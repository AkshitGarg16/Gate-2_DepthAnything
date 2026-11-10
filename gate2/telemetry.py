"""Time-keyed telemetry buffers and the fallback chains that read them.

Every sample is stored against its *header stamp* where the message has one,
because that is the only clock on which the camera and MAVROS agree: on the
fcb bags the image topic is logged a steady ~50 ms after its own stamp, so
pairing on receive time asks the IMU about the wrong moment. Attitude is
interpolated to the image stamp; everything else is nearest-in-time.

std_msgs/Float64 (rel_alt, compass_hdg) carries no stamp. Those are keyed on
receive time, which for MAVROS is within ~1 ms of the stamp it would have had.

Fallback chains (geo.altitude_sources / geo.heading_sources):
  altitude  rel_alt -> lidar /scan (near-nadir sweep fit, x lidar_range_scale)
            -> /mavros/local_position/pose z -> geo.fixed_altitude_m
  heading   compass_hdg -> IMU yaw (ENU, converted to a compass heading)
  attitude  IMU -> level (roll = pitch = 0) if geo.assume_nadir_if_no_imu
If a stamp lookup finds nothing within geo.max_age_s, but the newest sample
arrived within max_age_s of *wall* time (live only), that sample is used and
flagged -- this is what happens when the camera and the autopilot clocks are
not synced, and it is better to say so than to drop the frame.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


class TimeBuffer:
    """Ring of (stamp, receive_monotonic, value) covering the last `span` seconds."""

    def __init__(self, span=5.0):
        self.span = span
        self.d = deque()
        self.lock = threading.Lock()

    def push(self, t, v):
        with self.lock:
            # bags replayed with a loop or a seek can jump backwards: start over
            if self.d and t < self.d[-1][0] - 1.0:
                self.d.clear()
            self.d.append((t, time.monotonic(), v))
            while self.d and t - self.d[0][0] > self.span:
                self.d.popleft()

    def __len__(self):
        return len(self.d)

    def nearest(self, t, max_age, live=False):
        """(value, age_s, flag) nearest to stamp t; flag in {'', 'wallclock'} or None."""
        with self.lock:
            s = list(self.d)
        if not s:
            return None, float("inf"), None
        ts = np.fromiter((a for a, _, _ in s), float, len(s))
        i = int(np.argmin(np.abs(ts - t)))
        age = abs(ts[i] - t)
        if age <= max_age:
            return s[i][2], age, ""
        if live and time.monotonic() - s[-1][1] <= max_age:
            return s[-1][2], time.monotonic() - s[-1][1], "wallclock"
        return None, age, None

    def interp(self, t, max_age, live=False, wrap_last=False):
        """Linear interpolation of a tuple value at t (last component angle-wrapped)."""
        with self.lock:
            s = list(self.d)
        if len(s) >= 2:
            ts = np.fromiter((a for a, _, _ in s), float, len(s))
            if ts[0] <= t <= ts[-1]:
                v = np.array([c for _, _, c in s], float)
                out = []
                for j in range(v.shape[1]):
                    col = np.unwrap(v[:, j]) if (wrap_last and j == v.shape[1] - 1) else v[:, j]
                    out.append(float(np.interp(t, ts, col)))
                if wrap_last:
                    out[-1] = (out[-1] + math.pi) % (2 * math.pi) - math.pi
                return tuple(out), float(np.min(np.abs(ts - t))), ""
        return self.nearest(t, max_age, live)


def quat_to_rpy(x, y, z, w):
    """Intrinsic ZYX Euler angles (roll, pitch, yaw) -- same as Gate-1's giver_node."""
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def sweep_ground_range(angles, ranges, rmin, rmax, max_tilt_deg=25.0, near_deg=5.0):
    """Vertical distance to the ground from one 2D lidar sweep.

    Fits r = h / cos(theta - theta0): theta0 (the beam pointing straight down,
    absorbing mount tilt and vehicle attitude) by a grid search over the whole
    sweep, then h from the beams within near_deg of it only -- oblique beams
    skate across the grass canopy and read long. A simplified port of
    tile_depth.fit_sweep from the DA3 work.
    """
    a = np.asarray(angles, float)
    r = np.asarray(ranges, float)
    ok = np.isfinite(r) & (r > max(rmin, 0.05)) & (r < rmax)
    if ok.sum() < 8:
        return float("nan")
    a, r = a[ok], r[ok]
    best = (np.inf, 0.0)
    for th0 in np.radians(np.arange(-max_tilt_deg, max_tilt_deg + 1e-9, 0.5)):
        c = np.cos(a - th0)
        good = c > 0.3
        if good.sum() < 8:
            continue
        hv = r[good] * c[good]
        h = np.median(hv)
        res = np.median(np.abs(hv - h))
        if res < best[0]:
            best = (res, th0)
    th0 = best[1]
    near = np.abs(a - th0) < math.radians(near_deg)
    if near.sum() < 3:
        near = np.abs(a - th0) < math.radians(3 * near_deg)
    if near.sum() < 3:
        return float("nan")
    return float(np.median(r[near] * np.cos(a[near] - th0)))


@dataclass
class Telemetry:
    stamp: float
    roll: float = 0.0           # rad, from the MAVROS IMU (ENU/FLU), as Gate-1 uses it
    pitch: float = 0.0          # rad, Gate-1 negates it inside CoordinateTransformer
    heading: float = 0.0        # rad, compass heading (0 = north, clockwise)
    altitude: float | None = None   # camera-carrier height above ground, m
    lat: float | None = None
    lon: float | None = None
    att_source: str = "none"
    heading_source: str = "none"
    alt_source: str = "none"
    notes: list = field(default_factory=list)


class TelemetryHub:
    """Collects telemetry (from ROS callbacks or a bag reader) and answers snapshot(t)."""

    def __init__(self, cfg, live=False):
        self.cfg = cfg
        self.live = live
        span = cfg.ros.buffer_span_s
        self.att = TimeBuffer(span)       # (roll, pitch, yaw_enu)
        self.rel_alt = TimeBuffer(span)
        self.scan = TimeBuffer(span)      # ground range in m, already fitted and scaled
        self.pose_z = TimeBuffer(span)
        self.compass = TimeBuffer(span)   # degrees
        self.gps = TimeBuffer(span)       # (lat, lon)
        self.camera_K = None              # (fx, fy, cx, cy, width, height) from camera_info
        self.no_lidar = bool(cfg.get("no_lidar", {}).get("enabled", False))
        self.alt_sources = [s for s in cfg.geo.altitude_sources
                            if not (self.no_lidar and s == "scan")]
        self.counts = {k: 0 for k in ("imu", "rel_alt", "scan", "pose", "compass", "gps", "info")}

    # -- ingestion: one method per message type, ROS-free arguments ---------
    def on_imu(self, t, qx, qy, qz, qw):
        self.att.push(t, quat_to_rpy(qx, qy, qz, qw))
        self.counts["imu"] += 1

    def on_rel_alt(self, t, v):
        if math.isfinite(v):
            self.rel_alt.push(t, float(v))
            self.counts["rel_alt"] += 1

    def on_scan(self, t, angles, ranges, rmin, rmax):
        if self.no_lidar:
            return
        h = sweep_ground_range(angles, ranges, rmin, rmax)
        if math.isfinite(h):
            self.scan.push(t, h * self.cfg.geo.lidar_range_scale)
            self.counts["scan"] += 1

    def on_pose(self, t, z):
        self.pose_z.push(t, float(z))
        self.counts["pose"] += 1

    def on_compass(self, t, deg):
        self.compass.push(t, float(deg))
        self.counts["compass"] += 1

    def on_gps(self, t, lat, lon, status_ok=True):
        if status_ok and math.isfinite(lat) and math.isfinite(lon) and (lat or lon):
            self.gps.push(t, (float(lat), float(lon)))
            self.counts["gps"] += 1

    def on_camera_info(self, k, width, height):
        if self.camera_K is None and k[0] > 0:
            self.camera_K = (k[0], k[4], k[2], k[5], width, height)
            self.counts["info"] += 1

    # -- query ---------------------------------------------------------------
    def snapshot(self, t):
        g = self.cfg.geo
        age = g.max_age_s
        tel = Telemetry(stamp=t)

        att, a_age, flag = self.att.interp(t, age, self.live, wrap_last=True)
        yaw_enu = None
        if att is not None:
            tel.roll, tel.pitch, yaw_enu = att
            tel.att_source = "imu" + (" (wall-clock)" if flag else "")
        elif g.assume_nadir_if_no_imu:
            tel.att_source = "assumed level"
            tel.notes.append("no IMU: camera assumed nadir")

        for src in g.heading_sources:
            if src == "compass":
                v, _, flag = self.compass.nearest(t, age, self.live)
                if v is not None:
                    tel.heading, tel.heading_source = math.radians(v), "compass"
                    break
            elif src == "imu" and yaw_enu is not None:
                # ENU yaw is counter-clockwise from east; heading is clockwise from north
                tel.heading = (math.pi / 2 - yaw_enu) % (2 * math.pi)
                tel.heading_source = "imu yaw"
                break

        for src in self.alt_sources:
            v = None
            if src == "rel_alt":
                v, _, flag = self.rel_alt.nearest(t, age, self.live)
            elif src == "scan":
                v, _, flag = self.scan.nearest(t, max(age, 1.5), self.live)  # a sweep is slow
            elif src == "pose":
                v, _, flag = self.pose_z.nearest(t, age, self.live)
            elif src == "fixed" and g.fixed_altitude_m:
                v, flag = float(g.fixed_altitude_m), ""
            if v is not None and v > 0.5:
                tel.altitude = float(v)
                tel.alt_source = src + (" (wall-clock)" if flag else "")
                break
        if tel.altitude is None:
            tel.notes.append("no altitude: geometry is unscaled")

        v, _, flag = self.gps.nearest(t, age, self.live)
        if v is not None:
            tel.lat, tel.lon = v
        return tel
