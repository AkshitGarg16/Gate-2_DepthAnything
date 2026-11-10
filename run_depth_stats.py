#!/usr/bin/env python3
"""Whole-frame depth from every lidar return plus Depth-Anything-3 relief, and
the statistics of that depth over the frame -- rendered as a video.

    python3 run_depth_stats.py ~/rosbags/25m_2x_1.5ms
    python3 run_depth_stats.py ~/rosbags/25m_2x_1.5ms --start 60 --duration 20

Everything comes from the bag's topics: the camera, /pointcloud, the IMU
(/mavros/imu/data), the EKF pose (/mavros/local_position/pose) and rel_alt.
No sidecar CSV or mp4 is read.  /scan is not read: its beam angles are wrong
(see ~/Depth-Anything-3/lidar_map.py).

How a frame's depth is made
---------------------------
1. Every lidar return, in the world.  /pointcloud is cut into sweep passes
   (lidar_map.SweepSplitter; every return gets its own time).  Each return is
   put in the world (ENU) frame with the attitude and position *at its own
   time*:
       P = pos(t_i) + R_imu(t_i) (lidar_lever + k r_i R_mount d(theta_i))
   R_imu carries roll, pitch and yaw; R_mount is lidar_map's fitted mount
   (roll +3.05, yaw +5.0 deg); k is the range scale.

2. Into this frame's camera.  At the frame's stamp t_f the camera sits at
   pos(t_f) + R_imu(t_f) camera_lever and looks along R_imu(t_f) R_cam.
   Each world point is projected through it.  Doing the two steps separately
   takes out both kinds of parallax:
     * the lidar and the camera are at different places on the airframe
       (geo.lidar_offset_body vs geo.camera_offset_body), and
     * the aircraft moved and turned between the return and the shutter
       (up to 0.27 s of sweep, plus the window below).
   That is also what lets returns from several sweeps be used together: as
   the aircraft flies, successive sweep lines land on different image rows.

3. All the returns in a window around the frame (+-0.6 s, widened up to
   +-2 s when there are too few) that land on the camera's footprint give a
   ground surface: a plane in the world, fitted by weighted least squares.
     * Weights come from each return's noise.  That noise is measured from
       the bag at start-up: on 25m_2x_1.5ms it is 1.4 cm in range and 0.52 deg
       in beam angle, so a return's error grows from ~2 cm at nadir to ~15 cm
       at 25 deg off.
     * Tukey reweighting drops spikes.
     * The ground offset may drift linearly in time inside the window.  The
       EKF's height jitters ~5 cm from sweep to sweep, and trusting it to carry
       one sweep's ground into the next tipped the plane by degrees.  With the
       drift term, the lidar is its own altimeter and the EKF error cancels.
     * The slope has a level-ground prior, which binds only where the returns
       cannot see the slope (along-track while hovering).
   Accuracy is measured, not assumed: each sweep pass in the window is left
   out in turn and predicted from the rest, at its near-nadir returns (the
   "held-out" rms, which includes those returns' own ~2 cm noise).

4. DA3: the heatmap, and the metric depth only if the lidar confirms it.
   DA3's relief is DA3 over its own low-pass (sigma --detrend 100 px, as in
   person_parallax.py's depth panel).  Raised objects are left out of the
   low-pass, so no dark halo forms around a person.  The heatmap shows this
   relief as the reference panel does (inferno, colour range eased over
   time, edges cropped).  The range is the ground's median +- 4 robust sds,
   so the grass stays visible when people are in view.  A robust quadratic
   surface is divided out before the detrend, which keeps the edges clean.  It is display only, with no metric scale.  At 25 m the
   lidar finds no relationship with it: over 5,694 ground returns the
   correlation is -0.05, the lidar's spread 3.0 cm against DA3's 15.3 cm,
   and people read 10-15x too tall.
   The metric depth is
       depth(u, v) = L(u, v) * (1 + g * (DA3 / lowpass(DA3) - 1))
   where L is the lidar plane's depth along each pixel's ray and g the
   relief gain (--relief-gain auto).  g is fitted against every in-frame
   return's departure from the plane, weighted by that return's noise, with
   a prior of 1.  It is kept only if, fitted with each sweep pass held out in
   turn, it predicts the held-out returns at least --relief-min-gain (5%)
   better than the plain lidar ground.  Otherwise g = 0 and the depth is the
   lidar ground.

5. Statistics of the depth over every pixel, the histogram with the normal
   curve N(mean, sd), and the residual of the depth at every lidar return
   inside the frame.

Range scale: --lidar-scale auto (default) fits it per bag, regressing each
sweep's attitude-corrected nadir range on rel_alt over the climb.  If that is
not possible, it falls back to geo.lidar_range_scale.

Fallbacks, each one shown on the frame and in the CSV:
  too few returns   the previous frame's plane, if it is under --hold s old
  no lidar at all   a level plane at rel_alt below the camera
  below 1 m AGL     the frame is not measured (on the ground)
  no IMU            level attitude;  no pose: no motion compensation

What limits the accuracy
------------------------
Random noise averages down over hundreds of returns; the held-out rms
measures what is left.  Three systematic terms do not average down, and each
needs a measurement on the airframe:
  * the range scale -- 0.1% is 2.5 cm at 25 m
  * the lidar-to-camera lever arm (geo.lidar_offset_body, not measured):
    its vertical part enters the depth 1:1
  * the camera boresight (geo.camera_mount_deg, not measured): 1 deg moves the
    depth ~13 cm at the frame edge at 25 m
And DA3 relief on objects is not metric: DA3 has been measured exaggerating a
person's height several-fold.  The ground surface is what the lidar measures.

Output, in output.dir (config/default.yaml):
  <bag>_depthstats_<time>.mp4  2880x1080, five panels per frame:
        [frame + lidar returns][DA3 relief heatmap][statistics]
        [depth histogram      ][lidar altitude hist][  (tall)  ]
      The lidar-altitude panel is one altitude per in-frame return: the
      aircraft's height above that ground point, with roll, pitch, yaw, the
      mount and the scan angle taken out and carried to the frame's time
      (DepthFusion.altitudes), against the ground fit's altitude at nadir and
      rel_alt.
  <bag>_depthstats_<time>.csv  the same, one row per frame (alt_* = the
                               altitude statistics)
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gate2.config import load_config  # noqa: E402
from gate2.depth_estimator import MEAN, STD, DepthEstimator  # noqa: E402
from gate2.geolocator import resolve_K  # noqa: E402
from gate2.image_codec import stamp_s  # noqa: E402
from gate2.utils import log  # noqa: E402

PANEL_W, PANEL_H = 960, 540

# chart colours, dark surface (dataviz reference palette, dark steps)
SURFACE = "#1a1a19"
TEXT_1, TEXT_2 = "#f0efec", "#c3c2b7"
GRID = "#383835"
C_HIST, C_NORMAL, C_MEAN, C_MEDIAN, C_ANCHOR = "#3987e5", "#d95926", "#c98500", "#199e70", "#e66767"
C_WARN = "#fab219"

# camera axes (x right, y down, z optical) -> body FLU: image x = starboard,
# image top = nose, optical axis = body down.  Symmetric, so it is its own inverse.
R_CB = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

SIGMA_RANGE = 0.03                 # m, SF45 range noise (near-nadir sweep residual 0.023 m)
SIGMA_BEAM = math.radians(0.25)    # beam-angle noise
TUKEY_C = 4.685
NEAR_NADIR_DEG = 6.0               # held-out accuracy is read at returns this close to nadir


def bgr(hexcol):
    h = hexcol.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="rosbag2 directory")
    ap.add_argument("--start", type=float, default=0.0, help="start offset into the bag, s")
    ap.add_argument("--duration", type=float, default=None, help="seconds to process (default: to the end)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--res", type=int, default=1008,
                    help="DA3 input, longest side (/14 grid); 1008 -> 1008x560 on a 1920x1080 frame")
    ap.add_argument("--zoom", choices=["1x", "2x"],
                    help="intrinsics profile fcb_1x/fcb_2x (default: from the bag name, else config)")
    ap.add_argument("--cloud-topic", default="/pointcloud")
    ap.add_argument("--lidar-scale", default="auto",
                    help="'auto' = fit against rel_alt per bag, else a factor on every range "
                         "(geo.lidar_range_scale is the fallback; 1.0 = raw)")
    ap.add_argument("--angle-lag", default="auto",
                    help="'auto' = measure the SF45's direction-dependent beam-angle error from the bag, "
                         "else degrees (0 = none)")
    ap.add_argument("--window", type=float, default=0.6,
                    help="use lidar returns within this many s of the frame (widened up to --max-window)")
    ap.add_argument("--max-window", type=float, default=2.0)
    ap.add_argument("--min-points", type=int, default=40, help="returns needed on the footprint for a fit")
    ap.add_argument("--footprint-margin", type=float, default=0.25,
                    help="returns up to this fraction of the frame size outside it still count")
    ap.add_argument("--slope-prior", type=float, default=0.005,
                    help="sd of the level-ground slope prior, rise/run (0.005 = 0.3 deg).  It binds only "
                         "where the returns cannot see the slope: along-track while hovering, when every "
                         "sweep lands on the same line.  On 25m_2x_1.5ms the measured cross-track slope is "
                         "0.04 +- 0.5 deg; 0.005 gives corner-minus-centre jitter 3.0 cm vs 13.5 cm at 0.1")
    ap.add_argument("--time-order", type=int, default=1, choices=[0, 1, 2],
                    help="the ground offset may drift in time with this polynomial order (0 = trust the "
                         "EKF's vertical motion between sweeps)")
    ap.add_argument("--detrend", type=float, default=100.0,
                    help="DA3 relief = DA3 minus its Gaussian low-pass of this sigma, full-frame px "
                         "(as person_parallax.py's depth panel); coarser shape comes from the lidar")
    ap.add_argument("--relief-gain", default="auto",
                    help="relief in the METRIC depth: 'auto' = fitted to the lidar and kept only if it "
                         "predicts held-out returns (default); a number = fixed (1 = DA3 is depth up to a "
                         "scale).  On 25m_2x_1.5ms at 25 m DA3's relief has corr -0.05 with the lidar on "
                         "ground and reads people 10-15x too tall, so auto keeps it out.  The heatmap "
                         "always shows DA3's relief, display only")
    ap.add_argument("--crop", type=int, default=80,
                    help="edge px left out of the heatmap (person_parallax.py used 48; DA3 leaves a thin "
                         "bright rim at the bottom edge ~75 px deep)")
    ap.add_argument("--ema", type=float, default=0.05, help="heatmap colour range easing per frame")
    ap.add_argument("--relief-prior", type=float, default=0.5, help="sd of the relief gain prior (mean 1)")
    ap.add_argument("--relief-min-gain", type=float, default=0.05,
                    help="DA3 relief is used only if it cuts the held-out lidar misfit by this fraction")
    ap.add_argument("--relief-k", type=float, default=0.0,
                    help="DA3 relief within this many robust sds of the frame's own is ripple, shrunk to 0 "
                         "(0 = keep all of DA3's relief)")
    ap.add_argument("--hold", type=float, default=2.0, help="reuse the last plane for at most this long, s")
    ap.add_argument("--min-alt", type=float, default=1.0, help="below this height the frame is not measured, m")
    ap.add_argument("--bins", type=int, default=100, help="histogram bins")
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: the camera rate / stride)")
    ap.add_argument("--out", help="output mp4 path (default <output.dir>/<bag>_depthstats_<time>.mp4)")
    ap.add_argument("--config", help="YAML overriding config/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# telemetry and lidar, read from the bag once
# ---------------------------------------------------------------------------
def import_lidar_map(da3_root):
    if da3_root not in sys.path:
        sys.path.insert(0, da3_root)
    import lidar_map
    return lidar_map


class Series:
    """Samples (t, vector), linearly interpolated; quaternions by nlerp, held flat past the ends."""

    def __init__(self, t, v, quat=False):
        t, v = np.asarray(t, float), np.asarray(v, float)
        o = np.argsort(t)
        t, v = t[o], v[o]
        keep = np.r_[True, np.diff(t) > 0]
        self.t, self.v, self.quat = t[keep], v[keep], quat
        if quat and len(self.v) > 1:
            # same hemisphere as the previous sample, so interpolation takes the short way
            flip = np.sign(np.sum(self.v[1:] * self.v[:-1], axis=1))
            flip[flip == 0] = 1
            self.v = self.v * np.cumprod(np.r_[1.0, flip])[:, None]

    def __len__(self):
        return len(self.t)

    def at(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        out = np.stack([np.interp(t, self.t, self.v[:, j]) for j in range(self.v.shape[1])], 1)
        if self.quat:
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out


def rot_xyz(deg):
    """Rotation by small angles (deg) about x, then y, then z."""
    rx, ry, rz = np.radians(deg)
    X = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Z = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Z @ Y @ X


def frd_to_flu(v):
    return np.array([v[0], -v[1], -v[2]], float)


class BagTelemetry:
    """IMU attitude, EKF position, rel_alt and every lidar return of the bag, in one read."""

    def __init__(self, bag, cfg, cloud_topic, lm):
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from gate2.sources.bag_offline import _metadata

        r = cfg.ros
        sid, _ = _metadata(bag)
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=sid), rosbag2_py.ConverterOptions("", ""))
        types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        if cloud_topic not in types:
            raise RuntimeError(f"{bag} has no {cloud_topic}; lidar topics present: "
                               f"{sorted(t for t in types if 'scan' in t or 'cloud' in t or 'lidar' in t)}")
        want = {cloud_topic: "cloud", r.imu_topic: "imu", r.pose_topic: "pose", r.rel_alt_topic: "rel_alt"}
        want = {t: k for t, k in want.items() if t in types}
        self.missing = [k for k in ("imu", "pose", "rel_alt") if k not in want.values()]
        for k in self.missing:
            log("warn", "telemetry", f"bag has no {k} topic -- its fallback will be used")
        msg_types = {t: get_message(types[t]) for t in want}
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(want)))

        split = lm.SweepSplitter()
        imu_t, imu_q, pose_t, pose_p, alt_t, alt_v = [], [], [], [], [], []
        passes = []
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            kind = want[topic]
            m = deserialize_message(data, msg_types[topic])
            if kind == "cloud":
                rr, aa, xyz = lm.read_cloud(m)
                passes += [(t, a, rg) for t, a, rg, _ in split.add(stamp_s(m.header), aa, rr, xyz)]
            elif kind == "imu":
                q = m.orientation
                imu_t.append(stamp_s(m.header))
                imu_q.append((q.x, q.y, q.z, q.w))
            elif kind == "pose":
                p = m.pose.position
                pose_t.append(stamp_s(m.header))
                pose_p.append((p.x, p.y, p.z))
            elif kind == "rel_alt" and math.isfinite(m.data):
                alt_t.append(t_ns * 1e-9)       # Float64 carries no header: receive time
                alt_v.append(m.data)
        self.imu = Series(imu_t, imu_q, quat=True) if len(imu_t) > 1 else None
        self.pose = Series(pose_t, pose_p) if len(pose_t) > 1 else None
        self.rel_alt = Series(alt_t, np.array(alt_v)[:, None]) if len(alt_t) > 1 else None
        for k, s in (("imu", self.imu), ("pose", self.pose), ("rel_alt", self.rel_alt)):
            if s is None and k not in self.missing:
                self.missing.append(k)

        # every valid return, flattened, with its pass index
        ts, th, rg, pid = [], [], [], []
        for k, (t, a, rng) in enumerate(passes):
            ok = np.isfinite(rng) & (rng > lm.RANGE_MIN) & (rng < lm.RANGE_MAX)
            ts.append(t[ok])
            th.append(a[ok])
            rg.append(rng[ok])
            pid.append(np.full(int(ok.sum()), k))
        if not passes or not sum(len(x) for x in ts):
            raise RuntimeError(f"no valid returns in {cloud_topic}")
        t, theta, rng, pid = (np.concatenate(x) for x in (ts, th, rg, pid))
        o = np.argsort(t, kind="stable")
        self.t, self.theta, self.range_raw, self.pass_id = t[o], theta[o], rng[o], pid[o]
        log("info", "lidar", f"{cloud_topic}: {len(passes)} sweep passes, {len(self.t)} valid returns "
                             f"({split.n_dropped} passes dropped at gaps); imu {len(imu_t)}, "
                             f"pose {len(pose_t)}, rel_alt {len(alt_t)} samples")

    def attitude(self, t):
        """(N, 3, 3) body FLU -> ENU at times t; identity (level, yaw 0) without an IMU."""
        from lidar_map import quat_matrices
        t = np.atleast_1d(t)
        if self.imu is None:
            return np.repeat(np.eye(3)[None], len(t), 0)
        return quat_matrices(self.imu.at(t))

    def position(self, t):
        t = np.atleast_1d(t)
        if self.pose is None:
            return np.zeros((len(t), 3))
        return self.pose.at(t)

    def sweep_direction(self):
        """+1 / -1 for every return: whether its pass sweeps toward increasing theta."""
        if not hasattr(self, "_dir"):
            starts = np.r_[0, np.flatnonzero(np.diff(self.pass_id)) + 1]
            ends = np.r_[starts[1:], len(self.pass_id)] - 1
            d = np.sign(self.theta[ends] - self.theta[starts])
            self._dir = np.repeat(np.where(d == 0, 1.0, d), ends - starts + 1)
        return self._dir

    def calibrate_angle_lag(self, mount, lever_lidar, scale, window=0.6, min_h=15.0, span=2.0, step=0.05,
                            max_off_deg=25.0):
        """Direction-dependent beam-angle error, measured from the bag: theta += dir * delta.

        The SF45's reported angle and its range sample are not simultaneous, so
        a return lands a fixed angle ahead on one sweep direction and behind on
        the other: successive passes tilt alternately across track.  On
        25m_2x_1.5ms delta ~ -0.6 deg halves the multi-pass ground misfit
        (8.1 -> 4.3 cm rms).  Scored by the robust spread of every +-window
        group of returns about one shared plane with a time drift, over the
        airborne part of the bag, using beams within max_off_deg of nadir (the
        ones that land in the image); the minimum is refined with a parabola.
        Returns (delta_rad, lag_s, spread at 0, spread at delta) or None.
        """
        d = self.sweep_direction()
        theta0 = self.theta
        pos = self.position(self.t)
        rate = np.abs(np.diff(theta0)) / np.maximum(np.diff(self.t), 1e-4)
        rate = float(np.median(rate[(np.diff(self.pass_id) == 0) & np.isfinite(rate)]))
        cos_max = math.cos(math.radians(max_off_deg))
        t_grid = np.arange(self.t[0] + window, self.t[-1] - window, 1.0)
        bounds = np.searchsorted(self.t, np.stack([t_grid - window, t_grid + window], 1))

        def spread(delta):
            self.theta = theta0 + delta * d
            self.world_points(mount, lever_lidar, scale)
            res = []
            for (i0, i1), t in zip(bounds, t_grid):
                if i1 - i0 < 60:
                    continue
                m = np.arange(i0, i1)[self.cos_off[i0:i1] > cos_max]      # beams the camera can see
                if len(m) < 60:
                    continue
                P = self.P[m]
                if np.median(pos[m, 2] - P[:, 2]) < min_h:
                    continue
                f = fit_plane(P[:, 0] - P[:, 0].mean(), P[:, 1] - P[:, 1].mean(), P[:, 2], self.sigma[m],
                              0.005, self.t[m] - t, 1)
                if f is not None:
                    res.append(f[2][f[1]])
            if not res:
                return math.nan
            r = np.concatenate(res)
            return 1.4826 * float(np.median(np.abs(r - np.median(r))))

        try:
            grid = np.radians(np.arange(-span, span + 1e-9, 0.25))
            s = np.array([spread(x) for x in grid])
            if not np.isfinite(s).any():
                return None
            best = grid[np.nanargmin(s)]
            fine = best + np.radians(np.arange(-0.25, 0.25 + 1e-9, step))
            sf = np.array([spread(x) for x in fine])
            k = int(np.nanargmin(sf))
            delta = fine[k]
            if 0 < k < len(fine) - 1:                    # parabola through the minimum
                y0, y1, y2 = sf[k - 1:k + 2]
                den = y0 - 2 * y1 + y2
                if den > 0:
                    delta = fine[k] + 0.5 * (y0 - y2) / den * (fine[1] - fine[0])
            s0, s1 = spread(0.0), spread(delta)
        finally:
            self.theta = theta0
        if not (np.isfinite(s1) and s1 < s0):
            return None
        return float(delta), float(delta) / rate, s0, s1

    def beam_dirs(self, mount):
        """(N, 3) unit beam directions in ENU, each with the attitude at its own time."""
        th = self.theta + getattr(self, "angle_lag", 0.0) * self.sweep_direction()
        d = np.stack([np.zeros_like(th), np.sin(th), -np.cos(th)], 1) @ mount.T      # body FLU
        return np.einsum("nij,nj->ni", self.attitude(self.t), d), d

    def world_points(self, mount, lever_lidar, scale):
        """Every return in ENU, deskewed with the attitude and position at its own time."""
        dw, _ = self.beam_dirs(mount)
        self.lever_lidar = np.asarray(lever_lidar, float)
        lever = np.einsum("nij,j->ni", self.attitude(self.t), lever_lidar)
        self.range = self.range_raw * scale
        self.P = self.position(self.t) + lever + self.range[:, None] * dw
        self.cos_off = np.clip(-dw[:, 2], 1e-3, 1.0)
        self.set_noise(SIGMA_RANGE, SIGMA_BEAM)

    def set_noise(self, sigma_range, sigma_beam):
        """Height noise of each return: range noise, plus beam-angle noise across its slant."""
        self.sigma_range, self.sigma_beam = sigma_range, sigma_beam
        self.sigma = np.sqrt((sigma_range * self.cos_off) ** 2
                             + (self.range * np.sqrt(1 - self.cos_off ** 2) * sigma_beam) ** 2)

    def calibrate_noise(self, min_h=5.0, iters=2, max_off_deg=25.0):
        """Measure the range and beam-angle noise from the bag itself.

        Each sweep pass over flat ground is fitted on its own; the robust
        spread of its residuals against off-nadir angle gives
            var(off) = (sigma_r cos off)^2 + (r sin off * sigma_theta)^2.
        Beams within max_off_deg of nadir only (the ones in the image): far out,
        the ground is no longer one plane with the nadir ground, which would
        read as angle noise.
        """
        slant = self.range * np.sqrt(1 - self.cos_off ** 2)
        pos_z = self.position(self.t)[:, 2]
        view = self.cos_off > math.cos(math.radians(max_off_deg))
        for _ in range(iters):
            res, cos_o, sl = [], [], []
            for k in np.unique(self.pass_id):
                m = (self.pass_id == k) & view                # beams the camera can see
                if m.sum() < 30 or np.median(pos_z[m] - self.P[m, 2]) < min_h:
                    continue
                P = self.P[m]
                f = fit_plane(P[:, 0] - P[:, 0].mean(), P[:, 1] - P[:, 1].mean(), P[:, 2], self.sigma[m], 0.1)
                if f is None:
                    continue
                res.append(f[2][f[1]])
                cos_o.append(self.cos_off[m][f[1]])
                sl.append(slant[m][f[1]])
            if not res:
                return None
            res, cos_o, sl = np.concatenate(res), np.concatenate(cos_o), np.concatenate(sl)
            off = np.degrees(np.arccos(cos_o))
            rows, var = [], []
            for lo in np.arange(0, max_off_deg, 2.0):
                b = (off >= lo) & (off < lo + 2)
                if b.sum() >= 100:
                    rows.append([np.mean(cos_o[b] ** 2), np.mean(sl[b] ** 2)])
                    var.append((1.4826 * np.median(np.abs(res[b] - np.median(res[b])))) ** 2)
            if len(rows) < 3:
                return None
            sr2, st2 = np.linalg.lstsq(np.array(rows), np.array(var), rcond=None)[0]
            self.set_noise(math.sqrt(max(sr2, 0.01 ** 2)), math.sqrt(max(st2, math.radians(0.05) ** 2)))
        return self.sigma_range, self.sigma_beam, len(res)

    def calibrate_scale(self, mount, min_alt=5.0, min_span=8.0):
        """(scale, note): lidar range factor from each pass's nadir range against rel_alt.

        Fits h_lidar = s * rel_alt + b over the passes above min_alt (a free b,
        since the lidar sits above the skids and rel_alt drifts), robustly.
        Returns (None, why) when the bag cannot support it.
        """
        if self.rel_alt is None or self.imu is None:
            return None, "no rel_alt or IMU"
        dw, _ = self.beam_dirs(mount)
        cos_off = -dw[:, 2]
        h = self.range_raw * cos_off
        near = cos_off > math.cos(math.radians(5.0))
        hs, ts = [], []
        for k in np.unique(self.pass_id[near]):
            m = near & (self.pass_id == k)
            if m.sum() >= 5:
                hs.append(np.median(h[m]))
                ts.append(np.median(self.t[m]))
        if len(hs) < 50:
            return None, f"only {len(hs)} passes with nadir returns"
        hs, ra = np.array(hs), self.rel_alt.at(np.array(ts))[:, 0]
        m = ra > min_alt
        if m.sum() < 50 or np.ptp(ra[m]) < min_span:
            return None, f"rel_alt spans only {np.ptp(ra[m]) if m.any() else 0:.1f} m above {min_alt} m"
        x, y = ra[m], hs[m]
        w = np.ones_like(x)
        for _ in range(8):                       # Tukey IRLS on the line
            sw = np.sqrt(w)
            A = np.column_stack([x, np.ones_like(x)]) * sw[:, None]
            s, b = np.linalg.lstsq(A, y * sw, rcond=None)[0]
            res = y - (s * x + b)
            sc = max(1.4826 * np.median(np.abs(res[w > 0])), 1e-3)
            u = res / (TUKEY_C * sc)
            w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if not 0.8 < s < 1.25:
            return None, f"fitted lidar/rel_alt slope {s:.4f} is implausible"
        return 1.0 / s, (f"lidar = {s:.4f} x rel_alt {b:+.2f} m over {int((w > 0).sum())} passes "
                         f"({x.min():.0f}-{x.max():.0f} m), resid {sc:.3f} m")


# ---------------------------------------------------------------------------
# the ground plane
# ---------------------------------------------------------------------------
def fit_plane(x, y, z, sig, slope_sd, dt=None, order=0, drift_sd=0.5, iters=8):
    """Weighted Tukey-robust plane z = c0 + c1 x + c2 y [+ d1 dt + d2 dt^2 ...].

    c1, c2 get a N(0, slope_sd) level-ground prior.  With dt (each return's
    time minus the frame's) and order >= 1, the offset may drift in time --
    that absorbs the EKF's vertical error between sweeps, and c0 is then the
    ground *at the frame's time*.  The drift terms get a weak N(0, drift_sd)
    prior (m/s, m/s^2).  Returns (c, inlier mask, residuals) or None.
    """
    n = len(z)
    if n < 3:
        return None
    cols = [np.ones(n), x, y]
    pri = [0.0, 1.0 / slope_sd ** 2, 1.0 / slope_sd ** 2]
    for k in range(1, order + 1 if dt is not None else 1):
        cols.append(dt ** k)
        pri.append(1.0 / drift_sd ** 2)
    A = np.column_stack(cols)
    prior = np.diag(pri)
    tw = np.ones(n)
    c = res = None
    for _ in range(iters):
        W = tw / sig ** 2
        M = A.T @ (W[:, None] * A) + prior
        try:
            c = np.linalg.solve(M, A.T @ (W * z))
        except np.linalg.LinAlgError:
            return None
        res = z - A @ c
        sr = res / sig
        live = tw > 0
        scale = max(1.4826 * np.median(np.abs(sr[live])), 1.0)
        u = sr / (TUKEY_C * scale)
        tw = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if (tw > 0).sum() < 3:
            return None
    return c, tw > 0, res


class Plane:
    """A ground plane z = c0 + c1 (e - e0) + c2 (n - n0), ENU, and where it came from."""

    def __init__(self, c, origin, source, stamp):
        self.c, self.origin, self.source, self.stamp = np.asarray(c, float), np.asarray(origin, float), source, stamp
        self.drift = np.empty(0)        # ground offset drift in time, m/s, m/s^2 (lidar fits only)

    def drift_at(self, dt):
        return sum(d * dt ** (k + 1) for k, d in enumerate(self.drift)) if len(self.drift) else 0.0

    def z_at(self, e, n):
        return self.c[0] + self.c[1] * (e - self.origin[0]) + self.c[2] * (n - self.origin[1])

    def height_below(self, C):
        return C[2] - self.z_at(C[0], C[1])

    def tilt_deg(self):
        return math.degrees(math.atan(math.hypot(self.c[1], self.c[2])))

    def depth_along(self, rays_c, C, R_wc):
        """z-depth (camera axis) where each camera ray (x, y, 1) meets the plane; NaN if it never does."""
        m = np.array([-self.c[1], -self.c[2], 1.0])
        num = self.z_at(C[0], C[1]) - C[2]              # < 0: the ground is below the camera
        den = rays_c @ (R_wc.T @ m)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = num / den
        s[~(s > 0)] = np.nan
        return s


# ---------------------------------------------------------------------------
# DA3 on the whole frame
# ---------------------------------------------------------------------------
def infer_size(h, w, s):
    """DA3's upper_bound_resize: longest side to s, both sides to a /14 grid."""
    sc = s / max(h, w)
    return max(int(round(h * sc / 14) * 14), 14), max(int(round(w * sc / 14) * 14), 14)


def da3_frame(est, frame, res):
    """DA3 relative depth of the whole frame at its own (h, w) grid."""
    torch = est.torch
    H, W = frame.shape[:2]
    h, w = infer_size(H, W, res)
    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)[None, None]).to(est.device)
    with torch.no_grad(), torch.autocast(est.device, dtype=est.acd, enabled=(est.device == "cuda")):
        pr = est.model(t)
    return pr["depth"].float().reshape(h, w).cpu().numpy()


def detrend(z, sigma, k=3.0):
    """DA3 relative relief z / lowpass(z) - 1, with raised objects left out of the low-pass.

    Two steps:
      * a robust quadratic surface is divided out first.  A Gaussian low-pass
        mirrored at the border cannot follow DA3's broad tilt and bowl there,
        which left one edge bright and the opposite one dark.
      * the Gaussian low-pass is a normalised convolution over the pixels
        within k robust sds of a first pass's relief.  A plain blur takes in a
        person too, so the ground around them reads low -- the dark halo in
        person_parallax.py's panel.
    """
    h, w = z.shape
    yy, xx = np.mgrid[0:h:4, 0:w:4]
    X = np.column_stack([np.ones(xx.size), xx.ravel() / w, yy.ravel() / h, (xx.ravel() / w) ** 2,
                         (xx.ravel() / w) * (yy.ravel() / h), (yy.ravel() / h) ** 2])
    zs = z[::4, ::4].ravel().astype(np.float64)
    keep = np.ones(zs.size, bool)
    for _ in range(3):
        c = np.linalg.lstsq(X[keep], zs[keep], rcond=None)[0]
        r = zs - X @ c
        sd = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) <= k * max(sd, 1e-9)
    Y, Xg = np.mgrid[0:h, 0:w]
    xn, yn = Xg / w, Y / h
    surf = c[0] + c[1] * xn + c[2] * yn + c[3] * xn ** 2 + c[4] * xn * yn + c[5] * yn ** 2
    z1 = (z / np.maximum(surf, 1e-6)).astype(np.float32)

    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    rel = z1 / np.maximum(blur(z1), 1e-6) - 1.0
    med = float(np.median(rel))
    sd = 1.4826 * float(np.median(np.abs(rel - med)))
    m = (np.abs(rel - med) <= k * max(sd, 1e-9)).astype(np.float32)
    low = blur(z1 * m) / np.maximum(blur(m), 1e-6)
    return z1 / np.maximum(low, 1e-6) - 1.0


def bilinear(img, x, y):
    """img sampled at float pixel coordinates (x, y), clamped to the edge."""
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
STAT_KEYS = ["n", "mean", "median", "mode", "sd", "sd_median", "var", "mad", "sd_mad", "mean_abs_dev",
             "iqr", "sd_iqr", "min", "max", "range", "p1", "p5", "p25", "p75", "p95", "p99",
             "skew", "kurt_excess", "cv", "within_1sd", "within_2sd", "within_3sd", "bin_w"]
FUSION_KEYS = ["source", "flags", "window_s", "n_window", "n_footprint", "n_inliers", "n_frame", "n_passes",
               "range_scale", "plane_h_m", "plane_tilt_deg", "heldout_rms_m", "heldout_p95_m", "heldout_n",
               "frame_resid_rms_m", "frame_resid_p95_m", "resid_noise_m", "relief_gain", "relief", "rel_alt_m", "plane_minus_rel_alt_m",
               "roll_deg", "pitch_deg", "yaw_deg", "alt_ground_fit_m", "alt_n_rejected"]


def depth_stats(D, bins):
    """Statistics of every finite pixel of D, plus the histogram (edges, density)."""
    x = D[np.isfinite(D)].astype(np.float64).ravel()
    n = x.size
    mean = float(x.mean())
    dev = x - mean
    var = float(np.mean(dev ** 2))
    sd = math.sqrt(var)
    p1, p5, p25, med, p75, p95, p99 = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    mad = float(np.median(np.abs(x - med)))
    lo, hi = float(x.min()), float(x.max())
    # plot range: the central 99.8% widened to mean +- 3.5 sd, never past the data
    q_lo, q_hi = np.percentile(x, [0.1, 99.9])
    h_lo, h_hi = max(lo, min(q_lo, mean - 3.5 * sd)), min(hi, max(q_hi, mean + 3.5 * sd))
    if h_hi <= h_lo:
        h_lo, h_hi = mean - 1e-3, mean + 1e-3
    counts, edges = np.histogram(x, bins=bins, range=(h_lo, h_hi))
    dens = counts / (n * np.diff(edges))
    k = int(np.argmax(counts))
    a = np.abs(dev)
    st = dict(
        n=n, mean=mean, median=float(med), mode=float((edges[k] + edges[k + 1]) / 2),
        sd=sd, sd_median=float(math.sqrt(np.mean((x - med) ** 2))), var=var,
        mad=mad, sd_mad=1.4826 * mad, mean_abs_dev=float(a.mean()),
        iqr=float(p75 - p25), sd_iqr=float((p75 - p25) / 1.349),
        min=lo, max=hi, range=hi - lo,
        p1=float(p1), p5=float(p5), p25=float(p25), p75=float(p75), p95=float(p95), p99=float(p99),
        skew=float(np.mean(dev ** 3) / sd ** 3) if sd > 0 else math.nan,
        kurt_excess=float(np.mean(dev ** 4) / sd ** 4 - 3.0) if sd > 0 else math.nan,
        cv=sd / mean if mean else math.nan,
        within_1sd=float(np.mean(a <= sd)), within_2sd=float(np.mean(a <= 2 * sd)),
        within_3sd=float(np.mean(a <= 3 * sd)),
        bin_w=float(edges[1] - edges[0]),
    )
    return st, edges, dens


def rms(v):
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else math.nan


def p95abs(v):
    v = np.asarray(v, float)
    return float(np.percentile(np.abs(v), 95)) if v.size else math.nan


def rpy_flu(R):
    """Roll, pitch, yaw (rad) of a body FLU -> ENU rotation."""
    return (math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0]))), math.atan2(R[1, 0], R[0, 0]))


# ---------------------------------------------------------------------------
# one frame
# ---------------------------------------------------------------------------
class DepthFusion:
    """Lidar ground plane + DA3 relief, one frame at a time."""

    def __init__(self, tel, a, K, W, H, h, w, R_cam, lever_cam):
        self.tel, self.a, self.K, self.W, self.H = tel, a, K, W, H
        self.R_cam, self.lever_cam = R_cam, lever_cam
        fx, fy, cx, cy = K
        # camera rays (x, y, 1) through the centres of DA3's grid cells, in full-frame pixels
        uu = (np.arange(w) + 0.5) * W / w - 0.5
        vv = (np.arange(h) + 0.5) * H / h - 0.5
        U, V = np.meshgrid(uu, vv)
        self.rays = np.stack([(U - cx) / fx, (V - cy) / fy, np.ones_like(U)], -1).reshape(-1, 3)
        self.grid = (h, w)
        self.last = None

    def camera(self, stamp):
        Rb = self.tel.attitude(stamp)[0]
        C = self.tel.position(stamp)[0] + Rb @ self.lever_cam
        return C, Rb @ self.R_cam, Rb

    def returns(self, stamp, C, R_wc):
        """The returns in the window that land on the camera's footprint, seen from this camera."""
        tel, a = self.tel, self.a
        fx, fy, cx, cy = self.K
        win = a.window
        while True:
            i0, i1 = np.searchsorted(tel.t, [stamp - win, stamp + win])
            idx = np.arange(i0, i1)
            pc = (tel.P[idx] - C) @ R_wc                 # each row is R_wc^T (P - C)
            with np.errstate(divide="ignore", invalid="ignore"):
                u = cx + fx * pc[:, 0] / pc[:, 2]
                v = cy + fy * pc[:, 1] / pc[:, 2]
            mW, mH = a.footprint_margin * self.W, a.footprint_margin * self.H
            foot = (pc[:, 2] > 0.3) & (u > -mW) & (u < self.W + mW) & (v > -mH) & (v < self.H + mH)
            if foot.sum() >= a.min_points or win >= a.max_window:
                break
            win = min(win * 2, a.max_window)
        inframe = foot & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return dict(idx=idx[foot], pc=pc[foot], u=u[foot], v=v[foot], inframe=inframe[foot],
                    window=win, n_window=len(idx))

    def fit(self, L, C, stamp):
        """The plane through the footprint returns, and each pass's held-out residuals."""
        tel, a = self.tel, self.a
        P, sig, pid = tel.P[L["idx"]], tel.sigma[L["idx"]], tel.pass_id[L["idx"]]
        x, y, z = P[:, 0] - C[0], P[:, 1] - C[1], P[:, 2]
        dt = tel.t[L["idx"]] - stamp
        f = fit_plane(x, y, z, sig, a.slope_prior, dt, a.time_order)
        if f is None:
            return None
        c, inl, _ = f
        # held out: each pass predicted from the others, read at its near-nadir returns,
        # where a return's own noise (~2-3 cm) is small next to the error being measured
        near = tel.cos_off[L["idx"]] >= math.cos(math.radians(NEAR_NADIR_DEG))
        ho = []
        passes = np.unique(pid)
        for k in passes:
            test = pid == k
            if (test & near).sum() < 3 or (~test).sum() < a.min_points // 2:
                continue
            g = fit_plane(x[~test], y[~test], z[~test], sig[~test], a.slope_prior, dt[~test], a.time_order)
            if g is not None:
                t = test & near & inl
                pred = g[0][0] + g[0][1] * x[t] + g[0][2] * y[t]
                for j in range(3, len(g[0])):
                    pred = pred + g[0][j] * dt[t] ** (j - 2)
                ho.append(z[t] - pred)
        ho = np.concatenate(ho) if ho else np.empty(0)
        plane = Plane(c[:3], C[:2], "lidar", stamp)
        plane.drift = c[3:]
        return plane, inl, ho, len(passes)

    def relief_gain(self, L, Lmap, rel, plane, stamp, R_wc):
        """(g, note): DA3 relief gain, kept only if it predicts lidar returns it was not fitted on.

        The gain is fitted to every in-frame return's departure from the plane
        (weighted by its measured noise, prior N(1, relief_prior)), with each
        sweep pass held out in turn.  If those held-out predictions beat the
        plain lidar ground (g = 0) on the held-out returns, g is fitted on all
        of them; otherwise DA3's relief is left out (g = 0).
        """
        a, tel = self.a, self.tel
        h, w = self.grid
        fa = L["inframe"]
        if plane.source != "lidar":
            return 1.0, "no lidar to test it: prior"
        if fa.sum() < 10:
            return 0.0, "too few returns to test it: off"
        ax_, ay_ = L["u"][fa] * w / self.W - 0.5, L["v"][fa] * h / self.H - 0.5
        Lp = bilinear(Lmap, ax_, ay_)
        x = Lp * bilinear(rel, ax_, ay_)          # DA3's departure from the plane, m
        dt = tel.t[L["idx"][fa]] - stamp
        y = L["pc"][fa, 2] - R_wc[2, 2] * plane.drift_at(dt) - Lp      # the lidar's at the frame's time, m
        wt = 1.0 / tel.sigma[L["idx"][fa]] ** 2
        keep = np.abs(y) < 1.0                    # a spike is not relief
        x, y, wt, pid = x[keep], y[keep], wt[keep], tel.pass_id[L["idx"][fa]][keep]
        pw = 1.0 / a.relief_prior ** 2

        def fit(m):
            return float(np.clip((np.sum(wt[m] * x[m] * y[m]) + pw) / (np.sum(wt[m] * x[m] ** 2) + pw), 0.0, 2.0))

        sse_g = sse_0 = 0.0
        for k in np.unique(pid):
            t = pid == k
            if t.all() or not t.any():
                continue
            gk = fit(~t)
            sse_g += np.sum(wt[t] * (y[t] - gk * x[t]) ** 2)
            sse_0 += np.sum(wt[t] * y[t] ** 2)
        if sse_0 <= 0:
            return 0.0, "off"
        gain = 1.0 - sse_g / sse_0
        if gain < a.relief_min_gain:
            return 0.0, (f"off: held-out {100 * gain:+.0f}% vs lidar ground alone "
                         f"(needs +{100 * a.relief_min_gain:.0f}%)")
        return fit(np.ones(len(y), bool)), f"on: held-out {100 * gain:+.0f}% vs lidar ground alone"

    def altitudes(self, L, stamp, plane, info):
        """The aircraft's altitude above the ground according to each in-frame return, at the frame's time.

        A return's world point already has roll, pitch and yaw (the IMU at its own
        time), the mount and the scan angle taken out, so its vertical offset from
        the lidar, -r * (R_imu R_mount d(theta)).z, is the lidar's height above that
        ground point when it fired.  It is carried to the frame's time with the
        aircraft's vertical motion since then and the ground fit's offset drift
        (the EKF's vertical error between sweeps):
            alt_i = lidar z(t_f) - (P_i z - drift(t_i - t_f))
        For a return fired at t_f this is exactly r * cos(true off-nadir angle).
        """
        tel = self.tel
        fa = L["inframe"]
        idx = L["idx"][fa]
        Rf = tel.attitude(stamp)[0]
        lid = tel.position(stamp)[0] + Rf @ tel.lever_lidar
        drift = plane.drift_at(tel.t[idx] - stamp) if plane is not None and plane.source == "lidar" else 0.0
        info["alt"] = lid[2] - (tel.P[idx, 2] - drift)
        info["alt_inlier"] = L["inliers"][fa]
        info["alt_plane"] = (lid[2] - plane.z_at(lid[0], lid[1])) if plane is not None else math.nan

    def __call__(self, stamp, z_da3):
        """(D or None, info): metric depth on DA3's grid, and everything about how it was made."""
        a, tel = self.a, self.tel
        C, R_wc, Rb = self.camera(stamp)
        flags = [f"no {k}" for k in tel.missing if k in ("imu", "pose")]
        roll, pitch, yaw = (math.degrees(v) for v in rpy_flu(Rb))
        info = dict(flags=flags, points=None, heldout=np.empty(0), resid=np.empty(0), n_inliers=0, n_passes=0,
                    window=math.nan, n_window=0, n_footprint=0, n_frame=0, gain=math.nan, C=C, plane=None,
                    roll=roll, pitch=pitch, yaw=yaw,
                    rel_alt=float(tel.rel_alt.at(stamp)[0, 0]) if tel.rel_alt is not None else math.nan)

        L = self.returns(stamp, C, R_wc)
        info.update(window=L["window"], n_window=L["n_window"], n_footprint=len(L["idx"]),
                    n_frame=int(L["inframe"].sum()), points=L)
        L["inliers"] = np.zeros(len(L["idx"]), bool)
        plane = None
        if len(L["idx"]) >= a.min_points:
            f = self.fit(L, C, stamp)
            if f is not None:
                plane, L["inliers"], ho, npass = f
                info.update(n_inliers=int(L["inliers"].sum()), heldout=ho, n_passes=npass)
        if plane is None:
            if self.last is not None and 0 <= stamp - self.last.stamp <= a.hold:
                plane = Plane(self.last.c, self.last.origin, "held", self.last.stamp)
                flags.append(f"too few returns: plane held from {stamp - self.last.stamp:.1f} s ago")
            elif math.isfinite(info["rel_alt"]):
                plane = Plane([C[2] - info["rel_alt"], 0, 0], C[:2], "rel_alt", stamp)
                flags.append("no lidar fit: level ground at rel_alt")
        info["plane"] = plane
        self.altitudes(L, stamp, plane, info)
        if plane is None:
            flags.append("no ground reference")
            return None, info
        if plane.height_below(C) < a.min_alt:
            flags.append(f"below {a.min_alt:.0f} m (on the ground): not measured")
            return None, info
        if plane.source == "lidar":
            self.last = plane

        h, w = self.grid
        Lmap = plane.depth_along(self.rays, C, R_wc).reshape(h, w)
        fr = L["inframe"] & L["inliers"]
        gx, gy = L["u"][fr] * w / self.W - 0.5, L["v"][fr] * h / self.H - 0.5
        if z_da3 is None:
            info["gain"], D = 0.0, Lmap
        else:
            # DA3 relief: DA3 over its own low-pass; the coarse shape is the lidar's
            rel = detrend(z_da3, max(a.detrend * w / self.W, 0.6))
            info["display_relief"] = -rel                # + = raised; DA3's own unit, for the heatmap
            # ripples: relief within relief_k x the frame's own robust spread is DA3 texture on
            # flat ground (the lidar finds no such relief there), so it is shrunk to zero; what
            # stands clear of it -- an object -- is kept, less that threshold
            if a.relief_k > 0:
                med = float(np.median(rel))
                thr = a.relief_k * 1.4826 * float(np.median(np.abs(rel - med)))
                rel = np.sign(rel - med) * np.maximum(np.abs(rel - med) - thr, 0.0)
            # relief gain from every in-frame return's departure from the plane, weighted by its
            # measured noise, prior N(1, relief_prior).  Ground returns carry ~no relief and so
            # do not pull it; a return that lands on an object does.
            if str(a.relief_gain) == "auto":
                g, info["relief"] = self.relief_gain(L, Lmap, rel, plane, stamp, R_wc)
            else:
                g, info["relief"] = float(a.relief_gain), "fixed: DA3 relief taken as depth up to a scale"
            info["gain"] = g
            D = Lmap * (1.0 + g * rel)
            info["relief_m"] = -Lmap * g * rel          # height above the local ground, m
        if fr.any():
            # each return is brought to the frame's time first: the fit's offset drift (the
            # EKF's vertical error between sweeps) moves it along the vertical
            dt = tel.t[L["idx"][fr]] - stamp
            z_now = L["pc"][fr, 2] - R_wc[2, 2] * plane.drift_at(dt)
            info["resid"] = z_now - bilinear(D, gx, gy)
            info["resid_noise"] = rms(tel.sigma[L["idx"][fr]])
        return D, info


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def put(img, text, org, scale=0.55, col=TEXT_1, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bgr(col) if isinstance(col, str) else col,
                thick, cv2.LINE_AA)


def lidar_dots(img, pts, sx, sy, col, width=None):
    """The in-frame lidar returns as dots; returns the fit rejected as amber rings."""
    if pts is None:
        return
    width = img.shape[1] if width is None else width
    for i in np.flatnonzero(pts["inframe"]):
        x, y = int(round(pts["u"][i] * sx)), int(round(pts["v"][i] * sy))
        if not (4 <= x < width - 4 and 4 <= y < img.shape[0] - 4):
            continue
        if pts["inliers"][i]:
            cv2.circle(img, (x, y), 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 2, bgr(col), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), 4, bgr(C_WARN), 1, cv2.LINE_AA)


def crop_view(img, frame_wh, crop):
    """img (any resolution of the frame) with `crop` full-frame px cut from every edge."""
    h, w = img.shape[:2]
    cx, cy = int(round(crop * w / frame_wh[0])), int(round(crop * h / frame_wh[1]))
    return img[cy:h - cy, cx:w - cx] if crop > 0 else img


def heatmap_panel(relief, vmin, vmax, pts, frame_wh, title, crop):
    """DA3's detrended relief, inferno, raised = bright -- rendered as person_parallax.py's depth
    panel is (colour range eased over time, edges cropped), but stretched about the ground
    (median +- 4 robust sds) so the grass stays visible with people in view.  Display only: no metric scale is claimed,
    because at survey height the lidar finds no relationship between it and the ground's height.
    """
    bar_w, pad = 70, 8
    img_w = PANEL_W - bar_w
    out = np.full((PANEL_H, PANEL_W, 3), bgr(SURFACE), np.uint8)
    if relief is not None:
        r = crop_view(relief, frame_wh, crop)
        q = np.clip((r - vmin) / max(vmax - vmin, 1e-12), 0, 1)
        u8 = (np.nan_to_num(q, nan=0.0) * 255).astype(np.uint8)
        out[:, :img_w] = cv2.applyColorMap(cv2.resize(u8, (img_w, PANEL_H), interpolation=cv2.INTER_CUBIC),
                                           cv2.COLORMAP_INFERNO)
        cw, ch = frame_wh[0] - 2 * crop, frame_wh[1] - 2 * crop
        lidar_dots(out, shift_points(pts, crop), img_w / cw, PANEL_H / ch, "#ffffff", img_w)
        y0, y1 = 50, PANEL_H - 40
        x0 = img_w + pad
        grad = (np.linspace(255, 0, y1 - y0)).astype(np.uint8)[:, None].repeat(16, 1)
        out[y0:y1, x0:x0 + 16] = cv2.applyColorMap(grad, cv2.COLORMAP_INFERNO)
        for i, lab in ((0, "+4sd"), (2, "med"), (4, "-4sd")):
            y = int(y0 + i / 4 * (y1 - y0 - 1))
            cv2.line(out, (x0 + 16, y), (x0 + 20, y), bgr(TEXT_2), 1)
            cv2.putText(out, lab, (x0 + 22, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr(TEXT_1), 1, cv2.LINE_AA)
        cv2.putText(out, "raised", (x0 - 6, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr(TEXT_2), 1, cv2.LINE_AA)
        cv2.putText(out, "low", (x0 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr(TEXT_2), 1, cv2.LINE_AA)
    else:
        put(out, "no DA3", (PANEL_W // 2 - 50, PANEL_H // 2), 0.8, C_WARN)
    put(out, title, (12, 26), 0.6)
    return out


def shift_points(pts, crop):
    """The lidar points with their pixels moved into the cropped view."""
    if pts is None or crop <= 0:
        return pts
    q = dict(pts)
    q["u"], q["v"] = pts["u"] - crop, pts["v"] - crop
    return q


class HistPanel:
    """Histogram with N(mean, sd) over it, and reference lines (matplotlib, Agg)."""

    def __init__(self, title, xlabel, sample_label):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        self.title, self.xlabel, self.sample_label = title, xlabel, sample_label
        self.fig = Figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100, facecolor=SURFACE)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0.085, 0.13, 0.885, 0.63])

    def draw(self, st, edges, dens, refs=(), note=""):
        """refs: (x, label, colour, linestyle) reference lines, drawn when they fall in range."""
        ax = self.ax
        ax.clear()
        ax.set_facecolor(SURFACE)
        for lg in self.fig.legends:
            lg.remove()
        self.fig.texts.clear()
        self.fig.text(0.012, 0.955, self.title, color=TEXT_1, fontsize=11)
        if note:
            self.fig.text(0.988, 0.955, note, color=TEXT_2, fontsize=9, ha="right")
        if st is None:
            ax.set_axis_off()
        else:
            ax.set_axis_on()
            self._plot(ax, st, edges, dens, refs)
        self.canvas.draw()
        return cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)

    def _plot(self, ax, st, edges, dens, refs):
        mu, sd, med = st["mean"], st["sd"], st["median"]
        ax.stairs(dens, edges, fill=True, color=C_HIST, alpha=0.55, lw=0, label=self.sample_label)
        ax.stairs(dens, edges, color=C_HIST, lw=1.0)
        xs = np.linspace(edges[0], edges[-1], 400)
        if sd > 0:
            ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)),
                    color=C_NORMAL, lw=2, label=f"normal N({mu:.3f}, {sd:.3f}²)")
            ax.axvspan(mu - sd, mu + sd, color=C_NORMAL, alpha=0.08, lw=0)
        ax.axvline(mu, color=C_MEAN, lw=1.6, label=f"mean {mu:.3f}")
        ax.axvline(med, color=C_MEDIAN, lw=1.6, ls="--", label=f"median {med:.3f}")
        for x, lab, colr, ls in refs:
            if x is not None and math.isfinite(x) and edges[0] <= x <= edges[-1]:
                ax.axvline(x, color=colr, lw=1.4, ls=ls, label=f"{lab} {x:.3f}")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, max(dens.max(), 1.0 / (sd * math.sqrt(2 * math.pi)) if sd > 0 else 0) * 1.12)
        ax.set_xlabel(self.xlabel, color=TEXT_2, fontsize=10)
        ax.set_ylabel("density (1/m)", color=TEXT_2, fontsize=10)
        ax.tick_params(colors=TEXT_2, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        sec = ax.secondary_xaxis("top", functions=(lambda x: (x - mu) * 100, lambda d: mu + d / 100))
        sec.set_xlabel("deviation from mean (cm)", color=TEXT_2, fontsize=9)
        sec.tick_params(colors=TEXT_2, labelsize=8)
        sec.spines["top"].set_color(GRID)
        h, lab = ax.get_legend_handles_labels()
        leg = self.fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=min(len(h), 6),
                              fontsize=8, frameon=False, handlelength=1.6, columnspacing=1.0)
        for t in leg.get_texts():
            t.set_color(TEXT_1)


def stats_panel(st, sa, fu, frame_no, t_rel):
    """The statistics column, full height: the frame's depth, the lidar altitudes, the fusion."""
    img = np.full((2 * PANEL_H, PANEL_W, 3), bgr(SURFACE), np.uint8)
    put(img, f"frame {frame_no}   t = {t_rel:7.2f} s    roll {fu['roll_deg']:+.1f}  pitch {fu['pitch_deg']:+.1f}"
             f"  yaw {fu['yaw_deg']:+.1f} deg", (16, 28), 0.5, TEXT_2)

    def col(x, y, rows, dx=205, step=21, scale=0.5):
        for i, (k, v) in enumerate(rows):
            put(img, k, (x, y + step * i), scale, TEXT_2)
            put(img, v, (x + dx, y + step * i), scale, TEXT_1)

    def cm(v):
        return f"{100 * v:.1f} cm" if v is not None and math.isfinite(v) else "-"

    def m(v, fmt):
        return format(v, fmt) if v is not None and math.isfinite(v) else "-"

    def block(y, head, st, n_label):
        cv2.line(img, (16, y - 18), (PANEL_W - 16, y - 18), bgr(GRID), 1)
        put(img, head, (16, y + 4), 0.6)
        if st is None:
            put(img, "none for this frame", (16, y + 34), 0.55, C_WARN)
            return
        f3 = lambda v: f"{v:.3f}"  # noqa: E731
        put(img, f"{n_label} = {st['n']:,}", (16, y + 28), 0.47, TEXT_2)
        col(16, y + 54, [
            ("mean", f3(st["mean"])), ("median", f3(st["median"])), ("mode (bin centre)", f3(st["mode"])),
            ("sd about mean", f3(st["sd"])), ("sd about median", f3(st["sd_median"])),
            ("variance (m^2)", f"{st['var']:.5f}"), ("mean abs dev", f3(st["mean_abs_dev"])),
            ("MAD (median)", f3(st["mad"])), ("1.4826 x MAD", f3(st["sd_mad"])),
            ("IQR (P75-P25)", f3(st["iqr"])), ("IQR / 1.349", f3(st["sd_iqr"])),
            ("coeff. of variation", f"{100 * st['cv']:.3f} %"),
        ])
        col(490, y + 54, [
            ("min / max", f"{st['min']:.3f} / {st['max']:.3f}"), ("range", f3(st["range"])),
            ("P1 / P99", f"{st['p1']:.3f} / {st['p99']:.3f}"), ("P5 / P95", f"{st['p5']:.3f} / {st['p95']:.3f}"),
            ("P25 / P75", f"{st['p25']:.3f} / {st['p75']:.3f}"),
            ("skewness", f"{st['skew']:+.3f}"), ("excess kurtosis", f"{st['kurt_excess']:+.3f}"),
            ("within mean+-1 sd", f"{100 * st['within_1sd']:.1f} %  (68.3)"),
            ("within mean+-2 sd", f"{100 * st['within_2sd']:.1f} %  (95.4)"),
            ("within mean+-3 sd", f"{100 * st['within_3sd']:.1f} %  (99.7)"),
        ])

    block(70, "Depth over the frame (m)", st, "N pixels")
    block(400, "Lidar altitude, one per in-frame return (m)", sa, "N returns")
    if sa is not None:
        col(16, 400 + 54 + 21 * 12 + 4, [
            ("rejected by ground fit", f"{fu['alt_n_rejected']}  (objects, spikes)"),
        ], dx=235, scale=0.47)
        col(490, 400 + 54 + 21 * 12 + 4, [
            ("ground fit at nadir", f"{m(fu['alt_ground_fit_m'], '.3f')}   rel_alt {m(fu['rel_alt_m'], '.2f')}"),
        ], dx=185, scale=0.47)

    y = 760
    cv2.line(img, (16, y - 18), (PANEL_W - 16, y - 18), bgr(GRID), 1)
    warn = bool(fu["flags"])
    put(img, f"Lidar + DA3 fusion   [ground from: {fu['source']}]", (16, y + 4), 0.6, C_WARN if warn else TEXT_1)
    col(16, y + 34, [
        ("returns window/foot/frame", f"{fu['n_window']} / {fu['n_footprint']} / {fu['n_frame']}"),
        ("inliers / passes / +-s", f"{fu['n_inliers']} / {fu['n_passes']} / {m(fu['window_s'], '.1f')}"),
        ("lidar range scale", f"{fu['range_scale']:.4f}"),
        ("ground below camera", f"{m(fu['plane_h_m'], '.3f')} m"),
        ("ground tilt", f"{m(fu['plane_tilt_deg'], '.2f')} deg"),
    ], dx=225, scale=0.47)
    col(490, y + 34, [
        ("held-out lidar rms", f"{cm(fu['heldout_rms_m'])}  (n {fu['heldout_n']})"),
        ("held-out lidar p95", cm(fu["heldout_p95_m"])),
        ("depth - lidar rms", f"{cm(fu['frame_resid_rms_m'])}  (noise {cm(fu['resid_noise_m'])})"),
        ("DA3 relief gain g", f"{m(fu['relief_gain'], '.2f')}  {fu['relief'].split(':')[0]}"),
        ("ground - rel_alt", f"{m(fu['plane_minus_rel_alt_m'], '+.2f')} m"),
    ], dx=200, scale=0.47)
    if warn:
        put(img, "; ".join(fu["flags"])[:110], (16, 2 * PANEL_H - 14), 0.45, C_WARN)
    return img


def frame_panel(frame, pts, bag_name, t_rel, n_frame, crop):
    H, W = frame.shape[:2]
    img = cv2.resize(crop_view(frame, (W, H), crop), (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    lidar_dots(img, shift_points(pts, crop), PANEL_W / (W - 2 * crop), PANEL_H / (H - 2 * crop), C_ANCHOR)
    put(img, f"{bag_name}   t = {t_rel:.2f} s", (12, 26), 0.6)
    put(img, f"{n_frame} lidar returns in frame (rings: rejected)", (12, PANEL_H - 12), 0.5)
    return img




def render_loop(jobs, ff, name, crop):
    """Compose the five panels of each job and pipe them to ffmpeg, until a None.

    [frame      ][heatmap      ][statistics]
    [depth hist ][altitude hist][  (tall)  ]
    """
    hist = HistPanel("Depth variation over the frame: histogram and normal curve",
                     "depth along the optical axis (m)", "pixels (histogram)")
    hist_alt = HistPanel("Lidar altitude per return (roll/pitch/yaw + scan angle removed)",
                         "aircraft altitude above the ground point (m)", "returns (histogram)")
    while True:
        job = jobs.get()
        if job is None:
            return
        frame, relief, vrange, st, edges, dens, fu, pts, z_lid, n, t_rel, alt = job
        try:
            H, W = frame.shape[:2]
            title = ("DA3 relief, detrended (display only)" if not fu["relief_gain"] > 0 else
                     f"DA3 relief, detrended (lidar-verified, g {fu['relief_gain']:.2f})")
            top = np.hstack([frame_panel(frame, pts, name, t_rel, fu["n_frame"], crop),
                             heatmap_panel(relief, vrange[0], vrange[1], pts, (W, H), title, crop)])
            sa, ea, da = alt
            refs = [(fu["alt_ground_fit_m"], "ground fit", C_ANCHOR, ":"), (fu["rel_alt_m"], "rel_alt", C_WARN, "-.")]
            bot = np.hstack([hist.draw(st, edges, dens, [(z_lid, "lidar median", C_ANCHOR, ":")]),
                             hist_alt.draw(sa, ea, da, refs, note=f"{fu['alt_n_rejected']} rejected by the ground fit"
                                           if sa is not None else "")])
            img = np.hstack([np.vstack([top, bot]), stats_panel(st, sa, fu, n, t_rel)])
        except Exception as e:  # noqa: BLE001 - one bad panel must not end the video
            log("warn", "render", f"frame {n}: {type(e).__name__}: {e}")
            img = np.zeros((2 * PANEL_H, 3 * PANEL_W, 3), np.uint8)
            img[:PANEL_H, :PANEL_W] = cv2.resize(frame, (PANEL_W, PANEL_H))
        ff.stdin.write(np.ascontiguousarray(img).tobytes())


# ---------------------------------------------------------------------------
def camera_fps(bag, topic):
    import yaml
    try:
        with open(os.path.join(bag, "metadata.yaml")) as f:
            info = yaml.safe_load(f)["rosbag2_bagfile_information"]
        dur = info["duration"]["nanoseconds"] * 1e-9
        n = next(t["message_count"] for t in info["topics_with_message_count"]
                 if t["topic_metadata"]["name"] == topic)
        fps = n / dur
        return float(round(fps)) if abs(fps - round(fps)) < 0.03 * fps else fps
    except Exception:  # noqa: BLE001
        return 30.0


def fusion_row(info, scale):
    """The per-frame fusion numbers, as shown on the panel and written to the CSV."""
    plane, C = info["plane"], info["C"]
    ho, res = info["heldout"], info["resid"]
    ph = plane.height_below(C) if plane is not None else math.nan
    return dict(
        source=plane.source if plane is not None else "none", flags=list(info["flags"]),
        window_s=info["window"], n_window=info["n_window"], n_footprint=info["n_footprint"],
        n_inliers=info["n_inliers"], n_frame=info["n_frame"], n_passes=info["n_passes"], range_scale=scale,
        plane_h_m=ph, plane_tilt_deg=plane.tilt_deg() if plane is not None else math.nan,
        heldout_rms_m=rms(ho), heldout_p95_m=p95abs(ho), heldout_n=len(ho),
        frame_resid_rms_m=rms(res), frame_resid_p95_m=p95abs(res), resid_noise_m=info.get("resid_noise", math.nan),
        relief_gain=info["gain"], relief=info.get("relief", ""),
        rel_alt_m=info["rel_alt"], plane_minus_rel_alt_m=ph - info["rel_alt"],
        roll_deg=info["roll"], pitch_deg=info["pitch"], yaw_deg=info["yaw"],
        alt_ground_fit_m=info.get("alt_plane", math.nan),
        alt_n_rejected=int((~info["alt_inlier"]).sum()) if info.get("alt_inlier") is not None else 0)


def setup(a):
    """(cfg, name, bag, telemetry with world points, range scale, camera mount, camera lever)."""
    bag = os.path.abspath(os.path.expanduser(a.bag))
    if not os.path.isdir(bag):
        sys.exit(f"[abort] no such bag directory: {bag}")
    name = os.path.basename(os.path.normpath(bag))
    sets = list(a.set)
    zoom = a.zoom or ("2x" if "_2x" in name or "2xzoom" in name else
                      "1x" if "_1x" in name or "1xzoom" in name else None)
    if zoom:
        sets.append(f"geo.intrinsics=fcb_{zoom}")
    sets.append('ros.scan_topic=""')    # /scan's beam angles are wrong; the cloud is read instead
    cfg = load_config(a.config, sets)
    g = cfg.geo

    lm = import_lidar_map(cfg.paths.da3_root)
    mount = lm.mount_matrix(lm.MOUNT_ROLL_DEG, lm.MOUNT_PITCH_DEG, lm.MOUNT_YAW_DEG)
    lever_cam = frd_to_flu(g.camera_offset_body)
    if g.get("lidar_offset_body") is None:
        lever_lidar = lever_cam.copy()
        log("warn", "geo", "geo.lidar_offset_body is not set: the lidar is taken to sit at the camera "
                           "(no lidar-camera parallax). Measure it on the airframe -- its vertical part "
                           "enters the depth 1:1.")
    else:
        lever_lidar = frd_to_flu(g.lidar_offset_body)
    R_cam = R_CB @ rot_xyz(g.get("camera_mount_deg") or [0.0, 0.0, 0.0])

    tel = BagTelemetry(bag, cfg, a.cloud_topic, lm)
    if str(a.lidar_scale) == "auto":
        scale, why = tel.calibrate_scale(mount)
        if scale is None:
            scale = float(g.lidar_range_scale)
            log("warn", "lidar", f"range scale: auto calibration not possible ({why}); "
                                 f"using geo.lidar_range_scale {scale}")
        else:
            log("info", "lidar", f"range scale {scale:.4f}, auto: {why}  (config: {g.lidar_range_scale})")
    else:
        scale = float(a.lidar_scale)
        log("info", "lidar", f"range scale {scale} (given)")
    tel.world_points(mount, lever_lidar, scale)
    if str(a.angle_lag) == "auto":
        lag = tel.calibrate_angle_lag(mount, lever_lidar, scale)
        if lag is None:
            log("warn", "lidar", "beam-angle lag: could not be measured (too little airborne data); none applied")
        else:
            tel.angle_lag = lag[0]
            log("info", "lidar", f"beam-angle lag {math.degrees(lag[0]):+.2f} deg by sweep direction "
                                 f"({1e3 * lag[1]:+.2f} ms at the sweep rate): ground spread "
                                 f"{100 * lag[2]:.1f} -> {100 * lag[3]:.1f} cm")
    else:
        tel.angle_lag = math.radians(float(a.angle_lag))
        log("info", "lidar", f"beam-angle lag {float(a.angle_lag):+.2f} deg (given)")
    tel.world_points(mount, lever_lidar, scale)
    noise = tel.calibrate_noise()
    if noise is None:
        log("warn", "lidar", f"noise model: not enough airborne passes to measure it; using range "
                             f"{100 * SIGMA_RANGE:.1f} cm, angle {math.degrees(SIGMA_BEAM):.2f} deg")
    else:
        log("info", "lidar", f"noise model measured on {noise[2]} returns: range {100 * noise[0]:.1f} cm, "
                             f"beam angle {math.degrees(noise[1]):.2f} deg")
    return cfg, name, bag, tel, scale, R_cam, lever_cam


def main():
    a = parse()
    cfg, name, bag, tel, scale, R_cam, lever_cam = setup(a)

    est = DepthEstimator(cfg)
    if not est.available:
        log("warn", "depth", "no DA3 model could be loaded: depth is the lidar ground alone")

    from gate2.sources.bag_offline import BagSource
    src = BagSource(cfg, bag, a.start, a.duration, a.stride, a.max_frames)
    fps = a.fps or camera_fps(bag, src.image_topic) / max(a.stride, 1)

    os.makedirs(cfg.output.dir, exist_ok=True)
    out = a.out or os.path.join(cfg.output.dir, f"{name}_depthstats_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
    out_csv = os.path.splitext(out)[0] + ".csv"
    log("info", "run", f"intrinsics {cfg.geo.intrinsics}, DA3 {est.name if est.available else 'off'} @ {a.res}, "
                       f"window +-{a.window} s, {fps:g} fps -> {out}")

    jobs = queue.Queue(maxsize=4)
    worker = None
    ff = cf = writer = fusion = None
    t0_bag = src.t0_ns * 1e-9 if src.t0_ns else None
    vrange = None
    n = 0
    heldout_all, resid_all, sources = [], [], {}
    t_start = time.monotonic()
    try:
        for frame, stamp in src.frames():
            H, W = frame.shape[:2]
            if fusion is None:
                K, prof = resolve_K(cfg, W, H)
                h, w = infer_size(H, W, a.res)
                fusion = DepthFusion(tel, a, K, W, H, h, w, R_cam, lever_cam)
                log("info", "run", f"{W}x{H}, K ({prof}) fx {K[0]:.1f} fy {K[1]:.1f} cx {K[2]:.1f} cy {K[3]:.1f}")
                ff = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{3 * PANEL_W}x{2 * PANEL_H}", "-r", f"{fps:g}", "-i", "-",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                     # fragmented mp4: playable while it is still being written, and a
                     # killed run keeps everything up to its last 1 s fragment
                     "-g", f"{max(int(round(fps)), 1)}",
                     "-movflags", "+frag_keyframe+empty_moov+default_base_moof", out],
                    stdin=subprocess.PIPE)
                cf = open(out_csv, "w", newline="")
                writer = csv.writer(cf)
                writer.writerow(["frame", "stamp", "t_rel"] + FUSION_KEYS + STAT_KEYS + ["alt_" + k for k in STAT_KEYS])
            t_rel = stamp - t0_bag if t0_bag else 0.0

            z = None
            if est.available:
                try:
                    z = da3_frame(est, frame, a.res)
                except Exception as e:  # noqa: BLE001
                    log("warn", "depth", f"DA3 failed on frame {n}: {e}; lidar ground only")
            try:
                D, info = fusion(stamp, z)
            except Exception as e:  # noqa: BLE001 - one bad frame must not end the run
                log("warn", "fusion", f"frame {n}: {type(e).__name__}: {e}")
                C, _, Rb = fusion.camera(stamp)
                D, info = None, dict(flags=[f"fusion error: {type(e).__name__}"], points=None,
                                     heldout=np.empty(0), resid=np.empty(0), n_inliers=0, n_passes=0,
                                     window=math.nan, n_window=0, n_footprint=0, n_frame=0, gain=math.nan,
                                     C=C, plane=None, rel_alt=math.nan,
                                     **dict(zip(("roll", "pitch", "yaw"), (math.degrees(v) for v in rpy_flu(Rb)))))
            fu = fusion_row(info, scale)
            sources[fu["source"]] = sources.get(fu["source"], 0) + 1
            st = edges = dens = None
            if D is not None and np.isfinite(D).sum() > 100:
                st, edges, dens = depth_stats(D, a.bins)
                relief = info.get("display_relief")
                if relief is not None:
                    # display: a light blur hides DA3's 14-px patch grid, which a strong stretch shows
                    relief = cv2.GaussianBlur(relief.astype(np.float32), (0, 0), 1.5)
                    # colour range: the ground's median +- 4 robust sds, eased as person_parallax.py
                    # does.  P1..P99 let a few people take the whole range and the grass went black.
                    v = crop_view(relief, (W, H), a.crop)[::2, ::2]
                    med = float(np.median(v))
                    sd = 1.4826 * float(np.median(np.abs(v - med)))
                    lo, hi = med - 4 * sd, med + 4 * sd
                    k = 1.0 - (1.0 - a.ema) ** max(a.stride, 1)
                    vrange = (lo, hi) if vrange is None else ((1 - k) * vrange[0] + k * lo,
                                                              (1 - k) * vrange[1] + k * hi)
                if fu["source"] == "lidar":
                    heldout_all.append(info["heldout"])
                    resid_all.append(info["resid"])
            else:
                D = None
            if D is None:
                relief = None
            # the lidar altitude according to each in-frame return
            sa = ea = da = None
            alt = info.get("alt")
            if alt is not None and np.isfinite(alt).sum() >= 5:
                sa, ea, da = depth_stats(alt, int(np.clip(2 * np.sqrt(alt.size), 10, 40)))
            pts = info["points"]
            fr = pts["inframe"] & pts["inliers"] if pts is not None else np.zeros(0, bool)
            z_lid = float(np.median(pts["pc"][fr, 2])) if fr.any() else None

            if worker is None:
                worker = threading.Thread(target=render_loop, args=(jobs, ff, name, a.crop), daemon=True)
                worker.start()
            jobs.put((frame, relief, vrange or (-1.0, 1.0), st, edges, dens, fu, pts, z_lid, n, t_rel, (sa, ea, da)))
            row = [n, f"{stamp:.6f}", f"{t_rel:.4f}"]
            for k in FUSION_KEYS:
                v = fu[k]
                row.append("; ".join(v) if k == "flags" else v if isinstance(v, (str, int)) else f"{v:.6g}")
            row += ([st[k] if k == "n" else f"{st[k]:.6g}" for k in STAT_KEYS] if st else [""] * len(STAT_KEYS))
            row += ([sa[k] if k == "n" else f"{sa[k]:.6g}" for k in STAT_KEYS] if sa else [""] * len(STAT_KEYS))
            writer.writerow(row)
            n += 1
            if n == 1 or n % 100 == 0:
                el = time.monotonic() - t_start
                msg = (f"frame {n}  t {t_rel:.1f} s  {n / el:.2f} fps  [{fu['source']}] "
                       f"{fu['n_footprint']} returns, ground {fu['plane_h_m']:.2f} m, held-out "
                       f"{100 * fu['heldout_rms_m']:.1f} cm, depth-lidar {100 * fu['frame_resid_rms_m']:.1f} cm, "
                       f"g {fu['relief_gain']:.2f}")
                if st:
                    msg += f"  mean {st['mean']:.3f} sd {st['sd']:.3f} m"
                log("info", "run", msg + (f"  ({'; '.join(fu['flags'])})" if fu["flags"] else ""))
    except KeyboardInterrupt:
        pass
    finally:
        if worker is not None:
            jobs.put(None)
            worker.join()
        if ff is not None:
            ff.stdin.close()
            ff.wait()
        if cf is not None:
            cf.close()
        src.close()
    ho = np.concatenate(heldout_all) if heldout_all else np.empty(0)
    rs = np.concatenate(resid_all) if resid_all else np.empty(0)
    log("info", "run", f"{n} frames in {time.monotonic() - t_start:.1f} s -> {out}, {out_csv}")
    log("info", "run", "ground from: " + ", ".join(f"{k} {v}" for k, v in sorted(sources.items())))
    if ho.size:
        log("info", "run", f"lidar frames: held-out ground error rms {100 * rms(ho):.1f} cm, "
                           f"p95 {100 * p95abs(ho):.1f} cm; depth - lidar at in-frame returns rms "
                           f"{100 * rms(rs):.1f} cm, p95 {100 * p95abs(rs):.1f} cm")


if __name__ == "__main__":
    main()
