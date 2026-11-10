"""Stage 3 -- put keypoints on the ground in metres (Gate-1 geolocalisation).

The Gate-1 pipeline geolocates a casualty by casting the ray through a pixel
(intrinsics K), rotating it camera -> body -> world NED with the vehicle's
roll/pitch/heading, and intersecting it with the ground plane at the
vehicle's height (GATE_1_PIPELINE/deployment_gate_1/coordinate_transformer.py).
This module imports that exact class and uses it for two things:

  * the casualty's GPS position (pixel_to_gps on the mid-hip, unchanged), and
  * a 3D position for every keypoint. Gate-1 intersects with the ground; a
    keypoint is *above* the ground (a raised shoulder), so here the ray is
    stopped at the keypoint's height h instead:  P = C + s * r  with
    P_down = -h. With h = 0 this is exactly Gate-1's ground intersection.

If the Gate-1 checkout is missing, a vendored copy of the same class
(_gate1_transformer.py) is used and the HUD says so.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .utils import STATUS, log

# Measured calibration of the FCB camera on the fcb bags, for the full
# 3840x2160 sensor (from ~/Depth-Anything-3/ros_io.py REAL_K).
FCB_K = (3168.05596, 3163.04946, 1906.55092, 1037.94710)
FCB_K_SIZE = (3840, 2160)
# The matrix hard-coded in GATE_1 v2/Pipeline_yoloe_logs_robust.py (1920x1080).
GATE1_K = (4335.26660285, 4335.94313683, 971.38807875, 499.13445932)


def _load_transformer_class(gate1_root):
    p = Path(gate1_root) / "coordinate_transformer.py"
    if p.exists():
        try:
            spec = importlib.util.spec_from_file_location("gate1_coordinate_transformer", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.CoordinateTransformer, f"Gate-1 ({p.parent.name})"
        except Exception as e:  # noqa: BLE001
            log("warn", "geo", f"could not import {p}: {e}")
    from ._gate1_transformer import CoordinateTransformer
    return CoordinateTransformer, "vendored copy"


def resolve_K(cfg, W, H, camera_info=None):
    """(fx, fy, cx, cy) for a W x H stream from the configured profile."""
    g = cfg.geo
    prof = g.intrinsics
    if prof == "camera_info":
        if camera_info is None:
            return None, "camera_info (not received yet)"
        fx, fy, cx, cy, w, h = camera_info
        if w and h and (w, h) != (W, H):
            sx, sy = W / w, H / h
            fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        return (fx, fy, cx, cy), "camera_info"
    if prof == "gate1":
        fx, fy, cx, cy = GATE1_K
        if (W, H) != (1920, 1080):
            s = W / 1920.0
            fx, fy, cx, cy = fx * s, fy * s, cx * s, cy * s
        return (fx, fy, cx, cy), "gate1"
    if prof in ("fcb_2x", "fcb_1x"):
        K, (cw, ch), mode = FCB_K, FCB_K_SIZE, ("crop" if prof == "fcb_2x" else "scale")
    elif prof == "custom":
        K, (cw, ch), mode = tuple(g.custom_K), tuple(g.custom_K_size), g.custom_K_mode
    else:
        raise ValueError(f"unknown geo.intrinsics profile '{prof}'")
    fx, fy, cx, cy = K
    if mode == "crop":       # digital zoom: centred crop, focal unchanged
        return (fx, fy, cx - (cw - W) / 2.0, cy - (ch - H) / 2.0), prof
    if mode == "scale":      # resized frame: everything scales
        sx, sy = W / cw, H / ch
        return (fx * sx, fy * sy, cx * sx, cy * sy), prof
    return (fx, fy, cx, cy), prof


class Geolocator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cls, self.impl = _load_transformer_class(cfg.paths.gate1_root)
        self.tf = None
        self.size = None
        self.K = None
        self.offset = np.asarray(cfg.geo.camera_offset_body, float)

    def ensure(self, W, H, camera_info=None):
        """(Re)build the transformer when the frame size or intrinsics become known."""
        if self.tf is not None and self.size == (W, H):
            return True
        K, how = resolve_K(self.cfg, W, H, camera_info)
        if K is None:
            STATUS.set("geo", f"waiting for {how}", degraded=True)
            return False
        fx, fy, cx, cy = K
        self.K = K
        self.tf = self.cls(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
        self.size = (W, H)
        log("info", "geo", f"intrinsics '{how}' for {W}x{H}: fx={fx:.1f} fy={fy:.1f} "
                           f"cx={cx:.1f} cy={cy:.1f}  [{self.impl}]")
        if not (0 < cx < W and 0 < cy < H):
            log("warn", "geo", "principal point outside the frame -- wrong intrinsics profile?")
        return True

    # -- geometry ----------------------------------------------------------
    def rays(self, uv, tel):
        """World NED unit rays (N,3) through pixels uv (N,2)."""
        z = self.cfg.geo.zoom_ratio
        out = []
        for u, v in uv:
            r = self.tf.pixel_to_camera_ray(float(u), float(v), z)
            r = self.tf.camera_to_body_frame(r)
            out.append(self.tf.body_to_world_frame(r, tel.roll, tel.pitch, tel.heading))
        return np.asarray(out, float)

    def camera_ned(self, tel, altitude, offset_scale=1.0):
        """Camera position in local NED (origin = point on the ground below the vehicle).

        offset_scale converts the metric lever arm (geo.camera_offset_body)
        into the geometry's units: 1 for metres, 1/altitude for units of the
        camera height, 0 when that conversion is unknown.
        """
        off = self.tf.body_to_world_frame(self.offset * offset_scale, tel.roll, tel.pitch,
                                          tel.heading)
        return np.array([off[0], off[1], -altitude + off[2]])

    def lift(self, uv, heights, tel, altitude, offset_scale=1.0):
        """3D NED points of pixels uv that sit `heights` metres above the ground.

        Returns (N,3) with z = -height, or rows of nan where a ray does not
        point down far enough to reach that height.
        """
        r = self.rays(uv, tel)
        C = self.camera_ned(tel, altitude, offset_scale)
        h = np.asarray(heights, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (-h - C[2]) / r[:, 2]
        bad = (r[:, 2] <= 1e-6) | ~(s > 0)
        P = C[None, :] + s[:, None] * r
        P[bad] = np.nan
        return P

    def gps(self, u, v, tel):
        if tel.lat is None or tel.altitude is None:
            return None
        res = self.tf.pixel_to_gps(float(u), float(v), tel.lat, tel.lon, tel.altitude,
                                   tel.roll, tel.pitch, tel.heading, 0.0,
                                   self.cfg.geo.zoom_ratio)
        return res

    def gsd(self, u, v, tel, altitude):
        """Ground sample distance (m/px) at pixel (u, v)."""
        P = self.lift([(u, v), (u + 1.0, v)], [0.0, 0.0], tel, altitude)
        return float(np.linalg.norm(P[1, :2] - P[0, :2]))
