"""Synthetic end-to-end checks of the angle geometry (no models needed).

A 3D body is posed with a known hip angle, projected into the camera through
the same Gate-1 transformer the pipeline uses, and given a fake DA3 relief
(true height x an arbitrary unknown scale + noise). The estimator must recover
the angle without being told the scale. Run:  python3 -m pytest tests -q
or simply:  python3 tests/test_angle.py
"""
import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gate2 import keypoints as K                      # noqa: E402
from gate2.angle import WaistAngleEstimator           # noqa: E402
from gate2.config import load_config                  # noqa: E402
from gate2.depth_estimator import Relief              # noqa: E402
from gate2.geolocator import Geolocator               # noqa: E402
from gate2.telemetry import Telemetry                 # noqa: E402

W, H = 1920, 1080


def body(hip_angle_deg, stature=1.75, torso_pitch_deg=0.0, lying_dir_deg=30.0, hip_h=0.12):
    """17 keypoints in local NED for a person whose thighs lie on the ground.

    The torso is raised by (180 - hip_angle) from the thigh line, i.e.
    hip_angle 180 = lying flat, 90 = sitting upright. torso_pitch rotates the
    whole body up (90 = standing straight when hip_angle is 180).
    """
    psi = math.radians(lying_dir_deg)
    fwd = np.array([math.cos(psi), math.sin(psi), 0.0])      # hip -> head direction on ground
    side = np.array([-math.sin(psi), math.cos(psi), 0.0])
    up = np.array([0, 0, -1.0])
    bend = math.radians(180.0 - hip_angle_deg)
    tp = math.radians(torso_pitch_deg)
    # leg direction: from the hip away from the head, pitched by torso_pitch (down for standing)
    leg_dir = -math.cos(tp) * fwd - math.sin(tp) * up
    torso_dir = math.cos(tp + bend) * fwd + math.sin(tp + bend) * up
    hip_c = np.array([3.0, -2.0, -hip_h])
    if torso_pitch_deg:
        hip_c = hip_c + up * (0.49 * stature * math.sin(tp))   # standing: hips at leg length
    P = np.zeros((17, 3))
    for sgn, (s, h, k, a), (e, w) in ((+1, K.SIDES["L"], (K.L_ELB, K.L_WRI)),
                                      (-1, K.SIDES["R"], (K.R_ELB, K.R_WRI))):
        P[h] = hip_c + sgn * side * 0.0955 * stature
        P[s] = hip_c + sgn * side * 0.1295 * stature + torso_dir * 0.288 * stature
        P[k] = P[h] + leg_dir * 0.245 * stature
        P[a] = P[k] + leg_dir * 0.246 * stature
        P[e] = P[s] - torso_dir * 0.186 * stature + sgn * side * 0.03
        P[w] = P[e] - torso_dir * 0.146 * stature
    head = (P[K.L_SHO] + P[K.R_SHO]) / 2 + torso_dir * 0.13 * stature
    for i in range(5):
        P[i] = head
    return P


def project(geo, P, tel, alt):
    """Inverse of Geolocator.lift: NED points -> pixels."""
    R_wb = np.column_stack([geo.tf.body_to_world_frame(e, tel.roll, tel.pitch, tel.heading)
                            for e in np.eye(3)])
    R_bc = np.column_stack([geo.tf.camera_to_body_frame(e) for e in np.eye(3)])
    C = geo.camera_ned(tel, alt)
    fx, fy, cx, cy = geo.K
    out = []
    for p in P:
        q = R_bc.T @ R_wb.T @ (p - C)
        out.append((fx * q[0] / q[2] + cx, fy * q[1] / q[2] + cy))
    return np.array(out)


def make_case(hip_angle, alt=25.0, roll=0.04, pitch=-0.03, heading=1.1, noise=0.01,
              da3_scale=7.3, stature=1.75, standing=False, seed=0):
    cfg = load_config()
    geo = Geolocator(cfg)
    geo.ensure(W, H)
    tel = Telemetry(stamp=0.0, roll=roll, pitch=pitch, heading=heading, altitude=alt)
    P = body(hip_angle, stature, torso_pitch_deg=90.0 if standing else 0.0)
    uv = project(geo, P, tel, alt)
    x1, y1 = uv.min(0) - 10
    x2, y2 = uv.max(0) + 10
    box = np.array([x1, y1, x2, y2])
    # fake DA3 relief on a crop: unit-less height, unknown scale, noise
    side = int(max(x2 - x1, y2 - y1) * 1.6)
    ox, oy = int((x1 + x2) / 2 - side / 2), int((y1 + y2) / 2 - side / 2)
    rng = np.random.default_rng(seed)
    rel = rng.normal(0, noise / da3_scale, (side, side)).astype(np.float32)
    # paint the body as solid limbs: each skeleton link a capsule ~12 cm wide,
    # the trunk a filled quad, heights interpolated along each link
    gsd = alt / geo.K[0]
    body_h = np.zeros((side, side), np.float32)
    links = list(K.SKELETON) + [(K.L_SHO, K.R_HIP), (K.R_SHO, K.L_HIP)]
    import cv2
    for a, b in links:
        for t in np.linspace(0, 1, 40):
            u, v = uv[a] * (1 - t) + uv[b] * t
            h = -(P[a, 2] * (1 - t) + P[b, 2] * t)
            m = np.zeros_like(body_h, np.uint8)
            cv2.circle(m, (int(round(u - ox)), int(round(v - oy))), max(int(0.06 / gsd), 2), 1, -1)
            body_h = np.where(m > 0, np.maximum(body_h, h), body_h)
    # trunk: a solid quad, height interpolated along the mid-hip -> mid-shoulder axis
    quad = np.array([uv[K.L_SHO], uv[K.R_SHO], uv[K.R_HIP], uv[K.L_HIP]]) - (ox, oy)
    m = np.zeros_like(body_h, np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
    ys, xs = np.nonzero(m)
    mh, ms = (uv[K.L_HIP] + uv[K.R_HIP]) / 2 - (ox, oy), (uv[K.L_SHO] + uv[K.R_SHO]) / 2 - (ox, oy)
    ax = ms - mh
    t = np.clip(((np.column_stack([xs, ys]) - mh) @ ax) / max(ax @ ax, 1e-9), 0, 1)
    hh, hs = -(P[K.L_HIP, 2] + P[K.R_HIP, 2]) / 2, -(P[K.L_SHO, 2] + P[K.R_SHO, 2]) / 2
    body_h[ys, xs] = np.maximum(body_h[ys, xs], hh + t * (hs - hh) + 0.1)
    rel += body_h / da3_scale
    relief = Relief(ox, oy, side, rel, noise / da3_scale, False, "synthetic")
    det = {"bbox": box, "keypoints": uv, "scores": np.full(17, 0.9)}
    return cfg, geo, tel, det, relief


def run(hip_angle, **kw):
    cfg, geo, tel, det, relief = make_case(hip_angle, **kw)
    est = WaistAngleEstimator(cfg)
    return est.estimate(det, relief, geo, tel), est, cfg, geo, tel, det


def test_depth_recovers_angle():
    for true in (180, 150, 120, 90, 60):
        r, *_ = run(true)
        assert r.method.startswith("3d_depth"), r.method
        assert abs(r.angle - true) < 8.0, (true, r.angle, r.method, r.notes)


def test_standing_needs_depth():
    # from straight above a standing person's thighs point at the camera, which
    # is ambiguous with knees drawn up; posture=upright resolves it
    cfg0 = load_config(overrides=["angle.posture=upright"])
    _, geo, tel, det, relief = make_case(180, standing=True)
    est = WaistAngleEstimator(cfg0)
    r = est.estimate(det, relief, geo, tel)
    cfg = cfg0
    assert r.method == "3d_depth"
    assert abs(r.angle - 180) < 8.0, r.angle
    # without depth, the anthropometric fallback assumes a person on the ground
    cfg.angle.methods = ["3d_anthro"]
    r2 = est.estimate(det, None, geo, tel)
    assert r2.method == "3d_anthro"


def test_scale_error_tolerance():
    """Altitude / focal-length (scale) errors: what the solve does and does not absorb.

    The stature is solved jointly, so a scale error is absorbed only while the
    implied stature stays near the 1.70 m prior and inside its bounds. Measured
    on the synthetic poses (worst |error| over 180/150/120/100/90/60 deg):
        scale x0.6 29  x0.7 24  x0.8 19  x0.9 13  x1.0 6  x1.1..x1.4 26 deg
    A mid-range pose stays within 8 deg over x0.7..x1.4; that is asserted here,
    the rest is documented in the README (section 9).
    """
    for alt_err in (0.7, 1.4):
        cfg, geo, tel, det, relief = make_case(100)
        tel.altitude *= alt_err
        r = WaistAngleEstimator(cfg).estimate(det, relief, geo, tel)
        assert abs(r.angle - 100) < 8.0, (alt_err, r.angle)


def test_no_altitude_unscaled():
    cfg, geo, tel, det, relief = make_case(110)
    tel.altitude = None
    r = WaistAngleEstimator(cfg).estimate(det, relief, geo, tel)
    assert r.method == "3d_depth" and not r.metric
    assert abs(r.angle - 110) < 8.0, r.angle


def test_anthro_fallback_sitting():
    r, est, cfg, geo, tel, det = run(90)
    cfg.angle.methods = ["3d_anthro"]
    r2 = est.estimate(det, None, geo, tel)
    assert r2.method == "3d_anthro"
    assert abs(r2.angle - 90) < 15.0, r2.angle


def _flat_body(hip_angle, stature=1.75):
    """A body lying flat with legs spread in the ground plane: all joints at one height.

    hip_angle is the in-plane angle between trunk and (both) thighs.
    """
    P = body(180, stature)
    rot = math.radians(180 - hip_angle)
    c, s_ = math.cos(rot), math.sin(rot)
    for j in (K.L_KNE, K.R_KNE, K.L_ANK, K.R_ANK):
        h = P[K.L_HIP] if j in (K.L_KNE, K.L_ANK) else P[K.R_HIP]
        v = P[j] - h
        P[j] = h + np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1], v[2]])
    return P


def test_equal_depth_is_exact():
    """Keypoints at equal depth: the 3D-vector angle must equal the in-plane angle.

    Under camera tilt the raw pixel angle is distorted by perspective; the 3D
    vectors are not. Also checked: no depth signal + a short person (whose
    limbs look foreshortened against the 1.70 m prior) must not get an
    invented rise.
    """
    cfg = load_config()
    geo = Geolocator(cfg)
    geo.ensure(W, H)
    est = WaistAngleEstimator(cfg)
    for true in (170, 140, 110):
        for roll, pitch in ((0.0, 0.0), (0.25, -0.2)):         # up to ~14 deg of tilt
            for stature in (1.75, 1.45):
                tel = Telemetry(stamp=0, roll=roll, pitch=pitch, heading=0.7, altitude=20.0)
                P = _flat_body(true, stature)
                P[:, 2] = -0.12                               # every joint 12 cm up
                uv = project(geo, P, tel, 20.0)
                det = {"bbox": np.r_[uv.min(0) - 10, uv.max(0) + 10], "keypoints": uv,
                       "scores": np.full(17, 0.9)}
                # depth-free: equal-depth method
                r = est.estimate(det, None, geo, tel)
                assert r.method == "3d_equal_depth"
                assert abs(r.angle - true) < 1.0, (true, roll, stature, r.angle)
                # with a flat (noise-only) relief: 3d_depth must not invent a rise
                side = int(np.ptp(uv, 0).max() * 1.8)
                ox, oy = (uv.min(0) + uv.max(0)) / 2 - side / 2
                rel = np.random.default_rng(1).normal(0, 1e-3, (side, side)).astype(np.float32)
                relief = Relief(int(ox), int(oy), side, rel + 0.05, 1e-3, False, "flat")
                r = est.estimate(det, relief, geo, tel)
                assert r.method.startswith("3d_depth"), r.method
                assert abs(r.angle - true) < 1.5, (true, roll, stature, r.angle, r.notes)


def test_equal_depth_without_telemetry_and_missing_joints():
    r, est, cfg, geo, tel, det = run(120)
    r2 = est.estimate(det, None, None, None)
    assert r2.method == "3d_equal_depth" and r2.points.shape == (17, 3)
    det["scores"][[K.L_KNE, K.R_KNE, K.L_ANK, K.R_ANK]] = 0.0
    r3 = est.estimate(det, None, None, None)
    assert r3.method == "none"


def test_ankle_primary_knee_fallback():
    r, est, cfg, geo, tel, det = run(120)
    assert all(ch[5] == "ankle" for ch in r.chains)
    det["scores"][[K.L_ANK, K.R_ANK]] = 0.0
    r2 = est.estimate(det, None, geo, tel)
    assert r2.chains and all(ch[5] == "knee" for ch in r2.chains)
    assert any("hip->knee" in n for n in r2.notes)


def test_orientation():
    # lying flat: both vectors parallel to the ground
    r, *_ = run(180)
    o = r.orientation
    assert o["torso"]["relation"] == "parallel" and o["leg_L"]["relation"] == "parallel", o
    assert o["aligned"]
    # sitting upright: torso perpendicular (up), legs parallel
    r, *_ = run(90)
    o = r.orientation
    assert o["torso"]["relation"] == "perpendicular" and o["torso"]["direction"] == "up", o
    assert o["leg_L"]["relation"] == "parallel" and o["leg_R"]["relation"] == "parallel", o
    assert not o["aligned"]
    # standing: torso up, legs down, both perpendicular
    _, geo, tel, det, relief = make_case(180, standing=True)
    est = WaistAngleEstimator(load_config(overrides=["angle.posture=upright"]))
    o = est.estimate(det, relief, geo, tel).orientation
    assert o["torso"]["relation"] == "perpendicular" and o["torso"]["direction"] == "up", o
    assert o["leg_L"]["relation"] == "perpendicular" and o["leg_L"]["direction"] == "down", o


def _no_lidar_cfg(*extra):
    from gate2.config import DEFAULT_CONFIG
    return load_config(overrides=list(extra), layers=[DEFAULT_CONFIG.parent / "no_lidar.yaml"])


def test_no_lidar_relative_depth_recovers_angle():
    """Relative DA3 depth (depth / ground ratio), no altitude used for geometry."""
    from gate2.no_lidar import RelativeWaistAngleEstimator
    est = RelativeWaistAngleEstimator(_no_lidar_cfg())
    for true in (180, 150, 120, 90, 60):
        _, geo, tel, det, relief = make_case(true)
        relief.ground = np.full_like(relief.relief, 3.7)   # DA3's ground depth, arbitrary units
        r = est.estimate(det, relief, geo, tel)
        assert r.geometry == "relative" and r.method.startswith("3d_depth"), (r.method, r.notes)
        assert abs(r.angle - true) < 8.0, (true, r.angle)
        assert r.stature_m is not None and abs(r.stature_m - 1.75) < 0.35, r.stature_m
        # Same inputs, same altitude: the relative path must reproduce the
        # metric one (it is the same geometry up to a uniform scale)
        r_def = WaistAngleEstimator(load_config()).estimate(det, relief, geo, tel)
        assert abs(r.angle - r_def.angle) < 0.5, (true, r.angle, r_def.angle)
        # with no altitude at all the stature is free and the solve purely relative
        tel3 = copy.copy(tel); tel3.altitude = None
        r3 = est.estimate(det, relief, geo, tel3)
        assert r3.stature_m is None and abs(r3.angle - true) < 10.0, (true, r3.angle)


def test_no_lidar_attitude_normalisation():
    """Denormalised but equivalent attitudes give the same answer."""
    from gate2.no_lidar import RelativeWaistAngleEstimator, normalize_attitude
    est = RelativeWaistAngleEstimator(_no_lidar_cfg())
    _, geo, tel, det, relief = make_case(110)
    relief.ground = np.full_like(relief.relief, 2.0)
    ref = est.estimate(det, relief, geo, tel)
    r0, p0, h0 = tel.roll, tel.pitch, tel.heading
    for roll, pitch, hdg in ((r0 + 2 * math.pi, p0, h0 - 2 * math.pi),
                             (r0 - 4 * math.pi, p0 + 2 * math.pi, h0 + 6 * math.pi),
                             (r0 + math.pi, math.pi - p0, h0 + math.pi)):   # same rotation, |pitch| > 90
        t = copy.copy(tel); t.roll, t.pitch, t.heading = roll, pitch, hdg
        n = normalize_attitude(roll, pitch, hdg)
        assert abs(n[0] - r0) < 1e-9 and abs(n[1] - p0) < 1e-9 and abs(n[2] - h0) < 1e-9, n
        r = est.estimate(det, relief, geo, t)
        assert abs(r.angle - ref.angle) < 1e-6, (roll, pitch, hdg, r.angle, ref.angle)


def test_no_lidar_ignores_scan():
    from gate2.telemetry import TelemetryHub
    hub = TelemetryHub(_no_lidar_cfg())
    assert "scan" not in hub.alt_sources
    hub.on_scan(1.0, np.linspace(-0.6, 0.6, 100), np.full(100, 25.0), 0.2, 50.0)
    assert hub.counts["scan"] == 0 and len(hub.scan) == 0
    assert hub.snapshot(1.0).alt_source == "none"
    cfg = _no_lidar_cfg()
    assert cfg.ros.scan_topic == "" and cfg.no_lidar.enabled


if __name__ == "__main__":
    for true in (180, 150, 120, 90, 60):
        r, est, cfg, geo, tel, det = run(true)
        r2 = est.estimate(det, None, geo, tel)
        print(f"true {true:5.1f} | 3d_depth {r.angle:6.1f} ({r.method}, S={r.stature:.2f} "
              f"snr={r.relief_snr:.0f} rms={r.fit_rms:.3f}) | equal-depth {r2.angle:6.1f} "
              f"| pixel {r.angle_2d:6.1f}")
    _, geo, tel, det, relief = make_case(180, standing=True)
    est = WaistAngleEstimator(load_config(overrides=["angle.posture=upright"]))
    r = est.estimate(det, relief, geo, tel)
    r2 = est.estimate(det, None, geo, tel)
    print(f"standing 180 (posture=upright) | 3d_depth {r.angle:6.1f} | equal-depth {r2.angle:6.1f} | pixel {r.angle_2d:6.1f}")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
