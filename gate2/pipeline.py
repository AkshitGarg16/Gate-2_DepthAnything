"""Gate2Pipeline: one frame in, people with waist angles out.

    frame ──► PoseEstimator (FlyPose)        boxes + 17 keypoints per person
          ──► DepthEstimator (DA3)           relief map of each person crop
          ──► Geolocator (Gate-1)            keypoint rays, ground positions, GPS
          ──► WaistAngleEstimator            3D hip angle (+ fallbacks)
          ──► Tracker                        identity + rolling median per person

Each stage is isolated: if one raises, the frame is still returned with what
the other stages produced and the failure is shown on the HUD.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .angle import AngleResult, WaistAngleEstimator, build_chains
from .depth_estimator import DepthEstimator
from .geolocator import Geolocator
from .pose_estimator import PoseEstimator
from .tracker import Tracker
from .utils import STATUS, RateMeter, Timer, log
from . import visualize as viz


@dataclass
class Person:
    det: dict
    relief: object
    result: AngleResult
    track: object


@dataclass
class FrameResult:
    frame: np.ndarray
    stamp: float
    tel: object
    people: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


class Gate2Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        log("info", "pipeline", "loading models ...")
        self.pose = PoseEstimator(cfg)
        self.depth = DepthEstimator(cfg)
        self.geo = Geolocator(cfg)
        self.no_lidar = bool(cfg.get("no_lidar", {}).get("enabled", False))
        if self.no_lidar:
            from .no_lidar import RelativeWaistAngleEstimator
            self.angle = RelativeWaistAngleEstimator(cfg)
            STATUS.set("mode", "NO-LIDAR: relative DA3 depth, attitude-normalised frame")
        else:
            self.angle = WaistAngleEstimator(cfg)
        self.tracker = Tracker(cfg)
        self.rate = RateMeter()
        self.need_depth = "3d_depth" in cfg.angle.methods and self.depth.available
        self.max_depth_persons = cfg.depth.get("max_persons", 3)
        self._origin = None

    def process(self, frame, stamp, tel, camera_info=None):
        t_start = time.perf_counter()
        fr = FrameResult(frame=frame, stamp=stamp, tel=tel)
        H, W = frame.shape[:2]
        try:
            geo_ok = self.geo.ensure(W, H, camera_info)
        except Exception as e:  # noqa: BLE001
            geo_ok = False
            fr.errors.append(f"geo: {e}")
        geo = self.geo if geo_ok else None
        if geo_ok:
            STATUS.set("geo", f"{self.geo.impl} | alt {tel.alt_source} | att {tel.att_source} | "
                              f"hdg {tel.heading_source}",
                       degraded=tel.altitude is None or tel.att_source != "imu")

        try:
            with Timer() as t:
                dets = self.pose.detect(frame)
            fr.timings["pose"] = t.ms
        except Exception as e:  # noqa: BLE001
            fr.errors.append(f"pose: {e}")
            log("error", "pose", str(e))
            return fr

        world = None
        if geo is not None and tel is not None and tel.altitude is not None and dets:
            try:
                world = self._world_xy(dets, tel)
            except Exception as e:  # noqa: BLE001
                fr.errors.append(f"geo: {e}")
        tracks = self.tracker.update([d["bbox"] for d in dets], world)
        t_depth = t_angle = 0.0
        # depth for the most confident people only -- it is the expensive part
        order = np.argsort([-d.get("det_score", 0) for d in dets])
        depth_ok = set(order[: self.max_depth_persons].tolist())
        for i, (det, trk) in enumerate(zip(dets, tracks)):
            relief = None
            valid = det["scores"] >= self.cfg.pose.kp_thr
            measurable = bool(build_chains(valid, self.cfg.angle.use_knee_if_no_ankle))
            if self.need_depth and i in depth_ok and geo is not None and measurable:
                try:
                    with Timer() as t:
                        relief = self.depth.relief(frame, det["bbox"])
                    t_depth += t.ms
                except Exception as e:  # noqa: BLE001
                    fr.errors.append(f"depth: {e}")
            try:
                with Timer() as t:
                    res = self.angle.estimate(det, relief, geo, tel, trk.scale_prior())
                t_angle += t.ms
            except Exception as e:  # noqa: BLE001
                res = AngleResult(notes=[f"angle error: {e}"])
                fr.errors.append(f"angle: {e}")
            trk.add(res)
            fr.people.append(Person(det, relief, res, trk))
        fr.timings.update(depth=t_depth, angle=t_angle,
                          total=1000 * (time.perf_counter() - t_start))
        self.rate.tick(fr.timings["total"] / 1000.0)
        return fr

    def _world_xy(self, dets, tel):
        """Ground position (north, east metres) of each box centre.

        Relative to the first GPS fix when there is GPS, otherwise relative to
        the vehicle (which still separates people within a frame, but drifts
        with the vehicle between frames -- the gate then rarely matches and
        the tracker falls back to IoU).
        """
        uv = [((d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2) for d in dets]
        P = self.geo.lift(uv, np.zeros(len(uv)), tel, tel.altitude)
        if tel.lat is None:
            return None
        if self._origin is None:
            self._origin = (tel.lat, tel.lon)
        lat0, lon0 = self._origin
        mlat = 111132.92 - 559.82 * math.cos(2 * math.radians(lat0))
        mlon = 111412.84 * math.cos(math.radians(lat0))
        dn, de = (tel.lat - lat0) * mlat, (tel.lon - lon0) * mlon
        return [None if not np.all(np.isfinite(p[:2])) else (dn + p[0], de + p[1]) for p in P]

    def dump(self, fr, directory, n):
        """Save each person's inputs to the angle stage, for offline analysis."""
        import os
        os.makedirs(directory, exist_ok=True)
        tel = fr.tel
        for p in fr.people:
            rl = p.relief
            np.savez_compressed(
                os.path.join(directory, f"f{n:05d}_t{p.track.id}.npz"),
                keypoints=p.det["keypoints"], scores=p.det["scores"], bbox=p.det["bbox"],
                relief=rl.relief if rl is not None else np.zeros((0, 0)),
                relief_origin=np.array([rl.ox, rl.oy, rl.side]) if rl is not None else np.zeros(3),
                ring_sigma=rl.ring_sigma if rl is not None else np.nan,
                tel=np.array([tel.roll, tel.pitch, tel.heading,
                              np.nan if tel.altitude is None else tel.altitude]),
                frame_size=np.array(fr.frame.shape[:2]),
                crop=fr.frame[rl.oy:rl.oy + rl.side, rl.ox:rl.ox + rl.side] if rl is not None
                else np.zeros((0, 0, 3), np.uint8),
                angle=p.result.angle, method=p.result.method)

    # -- presentation ---------------------------------------------------------
    def render(self, fr, source_name="", extra_hud=None, frame=None, age_s=None):
        img = (frame if frame is not None else fr.frame).copy()
        s = img.shape[1] / 1280.0
        for p in fr.people:
            viz.draw_person(img, p.det, p.result, p.track, self.cfg.pose.kp_thr, s,
                            self.cfg.angle.parallel_max_deg, self.cfg.angle.perpendicular_min_deg)
        if self.cfg.display.show_relief_inset:
            best = max((p for p in fr.people if p.relief is not None),
                       key=lambda p: p.det.get("det_score", 0), default=None)
            if best is not None:
                viz.relief_inset(img, best.relief, best.det, best.result,
                                 size=int(220 * max(s, 0.8)), track_id=best.track.id)
        tel = fr.tel
        lines = [(f"{source_name}  {self.rate.hz:4.1f} Hz  {fr.timings.get('total', 0):5.0f} ms"
                  f"  (pose {fr.timings.get('pose', 0):.0f} | depth {fr.timings.get('depth', 0):.0f})"
                  + (f"  age {age_s * 1000:.0f} ms" if age_s is not None else ""), False)]
        if tel is not None:
            alt = f"{tel.altitude:.1f} m ({tel.alt_source})" if tel.altitude else "n/a"
            gps = f"{tel.lat:.6f},{tel.lon:.6f}" if tel.lat is not None else "no GPS"
            lines.append((f"alt {alt}  roll {math.degrees(tel.roll):+.1f}  pitch "
                          f"{math.degrees(tel.pitch):+.1f}  hdg {math.degrees(tel.heading):.0f}"
                          f" ({tel.heading_source})  {gps}", tel.altitude is None))
        for stage, (text, degraded) in STATUS.items():
            lines.append((f"{stage}: {text}", degraded))
        for e in fr.errors[:3]:
            lines.append((e[:110], True))
        if extra_hud:
            lines += extra_hud
        lines.append((f"people {len(fr.people)}", False))
        viz.hud(img, lines, s)
        return img

    def record(self, fr):
        """One JSON-serialisable dict per person, for the log and the ROS topic."""
        out = []
        tel = fr.tel
        for p in fr.people:
            r = p.result
            ang, meth, n = p.track.smoothed()
            out.append({
                "stamp": fr.stamp, "track": p.track.id,
                "bbox": [round(float(v), 1) for v in p.det["bbox"]],
                "det_score": round(float(p.det.get("det_score", 0)), 3),
                "method": r.method,
                "hip_angle_deg": None if not math.isfinite(r.angle) else round(r.angle, 2),
                "bend_deg": None if not math.isfinite(r.angle) else round(r.bend, 2),
                "hip_angle_smoothed_deg": None if not math.isfinite(ang) else round(ang, 2),
                "sides_deg": {k: round(v, 2) for k, v in r.sides.items() if math.isfinite(v)},
                "angle_2d_deg": None if not math.isfinite(r.angle_2d) else round(r.angle_2d, 2),
                "confidence": round(r.confidence, 3),
                "mode": "no_lidar" if self.no_lidar else "default",
                "geometry": r.geometry,
                "metric": r.metric,
                "stature_m": None if r.stature_m is None else round(r.stature_m, 3),
                "relative_gain": (None if getattr(r, "relative_gain", None) is None
                                  else round(r.relative_gain, 4)),
                "attitude_deg": (list(getattr(r, "attitude_deg", ())) or
                                 (None if tel is None else
                                  [round(math.degrees(v), 2) for v in (tel.roll, tel.pitch, tel.heading)])),
                "relief_snr": None if r.relief_snr is None else round(r.relief_snr, 1),
                "fit_rms": None if r.fit_rms is None else round(r.fit_rms, 3),
                "gps": None if r.gps is None else [round(r.gps[0], 7), round(r.gps[1], 7)],
                "altitude_m": None if tel is None or tel.altitude is None else round(tel.altitude, 2),
                "alt_source": None if tel is None else tel.alt_source,
                "keypoints": np.round(p.det["keypoints"], 1).tolist(),
                "kp_scores": np.round(p.det["scores"], 3).tolist(),
                "orientation": r.orientation,
                "elevation_smoothed_deg": {k: round(v, 1) for k, v in p.track.smoothed_elevation().items()},
                "leg_joint": sorted({ch[5] for ch in r.chains}),
                "notes": r.notes,
                "timings_ms": {k: round(v, 1) for k, v in fr.timings.items()},
            })
        return out


class Outputs:
    """mp4 writer + JSONL log, both optional."""

    def __init__(self, cfg, run_name):
        import os
        o = cfg.output
        os.makedirs(o.dir, exist_ok=True)
        self.video_path = (os.path.join(o.dir, f"{run_name}.mp4") if o.video == "auto" else o.video)
        self.log_path = (os.path.join(o.dir, f"{run_name}.jsonl") if o.log == "auto" else o.log)
        self.writer = None
        self.logf = open(self.log_path, "w") if self.log_path else None
        fr = o.get("frames", "auto")
        self.frames_dir = (os.path.join(o.dir, f"{run_name}_frames") if fr == "auto" else fr)
        self.frames_raw = bool(o.get("frames_raw", False))
        self.frames_all = bool(o.get("frames_all", False))
        self.n_frames = 0
        if self.frames_dir:
            os.makedirs(self.frames_dir, exist_ok=True)
            log("info", "output", f"frames with detections -> {self.frames_dir}/")
        if self.log_path:
            log("info", "output", f"measurements -> {self.log_path}")

    def save_frame(self, idx, stamp, annotated, raw, records):
        """Annotated JPEG (+ optional raw frame) and a JSON sidecar, for frames with people."""
        import os
        if not self.frames_dir or (not records and not self.frames_all):
            return None
        base = os.path.join(self.frames_dir, f"frame_{idx:06d}_{stamp:.3f}")
        cv2.imwrite(base + ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if self.frames_raw:
            cv2.imwrite(base + "_raw.jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(base + ".json", "w") as f:
            json.dump({"frame": idx, "stamp": stamp, "people": records}, f, indent=1)
        self.n_frames += 1
        return base + ".jpg"

    def write(self, annotated, records, fps=10.0):
        if self.video_path:
            if self.writer is None:
                h, w = annotated.shape[:2]
                self.writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                              fps, (w, h))
                log("info", "output", f"video -> {self.video_path}")
            self.writer.write(annotated)
        if self.logf:
            for r in records:
                self.logf.write(json.dumps(r) + "\n")
            self.logf.flush()

    def close(self):
        if self.frames_dir:
            log("info", "output", f"{self.n_frames} frames saved in {self.frames_dir}/")
        if self.writer is not None:
            self.writer.release()
        if self.logf:
            self.logf.close()
