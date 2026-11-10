"""Stage 1 -- person detection and 2D keypoints with FlyPose.

FlyPose is a top-down pipeline: an RT-DETR style detector proposes person
boxes, then a ViTPose model (S or H) regresses 17 COCO keypoints per box.
Both are ONNX and run on onnxruntime; this module reuses FlyPose's own
pre/post-processing (model/src/flypose/inference.py) unchanged, so results
match `run_image_inference.py` exactly.

Fallbacks, in the order they are tried:
  * execution provider: TensorRT -> CUDA -> CPU, whichever this onnxruntime
    build actually has (the stock `onnxruntime` wheel is CPU-only);
  * pose model: pose.model (default H) -> pose.fallback_models (default S),
    both at load time and if inference raises at run time;
  * detector: FlyPose detector -> Ultralytics YOLO/YOLOE (pose.detector_fallback).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from .config import AttrDict
from .utils import STATUS, log


def _load_flypose_module(flypose_root):
    """Import FlyPose's inference.py under a unique name (its own names are generic)."""
    src = Path(flypose_root) / "src" / "flypose"
    if not (src / "inference.py").exists():
        raise FileNotFoundError(f"FlyPose inference.py not found under {src}")
    if "flypose_inference" in sys.modules:
        return sys.modules["flypose_inference"]
    # inference.py does `from visualization import draw_results`
    vspec = importlib.util.spec_from_file_location("visualization", src / "visualization.py")
    vis = importlib.util.module_from_spec(vspec)
    vspec.loader.exec_module(vis)
    sys.modules.setdefault("visualization", vis)
    spec = importlib.util.spec_from_file_location("flypose_inference", src / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["flypose_inference"] = mod
    return mod


def _providers(wanted):
    import onnxruntime as ort
    have = ort.get_available_providers()
    use = [p for p in wanted if p in have] or ["CPUExecutionProvider"]
    if "CPUExecutionProvider" not in use:
        use.append("CPUExecutionProvider")
    return use


def _session(fp, path, providers):
    """Create an ORT session, dropping to CPU if a GPU provider fails to initialise."""
    try:
        s = fp.make_session(path, providers)
    except Exception as e:  # noqa: BLE001 - provider init errors come in many types
        if providers != ["CPUExecutionProvider"]:
            log("warn", "pose", f"{Path(path).name}: {providers[0]} failed ({e}); retrying on CPU")
            s = fp.make_session(path, ["CPUExecutionProvider"])
        else:
            raise
    return s, s.get_providers()[0].replace("ExecutionProvider", "")


class YoloDetector:
    """Ultralytics YOLO / YOLOE person detector, the fallback for FlyPose's detector."""

    def __init__(self, weights, conf):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        try:  # YOLOE is open-vocabulary: tell it what to look for
            self.model.set_classes(["person"], self.model.get_text_pe(["person"]))
            self.person = 0
        except Exception:  # noqa: BLE001 - plain YOLO has no set_classes
            names = getattr(self.model, "names", {0: "person"})
            self.person = next((k for k, v in names.items() if v == "person"), 0)

    def __call__(self, frame):
        r = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        keep = cls == self.person
        return (r.boxes.xyxy.cpu().numpy()[keep].astype(np.float32),
                r.boxes.conf.cpu().numpy()[keep].astype(np.float32))


class PoseEstimator:
    """detect(frame_bgr) -> list of {bbox, det_score, keypoints (17,2), scores (17,)}."""

    def __init__(self, cfg):
        self.cfg = cfg
        p = cfg.pose
        self.fp = _load_flypose_module(cfg.paths.flypose_root)
        self.providers = _providers(p.providers)
        ck = Path(cfg.paths.flypose_root) / "checkpoints"

        self.det_cfg = AttrDict(
            onnx_path=str(ck / "detector" / "flyposeDetector.onnx"),
            input_size=(1280, 1280), input_name="images",
            orig_size_input_name="orig_target_sizes", person_label=0,
            conf_threshold=p.det_conf, iou_threshold=p.det_iou,
            resize_mode=p.resize_mode, letterbox_long_side=p.letterbox_long_side,
            pad_value=114, to_rgb=p.get("det_to_rgb", True))
        self.pose_cfg = AttrDict(
            input_size=(192, 256), padding=1.1, to_rgb=True,
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
            keypoint_threshold=p.kp_thr)

        self.detector = None
        self.det_name = None
        for name in [p.detector] + ([p.detector_fallback] if p.detector_fallback else []):
            try:
                self.detector, self.det_name = self._make_detector(name), name
                break
            except Exception as e:  # noqa: BLE001
                log("warn", "pose", f"detector '{name}' unavailable: {e}")
        if self.detector is None:
            raise RuntimeError("no person detector could be loaded (see warnings above)")

        self.pose_chain = [p.model] + [m for m in p.fallback_models if m != p.model]
        self.pose_sess = None
        self._next_pose_model(ck)
        self._report()

    # -- construction ------------------------------------------------------
    def _make_detector(self, name):
        if name == "flypose":
            sess, prov = _session(self.fp, self.det_cfg.onnx_path, self.providers)
            self.det_prov = prov

            def run(frame):
                return self.fp.run_detector(sess, frame, self.det_cfg, return_scores=True)
            return run
        if name == "yolo":
            self.det_prov = "torch"
            return YoloDetector(self.cfg.pose.yolo_weights, self.cfg.pose.det_conf)
        raise ValueError(f"unknown detector '{name}'")

    def _next_pose_model(self, ck=None):
        ck = ck or Path(self.cfg.paths.flypose_root) / "checkpoints"
        while self.pose_chain:
            m = self.pose_chain.pop(0)
            path = ck / "pose" / f"flypose_{m}" / "end2end.onnx"
            try:
                self.pose_sess, self.pose_prov = _session(self.fp, str(path), self.providers)
                self.pose_name = f"flypose_{m}"
                return
            except Exception as e:  # noqa: BLE001
                log("warn", "pose", f"pose model flypose_{m} failed to load: {e}")
        raise RuntimeError("no FlyPose pose model could be loaded")

    def _report(self):
        degraded = (self.pose_name != f"flypose_{self.cfg.pose.model}"
                    or self.det_name != self.cfg.pose.detector)
        STATUS.set("pose", f"{self.det_name} det [{self.det_prov}] + {self.pose_name} "
                           f"[{self.pose_prov}]", degraded=degraded)

    # -- inference ---------------------------------------------------------
    def detect(self, frame):
        boxes, scores = self.detector(frame)
        if len(boxes):
            side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
            keep = side >= self.cfg.pose.min_box_px
            boxes, scores = boxes[keep], scores[keep]
            order = np.argsort(-scores)[: self.cfg.pose.max_persons]
            boxes, scores = boxes[order], scores[order]
        while True:
            try:
                dets = self.fp.run_pose(self.pose_sess, frame, boxes, self.pose_cfg, scores)
                break
            except Exception as e:  # noqa: BLE001 - e.g. CUDA OOM inside ORT
                log("warn", "pose", f"{self.pose_name} failed at run time: {e}")
                self._next_pose_model()
                self._report()
        if self.cfg.pose.primary_only:
            dets = self.fp.select_primary(dets)
        return dets
