#!/usr/bin/env python3
"""Check what Gate-2 will run on, before running it.

Prints, per stage, whether the primary choice is available and which fallback
will be used otherwise. Loads nothing heavy (no model weights).
    python3 tools/check_env.py [--config FILE] [--set key=value ...]
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gate2.config import load_config  # noqa: E402

OK, WARN, BAD = "\033[32m ok \033[0m", "\033[33mwarn\033[0m", "\033[31mFAIL\033[0m"


def line(state, what, detail=""):
    print(f"[{state}] {what:42s} {detail}")


def has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    p = cfg.paths

    print("== Python packages")
    for m, need in [("numpy", True), ("cv2", True), ("yaml", True), ("scipy", True),
                    ("onnxruntime", True), ("torch", False), ("safetensors", False),
                    ("av", False), ("ultralytics", False), ("rclpy", False), ("rosbag2_py", False),
                    ("ffmpeg_image_transport_msgs.msg", False)]:
        line(OK if has(m) else (BAD if need else WARN), m,
             "" if has(m) else ("required" if need else "optional: see README fallbacks"))

    print("== Stage 1: FlyPose")
    import onnxruntime as ort
    prov = ort.get_available_providers()
    gpu = [x for x in ("TensorrtExecutionProvider", "CUDAExecutionProvider") if x in prov]
    line(OK if gpu else WARN, "onnxruntime providers", ", ".join(prov) +
         ("" if gpu else "  -> FlyPose runs on CPU (pip install onnxruntime-gpu for CUDA)"))
    ck = os.path.join(p.flypose_root, "checkpoints")
    for rel in ("detector/flyposeDetector.onnx", f"pose/flypose_{cfg.pose.model}/end2end.onnx",
                *[f"pose/flypose_{m}/end2end.onnx" for m in cfg.pose.fallback_models]):
        f = os.path.join(ck, rel)
        line(OK if os.path.exists(f) else BAD, rel, "" if os.path.exists(f) else f"missing: {f}")
    if cfg.pose.detector_fallback == "yolo":
        f = cfg.pose.yolo_weights
        line(OK if os.path.exists(f) and has("ultralytics") else WARN, "YOLO fallback detector",
             f if os.path.exists(f) else f"missing {f}")

    print("== Stage 2: Depth-Anything-3")
    src = os.path.join(p.da3_root, "src", "depth_anything_3")
    line(OK if os.path.isdir(src) else BAD, "depth_anything_3 package", src)
    from gate2.depth_estimator import DA3_MODELS
    for m in cfg.depth.models:
        leaf = DA3_MODELS[m][1].split("/")[-1]
        hit = next((os.path.join(b, leaf) for b in p.da3_weights_dirs
                    if os.path.exists(os.path.join(b, leaf, "model.safetensors"))), None)
        line(OK if hit else WARN, f"DA3 {m} weights", hit or "not local (HF download will be tried)")
    if has("torch"):
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            line(OK, "CUDA for DA3", f"{torch.cuda.get_device_name(0)}, {gb:.1f} GiB")
        else:
            line(WARN, "CUDA for DA3", "not available -> DA3 on CPU (~20x slower)")

    print("== Stage 3: Gate-1 geolocalisation")
    f = os.path.join(p.gate1_root, "coordinate_transformer.py")
    line(OK if os.path.exists(f) else WARN, "Gate-1 CoordinateTransformer",
         f if os.path.exists(f) else "missing -> vendored copy gate2/_gate1_transformer.py")
    line(OK, "intrinsics profile", cfg.geo.intrinsics)
    line(OK, "altitude fallback chain", " -> ".join(cfg.geo.altitude_sources))

    print("== Stage 4: angle")
    line(OK, "method chain", " -> ".join(cfg.angle.methods))
    line(OK, "posture prior", cfg.angle.get("posture", "ground"))

    print("== Display")
    d = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    line(OK if d else WARN, "display", d or "none -> headless (use --save-video)")


if __name__ == "__main__":
    main()
