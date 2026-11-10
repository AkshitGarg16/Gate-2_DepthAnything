#!/usr/bin/env python3
"""Re-run the angle stage on saved dumps (run_gate2.py --dump DIR), no models needed.

Lets the fusion be tuned on real detections in seconds. Prints one row per
person: method, angle, equal-depth angle, pixel angle, stature, scale, SNR.
    python3 tools/replay_dumps.py outputs/dump_25m [--set key=value ...]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gate2.angle import WaistAngleEstimator   # noqa: E402
from gate2.config import load_config           # noqa: E402
from gate2.depth_estimator import Relief       # noqa: E402
from gate2.geolocator import Geolocator        # noqa: E402
from gate2.telemetry import Telemetry          # noqa: E402


def load(path, cfg, geo):
    z = np.load(path)
    H, W = (int(v) for v in z["frame_size"])
    geo.ensure(W, H)
    roll, pitch, hdg, alt = z["tel"]
    tel = Telemetry(0.0, float(roll), float(pitch), float(hdg),
                    None if np.isnan(alt) else float(alt))
    det = {"bbox": z["bbox"], "keypoints": z["keypoints"], "scores": z["scores"]}
    rel = None
    if z["relief"].size:
        ox, oy, side = (int(v) for v in z["relief_origin"])
        rel = Relief(ox, oy, side, z["relief"], float(z["ring_sigma"]), False, "dump")
    return det, rel, tel, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(None, a.set)
    geo = Geolocator(cfg)
    est = WaistAngleEstimator(cfg)
    print(f"{'file':18s} {'method':15s} {'angle':>6s} {'eqdep':>6s} {'pixel':>6s} "
          f"{'S':>5s} {'a':>7s} {'snr':>6s} {'rms':>5s}  notes")
    from collections import defaultdict
    scales = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.dir, "*.npz"))):
        det, rel, tel, _ = load(f, cfg, geo)
        tid = os.path.basename(f).split("_t")[-1][:-4]
        sp = float(np.median(scales[tid][-21:])) if len(scales[tid]) >= 3 else None
        r = est.estimate(det, rel, geo, tel, sp)
        if r.method.startswith("3d_depth") and r.scale and r.fit_rms is not None and r.fit_rms < 0.5:
            scales[tid].append(r.scale)
        r0 = est.estimate(det, None, geo, tel)
        fmt = lambda v, p=1: "  -  " if v is None or not np.isfinite(v) else f"{v:.{p}f}"  # noqa
        print(f"{os.path.basename(f)[:-4]:18s} {r.method:15s} {fmt(r.angle):>6s} {fmt(r0.angle):>6s} "
              f"{fmt(r.angle_2d):>6s} {fmt(r.stature, 2):>5s} {fmt(r.scale, 2):>7s} "
              f"{fmt(r.relief_snr, 0):>6s} {fmt(r.fit_rms, 2):>5s}  {(r.orientation or {}).get('summary', '')[:80]}")


if __name__ == "__main__":
    main()
