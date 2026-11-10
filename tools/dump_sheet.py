#!/usr/bin/env python3
"""Contact sheet of dumped people: crop + skeleton | DA3 relief, one row per dump.
    python3 tools/dump_sheet.py outputs/dump_25m out.jpg f00051_t8 f00069_t11 ...
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gate2 import keypoints as K  # noqa: E402

d, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for n in names:
    z = np.load(os.path.join(d, n + ".npz"))
    crop, rel = z["crop"].copy(), z["relief"]
    if crop.size == 0:
        continue
    ox, oy, side = z["relief_origin"].astype(int)
    S = 320
    k = S / side
    a = cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(rel, 2), np.percentile(rel, 99.5)
    b = cv2.applyColorMap((np.clip((rel - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8),
                          cv2.COLORMAP_TURBO)
    b = cv2.resize(b, (S, S), interpolation=cv2.INTER_NEAREST)
    kp, sc = z["keypoints"], z["scores"]
    for img in (a, b):
        for li, (i, j) in enumerate(K.SKELETON):
            if sc[i] > 0.3 and sc[j] > 0.3:
                p = tuple(((kp[i] - (ox, oy)) * k).astype(int)); q = tuple(((kp[j] - (ox, oy)) * k).astype(int))
                cv2.line(img, p, q, K.LINK_COLORS[li], 1, cv2.LINE_AA)
    cv2.putText(a, f"{n} {float(z['angle']):.0f} {z['method']}", (4, 16), 0, 0.45, (255, 255, 255), 1)
    rows.append(np.hstack([a, b]))
cv2.imwrite(out, np.vstack(rows))
