"""Overlay: skeleton, torso/thigh vectors, the angle arc and a status HUD."""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import keypoints as K

METHOD_COLOR = {"3d_depth": (60, 220, 60), "3d_depth_flat": (60, 220, 60),
                "3d_equal_depth": (0, 200, 255), "3d_anthro": (0, 140, 255),
                "none": (160, 160, 160)}
METHOD_LABEL = {"3d_depth": "3D depth", "3d_depth_flat": "3D depth (flat)",
                "3d_equal_depth": "3D equal depth", "3d_anthro": "3D proportions",
                "none": "--"}


def _txt(img, s, org, scale=0.55, color=(255, 255, 255), thick=1, bg=True):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg:
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + b + 2), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return h + b + 8


def _relation(elev, par, perp):
    a = abs(elev)
    return "parallel" if a <= par else "perpendicular" if a >= perp else "oblique"


def draw_person(img, det, res, track, kp_thr, s=1.0, par=20.0, perp=70.0):
    """Draw one person. s = display scale factor (line widths, fonts)."""
    kp, sc = det["keypoints"], det["scores"]
    lw = max(1, int(round(2 * s)))
    x1, y1, x2, y2 = np.round(det["bbox"]).astype(int)
    col = METHOD_COLOR.get(res.method, (200, 200, 200))
    cv2.rectangle(img, (x1, y1), (x2, y2), col, lw)
    for li, (a, b) in enumerate(K.SKELETON):
        if sc[a] >= kp_thr and sc[b] >= kp_thr:
            cv2.line(img, tuple(np.round(kp[a]).astype(int)), tuple(np.round(kp[b]).astype(int)),
                     K.LINK_COLORS[li], lw, cv2.LINE_AA)
    for i in range(17):
        if sc[i] >= kp_thr:
            cv2.circle(img, tuple(np.round(kp[i]).astype(int)), max(2, lw + 1), (255, 255, 255), -1)

    # hip->shoulder (magenta) and hip->ankle (yellow) vectors, and the arc at the hip
    for name, sho, thip, vhip, leg, _ in res.chains:
        ph, pt = kp[vhip].mean(0), kp[thip].mean(0)
        ps, pl = kp[sho].mean(0), kp[leg].mean(0)
        cv2.line(img, tuple(np.round(pt).astype(int)), tuple(np.round(ps).astype(int)),
                 (255, 0, 255), lw + 1, cv2.LINE_AA)
        cv2.line(img, tuple(np.round(ph).astype(int)), tuple(np.round(pl).astype(int)),
                 (0, 255, 255), lw + 1, cv2.LINE_AA)
        a1 = math.degrees(math.atan2(*(ps - pt)[::-1]))
        a2 = math.degrees(math.atan2(*(pl - ph)[::-1]))
        d = (a2 - a1 + 540) % 360 - 180
        r = max(8, int(0.25 * min(np.linalg.norm(ps - pt) + 1, np.linalg.norm(pl - ph) + 1)))
        cv2.ellipse(img, tuple(np.round(ph).astype(int)), (r, r), 0, a1, a1 + d,
                    (255, 255, 255), lw, cv2.LINE_AA)

    ang, meth, n = track.smoothed() if track is not None else (res.angle, res.method, 1)
    tid = f"#{track.id} " if track is not None else ""
    fs = 0.55 * max(s, 0.8)
    y = y1 - 8
    lines = []
    if math.isfinite(ang):
        lines.append((f"{tid}hip {ang:5.1f} deg  bend {180 - ang:5.1f}", col, 2))
        sides = "  ".join(f"{k} {v:.0f}" for k, v in res.sides.items() if math.isfinite(v))
        extra = f"{METHOD_LABEL.get(meth, meth)}{' (relative)' if res.geometry == 'relative' else ''}" \
                f"  conf {res.confidence:.2f}"
        if sides:
            extra += f"  [{sides}]"
        lines.append((extra, (230, 230, 230), 1))
        el = track.smoothed_elevation() if track is not None else \
            {k: v["elevation_deg"] for k, v in res.orientation.items() if isinstance(v, dict) and v}
        if "torso" in el:
            tag = "" if res.orientation.get("measured", True) else " (assumed)"
            lines.append((f"hip-shoulder {el['torso']:+.0f} deg {_relation(el['torso'], par, perp)}{tag}",
                          (255, 0, 255), 1))
            legs = [(k[4:], v) for k, v in sorted(el.items()) if k.startswith("leg_")]
            if legs:
                kinds = {ch[0]: ch[5] for ch in res.chains}
                lines.append(("hip-" + "/".join(sorted(set(kinds.values())) or ["ankle"]) + " " + "  ".join(f"{n} {v:+.0f} {_relation(v, par, perp)}"
                                                        for n, v in legs), (0, 255, 255), 1))
            if res.orientation.get("aligned"):
                lines.append(("torso & leg aligned (straight)", (230, 230, 230), 1))
        if res.stature_m:
            lines.append((f"stature {res.stature_m:.2f} m", (200, 200, 200), 1))
        if res.gps:
            lines.append((f"{res.gps[0]:.6f}, {res.gps[1]:.6f}", (200, 200, 200), 1))
    else:
        lines.append((f"{tid}angle n/a: {res.notes[0] if res.notes else ''}", col, 1))
    hgt = int(22 * fs / 0.55)
    y = y - hgt * (len(lines) - 1)
    if y < int(170 * s):          # would sit under the HUD: put the label below the box
        y = y2 + hgt
    for text, c, th in lines:
        y += _txt(img, text, (x1, y), fs, c, th) - 4


def relief_inset(img, relief, det, res, size=220, track_id=None):
    """DA3 relief of one person as a colour map in the top-right corner."""
    if relief is None:
        return
    rel = relief.relief
    lo, hi = np.percentile(rel, 2), np.percentile(rel, 99.5)
    norm = np.clip((rel - lo) / max(hi - lo, 1e-9), 0, 1)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cm = cv2.resize(cm, (size, size), interpolation=cv2.INTER_NEAREST)
    k = size / relief.side
    for i in range(17):
        if det["scores"][i] >= 0.3:
            u, v = det["keypoints"][i]
            cv2.circle(cm, (int((u - relief.ox) * k), int((v - relief.oy) * k)), 3, (255, 255, 255), -1)
    H, W = img.shape[:2]
    x0, y0 = W - size - 10, 10
    if x0 < 0 or y0 + size > H:
        return
    img[y0:y0 + size, x0:x0 + size] = cm
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 1)
    who = f"#{track_id} " if track_id is not None else ""
    _txt(img, f"{who}DA3 relief ({relief.model})", (x0 + 4, y0 + size + 18), 0.45)


def hud(img, lines, s=1.0):
    y = int(24 * max(s, 0.8))
    for text, degraded in lines:
        y += _txt(img, text, (10, y), 0.5 * max(s, 0.8), (0, 200, 255) if degraded else (255, 255, 255))
