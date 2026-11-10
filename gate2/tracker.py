"""Frame-to-frame identity and smoothing of the angle per person.

A greedy tracker: on ground position (metres, from the geolocator) when the
telemetry allows it, else on image IoU.
Each track keeps a rolling window of its measurements and reports their
median, so one bad keypoint frame does not flick the number on screen. The
median is taken only over measurements of the best method seen in the window
(a 3D reading is never averaged with a 2D one).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .angle import METHOD_WEIGHT


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    def __init__(self, tid, box, window, world=None):
        self.id = tid
        self.box = np.asarray(box, float)
        self.world = world
        self.missed = 0
        self.hits = 0
        self.hist = deque(maxlen=window)    # (method, angle, confidence)
        self.scales = deque(maxlen=3 * window)   # DA3 m/unit from well-fitted frames
        self.elev = deque(maxlen=window)         # {vector name: elevation deg}

    def add(self, res):
        if math.isfinite(res.angle):
            self.hist.append((res.method, res.angle, res.confidence))
        if res.orientation:
            self.elev.append({k: v["elevation_deg"] for k, v in res.orientation.items()
                              if isinstance(v, dict) and v})
        if (res.method.startswith("3d_depth") and res.scale and res.scale > 0
                and res.fit_rms is not None and res.fit_rms < 0.5):
            self.scales.append(res.scale)

    def smoothed_elevation(self):
        """{vector name: median elevation (deg) over the window}."""
        keys = {k for e in self.elev for k in e}
        return {k: float(np.median([e[k] for e in self.elev if k in e])) for k in keys}

    def scale_prior(self):
        """Median DA3 scale of this person over recent frames (None until 3 samples)."""
        return float(np.median(self.scales)) if len(self.scales) >= 3 else None

    def smoothed(self):
        """(angle, method, n) -- median over the best method present in the window."""
        if not self.hist:
            return float("nan"), "none", 0
        best = max(self.hist, key=lambda h: METHOD_WEIGHT.get(h[0], 0))[0]
        rank = METHOD_WEIGHT.get(best, 0)
        vals = [a for m, a, _ in self.hist if METHOD_WEIGHT.get(m, 0) >= rank - 0.15]
        return float(np.median(vals)), best, len(vals)


class Tracker:
    def __init__(self, cfg):
        self.iou_thr = cfg.angle.track_iou
        self.max_missed = cfg.angle.track_max_missed
        self.window = cfg.angle.smoothing_window
        self.world_gate = cfg.angle.get("track_world_gate_m", 2.0)
        self.tracks = []
        self.next_id = 1

    def update(self, boxes, world=None):
        """Returns the track for each box (same order).

        world: optional (N,2) ground positions in metres (local north/east from
        the geolocator). The camera moves, so between two processed frames a
        person can shift hundreds of pixels while standing still on the
        ground; when both the track and the detection have a ground position
        they are matched on distance (within world_gate_m) instead of IoU.
        """
        world = [None] * len(boxes) if world is None else list(world)

        def score(t, b, w):
            if t.world is not None and w is not None:
                d = float(np.hypot(*(np.asarray(t.world) - np.asarray(w))))
                return 1.0 - d / self.world_gate if d < self.world_gate else 0.0
            return iou(t.box, b)

        pairs = sorted(((score(t, b, world[bi]), ti, bi) for ti, t in enumerate(self.tracks)
                        for bi, b in enumerate(boxes)), reverse=True)
        used_t, used_b, out = set(), set(), [None] * len(boxes)
        for v, ti, bi in pairs:
            if v < self.iou_thr or ti in used_t or bi in used_b:
                continue
            used_t.add(ti); used_b.add(bi)
            t = self.tracks[ti]
            t.box, t.missed = np.asarray(boxes[bi], float), 0
            if world[bi] is not None:
                t.world = world[bi]
            t.hits += 1
            out[bi] = t
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for bi, b in enumerate(boxes):
            if out[bi] is None:
                t = Track(self.next_id, b, self.window, world[bi])
                self.next_id += 1
                self.tracks.append(t)
                out[bi] = t
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return out
