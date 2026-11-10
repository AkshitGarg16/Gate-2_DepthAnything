"""Plain video file or camera index -- the no-ROS fallback.

There is no telemetry, so the angle runs without altitude (unscaled geometry,
camera assumed level) unless geo.fixed_altitude_m is set.
"""
from __future__ import annotations

import time

import cv2

from ..telemetry import TelemetryHub
from ..utils import log


class VideoSource:
    live = False

    def __init__(self, cfg, path, start_s=0.0, stride=1, max_frames=None, realtime=False):
        self.cfg = cfg
        self.hub = TelemetryHub(cfg, live=False)
        src = int(path) if str(path).isdigit() else path
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {path}")
        self.live = isinstance(src, int)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if start_s and not self.live:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        self.stride, self.max_frames, self.realtime = max(1, stride), max_frames, realtime
        self.name = str(path)
        log("info", "source", f"video {path} @ {self.fps:.2f} fps (no telemetry)")

    def frames(self):
        i = n = 0
        t0 = time.monotonic()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if not self.live else time.time()
            i += 1
            if (i - 1) % self.stride:
                continue
            if self.realtime and not self.live:
                lag = (time.monotonic() - t0) - (i / self.fps)
                if lag > 1.0 / self.fps:
                    continue   # behind: drop like a live feed would
            yield frame, t
            n += 1
            if self.max_frames and n >= self.max_frames:
                return

    def stats(self):
        return ""

    def close(self):
        self.cap.release()
