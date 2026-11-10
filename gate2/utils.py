"""Logging, timing and a registry that records which fallback each stage is on."""
from __future__ import annotations

import sys
import threading
import time
from collections import deque

_COL = {"info": "", "warn": "\033[33m", "error": "\033[31m", "ok": "\033[32m"}
_lock = threading.Lock()


def log(level, stage, msg):
    """One line on stderr: [gate2:<stage>] message. Warnings in yellow on a tty."""
    col = _COL.get(level, "") if sys.stderr.isatty() else ""
    end = "\033[0m" if col else ""
    with _lock:
        print(f"{col}[gate2:{stage}] {level.upper()}: {msg}{end}", file=sys.stderr, flush=True)


class StageStatus:
    """What each stage is actually running on, after its fallbacks resolved.

    Shown on the HUD and printed at start-up, so a run that silently fell back
    to the 2D angle or the CPU pose model says so on screen.
    """

    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    def set(self, stage, text, degraded=False):
        with self._lock:
            prev = self._d.get(stage)
            self._d[stage] = (text, degraded)
        if prev != (text, degraded):
            log("warn" if degraded else "info", stage, text)

    def items(self):
        with self._lock:
            return list(self._d.items())


STATUS = StageStatus()


class RateMeter:
    """Rolling rate and latency over the last N events."""

    def __init__(self, n=30):
        self.t = deque(maxlen=n)
        self.lat = deque(maxlen=n)

    def tick(self, latency=None):
        self.t.append(time.monotonic())
        if latency is not None:
            self.lat.append(latency)

    @property
    def hz(self):
        if len(self.t) < 2:
            return 0.0
        return (len(self.t) - 1) / max(self.t[-1] - self.t[0], 1e-6)

    @property
    def latency_ms(self):
        return 1000.0 * sum(self.lat) / len(self.lat) if self.lat else 0.0


class Timer:
    """with Timer() as t: ...; t.ms"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = 1000.0 * (time.perf_counter() - self.t0)
        return False
