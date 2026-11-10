"""Stage 2 -- relative depth relief of each person with Depth-Anything-3.

For every detected person a square crop is taken around the box with a ring
of ground around it (the margin), DA3 is run on that crop alone, a plane is
fitted robustly to the ground ring, and the person's relief is

    relief(u, v) = plane(u, v) - depth(u, v)        (positive = above ground)

Why a zoomed crop and a plane (both measured on these bags, see the DA3 work
in ~/Depth-Anything-3/box_altitude.py):
  * DA3 samples once per 14 px of its input, so a 150 px person in a full
    1920x1080 frame gets a handful of depth samples. Zooming the crop to
    tile_res (1008) resolves torso, thigh and arm separately (~16 cm).
  * DA3's relative output has a smooth trend across any crop that is the same
    size as a prone person; a plane through the ring removes it, a constant
    does not.

The relief is in DA3's own unit, NOT metres: on a nadir view the affine scale
of relative depth is not observable from the ground alone. gate2/angle.py
gives it a metric scale from body proportions; only its *shape* (which joints
are raised, and by how much relative to each other) is used from here.

Fallbacks: CUDA OOM halves tile_res down to depth.min_tile_res, then the next
model in depth.models (mono-large -> base -> small), then CPU; if nothing
loads the stage reports itself disabled and the angle falls back to the
anthropometric / 2D methods.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

from .utils import STATUS, log

# name -> (yaml config in depth_anything_3's registry, HF repo)
DA3_MODELS = {
    "mono-large": ("da3mono-large", "depth-anything/DA3MONO-LARGE"),
    "large": ("da3-large", "depth-anything/DA3-LARGE-1.1"),
    "base": ("da3-base", "depth-anything/DA3-BASE"),
    "small": ("da3-small", "depth-anything/DA3-SMALL"),
    "metric-large": ("da3metric-large", "depth-anything/DA3METRIC-LARGE"),
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Relief:
    """Relief map of one person crop, in full-frame pixel coordinates."""
    ox: int
    oy: int
    side: int
    relief: np.ndarray        # (side, side) float32, DA3 units, + = above ground
    ring_sigma: float         # robust sd of the relief on the ground ring (noise floor)
    ring_clipped: bool        # box touches the crop edge: ground ring is one-sided
    model: str
    ground: np.ndarray | None = None   # (side, side) fitted ground-plane depth, DA3 units

    def sample(self, u, v, radius):
        """Median relief in a disc around full-frame pixel (u, v); nan if outside."""
        x, y = u - self.ox, v - self.oy
        r = max(int(round(radius)), 1)
        x0, x1 = int(math.floor(x - r)), int(math.ceil(x + r)) + 1
        y0, y1 = int(math.floor(y - r)), int(math.ceil(y + r)) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= self.side or y0 >= self.side:
            return float("nan")
        x0c, y0c = max(x0, 0), max(y0, 0)
        x1c, y1c = min(x1, self.side), min(y1, self.side)
        patch = self.relief[y0c:y1c, x0c:x1c]
        yy, xx = np.mgrid[y0c:y1c, x0c:x1c]
        m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        vals = patch[m]
        return float(np.median(vals)) if vals.size else float("nan")

    def profile(self, p0, p1, radius, n=9, t0=0.15, t1=0.85):
        """Relief change from p0 to p1, fitted along the segment's interior.

        Sampling at the two joints themselves is biased: a joint where a
        raised segment meets a flat one picks up the raised one's height
        (DA3 resolves ~16 cm), and a hip under an upright torso is hidden
        from a nadir camera altogether. A Theil-Sen line through samples at
        15-85% of the segment, extrapolated to its ends, avoids both.
        Returns (delta, ok); ok False when the segment is too short in the
        image for interior samples to be distinct.
        """
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 4.0 * radius:
            a, b = self.sample(*p0, radius), self.sample(*p1, radius)
            return b - a, False
        ts = np.linspace(t0, t1, n)
        vs = np.array([self.sample(*(p0 + (p1 - p0) * t), radius) for t in ts])
        ok = np.isfinite(vs)
        if ok.sum() < 3:
            return float("nan"), False
        ts, vs = ts[ok], vs[ok]
        i, j = np.triu_indices(len(ts), 1)
        slope = float(np.median((vs[j] - vs[i]) / (ts[j] - ts[i])))
        return slope, True


def zoom_tile(box, W, H, margin, min_side=32):
    """Square crop around the box with a ground margin, kept inside the frame."""
    x1, y1, x2, y2 = box
    side = int(round(max(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin), min_side)))
    side = min(side, W, H)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ox = int(round(min(max(cx - side / 2.0, 0), W - side)))
    oy = int(round(min(max(cy - side / 2.0, 0), H - side)))
    return ox, oy, side


def ring_mask(side, box_in_tile, pad):
    m = np.ones((side, side), bool)
    x1, y1, x2, y2 = box_in_tile
    m[int(max(0, math.floor(y1 - pad))):int(min(side, math.ceil(y2 + pad))),
      int(max(0, math.floor(x1 - pad))):int(min(side, math.ceil(x2 + pad)))] = False
    return m


def ground_plane(z, ring, iters=2, clip=2.5):
    """Robust plane fit of depth over the ground ring, evaluated over the crop."""
    n = z.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    if ring.sum() < 50:
        return np.full_like(z, float(np.median(z[ring])) if ring.any() else float(np.median(z)))
    m = ring.copy()
    g = None
    for _ in range(iters + 1):
        A = np.column_stack([np.ones(m.sum()), xx[m], yy[m]])
        c, *_ = np.linalg.lstsq(A, z[m].astype(np.float64), rcond=None)
        g = c[0] + c[1] * xx + c[2] * yy
        r = z - g
        med = np.median(r[ring])
        sd = 1.4826 * np.median(np.abs(r[ring] - med))
        if not np.isfinite(sd) or sd <= 0:
            break
        m2 = ring & (np.abs(r - med) < clip * sd)
        if m2.sum() < 50:
            break
        m = m2
    return g.astype(np.float32)


class DepthEstimator:
    """relief(frame, box) -> Relief or None. `available` is False if DA3 never loaded."""

    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg.depth
        self.available = False
        self.model = None
        if not d.enabled:
            STATUS.set("depth", "disabled by config", degraded=True)
            return
        try:
            import torch
            self.torch = torch
        except ImportError as e:
            STATUS.set("depth", f"disabled: torch not importable ({e})", degraded=True)
            return
        want = d.device
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if want == "auto" else want
        if self.device == "cuda" and not torch.cuda.is_available():
            log("warn", "depth", "depth.device=cuda but CUDA is not available; using CPU")
            self.device = "cpu"
        self.res = int(d.tile_res)
        self.chain = list(d.models)
        self._load_next()

    # -- model loading -----------------------------------------------------
    def _weights(self, name):
        leaf = DA3_MODELS[name][1].split("/")[-1]
        for base in self.cfg.paths.da3_weights_dirs:
            p = os.path.join(base, leaf, "model.safetensors")
            if os.path.exists(p):
                return p
        from huggingface_hub import hf_hub_download   # last resort: HF cache / download
        return hf_hub_download(DA3_MODELS[name][1], "model.safetensors")

    def _build(self, name):
        src = os.path.join(self.cfg.paths.da3_root, "src")
        if not os.path.isdir(os.path.join(src, "depth_anything_3")):
            raise FileNotFoundError(f"no depth_anything_3 package under {src}")
        if src not in sys.path:
            sys.path.insert(0, src)
        from safetensors.torch import load_file
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        cfg_name, _ = DA3_MODELS[name]
        ck = self._weights(name)
        net = create_object(load_config(MODEL_REGISTRY[cfg_name]))
        sd = load_file(ck)
        # the published checkpoint wraps the net under .model
        sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
        missing, _ = net.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "gs_head" not in k]
        if missing:
            log("warn", "depth", f"{name}: {len(missing)} missing weights, e.g. {missing[:2]}")
        return net.to(self.device).eval()

    def _load_next(self):
        while self.chain:
            name = self.chain.pop(0)
            try:
                if self.model is not None:
                    del self.model
                    self._empty_cache()
                self.model = self._build(name)
                self.name = name
                self.available = True
                self.acd = (self.torch.bfloat16 if self.device == "cuda"
                            and self.torch.cuda.is_bf16_supported() else self.torch.float16)
                self._report()
                return True
            except Exception as e:  # noqa: BLE001
                log("warn", "depth", f"DA3 {name} failed to load on {self.device}: {e}")
                self.model = None
                self._empty_cache()
        if self.device == "cuda":
            log("warn", "depth", "no DA3 model fits on CUDA; retrying the chain on CPU")
            self.device = "cpu"
            self.chain = list(self.cfg.depth.models)
            return self._load_next()
        self.available = False
        STATUS.set("depth", "disabled: no DA3 model could be loaded", degraded=True)
        return False

    def _report(self):
        degraded = (self.name != self.cfg.depth.models[0] or self.device == "cpu"
                    or self.res < self.cfg.depth.tile_res)
        STATUS.set("depth", f"DA3 {self.name} [{self.device}] @ {self.res}px", degraded=degraded)

    def _empty_cache(self):
        if getattr(self, "torch", None) is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    # -- inference ---------------------------------------------------------
    def _infer(self, tile_bgr):
        torch = self.torch
        import torch.nn.functional as F
        n = tile_bgr.shape[0]
        s = max(int(round(self.res / 14)) * 14, 14)
        small = cv2.resize(tile_bgr, (s, s), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)[None, None]).to(self.device)
        with torch.no_grad(), torch.autocast(self.device, dtype=self.acd,
                                             enabled=(self.device == "cuda")):
            pr = self.model(t)
        z = pr["depth"].float().reshape(1, 1, s, s)
        return F.interpolate(z, (n, n), mode="bilinear", align_corners=True)[0, 0].cpu().numpy()

    def relief(self, frame, box):
        if not self.available:
            return None
        H, W = frame.shape[:2]
        d = self.cfg.depth
        ox, oy, side = zoom_tile(box, W, H, d.margin)
        tile = frame[oy:oy + side, ox:ox + side]
        bt = (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)
        while True:
            try:
                z = self._infer(tile)
                break
            except Exception as e:  # noqa: BLE001
                oom = "out of memory" in str(e).lower()
                self._empty_cache()
                if oom and self.res // 2 >= d.min_tile_res:
                    self.res //= 2
                    log("warn", "depth", f"CUDA OOM: tile_res -> {self.res}")
                    self._report()
                    continue
                log("warn", "depth", f"DA3 {self.name} inference failed: {e}")
                if not self._load_next():
                    return None
        ring = ring_mask(side, bt, d.ring_pad)
        g = ground_plane(z, ring)
        rel = (g - z).astype(np.float32)
        rr = rel[ring]
        sigma = float(1.4826 * np.median(np.abs(rr - np.median(rr)))) if rr.size else float("nan")
        need = 6
        clipped = (bt[0] - d.ring_pad < need or bt[1] - d.ring_pad < need
                   or bt[2] + d.ring_pad > side - need or bt[3] + d.ring_pad > side - need)
        return Relief(ox, oy, side, rel, sigma, clipped, self.name, ground=g)
