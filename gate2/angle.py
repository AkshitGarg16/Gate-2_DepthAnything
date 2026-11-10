"""Stage 4 -- the waist (hip) angle, fusing pose, depth and geolocalisation.

The angle reported is the hip angle: the angle at the hip between the torso
(hip -> shoulder) and the leg (hip -> ankle; hip -> knee when the ankle is not
detected), in 3D. For each person the 3D direction of both vectors is also
reported: elevation above the ground, compass bearing, and whether each is
parallel, perpendicular or oblique to the ground.

    180 deg  body straight (lying flat or standing)
     90 deg  torso and thigh at a right angle (sitting upright)
    < 90     jack-knifed / curled
    bend = 180 - angle is shown alongside it.

Why 3D matters: from above, a person sitting up has their torso pointing at
the camera, so its image is foreshortened to almost nothing and the 2D angle
in the image is meaningless. The depth model says which joints are raised.

How each 3D keypoint is built
-----------------------------
  pixel (u, v)           FlyPose keypoint
  relief r (DA3 units)   DA3 relief at that keypoint (gate2/depth_estimator.py)
  height  h = a * r      a = metres per DA3 unit, solved below
  3D point               Gate-1 ray through (u, v), stopped at height h
                         (gate2/geolocator.py Geolocator.lift)

Solving the scale `a` (and the person's stature S)
--------------------------------------------------
DA3's relative depth has no metric scale at nadir, so it is taken from the
body itself: every limb segment has a known length as a fraction rho of the
standing height S (keypoints.SEGMENTS). For each visible segment i, its
ground-plane length d_i (metres, from the geolocator) and relief difference
dr_i give a 3D length sqrt(d_i^2 + (a dr_i)^2) that must equal S rho_i:

    minimise  sum_i w_i [ (sqrt(d_i^2 + a^2 dr_i^2) - S rho_i) / (S0 rho_i) ]^2
              + [ 0.1 (S - S0) / sigma_S ]^2         (soft stature prior)

Flat segments pin S, raised segments then fix a, and the sign of each rise
comes from DA3. S being free makes the angle insensitive to a wrong ground
sample distance (wrong altitude or focal length scales every d_i and S
together and cancels). With no altitude at all the same solve runs in
unscaled units with no stature prior ("unscaled" in the HUD).

Every method measures 3D vectors
--------------------------------
The angle is always arccos of the dot product of two 3D vectors, torso
(mid-hip -> mid-shoulder) and leg (hip -> ankle, or knee), built from 3D keypoints.
When the keypoints are at equal depth the vertical components are zero and
the result is exactly the angle in that plane -- in metres, so it is also
correct under camera tilt and perspective, which the raw pixel angle is not.

Fallback chain (angle.methods)
------------------------------
  3d_depth        as above. Rises DA3 does not see above its noise floor stay
                  at their measured (near-zero) value -- equal depth is never
                  overridden by body proportions. "flat" in the HUD when no
                  segment rises significantly.
  3d_equal_depth  no usable depth: every keypoint at equal height above the
                  ground (Gate-1 ray to the ground plane), or at one common
                  camera depth if there is no telemetry. Correct for a person
                  lying flat; underestimates the bend of one sitting up.
  3d_anthro       opt-in: rise guessed from foreshortening with the ASSUMPTION
                  that torso and thigh rise away from the hip (true for someone
                  on the ground, wrong for someone standing).
The raw image-plane angle is logged as angle_2d_deg for reference only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import keypoints as K

METHOD_WEIGHT = {"3d_depth": 1.0, "3d_depth_flat": 0.9, "3d_equal_depth": 0.5, "3d_anthro": 0.4}


def backproject_equal_depth(uv, K):
    """Camera-frame 3D points of pixels that all lie at the same depth Z = 1.

    X = (u - cx) / fx,  Y = (v - cy) / fy,  Z = 1. The angle between vectors
    built from these is the true angle of a figure lying in a plane parallel
    to the image -- unlike the raw pixel angle, it is correct when fx != fy,
    and it is a proper 3D vector the same code path measures.
    """
    fx, fy, cx, cy = K
    uv = np.asarray(uv, float)
    return np.column_stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))])


@dataclass
class AngleResult:
    method: str = "none"
    angle: float = float("nan")          # hip angle, degrees (180 = straight)
    sides: dict = field(default_factory=dict)   # 'L'/'R'/'mid' -> degrees
    angle_2d: float = float("nan")
    confidence: float = 0.0
    metric: bool = False
    stature: float | None = None         # in the geometry's units (metres if metric)
    stature_m: float | None = None       # metres, when known
    geometry: str = "metric"             # metric | unscaled | relative
    scale: float | None = None           # metres per DA3 unit
    heights: np.ndarray | None = None    # (17,) metres (or unscaled units) above ground
    points: np.ndarray | None = None     # (17,3) local NED
    chains: list = field(default_factory=list)
    gps: tuple | None = None
    vectors: dict = field(default_factory=dict)       # name -> (torso vec, leg vec), 3D
    orientation: dict = field(default_factory=dict)   # elevation/azimuth/relation per vector
    relief_snr: float | None = None
    fit_rms: float | None = None
    notes: list = field(default_factory=list)

    @property
    def bend(self):
        return 180.0 - self.angle if math.isfinite(self.angle) else float("nan")


def _angle(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if not (n1 > 0 and n2 > 0):
        return float("nan")
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(c))


def build_chains(valid, use_knee=True):
    """Joint groups the angle is measured on, one chain per usable leg.

    The leg is hip -> ankle; hip -> knee is the fallback when the ankle is
    not detected (use_knee). Note the two differ when the knee is bent.

    The torso is the trunk midline (mid-hip -> mid-shoulder) when both
    shoulders and both hips are visible -- the anatomical definition of hip
    flexion. A per-side shoulder->hip line is not usable as "the torso" for a
    straight body: the shoulders are wider than the hips, so it meets the
    thigh at ~173 deg, not 180. With a shoulder or hip missing, the visible
    side's own shoulder/hip are used instead.

    Each chain: (name, [sho], [torso_hip], [vertex_hip], [leg], leg_kind);
    angle = angle between mean(sho)-mean(torso_hip) and leg-vertex_hip.
    """
    both_sho = valid[K.L_SHO] and valid[K.R_SHO]
    both_hip = valid[K.L_HIP] and valid[K.R_HIP]
    chains = []
    for side, (s, h, k, a) in K.SIDES.items():
        if not valid[h]:
            continue
        if valid[a]:
            leg, kind = [a], "ankle"
        elif use_knee and valid[k]:
            leg, kind = [k], "knee"
        else:
            continue
        if both_sho and both_hip:
            chains.append((side, [K.L_SHO, K.R_SHO], [K.L_HIP, K.R_HIP], [h], leg, kind))
        elif valid[s]:
            chains.append((side, [s], [h], [h], leg, kind))
        else:
            sho = [i for i in (K.L_SHO, K.R_SHO) if valid[i]]
            if sho:
                chains.append((side, sho, [h], [h], leg, kind))
    return chains


def chain_angles(P, chains):
    out = {}
    for name, sho, thip, vhip, leg, _ in chains:
        torso = P[sho].mean(0) - P[thip].mean(0)
        thigh = P[leg].mean(0) - P[vhip].mean(0)
        out[name] = _angle(torso, thigh)
    return out


def chain_vectors(P, chains):
    """{chain name: (torso vector hip->shoulder, leg vector hip->ankle/knee)} in 3D."""
    return {name: (P[sho].mean(0) - P[thip].mean(0), P[leg].mean(0) - P[vhip].mean(0))
            for name, sho, thip, vhip, leg, _ in chains}


def vector_orientation(v, par_max, perp_min, world=True):
    """Direction of a 3D vector (local NED or camera frame, up = -z) against the ground.

    elevation_deg   angle above (+) or below (-) the ground plane
    from_vertical   angle from the vertical (90 - |elevation|)
    azimuth_deg     compass bearing of its horizontal part (world frame only)
    relation        parallel / perpendicular / oblique to the ground
    """
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if not n > 0:
        return None
    elev = math.degrees(math.asin(float(np.clip(-v[2] / n, -1.0, 1.0))))
    rel = ("parallel" if abs(elev) <= par_max else
           "perpendicular" if abs(elev) >= perp_min else "oblique")
    az = math.degrees(math.atan2(v[1], v[0])) % 360.0 if world and np.hypot(v[0], v[1]) > 1e-9 else None
    return {"elevation_deg": round(elev, 1), "from_vertical_deg": round(90.0 - abs(elev), 1),
            "azimuth_deg": None if az is None else round(az, 1), "relation": rel,
            "direction": "up" if elev > 0 else "down" if elev < 0 else "level"}


def _joints(ch):
    return sorted({i for g in ch[1:5] for i in g})


def _combine(angles, chains, scores):
    """Confidence-weighted mean of the per-chain angles."""
    num = den = 0.0
    for ch in chains:
        a = angles.get(ch[0], float("nan"))
        if not math.isfinite(a):
            continue
        w = float(np.mean(scores[_joints(ch)]))
        num, den = num + w * a, den + w
    return num / den if den > 0 else float("nan")


class WaistAngleEstimator:
    def __init__(self, cfg):
        self.cfg = cfg

    # -----------------------------------------------------------------------
    def estimate(self, det, relief, geo, tel, scale_prior=None):
        """scale_prior: the person's track-median DA3 scale (m/unit), or None."""
        c = self.cfg.angle
        uv = np.asarray(det["keypoints"], float)
        sc = np.asarray(det["scores"], float)
        valid = sc >= self.cfg.pose.kp_thr
        res = AngleResult()
        # the metric camera lever arm in the geometry's units (see Geolocator.camera_ned)
        metric0 = tel is not None and self._metric(tel)[0]
        self._off = 1.0 if metric0 else (1.0 / tel.altitude if (tel is not None and tel.altitude)
                                         else 0.0)
        chains = build_chains(valid, c.use_knee_if_no_ankle)
        if not chains:
            res.notes.append("shoulder/hip/ankle(knee) not visible")
            return res
        # drop chains whose image segments are degenerate (overlapping joints)
        keep = []
        for ch in chains:
            _, s, th, vh, l, _ = ch
            if np.linalg.norm(uv[l].mean(0) - uv[vh].mean(0)) >= c.min_segment_px:
                keep.append(ch)
        chains = keep
        if not chains:
            res.notes.append("leg too short in the image to measure")
            return res
        res.chains = chains
        if any(ch[5] == "knee" for ch in chains):
            res.notes.append("ankle not visible: leg direction taken hip->knee")

        # raw pixel angle, kept only as a reference in the log
        P2 = np.column_stack([uv, np.zeros(len(uv))])
        res.angle_2d = _combine(chain_angles(P2, chains), chains, sc)

        geo_ok = geo is not None and geo.tf is not None and tel is not None
        for method in c.methods:
            if method == "3d_depth" and geo_ok and relief is not None:
                if self._solve_depth(res, uv, sc, valid, relief, geo, tel, chains, scale_prior):
                    break
            elif method == "3d_anthro" and geo_ok:
                if self._solve_anthro(res, uv, sc, valid, geo, tel, chains):
                    break
            elif method in ("3d_equal_depth", "2d"):
                self._solve_equal_depth(res, uv, sc, geo, tel, chains, geo_ok)
                break
        if res.method == "none":
            return res

        if geo_ok and tel.altitude is not None:
            hips = [i for i in (K.L_HIP, K.R_HIP) if valid[i]]
            if hips:
                u, v = uv[hips].mean(0)
                try:
                    res.gps = geo.gps(u, v, tel)
                except Exception:  # noqa: BLE001 - GPS is informational only
                    res.gps = None

        used = sorted({i for ch in chains for i in _joints(ch)})
        conf = float(np.mean(sc[used])) * METHOD_WEIGHT.get(res.method, 0.3)
        if res.fit_rms is not None:
            conf *= math.exp(-res.fit_rms / 0.25)
        res.confidence = conf
        if res.stature is not None:
            if res.metric:
                res.stature_m = res.stature
            elif tel is not None and tel.altitude:
                res.stature_m = res.stature * tel.altitude   # units of camera height -> m
        res.geometry = "metric" if res.metric else "unscaled"
        self._orientation(res, geo_ok)
        return res

    # -----------------------------------------------------------------------
    def _orientation(self, res, world):
        """How the hip->shoulder and hip->ankle vectors sit in 3D space.

        Torso: the mean torso vector over the chains (one vector when the
        trunk midline is used). Legs: one per side. 'measured' is False for
        the equal-depth method, where every elevation is 0 by assumption.
        """
        c = self.cfg.angle
        par, perp = c.get("parallel_max_deg", 20.0), c.get("perpendicular_min_deg", 70.0)
        if not res.vectors:
            return
        torso = np.mean([vt / max(np.linalg.norm(vt), 1e-12) for vt, _ in res.vectors.values()], 0)
        out = {"torso": vector_orientation(torso, par, perp, world)}
        for name, (_, vl) in res.vectors.items():
            out[f"leg_{name}"] = vector_orientation(vl, par, perp, world)
        legs = [o for k, o in out.items() if k.startswith("leg_") and o]
        out["measured"] = res.method in ("3d_depth", "3d_depth_flat", "3d_anthro")
        out["aligned"] = (math.isfinite(res.angle) and res.angle >= c.get("aligned_min_deg", 165.0))
        if legs and out["torso"]:
            out["summary"] = (f"torso {out['torso']['elevation_deg']:+.0f} deg ({out['torso']['relation']}), "
                              + ", ".join(f"leg {k[4:]} {o['elevation_deg']:+.0f} deg ({o['relation']})"
                                          for k, o in out.items() if k.startswith("leg_") and o))
        res.orientation = out

    def _solve_equal_depth(self, res, uv, sc, geo, tel, chains, geo_ok):
        """All keypoints at equal height above the ground, as 3D vectors.

        With telemetry, each keypoint is the Gate-1 ray stopped at the ground
        plane (attitude- and perspective-correct; a tilted camera sees a flat
        body at unequal camera depths, and this accounts for it). Without
        telemetry, all keypoints are back-projected at one common camera depth
        through the intrinsics (or a nominal K if none are known).
        """
        if geo_ok:
            metric, alt = self._metric(tel)
            P = geo.lift(uv, np.zeros(17), tel, alt, offset_scale=self._off)
            how = "ground plane" + ("" if metric else " (unscaled)")
        else:
            if geo is not None and geo.K is not None:
                K_ = geo.K
            else:
                w = max(float(np.max(uv[:, 0])), 1.0) * 2
                K_ = (w, w, w / 2, w / 2)       # nominal: only the ratio fx/fy = 1 matters
            P = backproject_equal_depth(uv, K_)
            metric, how = False, "common camera depth"
        sides = chain_angles(P, chains)
        res.vectors = chain_vectors(P, chains)
        res.method, res.sides = "3d_equal_depth", sides
        res.angle = _combine(sides, chains, sc)
        res.metric, res.points = metric, P
        res.heights = np.zeros(17)
        res.notes.append(f"no depth: keypoints assumed at equal depth ({how})")
        return True

    def _metric(self, tel):
        if tel.altitude is not None:
            return True, tel.altitude
        return False, 1.0   # unscaled: everything in units of the camera height

    def _stature_prior(self, tel, metric):
        """(prior, sigma, (lo, hi)) for the stature in the solve's units, or None (free)."""
        c = self.cfg.angle
        if metric:
            return c.stature_prior_m, c.stature_sigma_m, tuple(c.stature_bounds_m)
        return None

    def _segments(self, valid, extra_ok=None):
        out = []
        for a, b, rho in K.SEGMENTS:
            if valid[a] and valid[b] and (extra_ok is None or (extra_ok[a] and extra_ok[b])):
                out.append((a, b, rho))
        return out

    def _solve_depth(self, res, uv, sc, valid, relief, geo, tel, chains, scale_prior=None):
        from scipy.optimize import least_squares
        c = self.cfg.angle
        metric, alt = self._metric(tel)
        long_side = max(np.ptp(uv[valid, 0]), np.ptp(uv[valid, 1]), 8.0)
        rad = max(self.cfg.depth.sample_radius_frac * long_side, 1.5)
        r_pt = np.array([relief.sample(u, v, rad) if valid[i] else np.nan
                         for i, (u, v) in enumerate(uv)])
        if not np.all(np.isfinite(r_pt[[i for ch in chains for i in _joints(ch)]])):
            res.notes.append("depth: a chain joint lies outside the depth crop")
            return False

        # ground-plane positions (h = 0) for the horizontal lengths
        P0 = geo.lift(uv, np.zeros(17), tel, alt, offset_scale=self._off)
        if not np.all(np.isfinite(P0[valid])):
            res.notes.append("depth: keypoint ray misses the ground")
            return False

        def seg(ja, jb):
            """(d horizontal, dr relief rise a->b, interior_ok) for joint groups."""
            pa, pb = P0[ja].mean(0), P0[jb].mean(0)
            dr, ok = relief.profile(uv[ja].mean(0), uv[jb].mean(0), rad)
            return float(np.linalg.norm(pb[:2] - pa[:2])), dr, ok

        segs = []   # (d, dr, rho, weight)
        for ja, jb, rho in K.SEGMENTS:
            if valid[ja] and valid[jb]:
                d, dr, ok = seg([ja], [jb])
                if np.isfinite(dr):
                    segs.append((d, dr, rho, min(sc[ja], sc[jb]), ok))
        if len(segs) < 3:
            res.notes.append("depth: too few segments with relief")
            return False
        d = np.array([x[0] for x in segs]); dr = np.array([x[1] for x in segs])
        rho = np.array([x[2] for x in segs])
        prior = self._stature_prior(tel, metric)
        if prior is not None:
            S0, sig, (lo, hi) = prior
            Sref = S0
        else:
            Sref = float(np.median(d / rho)) * 1.1
            lo, hi = Sref * 0.2, Sref * 5.0
        # Foreshortened segments carry occlusion-biased relief; let the long
        # (in-image) ones set the scale.
        fore = np.clip(d / (Sref * rho), 0.1, 1.0)
        w = np.sqrt(np.array([x[3] for x in segs])) * fore * \
            np.array([1.0 if x[4] else 0.3 for x in segs])
        rmax = float(np.nanmax(np.abs(r_pt[valid])))
        a_hi = hi / rmax if rmax > 0 else 1e6       # no joint higher than one stature

        # The DA3 scale of one person's crop is stable within a track (+-7% on
        # these bags), while one frame's limb lengths are noisy (FlyPose's
        # aerial keypoints imply segment lengths that vary +-40% within a
        # person). So a frame's scale is pulled toward its track's median.
        sp = scale_prior if (scale_prior is not None and scale_prior > 0) else None
        w_sp = float(self.cfg.angle.get("scale_prior_weight", 1.0))

        def fun(x):
            aa, SS = x
            out = w * (np.sqrt(d ** 2 + (aa * dr) ** 2) - SS * rho) / (Sref * rho)
            if prior is not None:
                out = np.append(out, 0.1 * (SS - S0) / sig)
            if sp is not None:
                out = np.append(out, w_sp * (aa - sp) / (0.15 * sp))
            return out

        need = np.sqrt(np.maximum((Sref * rho) ** 2 - d ** 2, 0.0))
        big = np.abs(dr) > 1e-9
        a0 = float(np.median(need[big] / np.abs(dr[big]))) if big.any() else 0.0
        best = None
        for start in (0.0, a0, 3.0 * a0) + ((sp,) if sp is not None else ()):
            try:
                sol = least_squares(fun, [float(np.clip(start, 0, a_hi)), Sref],
                                    bounds=([0.0, lo], [a_hi, hi]), loss="soft_l1", f_scale=0.15)
            except ValueError:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        if best is None:
            res.notes.append("depth: scale solve failed")
            return False
        a, S = (float(v) for v in best.x)
        rms = float(np.sqrt(np.mean((fun(best.x)[: len(segs)] / np.maximum(w, 1e-6)) ** 2)))
        noise = a * relief.ring_sigma if np.isfinite(relief.ring_sigma) else 0.0
        k_sig = self.cfg.depth.min_relief_snr

        fore_min = float(self.cfg.angle.get("foreshortened_frac", 0.55))
        posture = self.cfg.angle.get("posture", "ground")
        # metric constants in the solve's units: metres, or -- when the geometry
        # is unscaled/relative -- converted through the solved stature
        to_units = 1.0 if metric else S / float(c.stature_prior_m)
        hip_cap = float(self.cfg.angle.get("hip_height_ground_m", 0.12)) * to_units
        knee_ratio = tuple(self.cfg.angle.get("knee_ratio", [0.35, 0.55]))
        occl_notes = set()

        def rise(ja, jb, rho_seg, occluded=False, torso=False):
            """Vertical rise a->b in metres, from the depth of the two ends.

            Equal depth stays equal: if DA3 shows no rise above its noise floor
            the measured (near-zero) rise is used as it is, and the 3D angle
            reduces exactly to the angle in the ground plane. Body proportions
            never invent a rise the depth did not see.

            When DA3 does see a significant rise, its sign is trusted but its
            size is biased low on foreshortened segments (the lower end is
            hidden under the upper one, and DA3 resolves ~16 cm), so the size
            is blended with the length-model rise sqrt((S rho)^2 - d^2) with
            weight (d / S rho)^2 on depth: a segment lying in the image plane
            is measured by DA3, one pointing at the camera by its
            foreshortening.

            occluded: the segment lies under another in the image (thigh under
            a folded torso), so its depth is the cover's surface; its rise is
            taken from the length model alone.
            """
            dd, ddr, ok = seg(ja, jb)
            z_depth = a * ddr if ok else a * (np.mean(r_pt[jb]) - np.mean(r_pt[ja]))
            z_len = math.sqrt(max((S * rho_seg) ** 2 - dd ** 2, 0.0))
            if occluded:
                return z_len, dd
            if dd < fore_min * S * rho_seg and a > 0:
                # Strongly foreshortened: the segment points at the camera and
                # its lower end is hidden under its upper end, so DA3 sees one
                # covering surface and its reading -- level or not -- is not
                # this segment's rise. Keypoints truly at equal depth are never
                # this short, so this never overrides an equal-depth reading.
                if torso:
                    # for a person on the ground the torso can only point up
                    occl_notes.add("torso points at the camera: rise from its length")
                    return z_len, dd
                # Thigh pointing at the camera. From straight above, knees drawn
                # up (sitting) and a thigh hanging down (kneeling/standing) look
                # alike; angle.posture says which world we are in.
                if posture == "upright":
                    occl_notes.add("leg points at the camera: hanging down (posture=upright)")
                    return -z_len, dd
                # posture "ground" (a casualty): how high the knee's surface
                # stands relative to the top of the body is scale-free and
                # separates knees drawn up (>= knee_up_ratio) from a leg lying
                # flat whose knee keypoint merely sits close to the hip
                # (<= knee_flat_ratio; the measured depth then stands).
                top_i = [i for i in (K.L_SHO, K.R_SHO, 0) if valid[i] and np.isfinite(r_pt[i])]
                top = max(float(np.max(r_pt[top_i])) if top_i else 0.0, 1e-9)
                ratio = float(np.mean(r_pt[jb])) / top
                lo_r, hi_r = knee_ratio
                wu = float(np.clip((ratio - lo_r) / max(hi_r - lo_r, 1e-6), 0.0, 1.0))
                occl_notes.add("leg points at the camera: rise from its length")
                return wu * z_len + (1.0 - wu) * z_depth, dd
            if not abs(z_depth) > k_sig * noise:
                return z_depth, dd                  # equal depth (within noise)
            wd = min(dd / (S * rho_seg), 1.0) ** 2 if ok else 0.0
            size = wd * abs(z_depth) + (1.0 - wd) * max(z_len, abs(z_depth))
            return math.copysign(size, z_depth), dd

        sides, snr = {}, 0.0
        heights = np.full(17, np.nan)
        ov = self.cfg.angle.get("overlap_deg", 30.0)
        for name, sho, thip, vhip, leg, kind in chains:
            # torso folded over the thigh in the image: the thigh's relief is
            # the torso's top surface, so its rise comes from its length alone
            a2 = _angle(uv[sho].mean(0) - uv[thip].mean(0), uv[leg].mean(0) - uv[vhip].mean(0))
            occl = math.isfinite(a2) and a2 < ov
            if occl and "leg under torso" not in " ".join(res.notes):
                res.notes.append("leg under torso in the image: leg rise from its length")
            zt, _ = rise(thip, sho, 0.288, torso=True)
            zl, _ = rise(vhip, leg, 0.245 if kind == "knee" else 0.491, occluded=occl)
            hh = max(a * float(np.mean(r_pt[vhip])), 0.0)
            ht = max(a * float(np.mean(r_pt[thip])), 0.0)
            if posture == "ground":
                # DA3 sees the surface over the hip (belly, buttocks, or the
                # thighs/torso covering it), not the joint. For someone on the
                # ground the pelvis rests on it, so the hip joint is capped at
                # hip_height_ground_m; every other joint keeps its height
                # *relative* to the hip, so equal depth stays equal.
                hh, ht = min(hh, hip_cap), min(ht, hip_cap)
            heights[vhip] = hh
            heights[thip] = ht
            heights[sho] = ht + zt
            heights[leg] = hh + zl
            if relief.ring_sigma > 0:
                for ja, jb in ((thip, sho), (vhip, leg)):
                    snr = max(snr, abs(seg(ja, jb)[1]) / relief.ring_sigma)
        # No joint is below the ground plane. DA3 samples the body's *surface*,
        # so the hip reads the belly/buttocks, not the joint centre, and a leg
        # dropping from there to the ankle would otherwise reach under the
        # ground (measured: -54 deg legs on a person lying on the grass).
        if np.nanmin(heights) < 0:
            occl_notes.add("joint below ground clamped to ground")
        heights = np.where(np.isfinite(heights), np.maximum(heights, 0.0), np.nan)
        # re-lift every joint at its height: a raised joint seen off-nadir sits
        # radially inward of where its ray meets the ground
        hz = np.where(np.isfinite(heights), heights, 0.0)
        P1 = geo.lift(uv, hz, tel, alt, offset_scale=self._off)
        res.vectors = chain_vectors(P1, chains)
        for name, (vt, vl) in res.vectors.items():
            sides[name] = _angle(vt, vl)
        ang = _combine(sides, chains, sc)
        if not math.isfinite(ang):
            return False
        flat = all(abs(x[1]) * a <= k_sig * noise for x in segs)
        res.method = "3d_depth_flat" if flat else "3d_depth"
        res.angle, res.sides = ang, sides
        res.metric, res.stature, res.scale = metric, S, a
        res.relief_snr = snr if relief.ring_sigma > 0 else None
        res.fit_rms = rms
        res.heights = heights
        res.points = P1
        if not metric:
            res.notes.append("no altitude: unscaled geometry (angle still valid)")
        if relief.ring_clipped:
            res.notes.append("person at frame edge: one-sided ground ring")
        res.notes.extend(sorted(occl_notes))
        return True

    def _solve_anthro(self, res, uv, sc, valid, geo, tel, chains):
        c = self.cfg.angle
        metric, alt = self._metric(tel)
        P0 = geo.lift(uv, np.zeros(17), tel, alt, offset_scale=self._off)
        segs = self._segments(valid)
        if not segs:
            return False
        d_over_rho = [np.linalg.norm(P0[a, :2] - P0[b, :2]) / rho for a, b, rho in segs]
        S = max(max(d_over_rho), c.stature_prior_m if metric else 0.0)
        sides = {}
        for name, sho, thip, vhip, leg, kind in chains:
            ph, ps, pl = P0[vhip].mean(0), P0[sho].mean(0), P0[leg].mean(0)
            pth = P0[thip].mean(0)
            rho_t, rho_l = 0.288, (0.245 if kind == "knee" else 0.491)
            dt = np.linalg.norm(ps[:2] - pth[:2])
            dl = np.linalg.norm(pl[:2] - ph[:2])
            zt = math.sqrt(max((S * rho_t) ** 2 - dt ** 2, 0.0))
            zl = math.sqrt(max((S * rho_l) ** 2 - dl ** 2, 0.0))
            vt = np.array([ps[0] - pth[0], ps[1] - pth[1], -zt])   # NED: up is -z
            vl = np.array([pl[0] - ph[0], pl[1] - ph[1], -zl])
            sides[name] = _angle(vt, vl)
            res.vectors[name] = (vt, vl)
        ang = _combine(sides, chains, sc)
        if not math.isfinite(ang):
            return False
        res.method, res.angle, res.sides = "3d_anthro", ang, sides
        res.metric, res.stature, res.points = metric, S, P0
        res.notes.append("no depth: assumed torso and thigh rise away from the hip")
        return True
