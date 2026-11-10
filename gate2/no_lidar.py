"""No-lidar mode (--no-lidar): relative Depth-Anything-3 depth, attitude-normalised.

The parallel path to the default pipeline. Everything upstream and downstream
is shared (FlyPose, the DA3 crop, the Gate-1 transformer, tracking, display,
outputs); what changes is how depth becomes geometry, and what it may use.

What is never used here
-----------------------
The lidar. /scan is not subscribed live, not read from bags, and is removed
from the altitude chain (config/no_lidar.yaml). The altitude that remains
(rel_alt -> local pose -> fixed) is never used for the angle either: only for
the casualty's GPS, to print the stature in metres, and to put the soft
stature prior into camera-height units (see below).

Relative depth
--------------
DA3 runs on the person's zoomed crop exactly as in the default mode, and a
plane is fitted to the ground ring. Instead of treating `plane - depth` as a
height in DA3 units, each pixel's depth is taken as a *ratio to the ground*:

    q(u, v) = depth(u, v) / plane(u, v)          ground = 1, a raised joint < 1

On a keypoint's ray, the point at fraction q of the way from the camera to the
ground plane stands at height (1 - q) x camera height above that plane --
exactly, at any off-nadir angle, for a level ground plane. So the relative
height of a joint is

    h / H_camera = k * (1 - q)

in units of the camera's own height: dimensionless, no metres anywhere. k is
DA3's relative gain: 1 if its crop depth were physically consistent (measured
median 0.021 on 25m_2x_1.5ms -- the zoomed crop exaggerates relief ~50x), solved per
frame from body proportions and pulled toward the person's track median --
the same solve the default mode uses, now on dimensionless relief. Where an
altitude exists (rel_alt / pose, not lidar) it sets only the soft stature
prior in camera-height units (1.70 m / altitude); with none, the stature is
free.

Normalised roll, pitch and yaw
------------------------------
Before any geometry the attitude is normalised: roll and pitch wrapped to
[-180, 180) deg (and an Euler triple with |pitch| > 90 deg mapped to its
equivalent form), heading wrapped to [0, 360). Every ray is then rotated by
that attitude into a level, north-aligned frame, so "up" is gravity's up and
the torso/leg elevations are against the true horizontal whatever the
aircraft was doing. The normalised attitude is logged per frame.

Because the geometry is a similarity of the true one (uniform scale), every
angle -- the hip angle and both vectors' elevations -- is the same as it would
be in metres; only lengths lose their unit.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .angle import WaistAngleEstimator
from .depth_estimator import Relief


def wrap_pi(a):
    """Angle in radians to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def normalize_attitude(roll, pitch, heading):
    """(roll, pitch, heading) in canonical ranges, the same rotation.

    roll, pitch -> [-pi, pi), |pitch| <= pi/2; heading -> [0, 2 pi).
    A ZYX triple with |pitch| > 90 deg describes the same rotation as
    (roll + pi, pi - pitch, heading + pi); that form is used instead.
    """
    roll, pitch, heading = wrap_pi(roll), wrap_pi(pitch), heading
    if abs(pitch) > math.pi / 2:
        roll = wrap_pi(roll + math.pi)
        pitch = wrap_pi(math.pi - pitch)
        heading = heading + math.pi
    return roll, pitch, heading % (2.0 * math.pi)


def normalized_telemetry(tel):
    """Copy of a Telemetry with its attitude normalised (None stays None)."""
    if tel is None:
        return None
    t = copy.copy(tel)
    t.notes = list(tel.notes)
    t.roll, t.pitch, t.heading = normalize_attitude(tel.roll, tel.pitch, tel.heading)
    return t


def relative_relief(relief):
    """Relief as 1 - depth/ground: the joint's height in units of the camera height (x k).

    Returns a Relief whose map is dimensionless, with the ring noise in the
    same units, or None if the depth stage did not keep its ground plane.
    """
    if relief is None or relief.ground is None:
        return None
    g = np.where(np.abs(relief.ground) > 1e-9, relief.ground, np.nan)
    rel = (relief.relief / g).astype(np.float32)          # (g - z) / g = 1 - q
    rel = np.where(np.isfinite(rel), rel, 0.0).astype(np.float32)
    gmed = float(np.nanmedian(np.abs(g)))
    sigma = relief.ring_sigma / gmed if gmed > 0 else relief.ring_sigma
    return Relief(relief.ox, relief.oy, relief.side, rel, sigma, relief.ring_clipped,
                  relief.model + " (relative)", ground=relief.ground)


class RelativeWaistAngleEstimator(WaistAngleEstimator):
    """The default estimator, run on relative depth in the attitude-normalised frame."""

    def _metric(self, tel):
        # geometry in units of the camera height, never metres -- whether or not
        # an altitude exists
        return False, 1.0

    def _stature_prior(self, tel, metric):
        # The stature prior in units of the camera height: 1.70 m / altitude.
        # The altitude here is rel_alt / local pose (barometric/EKF), never the
        # lidar, and it only sets this soft prior -- the geometry stays
        # relative. Without it the stature is free, the solve can absorb limbs
        # FlyPose draws long into a larger person, and the DA3 gain collapses
        # to 0 (measured: 0 on most frames of 25m_2x_1.5ms without the prior).
        if tel is None or not tel.altitude:
            return None
        c, H = self.cfg.angle, float(tel.altitude)
        return (c.stature_prior_m / H, c.stature_sigma_m / H,
                tuple(b / H for b in c.stature_bounds_m))

    def estimate(self, det, relief, geo, tel, scale_prior=None):
        tel_n = normalized_telemetry(tel)
        rel_n = relative_relief(relief)
        res = super().estimate(det, rel_n, geo, tel_n, scale_prior)
        res.geometry = "relative"
        res.notes = [n for n in res.notes if not n.startswith("no altitude")]
        res.notes.append("no lidar: relative DA3 depth (x camera height), attitude-normalised")
        if relief is not None and rel_n is None:
            res.notes.append("depth stage kept no ground plane: relative depth unavailable")
        res.relative_gain = res.scale     # k: 1 if DA3's crop depth were physically consistent
        if tel_n is not None:
            res.attitude_deg = tuple(round(math.degrees(v), 2)
                                     for v in (tel_n.roll, tel_n.pitch, tel_n.heading))
        return res
