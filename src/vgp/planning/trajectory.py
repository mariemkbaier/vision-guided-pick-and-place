"""Time-parameterised trajectories.

A PATH SAYS WHERE. A TRAJECTORY SAYS WHEN.
------------------------------------------
A list of waypoints is not a plan. The moment you hand a controller bare
waypoints, something downstream has to decide how fast to traverse them -- and
whatever it decides is invisible, untunable, and usually "as fast as the gain
allows", which is how you get an arm that snaps to position, overshoots, and
knocks the object over before the gripper closes.

Time-parameterising here makes velocity and acceleration EXPLICIT and bounded,
and puts the decision somewhere it can be reasoned about and tested.

TRAPEZOIDAL VS QUINTIC -- WHY BOTH
----------------------------------
`trapezoidal`: accelerate at a_max, cruise at v_max, decelerate. It is
TIME-OPTIMAL subject to those bounds, which matters for cycle time. Its cost is
discontinuous acceleration at the corners -- infinite jerk -- which excites
structural modes and, on a real arm, makes an audible clunk.

`quintic`: a fifth-order polynomial matching position, velocity AND
acceleration at both ends. Smooth, zero jerk at the endpoints, and noticeably
slower for the same limits (its peak velocity is 1.875x the average, so it must
run longer to respect v_max).

We use trapezoidal for long free motions where time matters, and quintic for
short approach and retreat moves where smoothness near the object matters more
than a tenth of a second. That is the actual engineering trade, not a
preference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vgp.msgs import Header, JointTrajectory, JointTrajectoryPoint

# Per-joint limits. Conservative: this is a 0.6 m tabletop arm, not a delta
# robot, and the objects are unfixtured -- a fast move that shakes the base
# moves the cubes before we ever touch them.
DEFAULT_VMAX = np.array([2.0, 1.6, 1.8, 2.5, 2.5, 3.0])     # rad/s
DEFAULT_AMAX = np.array([4.0, 3.2, 3.6, 6.0, 6.0, 8.0])     # rad/s^2


@dataclass
class Profile:
    duration: float
    sample: callable          # t -> (q, dq)


def trapezoidal(q0: np.ndarray, q1: np.ndarray,
                vmax: np.ndarray, amax: np.ndarray) -> Profile:
    """Synchronised trapezoidal profile: all joints start and stop together.

    SYNCHRONISATION IS THE POINT. Give each joint its own optimal timing and
    they finish at different moments, so the end-effector traces an arbitrary
    curve through space -- fine in free air, not fine near an object. We
    compute each joint's minimum time, take the LONGEST, and slow every other
    joint to match. The result is a straight line in JOINT space, which is
    predictable even though it is not straight in Cartesian space.
    """
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    d = q1 - q0
    dist = np.abs(d)

    # Per-joint minimum time. Triangular profile if the move is too short to
    # ever reach vmax -- the classic case people forget, which then produces a
    # negative cruise time and a NaN.
    t_acc = vmax / amax
    d_acc = 0.5 * vmax * t_acc                    # distance in one ramp
    t_min = np.where(
        dist >= 2 * d_acc,
        2 * t_acc + (dist - 2 * d_acc) / vmax,    # trapezoid
        2 * np.sqrt(np.maximum(dist, 0) / amax),  # triangle: never reaches vmax
    )
    T = float(max(t_min.max(), 1e-3))

    def sample(t: float):
        t = float(np.clip(t, 0.0, T))
        # Normalised S-curve in [0,1], shared by all joints so they stay
        # synchronised; each joint just scales it by its own displacement.
        s, ds = _trap_unit(t, T)
        return q0 + d * s, d * ds

    return Profile(T, sample)


def _trap_unit(t: float, T: float) -> tuple[float, float]:
    """Unit trapezoid on [0,T] with a fixed 25% accel / 50% cruise / 25% decel.

    Fixing the ramp FRACTION rather than the ramp duration keeps the profile
    similar at every move length, so behaviour does not change character
    between a long and a short move.
    """
    ta = 0.25 * T
    # Peak velocity that makes the area (total displacement) exactly 1.
    vp = 1.0 / (T - ta)
    if t < ta:
        return 0.5 * vp * t * t / ta, vp * t / ta
    if t < T - ta:
        return 0.5 * vp * ta + vp * (t - ta), vp
    td = T - t
    return 1.0 - 0.5 * vp * td * td / ta, vp * td / ta


def quintic(q0: np.ndarray, q1: np.ndarray,
            vmax: np.ndarray, amax: np.ndarray) -> Profile:
    """Fifth-order polynomial with zero velocity AND acceleration at both ends.

    Duration is chosen so the polynomial's PEAK velocity and acceleration stay
    within limits. For the standard quintic those peaks are

        v_peak = 1.875 * d / T        a_peak = 5.7735 * d / T^2

    Solving each for T and taking the larger is why a quintic is slower than a
    trapezoid for the same limits -- it spends time ramping rather than
    cruising. That is the price of smoothness, and it is worth stating
    numerically rather than hand-waving.
    """
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    d = q1 - q0
    dist = np.abs(d)
    T = float(max(
        (1.875 * dist / vmax).max(),
        np.sqrt((5.7735 * dist / amax).max()) if dist.max() > 0 else 0.0,
        1e-3,
    ))

    def sample(t: float):
        u = float(np.clip(t, 0.0, T)) / T
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        ds = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
        return q0 + d * s, d * ds

    return Profile(T, sample)


def to_message(profile: Profile, joint_names, stamp: float,
               rate_hz: float = 100.0) -> JointTrajectory:
    """Discretise a profile into a JointTrajectory message.

    Sampled at the CONTROLLER's rate, not some arbitrary waypoint count, so
    the consumer never has to re-interpolate and introduce its own timing.
    """
    n = max(2, int(np.ceil(profile.duration * rate_hz)) + 1)
    pts = []
    for i in range(n):
        t = min(i / rate_hz, profile.duration)
        q, dq = profile.sample(t)
        pts.append(JointTrajectoryPoint(q, t, dq))
    return JointTrajectory(Header(stamp, "base_link"), tuple(joint_names), pts)


def plan_joint_move(q0, q1, joint_names, stamp: float, smooth: bool = False,
                    vmax=None, amax=None, speed_scale: float = 1.0,
                    rate_hz: float = 100.0) -> JointTrajectory:
    """Convenience: trapezoidal (fast) or quintic (smooth) joint-space move."""
    vmax = (DEFAULT_VMAX if vmax is None else np.asarray(vmax)) * speed_scale
    amax = (DEFAULT_AMAX if amax is None else np.asarray(amax)) * speed_scale ** 2
    f = quintic if smooth else trapezoidal
    return to_message(f(q0, q1, vmax, amax), joint_names, stamp, rate_hz)
