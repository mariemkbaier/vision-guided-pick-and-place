"""Straight-line Cartesian motion, and why it is worth the trouble.

JOINT-SPACE VS CARTESIAN -- USE BOTH, FOR DIFFERENT REASONS
------------------------------------------------------------
A joint-space move interpolates joint angles. It is fast, always feasible (it
never leaves the reachable set), and cannot hit a singularity mid-path. What it
does NOT do is travel in a straight line: the tool sweeps an arc whose shape
depends on the arm's configuration.

That arc is fine in free air and unacceptable for the last 10 cm of a grasp.
Approaching a cube, the gripper must come straight down between its neighbours;
an arc can sweep a fingertip through the object it is about to pick, or through
the one beside it. Same on retreat, where an arc drags the grasped object
sideways before lifting it clear.

So: joint-space for transit, Cartesian for approach and retreat. This is the
standard division and it is a genuine trade, not a style choice.

WHAT MAKES CARTESIAN HARDER
---------------------------
Every waypoint needs IK, so a straight line can be interrupted by a singularity
or a joint limit partway along -- and unlike a joint move, you only discover
that halfway through. We plan the WHOLE path before executing any of it, and
fail as a unit. Finding out at waypoint 7 of 20 that the arm cannot continue is
how you end up stopping with the gripper inside an object.

We also seed each IK solve from the previous solution. That keeps the arm on
one continuous branch: without it, consecutive waypoints can land in different
IK solutions and the arm flips configuration mid-approach -- a large, fast,
alarming motion in the middle of a delicate move.
"""

from __future__ import annotations

import numpy as np

from vgp.msgs import JointTrajectory
from vgp.planning.trajectory import (
    DEFAULT_AMAX, DEFAULT_VMAX, plan_joint_move, to_message, trapezoidal,
)
from vgp.robot.arm import Arm
from vgp.robot.ik import solve_ik
from vgp.utils.transforms import SE3, interpolate


class CartesianPlanFailure(RuntimeError):
    """A waypoint along the straight line was unreachable."""


def plan_cartesian_path(
    arm: Arm, start_q: np.ndarray, target: SE3, steps: int = 24,
    site: int | None = None, max_joint_jump: float = 0.5,
) -> list[np.ndarray]:
    """Joint waypoints tracing a straight Cartesian line to `target`.

    `max_joint_jump` catches configuration flips: a small Cartesian step that
    produces a large joint step means IK jumped to a different branch. We
    refuse rather than execute it, because that flip is a fast unplanned swing
    with the gripper near an object.
    """
    site = arm.grasp_sid if site is None else site
    start_pose = arm.fk(site, start_q)
    q = np.asarray(start_q, float).copy()
    out = [q.copy()]
    for i in range(1, steps + 1):
        waypoint = interpolate(SE3(start_pose.matrix), SE3(target.matrix), i / steps)
        r = solve_ik(arm, SE3(waypoint.matrix, "base_link", "grasp"), q_init=q,
                     site=site, q_rest=q, nullspace_gain=0.02)
        if not r.success:
            raise CartesianPlanFailure(
                f"straight-line path blocked at waypoint {i}/{steps} "
                f"({100 * i / steps:.0f}% along): {r.message}. "
                "Plan the whole path before moving -- discovering this mid-motion "
                "means stopping with the gripper somewhere unhelpful."
            )
        jump = float(np.abs(r.q - q).max())
        if jump > max_joint_jump:
            raise CartesianPlanFailure(
                f"IK branch flip at waypoint {i}/{steps}: one joint moved "
                f"{jump:.2f} rad for a {1/steps:.0%} Cartesian step. Executing "
                "that would be a large unplanned swing near the object."
            )
        q = r.q
        out.append(q.copy())
    return out


def plan_cartesian_move(
    arm: Arm, start_q: np.ndarray, target: SE3, joint_names, stamp: float,
    steps: int = 24, site: int | None = None, speed_scale: float = 0.35,
    rate_hz: float = 100.0,
) -> JointTrajectory:
    """Straight-line Cartesian move as a time-parameterised joint trajectory.

    `speed_scale` defaults low: approach moves happen near objects, and the
    time saved by rushing the last 10 cm is not worth the contact forces.
    """
    waypoints = plan_cartesian_path(arm, start_q, target, steps, site)
    # Time-parameterise using the TOTAL joint displacement, so a long Cartesian
    # move takes proportionally longer. Using per-segment trapezoids instead
    # would stop and restart at every waypoint -- 24 tiny accel/decel cycles.
    total = np.sum([np.abs(waypoints[i + 1] - waypoints[i]) for i in range(len(waypoints) - 1)],
                   axis=0)
    prof = trapezoidal(np.zeros_like(total), total,
                       DEFAULT_VMAX * speed_scale, DEFAULT_AMAX * speed_scale ** 2)
    cum = np.concatenate([[0.0], np.cumsum(
        [np.linalg.norm(waypoints[i + 1] - waypoints[i]) for i in range(len(waypoints) - 1)])])
    arc = cum / max(cum[-1], 1e-9)

    def sample(t: float):
        s, _ = prof.sample(t)
        frac = float(np.clip(s[np.argmax(np.abs(total))] / max(np.abs(total).max(), 1e-9), 0, 1)) \
            if np.abs(total).max() > 1e-9 else 1.0
        i = int(np.searchsorted(arc, frac, side="right")) - 1
        i = int(np.clip(i, 0, len(waypoints) - 2))
        span = max(arc[i + 1] - arc[i], 1e-9)
        a = (frac - arc[i]) / span
        q = waypoints[i] + a * (waypoints[i + 1] - waypoints[i])
        return q, np.zeros_like(q)

    from vgp.planning.trajectory import Profile
    return to_message(Profile(prof.duration, sample), joint_names, stamp, rate_hz)
