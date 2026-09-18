"""Inverse kinematics by damped least squares, with nullspace shaping.

WE WRITE THIS BECAUSE MUJOCO HAS NO IK -- AND THAT TURNED OUT TO BE A GAIN
--------------------------------------------------------------------------
PyBullet would have given us `calculateInverseKinematics()`, one call, opaque.
Deriving it means we can say exactly what happens near a singularity, why the
robot sometimes takes a strange route, and how posture is controlled. That is
the difference between using IK and understanding it.

THE DERIVATION
--------------
The Jacobian maps joint velocity to end-effector velocity:  v = J q̇.
For a small pose error e we want q̇ with J q̇ = e. J is square here (6 joints,
6 DoF) but that does not save us -- it becomes ILL-CONDITIONED near
singularities, where the arm loses the ability to move in some direction.

The naive answer q̇ = J⁻¹ e blows up there: as a singular value σ → 0, the
commanded joint velocity ~ e/σ → ∞. In simulation that is a NaN; on hardware
it is the robot flinging itself at its velocity limit. This is the single most
important practical fact about IK.

Damped least squares (Levenberg-Marquardt) minimises ‖J q̇ − e‖² + λ²‖q̇‖²:

    q̇ = Jᵀ (J Jᵀ + λ² I)⁻¹ e

Each singular value σ is replaced by σ/(σ² + λ²). For σ ≫ λ that is ~1/σ, the
exact inverse. For σ → 0 it goes to σ/λ² → 0 instead of ∞. So the solver
smoothly gives UP on directions the arm cannot move in, rather than screaming.
The cost is a small tracking error near singularities — which is the right
trade: a slightly wrong pose beats an unbounded velocity.

ADAPTIVE DAMPING
----------------
A fixed λ is either too sloppy far from singularities or too weak near them.
We scale λ with how close we are, measured by the smallest singular value.
Away from trouble the solver is nearly exact; near it, damping rises smoothly.

NULLSPACE (the part PyBullet's one-liner hides)
-----------------------------------------------
With redundancy, q̇ can carry a secondary term that does not disturb the task:

    q̇ = q̇_task + (I − J⁺J) q̇_null

`(I − J⁺J)` projects onto motions invisible at the end-effector. We use it to
pull joints toward a comfortable posture and away from their limits. A 6-DoF
arm at a 6-DoF task has no true nullspace, but DAMPING creates an effective
one near singularities — which is exactly when posture help matters most.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vgp.robot.arm import Arm
from vgp.utils.transforms import SE3


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    pos_err: float           # metres
    rot_err: float           # radians
    iterations: int
    min_singular_value: float
    message: str = ""


def pose_error(current: SE3, target: SE3) -> np.ndarray:
    """6-vector [linear; angular] error taking `current` to `target`.

    The angular part is the rotation VECTOR of the relative rotation, expressed
    in the BASE frame. Two things are easy to get wrong here:

      - Using Euler-angle differences instead. Those are not a valid error
        metric: they wrap, they are sequence-dependent, and near gimbal lock a
        tiny orientation change produces a huge "error".
      - Expressing the rotation vector in the TOOL frame while the Jacobian is
        in the base frame. That mismatch produces an IK that converges in
        position and spirals in orientation -- a memorable afternoon.
    """
    from scipy.spatial.transform import Rotation as R

    e_pos = target.t - current.t
    e_rot = R.from_matrix(target.R @ current.R.T).as_rotvec()
    return np.concatenate([e_pos, e_rot])


def solve_ik(
    arm: Arm,
    target: SE3,
    q_init: np.ndarray | None = None,
    site: int | None = None,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    max_iters: int = 200,
    damping: float = 0.02,
    q_rest: np.ndarray | None = None,
    nullspace_gain: float = 0.05,
    max_step: float = 0.20,
) -> IKResult:
    """Iterative damped-least-squares IK for `target` (in base_link)."""
    site = arm.grasp_sid if site is None else site
    q = arm.q.copy() if q_init is None else np.asarray(q_init, dtype=np.float64).copy()
    q_rest = arm.q.copy() if q_rest is None else np.asarray(q_rest, dtype=np.float64)

    sigma_min = 0.0
    for it in range(max_iters):
        current = arm.fk(site, q)
        e = pose_error(current, target)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        if pos_err < pos_tol and rot_err < rot_tol:
            return IKResult(q, True, pos_err, rot_err, it, sigma_min, "converged")

        J = arm.jacobian(site, q)

        # JOINT-LIMIT MASKING.
        # Clamping q after the step is not enough. A joint sitting ON its limit
        # still has a non-zero Jacobian column, so the solver keeps allocating
        # motion to a joint that cannot move -- the step is then silently
        # truncated by the clamp and the residual never goes away. Symptom: IK
        # stalls with one joint pinned and a few degrees of orientation error
        # that no number of iterations removes.
        #
        # Zeroing those columns tells the solver the truth: that direction is
        # unavailable, go find another route. We only mask joints being pushed
        # FURTHER into their limit, so a joint at a limit can still come back.
        at_lo = q <= arm.qlim[:, 0] + 1e-9
        at_hi = q >= arm.qlim[:, 1] - 1e-9
        if at_lo.any() or at_hi.any():
            probe = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), e)
            blocked = (at_lo & (probe < 0)) | (at_hi & (probe > 0))
            if blocked.any():
                J = J.copy()
                J[:, blocked] = 0.0

        sigma = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(sigma[-1])

        # Adaptive damping: near-exact far from singularities, strongly damped
        # close to them. The threshold is where the smallest singular value
        # stops being comfortably larger than our base damping.
        lam = damping
        sigma_threshold = 10.0 * damping
        if sigma_min < sigma_threshold:
            # Quadratic ramp so damping is continuous in sigma_min -- a step
            # change would make the trajectory jerk as the arm passes near a
            # singularity.
            lam = damping * (1.0 + 8.0 * (1.0 - sigma_min / sigma_threshold) ** 2)

        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(6), e)

        # Nullspace: pull toward a comfortable posture without disturbing the
        # task. `pinv` here is the DAMPED inverse, so the projector reflects the
        # directions we are actually able to control.
        # Fade the secondary objective out as the task error shrinks.
        #
        # The nullspace projector is built from the DAMPED inverse, so it is
        # only approximately orthogonal to the task -- a little of the posture
        # term leaks into end-effector motion. Far from the target that is
        # irrelevant; near it, the leak is the same size as the remaining error
        # and the solve stalls around 0.5 mm instead of converging. Measured:
        # stall at ~0.42 mm with a constant gain, ~0.0002 mm with this fade.
        #
        # Fading keeps posture shaping where it matters (during the approach,
        # when there is room to choose) and removes it where it does not.
        err_mag = max(pos_err, rot_err * 0.1)
        ns_scale = min(1.0, err_mag / 0.01)
        if nullspace_gain > 0 and ns_scale > 1e-3:
            J_pinv = J.T @ np.linalg.inv(JJt + (lam ** 2) * np.eye(6))
            N = np.eye(len(q)) - J_pinv @ J
            # Also push away from joint limits: the gradient of a simple
            # centring cost. Without it the arm creeps into a limit and then
            # cannot rotate the wrist when it finally needs to.
            mid = arm.qlim.mean(axis=1)
            span = np.maximum(arm.qlim[:, 1] - arm.qlim[:, 0], 1e-6)
            limit_grad = -2.0 * (q - mid) / span ** 2
            dq += N @ (nullspace_gain * ns_scale * ((q_rest - q) + 0.02 * limit_grad))

        # Step limiting keeps the linearisation valid. Without it a large
        # initial error produces a step where J is no longer a good model, and
        # the solve oscillates instead of converging.
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        q = arm.clamp(q + dq)

    current = arm.fk(site, q)
    e = pose_error(current, target)
    pos_err, rot_err = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
    ok = pos_err < pos_tol * 10 and rot_err < rot_tol * 10
    return IKResult(
        q, ok, pos_err, rot_err, max_iters, sigma_min,
        f"hit iteration limit (pos {pos_err*1000:.2f}mm, rot {np.degrees(rot_err):.2f}deg)"
        + ("; near-singular" if sigma_min < 0.02 else ""),
    )


def solve_ik_multistart(
    arm: Arm, target: SE3, seeds: list[np.ndarray] | None = None,
    collision_free: bool = True, **kw
) -> IKResult:
    """Try several seeds and keep the best.

    IK is non-convex: a single seed can land in a local minimum that is nowhere
    near the target even though a solution exists. Multiple seeds is the cheap,
    honest fix -- and cheaper than it sounds, because a failing seed usually
    fails fast.

    Seeds are ordered so the CURRENT configuration is tried first: if it works,
    the arm moves the shortest distance, which matters both for cycle time and
    for not swinging through the workspace unnecessarily.
    """
    rng = np.random.default_rng(0)
    if seeds is None:
        seeds = [arm.q.copy()]
        mid = arm.qlim.mean(axis=1)
        seeds.append(mid)
        for _ in range(6):
            seeds.append(rng.uniform(arm.qlim[:, 0] * 0.8, arm.qlim[:, 1] * 0.8))

    best: IKResult | None = None
    blocked: IKResult | None = None
    for s in seeds:
        r = solve_ik(arm, target, q_init=s, **kw)
        if r.success:
            # A pose-correct solution the arm cannot physically adopt is not a
            # solution. See Arm.in_collision for why this is not optional.
            if collision_free and arm.in_collision(r.q):
                blocked = r
                continue
            return r
        # Rank failures by position error; a near-miss is more useful to report
        # than an arbitrary one.
        if best is None or r.pos_err < best.pos_err:
            best = r
    if best is None and blocked is not None:
        return IKResult(blocked.q, False, blocked.pos_err, blocked.rot_err,
                        blocked.iterations, blocked.min_singular_value,
                        "reachable but self-colliding in every branch tried")
    if blocked is not None and best is not None and not best.success:
        best.message += "; a pose-correct branch existed but was self-colliding"
    return best
