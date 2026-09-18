"""Hand-eye calibration: where is the camera, relative to the robot?

THE STEP THAT MAKES PERCEPTION ACTIONABLE
-----------------------------------------
Steps 2-3 give an object's pose in the CAMERA frame. The robot moves in its
BASE frame. Nothing so far connects them, and no amount of better perception
will: a perfect 6D pose in the wrong frame is useless. Hand-eye calibration is
that missing link, and it is where these pipelines most often quietly break --
because its errors look exactly like perception errors at the gripper.

THE FORMULATION (eye-in-hand: camera bolted to the wrist)
---------------------------------------------------------
At each robot pose i we know two things:

    A_i = T_base_tool     from forward kinematics (essentially exact)
    B_i = T_cam_target    from PnP on a marker fixed to the table

The unknown X = T_tool_camera is rigid -- the camera does not move on the
wrist. The marker does not move either, so T_base_target is the same at every
pose:

    A_i · X · B_i  =  A_j · X · B_j      for all i, j

Rearranged into the classical form:

    (A_j⁻¹ A_i) · X  =  X · (B_j B_i⁻¹)          i.e.   A X = X B

WHY YOU MUST ROTATE ABOUT AT LEAST TWO NON-PARALLEL AXES
--------------------------------------------------------
This is the gate that actually matters, and it is pure geometry, not numerics.

Take rotations about a single axis only. Then every A has its rotation about
that same axis, and ANY X that commutes with those rotations -- in particular
X composed with an arbitrary extra rotation about that axis -- satisfies the
equation equally well. X is not determined: an entire one-parameter family of
solutions fits the data perfectly.

The solver does not fail. It returns one member of that family, with an
excellent residual, and the camera is mounted at a rotation you never measured.
Exactly the Step 2 lesson in a new costume: **the residual cannot tell you the
problem was well-posed.** So `rotation_axis_spread` is a hard gate.

Pure translations are worse still: they carry NO information about X's rotation
at all.

WHY WE IMPLEMENT PARK'S METHOD AS WELL AS CALLING OpenCV
--------------------------------------------------------
`cv2.calibrateHandEye` is one line and opaque. Park & Martin's method is short
enough to write and derive, and having both lets each check the other -- a
cross-validation with no shared code path, which is worth far more than one
number from one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.spatial.transform import Rotation as R

from vgp.utils.transforms import SE3


class HandEyeError(ValueError):
    pass


# --------------------------------------------------------------------------
# Park & Martin (1994), written out
# --------------------------------------------------------------------------

def solve_park(motions: list[tuple[SE3, SE3]]) -> SE3:
    """Solve A X = X B for X, given (A, B) relative-motion pairs.

    ROTATION.  Take logs: a = log(R_A), b = log(R_B) as rotation VECTORS. The
    constraint R_A R_X = R_X R_B means R_A and R_B are similar, so they share a
    rotation ANGLE and their axes are related by exactly R_X:

        a = R_X b

    That is a pure rotation-fitting problem over all motion pairs -- Wahba's
    problem. Park's closed form:

        M = Σ bᵢ aᵢᵀ,      R_X = (Mᵀ M)^(-1/2) Mᵀ

    The matrix inverse square root is the orthogonal Procrustes projection: it
    returns the closest rotation to M in the least-squares sense.

    TRANSLATION.  With R_X known, the translation part of A X = X B expands to

        R_A t_X + t_A = R_X t_B + t_X
        (R_A − I) t_X = R_X t_B − t_A

    which is linear in t_X, stacked over all pairs and solved by least squares.
    Note (R_A − I) is singular for any single motion (a rotation fixes its own
    axis) -- which is the same two-axis requirement showing up in the algebra.
    """
    if len(motions) < 2:
        raise HandEyeError(f"need >= 2 motion pairs, got {len(motions)}")

    M = np.zeros((3, 3))
    for A, B in motions:
        a = R.from_matrix(A.R).as_rotvec()
        b = R.from_matrix(B.R).as_rotvec()
        M += np.outer(b, a)

    MtM = M.T @ M
    if np.linalg.det(MtM) < 1e-18:
        raise HandEyeError(
            "rotation system is rank-deficient: motions do not rotate about "
            "enough independent axes. X is not uniquely determined."
        )
    R_x = np.real(fractional_matrix_power(MtM, -0.5)) @ M.T
    # Numerical hygiene: project back onto SO(3). The formula is exact in
    # theory; in floats it drifts, and an SE3 that is 1e-9 off orthonormal
    # will be rejected downstream for no useful reason.
    u, _, vt = np.linalg.svd(R_x)
    R_x = u @ vt
    if np.linalg.det(R_x) < 0:
        u[:, -1] *= -1
        R_x = u @ vt

    C, d = [], []
    for A, B in motions:
        C.append(A.R - np.eye(3))
        d.append(R_x @ B.t - A.t)
    t_x, *_ = np.linalg.lstsq(np.vstack(C), np.concatenate(d), rcond=None)
    return SE3.from_rt(R_x, t_x, parent="tool0", child="camera_link")


# --------------------------------------------------------------------------
# Collector + quality gates
# --------------------------------------------------------------------------

@dataclass
class HandEyeResult:
    X: SE3                                 # T_tool_camera
    X_opencv: SE3                          # independent cross-check
    n_poses: int
    rotation_axis_spread: float            # deg between the two most distinct axes
    max_rotation_angle: float              # deg
    residual_t_mm: float                   # scatter of the implied target pose
    residual_r_deg: float
    cross_check_t_mm: float                # our solver vs OpenCV's
    cross_check_r_deg: float
    per_pose_t_mm: np.ndarray = field(default_factory=lambda: np.array([]))

    MIN_POSES = 8
    MIN_AXIS_SPREAD_DEG = 30.0
    MIN_ROTATION_DEG = 20.0
    MAX_RESIDUAL_T_MM = 3.0
    MAX_RESIDUAL_R_DEG = 1.5
    MAX_CROSS_CHECK_MM = 2.0

    def problems(self) -> list[str]:
        out = []
        if self.n_poses < self.MIN_POSES:
            out.append(f"only {self.n_poses} poses (want >= {self.MIN_POSES})")
        if self.rotation_axis_spread < self.MIN_AXIS_SPREAD_DEG:
            out.append(
                f"rotation axes span only {self.rotation_axis_spread:.0f} deg "
                f"(want >= {self.MIN_AXIS_SPREAD_DEG:.0f}). Rotations about a SINGLE "
                "axis leave a whole one-parameter family of X fitting the data "
                "perfectly -- the residual will look excellent and the answer "
                "will be wrong. Rotate about a genuinely different axis."
            )
        if self.max_rotation_angle < self.MIN_ROTATION_DEG:
            out.append(
                f"largest rotation is only {self.max_rotation_angle:.0f} deg "
                f"(want >= {self.MIN_ROTATION_DEG:.0f}); small rotations give a "
                "poor signal-to-noise ratio on the axis direction"
            )
        if self.residual_t_mm > self.MAX_RESIDUAL_T_MM:
            out.append(f"target-pose scatter {self.residual_t_mm:.2f} mm "
                       f"> {self.MAX_RESIDUAL_T_MM} mm")
        if self.residual_r_deg > self.MAX_RESIDUAL_R_DEG:
            out.append(f"target-orientation scatter {self.residual_r_deg:.2f} deg "
                       f"> {self.MAX_RESIDUAL_R_DEG} deg")
        if self.cross_check_t_mm > self.MAX_CROSS_CHECK_MM:
            out.append(
                f"our solver and OpenCV's disagree by {self.cross_check_t_mm:.2f} mm "
                "-- two independent methods disagreeing means the DATA is "
                "ambiguous, not that one implementation is buggy"
            )
        return out

    @property
    def ok(self) -> bool:
        return not self.problems()

    def summary(self) -> str:
        lines = [
            f"poses            : {self.n_poses}",
            f"T_tool_camera    : {self.X}",
            f"axis spread      : {self.rotation_axis_spread:.1f} deg",
            f"max rotation     : {self.max_rotation_angle:.1f} deg",
            f"target scatter   : {self.residual_t_mm:.3f} mm / {self.residual_r_deg:.3f} deg",
            f"vs OpenCV        : {self.cross_check_t_mm:.3f} mm / {self.cross_check_r_deg:.3f} deg",
        ]
        probs = self.problems()
        lines.append("status           : OK" if not probs else "status           : PROBLEMS")
        lines += [f"  ! {p}" for p in probs]
        return "\n".join(lines)


class HandEyeCalibrator:
    """Eye-in-hand collector: pairs of (T_base_tool, T_camera_target)."""

    def __init__(self) -> None:
        self.A: list[SE3] = []      # T_base_tool
        self.B: list[SE3] = []      # T_camera_target

    def add(self, T_base_tool: SE3, T_cam_target: SE3) -> None:
        self.A.append(SE3(T_base_tool.matrix))
        self.B.append(SE3(T_cam_target.matrix))

    @property
    def n(self) -> int:
        return len(self.A)

    def _motions(self) -> list[tuple[SE3, SE3]]:
        """Relative motions between CONSECUTIVE poses.

        Consecutive rather than all-pairs: all-pairs is O(n^2) and, more
        importantly, the pairs are not independent -- the extra equations add
        correlated noise and inflate confidence without adding information.
        """
        out = []
        for i in range(1, self.n):
            # From  A_j X B_j = A_i X B_i  (the target does not move):
            #
            #     A_j⁻¹ A_i · X  =  X · B_j B_i⁻¹
            #       \_______/            \______/
            #           A                    B
            #
            # With j = i-1. Note the ASYMMETRY: A composes as (older)⁻¹(newer)
            # while B composes as (older)(newer)⁻¹. Writing B the same way round
            # as A gives B⁻¹ instead of B, and the solver then returns an X that
            # is wrong by roughly 180 degrees -- with a large, honest residual,
            # which is how it was caught. Derive it, do not pattern-match it.
            A = self.A[i - 1].inv() @ self.A[i]
            B = self.B[i - 1] @ self.B[i].inv()
            out.append((SE3(A.matrix), SE3(B.matrix)))
        return out

    def axis_spread_deg(self) -> tuple[float, float]:
        """(max angle between motion rotation axes, max rotation magnitude).

        This is the identifiability check, computed from the DATA alone --
        before any solve, so it can be reported live while collecting.
        """
        axes, angles = [], []
        for A, _ in self._motions():
            rv = R.from_matrix(A.R).as_rotvec()
            ang = np.linalg.norm(rv)
            if ang < np.deg2rad(2.0):
                continue            # too small: axis direction is pure noise
            axes.append(rv / ang)
            angles.append(np.degrees(ang))
        if len(axes) < 2:
            return 0.0, (max(angles) if angles else 0.0)
        axes = np.array(axes)
        # |cos| because an axis and its negation are the same axis.
        cos = np.clip(np.abs(axes @ axes.T), 0, 1)
        return float(np.degrees(np.arccos(cos.min()))), float(max(angles))

    def solve(self) -> HandEyeResult:
        if self.n < 3:
            raise HandEyeError(f"need >= 3 poses, got {self.n}")
        motions = self._motions()
        X = solve_park(motions)

        # Independent cross-check. OpenCV wants T_base_gripper and T_cam_target
        # as separate R/t lists, and returns T_gripper_cam -- the same X.
        R_g2b = [a.R for a in self.A]
        t_g2b = [a.t for a in self.A]
        R_t2c = [b.R for b in self.B]
        t_t2c = [b.t for b in self.B]
        R_cv, t_cv = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                          method=cv2.CALIB_HAND_EYE_PARK)
        X_cv = SE3.from_rt(R_cv, t_cv.ravel(), parent="tool0", child="camera_link")

        # RESIDUAL: with the true X, T_base_target = A_i X B_i is the SAME at
        # every pose, because the marker does not move. Its scatter is the most
        # meaningful error measure we have -- it is in millimetres, it is the
        # quantity the robot actually cares about, and unlike an algebraic
        # residual it has a physical interpretation.
        implied = [SE3((a @ X @ b).matrix) for a, b in zip(self.A, self.B)]
        centre_t = np.mean([p.t for p in implied], axis=0)
        per_pose = np.array([np.linalg.norm(p.t - centre_t) * 1000 for p in implied])
        mean_R = R.from_matrix(np.array([p.R for p in implied])).mean().as_matrix()
        res_r = max(np.degrees(np.linalg.norm(
            R.from_matrix(p.R @ mean_R.T).as_rotvec())) for p in implied)

        spread, max_ang = self.axis_spread_deg()
        dt, dr = X.distance_to(X_cv)
        return HandEyeResult(
            X=X, X_opencv=X_cv, n_poses=self.n,
            rotation_axis_spread=spread, max_rotation_angle=max_ang,
            residual_t_mm=float(per_pose.max()), residual_r_deg=float(res_r),
            cross_check_t_mm=float(dt * 1000), cross_check_r_deg=float(np.degrees(dr)),
            per_pose_t_mm=per_pose,
        )
