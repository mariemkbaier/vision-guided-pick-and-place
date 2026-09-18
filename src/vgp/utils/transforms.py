"""Rigid-body geometry: SE(3) with frame labels.

WHY THIS FILE IS FIRST
----------------------
Almost every bug in a perception-to-control pipeline is a frame bug, and frame
bugs are silent: the arm moves smoothly to confidently wrong place. There is no
exception, no NaN. You lose a day.

Three specific traps, and what we do about each:

1. DIRECTION AMBIGUITY.  "The camera-to-robot transform" names nothing. Does it
   map camera points into robot coordinates, or describe where the camera sits
   in robot coordinates? Those are inverses of each other.
   -> We use one naming rule EVERYWHERE, encoded in the type:

          T_a_b  maps points expressed in frame b  into frame a.
          p_a = T_a_b @ p_b

      Read it as "a from b" (hence `.parent`=a, `.child`=b). Composition then
      reads like cancelling fractions, which is the whole point:

          T_a_c = T_a_b @ T_b_c        (the two b's touch and cancel)

      And `T_a_b` also literally IS the pose of frame b expressed in frame a --
      its translation column is b's origin seen from a. Both readings are true;
      that is not a coincidence, and it is worth being able to say why.

2. QUATERNION ORDER.  scipy and ROS use (x, y, z, w). MuJoCo uses (w, x, y, z).
   A silent swap is a rotation that looks plausible and is wrong.
   -> No function here takes a bare "quat". Every entry point says which order
      in its NAME (`from_quat_xyzw` / `from_quat_wxyz`). Verbose on purpose:
      you cannot pass the wrong one without typing the wrong word.

3. MIXING FRAMES.  Composing T_base_cam with T_tag_cam is a type error in
   spirit but plain matrix multiply in numpy -- it happily returns garbage.
   -> SE3 carries optional frame labels and REFUSES to compose when the inner
      frames disagree. This has caught more of our own mistakes than any test.

WHY A CLASS AND NOT RAW 4x4 ARRAYS
----------------------------------
Raw arrays compose with `@` and hand straight to cv2/MuJoCo, which is genuinely
convenient. We gave that up for the frame checking in trap 3 -- the bug class it
removes is worth more than the convenience. `.as_matrix()` is the escape hatch
at every library boundary, and it is deliberately one call so those boundaries
stay visible in the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

# Tolerance for "is this really a rotation matrix". Loose enough to survive
# float32 round-trips through cv2/MuJoCo, tight enough to catch a real error
# (a shear, a scale, a reflection) rather than accumulated noise.
_ORTHO_ATOL = 1e-5


class FrameMismatch(ValueError):
    """Raised when composing transforms whose inner frames do not agree.

    Deliberately its own type: in a task state machine you may want to catch
    this and abort the task, distinct from a numerical failure.
    """


@dataclass(frozen=True)
class SE3:
    """A rigid transform, optionally labelled with the frames it connects.

    `matrix` is 4x4 homogeneous. `parent`/`child` are frame names such that
    this object maps points from `child` coordinates into `parent` coordinates.
    Labels are optional -- `None` means "unlabelled", and unlabelled transforms
    compose with anything. That escape valve matters for pure math helpers
    (interpolation, test fixtures) that are genuinely frame-agnostic.

    Frozen because transforms are values, not state. Sharing a mutable pose
    between a perception node and a control node is a data race waiting to
    happen; making it immutable means we can pass it over the bus without
    copying and without fear.
    """

    matrix: np.ndarray
    parent: str | None = None
    child: str | None = None

    # ---------------------------------------------------------------- ctors

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"SE3 needs a 4x4 matrix, got {m.shape}")
        # Validate the rotation block. We check here, at construction, rather
        # than lazily -- a bad rotation that enters the system propagates into
        # every downstream pose and the origin becomes untraceable.
        rot = m[:3, :3]
        if not np.allclose(rot @ rot.T, np.eye(3), atol=_ORTHO_ATOL):
            raise ValueError(
                "SE3 rotation block is not orthonormal "
                f"(max |RR^T - I| = {np.abs(rot @ rot.T - np.eye(3)).max():.2e}). "
                "Usual cause: a matrix built column-major, or a scale baked in."
            )
        det = float(np.linalg.det(rot))
        if det < 0:
            # det == -1 is a reflection: a mirrored frame. This is what you get
            # from a left-handed convention or a flipped axis, and it will make
            # a gripper approach from the wrong side.
            raise ValueError(
                f"SE3 rotation has det={det:.3f} < 0 -- that is a reflection, "
                "not a rotation. Usual cause: a left-handed axis convention."
            )
        # Re-orthonormalise the stored copy so tiny drift never accumulates
        # across long composition chains (tag -> cam -> base -> tool -> ...).
        u, _, vt = np.linalg.svd(rot)
        clean = np.eye(4)
        clean[:3, :3] = u @ vt
        clean[:3, 3] = m[:3, 3]
        object.__setattr__(self, "matrix", clean)

    @staticmethod
    def identity(parent: str | None = None, child: str | None = None) -> "SE3":
        return SE3(np.eye(4), parent, child)

    @staticmethod
    def from_rt(
        rot: np.ndarray, t: np.ndarray, parent: str | None = None, child: str | None = None
    ) -> "SE3":
        """Build from a 3x3 rotation and a 3-vector translation.

        This is the shape cv2 and MuJoCo hand back, so it is the most-used ctor.
        """
        m = np.eye(4)
        m[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        m[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        return SE3(m, parent, child)

    @staticmethod
    def from_rotvec(
        rvec: np.ndarray, t: np.ndarray, parent: str | None = None, child: str | None = None
    ) -> "SE3":
        """Build from an axis-angle (Rodrigues) vector -- what solvePnP returns.

        cv2 gives you `rvec` whose direction is the rotation axis and whose norm
        is the angle in radians. We convert here rather than calling
        cv2.Rodrigues at each call site, so PnP results enter the system as
        labelled SE3s immediately and never float around as bare arrays.
        """
        rot = R.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
        return SE3.from_rt(rot, t, parent, child)

    @staticmethod
    def from_quat_xyzw(
        q: np.ndarray, t: np.ndarray, parent: str | None = None, child: str | None = None
    ) -> "SE3":
        """(x, y, z, w) order -- the ROS and scipy convention."""
        rot = R.from_quat(np.asarray(q, dtype=np.float64).reshape(4)).as_matrix()
        return SE3.from_rt(rot, t, parent, child)

    @staticmethod
    def from_quat_wxyz(
        q: np.ndarray, t: np.ndarray, parent: str | None = None, child: str | None = None
    ) -> "SE3":
        """(w, x, y, z) order -- the MuJoCo convention.

        Exists purely so the MuJoCo boundary is explicit. Every call site that
        touches `mjData.xquat` uses this one and is greppable.
        """
        q = np.asarray(q, dtype=np.float64).reshape(4)
        return SE3.from_quat_xyzw(np.array([q[1], q[2], q[3], q[0]]), t, parent, child)

    @staticmethod
    def from_euler(
        seq: str,
        angles: Iterable[float],
        t: np.ndarray | None = None,
        degrees: bool = False,
        parent: str | None = None,
        child: str | None = None,
    ) -> "SE3":
        """Euler angles, with the convention string MANDATORY (e.g. "xyz", "ZYX").

        There is no default sequence on purpose. "Euler angles" without a
        sequence and a case (intrinsic vs extrinsic) is not a specification, and
        defaulting one invites exactly the silent error this module exists to
        prevent. scipy's rule: lowercase = extrinsic, uppercase = intrinsic.
        """
        rot = R.from_euler(seq, list(angles), degrees=degrees).as_matrix()
        return SE3.from_rt(rot, np.zeros(3) if t is None else t, parent, child)

    # ----------------------------------------------------------- accessors

    @property
    def R(self) -> np.ndarray:  # noqa: N802 - matches maths notation, not PEP8
        return self.matrix[:3, :3].copy()

    @property
    def t(self) -> np.ndarray:
        return self.matrix[:3, 3].copy()

    def as_matrix(self) -> np.ndarray:
        """Escape hatch to raw 4x4 for library boundaries (cv2, MuJoCo)."""
        return self.matrix.copy()

    def quat_xyzw(self) -> np.ndarray:
        return R.from_matrix(self.R).as_quat()

    def quat_wxyz(self) -> np.ndarray:
        q = self.quat_xyzw()
        return np.array([q[3], q[0], q[1], q[2]])

    def rotvec(self) -> np.ndarray:
        return R.from_matrix(self.R).as_rotvec()

    # --------------------------------------------------------- composition

    def inv(self) -> "SE3":
        """Inverse -- and note the labels swap, which is the whole point.

        Computed as [R^T | -R^T t] rather than np.linalg.inv: exact for a rigid
        transform, ~10x faster, and cannot introduce the tiny non-orthonormality
        that a general inverse would.
        """
        rt = self.R.T
        m = np.eye(4)
        m[:3, :3] = rt
        m[:3, 3] = -rt @ self.t
        return SE3(m, parent=self.child, child=self.parent)

    def __matmul__(self, other: "SE3") -> "SE3":
        """T_a_b @ T_b_c -> T_a_c, refusing the compose if the b's disagree."""
        if not isinstance(other, SE3):
            return NotImplemented
        # Only check when BOTH labels are present. Unlabelled transforms are the
        # documented escape valve for frame-agnostic maths.
        if self.child is not None and other.parent is not None and self.child != other.parent:
            raise FrameMismatch(
                f"cannot compose ({self.parent}<-{self.child}) @ "
                f"({other.parent}<-{other.child}): inner frames "
                f"'{self.child}' != '{other.parent}'. "
                "Did you mean to invert one of them?"
            )
        return SE3(self.matrix @ other.matrix, parent=self.parent, child=other.child)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform points. Accepts (3,) or (N, 3); returns the same shape.

        Uses R @ p + t rather than building homogeneous coordinates -- same
        result, no allocation of an (N,4) intermediate. Matters when this runs
        per-frame on a point cloud.
        """
        p = np.asarray(points, dtype=np.float64)
        single = p.ndim == 1
        p = np.atleast_2d(p)
        if p.shape[-1] != 3:
            raise ValueError(f"expected points with last dim 3, got {p.shape}")
        out = p @ self.R.T + self.t
        return out[0] if single else out

    # ------------------------------------------------------------- metrics

    def translation_norm(self) -> float:
        return float(np.linalg.norm(self.t))

    def rotation_angle(self) -> float:
        """Geodesic rotation magnitude in radians, in [0, pi]."""
        return float(np.linalg.norm(R.from_matrix(self.R).as_rotvec()))

    def distance_to(self, other: "SE3") -> tuple[float, float]:
        """(metres, radians) between two poses.

        Returned as a PAIR, never a single blended scalar. Combining them needs
        a metres-per-radian weight that depends entirely on the task -- for a
        5 cm cube, 1 degree of gripper yaw is nearly free, while 1 degree of
        approach-axis tilt can miss the grasp. We refuse to pick that weight
        here and make the caller face it. Every eval metric in Step 7 reports
        the two separately for the same reason.
        """
        delta = self.inv() @ SE3(other.matrix)  # labels dropped: pure difference
        return delta.translation_norm(), delta.rotation_angle()

    def __repr__(self) -> str:
        x, y, z = self.t
        rpy = np.degrees(R.from_matrix(self.R).as_euler("xyz"))
        lbl = f"{self.parent}<-{self.child}" if self.parent or self.child else "unlabelled"
        return (
            f"SE3({lbl}, t=[{x:+.4f} {y:+.4f} {z:+.4f}]m, "
            f"rpy=[{rpy[0]:+.1f} {rpy[1]:+.1f} {rpy[2]:+.1f}]deg)"
        )


# --------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------

def interpolate(a: SE3, b: SE3, s: float) -> SE3:
    """Interpolate a->b at s in [0,1]: SLERP the rotation, LERP the translation.

    WHY NOT a full screw-motion (SE(3) exponential) interpolation? That would be
    the geometrically "pure" answer and traces a helix. We do not want a helix:
    for Cartesian approach moves we want the tool to travel in a straight line
    while its orientation turns smoothly, which is exactly decoupled LERP+SLERP.
    Straight-line approach is also what makes collision reasoning tractable.
    """
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"interpolation parameter must be in [0,1], got {s}")
    key = R.from_matrix(np.stack([a.R, b.R]))
    # Slerp handles the double-cover sign issue (q and -q are the same rotation)
    # by picking the shorter arc -- doing this by hand is a classic source of a
    # robot taking the 350-degree way round.
    from scipy.spatial.transform import Slerp

    rot = Slerp([0.0, 1.0], key)([s])[0].as_matrix()
    t = (1.0 - s) * a.t + s * b.t
    return SE3.from_rt(rot, t, parent=a.parent, child=a.child)
