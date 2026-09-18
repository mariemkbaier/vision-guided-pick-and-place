"""Grasp pose construction, exploiting the gripper's own symmetries.

WHY THIS FILE EXISTS RATHER THAN BUILDING A ROTATION MATRIX INLINE
------------------------------------------------------------------
A top-down grasp on a cube looks like it has one answer: point the tool down,
align the fingers with the object's yaw. It actually has FOUR, and knowing that
is the difference between an IK that occasionally fails for no visible reason
and one that does not.

A parallel-jaw gripper is symmetric under a 180 degree rotation about its own
approach axis -- swapping which finger is on which side grasps the same object
identically. So yaw and yaw+pi are the SAME physical grasp. And a square cube
adds its own 90 degree symmetry, giving four equivalent yaws.

THAT SYMMETRY IS NOT COSMETIC -- IT RESCUES SINGULAR CONFIGURATIONS
-------------------------------------------------------------------
Measured on this arm: a top-down grasp at (0.30, 0, 0.15) with yaw = 0 requires
a wrist roll of exactly pi. The wrist is Y-Z-Y, so roll = pi is a gimbal-lock
singularity -- the damped solver correctly refuses to move in the lost
direction and stalls with ~4.5 degrees of orientation error that no number of
iterations removes.

The equivalent grasp at yaw = pi needs a roll of 0, which is perfectly
conditioned. Same physical grasp, same fingers on the same faces, trivially
solvable. Enumerating the symmetry costs nothing and turns an unexplained IK
failure into a non-event.

The general lesson: when IK fails, ask whether the TASK really required that
exact pose, or whether you over-specified it. Most manipulation tasks have
symmetries, and spending them is free.
"""

from __future__ import annotations

import numpy as np

from vgp.utils.transforms import SE3


def top_down_grasp(position, yaw: float = 0.0, approach_axis="-z") -> SE3:
    """Grasp pose with the tool pointing along `approach_axis` in base_link.

    Tool convention (see scene.py): the gripper's +Z points from the flange out
    toward the fingertips, so "pointing down" means tool +Z = base -Z. The
    fingers open along tool +/-X, so `yaw` rotates the finger line.
    """
    if approach_axis != "-z":
        raise ValueError("only top-down (-z) approach is implemented")
    # Tool Z = base -Z. Tool X = the finger-opening direction at `yaw`.
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.cross(z, x)
    return SE3.from_rt(np.column_stack([x, y, z]), np.asarray(position, float),
                       parent="base_link", child="grasp")


def equivalent_yaws(yaw: float, object_symmetry: int = 4) -> list[float]:
    """Yaws giving a physically identical grasp, best-conditioned first.

    `object_symmetry` is the object's own rotational symmetry about the
    approach axis: 4 for a square cube, 2 for a rectangular box, 1 for an
    asymmetric object. The GRIPPER contributes a factor of 2 regardless,
    because swapping its fingers changes nothing.

    Ordered by |wrapped angle| so the first candidate is the one closest to
    zero yaw, which empirically needs the least wrist travel.
    """
    n = max(1, object_symmetry)
    step = 2 * np.pi / n
    # Round before de-duplicating. Wrapping through arctan2 leaves float noise,
    # so 0.3 and 0.2999999999999996 survive as distinct set members -- which
    # returned 6 "unique" yaws for a 4-fold symmetry, two of them duplicates
    # that each cost a full multistart IK solve. 1e-6 rad is far below any
    # mechanical relevance and far above the noise.
    seen: dict[float, float] = {}
    for k in range(n):
        for m in (0, 1):
            a = yaw + k * step + m * np.pi
            w = float(np.arctan2(np.sin(a), np.cos(a)))
            seen.setdefault(round(w, 6), w)
    return sorted(seen.values(), key=abs)


def grasp_candidates(position, yaw: float = 0.0, object_symmetry: int = 4) -> list[SE3]:
    """All equivalent grasp poses for this position, best-conditioned first."""
    return [top_down_grasp(position, y) for y in equivalent_yaws(yaw, object_symmetry)]
