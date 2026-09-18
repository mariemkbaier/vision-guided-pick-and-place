"""Collecting hand-eye poses: generating motions that make X identifiable.

The solver is the easy half. The hard half is collecting data that DETERMINES
X, and the constraints pull against each other:

  - rotate about several independent axes    (or X is not identifiable at all)
  - keep the marker in view the whole time   (or there is no measurement)
  - keep the marker large and well-resolved  (depth error scales as 1/size)
  - stay inside joint limits and reachable   (or the pose does not exist)

Rotating the wrist hard is exactly what swings the marker out of frame. In
practice a human jogs the robot until the target is visible and repeats;
we automate that with rejection sampling, which is the same strategy Step 2
used for calibration-board poses and for the same reason.

Poses are generated as CAMERA poses looking at the target, then converted to
tool poses through a NOMINAL guess at X. That guess does not need to be good --
it only has to put the marker in frame. The calibration then measures the true
X, including however wrong the guess was. That distinction matters: using a
nominal X to generate poses is legitimate; using it in the solve would be
circular.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from vgp.robot.arm import Arm
from vgp.robot.ik import solve_ik_multistart
from vgp.utils.transforms import SE3


def look_at(eye: np.ndarray, target: np.ndarray, roll: float = 0.0) -> SE3:
    """Camera pose (OPTICAL convention) at `eye` looking at `target`.

    Optical convention: +Z toward the scene, +X right, +Y down. Building the
    basis explicitly rather than with a library call keeps the convention
    visible -- a look-at helper that silently assumes OpenGL (-Z forward) is a
    classic way to get a 180-degree error.
    """
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    # World -Z as the "up" hint, because optical +Y points DOWN.
    up = np.array([0.0, 0.0, -1.0])
    if abs(up @ z) > 0.98:                      # degenerate: looking straight down
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rm = np.column_stack([x, y, z]) @ R.from_euler("z", roll).as_matrix()
    return SE3.from_rt(Rm, eye, parent="base_link", child="camera_optical")


def generate_poses(
    arm: Arm,
    target_xy: tuple[float, float],
    X_nominal: SE3,
    n: int = 14,
    dist_range=(0.22, 0.34),
    polar_range=(12.0, 45.0),
    roll_range=(-0.7, 0.7),
    q_seed: np.ndarray | None = None,
    seed: int = 0,
) -> list[np.ndarray]:
    """Joint configurations whose wrist camera looks at `target_xy` on the table.

    Viewpoints are sampled on a spherical cap above the target. `polar_range`
    is the tilt away from straight-down: a spread there is what produces
    rotations about DIFFERENT axes, which is the identifiability requirement.
    Sampling only straight-down views gives beautiful images and an
    unidentifiable calibration.
    """
    rng = np.random.default_rng(seed)
    tx, ty = target_xy
    target = np.array([tx, ty, 0.0])
    out: list[np.ndarray] = []
    q_prev = arm.q.copy() if q_seed is None else q_seed.copy()

    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        polar = np.deg2rad(rng.uniform(*polar_range))
        azim = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(*dist_range)
        eye = target + dist * np.array([
            np.sin(polar) * np.cos(azim), np.sin(polar) * np.sin(azim), np.cos(polar)
        ])
        T_base_cam = look_at(eye, target, roll=rng.uniform(*roll_range))
        # Convert the desired CAMERA pose into a TOOL pose using the nominal X.
        T_base_tool = SE3((T_base_cam @ SE3(X_nominal.matrix).inv()).matrix,
                          "base_link", "tool0")
        r = solve_ik_multistart(arm, T_base_tool, site=arm.tool_sid,
                                q_rest=q_prev, pos_tol=2e-3, rot_tol=2e-2)
        if not r.success:
            continue
        out.append(r.q.copy())
        q_prev = r.q.copy()
    return out
