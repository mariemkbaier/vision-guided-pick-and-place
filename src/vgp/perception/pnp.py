"""6D pose from 2D-3D correspondences, with the honesty signals attached.

WHAT PnP IS
-----------
Given N points whose 3D positions are known in an object frame, and where they
landed in the image, plus the camera intrinsics, recover the rigid transform
between object and camera. That is the step that turns pixels into metres and
is where the pipeline stops being computer vision and starts being robotics.

THE TWO THINGS THIS MODULE ADDS BEYOND CALLING cv2.solvePnP
------------------------------------------------------------

1. REPROJECTION ERROR AS THE CONFIDENCE SIGNAL.
   A detector's score answers "is this a marker?". It says nothing about
   whether the recovered geometry is sane. Reprojection error answers the
   question we actually care about -- does this pose explain the pixels we saw
   -- in units we can threshold and reason about.

   This matters because the failure modes that hurt a robot are geometric, not
   recognition failures: a marker detected through motion blur decodes fine and
   yields a tilted pose. Its reprojection error goes up. Its detection score
   does not.

2. PLANAR AMBIGUITY DETECTION.
   A planar target viewed near-frontally has TWO poses that project to almost
   the same image -- mirror images through the marker plane. This is not
   numerical noise; it is a genuine geometric two-fold ambiguity, and it is
   worst exactly when the marker is small, far, or square-on, which is the
   normal case on a tabletop.

   Picking the wrong one flips the approach axis. The gripper dives at the
   object from a tilted direction and collides with the table. The classic
   symptom is a pose that "flickers" between two orientations frame to frame.

   We detect it by asking for BOTH solutions and comparing their residuals. If
   the runner-up explains the image nearly as well, we do not silently take the
   winner -- we flag it, and the task layer can request another view instead of
   guessing. Reporting an honest "I am not sure" beats a confident coin flip
   when the consequence is a collision.

WHY IPPE_SQUARE FOR MARKERS
---------------------------
For four coplanar points forming a square, IPPE_SQUARE is a closed-form
solution that returns BOTH ambiguous poses with their residuals -- exactly what
we need for (2). The default ITERATIVE flag needs an initial guess, converges
to ONE local minimum, and will happily return the wrong branch with no
indication that another existed. That silence is the problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vgp.msgs import CameraInfo, Detection2D, Header, ObjectPose
from vgp.utils.transforms import SE3

# If the runner-up solution's reprojection error is within this factor of the
# best, the pose is ambiguous. 3.0 is deliberately conservative: false alarms
# cost one extra camera view, a missed ambiguity costs a collision. Measured
# behaviour vs. viewing angle is in the Step 3 tests.
AMBIGUITY_RATIO = 3.0


class PnPFailure(RuntimeError):
    """PnP did not converge, or the correspondences were degenerate."""


@dataclass(frozen=True)
class PnPResult:
    pose: SE3                       # T_camera_object
    rmse_px: float
    ambiguous: bool
    alternative: SE3 | None         # the runner-up, when the pose is ambiguous
    error_ratio: float              # runner-up RMSE / best RMSE; 1.0 = fully ambiguous


def _rmse(model_pts, img_pts, rvec, tvec, K, dist) -> float:
    proj, _ = cv2.projectPoints(model_pts, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - img_pts.reshape(-1, 2)
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def solve_pnp(
    model_points: np.ndarray,
    image_points: np.ndarray,
    info: CameraInfo,
    square_planar: bool = True,
) -> PnPResult:
    """Recover T_camera_object from 3D-2D correspondences.

    `square_planar` selects IPPE_SQUARE, valid only for exactly four coplanar
    points in the marker corner order. For anything else we use SQPNP, which is
    a global solver (no initial guess, no local minimum to fall into) and is
    the right default for general point sets.
    """
    model = np.ascontiguousarray(model_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if len(model) != len(img):
        raise PnPFailure(f"{len(model)} model points vs {len(img)} image points")
    if len(model) < 4:
        raise PnPFailure(f"need >= 4 correspondences, got {len(model)}")

    flag = (cv2.SOLVEPNP_IPPE_SQUARE
            if (square_planar and len(model) == 4)
            else cv2.SOLVEPNP_SQPNP)

    # solvePnPGeneric returns ALL solutions the method found, which is the
    # whole reason we call it instead of solvePnP.
    n, rvecs, tvecs, _reproj = cv2.solvePnPGeneric(
        model, img, info.K, info.dist, flags=flag
    )
    if n < 1:
        raise PnPFailure("PnP found no solution (degenerate correspondences?)")

    # Rank by our own reprojection RMSE rather than trusting the ordering.
    # OpenCV's returned `reproj` is not RMSE in pixels across versions, and we
    # want one number whose definition we control and can threshold.
    scored = sorted(
        ((_rmse(model, img, rvecs[i], tvecs[i], info.K, info.dist), i) for i in range(n)),
        key=lambda t: t[0],
    )
    best_rmse, bi = scored[0]
    best = SE3.from_rotvec(rvecs[bi].ravel(), tvecs[bi].ravel())

    if best.t[2] <= 0:
        # The object is behind the camera. Physically impossible for something
        # we just saw, so this means the correspondences or the model points
        # are wrong -- almost always a corner-ordering mistake.
        raise PnPFailure(
            f"PnP put the object behind the camera (z={best.t[2]:.3f}m). "
            "Usual cause: model points ordered differently from image points."
        )

    alt, ratio, ambiguous = None, float("inf"), False
    if len(scored) > 1:
        second_rmse, si = scored[1]
        # Guard the degenerate case where the best fit is essentially exact:
        # a ratio against ~0 is meaningless, so floor the denominator at a
        # fraction of a pixel, below which corner noise dominates anyway.
        ratio = second_rmse / max(best_rmse, 0.05)
        ambiguous = ratio < AMBIGUITY_RATIO
        if ambiguous:
            alt = SE3.from_rotvec(rvecs[si].ravel(), tvecs[si].ravel())

    return PnPResult(best, best_rmse, ambiguous, alt, ratio)


def pose_from_detection(
    det: Detection2D,
    info: CameraInfo,
    stamp: float,
    object_id: str,
    frame_id: str,
) -> ObjectPose:
    """Convenience wrapper producing a fully-labelled ObjectPose message."""
    if det.corners is None or det.model_points is None:
        raise PnPFailure(
            f"detection '{det.label}' has no corner/model correspondences. "
            "A bounding box alone cannot yield orientation -- a square seen "
            "rotated has the same axis-aligned bbox. Use plane.py for objects "
            "without known geometry."
        )
    r = solve_pnp(det.model_points, det.corners, info)
    return ObjectPose(
        header=Header(stamp, frame_id),
        object_id=object_id,
        label=det.label,
        pose=SE3(r.pose.matrix, parent=frame_id, child=f"obj/{object_id}"),
        reprojection_rmse_px=r.rmse_px,
        pose_ambiguous=r.ambiguous,
        detection_score=det.score,
    )


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def depth_sensitivity(size_m: float, distance_m: float, fx: float,
                      corner_noise_px: float = 0.3) -> tuple[float, float]:
    """First-order (lateral, depth) position uncertainty in metres.

    WHY THIS FUNCTION EXISTS RATHER THAN A SINGLE "ACCURACY" NUMBER
    --------------------------------------------------------------
    Monocular pose error is deeply ANISOTROPIC and people quote it as if it
    were not. Sideways, a pixel of corner noise maps to

        dx ~ Z / fx  per pixel

    Depth is recovered from the marker's apparent SIZE, so a pixel of noise on
    an apparent size of (s * fx / Z) pixels gives

        dZ ~ Z^2 / (s * fx)  per pixel

    The ratio dZ/dx = Z/s. A 4 cm marker at 60 cm is FIFTEEN TIMES less certain
    in depth than sideways -- and it degrades with the SQUARE of distance while
    lateral error is only linear.

    Consequences we actually design around:
      - a grasp should approach ALONG the camera axis where possible, so the
        badly-known direction is absorbed by the approach rather than by a
        lateral miss;
      - bigger markers beat closer cameras (dZ ~ 1/s is cheaper to fix than
        moving the camera);
      - a single monocular view is a poor way to measure depth, which is why
        Step 6 servos rather than doing one-shot open-loop reaching.
    """
    dx = distance_m / fx * corner_noise_px
    dz = distance_m ** 2 / (size_m * fx) * corner_noise_px
    return dx, dz
