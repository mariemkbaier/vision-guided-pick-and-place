"""Lifting 2D detections to 3D using a known support plane.

THE IDEA
--------
A single camera cannot recover depth from one point: every pixel corresponds to
a RAY, not a position. But if you know the object is sitting on a known plane,
the ray meets that plane at exactly one point, and depth is determined.

This is how a huge amount of practical bin-picking and tabletop manipulation
actually works, and it is worth being able to derive:

    A pixel (u,v) back-projects to the ray  d = K^-1 [u v 1]^T  through the
    optical centre. A plane is {p : n . p = c}. Substituting p = t*d:

        t = c / (n . d)          and the 3D point is  t * d

WHERE IT BREAKS, WHICH IS THE PART WORTH KNOWING
------------------------------------------------
1. GRAZING RAYS. As n.d approaches zero the ray becomes parallel to the plane
   and t explodes. A pixel of noise then moves the 3D point by metres. The
   conditioning is entirely controlled by the angle between the ray and the
   plane, so a camera mounted low and looking across a table is far worse than
   one looking down at it -- and no amount of better detection fixes it.
   `incidence_deg` is returned so callers can refuse grazing geometry.

2. THE CENTROID IS NOT ON THE PLANE. This is the subtle one. For a box of
   height h standing on the table, the centroid of its SILHOUETTE is somewhere
   up the object's side, not on the table. Intersect that ray with the table
   and you get a point pulled toward the camera, by an amount proportional to
   h and to how obliquely you are looking. It is a systematic bias, not noise,
   so averaging over frames does not remove it -- and it is the right size to
   make a gripper clip the near edge of an object.

   We handle it by intersecting with a plane offset by the object's half-height,
   which recovers the CENTRE of an object of known height. That requires
   knowing h, which we state as an explicit argument rather than hide.

3. THE PLANE ITSELF IS ESTIMATED. Every error in the table's pose propagates
   directly into object position. In our pipeline the plane comes from the
   robot's own kinematics and the hand-eye calibration, so this is a place
   where a Step 4 error shows up disguised as a Step 3 error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vgp.msgs import CameraInfo, Detection2D, Header, ObjectPose
from vgp.utils.transforms import SE3


class PlaneLiftError(ValueError):
    """Ray does not meet the plane usefully (parallel, behind, or grazing)."""


@dataclass(frozen=True)
class Plane:
    """{p : normal . p = offset}, expressed in the CAMERA optical frame.

    Stored in camera coordinates rather than robot coordinates because that is
    where the ray lives, and converting the plane once per frame is cheaper and
    less error-prone than converting every ray.
    """

    normal: np.ndarray
    offset: float

    @staticmethod
    def from_pose(T_cam_plane: SE3) -> "Plane":
        """Plane through a frame's origin, with its +Z as the normal."""
        n = T_cam_plane.R[:, 2]
        return Plane(n / np.linalg.norm(n), float(n @ T_cam_plane.t))

    def offset_by(self, distance: float) -> "Plane":
        """Parallel plane `distance` ALONG the normal. Raw geometry.

        Rarely what you want directly -- see `offset_toward_camera`, because
        which way `normal` points is an accident of how the plane was built.
        """
        return Plane(self.normal, self.offset + distance)

    def offset_toward_camera(self, distance: float) -> "Plane":
        """Parallel plane `distance` closer to the camera, whichever way the
        normal happens to point.

        WHY THIS EXISTS AS A SEPARATE METHOD
        ------------------------------------
        A plane normal has two valid directions and nothing in the maths picks
        one. `Plane.from_pose` takes the frame's +Z, so a table frame with Z up
        gives a normal pointing at the camera, while a normal derived from a
        fitted point cloud may point either way depending on the fit.

        `offset_by(+h/2)` therefore moves the plane toward the camera in one
        case and away in the other -- silently doubling the half-height bias
        instead of removing it. Caught by a test asserting the lifted plane is
        NEARER the camera; it was 25 mm further.

        Objects rest between the support surface and an overhead camera, so
        "toward the camera" is the direction we always actually mean. Encoding
        the intent rather than the sign makes the call site correct regardless
        of where the plane came from.
        """
        # `offset` is the signed distance from the camera origin to the plane
        # along `normal`, so its sign tells us which way the plane lies.
        return Plane(self.normal, self.offset - np.sign(self.offset) * distance)


def backproject_ray(uv: np.ndarray, info: CameraInfo) -> np.ndarray:
    """Unit ray through pixel `uv` in the camera optical frame.

    Undistorts first. Skipping that is a quiet error that grows toward the image
    edges -- exactly where objects at the workspace boundary appear.
    """
    import cv2

    p = np.asarray(uv, dtype=np.float64).reshape(1, 1, 2)
    # P=None -> returns NORMALISED coordinates (already through K^-1).
    n = cv2.undistortPoints(p, info.K, info.dist).reshape(2)
    d = np.array([n[0], n[1], 1.0])
    return d / np.linalg.norm(d)


def intersect(uv: np.ndarray, info: CameraInfo, plane: Plane,
              min_incidence_deg: float = 10.0) -> tuple[np.ndarray, float]:
    """3D point where the pixel's ray meets the plane, plus incidence angle."""
    d = backproject_ray(uv, info)
    denom = float(plane.normal @ d)
    # Incidence measured from the plane SURFACE (90 deg = looking straight down).
    incidence = np.degrees(np.arcsin(min(1.0, abs(denom))))
    if incidence < min_incidence_deg:
        raise PlaneLiftError(
            f"ray meets the plane at {incidence:.1f} deg (min {min_incidence_deg:.0f}). "
            "Near-grazing geometry: a pixel of noise moves the 3D point by a large "
            "distance. Raise the camera or tilt it down."
        )
    t = plane.offset / denom
    if t <= 0:
        raise PlaneLiftError("plane intersection is behind the camera")
    return t * d, incidence


def pose_on_plane(
    det: Detection2D,
    info: CameraInfo,
    plane: Plane,
    object_height_m: float,
    stamp: float,
    object_id: str,
    frame_id: str,
    min_incidence_deg: float = 10.0,
) -> ObjectPose:
    """6D pose of an object resting on `plane`, from a 2D detection.

    Only 4 of the 6 DoF are actually measured: position (3) and yaw about the
    plane normal (1). Roll and pitch are ASSUMED zero -- that is the "resting
    flat on the table" assumption, and it is an assumption, not a measurement.
    We report a large `reprojection_rmse_px` sentinel rather than a small one,
    because there is no reprojection residual to speak of here and pretending
    otherwise would let this pose masquerade as a well-constrained marker pose
    in any downstream threshold.
    """
    # Lift the plane by half the object's height so we recover the object's
    # CENTRE, not a biased point on the table. See note 2 in the module
    # docstring, and offset_toward_camera for why the sign is not obvious.
    lifted = plane.offset_toward_camera(object_height_m / 2.0)
    p_cam, incidence = intersect(det.center, info, lifted, min_incidence_deg)

    # Yaw from the min-area-rect's long axis, lifted onto the plane. The rect's
    # angle is ambiguous modulo 90 degrees for a near-square object; we do not
    # pretend otherwise -- a symmetric object simply has no unique yaw, and for
    # a parallel-jaw gripper that is fine (two grasps are equivalent).
    if det.corners is not None and len(det.corners) >= 4:
        edges = det.corners[[1, 2, 3, 0]] - det.corners
        long_edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
        p_edge, _ = intersect(det.center + long_edge, info, lifted, min_incidence_deg)
        x_axis = p_edge - p_cam
    else:
        x_axis = np.array([1.0, 0.0, 0.0])

    z_axis = -plane.normal if plane.normal @ p_cam > 0 else plane.normal
    z_axis = z_axis / np.linalg.norm(z_axis)
    # Gram-Schmidt: force the measured in-plane direction perpendicular to the
    # plane normal, so the result is a proper rotation rather than an almost-one.
    x_axis = x_axis - (x_axis @ z_axis) * z_axis
    nx = np.linalg.norm(x_axis)
    x_axis = x_axis / nx if nx > 1e-9 else np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])

    return ObjectPose(
        header=Header(stamp, frame_id),
        object_id=object_id,
        label=det.label,
        pose=SE3.from_rt(R, p_cam, parent=frame_id, child=f"obj/{object_id}"),
        # Sentinel, not a measurement: this path has no reprojection residual.
        # Encoding the incidence angle keeps the conditioning visible downstream.
        reprojection_rmse_px=float(90.0 / max(incidence, 1.0)),
        pose_ambiguous=False,
        detection_score=det.score,
    )
