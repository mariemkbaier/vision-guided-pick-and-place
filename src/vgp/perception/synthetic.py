"""Render ArUco markers at exactly known poses, to test PnP against truth.

Same trick as calib/synthetic.py: a marker is planar, so the marker plane maps
to the image by a homography and one warpPerspective renders the view exactly.

The value is the same too -- we CHOOSE the pose, so we can report "translation
error 0.4 mm, rotation error 0.2 deg" rather than "the reprojection error was
small". Those are different claims, and only the first is about correctness.
"""

from __future__ import annotations

import cv2
import numpy as np

from vgp.perception.aruco import DEFAULT_DICT, generate_marker
from vgp.utils.transforms import SE3

_MARKER_PX = 240
_QUIET = 0.25


def marker_texture(marker_id: int, size_m: float, dictionary: int = DEFAULT_DICT):
    """Marker image plus its board-pixel -> marker-metre map.

    THE Y FLIP IS THE TRAP. The marker frame has Y UP (see aruco.py), while
    image row v increases DOWNWARD. So the mapping carries a negative on the Y
    row. Omit it and the rendered marker is vertically mirrored -- which still
    DECODES (ArUco is not chiral enough to reject it), still gives a low
    reprojection error, and yields a pose rotated 180 degrees about X. Another
    self-consistent wrong answer.
    """
    img = generate_marker(marker_id, _MARKER_PX, dictionary, _QUIET)
    n = img.shape[0]
    cu = cv = (n - 1) / 2.0
    s = size_m / _MARKER_PX              # metres per texture pixel
    M = np.array([
        [s, 0.0, -cu * s],
        [0.0, -s, cv * s],               # negative: marker Y up vs image v down
        [0.0, 0.0, 1.0],
    ])
    return img, M


def render_marker_view(
    marker_id: int,
    size_m: float,
    K: np.ndarray,
    T_cam_marker: SE3,
    width: int,
    height: int,
    dist: np.ndarray | None = None,
    background: int = 160,
    blur_sigma: float = 0.7,
    noise_std: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render one marker seen from `T_cam_marker`. Returns an RGB image."""
    rng = rng or np.random.default_rng(0)
    tex, M = marker_texture(marker_id, size_m)
    R, t = T_cam_marker.R, T_cam_marker.t
    # Third column of R is annihilated by Z = 0 on the marker plane.
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ M

    # Composite explicitly: warp the texture and a coverage mask, then paste.
    # The background must NOT be white -- detection needs the marker's white
    # quiet zone to contrast with its surroundings, and an infinite white plane
    # gives it nothing to contrast against.
    view = np.full((height, width), background, np.uint8)
    warped = cv2.warpPerspective(tex, H, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(tex.shape, 255, np.uint8), H, (width, height),
                               flags=cv2.INTER_NEAREST)
    view = np.where(mask > 0, warped, view).astype(np.uint8)

    if dist is not None and np.any(dist):
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        grid = np.stack([xs, ys], -1).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(grid, K, dist, P=K).reshape(height, width, 2)
        view = cv2.remap(view, ideal[..., 0].copy(), ideal[..., 1].copy(),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if blur_sigma > 0:
        view = cv2.GaussianBlur(view, (0, 0), blur_sigma)
    if noise_std > 0:
        view = np.clip(view.astype(np.float32) + rng.normal(0, noise_std, view.shape), 0, 255)
        view = view.astype(np.uint8)
    return cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)


# A marker squarely FACING the camera is rotated 180 degrees about X, not
# identity. The ArUco marker frame has +Z pointing OUT of the marker toward the
# viewer; the camera's +Z points INTO the scene. So the two Z axes are
# antiparallel when the marker faces you, and marker +Y (up) maps to camera -Y
# (because camera +Y is DOWN).
#
# Verified by projecting the model points: with identity rotation the corner
# labelled TL lands at the LARGEST image y -- the marker is facing away, renders
# mirrored, and then does not decode at all. That is how this was found.
T_FACING = SE3.from_rt(np.diag([1.0, -1.0, -1.0]), np.zeros(3))


def marker_pose(distance: float, tilt_deg: float, azim_deg: float = 0.0,
                offset=(0.0, 0.0)) -> SE3:
    """T_camera_marker at a given stand-off and tilt away from frontal.

    tilt_deg = 0 is face-on, which is precisely the pathological case for the
    planar two-fold ambiguity.
    """
    tilt = np.deg2rad(tilt_deg)
    az = np.deg2rad(azim_deg)
    # Tilt about an axis in the image plane, then apply the facing flip. Order
    # matters: tilt is expressed in CAMERA axes, so it pre-multiplies.
    tilt_R = SE3.from_rotvec(tilt * np.array([np.cos(az), np.sin(az), 0.0]), np.zeros(3))
    R = (tilt_R @ T_FACING).R
    return SE3.from_rt(R, [offset[0], offset[1], distance],
                       parent="camera_optical", child="marker")
