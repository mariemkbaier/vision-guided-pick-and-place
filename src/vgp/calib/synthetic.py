"""Synthetic checkerboard rendering, for testing calibration against known truth.

WHY SYNTHESISE INSTEAD OF USING REAL PHOTOS
-------------------------------------------
To test a calibrator you must know the right answer. With real photos you never
do -- you can only check that the residual is small, which (see intrinsics.py
note 3) is exactly the thing that can be small while the answer is wrong.

Here we CHOOSE K, render views through it, and then ask the calibrator to
recover it. That converts "the RMSE looks fine" into "fx was recovered to
within 0.3%", which is a real claim.

It also lets us demonstrate the fronto-parallel degeneracy on purpose: render
only face-on views, watch RMSE stay beautiful while fx comes back badly wrong,
and confirm the tilt gate catches it. That experiment is worth more than the
paragraph explaining it.

HOW IT WORKS
------------
A checkerboard is planar, so the board plane maps to the image by a HOMOGRAPHY
-- no ray tracing needed. For board points with Z = 0:

    [u v 1]^T  ~  K [r1 r2 t] [X Y 1]^T

where r1, r2 are the first two columns of R (the third is multiplied by Z = 0
and drops out). Compose that with the board-pixels -> board-metres scaling and
one `cv2.warpPerspective` renders the view exactly.
"""

from __future__ import annotations

import cv2
import numpy as np

from vgp.calib.intrinsics import CheckerboardSpec
from vgp.utils.transforms import SE3

_PPS = 40          # rendered pixels per board square
_MARGIN_SQ = 1     # white quiet zone; findChessboardCorners needs it to find the outer corners


def board_image(spec: CheckerboardSpec) -> tuple[np.ndarray, np.ndarray]:
    """Render the board face-on, and return the board-pixel -> board-metre map.

    A `cols x rows` INNER-corner board has (cols+1) x (rows+1) squares, with
    inner corner (i, j) at metric (i*s, j*s). Square (0,0) therefore spans
    x in [-s, 0] -- the origin sits on a square boundary, not a square centre.
    Off-by-one here silently shifts the whole board frame.
    """
    nx, ny = spec.cols + 1, spec.rows + 1
    m = _MARGIN_SQ
    W, H = (nx + 2 * m) * _PPS, (ny + 2 * m) * _PPS
    img = np.full((H, W), 255, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                y0, x0 = (j + m) * _PPS, (i + m) * _PPS
                img[y0 : y0 + _PPS, x0 : x0 + _PPS] = 0

    # board pixel (u,v) -> board metres. Pixel (m*_PPS, m*_PPS) is the top-left
    # of square (0,0), which is metric (-s, -s).
    s_per_px = spec.square_size_m / _PPS
    M = np.array([
        [s_per_px, 0.0, -spec.square_size_m - m * _PPS * s_per_px],
        [0.0, s_per_px, -spec.square_size_m - m * _PPS * s_per_px],
        [0.0, 0.0, 1.0],
    ])
    return img, M


def render_view(
    spec: CheckerboardSpec,
    K: np.ndarray,
    T_cam_board: SE3,
    width: int,
    height: int,
    dist: np.ndarray | None = None,
    blur_sigma: float = 0.8,
    noise_std: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render the board seen from `T_cam_board`. Returns an RGB image.

    `blur_sigma` and `noise_std` are not decoration. A perfectly sharp, noiseless
    checkerboard lets sub-pixel refinement hit accuracy no real camera achieves,
    so a calibrator validated only on clean renders has thresholds calibrated to
    a fantasy. A little blur and sensor noise keeps the numbers honest.
    """
    rng = rng or np.random.default_rng(0)
    board, M = board_image(spec)

    R, t = T_cam_board.R, T_cam_board.t
    # [r1 r2 t]: the third column of R is annihilated by Z = 0 on the board plane.
    P = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H = P @ M

    view = cv2.warpPerspective(
        board, H, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    if dist is not None and np.any(dist):
        # APPLYING distortion is the opposite direction from REMOVING it, and
        # OpenCV only ships the removal direction. Getting this backwards does
        # not crash -- it renders a plausibly barrel-warped image with the
        # inverse coefficients, and the calibrator then dutifully recovers
        # k1 = +0.15 when we asked for -0.15. Sign-flipped, not noisy.
        #
        # What we need: for each pixel of the OUTPUT (distorted) image, where
        # did it come from in the IDEAL pinhole render? That is exactly
        # undistortPoints (distorted pixel -> ideal pixel) evaluated on the
        # output grid.
        #
        # `initUndistortRectifyMap` answers the reverse question -- for each
        # IDEAL pixel, where to sample in a DISTORTED source -- which is what
        # you want when rectifying a real photo, and wrong here.
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        grid = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(grid, K, dist, P=K).reshape(height, width, 2)
        view = cv2.remap(
            view, ideal[..., 0].copy(), ideal[..., 1].copy(), cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )
    if blur_sigma > 0:
        view = cv2.GaussianBlur(view, (0, 0), blur_sigma)
    if noise_std > 0:
        view = np.clip(view.astype(np.float32) + rng.normal(0, noise_std, view.shape), 0, 255)
        view = view.astype(np.uint8)
    return cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)


def pose_grid(
    spec: CheckerboardSpec,
    n: int,
    K: np.ndarray | None = None,
    width: int | None = None,
    height: int | None = None,
    distance: float = 0.38,
    distance_jitter: float = 0.35,
    max_tilt_deg: float = 35.0,
    lateral: float = 0.30,
    seed: int = 0,
    margin_px: float = 6.0,
) -> list[SE3]:
    """Sample `n` board poses in front of the camera, framed so the board fits.

    WHY REJECTION SAMPLING ON VISIBILITY
    ------------------------------------
    There is a real tension in collecting calibration views. The coverage and
    edge gates want the board pushed out to the image corners; pushing it out
    makes part of it leave the frame, and a partly-visible checkerboard is not
    detected AT ALL (findChessboardCorners is all-or-nothing). So naive lateral
    jitter silently throws away ~45% of views -- measured -- and the ones it
    keeps are the central, low-coverage ones. You get the worst of both.

    Worse, the loss is invisible: you asked for 20 views, got 11, and the
    calibration is unstable for a reason that never appears in its output.

    So we sample the full lateral range and KEEP only poses whose entire board
    projects inside the image. That pushes views as far out as geometry allows
    while guaranteeing they are usable -- which is also exactly what a human
    does when calibrating well: slide the board toward the edge until it is
    about to leave frame, then stop.

    Without `K`/`width`/`height` we cannot check visibility, so we fall back to
    a conservative lateral range rather than silently producing clipped views.
    """
    rng = np.random.default_rng(seed)
    cx = spec.cols * spec.square_size_m / 2
    cy = spec.rows * spec.square_size_m / 2
    check = K is not None and width is not None and height is not None
    if not check:
        lateral = min(lateral, 0.05)     # conservative: see docstring

    # Include the quiet-zone margin: findChessboardCorners needs the white
    # border visible too, not just the inner corners.
    s_ = spec.square_size_m
    hull = np.array([
        [-s_, -s_, 0.0], [spec.cols * s_ + s_, -s_, 0.0],
        [-s_, spec.rows * s_ + s_, 0.0], [spec.cols * s_ + s_, spec.rows * s_ + s_, 0.0],
    ])

    out: list[SE3] = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        tilt = np.deg2rad(rng.uniform(0, max_tilt_deg))
        azim = rng.uniform(0, 2 * np.pi)
        # Tilt about an axis lying in the image plane, direction set by `azim`,
        # so orientations spread in every direction rather than about one axis.
        rotvec = tilt * np.array([np.cos(azim), np.sin(azim), 0.0])
        spin = rng.uniform(-0.25, 0.25)
        R_cb = (SE3.from_rotvec(rotvec, np.zeros(3)) @ SE3.from_euler("z", [spin])).R
        # Vary depth widely: a board held CLOSE fills the frame and is what
        # actually produces edge corners, which is what constrains the
        # distortion coefficients. A fixed stand-off gives neat views and a
        # distortion model fitted to the image centre only.
        z = distance * rng.uniform(1.0 - distance_jitter, 1.0 + distance_jitter)
        t = R_cb @ np.array([-cx, -cy, 0.0]) + np.array(
            [rng.uniform(-lateral, lateral), rng.uniform(-lateral, lateral), z]
        )
        pose = SE3.from_rt(R_cb, t, parent="camera_optical", child="board")

        if check:
            p_cam = pose.apply(hull)
            if np.any(p_cam[:, 2] <= 1e-3):
                continue
            uv = p_cam @ np.asarray(K).T
            uv = uv[:, :2] / uv[:, 2:3]
            if (uv[:, 0].min() < margin_px or uv[:, 0].max() > width - 1 - margin_px
                    or uv[:, 1].min() < margin_px or uv[:, 1].max() > height - 1 - margin_px):
                continue
        out.append(pose)
    return out
