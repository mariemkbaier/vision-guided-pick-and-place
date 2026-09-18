"""Sensor messages: images and the intrinsics that make them metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vgp.msgs.header import Header


@dataclass(frozen=True)
class Image:
    """A single frame.

    `data` is (H, W, 3) uint8. We fix ONE internal colour order -- RGB -- and
    convert at the boundary, because cv2 reads BGR while MuJoCo renders RGB and
    mixing them is a bug that looks like bad lighting rather than an error. The
    `encoding` field records it so the invariant is checkable, not assumed.

    `frame_id` on the header is normally `camera_optical`: an image is vision
    data and lives in the vision convention. See utils/frames.py.
    """

    header: Header
    data: np.ndarray
    encoding: str = "rgb8"

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]


@dataclass(frozen=True)
class CameraInfo:
    """Pinhole intrinsics + distortion. Without this, pixels are not geometry.

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

    `dist` is OpenCV's (k1, k2, p1, p2, k3) radial-tangential model.

    WHY WE CARRY THIS AS A MESSAGE rather than a global config: intrinsics
    change. Autofocus shifts fx/fy, and any resize or crop rescales fx, fy, cx,
    cy. Pinning them once at startup means a pipeline that is quietly wrong
    after someone changes the capture resolution. Bundling them with the frames
    they describe makes that impossible to get wrong.
    """

    header: Header
    K: np.ndarray                 # (3,3) float64
    dist: np.ndarray              # (5,) float64
    width: int
    height: int

    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])

    def scaled(self, sx: float, sy: float) -> "CameraInfo":
        """Intrinsics after resizing the image by (sx, sy).

        All four of fx, fy, cx, cy scale -- forgetting cx/cy is the classic
        half-done version, and it puts the principal point off-centre, which
        shows up as a pose error that grows with distance from image centre.
        Distortion coefficients are dimensionless in normalised coords and do
        NOT scale.
        """
        K = self.K.copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        return CameraInfo(
            header=self.header, K=K, dist=self.dist.copy(),
            width=int(round(self.width * sx)), height=int(round(self.height * sy)),
        )
