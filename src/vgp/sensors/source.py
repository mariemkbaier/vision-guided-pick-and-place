"""CameraSource: one interface, so perception never knows where frames come from.

WHY THIS ABSTRACTION EARNS ITS KEEP
-----------------------------------
It is the difference between "we tested in sim" and "the same code ran in sim
and on hardware". Three concrete payoffs:

1. THE SAME DETECTOR CODE RUNS ON BOTH. If the sim path went through a
   different function than the webcam path, every sim result would carry an
   asterisk -- you would never know whether a real-world failure was the
   environment or the untested code path.

2. GROUND TRUTH BECOMES AVAILABLE WITHOUT CHANGING THE CONSUMER. The sim source
   knows the exact object pose and the exact camera pose. It exposes them on a
   SEPARATE channel (`ground_truth()`), never through `read()`, so no perception
   code can accidentally consume them. That separation is what makes the Step 7
   error numbers trustworthy.

3. RECORDED DATA IS JUST A THIRD IMPLEMENTATION. A replay source over saved
   frames needs no changes anywhere else, which is how a flaky real-world
   failure becomes a regression test.

WHAT THE INTERFACE DELIBERATELY REFUSES TO DO
---------------------------------------------
`read()` returns an Image AND a CameraInfo, every frame, together. It would be
tidier to fetch intrinsics once at startup. That would also be a bug generator:
autofocus shifts fx/fy, and any resize rescales all four of fx, fy, cx, cy. Data
and the calibration that interprets it travel together or they drift apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from vgp.msgs import CameraInfo, Image
from vgp.utils.transforms import SE3


@dataclass(frozen=True)
class GroundTruth:
    """Privileged state only a simulator can know.

    Kept OUT of `read()` on purpose. If ground truth flowed through the normal
    message path, a perception node could read it -- by accident or by a
    well-meaning shortcut -- and every accuracy number afterwards would be a
    measurement of nothing. Physically separating the channel makes that
    mistake impossible rather than merely discouraged.
    """

    stamp: float
    camera_pose: SE3                      # T_world_cameraOptical
    object_poses: dict[str, SE3]          # object_id -> T_world_object


class CameraSource(ABC):
    @abstractmethod
    def read(self) -> tuple[Image, CameraInfo]:
        """Grab one frame with the intrinsics that describe it."""

    @abstractmethod
    def close(self) -> None: ...

    def ground_truth(self) -> GroundTruth | None:
        """Privileged state, or None for a real camera. Never used by perception."""
        return None

    # Context-manager so a camera handle is released even when a demo crashes.
    # A leaked cv2.VideoCapture on macOS keeps the green light on and blocks the
    # next run -- a small thing that wastes real time during a demo.
    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def intrinsics_from_fov(
    width: int, height: int, fovy_deg: float, stamp: float = 0.0,
    frame_id: str = "camera_optical",
) -> CameraInfo:
    """Build an exact pinhole K from a renderer's vertical field of view.

    THE DERIVATION (worth being able to reproduce on a whiteboard)
    -------------------------------------------------------------
    A pinhole camera with vertical FOV `fovy` sees, at depth Z, a vertical
    extent of  2 * Z * tan(fovy/2)  world units mapped onto `height` pixels.
    The focal length in pixels is the constant that converts one to the other:

        fy = (height / 2) / tan(fovy / 2)

    fx == fy because the renderer uses square pixels and expresses its frustum
    through the vertical FOV, letting the horizontal extent follow from the
    aspect ratio. (A real camera has fx != fy only if its sensor pixels are
    non-square or the image was anisotropically scaled.)

    THE HALF-PIXEL THAT PEOPLE GET WRONG
    ------------------------------------
    The principal point is NOT (width/2, height/2). In OpenCV's convention a
    pixel's coordinate is its CENTRE, so pixel 0 spans [-0.5, +0.5] and an image
    of W pixels spans [-0.5, W-0.5]. The centre of that span is

        cx = (width - 1) / 2

    Using width/2 puts the principal point half a pixel off. That sounds
    ignorable and mostly is -- until you propagate it through PnP at range,
    where a half-pixel is a systematic bias that shows up as a small constant
    offset in every object pose and gets misattributed to hand-eye calibration.
    We verify this empirically against actual rendered pixels rather than
    trusting the reasoning (tests/test_sim_camera.py).
    """
    fy = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg))
    fx = fy
    K = np.array([
        [fx, 0.0, (width - 1) / 2.0],
        [0.0, fy, (height - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ])
    from vgp.msgs import Header
    # A rendered image is a perfect pinhole: zero distortion, exactly. This is
    # NOT true of the webcam, and the difference is a large part of why sim
    # numbers flatter you.
    return CameraInfo(Header(stamp, frame_id), K, np.zeros(5), width, height)
