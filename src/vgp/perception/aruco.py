"""ArUco marker detection.

WHY FIDUCIAL MARKERS ARE THE RIGHT FIRST PERCEPTION PATH
--------------------------------------------------------
They are not the impressive answer. A learned 6D pose estimator is. They are
the right answer *first*, for a reason that matters more than impressiveness:

A pick-and-place pipeline has five stages that can each be wrong, and an error
at the gripper is indistinguishable between them. Markers give a perception
stage whose accuracy is known, bounded and independently verifiable -- so when
the arm misses, you can rule perception OUT and go look at hand-eye
calibration. Start with a learned estimator and every failure is a
four-way argument.

They also give the *rest* of the pipeline something to be tested against. The
colour/plane detector in plane.py is graded against marker poses, and the
hand-eye calibration in Step 4 is driven by them.

THE CONVENTION (the part worth memorising)
------------------------------------------
`detectMarkers` returns corners in image order TL, TR, BR, BL -- clockwise
when y points down. The matching 3D model points put the marker frame at the
marker CENTRE with:

    X right, Y up, Z out of the marker plane toward the camera

so the corners are (-s/2, +s/2), (+s/2, +s/2), (+s/2, -s/2), (-s/2, -s/2),
all at Z = 0. Note Y is UP in the marker frame while v is DOWN in the image;
that sign flip is exactly why the order looks "reversed" and is the thing
people get wrong when they hand-roll the object points.

A QUIET ZONE IS NOT OPTIONAL. A marker printed to the edge of its border is
not detected at all -- verified: the same image detects as None without
padding and as id 7 with it. Every marker we generate includes one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vgp.msgs import Detection2D, Detection2DArray, Header, Image

# 4x4_50: the smallest dictionary that covers our needs. Smaller bit grids are
# detectable at greater distance and lower resolution (fewer pixels per bit);
# larger dictionaries buy more unique ids at the cost of inter-marker Hamming
# distance, i.e. more false ids under noise. 50 ids is far more than a tabletop
# needs, so we take the robustness.
DEFAULT_DICT = cv2.aruco.DICT_4X4_50


def marker_model_points(size_m: float) -> np.ndarray:
    """3D corners of a marker in ITS OWN frame, ordered to match detectMarkers.

    Centre-origin (not corner-origin) so the reported pose is the marker's
    centre -- which is what you want to grasp toward, and what makes a marker's
    pose independent of its size.
    """
    h = size_m / 2.0
    return np.array([
        [-h, +h, 0.0],   # TL
        [+h, +h, 0.0],   # TR
        [+h, -h, 0.0],   # BR
        [-h, -h, 0.0],   # BL
    ], dtype=np.float64)


@dataclass(frozen=True)
class MarkerSpec:
    """What a marker id means physically.

    `size_m` is per-id, not global: a scene legitimately mixes a large marker
    on the table (for hand-eye) with small ones on objects. Assuming one global
    size scales every pose of the odd-sized markers -- silently, since the
    reprojection error stays perfect.
    """

    size_m: float
    label: str = "marker"


class ArucoDetector:
    def __init__(
        self,
        specs: dict[int, MarkerSpec] | None = None,
        default_size_m: float = 0.04,
        dictionary: int = DEFAULT_DICT,
        refine: bool = True,
    ) -> None:
        self.specs = specs or {}
        self.default_size_m = default_size_m
        self._dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        if refine:
            # Sub-pixel corner refinement. This is the single highest-leverage
            # flag here: corner accuracy enters PnP directly, and depth error
            # scales with it. Costs a few ms; buys roughly an order of
            # magnitude in pose precision. Quantified in the Step 3 tests.
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
        self._det = cv2.aruco.ArucoDetector(self._dict, params)

    def detect(self, image: Image) -> Detection2DArray:
        # cv2 wants BGR or grey; our internal convention is RGB (msgs/sensor.py).
        # Greyscale is both correct and faster -- marker decoding is a binary
        # threshold operation, colour carries no information here.
        gray = cv2.cvtColor(image.data, cv2.COLOR_RGB2GRAY)
        corners, ids, _rejected = self._det.detectMarkers(gray)

        out: list[Detection2D] = []
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                spec = self.specs.get(int(i), MarkerSpec(self.default_size_m, f"tag{int(i)}"))
                pts = c.reshape(4, 2).astype(np.float64)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                out.append(Detection2D(
                    # Underscore, not colon: this label becomes a TF frame segment
                    # ("obj/cube_7"), and a colon there reads as a namespace
                    # separator in most tf tooling. Keeping labels frame-safe at
                    # the source beats sanitising them at every consumer.
                    label=f"{spec.label}_{int(i)}",
                    # A marker either decodes or it does not -- there is no
                    # meaningful graded confidence, so 1.0 is honest. The real
                    # confidence signal arrives downstream as reprojection
                    # error, which is about GEOMETRY rather than recognition.
                    score=1.0,
                    bbox_xyxy=np.array([x0, y0, x1, y1]),
                    corners=pts,
                    model_points=marker_model_points(spec.size_m),
                ))
        return Detection2DArray(Header(image.header.stamp, image.header.frame_id), out)


def generate_marker(marker_id: int, pixels: int = 300, dictionary: int = DEFAULT_DICT,
                    quiet_zone_frac: float = 0.25) -> np.ndarray:
    """Render a marker WITH its mandatory quiet zone. Returns greyscale uint8."""
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    img = cv2.aruco.generateImageMarker(d, marker_id, pixels)
    pad = int(pixels * quiet_zone_frac)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
