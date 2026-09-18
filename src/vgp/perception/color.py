"""Colour-and-contour detection for objects with no fiducial marker.

WHY BOTHER, GIVEN MARKERS WORK
------------------------------
Because in any real cell most objects are not tagged, and because it makes the
pipeline's assumptions explicit. This detector cannot produce a 6D pose on its
own -- and being forced to say WHY is the point:

A bounding box has four numbers. A 6D pose has six, and orientation is not
recoverable from an axis-aligned box at all: a square seen rotated has the same
bbox as one seen straight. So a detector like this needs an extra constraint
from somewhere. We get it from the table plane (see plane.py). Markers carry
their own constraint -- known 3D geometry -- which is exactly why they are
easier and why we built them first.

WHY HSV AND NOT RGB
-------------------
In RGB, "red" is a diagonal region that moves as lighting changes -- brightness
is smeared across all three channels, so a threshold tuned indoors fails by a
window. HSV separates hue (what colour) from value (how bright), so a hue band
plus loose saturation/value bounds survives illumination change far better.

Its own failure mode, stated plainly: hue is meaningless at low saturation.
Greys, whites and blacks have essentially random hue, so this cannot detect an
uncoloured object at all, and a washed-out highlight on a red cube reads as
"not red". That is a hard limit of the method, not a tuning problem -- the
honest fix is a learned detector, which is the natural upgrade path here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vgp.msgs import Detection2D, Detection2DArray, Header, Image


@dataclass(frozen=True)
class ColorSpec:
    """An HSV band. OpenCV hue is 0-179 (half of 0-359, to fit in a uint8)."""

    label: str
    hue_lo: int
    hue_hi: int
    sat_min: int = 90
    val_min: int = 60
    min_area_px: int = 400

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        # Red straddles the 0/180 wrap, so a band may need two ranges. Handling
        # the wrap explicitly avoids the classic bug where red is half-detected.
        lo, hi = np.array([self.hue_lo, self.sat_min, self.val_min]), \
                 np.array([self.hue_hi, 255, 255])
        if self.hue_lo <= self.hue_hi:
            return cv2.inRange(hsv, lo, hi)
        a = cv2.inRange(hsv, np.array([0, self.sat_min, self.val_min]),
                        np.array([self.hue_hi, 255, 255]))
        b = cv2.inRange(hsv, np.array([self.hue_lo, self.sat_min, self.val_min]),
                        np.array([179, 255, 255]))
        return cv2.bitwise_or(a, b)


# Sensible starting bands. Tuned per-cell in practice -- which is itself an
# argument for a learned detector in anything beyond a demo.
RED = ColorSpec("red", hue_lo=170, hue_hi=10)
GREEN = ColorSpec("green", hue_lo=45, hue_hi=85)
BLUE = ColorSpec("blue", hue_lo=100, hue_hi=130)


class ColorDetector:
    def __init__(self, specs: list[ColorSpec] | None = None, open_kernel: int = 5) -> None:
        self.specs = specs or [RED, GREEN, BLUE]
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))

    def detect(self, image: Image) -> Detection2DArray:
        hsv = cv2.cvtColor(image.data, cv2.COLOR_RGB2HSV)
        out: list[Detection2D] = []
        for spec in self.specs:
            m = spec.mask(hsv)
            # Opening then closing: remove salt noise, then fill small holes
            # (specular highlights punch holes in a coloured face and would
            # otherwise split one object into several contours).
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._kernel)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < spec.min_area_px:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                # Min-area rect gives an ORIENTED box. Its angle is the only
                # orientation cue colour offers, and it is ambiguous modulo 90
                # degrees for a square object -- resolved (or not) downstream.
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype(np.float64)
                out.append(Detection2D(
                    label=spec.label,
                    # Solidity (contour area / its convex hull area) as a crude
                    # score: a clean convex object scores ~1, a fragmented or
                    # partly-occluded blob scores lower. Not a probability, and
                    # deliberately not treated as one.
                    score=float(area / max(cv2.contourArea(cv2.convexHull(c)), 1.0)),
                    bbox_xyxy=np.array([x, y, x + w, y + h], dtype=np.float64),
                    corners=box,
                    # No model_points: we do NOT know this object's 3D geometry.
                    # pose_from_detection() will refuse it, by design -- the
                    # plane constraint has to be supplied explicitly.
                    model_points=None,
                ))
        return Detection2DArray(Header(image.header.stamp, image.header.frame_id), out)
