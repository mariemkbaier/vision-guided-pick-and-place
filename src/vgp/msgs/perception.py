"""Perception messages: 2D detections and 6D object poses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vgp.msgs.header import Header
from vgp.utils.transforms import SE3


@dataclass(frozen=True)
class Detection2D:
    """A 2D detection in image space.

    `corners` is optional and is the bridge to 6D pose: for a planar marker or
    a known-geometry object, four or more image points with known 3D model
    correspondences are exactly what solvePnP consumes. A bbox alone cannot
    give orientation -- a square seen rotated has the same axis-aligned bbox.
    That is precisely why this stays a separate field rather than us pretending
    a bbox is enough.
    """

    label: str
    score: float
    bbox_xyxy: np.ndarray                      # (4,) pixels
    corners: np.ndarray | None = None          # (N, 2) pixels, ordered
    model_points: np.ndarray | None = None     # (N, 3) metres, object frame

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox_xyxy
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])


@dataclass(frozen=True)
class Detection2DArray:
    header: Header
    detections: list[Detection2D] = field(default_factory=list)


@dataclass(frozen=True)
class ObjectPose:
    """A 6D object pose with the uncertainty needed to decide whether to act.

    `pose` is labelled parent=<observing frame>, child=obj/<id>.

    `reprojection_rmse_px` is our confidence signal, and it is deliberately NOT
    a neural-net softmax score. A detector can be 0.99 confident about a box
    that yields a geometrically impossible pose. Reprojection error measures the
    thing we actually care about -- does this pose explain the pixels we saw --
    in units (pixels) we can reason about and threshold. It is the single most
    useful number for deciding "do I trust this enough to move a robot".

    `pose_ambiguous` flags the planar-marker two-solution problem: a small,
    near-frontal square marker has two PnP solutions with almost equal
    reprojection error, mirrored about the marker plane. Acting on the wrong one
    tilts the approach axis badly. Reporting it lets the task layer request a
    second view instead of guessing.
    """

    header: Header
    object_id: str
    label: str
    pose: SE3
    reprojection_rmse_px: float
    pose_ambiguous: bool = False
    detection_score: float = 1.0


@dataclass(frozen=True)
class ObjectPoseArray:
    header: Header
    objects: list[ObjectPose] = field(default_factory=list)
