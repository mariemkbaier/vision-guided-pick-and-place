"""Perception nodes: detection and pose estimation on the bus.

WHY TWO NODES AND NOT ONE
-------------------------
Detection and pose estimation have genuinely different characters, and
splitting them buys three concrete things:

  - They run at different rates. Detection is the expensive part; pose
    estimation on already-found corners is microseconds. Splitting lets us
    throttle one without the other.
  - Detection is swappable (ArUco, colour, a learned model) while PnP is not.
    The seam belongs exactly where the interchangeable part ends.
  - /perception/detections is independently useful -- the visualiser and the
    recorder subscribe to it without caring whether a pose was solvable.

WHERE POSES GO
--------------
Into the TF tree, stamped with the IMAGE's timestamp, not "now". That is the
whole reason Step 1's buffer interpolates: a consumer asking "where is the cube
relative to the robot base" must resolve the arm's state at the instant the
photo was taken, which on a moving arm is centimetres away from its state now.
"""

from __future__ import annotations

import numpy as np

from vgp.msgs import (
    CameraInfo, Detection2DArray, Image, ObjectPose, ObjectPoseArray,
)
from vgp.perception.aruco import ArucoDetector
from vgp.perception.pnp import PnPFailure, pose_from_detection
from vgp.runtime.node import Node
from vgp.sensors.camera_node import TOPIC_IMAGE, TOPIC_INFO

TOPIC_DETECTIONS = "/perception/detections"
TOPIC_POSES = "/perception/object_poses"


class DetectorNode(Node):
    """Runs a detector on incoming frames.

    Event-driven (a subscription callback), not timer-driven: there is nothing
    to do without a frame, and polling would either duplicate work on the same
    image or miss frames depending on phase. `max_hz` throttles by DROPPING
    frames rather than queueing them -- for perception a stale frame has no
    value, so dropping is strictly better than falling further behind.
    """

    def __init__(self, name, bus, tf, clock, detector=None, max_hz: float | None = None):
        super().__init__(name, bus, tf, clock)
        self.detector = detector or ArucoDetector()
        self.min_period = None if max_hz is None else 1.0 / max_hz
        self._last_run = -np.inf
        self.frames_seen = 0
        self.frames_processed = 0
        self.detections_published = 0

    def on_configure(self) -> None:
        self.declare(TOPIC_DETECTIONS, Detection2DArray)
        self.subscribe(TOPIC_IMAGE, Image, self._on_image)

    def _on_image(self, img: Image) -> None:
        self.frames_seen += 1
        if self.min_period is not None and img.header.stamp - self._last_run < self.min_period:
            return
        self._last_run = img.header.stamp
        self.frames_processed += 1
        dets = self.detector.detect(img)
        self.detections_published += len(dets.detections)
        self.publish(TOPIC_DETECTIONS, dets)

    @property
    def drop_fraction(self) -> float:
        return 1.0 - self.frames_processed / max(1, self.frames_seen)


class PoseEstimatorNode(Node):
    """Turns detections with known geometry into 6D poses, and into TF.

    `max_rmse_px` is the gate that keeps a bad pose from reaching the robot.
    It is deliberately a GEOMETRIC threshold: a detection can be recognised
    with total confidence and still yield a pose that does not explain the
    pixels (motion blur, a partly-occluded marker, a wrong marker size).

    Ambiguous poses are published but FLAGGED, not dropped. The task layer is
    the right place to decide -- it can move the camera and look again, which
    perception cannot.
    """

    def __init__(self, name, bus, tf, clock, max_rmse_px: float = 2.0,
                 publish_tf: bool = True):
        super().__init__(name, bus, tf, clock)
        self.max_rmse_px = max_rmse_px
        self.publish_tf = publish_tf
        self._info: CameraInfo | None = None
        self.poses_published = 0
        self.rejected_rmse = 0
        self.rejected_no_geometry = 0
        self.ambiguous_count = 0

    def on_configure(self) -> None:
        self.declare(TOPIC_POSES, ObjectPoseArray)
        # Latched: intrinsics are published once and we may subscribe later.
        self.subscribe(TOPIC_INFO, CameraInfo, self._on_info)
        self.subscribe(TOPIC_DETECTIONS, Detection2DArray, self._on_detections)

    def _on_info(self, info: CameraInfo) -> None:
        self._info = info

    def _on_detections(self, dets: Detection2DArray) -> None:
        if self._info is None:
            # Not an error: the camera simply has not published yet. Log it so
            # a permanently-missing CameraInfo is visible rather than looking
            # like "perception found nothing".
            self.log("detections arrived before CameraInfo; skipping", "warn")
            return

        out: list[ObjectPose] = []
        for det in dets.detections:
            if det.model_points is None or det.corners is None:
                self.rejected_no_geometry += 1
                continue
            try:
                p = pose_from_detection(
                    det, self._info, dets.header.stamp,
                    object_id=det.label,
                    frame_id=dets.header.frame_id,
                )
            except PnPFailure as e:
                self.log(f"PnP failed for {det.label}: {e}", "warn")
                continue

            if p.reprojection_rmse_px > self.max_rmse_px:
                self.rejected_rmse += 1
                self.log(
                    f"rejected {det.label}: reprojection {p.reprojection_rmse_px:.2f}px "
                    f"> {self.max_rmse_px}px", "warn")
                continue
            if p.pose_ambiguous:
                self.ambiguous_count += 1
            out.append(p)
            if self.publish_tf:
                # Stamped with the IMAGE time, not now -- see module docstring.
                self.tf.set_transform(p.pose, stamp=p.header.stamp)

        self.poses_published += len(out)
        self.publish(TOPIC_POSES, ObjectPoseArray(dets.header, out))
