"""Typed messages passed between nodes.

All frozen dataclasses. Immutability is the point: a message published to two
subscribers must not be mutable by one of them, or you get action-at-a-distance
bugs that only appear under a particular subscriber ordering. Frozen also means
we can publish without defensive copying.
"""

from vgp.msgs.header import Header
from vgp.msgs.sensor import Image, CameraInfo
from vgp.msgs.perception import (
    Detection2D, Detection2DArray, ObjectPose, ObjectPoseArray,
)
from vgp.msgs.control import (
    JointState, JointTrajectoryPoint, JointTrajectory, GripperCommand,
)

__all__ = [
    "Header", "Image", "CameraInfo",
    "Detection2D", "Detection2DArray", "ObjectPose", "ObjectPoseArray",
    "JointState", "JointTrajectoryPoint", "JointTrajectory", "GripperCommand",
]
