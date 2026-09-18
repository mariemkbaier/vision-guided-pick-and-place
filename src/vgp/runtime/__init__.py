"""ROS-shaped runtime: nodes, a typed bus, a transform tree, a swappable clock.

Deliberately small and self-contained. The design invariant to defend: porting
to rclpy should mean rewriting THIS package and leaving node bodies and message
definitions alone. If a port would force changes in nodes/, runtime detail has
leaked and that is a bug.
"""

from vgp.runtime.bus import Bus, TopicTypeError
from vgp.runtime.clock import Clock, SimClock, WallClock, Rate
from vgp.runtime.node import Node
from vgp.runtime.tf import TFTree, TFLookupError
from vgp.runtime.executor import Executor

__all__ = [
    "Bus", "TopicTypeError", "Clock", "SimClock", "WallClock", "Rate",
    "Node", "TFTree", "TFLookupError", "Executor",
]
