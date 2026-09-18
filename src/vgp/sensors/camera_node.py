"""Node that pumps a CameraSource onto the bus.

Thin by design: it owns the timer and the topics, and knows nothing about
whether frames come from MuJoCo, a webcam, or a recording. That is the payoff
of the CameraSource interface -- switching sim to real is one constructor
argument at wiring time, with no node changes.

Two topics, different durability:
  /camera/image  volatile   -- a stale frame is worthless, never replay it
  /camera/info   LATCHED    -- published once, must reach late subscribers

That asymmetry is the whole reason the bus has a `latched` flag. Without it, a
detector node constructed a moment after the camera waits forever for
intrinsics that already went by, and the symptom is a hang with no error.
"""

from __future__ import annotations

from vgp.msgs import CameraInfo, Image
from vgp.runtime.node import Node
from vgp.sensors.source import CameraSource

TOPIC_IMAGE = "/camera/image"
TOPIC_INFO = "/camera/info"


class CameraNode(Node):
    def __init__(self, name, bus, tf, clock, source: CameraSource, hz: float = 30.0) -> None:
        super().__init__(name, bus, tf, clock)
        self.source = source
        self.hz = hz
        self.frames_published = 0

    def on_configure(self) -> None:
        self.declare(TOPIC_IMAGE, Image)
        self.declare(TOPIC_INFO, CameraInfo, latched=True)
        self.create_timer(self.hz, self._capture, name="capture")

    def _capture(self) -> None:
        img, info = self.source.read()
        # Info first: a subscriber processing the image in its callback must
        # already have the intrinsics, or it has to buffer the frame and wait.
        self.publish(TOPIC_INFO, info)
        self.publish(TOPIC_IMAGE, img)
        self.frames_published += 1

    def on_shutdown(self) -> None:
        self.source.close()
