"""Message header -- timestamp, frame, sequence number.

WHY EVERY MESSAGE CARRIES ONE
-----------------------------
Two fields, both of which exist to prevent a specific class of silent failure.

`frame_id`: geometric data without its frame is meaningless. A pose is not "at
(0.4, 0.1, 0.2)" -- it is at that point *in some frame*, and the whole pipeline
is the business of moving between frames. Attaching the frame to the data means
the TFTree can validate a lookup instead of trusting the caller.

`stamp`: the detector runs slower than the camera. By the time a pose reaches
the controller it describes where the object was 80 ms ago. For a static object
that is fine; for a moving one, acting on a stale pose is how you crush things.
Carrying the stamp lets the consumer decide -- see `age_s` and the staleness
guard in the task layer. A pipeline that drops timestamps cannot even detect
the problem.

`seq` exists to spot dropped frames, which is the difference between "the
detector is slow" and "the camera is dropping buffers" -- very different fixes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

_seq_counter = itertools.count()


@dataclass(frozen=True)
class Header:
    stamp: float                 # seconds, from the runtime Clock (sim or wall)
    frame_id: str                # the frame this message's data is expressed in
    seq: int = field(default_factory=lambda: next(_seq_counter))

    def age_s(self, now: float) -> float:
        """How stale this message is. Consumers gate on this; see task/."""
        return now - self.stamp
