"""Typed publish/subscribe message bus.

WHY A BUS AT ALL, when direct function calls would work
-------------------------------------------------------
Because the coupling a bus removes is exactly the coupling that makes these
pipelines rigid. With direct calls, the detector must know the pose estimator
exists, which must know the planner exists. Adding a recorder, or a second
consumer of detections for visualisation, means editing the producer.

Concretely, it buys us three things we use later:
  - the recorder node subscribes to everything and nobody knows it is there;
  - swapping the webcam source for the sim source changes one line of wiring;
  - a node can be tested by publishing a message and asserting on what appears,
    with no other node constructed.

The cost is indirection: you cannot jump-to-definition from producer to
consumer. We pay that back with `Bus.graph()`, which prints the actual wiring.

TYPE CHECKING ON PUBLISH
------------------------
Topics declare their message type and publishing the wrong type raises
immediately. Without it, a type error surfaces later inside a subscriber as an
AttributeError, at which point the traceback points at the innocent consumer
rather than the guilty producer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


class TopicTypeError(TypeError):
    """Published payload does not match the topic's declared type."""


@dataclass
class _Topic:
    name: str
    msg_type: type
    latched: bool          # replay the last message to late subscribers
    last: Any = None
    publish_count: int = 0


class Bus:
    def __init__(self) -> None:
        self._topics: dict[str, _Topic] = {}
        self._subs: dict[str, list[tuple[str, Callable[[Any], None]]]] = defaultdict(list)

    # ------------------------------------------------------------ declare

    def declare(self, name: str, msg_type: type, latched: bool = False) -> None:
        """Register a topic. Idempotent, but the type must not change.

        `latched` mirrors ROS's transient-local durability. It exists for
        low-rate, slow-changing data -- CameraInfo above all. Intrinsics are
        published once at startup; without latching, any node that starts a
        moment later waits forever for a message that already came and went.
        That failure looks like a hang with no error, which is the worst kind.
        """
        existing = self._topics.get(name)
        if existing is not None:
            if existing.msg_type is not msg_type:
                raise TopicTypeError(
                    f"topic '{name}' already declared as {existing.msg_type.__name__}, "
                    f"cannot redeclare as {msg_type.__name__}"
                )
            return
        self._topics[name] = _Topic(name, msg_type, latched)

    # ------------------------------------------------------------ pub/sub

    def publish(self, name: str, msg: Any) -> None:
        topic = self._topics.get(name)
        if topic is None:
            raise KeyError(f"publish to undeclared topic '{name}'. Declare it in on_configure().")
        if not isinstance(msg, topic.msg_type):
            raise TopicTypeError(
                f"topic '{name}' expects {topic.msg_type.__name__}, "
                f"got {type(msg).__name__}"
            )
        topic.publish_count += 1
        if topic.latched:
            topic.last = msg
        # Synchronous fan-out in subscription order. Deterministic on purpose:
        # see runtime/executor.py for why we chose single-threaded.
        for _owner, cb in self._subs[name]:
            cb(msg)

    def subscribe(self, name: str, cb: Callable[[Any], None], owner: str = "?") -> None:
        topic = self._topics.get(name)
        if topic is None:
            raise KeyError(f"subscribe to undeclared topic '{name}'")
        self._subs[name].append((owner, cb))
        # Late-subscriber replay for latched topics -- the whole reason latching
        # exists. Delivered immediately so the subscriber is configured before
        # its first real callback.
        if topic.latched and topic.last is not None:
            cb(topic.last)

    # ----------------------------------------------------- introspection

    def graph(self) -> str:
        """Render the real wiring. Our substitute for `rqt_graph`.

        Worth having: when the demo silently does nothing, the first question is
        always "is anyone actually subscribed to that topic", and this answers
        it in one line instead of a debugger session.
        """
        lines = ["topic".ljust(28) + "type".ljust(20) + "pubs  subscribers"]
        for name in sorted(self._topics):
            t = self._topics[name]
            subs = ", ".join(o for o, _ in self._subs[name]) or "(none)"
            flag = " [latched]" if t.latched else ""
            lines.append(
                f"{name:<28}{t.msg_type.__name__:<20}{t.publish_count:<6}{subs}{flag}"
            )
        return "\n".join(lines)

    def publish_count(self, name: str) -> int:
        return self._topics[name].publish_count
