"""Node base class -- the unit of computation.

A node owns one job (capture frames, detect markers, plan a motion) and talks to
the rest of the system ONLY through the bus and the TF tree. That restriction is
the entire design, and it is what makes the ROS port mechanical later: a node
body references `self.publish`, `self.tf`, `self.clock` and nothing else about
the runtime.

WHAT A NODE MUST NOT DO
-----------------------
  - call another node's methods directly (use a topic)
  - call time.time() (use self.clock -- see runtime/clock.py)
  - hand-compose transforms it could look up (use self.tf)
  - keep mutable shared state visible to other nodes

Each of those is a rule with a victim: the first breaks the ROS port, the second
breaks reproducible tests, the third re-introduces the frame bugs TFTree exists
to kill, the fourth makes behaviour depend on subscriber ordering.

LIFECYCLE
---------
`on_configure` (declare topics/subscriptions) then `on_start` (first-tick work),
mirroring ROS 2 managed nodes. Split in two because declarations must ALL happen
before any node starts publishing -- otherwise a fast producer can publish to a
topic a slower consumer has not subscribed to yet, and the message is simply
lost. That race is miserable to debug; a two-phase startup makes it impossible.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Callable

from vgp.runtime.bus import Bus
from vgp.runtime.clock import Clock
from vgp.runtime.tf import TFTree


class Timer:
    """A periodic callback.

    WHY THE DEADLINE IS COMPUTED, NOT ACCUMULATED
    ---------------------------------------------
    The obvious implementation is `self.next_due += self.period` after each
    firing. It is wrong, and wrong in a way that hides: 0.01 is not exactly
    representable in binary floating point, so adding it 100 times gives
    1.0000000000000007, not 1.0. A 100 Hz timer asked to run for exactly one
    second then fires 99 times, because its 100th deadline landed a hair past
    the end.

    Worse, the error is not deterministic across rates -- accumulating 0.1 ten
    times gives 0.9999999999999999, which errs the *other* way. So a 10 Hz
    timer passes the same test that a 100 Hz timer fails. That is how this
    survives casual testing and then shows up as an off-by-one in a frame count
    much later, where nobody suspects the clock.

    Keeping an integer tick index and computing `start + n * period` bounds the
    error at one rounding instead of letting it accumulate with n. Real-time
    schedulers do the same thing.
    """

    def __init__(self, name: str, period: float, cb: Callable[[], None], start: float) -> None:
        self.name = name
        self.period = period
        self.cb = cb
        self.start = start
        self.n = 1                 # tick index; deadline is start + n * period
        self.calls = 0
        self.overruns = 0

    @property
    def next_due(self) -> float:
        return self.start + self.n * self.period

    def advance(self, ticks: int = 1) -> None:
        self.n += ticks


class Node(ABC):
    def __init__(self, name: str, bus: Bus, tf: TFTree, clock: Clock) -> None:
        self.name = name
        self.bus = bus
        self.tf = tf
        self.clock = clock
        self.timers: list[Timer] = []
        self._log: list[tuple[float, str, str]] = []

    # ------------------------------------------------------------ lifecycle

    def on_configure(self) -> None:
        """Declare topics, subscriptions and timers. No publishing here."""

    def on_start(self) -> None:
        """Called once after every node is configured. Publishing allowed."""

    def on_shutdown(self) -> None:
        """Release hardware (camera handles, sim contexts)."""

    # -------------------------------------------------------------- helpers

    def declare(self, topic: str, msg_type: type, latched: bool = False) -> None:
        self.bus.declare(topic, msg_type, latched)

    def publish(self, topic: str, msg: Any) -> None:
        self.bus.publish(topic, msg)

    def subscribe(self, topic: str, msg_type: type, cb: Callable[[Any], None]) -> None:
        # Declaring on subscribe too means startup order does not matter: either
        # side can be the first to mention a topic, and the type check still
        # catches a producer/consumer type disagreement.
        self.bus.declare(topic, msg_type)
        self.bus.subscribe(topic, cb, owner=self.name)

    def create_timer(self, hz: float, cb: Callable[[], None], name: str = "timer") -> Timer:
        t = Timer(f"{self.name}/{name}", 1.0 / hz, cb, self.clock.now())
        self.timers.append(t)
        return t

    def now(self) -> float:
        return self.clock.now()

    def log(self, msg: str, level: str = "info") -> None:
        """Timestamped log kept in-process.

        In-memory rather than printed so tests can assert on what a node
        reported, and so a run's log can be dumped alongside its eval metrics
        in Step 7. Printing is the executor's choice, not the node's.
        """
        self._log.append((self.clock.now(), level, msg))

    @property
    def log_lines(self) -> list[str]:
        return [f"[{t:8.3f}] {self.name:<16} {lvl:<5} {m}" for t, lvl, m in self._log]
