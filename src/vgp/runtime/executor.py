"""Single-threaded deterministic executor.

THE DECISION: ONE THREAD, NOT MANY
----------------------------------
Real ROS 2 runs callbacks across a thread pool, and for good reason -- a slow
detector should not stall the 500 Hz control loop. We deliberately do not.

What we gain: a run is *bit-reproducible*. Same seed and same inputs give the
same trajectory, every time. That is what makes the Step 7 eval harness
meaningful (a grasp-success number that wobbles run to run measures nothing),
and it is what makes a failure reproducible instead of a story about something
that happened once.

What we give up: no real parallelism, so wall-clock throughput is bounded by the
slowest callback, and we cannot demonstrate a genuine multi-rate system under
load. For a sim-plus-webcam pipeline on one laptop that is the right trade --
determinism is the scarce resource here, CPU is not.

What we must be careful about: this executor lets a slow node silently dominate
the loop. So timers COUNT their overruns (see `report()`), which keeps the
multi-rate story honest: we can still say "the detector missed 12% of its 15 Hz
deadlines" even though nothing ran concurrently.

Exit path: swapping in a thread-pool executor means changing this file only,
provided nodes obeyed the no-shared-mutable-state rule in node.py.

SCHEDULING
----------
Earliest-deadline-first over the registered timers. With a SimClock we jump time
straight to the next deadline -- no idle spinning, so a 60-second simulated
run finishes in whatever real time the work takes. With a WallClock we sleep.
Ties break by registration order, deterministically.
"""

from __future__ import annotations

from vgp.runtime.bus import Bus
from vgp.runtime.clock import Clock, SimClock
from vgp.runtime.node import Node, Timer
from vgp.runtime.tf import TFTree

# One nanosecond: far below any real scheduling granularity, far above the
# ~1e-16 rounding we are guarding against.
_EPS = 1e-9


class Executor:
    def __init__(self, clock: Clock, bus: Bus | None = None, tf: TFTree | None = None) -> None:
        self.clock = clock
        self.bus = bus or Bus()
        self.tf = tf or TFTree()
        self.nodes: list[Node] = []
        self._started = False

    def add(self, node: Node) -> Node:
        if self._started:
            raise RuntimeError("cannot add nodes after start(); declarations must precede publishes")
        self.nodes.append(node)
        return node

    def start(self) -> None:
        """Two-phase startup: configure everything, then start everything."""
        for n in self.nodes:
            n.on_configure()
        for n in self.nodes:
            n.on_start()
        self._started = True

    def _timers(self) -> list[Timer]:
        return [t for n in self.nodes for t in n.timers]

    def spin_once(self) -> bool:
        """Fire the single earliest-due timer. Returns False if none remain."""
        timers = self._timers()
        if not timers:
            return False
        # min() is stable, so equal deadlines resolve in registration order --
        # the detail that makes runs reproducible rather than merely repeatable.
        due = min(timers, key=lambda t: t.next_due)
        now = self.clock.now()
        if due.next_due > now:
            self.clock.sleep_until(due.next_due)
        else:
            # Already late. Count it, then run immediately.
            due.overruns += 1
        due.calls += 1
        due.cb()
        # Advance the tick INDEX, never the deadline itself -- see Timer's
        # docstring for the float-accumulation bug this avoids.
        due.advance()
        # If the callback overran by more than a whole period, drop the missed
        # slots rather than replaying them in a burst: stale sensor slots have
        # no value, and a catch-up burst would starve every other timer.
        t_now = self.clock.now()
        if due.next_due < t_now:
            missed = int((t_now - due.next_due) // due.period) + 1
            due.overruns += missed
            due.advance(missed)
        return True

    def spin(self, duration: float | None = None, max_iterations: int | None = None) -> None:
        """Run for `duration` seconds of clock time, or `max_iterations` ticks."""
        if not self._started:
            self.start()
        end = None if duration is None else self.clock.now() + duration
        i = 0
        try:
            while True:
                if max_iterations is not None and i >= max_iterations:
                    break
                if end is not None:
                    timers = self._timers()
                    # EPS guards the boundary case: a deadline landing exactly
                    # on `end` should fire (spin(1.0) at 10 Hz means 10 ticks,
                    # the last one AT t=1.0). Without the tolerance, one ulp of
                    # rounding decides whether the final tick happens.
                    if not timers or min(t.next_due for t in timers) > end + _EPS:
                        if isinstance(self.clock, SimClock):
                            self.clock.sleep_until(end)   # land exactly on the boundary
                        break
                if not self.spin_once():
                    break
                i += 1
        finally:
            pass

    def shutdown(self) -> None:
        # Reverse order so consumers stop before the producers they depend on.
        for n in reversed(self.nodes):
            n.on_shutdown()

    def report(self) -> str:
        """Per-timer call counts and deadline overruns -- our rqt_top."""
        lines = ["timer".ljust(34) + "hz".rjust(7) + "calls".rjust(8) + "overruns".rjust(10)]
        for t in self._timers():
            lines.append(
                f"{t.name:<34}{1.0 / t.period:>7.1f}{t.calls:>8}{t.overruns:>10}"
            )
        return "\n".join(lines)

    def logs(self) -> str:
        merged = [(t, line) for n in self.nodes for (t, _, _), line in zip(n._log, n.log_lines)]
        return "\n".join(line for _, line in sorted(merged, key=lambda p: p[0]))
