"""Time source. Swappable, because determinism is worth more than realism.

WHY NOT JUST time.time()
------------------------
A pipeline wired to the wall clock cannot be tested. "Did the servo loop
converge?" becomes a question whose answer depends on how loaded the laptop was,
and a flaky robotics test is a test that gets deleted.

So time is an injected dependency. `SimClock` advances only when the executor
steps it, which makes a whole run bit-reproducible: same seed, same inputs,
same trajectory, every time. `WallClock` is for live webcam and demo runs.

This mirrors ROS's `/use_sim_time`, and for the same reason. The cost is that
every component must take a clock instead of calling `time.time()` -- slightly
more plumbing, and a discipline that is easy to break with one careless import.
The `Rate` helper below is the main place that temptation shows up, which is
why it holds a clock rather than calling `time.sleep` directly.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod


class Clock(ABC):
    @abstractmethod
    def now(self) -> float:
        """Current time in seconds."""

    @abstractmethod
    def sleep_until(self, t: float) -> None:
        """Block (or advance) until time `t`."""


class SimClock(Clock):
    """Deterministic virtual time, advanced explicitly.

    Note `sleep_until` never goes backwards: a node that overruns its period
    must not rewind time, it must just run late -- the same semantics as a real
    scheduler missing a deadline.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        if dt < 0:
            raise ValueError("cannot advance simulated time backwards")
        self._t += dt
        return self._t

    def sleep_until(self, t: float) -> None:
        self._t = max(self._t, float(t))


class WallClock(Clock):
    """Real time. Used for webcam capture and live demos."""

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, t: float) -> None:
        dt = t - self.now()
        if dt > 0:
            time.sleep(dt)


class Rate:
    """Fixed-frequency loop helper that measures overrun instead of hiding it.

    A naive `sleep(1/hz)` drifts, because it ignores how long the body took.
    Worse, it silently absorbs overruns: a loop that should run at 30 Hz but
    takes 50 ms per iteration just quietly runs at 20 Hz and nobody notices
    until the servo gains are wrong.

    We track the next deadline absolutely (no drift) and COUNT missed deadlines,
    so "the detector cannot keep up" becomes an observable number rather than a
    vague feeling that the demo is sluggish.
    """

    def __init__(self, clock: Clock, hz: float) -> None:
        if hz <= 0:
            raise ValueError("rate must be positive")
        self.clock = clock
        self.period = 1.0 / hz
        self._next = clock.now() + self.period
        self.missed_deadlines = 0
        self.iterations = 0

    def sleep(self) -> None:
        now = self.clock.now()
        self.iterations += 1
        if now > self._next:
            self.missed_deadlines += 1
            # Re-base rather than trying to catch up with a burst of fast
            # iterations, which would make a momentarily-slow loop suddenly
            # saturate the CPU. Dropping the missed slots is the right policy
            # for sensor-driven work: a stale frame has no value.
            self._next = now + self.period
            return
        self.clock.sleep_until(self._next)
        self._next += self.period

    @property
    def overrun_fraction(self) -> float:
        return self.missed_deadlines / max(1, self.iterations)
