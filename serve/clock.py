"""Clocks. The engine reads time only through `clock.now()`, so the same engine code can run on
real time (real model) or on simulated time (SimRunner advances a VirtualClock).
"""

import time


class RealClock:
    """Wall-clock time in seconds since the clock was created."""

    def __init__(self) -> None:
        """Start counting from zero now."""
        self._start = time.perf_counter()

    def now(self) -> float:
        """Seconds since start."""
        return time.perf_counter() - self._start

    def advance_to(self, t: float) -> None:
        """Idle until time t (when nothing is running and the next request is in the future)."""
        delay = t - self.now()
        if delay > 0:
            time.sleep(delay)


class VirtualClock:
    """Simulated time: only moves when someone advances it."""

    def __init__(self) -> None:
        """Start at time zero."""
        self._now = 0.0

    def now(self) -> float:
        """Current simulated time in seconds."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Move time forward by `seconds` (e.g. the predicted duration of a model step)."""
        if seconds < 0:
            raise ValueError("time cannot go backwards")
        self._now += seconds

    def advance_to(self, t: float) -> None:
        """Jump forward to time t (never backwards)."""
        self._now = max(self._now, t)
