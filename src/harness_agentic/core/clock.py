"""Time as an injected dependency.

Golden prompt tests and rate-limit tests both need time to be controllable.
Patching ``datetime.now`` globally breaks in surprising ways once threads are
involved, so time is a collaborator the agent is constructed with.

Timestamps are always timezone-aware. Session resume does arithmetic across
them, and naive datetimes silently do the wrong thing across a DST boundary --
``ruff``'s ``DTZ`` rules keep the naive constructors out.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """Wall-clock and monotonic time."""

    def now(self) -> datetime:
        """Current time, timezone-aware."""
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin; unaffected by clock adjustments."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds``."""
        ...


class SystemClock:
    """The real clock."""

    def now(self) -> datetime:
        """Current UTC time."""
        return datetime.now(UTC)

    def monotonic(self) -> float:
        """Monotonic seconds."""
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds``."""
        time.sleep(seconds)


class ManualClock:
    """A clock tests advance by hand.

    Backoff and debounce tests then run in microseconds and are deterministic,
    rather than sleeping for real and hoping.
    """

    def __init__(self, start: datetime | None = None) -> None:
        """Start at ``start`` (default: a fixed instant, not the real time)."""
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._mono = 0.0
        self.slept: list[float] = []

    def now(self) -> datetime:
        """Current simulated time."""
        return self._now

    def monotonic(self) -> float:
        """Simulated monotonic seconds."""
        return self._mono

    def sleep(self, seconds: float) -> None:
        """Record the request and advance instantly."""
        self.slept.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Move both clocks forward."""
        from datetime import timedelta  # noqa: PLC0415  -- keeps import cost off the hot path

        self._mono += seconds
        self._now = self._now + timedelta(seconds=seconds)
