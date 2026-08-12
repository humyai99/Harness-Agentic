"""Time, for the asyncio half of the process.

The core has :class:`~harness_agentic.core.clock.Clock` and it is synchronous,
which is correct there and useless here: the delivery layer's whole job is
debouncing, and a test that has to wait 1.5 real seconds per edit is a test
nobody runs. So the gateway takes its sleeping as a dependency too.

:class:`ManualAsyncClock` advances only when a test says so, which makes
"does the debounce coalesce three chunks into one edit" a deterministic
assertion rather than a timing race.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from datetime import UTC, datetime, timedelta
from typing import Protocol


class AsyncClock(Protocol):
    """What the gateway needs from time."""

    def now(self) -> datetime:
        """The current instant, timezone-aware."""
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin, never going backwards."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Yield for ``seconds``."""
        ...


class RealAsyncClock:
    """Wall time and real sleeping."""

    def now(self) -> datetime:
        """The current UTC instant."""
        return datetime.now(UTC)

    def monotonic(self) -> float:
        """The event loop's monotonic clock."""
        return asyncio.get_running_loop().time()

    async def sleep(self, seconds: float) -> None:
        """Sleep on the event loop."""
        await asyncio.sleep(seconds)


class ManualAsyncClock:
    """A clock that only moves when a test moves it.

    Sleepers park on a heap keyed by wake time. :meth:`advance` releases
    everything due and yields to the loop between wakes, so a coroutine woken
    early can queue another sleep and still be released within the same call --
    without that, a debounce loop would need one ``advance`` per iteration and
    the test would encode the implementation's step count.
    """

    def __init__(self, start: datetime | None = None) -> None:
        """Start stopped, at ``start`` or the Unix epoch."""
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._t = 0.0
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        self._counter = itertools.count()

    def now(self) -> datetime:
        """The simulated instant."""
        return self._now

    def monotonic(self) -> float:
        """Simulated seconds since the clock was created."""
        return self._t

    async def sleep(self, seconds: float) -> None:
        """Park until the clock is advanced past ``seconds`` from now."""
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (self._t + seconds, next(self._counter), future))
        await future

    async def advance(self, seconds: float) -> None:
        """Move time forward, releasing sleepers as their deadlines pass."""
        target = self._t + seconds
        while self._waiters and self._waiters[0][0] <= target:
            when, _, future = heapq.heappop(self._waiters)
            self._step_to(when)
            if not future.done():
                future.set_result(None)
            # Let the woken coroutine run before deciding what is due next: it
            # may schedule another sleep that also falls before ``target``.
            await asyncio.sleep(0)
        self._step_to(target)
        await asyncio.sleep(0)

    def _step_to(self, when: float) -> None:
        """Move both clocks to a monotonic instant, never backwards."""
        delta = when - self._t
        if delta <= 0:
            return
        self._t = when
        self._now += timedelta(seconds=delta)

    @property
    def sleepers(self) -> int:
        """How many coroutines are parked. Useful for asserting on quiescence."""
        return len(self._waiters)
