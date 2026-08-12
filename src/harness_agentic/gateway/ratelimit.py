"""Keeping one sender from spending the whole budget.

Authorization answers "may this person talk to the agent". It does not answer
"may they do it four hundred times an hour", and on a metered API that second
question is the one with a bill attached. A paired user whose phone gets stuck
in a retry loop is indistinguishable from an attacker, and both are handled
here.

Two buckets, deliberately: messages and turns. A message is cheap to accept and
its limit exists to stop spam; a turn is a model call with tools attached and
its limit exists to stop cost. Separating them means a burst of ``/status``
commands does not consume someone's allowance for actual work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Limit:
    """A token bucket's shape."""

    capacity: float
    """The largest burst allowed."""
    per_second: float
    """The sustained rate it refills at."""

    @classmethod
    def per_minute(cls, count: float, *, burst: float | None = None) -> Limit:
        """A limit expressed the way an operator thinks about it."""
        return cls(capacity=burst if burst is not None else count, per_second=count / 60.0)

    @classmethod
    def per_hour(cls, count: float, *, burst: float | None = None) -> Limit:
        """As above, over an hour."""
        return cls(capacity=burst if burst is not None else count, per_second=count / 3600.0)


DEFAULT_MESSAGES = Limit.per_minute(20, burst=8)
DEFAULT_TURNS = Limit.per_hour(60, burst=6)


@dataclass
class _Bucket:
    """One sender's tokens for one limit."""

    tokens: float
    updated_at: float

    def take(self, limit: Limit, now: float, amount: float) -> bool:
        """Spend ``amount`` if it is there."""
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(limit.capacity, self.tokens + elapsed * limit.per_second)
        self.updated_at = now
        if self.tokens < amount:
            return False
        self.tokens -= amount
        return True

    def retry_after(self, limit: Limit, amount: float) -> float:
        """Seconds until ``amount`` would be available."""
        if limit.per_second <= 0:
            return float("inf")
        return max(0.0, (amount - self.tokens) / limit.per_second)


@dataclass(frozen=True, slots=True)
class RateDecision:
    """Whether a request fits inside the limits."""

    allowed: bool
    retry_after_s: float = 0.0
    which: str = ""

    def __bool__(self) -> bool:
        """Truthy when allowed."""
        return self.allowed

    def message(self) -> str:
        """What to tell the sender. Says when, not just no."""
        if self.allowed:
            return ""
        if math.isinf(self.retry_after_s):
            # A bucket that never refills is a hard cap. Promising a time would
            # be a lie, and rounding infinity is a crash.
            return f"Rate limit reached ({self.which}); this is a hard cap for now."
        wait = max(1, round(self.retry_after_s))
        return f"Rate limit reached ({self.which}). Try again in about {wait}s."


@dataclass
class RateLimiter:
    """Per-sender token buckets, refilled lazily.

    Lazy refill means no background task and no per-sender timer: a bucket
    catches up the moment it is touched. Idle senders cost one dict entry, and
    :meth:`prune` drops those once they are back to full.
    """

    messages: Limit = DEFAULT_MESSAGES
    turns: Limit = DEFAULT_TURNS
    exempt: frozenset[str] = frozenset()
    """Sender keys that bypass both buckets -- operators, and the CLI."""
    _messages: dict[str, _Bucket] = field(default_factory=dict)
    _turns: dict[str, _Bucket] = field(default_factory=dict)

    def check_message(self, key: str, now: float) -> RateDecision:
        """Account for one inbound message."""
        return self._take(self._messages, self.messages, key, now, "messages")

    def check_turn(self, key: str, now: float) -> RateDecision:
        """Account for one agent turn, which is the expensive one."""
        return self._take(self._turns, self.turns, key, now, "turns")

    def refund_turn(self, key: str, now: float) -> None:
        """Give back a turn that never ran.

        A turn rejected downstream -- unauthorized, deduplicated, interrupted
        before the first request -- cost nothing, and charging for it would let
        a misconfiguration quietly exhaust a legitimate user's allowance.
        """
        bucket = self._turns.get(key)
        if bucket is not None:
            bucket.tokens = min(self.turns.capacity, bucket.tokens + 1.0)
            bucket.updated_at = now

    def _take(
        self, store: dict[str, _Bucket], limit: Limit, key: str, now: float, which: str
    ) -> RateDecision:
        if key in self.exempt:
            return RateDecision(allowed=True)
        bucket = store.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=limit.capacity, updated_at=now)
            store[key] = bucket
        if bucket.take(limit, now, 1.0):
            return RateDecision(allowed=True)
        return RateDecision(
            allowed=False, retry_after_s=bucket.retry_after(limit, 1.0), which=which
        )

    def prune(self, now: float) -> int:
        """Forget senders whose buckets have refilled. Returns how many."""
        removed = 0
        for store, limit in ((self._messages, self.messages), (self._turns, self.turns)):
            stale = [
                key
                for key, bucket in store.items()
                if bucket.tokens + (now - bucket.updated_at) * limit.per_second >= limit.capacity
            ]
            for key in stale:
                del store[key]
            removed += len(stale)
        return removed

    def tracked(self) -> int:
        """How many senders currently have state. For ``/status``."""
        return len(set(self._messages) | set(self._turns))
