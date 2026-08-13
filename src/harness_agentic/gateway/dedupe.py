"""Not answering the same message twice.

Webhook platforms redeliver. LINE and Slack both retry when the endpoint does
not return 2xx quickly, and Slack additionally retries on its own timeout even
when the handler eventually succeeded. The failure this produces is not a
duplicate line of output -- it is the agent running a *second turn*, with tools,
against the same request, which on a deploy command means deploying twice.

So every inbound event is checked against recently-seen ids before anything
irreversible happens. The window is bounded in both time and size: unbounded
memoization of message ids is a slow memory leak on a process meant to run for
months.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

DEFAULT_TTL_S = 900.0
"""Fifteen minutes. Longer than any platform's retry schedule."""
DEFAULT_CAPACITY = 20_000


@dataclass
class Deduplicator:
    """Remembers which message ids have already been accepted.

    Insertion-ordered, so eviction is oldest-first and does not need a scan.
    Ids are namespaced by platform because nothing guarantees two platforms
    will not both hand out ``"1"``.
    """

    ttl_s: float = DEFAULT_TTL_S
    capacity: int = DEFAULT_CAPACITY
    _seen: OrderedDict[str, float] = field(default_factory=OrderedDict)
    duplicates: int = 0

    def seen(self, platform: str, message_id: str, now: float) -> bool:
        """Whether this message was already accepted; records it if not.

        An event with no id is always treated as new. Polling transports
        deduplicate by offset upstream and genuinely have nothing to key on,
        and refusing those would drop real messages to guard against a
        redelivery that cannot happen.
        """
        if not message_id:
            return False
        key = f"{platform}:{message_id}"
        self._expire(now)
        if key in self._seen:
            self.duplicates += 1
            return True
        self._seen[key] = now
        while len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
        return False

    def forget(self, platform: str, message_id: str) -> None:
        """Drop a record, so a failed accept can be retried honestly.

        Called when handling raised before the turn began. Without it, a
        transient failure would be permanently indistinguishable from a
        completed run and the platform's retry would be swallowed.
        """
        self._seen.pop(f"{platform}:{message_id}", None)

    def _expire(self, now: float) -> None:
        cutoff = now - self.ttl_s
        while self._seen:
            key, stamp = next(iter(self._seen.items()))
            if stamp >= cutoff:
                return
            del self._seen[key]

    def __len__(self) -> int:
        """How many ids are being remembered."""
        return len(self._seen)
