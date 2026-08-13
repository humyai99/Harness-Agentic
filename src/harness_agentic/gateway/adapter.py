"""The contract every chat platform is reduced to.

Three transports, one interface. A long-poll adapter yields events from
:meth:`PlatformAdapter.poll`, a webhook adapter registers
:meth:`PlatformAdapter.webhook_routes` and pushes events into a queue as
requests arrive, and a socket adapter runs its own connection loop. The router
consumes one stream and cannot tell the difference.

The deliberate omission is ``edit``. It is not on this class, because LINE
cannot do it and a method that raises on one implementation is a lie in the
type. Editing lives behind :attr:`Capabilities.edit_messages` and the delivery
layer checks before it commits to a streaming strategy -- which is the whole
reason LINE is in the first gateway milestone rather than the fourth. Building
against Telegram alone produces an interface that assumes edits, and then LINE
arrives and every adapter has to change.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence

from harness_agentic.errors import AdapterError
from harness_agentic.gateway.types import (
    Capabilities,
    DeliveryTarget,
    MessageEvent,
    OutboundMessage,
    Route,
    SentRef,
    Transport,
)


class PlatformAdapter(abc.ABC):
    """One chat platform, normalized.

    Subclasses own their wire format and nothing else: no session logic, no
    authorization, no chunking. Those live once, above this line, so a bug
    fixed for Telegram is fixed for LINE at the same moment.
    """

    platform: str
    transport: Transport
    capabilities: Capabilities = Capabilities()
    """Set at class level by real adapters; overridden per instance by the fake,
    which impersonates other platforms' limits so their delivery paths are
    exercised without an account on them."""

    async def connect(self) -> None:
        """Acquire whatever the platform needs before events can flow."""

    async def disconnect(self) -> None:
        """Release it again. Must be safe to call without a prior connect."""

    @abc.abstractmethod
    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        """Post a message and return a handle to it."""

    async def edit(self, ref: SentRef, message: OutboundMessage) -> None:
        """Replace an already-posted message.

        Only ever called when :attr:`Capabilities.edit_messages` is set, so the
        default is a hard error rather than a silent no-op -- an adapter that
        claims the capability and forgets the method should fail loudly in the
        first test, not stream into a void.
        """
        detail = f"{self.platform} declares edit_messages but does not implement edit()"
        raise AdapterError(detail)

    async def typing(self, target: DeliveryTarget) -> None:
        """Show a typing indicator, if the platform has one."""

    def webhook_routes(self) -> Sequence[Route]:
        """HTTP endpoints this adapter needs served. Webhook transports only."""
        return ()

    def poll(self) -> AsyncIterator[MessageEvent]:
        """Yield events as they arrive. Poll and socket transports only."""
        detail = f"{self.platform} is a {self.transport} adapter and does not poll"
        raise AdapterError(detail)

    def describe(self) -> str:
        """A one-line summary for ``harn gateway --check``."""
        caps = self.capabilities
        traits = [f"{caps.max_chars} chars"]
        traits.append("edit" if caps.edit_messages else "chunked")
        if caps.typing_indicator:
            traits.append("typing")
        if caps.threads:
            traits.append("threads")
        return f"{self.platform} ({self.transport}): {', '.join(traits)}"


class QueueAdapter(PlatformAdapter):
    """Base for adapters whose events arrive from somewhere else.

    A webhook handler runs on the web server's task and must return quickly;
    parking the event on a queue and returning 200 is what "quickly" means in
    practice. This holds the queue so every webhook adapter does not reinvent
    it, and bounds it so a flood of deliveries cannot exhaust memory -- when
    full, the oldest event is dropped and counted, because a gateway that
    silently stops accepting messages is worse than one that admits it lost
    some.
    """

    def __init__(self, *, maxsize: int = 1000) -> None:
        """Create an adapter with a bounded inbound queue."""
        self._queue: asyncio.Queue[MessageEvent] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def offer(self, event: MessageEvent) -> bool:
        """Hand an event to the router. Returns whether it was accepted."""
        if self._queue.full():
            self.dropped += 1
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover - only under concurrent offers
            self.dropped += 1
            return False
        return True

    def poll(self) -> AsyncIterator[MessageEvent]:
        """Drain the queue forever."""
        return self._drain()

    async def _drain(self) -> AsyncIterator[MessageEvent]:
        while True:
            yield await self._queue.get()
