"""A platform that exists only in memory.

This is not a testing convenience bolted on afterwards -- it is the adapter the
whole gateway is developed against, for the same reason ``FakeTransport`` is the
first provider. Every path through the router, the actors, the delivery layer
and the commands can be driven end to end with no token, no webhook URL, and no
network, which means those paths are exercised on every commit rather than
whenever someone remembers to open Telegram.

It also ships in the package rather than living under ``tests/``, so anyone
writing a plugin or a new adapter can drive their code the same way.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from harness_agentic.gateway.adapter import PlatformAdapter, QueueAdapter
from harness_agentic.gateway.types import (
    Capabilities,
    ChatKind,
    DeliveryTarget,
    MessageEvent,
    OutboundMessage,
    Sender,
    SentRef,
    Transport,
)


@dataclass
class Posted:
    """One message this adapter was asked to deliver."""

    target: DeliveryTarget
    message: OutboundMessage
    message_id: str
    edits: list[str] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        """What the reader ends up seeing, after every edit."""
        return self.edits[-1] if self.edits else self.message.text


class FakeAdapter(QueueAdapter):
    """Records what would have been sent, and lets a test inject messages."""

    platform = "fake"
    transport = Transport.POLL
    capabilities = Capabilities(
        max_chars=200,
        edit_messages=True,
        typing_indicator=True,
        min_edit_interval_s=1.5,
        markdown="commonmark",
    )

    def __init__(
        self, *, capabilities: Capabilities | None = None, maxsize: int = 1000, name: str = "fake"
    ) -> None:
        """Create an adapter, optionally impersonating another platform's limits.

        Passing LINE's capabilities here is how the chunked delivery path gets
        tested without a LINE account, which is the difference between that
        path being covered and being hoped about.
        """
        super().__init__(maxsize=maxsize)
        if capabilities is not None:
            self.capabilities = capabilities
        self.platform = name
        self.posted: list[Posted] = []
        self.typing_calls = 0
        self.connected = False
        self.fail_next_send: Exception | None = None
        self._counter = 0

    async def connect(self) -> None:
        """Mark the adapter live."""
        self.connected = True

    async def disconnect(self) -> None:
        """Mark it closed."""
        self.connected = False

    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        """Record a message as posted."""
        if self.fail_next_send is not None:
            error, self.fail_next_send = self.fail_next_send, None
            raise error
        self._counter += 1
        message_id = f"m{self._counter}"
        self.posted.append(Posted(target=target, message=message, message_id=message_id))
        return SentRef(platform=self.platform, chat_id=target.chat_id, message_id=message_id)

    async def edit(self, ref: SentRef, message: OutboundMessage) -> None:
        """Record an edit against a previously posted message."""
        if self.fail_next_send is not None:
            error, self.fail_next_send = self.fail_next_send, None
            raise error
        for posted in self.posted:
            if posted.message_id == ref.message_id:
                posted.edits.append(message.text)
                return
        detail = f"no such message {ref.message_id}"
        raise KeyError(detail)

    async def typing(self, target: DeliveryTarget) -> None:
        """Count a typing indicator."""
        self.typing_calls += 1

    # -- driving it from a test ------------------------------------------------

    def inject(
        self,
        text: str,
        *,
        sender: str = "u1",
        chat: str = "c1",
        kind: ChatKind = ChatKind.PRIVATE,
        thread: str = "",
        message_id: str = "",
        mentioned: bool = True,
        at: datetime | None = None,
    ) -> MessageEvent:
        """Queue an inbound message as if a user had sent it."""
        self._counter += 1
        event = MessageEvent(
            platform=self.platform,
            chat_id=chat,
            chat_kind=kind,
            sender=Sender(id=sender, display_name=sender),
            text=text,
            received_at=at or datetime(2026, 1, 1, tzinfo=UTC),
            message_id=message_id or f"in{self._counter}",
            thread_id=thread,
            mentioned=mentioned,
        )
        self.offer(event)
        return event

    # -- assertions ------------------------------------------------------------

    def texts(self) -> list[str]:
        """Everything the reader ends up seeing, in order."""
        return [p.final_text for p in self.posted]

    def transcript(self) -> str:
        """All delivered text joined, for a single readable assertion."""
        return "\n".join(self.texts())

    def of_kind(self, kind: str) -> list[Posted]:
        """Messages of one kind -- answers, statuses, approvals, errors."""
        return [p for p in self.posted if p.message.kind == kind]

    def assert_within_limits(self) -> None:
        """Fail if anything sent exceeds what the platform would accept.

        Worth asserting explicitly: an over-length message is rejected by the
        real platform with an error the user never sees, so the symptom in
        production is a missing answer rather than a crash.
        """
        for posted in self.posted:
            for text in [posted.message.text, *posted.edits]:
                if len(text) > self.capabilities.max_chars:
                    detail = (
                        f"{self.platform} message of {len(text)} chars exceeds "
                        f"the {self.capabilities.max_chars} limit: {text[:80]!r}"
                    )
                    raise AssertionError(detail)


def line_like() -> Capabilities:
    """LINE's shape: long messages, no edits, no typing indicator."""
    return Capabilities(max_chars=5000, edit_messages=False, typing_indicator=False)


def telegram_like() -> Capabilities:
    """Telegram's shape: shorter messages, edits, typing."""
    return Capabilities(
        max_chars=4096, edit_messages=True, typing_indicator=True, markdown="markdown_v2"
    )


async def drain(adapter: PlatformAdapter, *, limit: int) -> list[MessageEvent]:
    """Take up to ``limit`` events off an adapter. Test helper."""
    events: list[MessageEvent] = []
    stream = adapter.poll()
    for _ in range(limit):
        try:
            events.append(await asyncio.wait_for(stream.__anext__(), timeout=1.0))
        except (TimeoutError, StopAsyncIteration):
            break
    return events
