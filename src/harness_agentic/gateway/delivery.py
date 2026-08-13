"""Turning a stream of agent events into messages a platform will accept.

This is where the two streaming strategies live, chosen from
:class:`~harness_agentic.gateway.types.Capabilities` rather than from a platform
name:

* **Edit-in-place** (Telegram, Slack, Discord): post a placeholder, then rewrite
  it on a debounce. The user watches the answer appear. The constraint is the
  platform's edit rate limit, so edits coalesce -- three chunks arriving in
  200ms become one edit, not three.
* **Chunked** (LINE): messages are immutable once sent, so there is nothing to
  rewrite. Text accumulates until it reaches a natural boundary or the size cap
  and then goes out as its own message. Fewer, larger messages, and no edits.

Getting this wrong is loud in opposite ways. Editing too eagerly gets the bot
rate-limited off the platform; buffering everything means a two-minute silence
followed by a wall of text, and the user has already asked "are you there".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harness_agentic.core.events import (
    AgentEvent,
    ApprovalRequested,
    CompactionStarted,
    Notice,
    ProviderFallback,
    TextChunk,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
)
from harness_agentic.gateway.chunking import clip, split_for_platform
from harness_agentic.gateway.clock import AsyncClock, RealAsyncClock
from harness_agentic.gateway.types import DeliveryTarget, OutboundMessage, SentRef

if TYPE_CHECKING:
    from harness_agentic.gateway.adapter import PlatformAdapter

log = logging.getLogger(__name__)

FLUSH_AT_RATIO = 0.8
"""Flush a chunked buffer once it reaches this share of the size cap, so the
split lands on a paragraph rather than exactly on the limit."""
STATUS_PREVIEW_CHARS = 120


@dataclass
class DeliveryStats:
    """What a delivery actually cost, for tests and for ``/status``."""

    sends: int = 0
    edits: int = 0
    failures: int = 0
    """Platform calls that were rejected. The text they carried is retried."""
    edits_skipped: int = 0
    """Debounced away. High is good: it means coalescing worked."""
    chars: int = 0


@dataclass
class Delivery:
    """Streams one turn's output to one chat.

    Not reusable across turns: it holds the placeholder reference and the
    buffer for exactly one answer, and a fresh one per turn is cheaper than
    reasoning about reset.
    """

    adapter: PlatformAdapter
    target: DeliveryTarget
    clock: AsyncClock = field(default_factory=RealAsyncClock)
    show_tools: bool = True
    """Whether tool activity is narrated. Off for surfaces where it is noise."""
    stats: DeliveryStats = field(default_factory=DeliveryStats)

    _buffer: str = ""
    _posted: SentRef | None = None
    _posted_text: str = ""
    _edit_count: int = 0
    _last_edit_at: float = -1e9
    _pending: asyncio.Task[None] | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _closed: bool = False

    # -- the event sink -------------------------------------------------------

    async def handle(self, event: AgentEvent) -> None:
        """React to one agent event."""
        match event:
            case TextChunk(text=text):
                await self._append(text)
            case ToolCallStarted(tool=tool, summary=summary) if self.show_tools:
                await self.status(f"⚙ {tool}: {clip(summary, STATUS_PREVIEW_CHARS)}")
            case ToolCallFinished(tool=tool, is_error=True) if self.show_tools:
                await self.status(f"⚠ {tool} failed")
            case ApprovalRequested(tool=tool, summary=summary):
                await self.say(
                    f"🔒 *{tool}* needs approval:\n`{clip(summary, 300)}`\n"
                    f"Reply `/approve` to allow it, `/deny` to refuse.",
                    kind="approval",
                )
            case ProviderFallback(from_model=old, to_model=new, reason=reason):
                await self.status(f"↩ {old} unavailable ({reason}); using {new}")
            case CompactionStarted():
                await self.status("🗜 summarizing earlier messages to make room")
            case Notice(level="error", message=message):
                await self.say(f"⚠ {message}", kind="error")
            case TurnFinished():
                await self.finish()
            case _:
                return

    # -- text ------------------------------------------------------------------

    async def _append(self, text: str) -> None:
        """Add answer text, flushing or scheduling an edit as appropriate."""
        if self._closed or not text:
            return
        async with self._lock:
            self._buffer += text
        if self.adapter.capabilities.edit_messages:
            self._schedule_edit()
        else:
            await self._maybe_flush_chunk()

    async def _maybe_flush_chunk(self) -> None:
        """Send a whole message once enough has accumulated.

        Only for platforms that cannot edit. Waits for a paragraph break so the
        seam between messages falls somewhere a reader would have paused
        anyway.
        """
        cap = self.adapter.capabilities.max_chars
        async with self._lock:
            if len(self._buffer) < cap * FLUSH_AT_RATIO:
                return
            split = self._buffer.rfind("\n\n")
            if split < cap * 0.3:
                split = len(self._buffer) if len(self._buffer) >= cap else -1
            if split <= 0:
                return
            piece, self._buffer = self._buffer[:split], self._buffer[split:].lstrip("\n")
        await self._send_text(piece)

    def _schedule_edit(self) -> None:
        """Ensure exactly one debounced edit is in flight."""
        if self._pending is not None and not self._pending.done():
            self.stats.edits_skipped += 1
            return
        self._pending = asyncio.create_task(self._edit_after_debounce())

    async def _edit_after_debounce(self) -> None:
        """Wait out the platform's edit interval, then push the latest text.

        Failures are absorbed here rather than left to escape. This runs as a
        fire-and-forget task, so an exception would surface only as asyncio's
        "task exception was never retrieved" and the answer would go missing with
        no other trace. Because ``_posted_text`` is now only advanced on success,
        the next chunk's push retries the text that did not land.
        """
        interval = self.adapter.capabilities.min_edit_interval_s
        elapsed = self.clock.monotonic() - self._last_edit_at
        if elapsed < interval:
            await self.clock.sleep(interval - elapsed)
        try:
            await self._push_edit()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.stats.failures += 1
            log.warning("%s rejected an edit (%s); will retry", self.adapter.platform, exc)

    async def _push_edit(self) -> None:
        """Post or rewrite the streaming message with everything so far."""
        async with self._lock:
            text = self._buffer
            if not text.strip() or text == self._posted_text:
                return
            caps = self.adapter.capabilities
            if len(text) > caps.max_chars or self._edit_count >= caps.max_edits_per_message:
                # The message is full, or has been rewritten as many times as
                # the platform tolerates. Seal it and start another.
                head, *rest = split_for_platform(text, caps.max_chars)
                await self._commit(head)
                self._buffer = "\n\n".join(rest)
                self._posted, self._posted_text, self._edit_count = None, "", 0
                return

        # Recorded only *after* the platform accepted it. Setting it first meant
        # a rate-limited edit left `_posted_text` claiming text that never
        # landed, and every later attempt then saw "nothing changed" and skipped
        # it -- so the user got a silently truncated answer.
        if self._posted is None:
            self._posted = await self.adapter.send(self.target, OutboundMessage(text))
            self.stats.sends += 1
        else:
            await self.adapter.edit(self._posted, OutboundMessage(text))
            self.stats.edits += 1
            self._edit_count += 1
        self._posted_text = text
        self.stats.chars = len(text)
        self._last_edit_at = self.clock.monotonic()

    async def _commit(self, text: str) -> None:
        """Finalize the in-flight message with ``text``, or send it fresh."""
        if self._posted is None:
            await self.adapter.send(self.target, OutboundMessage(text))
            self.stats.sends += 1
        else:
            await self.adapter.edit(self._posted, OutboundMessage(text))
            self.stats.edits += 1
        self._last_edit_at = self.clock.monotonic()

    async def _send_text(self, text: str) -> None:
        """Send text as one or more whole messages."""
        for piece in split_for_platform(text, self.adapter.capabilities.max_chars):
            if not piece.strip():
                continue
            await self.adapter.send(self.target, OutboundMessage(piece))
            self.stats.sends += 1
            self.stats.chars += len(piece)

    # -- out-of-band messages --------------------------------------------------

    async def say(self, text: str, *, kind: str = "answer") -> None:
        """Send a message immediately, outside the streaming buffer.

        Approval prompts and errors go this way: they must arrive now, and they
        must not be swallowed when the streaming message is later rewritten.
        """
        for piece in split_for_platform(text, self.adapter.capabilities.max_chars):
            await self.adapter.send(
                self.target,
                OutboundMessage(piece, kind=kind, silent=kind == "status"),  # type: ignore[arg-type]
            )
            self.stats.sends += 1

    async def status(self, text: str) -> None:
        """Report progress, where the platform makes that cheap.

        Silently dropped on platforms without edits: a LINE conversation
        interleaved with twelve "running grep" notifications is worse for the
        reader than no progress at all, and every one of them is a push
        message the operator pays for.
        """
        if not self.adapter.capabilities.edit_messages:
            return
        await self.say(text, kind="status")

    async def typing(self) -> None:
        """Show a typing indicator if there is one."""
        if self.adapter.capabilities.typing_indicator:
            with contextlib.suppress(Exception):
                await self.adapter.typing(self.target)

    # -- shutdown --------------------------------------------------------------

    async def finish(self) -> None:
        """Flush whatever is left and stop accepting text.

        Idempotent, because a turn can end through several paths -- normal
        completion, interruption, an exception -- and each of them wants to be
        sure the user got the answer.
        """
        if self._closed:
            return
        self._closed = True
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending
        async with self._lock:
            leftover = self._buffer
            posted = self._posted_text
        if not leftover.strip():
            return
        if self.adapter.capabilities.edit_messages and leftover != posted:
            self._last_edit_at = -1e9  # the final write is never debounced away
            try:
                await self._push_edit()
            except Exception as exc:
                # The turn is over, so there is no later push to retry on. Say so
                # rather than ending silently with a half-delivered answer.
                self.stats.failures += 1
                log.warning("%s rejected the final edit: %s", self.adapter.platform, exc)
            return
        if not self.adapter.capabilities.edit_messages:
            await self._send_text(leftover)
            async with self._lock:
                self._buffer = ""


def sink_for(delivery: Delivery, loop: asyncio.AbstractEventLoop) -> object:
    """Adapt a :class:`Delivery` into the synchronous core's event sink.

    The agent loop is synchronous and runs on a worker thread; the adapter is
    asyncio and lives on the gateway's loop. This is the crossing, and it is
    fire-and-forget on purpose: a slow platform API must never be able to stall
    the agent's next iteration.
    """

    def sink(event: AgentEvent) -> None:
        asyncio.run_coroutine_threadsafe(delivery.handle(event), loop)

    return sink


def summarize_events(events: Sequence[AgentEvent]) -> str:
    """A short account of what a turn did, for ``/status`` and audit lines."""
    tools = [e.tool for e in events if isinstance(e, ToolCallStarted)]
    failed = sum(1 for e in events if isinstance(e, ToolCallFinished) and e.is_error)
    parts = [f"{len(tools)} tool call(s)"]
    if failed:
        parts.append(f"{failed} failed")
    if tools:
        parts.append("used " + ", ".join(dict.fromkeys(tools)))
    return "; ".join(parts)
