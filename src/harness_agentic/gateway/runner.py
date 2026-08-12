"""One agent run per conversation, and what happens to everything else.

The rule that shapes this module: **a session key has at most one turn in
flight.** Two turns against one conversation would interleave their writes to
the same history and produce a transcript where neither the user's questions
nor the agent's answers are in order -- and since the store is the agent's
memory, that damage is permanent.

So each session gets a :class:`SessionActor`: a queue and a task. Messages that
arrive mid-run join the queue *and* interrupt the run, because a user who sends
a follow-up almost always means "actually, do this instead", and making them
wait out a turn they have already corrected is the single most irritating thing
a chat agent does.

The agent loop itself is synchronous and blocking. It runs on a worker thread
via ``asyncio.to_thread``, behind a semaphore that bounds how many turns can be
in flight across all conversations at once -- without it, forty chats mean forty
threads and forty concurrent provider bills.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from harness_agentic.agent.build import AgentBundle
from harness_agentic.core.events import (
    AgentEvent,
    EventSink,
    Notice,
    SwitchableSink,
    TurnFinished,
    null_sink,
)
from harness_agentic.gateway.clock import AsyncClock, RealAsyncClock
from harness_agentic.gateway.delivery import Delivery
from harness_agentic.gateway.types import DeliveryTarget, MessageEvent

if TYPE_CHECKING:
    from harness_agentic.gateway.adapter import PlatformAdapter
    from harness_agentic.session.store import SessionRecord

log = logging.getLogger(__name__)

MAX_QUEUE_PER_SESSION = 20
"""Beyond this the user is not conversing, and the oldest waiting message is
almost certainly obsolete anyway."""
COALESCE_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class Turn:
    """One unit of work for an actor."""

    event: MessageEvent
    target: DeliveryTarget
    text: str


BundleFactory = Callable[[str, MessageEvent, EventSink], Awaitable[AgentBundle]]
"""Builds the agent for a session key, wired to the given sink. Async because
the first call for a conversation may have to open a database."""


@dataclass
class ActorStats:
    """What one conversation has done. Read by ``/status``."""

    turns: int = 0
    interruptions: int = 0
    dropped: int = 0
    errors: int = 0
    last_finished_at: datetime | None = None
    last_reason: str = ""


class SessionActor:
    """Serializes all work for one session key."""

    def __init__(
        self,
        session_key: str,
        *,
        adapter: PlatformAdapter,
        bundle_factory: BundleFactory,
        semaphore: asyncio.Semaphore,
        clock: AsyncClock | None = None,
        show_tools: bool = True,
        max_queue: int = MAX_QUEUE_PER_SESSION,
    ) -> None:
        """Create an idle actor. The pump starts on the first message."""
        self.key = session_key
        self.adapter = adapter
        self.stats = ActorStats()
        self._factory = bundle_factory
        self._semaphore = semaphore
        self._clock = clock or RealAsyncClock()
        self._show_tools = show_tools
        self._queue: asyncio.Queue[Turn] = asyncio.Queue(maxsize=max_queue)
        self._pump: asyncio.Task[None] | None = None
        self._bundle: AgentBundle | None = None
        self._running = False
        self._delivery: Delivery | None = None
        self._pending_approval: asyncio.Future[bool] | None = None
        # Built once with the agent, re-pointed at each turn's delivery.
        self._sink = SwitchableSink()

    # -- inbound ---------------------------------------------------------------

    def submit(self, turn: Turn) -> bool:
        """Enqueue work, interrupting any run in progress.

        Returns whether it was accepted. A full queue drops the *oldest*
        waiting message rather than the newest: when someone is typing faster
        than the agent can answer, the most recent thing they said is the one
        they still care about.
        """
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self.stats.dropped += 1
        try:
            self._queue.put_nowait(turn)
        except asyncio.QueueFull:  # pragma: no cover - only under concurrent submits
            self.stats.dropped += 1
            return False
        if self._running:
            self.interrupt()
        self._ensure_pump()
        return True

    def interrupt(self) -> bool:
        """Ask the running turn to stop at its next checkpoint."""
        bundle = self._bundle
        if bundle is None or not self._running:
            return False
        bundle.runner.cancel.cancel("a newer message arrived")
        bundle.context.cancel.cancel("a newer message arrived")
        self.stats.interruptions += 1
        return True

    def _ensure_pump(self) -> None:
        if self._pump is None or self._pump.done():
            self._pump = asyncio.create_task(self._run_pump(), name=f"actor:{self.key}")

    # -- state -----------------------------------------------------------------

    @property
    def running(self) -> bool:
        """Whether a turn is in flight."""
        return self._running

    def queue_depth(self) -> int:
        """How many messages are waiting."""
        return self._queue.qsize()

    def describe(self) -> str:
        """A line of human-readable state."""
        parts = [f"{self.stats.turns} turn(s)"]
        if self.stats.interruptions:
            parts.append(f"{self.stats.interruptions} interrupted")
        if self.stats.dropped:
            parts.append(f"{self.stats.dropped} dropped")
        if self.stats.last_reason:
            parts.append(f"last: {self.stats.last_reason}")
        return "; ".join(parts)

    def resolve_approval(self, granted: bool) -> bool:
        """Answer the approval this conversation is blocked on."""
        pending = self._pending_approval
        if pending is None or pending.done():
            return False
        pending.set_result(granted)
        return True

    async def wait_idle(self) -> None:
        """Block until this conversation has nothing running or queued.

        Awaits the pump rather than polling, so a graceful shutdown finishes
        the turn a user is waiting on instead of racing it -- and so tests
        assert on the gateway's actual state rather than on a sleep long enough
        to probably be right.
        """
        while True:
            pump = self._pump
            if pump is None or pump.done():
                if self._queue.empty() and not self._running:
                    return
                # The pump exited between the queue check and here; let the
                # replacement start.
                await asyncio.sleep(0)
                continue
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    async def aclose(self) -> None:
        """Stop the pump, abandoning anything still queued."""
        self.interrupt()
        pump = self._pump
        self._pump = None
        if pump is not None and not pump.done():
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    # -- the pump --------------------------------------------------------------

    async def _run_pump(self) -> None:
        """Take turns off the queue until it is empty, then exit.

        Exiting on empty rather than parking forever means an idle conversation
        holds a dict entry and nothing else. A gateway that has served ten
        thousand chats over a month should not be running ten thousand tasks.
        """
        while True:
            try:
                turn = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            turn = self._coalesce(turn)
            try:
                await self._run_turn(turn)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed turn must not take the conversation's pump with it;
                # the next message deserves a working agent.
                self.stats.errors += 1
                log.exception("turn failed in %s", self.key)

    def _coalesce(self, first: Turn) -> Turn:
        """Merge messages that piled up while the previous turn ran.

        Three lines typed in quick succession are one thought, and answering
        them as three turns produces three partial answers that each miss the
        others' context.
        """
        parts = [first.text]
        while True:
            try:
                extra = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            parts.append(extra.text)
            first = extra
        if len(parts) == 1:
            return first
        return Turn(event=first.event, target=first.target, text=COALESCE_SEPARATOR.join(parts))

    async def _run_turn(self, turn: Turn) -> None:
        """Run one turn end to end, streaming as it goes."""
        async with self._semaphore:
            bundle = self._bundle
            if bundle is None:
                bundle = await self._factory(self.key, turn.event, self._sink)
                self._bundle = bundle

            delivery = Delivery(
                adapter=self.adapter,
                target=turn.target,
                clock=self._clock,
                show_tools=self._show_tools,
            )
            self._delivery = delivery
            loop = asyncio.get_running_loop()

            def sink(event: AgentEvent) -> None:
                # The agent loop is on a worker thread; hop back to the loop
                # the adapter lives on. Fire-and-forget, so a slow platform API
                # cannot stall the agent's next iteration.
                asyncio.run_coroutine_threadsafe(delivery.handle(event), loop)

            self._sink.target = sink

            await delivery.typing()
            self._running = True
            try:
                session = _require_session(bundle)
                result = await asyncio.to_thread(bundle.runner.run_turn, turn.text, session=session)
            except Exception as exc:
                self.stats.errors += 1
                self.stats.last_reason = "error"
                await delivery.handle(Notice("error", f"the turn failed: {exc}"))
                await delivery.finish()
                raise
            finally:
                self._running = False
                self._delivery = None
                # Anything the loop emits after this point -- a late tool
                # progress line from a thread still winding down -- must not
                # reach a delivery that has already been finished.
                self._sink.target = null_sink

            self.stats.turns += 1
            self.stats.last_reason = result.exit_reason
            self.stats.last_finished_at = self._clock.now()
            await delivery.finish()
            if result.exit_reason == "interrupted" and self._queue.empty():
                await delivery.say("⏹ Stopped.", kind="status")
            elif not result.final_text.strip() and result.exit_reason != "interrupted":
                # Silence reads as a crash. Say something, even if it is only
                # that there was nothing to say.
                await delivery.say(_empty_answer_note(result.exit_reason), kind="status")


def _require_session(bundle: AgentBundle) -> SessionRecord:
    """Fetch the bundle's session, refusing to run a turn without one."""
    session = bundle.store.get(bundle.context.session_id)
    if session is None:  # pragma: no cover - the store created it moments ago
        detail = f"session {bundle.context.session_id} is missing from the store"
        raise RuntimeError(detail)
    return session


def _empty_answer_note(reason: str) -> str:
    """What to say when a turn produced no text."""
    match reason:
        case "max_iterations":
            return "⚠ Stopped after too many steps without finishing. Try narrowing the request."
        case "content_filter":
            return "⚠ The model declined to answer that."
        case "error":
            return "⚠ The turn ended with an error."
        case _:
            return "(no answer)"


@dataclass
class ActorRegistry:
    """Every live conversation, keyed by session key."""

    actors: dict[str, SessionActor] = field(default_factory=dict)

    def get(self, key: str) -> SessionActor | None:
        """The actor for a key, if one exists."""
        return self.actors.get(key)

    def put(self, actor: SessionActor) -> SessionActor:
        """Remember an actor."""
        self.actors[actor.key] = actor
        return actor

    async def drop(self, key: str) -> bool:
        """Close and forget one conversation."""
        actor = self.actors.pop(key, None)
        if actor is None:
            return False
        await actor.aclose()
        return True

    async def wait_idle(self) -> None:
        """Block until no conversation has work outstanding."""
        while True:
            pending = [a for a in list(self.actors.values()) if a.running or a.queue_depth()]
            if not pending:
                return
            for actor in pending:
                await actor.wait_idle()

    async def aclose(self) -> None:
        """Close every conversation."""
        actors = list(self.actors.values())
        self.actors.clear()
        for actor in actors:
            await actor.aclose()

    def running(self) -> int:
        """How many turns are in flight right now."""
        return sum(1 for a in self.actors.values() if a.running)

    def queued(self) -> int:
        """How many messages are waiting across all conversations."""
        return sum(a.queue_depth() for a in self.actors.values())


def turn_finished(event: AgentEvent) -> bool:
    """Whether an event ends a turn. Used by surfaces that mirror the stream."""
    return isinstance(event, TurnFinished)
