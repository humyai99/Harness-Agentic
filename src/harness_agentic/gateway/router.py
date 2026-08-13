"""The path every inbound message takes, in the order it must take it.

Deduplicate, authorize, rate-limit, then route -- and the order is not
arbitrary:

* **Dedupe first.** A redelivery of a message already being worked on must not
  consume a second rate-limit token, and must not be answered twice.
* **Authorize before rate-limiting.** An unauthorized sender should not be able
  to exhaust a shared bucket, and their rejection costs nothing.
* **Commands before the queue.** Inline, so ``/stop`` reaches a busy actor.
* **Only then build an agent.** Everything above this line is cheap; a bundle
  opens a database and resolves credentials, and doing that for an unauthorized
  sender is how a scanner turns into a bill.

Each rejection is answered, not dropped. A user who is rate-limited and hears
nothing concludes the bot is broken and sends five more messages.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harness_agentic.core.clock import Clock, SystemClock
from harness_agentic.gateway.chunking import split_for_platform
from harness_agentic.gateway.commands import (
    CommandContext,
    CommandRegistry,
    CommandState,
    default_commands,
)
from harness_agentic.gateway.dedupe import Deduplicator
from harness_agentic.gateway.keys import KeyPolicy, build_session_key
from harness_agentic.gateway.ratelimit import RateLimiter
from harness_agentic.gateway.runner import ActorRegistry, BundleFactory, SessionActor, Turn
from harness_agentic.gateway.types import (
    ChatKind,
    DeliveryTarget,
    MessageEvent,
    OutboundMessage,
)

if TYPE_CHECKING:
    from harness_agentic.gateway.adapter import PlatformAdapter
    from harness_agentic.gateway.authz import Authorizer

log = logging.getLogger(__name__)

PAIRING_HINT = "Send `/pair <code>` in a direct message to enrol."


@dataclass
class RouteOutcome:
    """What the router did with one event. Returned for tests and metrics."""

    accepted: bool
    reason: str
    session_key: str = ""
    was_command: bool = False


@dataclass
class Router:
    """Turns normalized platform events into agent turns."""

    adapters: dict[str, PlatformAdapter]
    authorizer: Authorizer
    bundle_factory: BundleFactory
    actors: ActorRegistry = field(default_factory=ActorRegistry)
    limiter: RateLimiter = field(default_factory=RateLimiter)
    dedupe: Deduplicator = field(default_factory=Deduplicator)
    commands: CommandRegistry = field(default_factory=default_commands)
    key_policies: dict[str, KeyPolicy] = field(default_factory=dict)
    clock: Clock = field(default_factory=SystemClock)
    max_concurrent_runs: int = 4
    show_tools: bool = True
    require_mention_in_groups: bool = True
    _sem: asyncio.Semaphore = field(init=False)
    _closing: set[asyncio.Task[None]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        """Create the concurrency gate every actor shares.

        Bounded here rather than per conversation: forty chats must not mean
        forty worker threads and forty concurrent provider calls.
        """
        self._sem = asyncio.Semaphore(self.max_concurrent_runs)

    # -- the path --------------------------------------------------------------

    async def handle(self, event: MessageEvent) -> RouteOutcome:  # noqa: PLR0911
        """Route one inbound message.

        Every gate is its own early return. Collapsing them would hide the
        order, and the order is the security property.
        """
        adapter = self.adapters.get(event.platform)
        if adapter is None:
            return RouteOutcome(False, f"no adapter for {event.platform}")

        now = self.clock.now().timestamp()
        if self.dedupe.seen(event.platform, event.message_id, now):
            return RouteOutcome(False, "duplicate delivery")

        if self.require_mention_in_groups and not event.mentioned and _is_group(event):
            # A bot that answers everything in a busy channel gets removed from
            # it. Not an error, and not worth a reply.
            return RouteOutcome(False, "not addressed")

        target = DeliveryTarget(
            platform=event.platform,
            chat_id=event.chat_id,
            thread_id=event.thread_id,
            reply_to=event.reply_to,
        )
        session_key = build_session_key(event, self.key_policies.get(event.platform))

        decision = self.authorizer.check(event)
        if not decision:
            handled = await self._try_pairing(adapter, event, target)
            if handled:
                return RouteOutcome(True, "pairing attempt", session_key, was_command=True)
            await self._reply(adapter, target, f"{decision.reason}\n{PAIRING_HINT}")
            return RouteOutcome(False, decision.reason, session_key)

        rate_key = f"{event.platform}:{event.sender.id}"
        if not (verdict := self.limiter.check_message(rate_key, now)):
            await self._reply(adapter, target, verdict.message())
            return RouteOutcome(False, "rate limited", session_key)

        if event.is_command():
            reply = await self._run_command(event, session_key)
            await self._reply(adapter, target, reply)
            return RouteOutcome(True, "command", session_key, was_command=True)

        if not event.text.strip():
            return RouteOutcome(False, "empty message", session_key)

        if not (turn_verdict := self.limiter.check_turn(rate_key, now)):
            await self._reply(adapter, target, turn_verdict.message())
            return RouteOutcome(False, "turn budget exhausted", session_key)

        actor = self._actor_for(session_key, adapter)
        if not actor.submit(Turn(event=event, target=target, text=event.text)):
            self.limiter.refund_turn(rate_key, now)
            await self._reply(adapter, target, "Too many messages queued; try again shortly.")
            return RouteOutcome(False, "queue full", session_key)
        return RouteOutcome(True, "queued", session_key)

    # -- pieces ----------------------------------------------------------------

    def _actor_for(self, session_key: str, adapter: PlatformAdapter) -> SessionActor:
        """The actor for a conversation, created on first use."""
        existing = self.actors.get(session_key)
        if existing is not None:
            return existing
        return self.actors.put(
            SessionActor(
                session_key,
                adapter=adapter,
                bundle_factory=self.bundle_factory,
                semaphore=self._sem,
                show_tools=self.show_tools,
            )
        )

    async def _try_pairing(
        self, adapter: PlatformAdapter, event: MessageEvent, target: DeliveryTarget
    ) -> bool:
        """Let an unauthorized sender redeem a code, and nothing else.

        This is the one thing someone who is not yet authorized may do, which
        is why it is handled here rather than in the command registry -- the
        registry runs after the authorization check has already passed.
        """
        text = event.text.strip()
        if not text.lower().startswith("/pair"):
            return False
        code = text[len("/pair") :].strip()
        if not code:
            await self._reply(adapter, target, "Send `/pair <code>` with the code you were given.")
            return True
        decision = self.authorizer.redeem(event, code, now=self.clock.now())
        await self._reply(
            adapter,
            target,
            "Paired. You can talk to the agent now." if decision else decision.reason,
        )
        return True

    def command_state(self) -> CommandState:
        """The narrow view of gateway state that commands may act on."""

        def interrupt(key: str) -> bool:
            actor = self.actors.get(key)
            return actor.interrupt() if actor else False

        def queue_depth(key: str) -> int:
            actor = self.actors.get(key)
            return actor.queue_depth() if actor else 0

        def is_running(key: str) -> bool:
            actor = self.actors.get(key)
            return actor.running if actor else False

        def describe(key: str) -> str:
            actor = self.actors.get(key)
            return actor.describe() if actor else "No history yet."

        def resolve_approval(key: str, granted: bool) -> bool:
            actor = self.actors.get(key)
            return actor.resolve_approval(granted) if actor else False

        return CommandState(
            interrupt=interrupt,
            queue_depth=queue_depth,
            is_running=is_running,
            reset=self._reset,
            describe=describe,
            resolve_approval=resolve_approval,
        )

    async def _run_command(self, event: MessageEvent, session_key: str) -> str:
        """Dispatch a slash command against live gateway state."""
        ctx = CommandContext(
            event=event,
            args="",
            session_key=session_key,
            authorizer=self.authorizer,
            now=self.clock.now(),
            state=self.command_state(),
        )
        return await self.commands.dispatch(ctx)

    def _reset(self, session_key: str) -> str:
        """Forget a conversation so the next message starts fresh.

        The stored history is not deleted -- ``harn sessions`` can still show
        it. Dropping the actor drops the *binding*, so a new agent and a new
        session id are built on the next message.
        """
        actor = self.actors.actors.pop(session_key, None)
        if actor is None:
            return "Already starting fresh."
        task = asyncio.create_task(actor.aclose())
        # Held so the task is not garbage-collected mid-flight, and discarded
        # when it finishes.
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)
        return "Started a fresh conversation. Earlier messages are still searchable."

    async def _reply(self, adapter: PlatformAdapter, target: DeliveryTarget, text: str) -> None:
        """Send one short out-of-band message, never raising."""
        if not text:
            return
        try:
            for piece in split_for_platform(text, adapter.capabilities.max_chars):
                await adapter.send(target, OutboundMessage(piece, kind="status"))
        except Exception:
            log.exception("failed to reply on %s", adapter.platform)

    async def aclose(self) -> None:
        """Close every conversation."""
        await self.actors.aclose()

    def status_lines(self) -> Sequence[str]:
        """A snapshot of the gateway, for the CLI and ``/status``."""
        return (
            (
                f"{len(self.actors.actors)} conversation(s), "
                f"{self.actors.running()} running, {self.actors.queued()} queued"
            ),
            (
                f"{len(self.dedupe)} message id(s) remembered, "
                f"{self.dedupe.duplicates} duplicate(s) rejected"
            ),
            f"{self.limiter.tracked()} sender(s) rate-tracked",
        )


def _is_group(event: MessageEvent) -> bool:
    """Whether more than one human can see this conversation."""
    return event.chat_kind in (ChatKind.GROUP, ChatKind.CHANNEL)
