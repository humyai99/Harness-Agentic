"""Slash commands, and why they never wait in line.

Every command here runs *inline*, ahead of the queue, and that is the entire
point of the module. A user whose agent is grinding through a long turn needs
``/stop`` to work right now; if it queued behind the run it was meant to cancel
it would be useless exactly when it is needed. Same for ``/status`` -- the
question "is this thing still alive" cannot be answered by a mechanism that is
also stuck.

So commands may not do anything expensive. They read state, set a flag, or
answer a question. Anything that needs the model is not a command.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from harness_agentic.gateway.types import ChatKind, MessageEvent

if TYPE_CHECKING:
    from harness_agentic.gateway.authz import Authorizer


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Everything a command may look at."""

    event: MessageEvent
    args: str
    session_key: str
    authorizer: Authorizer
    now: datetime
    state: CommandState


class CommandState:
    """The gateway operations a command is allowed to perform.

    A narrow surface deliberately: commands can interrupt, inspect and reset,
    and that is all. Widening this is how ``/status`` grows the ability to
    change the model, and then an unauthorized user finds it.
    """

    def __init__(
        self,
        *,
        interrupt: Callable[[str], bool],
        queue_depth: Callable[[str], int],
        is_running: Callable[[str], bool],
        reset: Callable[[str], str],
        describe: Callable[[str], str],
        resolve_approval: Callable[[str, bool], bool] | None = None,
    ) -> None:
        """Bind the callbacks the gateway exposes to commands."""
        self.interrupt = interrupt
        self.queue_depth = queue_depth
        self.is_running = is_running
        self.reset = reset
        self.describe = describe
        self.resolve_approval = resolve_approval


Handler = Callable[[CommandContext], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class Command:
    """One command's name, help text, and access level."""

    name: str
    help: str
    handler: Handler
    admin_only: bool = False
    private_only: bool = False
    """Refused outside a direct message. Pairing, and anything that names ids."""


@dataclass
class CommandRegistry:
    """The commands a gateway understands."""

    commands: dict[str, Command] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)

    def add(self, command: Command) -> None:
        """Register a command."""
        self.commands[command.name] = command

    def lookup(self, name: str) -> Command | None:
        """Find a command by name or alias."""
        canonical = self.aliases.get(name, name)
        return self.commands.get(canonical)

    async def dispatch(self, ctx: CommandContext) -> str:
        """Run the command named in ``ctx.event``, or explain why not."""
        name, _, args = ctx.event.text.lstrip().removeprefix("/").partition(" ")
        command = self.lookup(name.lower())
        if command is None:
            return f"Unknown command `/{name}`. Try `/help`."
        if command.private_only and ctx.event.chat_kind is not ChatKind.PRIVATE:
            return f"`/{name}` only works in a direct message."
        if command.admin_only and not ctx.authorizer.is_admin(
            ctx.event.platform, ctx.event.sender.id
        ):
            return f"`/{name}` is for administrators."
        return await command.handler(
            CommandContext(
                event=ctx.event,
                args=args.strip(),
                session_key=ctx.session_key,
                authorizer=ctx.authorizer,
                now=ctx.now,
                state=ctx.state,
            )
        )


def default_commands() -> CommandRegistry:
    """The commands every gateway has."""
    registry = CommandRegistry(
        aliases={"cancel": "stop", "abort": "stop", "reset": "new", "h": "help", "?": "help"}
    )

    async def stop(ctx: CommandContext) -> str:
        if not ctx.state.is_running(ctx.session_key):
            return "Nothing is running."
        stopped = ctx.state.interrupt(ctx.session_key)
        # "Stopping" rather than "stopped": cancellation is cooperative, so the
        # turn ends at the next checkpoint and claiming otherwise would be a
        # lie the user can see through when one more tool result arrives.
        return "Stopping at the next checkpoint." if stopped else "Could not interrupt that run."

    async def status(ctx: CommandContext) -> str:
        running = ctx.state.is_running(ctx.session_key)
        depth = ctx.state.queue_depth(ctx.session_key)
        lines = ["*Running*" if running else "*Idle*", ctx.state.describe(ctx.session_key)]
        if depth:
            lines.append(f"{depth} message(s) waiting.")
        return "\n".join(line for line in lines if line)

    async def queue(ctx: CommandContext) -> str:
        depth = ctx.state.queue_depth(ctx.session_key)
        return f"{depth} message(s) waiting." if depth else "The queue is empty."

    async def new(ctx: CommandContext) -> str:
        if ctx.state.is_running(ctx.session_key):
            ctx.state.interrupt(ctx.session_key)
        return ctx.state.reset(ctx.session_key)

    async def whoami(ctx: CommandContext) -> str:
        admin = ctx.authorizer.is_admin(ctx.event.platform, ctx.event.sender.id)
        return (
            f"{ctx.event.sender.label()} on {ctx.event.platform}\n"
            f"session `{ctx.session_key}`\n"
            f"{'administrator' if admin else 'standard user'}"
        )

    async def pair(ctx: CommandContext) -> str:
        if not ctx.args:
            return "Send `/pair <code>` with the code your operator gave you."
        decision = ctx.authorizer.redeem(ctx.event, ctx.args, now=ctx.now)
        return "Paired. You can talk to the agent now." if decision else decision.reason

    async def approve(ctx: CommandContext) -> str:
        return _resolve(ctx, granted=True)

    async def deny(ctx: CommandContext) -> str:
        return _resolve(ctx, granted=False)

    async def help_(ctx: CommandContext) -> str:
        admin = ctx.authorizer.is_admin(ctx.event.platform, ctx.event.sender.id)
        lines = [
            f"`/{c.name}` — {c.help}"
            for c in sorted(registry.commands.values(), key=lambda c: c.name)
            if admin or not c.admin_only
        ]
        return "\n".join(lines)

    for command in (
        Command("stop", "interrupt the current run", stop),
        Command("status", "what the agent is doing", status),
        Command("queue", "how many messages are waiting", queue),
        Command("new", "start a fresh conversation", new),
        Command("whoami", "your identity and session", whoami),
        Command("pair", "enrol with a pairing code", pair, private_only=True),
        Command("approve", "allow the pending tool call", approve),
        Command("deny", "refuse the pending tool call", deny),
        Command("help", "list commands", help_),
    ):
        registry.add(command)
    return registry


def _resolve(ctx: CommandContext, *, granted: bool) -> str:
    """Answer a pending approval, if the surface supports approvals at all."""
    if ctx.state.resolve_approval is None:
        return "This surface does not ask for approvals."
    if not ctx.state.resolve_approval(ctx.session_key, granted):
        return "Nothing is waiting for approval."
    return "Approved." if granted else "Refused."
