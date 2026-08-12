"""``harn gateway`` -- run the chat platforms, and the commands that support it.

Three subcommands, and the split is deliberate:

* ``run`` starts the process.
* ``check`` starts nothing and prints what *would* start, including every way
  the configuration is more open than the operator probably meant. Running that
  before exposing a bot is the difference between finding ``allow_all`` in a
  config review and finding it in a bill.
* ``pair`` mints an invitation, because allowlisting people by numeric platform
  id is miserable and nobody knows their own.

Credentials are read from the environment, never from arguments. A token passed
as ``--token`` is in the shell history, in ``ps`` output, and in any process
listing on the box.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.table import Table

from harness_agentic.agent.build import build_agent
from harness_agentic.cli.render import console
from harness_agentic.constants import harness_home
from harness_agentic.errors import AdapterError, CredentialError, HarnessError
from harness_agentic.gateway.authz import Authorizer, PlatformAuth
from harness_agentic.gateway.platforms import KNOWN, load
from harness_agentic.gateway.router import Router
from harness_agentic.gateway.service import Gateway
from harness_agentic.gateway.webserver import DEFAULT_HOST, DEFAULT_PORT
from harness_agentic.providers.credentials import SecretResolver
from harness_agentic.tools.approval import ApprovalPolicy

if TYPE_CHECKING:
    from harness_agentic.agent.build import AgentBundle
    from harness_agentic.core.events import EventSink
    from harness_agentic.gateway.adapter import PlatformAdapter
    from harness_agentic.gateway.types import MessageEvent

DEFAULT_TOOLSETS = "core"
"""What a chat surface gets unless the operator widens it.

Not ``file`` and certainly not ``terminal``. A LINE official account is
reachable by strangers, and the difference between a chat assistant and a
remote shell is exactly this default."""

_ENV_KEYS = {
    "telegram": ("TELEGRAM_BOT_TOKEN",),
    "line": ("LINE_CHANNEL_SECRET", "LINE_CHANNEL_ACCESS_TOKEN"),
}


def register(app: typer.Typer) -> None:
    """Attach the ``gateway`` command group."""
    group = typer.Typer(help="Run the agent on chat platforms.", no_args_is_help=True)
    app.add_typer(group, name="gateway")

    @group.command("run")
    def run(
        platforms: str = typer.Option(
            "telegram", "--platforms", "-p", help="Comma-separated platform names."
        ),
        model: str = typer.Option("anthropic/claude-sonnet-5", "--model", "-m"),
        toolsets: str = typer.Option(
            DEFAULT_TOOLSETS,
            "--toolsets",
            help="Toolsets chat users may reach. Widen this deliberately.",
        ),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        host: str = typer.Option(DEFAULT_HOST, "--host", help="Webhook bind address."),
        port: int = typer.Option(DEFAULT_PORT, "--port"),
        allow_all: bool = typer.Option(
            False, "--allow-all", help="Serve anyone who messages the bot. Rarely right."
        ),
        max_concurrent: int = typer.Option(4, "--max-concurrent"),
        show_tools: bool = typer.Option(True, "--show-tools/--quiet-tools"),
    ) -> None:
        """Run the gateway until interrupted."""
        names = _names(platforms)
        adapters = _build_adapters(names)
        authorizer = _authorizer(names, allow_all=allow_all)

        for warning in authorizer.warnings():
            console.print(f"[bold yellow]warning:[/] {warning}")

        gateway = _assemble(
            adapters,
            authorizer,
            model=model,
            toolsets=_names(toolsets),
            workspace=workspace,
            max_concurrent=max_concurrent,
            show_tools=show_tools,
        )
        for adapter in adapters:
            console.print(f"[dim]•[/] {adapter.describe()}")
        gateway.run(host=host, port=port, report=_report)

    @group.command("check")
    def check(
        platforms: str = typer.Option("telegram,line", "--platforms", "-p"),
        allow_all: bool = typer.Option(False, "--allow-all"),
    ) -> None:
        """Report what would start, and what is more open than it looks."""
        resolver = SecretResolver()
        table = Table("platform", "transport", "credentials", "delivery")
        for name in _names(platforms):
            try:
                adapter_class = load(name)
            except HarnessError as exc:
                table.add_row(name, "—", f"[red]{exc}[/]", "—")
                continue
            missing = [key for key in _ENV_KEYS.get(name, ()) if not _has(resolver, key)]
            credentials = (
                "[red]missing " + ", ".join(missing) + "[/]" if missing else "[green]ok[/]"
            )
            caps = adapter_class.capabilities
            table.add_row(
                name,
                str(adapter_class.transport),
                credentials,
                "edit-in-place" if caps.edit_messages else f"chunked at {caps.max_chars}",
            )
        console.print(table)

        authorizer = _authorizer(_names(platforms), allow_all=allow_all)
        for warning in authorizer.warnings():
            console.print(f"[bold yellow]warning:[/] {warning}")
        if not authorizer.warnings():
            console.print("[green]closed by default[/]: only paired or allowlisted senders.")

    @group.command("pair")
    def pair(
        platform: str = typer.Argument(..., help="Which platform the code is for."),
        admin: bool = typer.Option(False, "--admin", help="Grant administrative commands."),
        note: str = typer.Option("", "--note", help="Who this is for. Recorded locally."),
    ) -> None:
        """Mint a single-use pairing code."""
        authorizer = Authorizer(platforms={platform: PlatformAuth()}, state_path=_pairing_state())
        code = authorizer.issue_code(platform, now=datetime.now(UTC), note=note, make_admin=admin)
        console.print(f"Pairing code for [bold]{platform}[/]: [bold cyan]{code.code}[/]")
        console.print(
            "Have them send [bold]/pair "
            f"{code.code}[/] as a direct message. "
            f"Single use, expires {code.expires_at:%Y-%m-%d %H:%M} UTC."
        )
        if admin:
            console.print("[yellow]This code grants administrative commands.[/]")

    @group.command("paired")
    def paired(platform: str = typer.Argument(...)) -> None:
        """List who has paired on one platform."""
        authorizer = Authorizer(state_path=_pairing_state())
        senders = sorted(authorizer.paired_senders(platform))
        if not senders:
            console.print(f"Nobody has paired on {platform}.")
            return
        for sender in senders:
            flag = " [yellow](admin)[/]" if authorizer.is_admin(platform, sender) else ""
            console.print(f"  {sender}{flag}")


def _report(message: str) -> None:
    """Print one lifecycle note from the running gateway."""
    console.print(f"[dim]{message}[/]" if message.startswith(" ") else message)


# -- assembly ---------------------------------------------------------------------


def _assemble(
    adapters: list[PlatformAdapter],
    authorizer: Authorizer,
    *,
    model: str,
    toolsets: list[str],
    workspace: Path,
    max_concurrent: int,
    show_tools: bool,
) -> Gateway:
    """Wire adapters, router and agent factory into a runnable gateway."""
    sessions_root = harness_home() / "gateway"

    async def factory(key: str, _event: MessageEvent, sink: EventSink) -> AgentBundle:
        return build_agent(
            model=model,
            workspace=workspace,
            sessions_dir=sessions_root / _slug(key),
            toolsets=toolsets,
            surface="gateway",
            emit=sink,
            # Left at the surface default: allowlist-only. A chat user cannot
            # be prompted for approval mid-turn, so anything consequential must
            # already be written down by the operator.
            approval=ApprovalPolicy(surface="gateway"),
        )

    router = Router(
        adapters={a.platform: a for a in adapters},
        authorizer=authorizer,
        bundle_factory=factory,
        max_concurrent_runs=max_concurrent,
        show_tools=show_tools,
    )
    return Gateway(router=router, adapters=adapters)


def _build_adapters(names: list[str]) -> list[PlatformAdapter]:
    """Construct each requested adapter from environment credentials."""
    resolver = SecretResolver()
    adapters: list[PlatformAdapter] = []
    for name in names:
        try:
            adapters.append(_construct(name, resolver))
        except (AdapterError, CredentialError) as exc:
            # Named and skipped rather than fatal: one unconfigured platform
            # must not stop the configured ones from serving.
            console.print(f"[yellow]skipping {name}:[/] {exc}")
    if not adapters:
        console.print("[red]no platform could be started[/] — run `harn gateway check`")
        raise typer.Exit(code=1)
    return adapters


def _build_telegram(resolver: SecretResolver) -> PlatformAdapter:
    """Construct the Telegram adapter from environment credentials."""
    from harness_agentic.gateway.platforms.telegram import TelegramAdapter

    return TelegramAdapter(resolver.require("TELEGRAM_BOT_TOKEN"))


def _build_line(resolver: SecretResolver) -> PlatformAdapter:
    """Construct the LINE adapter from environment credentials."""
    from harness_agentic.gateway.platforms.line import LineAdapter

    return LineAdapter(
        channel_secret=resolver.require("LINE_CHANNEL_SECRET"),
        access_token=resolver.require("LINE_CHANNEL_ACCESS_TOKEN"),
    )


def _build_fake(_resolver: SecretResolver) -> PlatformAdapter:
    """Construct the in-memory adapter. For smoke-testing the wiring."""
    from harness_agentic.gateway.platforms.fake import FakeAdapter

    return FakeAdapter()


_BUILDERS: dict[str, Callable[[SecretResolver], PlatformAdapter]] = {
    "telegram": _build_telegram,
    "line": _build_line,
    "fake": _build_fake,
}
"""Adapters have different constructors because platforms need different
secrets. One builder each keeps that where the credential names already live,
rather than behind a cast that hides a signature mismatch until runtime."""


def _construct(name: str, resolver: SecretResolver) -> PlatformAdapter:
    """Build one adapter, reading its secrets by name."""
    builder = _BUILDERS.get(name)
    if builder is None:
        known = ", ".join(sorted(_BUILDERS))
        detail = f"unknown platform {name!r}; known platforms are {known}"
        raise AdapterError(detail)
    return builder(resolver)


def _authorizer(names: list[str], *, allow_all: bool) -> Authorizer:
    """Build the authorizer, reading allowlists from the environment."""
    resolver = SecretResolver()
    platforms = {}
    for name in names:
        platforms[name] = PlatformAuth(
            allowed_senders=_id_set(resolver, f"{name.upper()}_ALLOWED_SENDERS"),
            allowed_chats=_id_set(resolver, f"{name.upper()}_ALLOWED_CHATS"),
            admins=_id_set(resolver, f"{name.upper()}_ADMINS"),
        )
    return Authorizer(platforms=platforms, global_allow_all=allow_all, state_path=_pairing_state())


def _id_set(resolver: SecretResolver, key: str) -> frozenset[str]:
    """Read a comma-separated list of platform ids from the environment."""
    raw = resolver.get(key)
    if raw is None:
        return frozenset()
    return frozenset(part.strip() for part in raw.reveal().split(",") if part.strip())


def _has(resolver: SecretResolver, key: str) -> bool:
    """Whether a credential resolves, without revealing it."""
    return resolver.get(key) is not None


def _pairing_state() -> Path:
    """Where paired senders are recorded."""
    return harness_home() / "gateway" / "pairing.json"


def _names(raw: str) -> list[str]:
    """Split a comma-separated option, dropping blanks and duplicates."""
    return list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


def _slug(key: str) -> str:
    """Turn a session key into a directory name."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in key)


__all__ = ["KNOWN", "register"]
