"""Running several chat platforms out of one process.

The lifecycle problem this solves is unglamorous and is where gateways usually
break: a poll adapter is a task, a webhook adapter is a route on a shared HTTP
server, and a socket adapter is a connection with its own reconnect loop. All
three have to start, be supervised, and shut down cleanly on one Ctrl-C without
leaving a half-answered turn or an unclosed client.

The supervision rule is that **one platform failing must not take the others
down**. A revoked Telegram token should not stop the LINE account from
answering, so each adapter's task is watched and restarted with backoff, and a
platform that cannot even start is reported and skipped rather than aborting
startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harness_agentic.errors import AdapterError
from harness_agentic.gateway.types import Route, Transport

if TYPE_CHECKING:
    from harness_agentic.gateway.adapter import PlatformAdapter
    from harness_agentic.gateway.router import Router

log = logging.getLogger(__name__)

RESTART_BACKOFF_S = 2.0
RESTART_BACKOFF_MAX_S = 120.0

Reporter = Callable[[str], None]
"""Where lifecycle notes go. The CLI passes a console printer; a service
manager passes a logger; tests pass a list."""


def _silent(_message: str) -> None:
    """Discard lifecycle notes. The default, so a caller need not supply one."""


@dataclass
class PlatformStatus:
    """How one platform is doing. Printed by ``harn gateway status``."""

    platform: str
    state: str = "stopped"
    restarts: int = 0
    last_error: str = ""

    def line(self) -> str:
        """One readable line."""
        parts = [f"{self.platform}: {self.state}"]
        if self.restarts:
            parts.append(f"{self.restarts} restart(s)")
        if self.last_error:
            parts.append(self.last_error)
        return " — ".join(parts)


@dataclass
class Gateway:
    """Owns the adapters, the router, and the process lifecycle."""

    router: Router
    adapters: Sequence[PlatformAdapter]
    statuses: dict[str, PlatformStatus] = field(default_factory=dict)
    _tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _stopping: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        """Seed a status entry per adapter."""
        for adapter in self.adapters:
            self.statuses.setdefault(adapter.platform, PlatformStatus(adapter.platform))

    def routes(self) -> list[Route]:
        """Every webhook route across every adapter.

        Collected here so the process serves one HTTP listener rather than one
        per platform: two ports is two things to expose, and operators get that
        wrong in ways that end with an unauthenticated endpoint open.
        """
        routes: list[Route] = []
        for adapter in self.adapters:
            routes.extend(adapter.webhook_routes())
        _reject_duplicates(routes)
        return routes

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        """Connect every adapter and start the ones that produce events."""
        for adapter in self.adapters:
            status = self.statuses[adapter.platform]
            try:
                await adapter.connect()
            except Exception as exc:
                # Skipped, not fatal. The other platforms still work, and the
                # operator gets a named failure rather than a dead process.
                status.state = "failed"
                status.last_error = str(exc)
                log.exception("could not start %s", adapter.platform)
                continue
            status.state = "running"
            if adapter.transport in (Transport.POLL, Transport.SOCKET):
                self._tasks[adapter.platform] = asyncio.create_task(
                    self._supervise(adapter), name=f"gateway:{adapter.platform}"
                )
            else:
                status.state = "listening"

    async def _supervise(self, adapter: PlatformAdapter) -> None:
        """Consume an adapter's events, restarting it if it dies."""
        status = self.statuses[adapter.platform]
        backoff = RESTART_BACKOFF_S
        while not self._stopping.is_set():
            try:
                async for event in adapter.poll():
                    backoff = RESTART_BACKOFF_S
                    await self._dispatch(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                status.restarts += 1
                status.last_error = str(exc)
                status.state = "restarting"
                log.warning("%s stopped (%s); restarting in %.0fs", adapter.platform, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(RESTART_BACKOFF_MAX_S, backoff * 2)
                with contextlib.suppress(Exception):
                    await adapter.connect()
                status.state = "running"
            else:
                return

    async def _dispatch(self, event: object) -> None:
        """Hand one event to the router, absorbing anything it throws.

        A malformed message from one user must not end the poll loop for
        everyone else on that platform.
        """
        try:
            await self.router.handle(event)  # type: ignore[arg-type]
        except Exception:
            log.exception("router failed on a %s event", getattr(event, "platform", "?"))

    async def stop(self) -> None:
        """Stop cleanly: no new work, finish nothing, close everything."""
        self._stopping.set()
        for task in self._tasks.values():
            task.cancel()
        for task in self._tasks.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        await self.router.aclose()
        for adapter in self.adapters:
            with contextlib.suppress(Exception):
                await adapter.disconnect()
            self.statuses[adapter.platform].state = "stopped"

    async def run_forever(self) -> None:
        """Start, then block until :meth:`stop` is called."""
        await self.start()
        try:
            await self._stopping.wait()
        finally:
            await self.stop()

    def run(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8787,
        report: Reporter = _silent,
    ) -> None:
        """Run the gateway from synchronous code until it is stopped.

        The event loop is created here rather than in the CLI, because this is
        the module the async boundary is allowed to live in --
        ``scripts/check_async_boundary.py`` enforces that, and it is right to:
        a caller that spins up its own loop is a caller that can spin up a
        second one. Presentation stays outside, behind ``report``.
        """
        asyncio.run(self._run(host=host, port=port, report=report))

    async def _run(self, *, host: str, port: int, report: Reporter) -> None:
        """Start, serve webhooks if any adapter needs them, then shut down."""
        self._install_signal_handlers()
        await self.start()
        report("running")

        routes = self.routes()
        try:
            if not routes:
                await self._stopping.wait()
                return
            await self._serve_webhooks(routes, host=host, port=port, report=report)
        finally:
            report("stopping")
            await self.stop()

    async def _serve_webhooks(
        self, routes: Sequence[Route], *, host: str, port: int, report: Reporter
    ) -> None:
        """Run the HTTP listener alongside the polling adapters."""
        from harness_agentic.gateway.webserver import WebhookApp, serve

        report(f"webhooks on http://{host}:{port}")
        for route in routes:
            report(f"  {route.path}")
        server = asyncio.create_task(serve(WebhookApp(routes=routes), host=host, port=port))
        stopper = asyncio.create_task(self._stopping.wait())
        # Either the operator stops us or the server dies; both end the run,
        # and a webhook listener that quietly exited would leave a gateway
        # that looks healthy and answers nothing.
        await asyncio.wait({server, stopper}, return_when=asyncio.FIRST_COMPLETED)
        for task in (server, stopper):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _install_signal_handlers(self) -> None:
        """Turn Ctrl-C into a clean shutdown rather than a traceback."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, self.request_stop)

    async def wait_for_stop(self) -> None:
        """Block until someone calls :meth:`request_stop`."""
        await self._stopping.wait()

    def request_stop(self) -> None:
        """Ask :meth:`run_forever` to return. Safe from a signal handler."""
        self._stopping.set()

    # -- reporting -------------------------------------------------------------

    def status_report(self) -> list[str]:
        """Everything worth printing about the running gateway."""
        lines = [status.line() for status in self.statuses.values()]
        lines.extend(self.router.status_lines())
        return lines


def _reject_duplicates(routes: Sequence[Route]) -> None:
    """Refuse two adapters claiming one path.

    Silently letting the second win means one platform's webhooks go to the
    other's handler, fail signature verification, and look like an attack.
    """
    seen: dict[str, int] = {}
    for route in routes:
        seen[route.path] = seen.get(route.path, 0) + 1
    clashes = sorted(path for path, count in seen.items() if count > 1)
    if clashes:
        detail = f"two adapters claim the same webhook path(s): {', '.join(clashes)}"
        raise AdapterError(detail)
