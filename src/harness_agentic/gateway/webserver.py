"""Serving the webhook routes, without taking a web framework as a dependency.

The routes an adapter registers are plain callables over ``(body, headers)``,
which means the whole webhook path -- signature check, parsing, queueing -- is
testable by calling a function. This module is the small amount of glue that
puts those callables behind a socket, written directly against ASGI so that
``fastapi`` and ``starlette`` stay optional.

Two defaults here are security decisions rather than conveniences:

* **Bind to localhost.** A gateway is normally behind a reverse proxy that
  terminates TLS. Defaulting to ``0.0.0.0`` would put an unauthenticated HTTP
  endpoint on every interface the moment someone runs it on a VPS.
* **Reject oversized bodies before reading them.** A webhook endpoint is
  unauthenticated until the signature is checked, and the signature cannot be
  checked without the body -- so the body has to be bounded first.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from harness_agentic.gateway.types import Route, WebhookResponse

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 1_000_000
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787

Scope = MutableMapping[str, Any]
Receive = Callable[[], Any]
Send = Callable[[MutableMapping[str, Any]], Any]


@dataclass
class WebhookApp:
    """An ASGI application over a fixed set of adapter routes."""

    routes: Sequence[Route]
    max_body_bytes: int = MAX_BODY_BYTES
    health_path: str = "/healthz"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI event."""
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":  # pragma: no cover - websockets are not served here
            return
        response = await self.dispatch(
            scope.get("path", "/"),
            scope.get("method", "GET"),
            _headers(scope),
            await self._read_body(receive),
        )
        await _respond(send, response)

    async def dispatch(
        self, path: str, method: str, headers: Mapping[str, str], body: bytes | None
    ) -> WebhookResponse:
        """Route one request. Public so tests can drive it without a socket."""
        if path == self.health_path:
            return WebhookResponse(status=200, body=b"ok")
        if body is None:
            return WebhookResponse(status=413, body=b"payload too large")

        for route in self.routes:
            if route.path != path:
                continue
            if method.upper() not in route.methods:
                return WebhookResponse(status=405, body=b"method not allowed")
            try:
                return await route.handler(body, headers)
            except Exception:
                # The platform will redeliver on a 5xx, which is what we want
                # for an unexpected failure -- the event was not handled.
                log.exception("webhook handler failed for %s", path)
                return WebhookResponse(status=500, body=b"handler failed")
        return WebhookResponse(status=404, body=b"no such webhook")

    async def _read_body(self, receive: Receive) -> bytes | None:
        """Read the request body, or ``None`` if it exceeds the cap."""
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunk: bytes = message.get("body", b"")
            total += len(chunk)
            if total > self.max_body_bytes:
                return None
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        return b"".join(chunks)

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        """Answer the ASGI lifespan handshake until the server goes away.

        Cancellation is a normal ending, not a fault. The gateway stops its
        server by cancelling the task, and letting that propagate made uvicorn
        log ``ERROR: Exception in 'lifespan' protocol`` with a traceback on every
        clean Ctrl-C. An operator who is shown a stack trace each time they stop
        the process correctly learns to ignore this one, and then ignores the
        next one too.
        """
        try:
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        except asyncio.CancelledError:
            return


async def serve(
    app: WebhookApp, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> None:  # pragma: no cover - requires a real socket
    """Run the app under uvicorn, which is an optional dependency.

    Imported here rather than at module scope so that the gateway's webhook
    machinery -- and its tests -- work in an install without a web server.
    """
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        detail = "serving webhooks needs the gateway extra: pip install 'harness-agentic[gateway]'"
        raise RuntimeError(detail) from exc

    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "binding webhooks to %s exposes them beyond this machine; "
            "terminate TLS in front of it and keep signature verification on",
            host,
        )
    config = uvicorn.Config(app, host=host, port=port, log_level="info", lifespan="on")
    await uvicorn.Server(config).serve()


def _headers(scope: Scope) -> dict[str, str]:
    """ASGI header pairs as a string dict."""
    return {
        key.decode("latin-1"): value.decode("latin-1") for key, value in scope.get("headers", [])
    }


async def _respond(send: Send, response: WebhookResponse) -> None:
    """Write one ASGI response."""
    await send(
        {
            "type": "http.response.start",
            "status": response.status,
            "headers": [
                (b"content-type", response.content_type.encode()),
                (b"content-length", str(len(response.body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": response.body})
