"""The HTTP API behind the web UI.

Written against ASGI directly, like the webhook server, so ``fastapi`` stays
optional. The routes are plain functions over ``(body, headers)`` and the ASGI
layer is glue, which means the whole API -- auth, approval, turn dispatch -- is
testable by calling functions.

Three security decisions, and none of them is adjustable by accident.

**A token is required, always.** Not optional-with-a-warning: this endpoint runs
tools. A missing token is a refusal at startup, not a permissive default, because
"I'll add auth before I expose it" is how an agent ends up on the public internet
running shell commands.

**Localhost unless a token is set and the operator asked.** Binding to
``0.0.0.0`` needs both.

**Approvals go through the same policy as the CLI.** A web approval is a
:class:`~harness_agentic.tools.approval.ApprovalPolicy` prompt answered from a
browser rather than a terminal -- not a second gate with its own rules, which
would inevitably drift into being the weaker one.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from harness_agentic.api.events import (
    HEARTBEAT_COMMENT,
    Frame,
    Stream,
    preamble,
    sink_for,
)
from harness_agentic.errors import HarnessError

if TYPE_CHECKING:
    from harness_agentic.tools.spec import ApprovalRequest

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8788
MAX_BODY_BYTES = 200_000
APPROVAL_TIMEOUT_S = 300.0
"""How long a tool waits for a browser to answer. Past this it is refused --
denying is the safe outcome, and a turn blocked forever is worse than a turn
that reports it could not get permission."""


class ApiError(HarnessError):
    """The API was asked to do something it will not."""


@dataclass(frozen=True, slots=True)
class Response:
    """One HTTP response."""

    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"
    stream: Stream | None = None
    """Set for the SSE endpoint, which the ASGI layer serves incrementally."""

    @classmethod
    def json(cls, payload: object, *, status: int = 200) -> Response:
        """A JSON response."""
        return cls(status=status, body=json.dumps(payload).encode())

    @classmethod
    def error(cls, message: str, *, status: int = 400) -> Response:
        """A JSON error."""
        return cls.json({"error": message}, status=status)


@dataclass
class Pending:
    """One approval a tool is blocked on."""

    request: ApprovalRequest
    event: threading.Event = field(default_factory=threading.Event)
    granted: bool = False
    answered_by: str = ""

    def resolve(self, *, granted: bool, by: str) -> None:
        """Answer it, releasing the waiting tool."""
        self.granted = granted
        self.answered_by = by
        self.event.set()

    def wait(self, timeout: float = APPROVAL_TIMEOUT_S) -> bool:
        """Block until answered or the timeout passes.

        A timeout denies. The alternative -- waiting forever -- means a turn that
        nobody notices is stuck, holding a worker thread until the process
        restarts.
        """
        if not self.event.wait(timeout):
            log.warning("approval for %s timed out; refusing", self.request.tool)
            return False
        return self.granted


@dataclass
class Approvals:
    """The approvals currently waiting for a browser to answer them."""

    pending: dict[str, Pending] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _counter: int = 0

    def open(self, request: ApprovalRequest) -> tuple[str, Pending]:
        """Register a request and return its id."""
        with self._lock:
            self._counter += 1
            identifier = f"a{self._counter}"
            waiting = Pending(request=request)
            self.pending[identifier] = waiting
        return identifier, waiting

    def resolve(self, identifier: str, *, granted: bool, by: str) -> bool:
        """Answer one. Returns whether it was still waiting."""
        with self._lock:
            waiting = self.pending.pop(identifier, None)
        if waiting is None:
            return False
        waiting.resolve(granted=granted, by=by)
        return True

    def listed(self) -> list[dict[str, object]]:
        """Everything outstanding, for a browser that just connected."""
        with self._lock:
            return [
                {
                    "id": identifier,
                    "tool": waiting.request.tool,
                    "danger": int(waiting.request.danger),
                    "summary": waiting.request.summary,
                }
                for identifier, waiting in self.pending.items()
            ]


TurnRunner = Callable[[str, str], None]
"""Runs a turn for ``(session_id, prompt)``. Supplied by the caller, because the
API must not decide how an agent is built."""


@dataclass
class Api:
    """The routes, independent of any HTTP server."""

    token: str
    run_turn: TurnRunner
    approvals: Approvals = field(default_factory=Approvals)
    streams: dict[str, Stream] = field(default_factory=dict)
    sessions: Callable[[], list[dict[str, object]]] | None = None
    include_thinking: bool = False
    index_html: str = ""
    _turns: list[threading.Thread] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Refuse to exist without a token."""
        if not self.token:
            detail = (
                "the web API requires a token; generate one with "
                "`harn web token` and pass it as HARNESS_WEB_TOKEN"
            )
            raise ApiError(detail)

    # -- routing ---------------------------------------------------------------

    def dispatch(  # noqa: PLR0911
        self, method: str, path: str, headers: Mapping[str, str], body: bytes
    ) -> Response:
        """Route one request. Public so tests need no socket.

        One return per route. A dispatch table would be shorter and would put the
        auth check somewhere a new route could be added without passing it.
        """
        if path in ("/", "/index.html") and method == "GET":
            # The page itself is not behind the token: it contains no data, and
            # requiring a header to fetch HTML means no browser can load it.
            return Response(
                status=200,
                body=(self.index_html or INDEX_HTML).encode(),
                content_type="text/html",
            )
        if path == "/healthz":
            return Response.json({"ok": True})

        if not self.authorized(headers, path):
            # 401 with no detail. Telling an unauthenticated caller which part of
            # its credential was wrong is telling an attacker.
            return Response.error("unauthorized", status=401)

        verb = method.upper()
        route = path.split("?", maxsplit=1)[0]
        if verb == "GET" and route == "/api/sessions":
            return Response.json({"sessions": self.sessions() if self.sessions else []})
        if verb == "GET" and route == "/api/approvals":
            return Response.json({"approvals": self.approvals.listed()})
        if verb == "POST" and route.startswith("/api/approvals/"):
            return self._resolve_approval(route.rsplit("/", 1)[-1], body)
        if verb == "POST" and route.endswith("/turns"):
            return self._start_turn(_session_of(route), body)
        if verb == "GET" and route.endswith("/events"):
            return self._subscribe(_session_of(route))
        return Response.error("no such endpoint", status=404)

    def authorized(self, headers: Mapping[str, str], path: str) -> bool:
        """Whether a request carries the token.

        Compared in constant time. A token check with ``==`` leaks the correct
        prefix, and this token guards a shell.
        """
        provided = _bearer(headers) or _query_token(path)
        return bool(provided) and hmac.compare_digest(provided, self.token)

    # -- endpoints -------------------------------------------------------------

    def _start_turn(self, session: str, body: bytes) -> Response:
        """Begin a turn on a worker thread and answer immediately."""
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            return Response.error("the body is not valid JSON")
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            return Response.error("prompt is required")

        stream = self.streams.setdefault(session, Stream())
        # On a thread, and answered before the turn finishes: a turn takes
        # minutes and an HTTP request that waits for it times out in every proxy
        # between here and the browser.
        thread = threading.Thread(
            target=self._run_guarded,
            args=(session, prompt),
            name=f"web-turn:{session}",
            daemon=True,
        )
        thread.start()
        # Tracked so a shutting-down server can wait for in-flight turns rather
        # than dropping a half-finished answer on the floor.
        self._turns = [live for live in self._turns if live.is_alive()]
        self._turns.append(thread)
        del stream
        return Response.json({"accepted": True, "session": session}, status=202)

    def _run_guarded(self, session: str, prompt: str) -> None:
        """Run a turn, reporting a failure onto the stream rather than losing it."""
        try:
            self.run_turn(session, prompt)
        except Exception as exc:
            log.exception("web turn failed for %s", session)
            stream = self.streams.get(session)
            if stream is not None:
                stream.push(Frame("notice", {"level": "error", "message": str(exc)}))

    def _subscribe(self, session: str) -> Response:
        """Hand back the stream for a session, creating it if needed."""
        return Response(
            status=200,
            content_type="text/event-stream",
            stream=self.streams.setdefault(session, Stream()),
        )

    def _resolve_approval(self, identifier: str, body: bytes) -> Response:
        """Answer a pending approval from the browser."""
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            return Response.error("the body is not valid JSON")
        granted = bool(payload.get("granted", False))
        who = str(payload.get("by") or "web")
        if not self.approvals.resolve(identifier, granted=granted, by=who):
            return Response.error("that approval is no longer pending", status=409)
        return Response.json({"granted": granted})

    # -- wiring ----------------------------------------------------------------

    def wait_for_turns(self, timeout: float = 30.0) -> bool:
        """Block until every in-flight turn has finished, or the timeout passes.

        Returns whether they all finished. Used by shutdown, and by tests that
        need to observe what a turn put on the stream.
        """
        deadline = timeout
        for thread in list(self._turns):
            thread.join(timeout=deadline)
        return not any(thread.is_alive() for thread in self._turns)

    def sink(self, session: str) -> Any:
        """The event sink for one session's stream."""
        stream = self.streams.setdefault(session, Stream())
        return sink_for(stream, include_thinking=self.include_thinking)

    def prompter(self, session: str) -> Callable[[ApprovalRequest], bool]:
        """A prompter that asks the browser and blocks until it answers.

        Handed to :class:`~harness_agentic.tools.approval.ApprovalPolicy`, so a
        web approval is the same gate the CLI uses with a different way of
        asking -- not a second gate that drifts into being the weaker one.
        """

        def ask(request: ApprovalRequest) -> bool:
            identifier, waiting = self.approvals.open(request)
            stream = self.streams.setdefault(session, Stream())
            stream.push(
                Frame(
                    "approval_requested",
                    {
                        "id": identifier,
                        "tool": request.tool,
                        "danger": int(request.danger),
                        "summary": request.summary,
                        "detail": request.detail,
                    },
                )
            )
            granted = waiting.wait()
            stream.push(
                Frame(
                    "approval_resolved",
                    {"id": identifier, "granted": granted, "by": waiting.answered_by or "timeout"},
                )
            )
            return granted

        return ask


def new_token() -> str:
    """A fresh API token."""
    return secrets.token_urlsafe(32)


def _bearer(headers: Mapping[str, str]) -> str:
    """The bearer token from an Authorization header."""
    for key, value in headers.items():
        if key.lower() == "authorization" and value.lower().startswith("bearer "):
            return value[7:].strip()
    return ""


def _query_token(path: str) -> str:
    """A token from the query string.

    Only for ``EventSource``, which cannot set headers. It is a real cost --
    query strings end up in access logs -- so it is the one concession, and the
    token is single-purpose and revocable.
    """
    _, _, query = path.partition("?")
    for part in query.split("&"):
        key, _, value = part.partition("=")
        if key == "token":
            return value
    return ""


def _session_of(path: str) -> str:
    """The session id from ``/api/sessions/<id>/turns``."""
    parts = path.split("?", maxsplit=1)[0].strip("/").split("/")
    return parts[2] if len(parts) > 2 else "default"  # noqa: PLR2004


INDEX_HTML = """<!doctype html>
<meta charset="utf-8"><title>Harness-Agentic</title>
<style>
 body{font:14px/1.5 ui-monospace,monospace;margin:0;background:#111;color:#ddd}
 header{padding:.6rem 1rem;background:#000;border-bottom:1px solid #333}
 #log{padding:1rem;white-space:pre-wrap}
 .tool{color:#7cf}.err{color:#f77}.note{color:#fc7}.ask{color:#7f7}
 form{display:flex;gap:.5rem;padding:.6rem 1rem;background:#000;
      border-top:1px solid #333;position:sticky;bottom:0}
 input,button{font:inherit;padding:.4rem;background:#222;color:#ddd;border:1px solid #444}
 input{flex:1}
</style>
<header>Harness-Agentic &mdash; paste your token below</header>
<div id="log"></div>
<form id="f">
  <input id="token" type="password" placeholder="token" size="20">
  <input id="q" placeholder="ask something" autofocus>
  <button>send</button>
</form>
<script>
const log = document.getElementById("log");
function line(text, cls) {
  const el = document.createElement("div");
  if (cls) el.className = cls;
  el.textContent = text;
  log.appendChild(el);
  window.scrollTo(0, document.body.scrollHeight);
}
let answer = null;
let source = null;
function listen(token) {
  if (source) source.close();
  source = new EventSource("/api/sessions/default/events?token=" + encodeURIComponent(token));
  source.addEventListener("text", e => {
    const t = JSON.parse(e.data).text;
    if (!answer) { answer = document.createElement("div"); log.appendChild(answer); }
    answer.textContent += t;
    window.scrollTo(0, document.body.scrollHeight);
  });
  source.addEventListener("tool_started", e => {
    const d = JSON.parse(e.data);
    line("\\u2699 " + d.tool + ": " + d.summary, "tool");
  });
  source.addEventListener("notice", e => {
    const d = JSON.parse(e.data);
    line(d.message, d.level === "error" ? "err" : "note");
  });
  source.addEventListener("approval_requested", e => {
    const d = JSON.parse(e.data);
    line("\\ud83d\\udd12 " + d.tool + ": " + d.summary, "ask");
    const ok = confirm(d.tool + "\\n\\n" + d.summary + "\\n\\n" + (d.detail || ""));
    fetch("/api/approvals/" + d.id, {
      method: "POST",
      headers: {"Authorization": "Bearer " + token, "Content-Type": "application/json"},
      body: JSON.stringify({granted: ok, by: "web"})
    });
  });
  source.addEventListener("turn_finished", e => {
    answer = null;
    const d = JSON.parse(e.data);
    if (d.reason !== "completed") line("[" + d.reason + "]", "note");
  });
  source.addEventListener("gap", e => {
    line("[" + JSON.parse(e.data).dropped + " events dropped]", "note");
  });
}
document.getElementById("token").addEventListener("change", e => listen(e.target.value));
document.getElementById("f").addEventListener("submit", async e => {
  e.preventDefault();
  const token = document.getElementById("token").value;
  const q = document.getElementById("q");
  if (!q.value.trim()) return;
  if (!source) listen(token);
  line("> " + q.value);
  await fetch("/api/sessions/default/turns", {
    method: "POST",
    headers: {"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    body: JSON.stringify({prompt: q.value})
  });
  q.value = "";
});
</script>
"""
"""A single-file UI, shipped in the wheel.

Deliberately small and dependency-free. A build step at install time is a build
step that breaks on somebody's machine, and the point of this surface is to
prove the event stream works from a browser -- not to be a product."""


@dataclass
class AsgiApp:
    """The ASGI adapter. Thin by design: the routing above is what is tested."""

    api: Api

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Handle one ASGI event."""
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":  # pragma: no cover - no websockets here
            return

        path = scope.get("path", "/")
        if scope.get("query_string"):
            path = f"{path}?{scope['query_string'].decode('latin-1')}"
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers", [])}
        body = await self._read(receive)
        if body is None:
            await _send_simple(send, 413, b"payload too large", "text/plain")
            return

        response = self.api.dispatch(scope.get("method", "GET"), path, headers, body)
        if response.stream is not None:
            await self._serve_stream(response.stream, send)
            return
        await _send_simple(send, response.status, response.body, response.content_type)

    async def _serve_stream(self, stream: Stream, send: Any) -> None:
        """Hold the connection open, flushing frames and heartbeats."""
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    # Both required in practice: nginx buffers an SSE response
                    # into oblivion without the first, and a caching proxy will
                    # happily replay a finished stream without the second.
                    (b"cache-control", b"no-cache"),
                    (b"x-accel-buffering", b"no"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": preamble().encode(), "more_body": True})
        idle = 0.0
        while not stream.closed:
            chunk = "".join(stream.drain())
            if chunk:
                idle = 0.0
                await send(
                    {"type": "http.response.body", "body": chunk.encode(), "more_body": True}
                )
            else:
                await asyncio.sleep(0.1)
                idle += 0.1
                if idle >= 15.0:  # noqa: PLR2004 - well inside every proxy's idle timeout
                    idle = 0.0
                    await send(
                        {
                            "type": "http.response.body",
                            "body": HEARTBEAT_COMMENT.encode(),
                            "more_body": True,
                        }
                    )
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _read(self, receive: Any) -> bytes | None:
        """Read the body, or ``None`` if it is over the cap."""
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunk: bytes = message.get("body", b"")
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                return None
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        return b"".join(chunks)

    async def _lifespan(self, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return


async def _send_simple(send: Any, status: int, body: bytes, content_type: str) -> None:
    """Write one complete response."""
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", content_type.encode()),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def serve(api: Api, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run the API under uvicorn.

    Binding beyond localhost is allowed only because a token is mandatory --
    :class:`Api` refuses to exist without one -- and it still says so out loud,
    because an operator who did not mean to expose this should find out now.
    """
    try:
        import uvicorn  # noqa: PLC0415 - the [gateway] extra, imported on demand
    except ModuleNotFoundError as exc:  # pragma: no cover - needs the extra absent
        detail = "the web UI needs the gateway extra: pip install 'harness-agentic[gateway]'"
        raise ApiError(detail) from exc

    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "the web API is bound to %s, reachable beyond this machine. The token is "
            "the only thing in front of it; terminate TLS in front of that.",
            host,
        )
    uvicorn.run(AsgiApp(api), host=host, port=port, log_level="warning")
