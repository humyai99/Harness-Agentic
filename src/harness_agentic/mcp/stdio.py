"""Talking to an MCP server over its standard input and output.

Stdio first because it covers most servers and needs no network, no port, and no
token. It also means the server is a **child process this framework starts**,
which is the part with teeth: launching a subprocess with an argument list the
operator configured is fine, launching one with a command string is a shell
injection waiting for a config file to contain a semicolon. So the command is
always a list, never a string, and it is never passed through a shell.

The environment is an allowlist, not the parent's environment. A server that
needs a token gets that one variable named in config; it does not get every
credential the agent process happens to hold, and it does not get to read the
provider keys out of ``os.environ``.

Reading runs on a background thread and stderr is drained separately, because a
server that logs to stderr fills the pipe and blocks forever if nobody reads it
-- the classic subprocess deadlock, and it presents as "the MCP server hung".
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from harness_agentic.errors import HarnessError
from harness_agentic.mcp.protocol import Notification, Reply, Request, Session

log = logging.getLogger(__name__)

READ_TIMEOUT_S = 30.0
SHUTDOWN_GRACE_S = 5.0
MAX_STDERR_LINES = 50
"""Kept for diagnostics. A server that logs a megabyte is not worth storing."""


class McpError(HarnessError):
    """An MCP server could not be started, or stopped answering."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """How to start one server."""

    name: str
    command: tuple[str, ...]
    """An argument list. Never a string, and never run through a shell."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Extra variables, added to the allowlisted subset of the parent's."""
    env_passthrough: tuple[str, ...] = ()
    """Names to copy from this process's environment. Nothing else is copied."""
    cwd: Path | None = None
    timeout_s: float = READ_TIMEOUT_S
    enabled: bool = True

    def resolved_env(self) -> dict[str, str]:
        """The child's environment: a minimal base, an allowlist, then extras.

        Built up rather than filtered down. Starting from ``os.environ`` and
        removing what looks sensitive means every new credential name added
        anywhere is leaked until somebody remembers to add it to the deny list.
        """
        base = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SystemRoot")
            if key in os.environ
        }
        base.update({key: os.environ[key] for key in self.env_passthrough if key in os.environ})
        base.update(self.env)
        return base

    def describe(self) -> str:
        """One readable line for ``harn mcp list``."""
        state = "" if self.enabled else " [disabled]"
        return f"{self.name}: {' '.join(self.command)}{state}"


@dataclass
class StdioServer:
    """A running MCP server, spoken to over pipes."""

    config: ServerConfig
    session: Session = field(default_factory=Session)
    _process: subprocess.Popen[str] | None = None
    _inbox: queue.Queue[Reply | McpError] = field(default_factory=queue.Queue)
    _stderr: list[str] = field(default_factory=list)
    _readers: list[threading.Thread] = field(default_factory=list)
    _stopping: threading.Event = field(default_factory=threading.Event)

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> list[str]:
        """Launch the server and complete the handshake. Returns any warnings."""
        if not self.config.command:
            detail = f"{self.config.name} has no command configured"
            raise McpError(detail)
        try:
            self._process = subprocess.Popen(  # noqa: S603 - an operator-configured argv
                list(self.config.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.config.resolved_env(),
                cwd=str(self.config.cwd) if self.config.cwd else None,
                text=True,
                bufsize=1,
                # No shell, ever. A command list plus shell=True is how a
                # semicolon in a config file becomes arbitrary execution.
                shell=False,
            )
        except OSError as exc:
            detail = f"could not start {self.config.name}: {exc}"
            raise McpError(detail) from exc

        self._spawn_readers()
        reply = self._exchange(self.session.initialize())
        warnings = self.session.on_initialize_result(reply.unwrap())
        self.send(Session.initialized_notification())
        return warnings

    def _spawn_readers(self) -> None:
        """Drain stdout and stderr on their own threads.

        Both, always. A server that logs to stderr fills that pipe and blocks
        forever if nobody reads it, and the symptom is indistinguishable from a
        hung server.
        """
        assert self._process is not None  # noqa: S101 - set by start()
        for stream, handler in (
            (self._process.stdout, self._read_stdout),
            (self._process.stderr, self._read_stderr),
        ):
            if stream is None:  # pragma: no cover - pipes are always requested
                continue
            thread = threading.Thread(
                target=handler, args=(stream,), name=f"mcp:{self.config.name}", daemon=True
            )
            thread.start()
            self._readers.append(thread)

    def _read_stdout(self, stream: IO[str]) -> None:
        """Parse each line into a reply and queue it."""
        for line in stream:
            if self._stopping.is_set():
                return
            text = line.strip()
            if not text:
                continue
            try:
                self._inbox.put(self.session.on_message(text))
            except Exception as exc:
                self._inbox.put(McpError(f"{self.config.name}: {exc}"))
        if not self._stopping.is_set():
            self._inbox.put(McpError(f"{self.config.name} closed its output stream"))

    def _read_stderr(self, stream: IO[str]) -> None:
        """Keep the last few lines for diagnostics, and drop the rest."""
        for line in stream:
            if self._stopping.is_set():
                return
            self._stderr.append(line.rstrip())
            del self._stderr[:-MAX_STDERR_LINES]

    def stop(self) -> None:
        """Terminate the server and close every pipe.

        All three pipes, explicitly. Closing only stdin leaves stdout and stderr
        for the garbage collector, and a process that connects and disconnects
        servers over weeks -- a gateway -- runs out of file descriptors long
        before anything looks wrong.
        """
        self._stopping.set()
        process = self._process
        self._process = None
        if process is None:
            return

        with contextlib.suppress(OSError):
            if process.stdin is not None:
                process.stdin.close()
        try:
            process.terminate()
            process.wait(timeout=SHUTDOWN_GRACE_S)
        except (subprocess.TimeoutExpired, OSError):
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                process.wait(timeout=SHUTDOWN_GRACE_S)

        # After the child is gone the reader threads see EOF and return; joining
        # them briefly means the pipes are not closed from under a live read.
        for reader in self._readers:
            reader.join(timeout=1.0)
        self._readers.clear()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()

    @property
    def alive(self) -> bool:
        """Whether the process is still running."""
        return self._process is not None and self._process.poll() is None

    def diagnostics(self) -> str:
        """Whatever the server said on stderr. What ``harn mcp check`` prints."""
        return "\n".join(self._stderr[-MAX_STDERR_LINES:])

    # -- messages --------------------------------------------------------------

    def send(self, message: Request | Notification) -> None:
        """Write one message to the server."""
        if self._process is None or self._process.stdin is None:
            detail = f"{self.config.name} is not running"
            raise McpError(detail)
        try:
            self._process.stdin.write(message.encode() + "\n")
            self._process.stdin.flush()
        except OSError as exc:
            detail = f"{self.config.name} stopped accepting input: {exc}"
            raise McpError(detail) from exc

    def _exchange(self, request: Request) -> Reply:
        """Send a request and wait for the reply with a matching id.

        Matching on id rather than taking the next line: a server may interleave
        notifications and its own requests with the answer, and reading the next
        line works against every toy server and fails against a real one.
        """
        self.send(request)
        deadline = self.config.timeout_s
        while True:
            try:
                item = self._inbox.get(timeout=deadline)
            except queue.Empty as exc:
                detail = f"{self.config.name} did not answer {request.method} in {deadline:.0f}s"
                raise McpError(detail) from exc
            if isinstance(item, McpError):
                raise item
            if item.kind in ("notification", "request"):
                self._on_unsolicited(item)
                continue
            if item.id == request.id:
                return item
            log.debug("%s: ignoring a reply to id %s", self.config.name, item.id)

    def _on_unsolicited(self, reply: Reply) -> None:
        """Handle what the server sends unprompted.

        ``tools/list_changed`` is the one that matters: a server may gain or lose
        tools while connected, and the bridge re-reads the list. A request from
        the server -- sampling, roots -- is refused politely rather than ignored,
        because a server waiting on an answer that never comes hangs.
        """
        if reply.kind == "notification":
            log.debug("%s notification: %s", self.config.name, reply.method)
            return
        if reply.id is None:
            return
        process = self._process
        if process is None or process.stdin is None:
            return
        error = {
            "jsonrpc": "2.0",
            "id": reply.id,
            "error": {"code": -32601, "message": f"{reply.method} is not supported"},
        }
        with contextlib.suppress(OSError):
            process.stdin.write(json.dumps(error) + "\n")
            process.stdin.flush()

    def request(self, method: str, params: Mapping[str, object] | None = None) -> dict[str, object]:
        """Make one request and return its result, raising on a server error."""
        return self._exchange(self.session.next_request(method, dict(params or {}))).unwrap()

    def __enter__(self) -> StdioServer:
        """Start the server on entry."""
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop it on exit, however the block ended."""
        self.stop()


def _names(raw: object) -> tuple[str, ...]:
    """A tuple of strings from a config value, ignoring anything else."""
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    return ()


def load_servers(entries: Sequence[Mapping[str, object]]) -> list[ServerConfig]:
    """Build server configs from ``[mcp.servers.*]`` tables.

    A string command is refused rather than split. ``npx -y @scope/server`` looks
    harmless and ``sh -c "..."`` does not, and the difference is not something a
    config loader should be deciding by heuristic.
    """
    configs: list[ServerConfig] = []
    for entry in entries:
        name = str(entry.get("name") or "")
        command = entry.get("command")
        if not name:
            detail = f"an MCP server entry has no name: {entry!r}"
            raise ValueError(detail)
        if isinstance(command, str) or not isinstance(command, (list, tuple)) or not command:
            # A string is refused rather than split. `npx -y @scope/server` looks
            # harmless and `sh -c "..."` does not, and that difference is not
            # something a config loader should decide by heuristic.
            detail = (
                f"{name}: command must be a non-empty list of arguments -- "
                f'write ["npx", "-y", "server"] rather than "npx -y server"'
            )
            raise TypeError(detail)
        timeout = entry.get("timeout_s")

        raw_env = entry.get("env")
        configs.append(
            ServerConfig(
                name=name,
                command=tuple(str(part) for part in command),
                env={str(k): str(v) for k, v in (raw_env or {}).items()}
                if isinstance(raw_env, dict)
                else {},
                env_passthrough=_names(entry.get("env_passthrough")),
                cwd=Path(str(entry["cwd"])) if entry.get("cwd") else None,
                timeout_s=float(timeout) if isinstance(timeout, (int, float)) else READ_TIMEOUT_S,
                enabled=bool(entry.get("enabled", True)),
            )
        )
    return configs
