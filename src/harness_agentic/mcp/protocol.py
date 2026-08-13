"""The Model Context Protocol, as message construction and parsing.

JSON-RPC 2.0 over a byte stream, and the parts worth being careful about are
unglamorous:

* **A response is matched by id, not by arrival order.** A server is free to
  answer out of order and to interleave notifications and its own requests. Code
  that reads the next line and assumes it is the answer works against every
  simple server and breaks against a real one.
* **A notification has no id and gets no reply.** Answering one is a protocol
  error, and some servers close the connection over it.
* **The server's own declared danger level is not evidence.** Tool metadata is
  written by whoever wrote the server, so ``readOnlyHint: true`` is a claim, not
  a guarantee. It can lower nothing.

This module holds no transport and no socket: messages in, messages out. The
stdio pipe is :mod:`harness_agentic.mcp.stdio`, and the mapping onto the tool
registry is :mod:`harness_agentic.mcp.bridge`.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
"""The revision this client speaks. Sent in `initialize` and checked against
what the server answers -- a mismatch is worth a warning, not a refusal, since
servers routinely lag and mostly still work."""

CLIENT_NAME = "harness-agentic"
JSONRPC = "2.0"


class ProtocolError(Exception):
    """The peer sent something that is not valid MCP."""


class ServerError(Exception):
    """The server answered a request with an error."""

    def __init__(self, code: int, message: str, data: object = None) -> None:
        """Carry the JSON-RPC error triple."""
        super().__init__(f"{message} (code {code})")
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True, slots=True)
class Request:
    """An outbound call awaiting a reply."""

    id: int
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def encode(self) -> str:
        """Serialize as one JSON-RPC line."""
        return json.dumps(
            {"jsonrpc": JSONRPC, "id": self.id, "method": self.method, "params": self.params}
        )


@dataclass(frozen=True, slots=True)
class Notification:
    """An outbound message with no reply expected."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def encode(self) -> str:
        """Serialize as one JSON-RPC line, with no id."""
        return json.dumps({"jsonrpc": JSONRPC, "method": self.method, "params": self.params})


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool as the server describes it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = False
    """The server's own claim. Read for display; never used to lower a danger
    level, because the server wrote it."""
    destructive: bool = False
    title: str = ""

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> ToolSpec:
        """Read one entry from a ``tools/list`` result."""
        name = str(raw.get("name") or "")
        if not name:
            detail = f"a tool entry has no name: {raw!r}"
            raise ProtocolError(detail)
        annotations = raw.get("annotations") or {}
        schema = raw.get("inputSchema") or {"type": "object", "properties": {}}
        return cls(
            name=name,
            description=str(raw.get("description") or raw.get("title") or name),
            input_schema=schema if isinstance(schema, dict) else {"type": "object"},
            read_only=bool(annotations.get("readOnlyHint", False)),
            destructive=bool(annotations.get("destructiveHint", False)),
            title=str(raw.get("title") or ""),
        )


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What a ``tools/call`` returned."""

    text: str
    is_error: bool = False
    images: tuple[tuple[str, str], ...] = ()
    """``(media_type, base64)`` pairs, ready for an ImageBlock.

    Left as base64 rather than decoded: the wire format and the block format are
    both base64, so decoding here would only be to re-encode later."""

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> ToolOutcome:
        """Flatten MCP's content blocks into text plus images.

        ``isError`` in the *result* means the tool failed while the call
        succeeded -- distinct from a JSON-RPC error, which means the call itself
        did not happen. Both end up as a tool error for the model, but only the
        second is worth logging as a client problem.
        """
        parts: list[str] = []
        images: list[tuple[str, str]] = []
        for block in _blocks(raw.get("content")):
            match block.get("type"):
                case "text":
                    parts.append(str(block.get("text") or ""))
                case "image":
                    data = str(block.get("data") or "")
                    media = str(block.get("mimeType") or "image/png")
                    if _is_base64(data):
                        images.append((media, data))
                    else:
                        # Validated here rather than left for the provider to
                        # reject: a 400 from the model API for a malformed image
                        # is much harder to trace back to one MCP server.
                        parts.append("[an image block was not valid base64]")
                case "resource":
                    resource = block.get("resource") or {}
                    parts.append(
                        str(resource.get("text") or f"[resource {resource.get('uri', '?')}]")
                    )
                case _:
                    parts.append(f"[unsupported content block: {block.get('type')!r}]")
        return cls(
            text="\n".join(part for part in parts if part).strip(),
            is_error=bool(raw.get("isError", False)),
            images=tuple(images),
        )


@dataclass
class Session:
    """Tracks request ids and what the server said about itself.

    Not a connection -- it builds messages and interprets replies. The transport
    calls :meth:`next_request` to get something to send and :meth:`on_message` to
    hand back whatever arrived.
    """

    client_name: str = CLIENT_NAME
    client_version: str = "1.0"
    protocol_version: str = PROTOCOL_VERSION
    server_name: str = ""
    server_version: str = ""
    server_protocol: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    initialized: bool = False
    _next_id: int = 1
    _pending: dict[int, str] = field(default_factory=dict)

    def next_request(self, method: str, params: dict[str, Any] | None = None) -> Request:
        """Build the next request, remembering what it was."""
        request = Request(id=self._next_id, method=method, params=params or {})
        self._pending[request.id] = method
        self._next_id += 1
        return request

    def initialize(self) -> Request:
        """The handshake request."""
        return self.next_request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {"roots": {"listChanged": False}, "sampling": {}},
                "clientInfo": {"name": self.client_name, "version": self.client_version},
            },
        )

    @staticmethod
    def initialized_notification() -> Notification:
        """Sent after the handshake, before anything else."""
        return Notification("notifications/initialized")

    def on_initialize_result(self, result: dict[str, Any]) -> list[str]:
        """Record what the server said. Returns any warnings worth surfacing."""
        info = result.get("serverInfo") or {}
        self.server_name = str(info.get("name") or "unknown")
        self.server_version = str(info.get("version") or "")
        self.server_protocol = str(result.get("protocolVersion") or "")
        caps = result.get("capabilities")
        self.capabilities = caps if isinstance(caps, dict) else {}
        self.initialized = True

        warnings: list[str] = []
        if self.server_protocol and self.server_protocol != self.protocol_version:
            warnings.append(
                f"{self.server_name} speaks MCP {self.server_protocol}, this client "
                f"speaks {self.protocol_version}; most things will still work"
            )
        if "tools" not in self.capabilities:
            warnings.append(f"{self.server_name} declares no tools capability")
        return warnings

    def list_tools(self, cursor: str = "") -> Request:
        """Ask for the tool list, or the next page of it."""
        return self.next_request("tools/list", {"cursor": cursor} if cursor else {})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Request:
        """Ask the server to run a tool."""
        return self.next_request("tools/call", {"name": name, "arguments": arguments})

    def on_message(self, raw: str) -> Reply:
        """Interpret one inbound line."""
        try:
            message = json.loads(raw)
        except ValueError as exc:
            detail = f"the server sent a line that is not JSON: {raw[:120]!r}"
            raise ProtocolError(detail) from exc
        if not isinstance(message, dict):
            detail = f"expected a JSON object, got {type(message).__name__}"
            raise ProtocolError(detail)

        if "method" in message and "id" not in message:
            return Reply(
                kind="notification",
                method=str(message["method"]),
                params=message.get("params") or {},
            )
        if "method" in message:
            # The server is asking *us* something -- sampling, or roots. Handled
            # by the caller; answering nothing at all makes a server hang.
            return Reply(
                kind="request",
                method=str(message["method"]),
                params=message.get("params") or {},
                id=_as_id(message.get("id")),
            )

        identifier = _as_id(message.get("id"))
        if identifier is None:
            detail = f"a response has no usable id: {raw[:120]!r}"
            raise ProtocolError(detail)
        method = self._pending.pop(identifier, "")

        if "error" in message:
            error = message["error"] or {}
            return Reply(
                kind="error",
                id=identifier,
                method=method,
                error=ServerError(
                    code=int(error.get("code", -1)),
                    message=str(error.get("message") or "unknown error"),
                    data=error.get("data"),
                ),
            )
        return Reply(
            kind="result", id=identifier, method=method, result=message.get("result") or {}
        )

    def outstanding(self) -> tuple[int, ...]:
        """Request ids still awaiting an answer."""
        return tuple(sorted(self._pending))


@dataclass(frozen=True, slots=True)
class Reply:
    """One interpreted inbound message."""

    kind: str
    """``result``, ``error``, ``notification``, or ``request``."""
    id: int | None = None
    method: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    error: ServerError | None = None

    def unwrap(self) -> dict[str, Any]:
        """The result, or the server's error raised."""
        if self.error is not None:
            raise self.error
        return self.result


def parse_tool_list(result: dict[str, Any]) -> tuple[list[ToolSpec], str]:
    """Read a ``tools/list`` result into specs plus a pagination cursor."""
    entries = result.get("tools")
    if not isinstance(entries, list):
        return [], ""
    specs = [ToolSpec.parse(entry) for entry in entries if isinstance(entry, dict)]
    return specs, str(result.get("nextCursor") or "")


def _blocks(raw: object) -> list[dict[str, Any]]:
    """The content blocks from a result, ignoring anything malformed."""
    if not isinstance(raw, list):
        return []
    return [block for block in raw if isinstance(block, dict)]


def _is_base64(data: str) -> bool:
    """Whether a string decodes as base64."""
    try:
        base64.b64decode(data, validate=True)
    except (ValueError, TypeError):
        return False
    return bool(data)


def _as_id(raw: object) -> int | None:
    """JSON-RPC allows string ids; this client only issues integers."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None
