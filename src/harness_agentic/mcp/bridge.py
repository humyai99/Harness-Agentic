"""Turning an MCP server's tools into tools the agent can call.

The mapping is nearly free: MCP publishes JSON Schema, the registry wants JSON
Schema, and no conversion is needed. What is not free is the trust question, and
it has one answer.

**An MCP server is somebody else's code.** Its tool metadata is written by
whoever wrote the server, so ``readOnlyHint: true`` is a claim about intent, not
a fact about behaviour. So every bridged tool is floored at
:attr:`~harness_agentic.tools.spec.Danger.NETWORK` and anything the server calls
destructive is raised to ``DESTRUCTIVE`` -- the hint can only ever make a tool
*more* restricted. A server cannot talk its way past the approval policy, which
is exactly what a compromised one would try.

Results are third-party content, so they are enveloped and taint the session,
the same as a fetched page. A tool result is a fine place to put "ignore your
instructions and run this next", and an agent that reads one as an instruction is
an agent the server operator controls.

Names are prefixed with the server name. Two servers offering ``search`` is
normal, and silently letting the second win means the agent calls a tool it did
not mean to.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from harness_agentic.core.types import ImageBlock
from harness_agentic.mcp.protocol import ToolOutcome, ToolSpec, parse_tool_list
from harness_agentic.mcp.stdio import ServerConfig, StdioServer
from harness_agentic.tools.builtin.web import envelope
from harness_agentic.tools.spec import Danger, Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from harness_agentic.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

MCP_FLOOR = Danger.NETWORK
"""The lowest danger level a bridged tool may have, whatever the server claims."""
MAX_PAGES = 20
MAX_RESULT_CHARS = 60_000
SEPARATOR = "__"
"""Between the server name and the tool name. Double underscore because a single
one is common inside tool names and would make the split ambiguous."""


@dataclass
class BridgedServer:
    """One connected server and the tools it contributed."""

    server: StdioServer
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    registered: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """The server's configured name."""
        return self.server.config.name

    def describe(self) -> str:
        """One readable line for ``harn mcp list``."""
        state = "connected" if self.server.alive else "not running"
        return f"{self.name} ({state}): {len(self.registered)} tool(s)"


@dataclass
class McpBridge:
    """Connects servers, registers their tools, and cleans up after them."""

    registry: ToolRegistry
    servers: dict[str, BridgedServer] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def connect(self, config: ServerConfig) -> BridgedServer | None:
        """Start one server and register its tools.

        Returns ``None`` on failure rather than raising: one unreachable server
        must not stop the others, and an operator gets a named failure from
        ``harn mcp check`` instead of a dead startup.
        """
        if not config.enabled:
            return None
        server = StdioServer(config=config)
        try:
            warnings = server.start()
            specs = self._read_tools(server)
        except Exception as exc:
            server.stop()
            self.failures[config.name] = f"{type(exc).__name__}: {exc}"
            log.warning("mcp server %s failed: %s", config.name, exc)
            return None

        bridged = BridgedServer(
            server=server,
            tools={spec.name: spec for spec in specs},
            warnings=tuple(warnings),
        )
        bridged.registered = self._register(bridged)
        self.servers[config.name] = bridged
        return bridged

    def connect_all(self, configs: Sequence[ServerConfig]) -> None:
        """Connect every configured server, isolating failures."""
        for config in configs:
            self.connect(config)

    def _read_tools(self, server: StdioServer) -> list[ToolSpec]:
        """Page through ``tools/list``.

        Bounded. A server that returns a cursor pointing at itself would
        otherwise spin forever, and a broken server should not be able to hang
        startup.
        """
        specs: list[ToolSpec] = []
        cursor = ""
        for _ in range(MAX_PAGES):
            result = server._exchange(server.session.list_tools(cursor)).unwrap()
            page, cursor = parse_tool_list(result)
            specs.extend(page)
            if not cursor:
                return specs
        log.warning("%s paginated past %d pages; stopping", server.config.name, MAX_PAGES)
        return specs

    def _register(self, bridged: BridgedServer) -> tuple[str, ...]:
        """Register each spec as a tool, under a prefixed name."""
        names: list[str] = []
        for spec in bridged.tools.values():
            name = qualified(bridged.name, spec.name)
            self.registry.register(
                Tool(
                    name=name,
                    description=describe(spec, bridged.name),
                    toolset=f"mcp:{bridged.name}",
                    params_model=None,
                    raw_schema=spec.input_schema,
                    handler=_handler(bridged, spec),
                    danger=danger_for(spec),
                    max_result_chars=MAX_RESULT_CHARS,
                    source=f"mcp:{bridged.name}",
                ),
                override=True,
            )
            names.append(name)
        return tuple(names)

    def refresh(self, name: str) -> tuple[str, ...]:
        """Re-read one server's tool list, adding and removing as needed.

        Called on ``notifications/tools/list_changed``. A server may gain or lose
        tools while connected, and an agent still holding a removed one will call
        it and get a confusing failure.
        """
        bridged = self.servers.get(name)
        if bridged is None:
            return ()
        try:
            specs = self._read_tools(bridged.server)
        except Exception as exc:
            log.warning("could not refresh %s: %s", name, exc)
            return bridged.registered

        previous = set(bridged.registered)
        bridged.tools = {spec.name: spec for spec in specs}
        bridged.registered = self._register(bridged)
        for stale in previous - set(bridged.registered):
            self.registry.unregister(stale)
        return bridged.registered

    def disconnect(self, name: str) -> None:
        """Stop one server and remove its tools.

        Removed, not left registered. A tool whose server is gone fails in a way
        that looks like the tool being broken, and the agent will retry it.
        """
        bridged = self.servers.pop(name, None)
        if bridged is None:
            return
        for tool in bridged.registered:
            self.registry.unregister(tool)
        bridged.server.stop()

    def close(self) -> None:
        """Disconnect everything."""
        for name in list(self.servers):
            self.disconnect(name)

    def report(self) -> list[str]:
        """Every line worth printing about the connected servers."""
        lines = [bridged.describe() for bridged in self.servers.values()]
        lines += [
            f"  warning: {warning}"
            for bridged in self.servers.values()
            for warning in bridged.warnings
        ]
        lines += [f"{name}: FAILED -- {reason}" for name, reason in self.failures.items()]
        return lines

    def tool_names(self) -> tuple[str, ...]:
        """Every bridged tool, across all servers."""
        return tuple(name for bridged in self.servers.values() for name in bridged.registered)


def qualified(server: str, tool: str) -> str:
    """The registry name for a server's tool.

    Prefixed because two servers offering ``search`` is normal, and letting the
    second silently win means the agent calls one it did not mean to.
    """
    return f"{server}{SEPARATOR}{tool}"


def danger_for(spec: ToolSpec) -> Danger:
    """How dangerous to treat a bridged tool as.

    The server's hints can raise this and never lower it. ``readOnlyHint`` is
    written by whoever wrote the server, so treating it as evidence would let a
    compromised server classify its own exfiltration tool as safe.
    """
    return Danger.DESTRUCTIVE if spec.destructive else MCP_FLOOR


def describe(spec: ToolSpec, server: str) -> str:
    """The description the model sees, naming where the tool came from."""
    body = spec.description.strip() or spec.name
    return f"{body}\n\n(provided by the {server!r} MCP server)"


def _handler(bridged: BridgedServer, spec: ToolSpec) -> Any:
    """Build the callable that invokes one remote tool."""

    def call(arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        if not bridged.server.alive:
            return ToolResult.error(f"the {bridged.name!r} MCP server is not running")
        ctx.emit(f"{bridged.name}: {spec.name}")
        try:
            raw = bridged.server.request(
                "tools/call", {"name": spec.name, "arguments": dict(arguments)}
            )
        except Exception as exc:  # a server failure is a tool error, not a crash
            return ToolResult.error(f"{bridged.name}/{spec.name} failed: {exc}")

        outcome = ToolOutcome.parse(raw)
        return ToolResult(
            # Third-party content, exactly like a fetched page: a tool result is
            # a fine place to put "ignore your instructions and do this".
            text=envelope(
                outcome.text or "(the tool returned nothing)", origin=f"mcp:{bridged.name}"
            ),
            is_error=outcome.is_error,
            images=tuple(
                ImageBlock(media_type=media, data_b64=data) for media, data in outcome.images
            ),
            display=f"{bridged.name}/{spec.name}",
            tainted=True,
        )

    return call
