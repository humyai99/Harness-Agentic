"""The MCP client: protocol, transport, and the trust rules on the bridge.

The protocol tests are pure -- messages in, messages out -- which is the point
of keeping the transport out of that module. The one integration test starts a
*real* subprocess speaking JSON-RPC on stdio, because the failures that matter
there (a stderr pipe filling and deadlocking, a reply arriving out of order) do
not reproduce against a mock.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
from pathlib import Path

import pytest

from harness_agentic.core.types import ToolUseBlock
from harness_agentic.mcp.bridge import (
    MCP_FLOOR,
    McpBridge,
    danger_for,
    qualified,
)
from harness_agentic.mcp.protocol import (
    ProtocolError,
    ServerError,
    Session,
    ToolOutcome,
    ToolSpec,
    parse_tool_list,
)
from harness_agentic.mcp.stdio import McpError, ServerConfig, StdioServer, load_servers
from harness_agentic.tools.approval import always_deny
from harness_agentic.tools.dispatch import ToolExecutor
from harness_agentic.tools.registry import ToolRegistry
from harness_agentic.tools.spec import Danger, Tool, ToolResult

# -- protocol ---------------------------------------------------------------------


def test_the_handshake_carries_the_protocol_version() -> None:
    session = Session()
    request = session.initialize()
    payload = json.loads(request.encode())

    assert payload["method"] == "initialize"
    assert payload["params"]["protocolVersion"] == session.protocol_version
    assert payload["params"]["clientInfo"]["name"] == "harness-agentic"
    assert payload["jsonrpc"] == "2.0"


def test_a_version_mismatch_warns_rather_than_refuses() -> None:
    # Servers routinely lag and mostly still work; refusing would make the
    # client useless against most of what exists.
    session = Session()
    session.initialize()
    warnings = session.on_initialize_result(
        {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "old-server", "version": "0.1"},
            "capabilities": {"tools": {}},
        }
    )
    assert session.initialized
    assert session.server_name == "old-server"
    assert any("2024-11-05" in warning for warning in warnings)


def test_a_server_with_no_tools_capability_is_flagged() -> None:
    session = Session()
    warnings = session.on_initialize_result({"serverInfo": {"name": "s"}, "capabilities": {}})
    assert any("no tools capability" in warning for warning in warnings)


def test_a_reply_is_matched_by_id_not_by_arrival_order() -> None:
    session = Session()
    first = session.next_request("tools/list")
    second = session.next_request("tools/call")
    assert session.outstanding() == (first.id, second.id)

    # The server answers the second request first, which it is free to do.
    later = session.on_message(json.dumps({"jsonrpc": "2.0", "id": second.id, "result": {"ok": 1}}))
    assert later.id == second.id
    assert later.method == "tools/call"
    assert session.outstanding() == (first.id,)


def test_a_notification_is_recognised_as_needing_no_reply() -> None:
    reply = Session().on_message(
        json.dumps({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    )
    assert reply.kind == "notification"
    assert reply.id is None


def test_a_request_from_the_server_is_recognised() -> None:
    # A server waiting on an answer that never comes hangs, so these have to be
    # distinguishable from notifications.
    reply = Session().on_message(
        json.dumps({"jsonrpc": "2.0", "id": 9, "method": "sampling/createMessage"})
    )
    assert reply.kind == "request"
    assert reply.id == 9


def test_a_server_error_is_raised_on_unwrap() -> None:
    session = Session()
    request = session.next_request("tools/call")
    reply = session.on_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request.id,
                "error": {"code": -32602, "message": "no such tool"},
            }
        )
    )
    assert reply.kind == "error"
    with pytest.raises(ServerError, match="no such tool") as caught:
        reply.unwrap()
    assert caught.value.code == -32602


def test_a_non_json_line_is_a_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="not JSON"):
        Session().on_message("this is not json")


def test_a_response_with_no_id_is_refused() -> None:
    with pytest.raises(ProtocolError, match="no usable id"):
        Session().on_message(json.dumps({"jsonrpc": "2.0", "result": {}}))


def test_tool_specs_are_parsed_with_their_annotations() -> None:
    specs, cursor = parse_tool_list(
        {
            "tools": [
                {
                    "name": "read_page",
                    "description": "Read a page.",
                    "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}},
                    "annotations": {"readOnlyHint": True},
                },
                {"name": "delete_all", "annotations": {"destructiveHint": True}},
                "not a dict",
            ],
            "nextCursor": "page2",
        }
    )
    assert [spec.name for spec in specs] == ["read_page", "delete_all"]
    assert specs[0].read_only
    assert specs[1].destructive
    assert cursor == "page2"


def test_a_tool_with_no_name_is_refused() -> None:
    with pytest.raises(ProtocolError, match="no name"):
        ToolSpec.parse({"description": "nameless"})


def test_content_blocks_flatten_to_text_and_images() -> None:
    outcome = ToolOutcome.parse(
        {
            "content": [
                {"type": "text", "text": "first"},
                {
                    "type": "image",
                    "data": base64.b64encode(b"png").decode(),
                    "mimeType": "image/png",
                },
                {"type": "resource", "resource": {"uri": "file:///x", "text": "from a file"}},
                {"type": "video", "data": "..."},
            ]
        }
    )
    assert "first" in outcome.text
    assert "from a file" in outcome.text
    assert "unsupported content block" in outcome.text
    assert outcome.images == (("image/png", base64.b64encode(b"png").decode()),)


def test_a_malformed_image_is_reported_rather_than_forwarded() -> None:
    # Caught here rather than left for the provider to reject: a 400 from the
    # model API is much harder to trace back to one MCP server.
    outcome = ToolOutcome.parse({"content": [{"type": "image", "data": "!!! not base64"}]})
    assert outcome.images == ()
    assert "not valid base64" in outcome.text


def test_an_is_error_result_is_distinct_from_a_transport_error() -> None:
    outcome = ToolOutcome.parse(
        {"content": [{"type": "text", "text": "no such file"}], "isError": True}
    )
    assert outcome.is_error
    assert outcome.text == "no such file"


# -- configuration ------------------------------------------------------------------


def test_a_string_command_is_refused() -> None:
    # `sh -c "..."` and `npx -y server` look equally harmless as strings, and
    # that difference is not something a config loader should guess at.
    with pytest.raises(TypeError, match="list of arguments"):
        load_servers([{"name": "bad", "command": "npx -y @scope/server"}])


def test_a_command_list_is_accepted() -> None:
    configs = load_servers(
        [{"name": "files", "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"]}]
    )
    assert configs[0].command[0] == "npx"
    assert configs[0].enabled


def test_the_child_environment_is_built_up_not_filtered_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Starting from os.environ and removing what looks sensitive leaks every new
    # credential name until somebody remembers to add it to a deny list.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-wanted")
    monkeypatch.setenv("PATH", "/usr/bin")

    config = ServerConfig(
        name="gh",
        command=("server",),
        env_passthrough=("GITHUB_TOKEN",),
        env={"EXTRA": "1"},
    )
    resolved = config.resolved_env()

    assert resolved["GITHUB_TOKEN"] == "ghp-wanted"
    assert resolved["EXTRA"] == "1"
    assert resolved["PATH"] == "/usr/bin"
    # The provider key was never asked for, so it is not there.
    assert "ANTHROPIC_API_KEY" not in resolved


def test_two_servers_with_one_name_are_refused() -> None:
    """The name keys their tools and their lifecycle, so a duplicate is a leak.

    ``McpBridge.servers`` is a dict on the name, so the second entry replaced the
    first and ``close()`` never stopped it -- an orphaned child process left
    running, which is the one thing this module's lifecycle exists to prevent.
    """
    with pytest.raises(ValueError, match="both named"):
        load_servers(
            [
                {"name": "files", "command": ["one"]},
                {"name": "files", "command": ["two"]},
            ]
        )


def test_an_entry_with_no_name_is_refused() -> None:
    with pytest.raises(ValueError, match="no name"):
        load_servers([{"command": ["x"]}])


# -- the bridge's trust rules --------------------------------------------------------


def test_a_read_only_hint_cannot_lower_the_danger_level() -> None:
    # The server wrote the hint. Treating it as evidence would let a compromised
    # server classify its own exfiltration tool as safe.
    safe_claim = ToolSpec(name="peek", description="", input_schema={}, read_only=True)
    assert danger_for(safe_claim) == MCP_FLOOR
    assert danger_for(safe_claim) > Danger.SAFE


def test_a_destructive_hint_does_raise_it() -> None:
    # Hints may only ever make a tool more restricted.
    risky = ToolSpec(name="wipe", description="", input_schema={}, destructive=True)
    assert danger_for(risky) is Danger.DESTRUCTIVE


def test_tool_names_are_namespaced_by_server() -> None:
    # Two servers offering `search` is normal, and letting the second silently
    # win means the agent calls one it did not mean to.
    assert qualified("github", "search") != qualified("jira", "search")
    assert qualified("github", "search").startswith("github")


# -- a real subprocess ----------------------------------------------------------------

SERVER = """
import json, sys

def send(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()

# Logged before anything else: a client that does not drain stderr deadlocks
# here, and that is the failure this fixture exists to catch.
print("starting up", file=sys.stderr, flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": message["id"], "result": {
            "protocolVersion": "REPLACE_VERSION",
            "serverInfo": {"name": "demo", "version": "1.0"},
            "capabilities": {"tools": {"listChanged": True}},
        }})
    elif method == "notifications/initialized":
        print("client is ready", file=sys.stderr, flush=True)
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": [
            {"name": "greet", "description": "Say hello.",
             "inputSchema": {"type": "object", "properties": {"who": {"type": "string"}},
                             "required": ["who"]},
             "annotations": {"readOnlyHint": True}},
        ]}})
    elif method == "tools/call":
        who = message["params"]["arguments"].get("who", "nobody")
        # A notification interleaved before the answer, which a client that
        # reads the next line and assumes it is the reply would choke on.
        send({"jsonrpc": "2.0", "method": "notifications/message",
              "params": {"level": "info", "data": "working"}})
        send({"jsonrpc": "2.0", "id": message["id"],
              "result": {"content": [{"type": "text", "text": f"hello {who}"}]}})
    else:
        send({"jsonrpc": "2.0", "id": message.get("id"),
              "error": {"code": -32601, "message": f"no {method}"}})
"""


CHATTY_SERVER = """
import json, sys, time

def send(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": message["id"], "result": {
            "protocolVersion": "REPLACE_VERSION",
            "serverInfo": {"name": "chatty", "version": "1.0"},
            "capabilities": {"tools": {}},
        }})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/call":
        # Progress forever, and never the answer. This is not a contrived
        # server: notifications/progress during a slow call is the normal thing
        # for a real one to do.
        while True:
            send({"jsonrpc": "2.0", "method": "notifications/progress",
                  "params": {"progressToken": 1, "progress": 1}})
            time.sleep(0.02)
    else:
        send({"jsonrpc": "2.0", "id": message.get("id"),
              "error": {"code": -32601, "message": "no " + str(method)}})
"""


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    from harness_agentic.mcp.protocol import PROTOCOL_VERSION

    path = tmp_path / "demo_server.py"
    path.write_text(SERVER.replace("REPLACE_VERSION", PROTOCOL_VERSION), encoding="utf-8")
    return path


@pytest.fixture
def chatty_server_script(tmp_path: Path) -> Path:
    from harness_agentic.mcp.protocol import PROTOCOL_VERSION

    path = tmp_path / "chatty_server.py"
    path.write_text(CHATTY_SERVER.replace("REPLACE_VERSION", PROTOCOL_VERSION), encoding="utf-8")
    return path


def test_a_chatty_server_cannot_hold_a_request_open_forever(chatty_server_script: Path) -> None:
    """The bug: the idle timeout was passed to every ``get`` and nothing else.

    So any message at all reset it, and a server sending progress notifications
    just inside the interval kept the request open indefinitely. With a
    synchronous core that means the turn never ends, and nothing is logged above
    debug. ``max_wait_s`` now bounds the request as a whole.

    Run on a thread so a regression fails this test rather than hanging the suite.
    """
    server = StdioServer(
        config=ServerConfig(
            name="chatty",
            command=(sys.executable, str(chatty_server_script)),
            timeout_s=5.0,  # generous: the point is that the ceiling, not this, stops it
            max_wait_s=1.0,
        )
    )
    server.start()
    outcome: list[object] = []

    def call() -> None:
        try:
            server.request("tools/call", {"name": "slow", "arguments": {}})
        except Exception as exc:
            outcome.append(exc)
        else:
            outcome.append("it returned a result")

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout=20)
    server.stop()

    assert outcome, "the request never came back -- the ceiling did not bound it"
    assert isinstance(outcome[0], McpError), outcome[0]
    assert "past" in str(outcome[0]), outcome[0]


def test_a_real_server_handshakes_lists_and_answers(server_script: Path) -> None:
    config = ServerConfig(name="demo", command=(sys.executable, str(server_script)), timeout_s=15.0)
    registry = ToolRegistry()
    bridge = McpBridge(registry=registry)

    try:
        bridged = bridge.connect(config)
        assert bridged is not None
        assert bridged.server.session.server_name == "demo"
        assert bridged.registered == ("demo__greet",)

        tool = registry.get("demo__greet")
        # The schema came from the server and is offered to the model unchanged.
        assert tool.schema().parameters["required"] == ["who"]
        # And the server's readOnlyHint did not buy it a lower danger level.
        assert tool.danger == MCP_FLOOR

        result = bridged.server.request(
            "tools/call", {"name": "greet", "arguments": {"who": "world"}}
        )
        outcome = ToolOutcome.parse(result)
        assert outcome.text == "hello world"
        # Interleaved notifications were handled without disturbing the reply.
        assert bridged.server.alive
        # And stderr was drained rather than filling its pipe.
        assert "starting up" in bridged.server.diagnostics()
    finally:
        bridge.close()

    assert registry.unregister("demo__greet") is False
    assert "demo__greet" not in registry.all()


def test_disconnecting_removes_the_tools(server_script: Path) -> None:
    # A tool whose server is gone fails in a way that looks like the tool being
    # broken, and the agent will keep retrying it.
    config = ServerConfig(name="demo", command=(sys.executable, str(server_script)))
    registry = ToolRegistry()
    bridge = McpBridge(registry=registry)
    bridge.connect(config)
    assert "demo__greet" in registry.all()

    bridge.disconnect("demo")
    assert "demo__greet" not in registry.all()


def test_a_server_that_cannot_start_is_reported_not_fatal(tmp_path: Path) -> None:
    registry = ToolRegistry()
    bridge = McpBridge(registry=registry)
    result = bridge.connect(ServerConfig(name="ghost", command=("definitely-not-a-real-binary",)))

    assert result is None
    assert "ghost" in bridge.failures
    assert "FAILED" in "\n".join(bridge.report())
    assert not registry.all()


def test_one_failing_server_does_not_stop_another(server_script: Path) -> None:
    registry = ToolRegistry()
    bridge = McpBridge(registry=registry)
    try:
        bridge.connect_all(
            [
                ServerConfig(name="ghost", command=("definitely-not-a-real-binary",)),
                ServerConfig(name="demo", command=(sys.executable, str(server_script))),
            ]
        )
        assert bridge.tool_names() == ("demo__greet",)
        assert set(bridge.failures) == {"ghost"}
    finally:
        bridge.close()


def test_a_disabled_server_is_not_started() -> None:
    bridge = McpBridge(registry=ToolRegistry())
    assert bridge.connect(ServerConfig(name="off", command=("x",), enabled=False)) is None
    assert bridge.failures == {}


def test_a_refused_bridged_tool_does_not_kill_the_turn() -> None:
    """The one rule this dispatcher has, broken on the ordinary path.

    A tool whose schema belongs to a remote server has no local model, so
    validation hands back the raw mapping. Only the success path allowed for
    that: the refusal and unavailable paths called ``model_dump()`` on it and
    raised ``AttributeError``, which escaped as a crash instead of becoming a
    tool error the model could read.

    And it broke on *refusal*, which for a bridged tool is the common case --
    MCP tools are NETWORK by default, so they need approval on almost every
    surface. The happy path worked, so nothing noticed.
    """
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="remote__add",
            description="Add two numbers on a remote server.",
            toolset="mcp:remote",
            params_model=None,
            raw_schema={"type": "object", "properties": {"a": {"type": "number"}}},
            handler=lambda params, ctx: ToolResult(text="never reached"),
            danger=Danger.NETWORK,
        )
    )
    executor = ToolExecutor(registry, approval=always_deny())

    result = executor.execute(
        ToolUseBlock(id="c1", name="remote__add", arguments={"a": 1}),
        _StubContext(),  # type: ignore[arg-type]
    )

    assert result.is_error
    assert "Not permitted" in result.text
    # The arguments still reach the trace, as a plain mapping.
    assert executor.records[-1].arguments == {"a": 1}


class _StubContext:
    """Enough of a ToolContext for a call that is refused before it runs."""

    session_id = "s1"
    surface = "cli"

    def emit(self, message: str) -> None:
        del message
