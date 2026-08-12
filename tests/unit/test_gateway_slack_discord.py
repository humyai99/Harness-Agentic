"""Slack and Discord: the thread-keyed session, and the gateway handshake.

Two things here have no equivalent in the first two adapters and are the reason
those platforms are worth the code:

* Slack conversations are **threads**, not people. Several colleagues in one
  thread are having one conversation, and keying by sender would give each of
  them a private half of it.
* Discord is a **websocket**, so there is a handshake, a heartbeat and a
  session to resume. That is state machine work, and it is tested as one --
  frames in, frames out, no socket.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from harness_agentic.errors import AdapterError
from harness_agentic.gateway.keys import build_session_key
from harness_agentic.gateway.platforms.discord import (
    DEFAULT_INTENTS,
    MESSAGE_CONTENT,
    DiscordAdapter,
    GatewayProtocol,
    Op,
    ScriptedSocket,
    close_is_fatal,
)
from harness_agentic.gateway.platforms.slack import SlackAdapter, to_mrkdwn
from harness_agentic.gateway.types import ChatKind, DeliveryTarget, OutboundMessage, SentRef
from harness_agentic.providers.credentials import Secret

SIGNING_SECRET = "slack-signing-secret"
BOT = "UBOT01"


def client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# -- Slack --------------------------------------------------------------------


def slack_adapter(handler: Any = None, **kwargs: Any) -> SlackAdapter:
    return SlackAdapter(
        signing_secret=Secret(SIGNING_SECRET, source="test"),
        bot_token=Secret("xoxb-not-a-real-token", source="test"),
        bot_user_id=BOT,
        client=client(handler) if handler else None,
        **kwargs,
    )


def slack_sign(body: bytes, *, timestamp: str | None = None) -> dict[str, str]:
    stamp = timestamp or str(int(datetime.now(UTC).timestamp()))
    base = b"v0:" + stamp.encode() + b":" + body
    signature = "v0=" + hmac.new(SIGNING_SECRET.encode(), base, hashlib.sha256).hexdigest()
    return {"X-Slack-Signature": signature, "X-Slack-Request-Timestamp": stamp}


def slack_event(
    text: str = f"<@{BOT}> hello",
    *,
    user: str = "U1",
    channel: str = "C1",
    ts: str = "1700000001.000100",
    thread_ts: str = "",
    channel_type: str = "channel",
    event_id: str = "Ev1",
    kind: str = "app_mention",
) -> bytes:
    event: dict[str, Any] = {
        "type": kind,
        "user": user,
        "text": text,
        "ts": ts,
        "channel": channel,
        "channel_type": channel_type,
    }
    if thread_ts:
        event["thread_ts"] = thread_ts
    return json.dumps({"type": "event_callback", "event_id": event_id, "event": event}).encode()


async def test_slack_echoes_the_url_verification_challenge() -> None:
    # Slack refuses to save the endpoint without this.
    adapter = slack_adapter()
    body = json.dumps({"type": "url_verification", "challenge": "abc123"}).encode()
    response = await adapter.handle_webhook(body, slack_sign(body))
    assert response.status == 200
    assert response.body == b"abc123"


async def test_slack_refuses_an_unsigned_or_replayed_delivery() -> None:
    adapter = slack_adapter()
    body = slack_event()

    assert (await adapter.handle_webhook(body, {})).status == 400
    # A captured request replayed hours later must not be accepted.
    old = slack_sign(body, timestamp="1700000000")
    assert (await adapter.handle_webhook(body, old)).status == 400
    assert adapter.rejected == 2


async def test_slack_keys_the_session_by_thread_not_by_sender() -> None:
    # Two people in one thread are having one conversation with the agent.
    adapter = slack_adapter()
    first = adapter.normalize(json.loads(slack_event(user="U1", ts="1.1", thread_ts="1.0")))
    second = adapter.normalize(json.loads(slack_event(user="U2", ts="1.2", thread_ts="1.0")))

    assert first is not None
    assert second is not None
    assert build_session_key(first) == build_session_key(second)
    assert build_session_key(first).endswith("C1#1.0")


async def test_a_top_level_slack_message_starts_its_own_thread() -> None:
    adapter = slack_adapter()
    one = adapter.normalize(json.loads(slack_event(ts="1.1")))
    two = adapter.normalize(json.loads(slack_event(ts="2.2", event_id="Ev2")))

    assert one is not None
    assert two is not None
    # Each question gets its own thread, and its own session.
    assert one.thread_id == "1.1"
    assert build_session_key(one) != build_session_key(two)


async def test_slack_ignores_its_own_messages() -> None:
    # A bot that answers itself in a thread never stops.
    adapter = slack_adapter()
    assert adapter.normalize(json.loads(slack_event(user=BOT, kind="message"))) is None

    body = json.loads(slack_event(kind="message"))
    body["event"]["bot_id"] = "B1"
    assert adapter.normalize(body) is None


async def test_slack_needs_a_mention_in_a_channel_but_not_in_a_dm() -> None:
    adapter = slack_adapter()
    quiet = adapter.normalize(json.loads(slack_event("just chatting", kind="message")))
    assert quiet is not None
    assert not quiet.mentioned

    direct = adapter.normalize(json.loads(slack_event("hello", kind="message", channel_type="im")))
    assert direct is not None
    assert direct.mentioned
    assert direct.chat_kind is ChatKind.PRIVATE


async def test_slack_strips_the_mention_before_the_model_sees_it() -> None:
    adapter = slack_adapter()
    event = adapter.normalize(json.loads(slack_event(f"<@{BOT}> deploy staging")))
    assert event is not None
    assert event.text == "deploy staging"


async def test_slack_counts_retries_and_keeps_the_event_id_stable() -> None:
    # The deduplicator upstream keys on event_id, which Slack holds constant
    # across the retries its own three-second timeout causes.
    adapter = slack_adapter()
    body = slack_event()
    headers = slack_sign(body) | {"X-Slack-Retry-Num": "1"}
    await adapter.handle_webhook(body, headers)

    assert adapter.retries_seen == 1
    assert (await anext(adapter.poll())).message_id == "Ev1"


async def test_slack_replies_in_thread() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "ts": "9.9"})

    adapter = slack_adapter(handler)
    ref = await adapter.send(
        DeliveryTarget("slack", "C1", thread_id="1.0"), OutboundMessage("answer")
    )

    assert seen[0]["thread_ts"] == "1.0"
    assert ref.message_id == "9.9"


async def test_slack_edits_in_place() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(200, json={"ok": True, "ts": "9.9"})

    adapter = slack_adapter(handler)
    await adapter.edit(SentRef("slack", "C1", "9.9"), OutboundMessage("revised"))
    assert calls == ["chat.update"]


async def test_slack_surfaces_its_ok_false_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})

    adapter = slack_adapter(handler)
    with pytest.raises(AdapterError, match="channel_not_found"):
        await adapter.send(DeliveryTarget("slack", "C1"), OutboundMessage("hi"))


def test_mrkdwn_translates_the_markdown_a_model_writes() -> None:
    # Slack's dialect is not markdown, and untranslated `**bold**` renders with
    # the asterisks visible -- which reads as the bot being broken.
    assert to_mrkdwn("**bold** and _italic_") == "*bold* and _italic_"
    assert to_mrkdwn("see [the docs](https://x.test)") == "see <https://x.test|the docs>"
    assert to_mrkdwn("## Heading") == "*Heading*"


def test_mrkdwn_leaves_code_alone() -> None:
    # Rewriting punctuation inside a fence would corrupt the code.
    source = "before\n```python\nx = a ** b\n```\nafter **bold**"
    converted = to_mrkdwn(source)
    assert "a ** b" in converted
    assert converted.endswith("after *bold*")
    assert to_mrkdwn("`a ** b`") == "`a ** b`"


# -- Discord ---------------------------------------------------------------------


def protocol() -> GatewayProtocol:
    return GatewayProtocol(token=Secret("bot-token", source="test"))


HELLO = {"op": int(Op.HELLO), "d": {"heartbeat_interval": 41250}}
READY = {
    "op": int(Op.DISPATCH),
    "t": "READY",
    "s": 1,
    "d": {"session_id": "sess-1", "resume_gateway_url": "wss://resume.test"},
}


def test_the_handshake_identifies_then_becomes_resumable() -> None:
    state = protocol()
    replies = state.on_frame(HELLO)

    assert [f.op for f in replies] == [Op.IDENTIFY]
    assert replies[0].data["intents"] & MESSAGE_CONTENT
    assert replies[0].data["intents"] == DEFAULT_INTENTS
    assert not state.can_resume

    state.on_frame(READY)
    assert state.ready
    assert state.can_resume
    assert state.heartbeat_interval_s == pytest.approx(41.25)


def test_a_reconnect_resumes_from_the_last_sequence_number() -> None:
    state = protocol()
    state.on_frame(HELLO)
    state.on_frame(READY)
    state.on_frame({"op": int(Op.DISPATCH), "t": "MESSAGE_CREATE", "s": 7, "d": {}})

    # The socket dropped; a fresh HELLO arrives on the new connection.
    replies = state.on_frame(HELLO)
    assert [f.op for f in replies] == [Op.RESUME]
    assert replies[0].data["seq"] == 7
    assert replies[0].data["session_id"] == "sess-1"
    assert state.connect_url().startswith("wss://resume.test")


def test_an_unresumable_invalid_session_starts_over() -> None:
    state = protocol()
    state.on_frame(HELLO)
    state.on_frame(READY)

    state.on_frame({"op": int(Op.INVALID_SESSION), "d": False})
    assert not state.can_resume
    # And the next handshake identifies rather than trying a dead session.
    assert [f.op for f in state.on_frame(HELLO)] == [Op.IDENTIFY]


def test_a_resumable_invalid_session_keeps_the_session() -> None:
    state = protocol()
    state.on_frame(HELLO)
    state.on_frame(READY)
    state.on_frame({"op": int(Op.INVALID_SESSION), "d": True})
    assert state.can_resume


def test_an_unacknowledged_heartbeat_marks_the_connection_dead() -> None:
    # A TCP connection stays open long after the far end stops listening. This
    # is the only signal that the socket is a zombie.
    state = protocol()
    state.on_frame(HELLO)
    state.heartbeat()
    assert state.zombied()

    state.on_frame({"op": int(Op.HEARTBEAT_ACK)})
    assert not state.zombied()


def test_the_gateway_can_demand_a_heartbeat() -> None:
    state = protocol()
    state.on_frame(HELLO)
    replies = state.on_frame({"op": int(Op.HEARTBEAT)})
    assert [f.op for f in replies] == [Op.HEARTBEAT]


def test_fatal_close_codes_are_not_retried() -> None:
    # Reconnecting on a bad token is what gets the token disabled.
    assert close_is_fatal(4004)
    assert close_is_fatal(4014)
    assert not close_is_fatal(1006)
    assert not close_is_fatal(None)


def test_a_frame_serializes_to_what_discord_expects() -> None:
    from harness_agentic.gateway.platforms.discord import Frame

    assert json.loads(Frame(Op.HEARTBEAT, 42).encode()) == {"op": 1, "d": 42}


async def test_the_scripted_socket_drives_a_whole_handshake() -> None:
    message = {
        "op": int(Op.DISPATCH),
        "t": "MESSAGE_CREATE",
        "s": 2,
        "d": {
            "id": "m1",
            "channel_id": "chan1",
            "guild_id": "g1",
            "content": f"<@{BOT}> what is up",
            "author": {"id": "u1", "username": "someone"},
            "mentions": [{"id": BOT}],
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
    }
    socket = ScriptedSocket(inbound=[HELLO, READY, message])
    adapter = DiscordAdapter(
        Secret("bot-token", source="test"), bot_user_id=BOT, socket_factory=_fixed(socket)
    )

    events = [event async for event in adapter.poll()]

    assert socket.ops() == [Op.IDENTIFY]
    assert socket.closed
    assert len(events) == 1
    assert events[0].text == "what is up"
    assert events[0].chat_id == "chan1"
    assert events[0].chat_kind is ChatKind.CHANNEL


async def test_discord_ignores_bots_including_itself() -> None:
    adapter = DiscordAdapter(Secret("t", source="test"), bot_user_id=BOT)
    other_bot = {
        "op": int(Op.DISPATCH),
        "t": "MESSAGE_CREATE",
        "d": {"id": "m", "channel_id": "c", "content": "beep", "author": {"id": "x", "bot": True}},
    }
    itself = {
        "op": int(Op.DISPATCH),
        "t": "MESSAGE_CREATE",
        "d": {"id": "m", "channel_id": "c", "content": "hi", "author": {"id": BOT}},
    }
    assert adapter.normalize(other_bot) is None
    assert adapter.normalize(itself) is None


async def test_a_discord_dm_needs_no_mention() -> None:
    adapter = DiscordAdapter(Secret("t", source="test"), bot_user_id=BOT)
    event = adapter.normalize(
        {
            "op": int(Op.DISPATCH),
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "m",
                "channel_id": "dm1",
                "content": "hello",
                "author": {"id": "u1"},
            },
        }
    )
    assert event is not None
    assert event.mentioned
    assert event.chat_kind is ChatKind.PRIVATE


async def test_discord_posts_and_edits_over_rest() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"id": "msg1"})

    adapter = DiscordAdapter(Secret("t", source="test"), client=client(handler))
    ref = await adapter.send(DeliveryTarget("discord", "chan1"), OutboundMessage("hello"))
    await adapter.edit(ref, OutboundMessage("hello again"))
    await adapter.typing(DeliveryTarget("discord", "chan1"))

    assert ref.message_id == "msg1"
    assert calls == [
        ("POST", "/api/v10/channels/chan1/messages"),
        ("PATCH", "/api/v10/channels/chan1/messages/msg1"),
        ("POST", "/api/v10/channels/chan1/typing"),
    ]


async def test_discord_reports_rest_failures_without_the_token() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "Missing Access"})

    adapter = DiscordAdapter(
        Secret("super-secret-bot-token", source="test"), client=client(handler)
    )
    with pytest.raises(AdapterError) as caught:
        await adapter.send(DeliveryTarget("discord", "c"), OutboundMessage("hi"))
    assert "403" in str(caught.value)
    assert "super-secret-bot-token" not in str(caught.value)


def _fixed(socket: ScriptedSocket) -> Any:
    async def factory(_url: str) -> ScriptedSocket:
        return socket

    return factory
