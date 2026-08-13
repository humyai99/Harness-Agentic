"""The two real adapters, driven against recorded payloads.

No network: the Telegram adapter gets an ``httpx.MockTransport`` and the LINE
adapter gets its webhook handler called directly, which is the whole reason
routes are plain callables over ``(body, headers)``. What is being checked is
the part that is specific to each platform and therefore cannot be tested
anywhere else -- offset advancement, MarkdownV2 escaping, signature rejection,
and the single-use reply token.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from harness_agentic.core.secrets import Secret
from harness_agentic.errors import AdapterError
from harness_agentic.gateway.platforms import load
from harness_agentic.gateway.platforms.fake import FakeAdapter, line_like
from harness_agentic.gateway.platforms.line import LineAdapter
from harness_agentic.gateway.platforms.telegram import TelegramAdapter, escape_markdown_v2
from harness_agentic.gateway.types import ChatKind, DeliveryTarget, OutboundMessage, Transport
from harness_agentic.gateway.webserver import WebhookApp

CHANNEL_SECRET = "line-channel-secret"


# -- Telegram --------------------------------------------------------------------


def telegram_update(update_id: int, text: str, *, chat_type: str = "private") -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id * 10,
            "date": 1767225600,
            "chat": {"id": -100 if chat_type != "private" else 42, "type": chat_type},
            "from": {"id": 7, "username": "somebody"},
            "text": text,
        },
    }


def telegram_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_telegram_normalizes_an_update() -> None:
    adapter = TelegramAdapter(Secret("t", source="test"), bot_username="mybot")
    event = adapter.normalize(telegram_update(1, "hello"))

    assert event is not None
    assert (event.platform, event.chat_id, event.text) == ("telegram", "42", "hello")
    assert event.chat_kind is ChatKind.PRIVATE
    assert event.sender.id == "7"
    # Namespaced by chat: two chats can each have message_id 1.
    assert event.message_id == "42:10"


async def test_telegram_ignores_group_chatter_it_was_not_addressed_in() -> None:
    adapter = TelegramAdapter(Secret("t", source="test"), bot_username="mybot")
    ignored = adapter.normalize(telegram_update(1, "just talking", chat_type="supergroup"))
    addressed = adapter.normalize(telegram_update(2, "@mybot help", chat_type="supergroup"))

    assert ignored is not None
    assert not ignored.mentioned
    assert addressed is not None
    assert addressed.mentioned
    # The mention is stripped so the model does not see its own handle.
    assert addressed.text == "help"


async def test_telegram_advances_its_offset_only_after_yielding() -> None:
    # Advancing before the event is handed on would lose messages on a crash.
    # Advancing after means at-least-once, which the deduplicator absorbs.
    seen_offsets: list[int] = []
    batches = [[telegram_update(11, "one"), telegram_update(12, "two")], []]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path.endswith("getUpdates"):
            seen_offsets.append(payload["offset"])
            body = batches.pop(0) if batches else []
            return httpx.Response(200, json={"ok": True, "result": body})
        return httpx.Response(200, json={"ok": True, "result": {}})

    adapter = TelegramAdapter(
        Secret("t", source="test"), client=telegram_client(handler), bot_username="mybot"
    )
    stream = adapter.poll()
    first = await anext(stream)
    assert first.text == "one"
    second = await anext(stream)
    assert second.text == "two"

    assert seen_offsets[0] == 0
    # Suspended at the yield of update 12, the offset is still 12 -- it moved
    # past 11 only once 11 had been handed on, and will move past 12 only once
    # 12 has. That is the property: a crash here replays update 12 rather than
    # losing it, and the deduplicator upstream absorbs the replay.
    assert adapter._offset == 12


async def test_telegram_reports_api_errors_without_leaking_the_token() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "chat not found"})

    adapter = TelegramAdapter(
        Secret("super-secret-token", source="test"), client=telegram_client(handler)
    )
    with pytest.raises(AdapterError) as caught:
        await adapter.send(DeliveryTarget("telegram", "42"), OutboundMessage("hi"))

    assert "chat not found" in str(caught.value)
    # The URL carries the bot token; it must never reach a log line.
    assert "super-secret-token" not in str(caught.value)


async def test_telegram_tolerates_an_edit_to_identical_text() -> None:
    # Debouncing makes this reachable, and the Bot API calls it an error.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"ok": False, "description": "Bad Request: message is not modified"},
        )

    adapter = TelegramAdapter(Secret("t", source="test"), client=telegram_client(handler))
    from harness_agentic.gateway.types import SentRef

    await adapter.edit(SentRef("telegram", "42", "9"), OutboundMessage("same"))


def test_markdown_v2_escaping_covers_ordinary_prose() -> None:
    # A single unescaped '.' or '-' makes Telegram reject the whole message,
    # and the user simply never receives the answer.
    escaped = escape_markdown_v2("Version 1.2 (beta) - see docs!")
    assert escaped == r"Version 1\.2 \(beta\) \- see docs\!"


# -- LINE ------------------------------------------------------------------------


def sign(body: bytes) -> dict[str, str]:
    digest = hmac.new(CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return {"X-Line-Signature": base64.b64encode(digest).decode()}


def line_adapter(handler: Any = None) -> LineAdapter:
    client = telegram_client(handler) if handler else None
    return LineAdapter(
        channel_secret=Secret(CHANNEL_SECRET, source="test"),
        access_token=Secret("line-access-token", source="test"),
        client=client,
    )


def line_payload(
    text: str = "hello",
    *,
    message_id: str = "m1",
    user_id: str = "Uuser1",
    reply_token: str = "reply-token-1",
    timestamp_ms: int | None = None,
) -> bytes:
    """A signed-delivery body.

    The timestamp defaults to now because the adapter judges a reply token by
    its age: a fixture pinned to a past date makes every token look expired, and
    a test written against it asserts the push path while believing it covers
    the reply one.
    """
    return json.dumps(
        {
            "destination": "Uabc",
            "events": [
                {
                    "type": "message",
                    "replyToken": reply_token,
                    "timestamp": timestamp_ms
                    if timestamp_ms is not None
                    else int(datetime.now(UTC).timestamp() * 1000),
                    "source": {"type": "user", "userId": user_id},
                    "message": {"id": message_id, "type": "text", "text": text},
                }
            ],
        }
    ).encode()


async def test_line_accepts_a_signed_delivery() -> None:
    adapter = line_adapter()
    body = line_payload("สวัสดีครับ")
    response = await adapter.handle_webhook(body, sign(body))

    assert response.status == 200
    event = await anext(adapter.poll())
    assert event.text == "สวัสดีครับ"
    assert event.chat_id == "Uuser1"
    assert event.reply_to == "reply-token-1"


async def test_line_refuses_an_unsigned_delivery_and_queues_nothing() -> None:
    adapter = line_adapter()
    body = line_payload()

    assert (await adapter.handle_webhook(body, {})).status == 400
    assert (await adapter.handle_webhook(body, {"X-Line-Signature": "nope"})).status == 400
    assert adapter.rejected == 2
    # 400 rather than 500: LINE does not retry a 400, and redelivering a
    # request that failed verification will fail verification again.


async def test_line_answers_the_console_verification_ping() -> None:
    # Sent with no events when the endpoint is configured. A non-200 here means
    # the URL cannot be saved at all.
    adapter = line_adapter()
    body = json.dumps({"destination": "U", "events": []}).encode()
    assert (await adapter.handle_webhook(body, sign(body))).status == 200


async def test_line_does_not_redeliver_a_payload_it_cannot_parse() -> None:
    adapter = line_adapter()
    body = b"not json at all"
    response = await adapter.handle_webhook(body, sign(body))
    assert response.status == 200


async def test_line_uses_the_reply_token_once_then_pushes() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(200, json={})

    adapter = line_adapter(handler)
    body = line_payload()
    await adapter.handle_webhook(body, sign(body))
    event = await anext(adapter.poll())

    target = DeliveryTarget("line", event.chat_id, reply_to=event.reply_to)
    await adapter.send(target, OutboundMessage("first"))
    await adapter.send(DeliveryTarget("line", event.chat_id), OutboundMessage("second"))

    # A reply token is single-use; a second attempt with it costs a message.
    assert calls == ["reply", "push"]


async def test_line_does_not_replay_the_token_across_chunks_of_one_answer() -> None:
    """The delivery layer reuses one target for every chunk it sends.

    So a target's ``reply_to`` is present on chunk two and chunk three as well,
    and honouring it there replayed a single-use token: a guaranteed 400 and a
    push behind it, per chunk. The answer still arrived, which is why nothing
    caught it -- it just cost an extra round trip each time, on the one platform
    that meters messages.
    """
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        calls.append(endpoint)
        if endpoint == "reply" and calls.count("reply") > 1:
            return httpx.Response(400, text="Invalid reply token")
        return httpx.Response(200, json={})

    adapter = line_adapter(handler)
    body = line_payload()
    await adapter.handle_webhook(body, sign(body))
    event = await anext(adapter.poll())

    target = DeliveryTarget("line", event.chat_id, reply_to=event.reply_to)
    for index in range(3):
        await adapter.send(target, OutboundMessage(f"chunk {index}"))

    assert calls == ["reply", "push", "push"]


async def test_line_does_not_try_a_token_it_never_issued() -> None:
    """A token this process did not record cannot be spent usefully.

    It is either already used or from before a restart, so attempting it buys a
    round trip and a 400. The message still has to arrive, so it goes by push.
    """
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(200, json={})

    adapter = line_adapter(handler)
    target = DeliveryTarget("line", "Uuser1", reply_to="a-token-from-a-previous-process")
    await adapter.send(target, OutboundMessage("late answer"))

    assert calls == ["push"]


async def test_line_does_not_hoard_reply_tokens_for_turns_that_never_answered() -> None:
    # An unauthorized sender or a command answered inline leaves a token nobody
    # claims. One per chat that has ever spoken, held for the life of the
    # process, is a leak with a slow fuse.
    adapter = line_adapter()
    stale = line_payload(user_id="Uold", reply_token="old-token", timestamp_ms=1767225600000)
    await adapter.handle_webhook(stale, sign(stale))

    fresh = line_payload(message_id="m2")
    await adapter.handle_webhook(fresh, sign(fresh))

    assert "Uold" not in adapter._reply_tokens


async def test_line_does_not_redeliver_a_body_of_the_wrong_shape() -> None:
    # Valid JSON, not an object. It used to raise, which meant a 500 and LINE
    # redelivering a body no version of this code will ever parse.
    adapter = line_adapter()
    body = b"[1, 2, 3]"
    response = await adapter.handle_webhook(body, sign(body))
    assert response.status == 200


async def test_line_declares_that_it_cannot_edit() -> None:
    adapter = line_adapter()
    assert not adapter.capabilities.edit_messages
    assert adapter.transport is Transport.WEBHOOK
    assert "chunked" in adapter.describe()


async def test_an_adapter_claiming_edits_without_implementing_them_fails_loudly() -> None:
    from harness_agentic.gateway.types import SentRef

    adapter = line_adapter()
    with pytest.raises(AdapterError, match="does not implement edit"):
        await adapter.edit(SentRef("line", "U", ""), OutboundMessage("x"))


# -- the webhook server ------------------------------------------------------------


async def test_the_web_app_routes_health_and_unknown_paths() -> None:
    app = WebhookApp(routes=line_adapter().webhook_routes())
    assert (await app.dispatch("/healthz", "GET", {}, b"")).status == 200
    assert (await app.dispatch("/nope", "POST", {}, b"")).status == 404
    assert (await app.dispatch("/webhooks/line", "GET", {}, b"")).status == 405


async def test_the_web_app_rejects_an_oversized_body_before_verifying_it() -> None:
    # The endpoint is unauthenticated until the signature is checked, and the
    # signature cannot be checked without the body -- so the body is bounded.
    app = WebhookApp(routes=line_adapter().webhook_routes(), max_body_bytes=10)
    assert (await app.dispatch("/webhooks/line", "POST", {}, None)).status == 413


async def test_the_web_app_serves_a_signed_line_delivery() -> None:
    adapter = line_adapter()
    app = WebhookApp(routes=adapter.webhook_routes())
    body = line_payload("via the server")
    response = await app.dispatch("/webhooks/line", "POST", sign(body), body)

    assert response.status == 200
    assert (await anext(adapter.poll())).text == "via the server"


async def test_a_handler_that_explodes_returns_500_so_the_platform_retries() -> None:
    from harness_agentic.gateway.types import Route, WebhookResponse

    async def boom(_body: bytes, _headers: Any) -> WebhookResponse:
        detail = "storage is down"
        raise RuntimeError(detail)

    app = WebhookApp(routes=[Route("/boom", boom)])
    assert (await app.dispatch("/boom", "POST", {}, b"{}")).status == 500


# -- the registry --------------------------------------------------------------------


def test_platforms_resolve_by_name() -> None:
    assert load("line") is LineAdapter
    assert load("telegram") is TelegramAdapter
    with pytest.raises(AdapterError, match="unknown platform"):
        load("myspace")


async def test_the_fake_can_impersonate_another_platforms_limits() -> None:
    # This is what makes the chunked delivery path testable without a LINE
    # account, and it is why capabilities are per-instance.
    adapter = FakeAdapter(capabilities=line_like())
    assert adapter.capabilities.max_chars == 5000
    assert not adapter.capabilities.edit_messages
    assert FakeAdapter().capabilities.edit_messages


async def test_a_bounded_queue_drops_the_oldest_and_says_so() -> None:
    adapter = FakeAdapter(maxsize=2)
    for index in range(4):
        adapter.inject(f"m{index}")
    assert adapter.dropped == 2
    first = await anext(adapter.poll())
    assert first.text == "m2"
