"""Slack, over the Events API.

Third adapter, and the one that forces the session model to be right. Slack
conversations are *threads*, not people: several colleagues in one thread are
having one conversation with the agent, and keying the session by sender would
give each of them a private half of it. So the thread is the session, and a
top-level message starts a new one -- which is also how the agent decides where
to reply, since answering in-channel instead of in-thread is how a bot becomes
the loudest participant in a busy channel.

The other Slack-specific constraint is the **three-second acknowledgement**. If
the endpoint has not answered 200 within three seconds Slack retries, and then
retries again, and an agent turn takes far longer than that. So the handler
verifies, queues, and returns -- which is exactly what :class:`QueueAdapter`
exists for. The retries that slip through anyway carry ``X-Slack-Retry-Num``
and are caught by the deduplicator upstream, keyed on the event id Slack keeps
stable across them.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from harness_agentic.errors import AdapterError, Unauthorized
from harness_agentic.gateway.adapter import QueueAdapter
from harness_agentic.gateway.signature import verify_slack
from harness_agentic.gateway.types import (
    Capabilities,
    ChatKind,
    DeliveryTarget,
    MessageEvent,
    OutboundMessage,
    Route,
    Sender,
    SentRef,
    Transport,
    WebhookResponse,
)

if TYPE_CHECKING:
    from harness_agentic.providers.credentials import Secret

log = logging.getLogger(__name__)

API_ROOT = "https://slack.com/api"
_MENTION = re.compile(r"<@([A-Z0-9]+)>")
_BOT_SUBTYPES = frozenset({"bot_message", "message_changed", "message_deleted"})


class SlackAdapter(QueueAdapter):
    """Receives signed Events API deliveries and answers in-thread."""

    platform = "slack"
    transport = Transport.WEBHOOK
    capabilities = Capabilities(
        max_chars=3800,
        edit_messages=True,
        typing_indicator=False,
        min_edit_interval_s=1.2,
        markdown="mrkdwn",
        threads=True,
        max_edits_per_message=40,
    )

    def __init__(
        self,
        *,
        signing_secret: Secret,
        bot_token: Secret,
        bot_user_id: str = "",
        api_root: str = API_ROOT,
        client: httpx.AsyncClient | None = None,
        path: str = "/webhooks/slack",
        maxsize: int = 1000,
        require_mention_in_channels: bool = True,
    ) -> None:
        """Create an adapter for one Slack app."""
        super().__init__(maxsize=maxsize)
        self._secret = signing_secret
        self._token = bot_token
        self._api_root = api_root.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._path = path
        self.bot_user_id = bot_user_id
        self.require_mention_in_channels = require_mention_in_channels
        self.rejected = 0
        self.retries_seen = 0

    async def connect(self) -> None:
        """Open the HTTP client and learn the bot's own user id.

        Needed to recognise ``<@U123>`` mentions and, more importantly, to
        ignore the agent's own messages -- a bot that answers itself in a
        thread produces an infinite conversation with nobody.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        if not self.bot_user_id:
            identity = await self._call("auth.test", {})
            self.bot_user_id = str(identity.get("user_id", ""))

    async def disconnect(self) -> None:
        """Close the client if this adapter opened it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- inbound ---------------------------------------------------------------

    def webhook_routes(self) -> Sequence[Route]:
        """The single Events API endpoint."""
        return (Route(path=self._path, handler=self.handle_webhook),)

    async def handle_webhook(self, body: bytes, headers: Mapping[str, str]) -> WebhookResponse:
        """Verify, queue, and answer inside Slack's three-second window."""
        try:
            verify_slack(body, headers, self._secret, now=datetime.now(UTC).timestamp())
        except Unauthorized:
            self.rejected += 1
            return WebhookResponse(status=400, body=b"bad signature")

        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            log.warning("slack webhook body was not JSON")
            return WebhookResponse(status=200, body=b"ignored")

        if payload.get("type") == "url_verification":
            # Echoed once when the endpoint is configured. Slack refuses to
            # save the URL without it.
            challenge = str(payload.get("challenge", ""))
            return WebhookResponse(status=200, body=challenge.encode(), content_type="text/plain")

        if _header(headers, "x-slack-retry-num"):
            # Slack retried because the first attempt was slow, not because it
            # failed. Counted, then handled: the deduplicator keys on the event
            # id, which stays the same across retries.
            self.retries_seen += 1

        event = self.normalize(payload)
        if event is not None:
            self.offer(event)
        return WebhookResponse(status=200, body=b"ok")

    def normalize(self, payload: Mapping[str, Any]) -> MessageEvent | None:
        """Convert one ``event_callback`` envelope into a normalized event."""
        if payload.get("type") != "event_callback":
            return None
        raw = payload.get("event") or {}
        kind_name = str(raw.get("type", ""))
        if kind_name not in ("message", "app_mention"):
            return None
        if raw.get("subtype") in _BOT_SUBTYPES or raw.get("bot_id"):
            return None
        user = str(raw.get("user") or "")
        if not user or (self.bot_user_id and user == self.bot_user_id):
            # The agent's own message coming back. Answering it would loop.
            return None

        channel = str(raw.get("channel") or "")
        channel_type = str(raw.get("channel_type") or "")
        is_dm = channel_type == "im"
        text = str(raw.get("text") or "")
        mentioned = (
            is_dm
            or kind_name == "app_mention"
            or not self.require_mention_in_channels
            or self._addressed(text)
        )
        if self.bot_user_id:
            text = _MENTION.sub("", text).strip() if mentioned else text

        # The thread the reply belongs in -- a reply's own thread, or the
        # message itself if it starts one. This is the session, and it is why
        # several people in one thread share one conversation.
        thread = str(raw.get("thread_ts") or raw.get("ts") or "")

        return MessageEvent(
            platform=self.platform,
            chat_id=channel,
            chat_kind=ChatKind.PRIVATE if is_dm else ChatKind.CHANNEL,
            sender=Sender(id=user),
            text=text,
            received_at=datetime.fromtimestamp(float(raw.get("ts") or 0), UTC),
            message_id=str(payload.get("event_id") or raw.get("ts") or ""),
            thread_id=thread,
            mentioned=mentioned,
            raw=payload,
        )

    def _addressed(self, text: str) -> bool:
        """Whether a channel message mentioned this bot."""
        return bool(self.bot_user_id) and f"<@{self.bot_user_id}>" in text

    # -- outbound --------------------------------------------------------------

    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        """Post a message, in-thread when there is a thread."""
        payload: dict[str, Any] = {
            "channel": target.chat_id,
            "text": to_mrkdwn(message.text) if message.markdown else message.text,
        }
        if target.thread_id:
            payload["thread_ts"] = target.thread_id
        result = await self._call("chat.postMessage", payload)
        return SentRef(
            platform=self.platform, chat_id=target.chat_id, message_id=str(result.get("ts", ""))
        )

    async def edit(self, ref: SentRef, message: OutboundMessage) -> None:
        """Rewrite a message already posted."""
        await self._call(
            "chat.update",
            {
                "channel": ref.chat_id,
                "ts": ref.message_id,
                "text": to_mrkdwn(message.text) if message.markdown else message.text,
            },
        )

    async def _call(self, method: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """One Web API call, with Slack's ``ok: false`` shape unwrapped."""
        if self._client is None:
            detail = "slack adapter is not connected"
            raise AdapterError(detail)
        response = await self._client.post(
            f"{self._api_root}/{method}",
            json=dict(payload),
            headers={"Authorization": f"Bearer {self._token.reveal()}"},
        )
        try:
            body = response.json()
        except ValueError as exc:
            detail = f"slack {method} returned {response.status_code} with a non-JSON body"
            raise AdapterError(detail) from exc
        if not body.get("ok", False):
            detail = f"slack {method} failed: {body.get('error', response.status_code)}"
            raise AdapterError(detail)
        result: Mapping[str, Any] = body
        return result


def to_mrkdwn(text: str) -> str:
    """Translate the markdown the model writes into Slack's dialect.

    Slack's ``mrkdwn`` is not markdown and the differences are the ones a model
    hits constantly: bold is one asterisk, italic is one underscore, and links
    are ``<url|text>``. Left untranslated, ``**bold**`` renders with the
    asterisks visible, which reads as the bot being broken.

    Code fences are left exactly as they are -- they mean the same thing in
    both, and rewriting punctuation inside one would corrupt the code.
    """
    out: list[str] = []
    for index, part in enumerate(re.split(r"(```.*?```|`[^`]*`)", text, flags=re.DOTALL)):
        if index % 2:
            out.append(part)
            continue
        converted = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", part)
        converted = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"*\1*", converted, flags=re.DOTALL)
        converted = re.sub(r"(?<![\w*])_(?=\S)(.+?)(?<=\S)_(?![\w*])", r"_\1_", converted)
        converted = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", converted, flags=re.MULTILINE)
        out.append(converted)
    return "".join(out)


def _header(headers: Mapping[str, str], name: str) -> str:
    """Case-insensitive header lookup."""
    target = name.lower()
    return next((v for k, v in headers.items() if k.lower() == target), "")
