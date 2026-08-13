"""LINE, over signed webhooks.

Second adapter, and chosen second on purpose: it is the one that breaks the
assumptions Telegram lets you get away with, and finding that out now is much
cheaper than finding it out after four adapters have been written against those
assumptions.

What LINE does differently:

* **Messages cannot be edited.** There is no ``editMessage``. The streaming
  design most chat agents use -- post a placeholder, rewrite it as text arrives
  -- is simply unavailable, which is why :class:`Capabilities` exists and why
  the delivery layer has a chunked path from the first milestone.
* **The reply token is single-use and short-lived.** Roughly thirty seconds, one
  use. An agent turn routinely takes longer, so the token is used for the first
  message if it is still plausible and everything after it goes out over the
  push API. Push messages are metered, which is a real reason not to narrate
  progress on this platform.
* **The webhook is public.** Every request is verified against
  ``X-Line-Signature`` over the raw body before it is parsed. LINE also sends a
  verification request with no events when the endpoint is configured, and that
  must answer 200 or the console refuses to save the URL.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from harness_agentic.errors import AdapterError, Unauthorized
from harness_agentic.gateway.adapter import QueueAdapter
from harness_agentic.gateway.signature import verify_line
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

API_ROOT = "https://api.line.me/v2/bot"
REPLY_TOKEN_BUDGET_S = 20.0
"""How long a reply token is treated as usable. LINE allows about 30 seconds;
staying well inside that avoids burning the turn's first message on a token
that expired while the model was thinking."""
MAX_MESSAGES_PER_CALL = 5
"""LINE accepts at most five message objects per API call."""

_KIND = {"user": ChatKind.PRIVATE, "group": ChatKind.GROUP, "room": ChatKind.GROUP}


class LineAdapter(QueueAdapter):
    """Receives signed webhooks and answers over the Messaging API."""

    platform = "line"
    transport = Transport.WEBHOOK
    capabilities = Capabilities(
        max_chars=5000,
        edit_messages=False,
        typing_indicator=False,
        markdown="none",
        threads=False,
    )

    def __init__(
        self,
        *,
        channel_secret: Secret,
        access_token: Secret,
        api_root: str = API_ROOT,
        client: httpx.AsyncClient | None = None,
        path: str = "/webhooks/line",
        maxsize: int = 1000,
    ) -> None:
        """Create an adapter for one LINE channel."""
        super().__init__(maxsize=maxsize)
        self._secret = channel_secret
        self._token = access_token
        self._api_root = api_root.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._path = path
        self._reply_tokens: dict[str, tuple[str, float]] = {}
        self.rejected = 0
        """Requests that failed signature verification. Watch this number."""

    async def connect(self) -> None:
        """Open the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)

    async def disconnect(self) -> None:
        """Close it again."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- inbound ---------------------------------------------------------------

    def webhook_routes(self) -> Sequence[Route]:
        """The single endpoint LINE delivers to."""
        return (Route(path=self._path, handler=self.handle_webhook),)

    async def handle_webhook(self, body: bytes, headers: Mapping[str, str]) -> WebhookResponse:
        """Verify, parse, and queue the events in one delivery.

        Always answers 200 once the signature is good, including for payloads
        it could not understand. A non-2xx makes LINE redeliver, and redelivering
        a payload this build cannot parse will not help -- it will just arrive
        again, forever.
        """
        try:
            verify_line(body, headers, self._secret)
        except Unauthorized:
            self.rejected += 1
            # 400, not 401: LINE does not retry on 400, and a signature failure
            # is either a misconfiguration or someone probing the endpoint.
            # Neither is improved by redelivery.
            return WebhookResponse(status=400, body=b"bad signature")

        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            log.warning("line webhook body was not JSON")
            return WebhookResponse(status=200, body=b"ignored")
        if not isinstance(payload, dict):
            # Valid JSON, wrong shape. Answering 200 is the documented contract
            # above, and it is what stops LINE redelivering a body forever that
            # no version of this code will ever parse.
            log.warning("line webhook body was not a JSON object")
            return WebhookResponse(status=200, body=b"ignored")

        events = payload.get("events") or []
        if not events:
            # The console's verification ping. Must be 200 or the URL cannot
            # be saved.
            return WebhookResponse(status=200, body=b"ok")

        for raw in events:
            if not isinstance(raw, dict):
                continue
            event = self.normalize(raw)
            if event is not None:
                self.offer(event)
        return WebhookResponse(status=200, body=b"ok")

    def normalize(self, raw: Mapping[str, Any]) -> MessageEvent | None:
        """Convert one LINE webhook event into a normalized event."""
        if raw.get("type") != "message":
            return None
        message = raw.get("message") or {}
        if message.get("type") != "text":
            # Images, stickers and audio arrive as ids that must be fetched
            # separately; that belongs with the attachment work, not here.
            return None

        source = raw.get("source") or {}
        kind = _KIND.get(str(source.get("type", "user")), ChatKind.PRIVATE)
        chat_id = str(source.get("groupId") or source.get("roomId") or source.get("userId") or "")
        timestamp = float(raw.get("timestamp", 0)) / 1000.0
        reply_token = str(raw.get("replyToken") or "")
        if reply_token and chat_id:
            self._forget_expired_tokens()
            self._reply_tokens[chat_id] = (reply_token, timestamp)

        return MessageEvent(
            platform=self.platform,
            chat_id=chat_id,
            chat_kind=kind,
            sender=Sender(id=str(source.get("userId") or chat_id)),
            text=str(message.get("text") or ""),
            received_at=datetime.fromtimestamp(timestamp, UTC),
            message_id=str(message.get("id") or ""),
            reply_to=reply_token,
            mentioned=True,
            raw=raw,
        )

    # -- outbound --------------------------------------------------------------

    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        """Reply if the token is still good, otherwise push."""
        body = {"type": "text", "text": message.text}
        token = self._claim_reply_token(target)
        if token:
            try:
                await self._call("message/reply", {"replyToken": token, "messages": [body]})
            except AdapterError as exc:
                # An expired or already-used token is recoverable: push instead
                # rather than losing the message.
                log.info("line reply token unusable (%s); pushing", exc)
                await self._call("message/push", {"to": target.chat_id, "messages": [body]})
        else:
            await self._call("message/push", {"to": target.chat_id, "messages": [body]})
        # LINE returns no message id for either call, so there is nothing to
        # edit later -- consistent with the capability being off.
        return SentRef(platform=self.platform, chat_id=target.chat_id, message_id="")

    def _claim_reply_token(self, target: DeliveryTarget) -> str:
        """Take the reply token for this chat, if one is still unspent.

        What the adapter has recorded is the authority, not what the target
        carries. A target is one object reused for every chunk of an answer, so
        trusting its ``reply_to`` meant replaying a single-use token on chunk
        two, chunk three, and so on -- each one a guaranteed 400 followed by a
        push. The answer still arrived, so nothing failed visibly; it just cost
        an extra round trip per chunk, on a platform that meters messages.
        """
        entry = self._reply_tokens.get(target.chat_id)
        if entry is None:
            # Spent on an earlier chunk, or from a delivery this process never
            # saw. Either way the only thing left that works is push.
            return ""
        token, issued_at = entry
        if target.reply_to and target.reply_to != token:
            # The turn is answering an older message than the last one to
            # arrive. Leave the newer token for the turn it belongs to.
            return ""
        del self._reply_tokens[target.chat_id]
        age = datetime.now(UTC).timestamp() - issued_at
        return token if age <= REPLY_TOKEN_BUDGET_S else ""

    def _forget_expired_tokens(self) -> None:
        """Drop tokens too old to use.

        A token whose turn never sent anything -- an unauthorized sender, a
        command answered inline -- is otherwise held for the life of the process,
        one per chat that has ever spoken.
        """
        cutoff = datetime.now(UTC).timestamp() - REPLY_TOKEN_BUDGET_S
        for chat_id, (_, issued_at) in list(self._reply_tokens.items()):
            if issued_at < cutoff:
                del self._reply_tokens[chat_id]

    async def _call(self, path: str, payload: Mapping[str, Any]) -> None:
        """One Messaging API call."""
        if self._client is None:
            detail = "line adapter is not connected"
            raise AdapterError(detail)
        response = await self._client.post(
            f"{self._api_root}/{path}",
            json=dict(payload),
            headers={"Authorization": f"Bearer {self._token.reveal()}"},
        )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            detail = f"line {path} failed with {response.status_code}: {response.text[:200]}"
            raise AdapterError(detail)
