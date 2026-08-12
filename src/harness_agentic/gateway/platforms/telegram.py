"""Telegram, over long polling.

First adapter because it needs no public URL: ``getUpdates`` reaches out from
wherever the process runs, so the development loop is edit-save-message with no
tunnel in between. It also exercises the edit-in-place streaming path, which is
the one most platforms support.

Two Telegram-specific hazards are handled here and nowhere else:

* **MarkdownV2 escaping.** Telegram's dialect requires escaping a long list of
  punctuation *even inside* text that is not markup, and an unescaped character
  makes the API reject the whole message. Since the agent writes prose full of
  parentheses and hyphens, the safe default is to escape and let real formatting
  through deliberately.
* **The update offset.** ``getUpdates`` only advances when the next call passes
  ``offset = last_update_id + 1``. Advancing it before the event is queued would
  drop messages on a crash; advancing after means at-least-once delivery, which
  the deduplicator upstream is there to absorb.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from harness_agentic.errors import AdapterError
from harness_agentic.gateway.adapter import PlatformAdapter
from harness_agentic.gateway.types import (
    Capabilities,
    ChatKind,
    DeliveryTarget,
    MessageEvent,
    OutboundMessage,
    Sender,
    SentRef,
    Transport,
)

if TYPE_CHECKING:
    from harness_agentic.providers.credentials import Secret

log = logging.getLogger(__name__)

API_ROOT = "https://api.telegram.org"
LONG_POLL_S = 25
"""Telegram holds the request open this long when there is nothing to say."""
BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 60.0
EMPTY_POLL_PAUSE_S = 1.0
MAX_CONSECUTIVE_FAILURES = 8
"""After this many failures in a row, give up and let the supervisor decide.

Retrying forever inside the adapter is how a bot with a revoked token stays
"running" for three days while answering nobody."""

_MDV2_SPECIAL = r"_*[]()~`>#+-=|{}.!"
_ESCAPE = re.compile("([" + re.escape(_MDV2_SPECIAL) + "])")

_KIND = {
    "private": ChatKind.PRIVATE,
    "group": ChatKind.GROUP,
    "supergroup": ChatKind.GROUP,
    "channel": ChatKind.CHANNEL,
}


def escape_markdown_v2(text: str) -> str:
    """Escape every character Telegram's MarkdownV2 treats as markup.

    Aggressive on purpose. A single stray ``.`` in "version 1.2." is enough for
    the API to reject the message with a 400, and a rejected message is an
    answer the user never receives.
    """
    return _ESCAPE.sub(r"\\\1", text)


class TelegramAdapter(PlatformAdapter):
    """Talks to the Bot API over long polling."""

    platform = "telegram"
    transport = Transport.POLL
    capabilities = Capabilities(
        max_chars=4096,
        edit_messages=True,
        typing_indicator=True,
        min_edit_interval_s=1.6,
        markdown="markdown_v2",
        threads=True,
        max_edits_per_message=30,
    )

    def __init__(
        self,
        token: Secret,
        *,
        api_root: str = API_ROOT,
        client: httpx.AsyncClient | None = None,
        bot_username: str = "",
        timeout_s: float = LONG_POLL_S,
    ) -> None:
        """Create an adapter for one bot token."""
        self._token = token
        self._api_root = api_root.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._offset = 0
        self._timeout_s = timeout_s
        self.bot_username = bot_username

    # -- lifecycle -------------------------------------------------------------

    async def connect(self) -> None:
        """Open the HTTP client and learn the bot's own username.

        The username is needed to recognise ``@mention`` in groups, and asking
        for it doubles as a credential check: a bad token fails here, at
        startup, rather than silently never receiving anything.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s + 10)
        me = await self._call("getMe", {})
        self.bot_username = self.bot_username or str(me.get("username", ""))

    async def disconnect(self) -> None:
        """Close the client if this adapter opened it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- outbound --------------------------------------------------------------

    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        """Post a message to a chat."""
        payload: dict[str, Any] = {
            "chat_id": target.chat_id,
            "text": self._render(message),
            "disable_notification": message.silent,
        }
        if message.markdown:
            payload["parse_mode"] = "MarkdownV2"
        if target.thread_id:
            payload["message_thread_id"] = target.thread_id
        result = await self._call("sendMessage", payload)
        return SentRef(
            platform=self.platform,
            chat_id=target.chat_id,
            message_id=str(result.get("message_id", "")),
        )

    async def edit(self, ref: SentRef, message: OutboundMessage) -> None:
        """Rewrite a message already posted."""
        payload: dict[str, Any] = {
            "chat_id": ref.chat_id,
            "message_id": ref.message_id,
            "text": self._render(message),
        }
        if message.markdown:
            payload["parse_mode"] = "MarkdownV2"
        try:
            await self._call("editMessageText", payload)
        except AdapterError as exc:
            # Editing to identical text is an error in the Bot API and a no-op
            # in intent. Debouncing makes it reachable, so it is not a failure.
            if "message is not modified" not in str(exc):
                raise

    async def typing(self, target: DeliveryTarget) -> None:
        """Show the typing indicator for a few seconds."""
        await self._call("sendChatAction", {"chat_id": target.chat_id, "action": "typing"})

    def _render(self, message: OutboundMessage) -> str:
        """Apply MarkdownV2 escaping if the message asked for markup."""
        return escape_markdown_v2(message.text) if message.markdown else message.text

    # -- inbound ---------------------------------------------------------------

    def poll(self) -> AsyncIterator[MessageEvent]:
        """Yield messages as ``getUpdates`` returns them."""
        return self._poll()

    async def _poll(self) -> AsyncIterator[MessageEvent]:
        backoff = BACKOFF_START_S
        failures = 0
        while True:
            try:
                updates = await self._call(
                    "getUpdates",
                    {
                        "offset": self._offset,
                        "timeout": int(self._timeout_s),
                        "allowed_updates": ["message", "edited_message"],
                    },
                    expect_list=True,
                )
                backoff, failures = BACKOFF_START_S, 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A network blip must not end the poll loop; the gateway is
                # expected to survive its upstream being briefly unreachable.
                failures += 1
                if failures > MAX_CONSECUTIVE_FAILURES:
                    # Past this it is not a blip -- a revoked token fails
                    # identically forever. Restart policy belongs to the
                    # supervisor, which reports the platform as failed rather
                    # than retrying invisibly until someone notices the bot
                    # went quiet three days ago.
                    detail = f"telegram polling failed {failures} times in a row: {exc}"
                    raise AdapterError(detail) from exc
                log.warning("telegram poll failed (%s); retrying in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2)
                continue

            if not updates:
                # A real getUpdates blocks server-side for `timeout` seconds,
                # so an empty result should be rare and slow. When something in
                # front of it answers instantly instead, this is what keeps the
                # loop from becoming a hot spin against the API.
                await asyncio.sleep(EMPTY_POLL_PAUSE_S)
                continue

            for update in updates:
                if not isinstance(update, dict):
                    continue
                event = self.normalize(update)
                if event is not None:
                    yield event
                # Advanced only after the event has been handed on, so a crash
                # replays the update rather than losing it.
                self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)

    def normalize(self, update: Mapping[str, Any]) -> MessageEvent | None:
        """Convert one Bot API update into a normalized event."""
        raw = update.get("message") or update.get("edited_message")
        if not isinstance(raw, dict):
            return None
        chat = raw.get("chat") or {}
        sender = raw.get("from") or {}
        text = str(raw.get("text") or raw.get("caption") or "")
        kind = _KIND.get(str(chat.get("type", "private")), ChatKind.PRIVATE)

        mentioned = kind is ChatKind.PRIVATE or self._addressed(text, raw)
        if mentioned and self.bot_username:
            text = text.replace(f"@{self.bot_username}", "").strip()

        return MessageEvent(
            platform=self.platform,
            chat_id=str(chat.get("id", "")),
            chat_kind=kind,
            sender=Sender(
                id=str(sender.get("id", "")),
                display_name=str(sender.get("username") or sender.get("first_name") or ""),
            ),
            text=text,
            received_at=datetime.fromtimestamp(float(raw.get("date", 0)), UTC),
            message_id=f"{chat.get('id')}:{raw.get('message_id')}",
            thread_id=str(raw.get("message_thread_id") or ""),
            mentioned=mentioned,
            raw=update,
        )

    def _addressed(self, text: str, raw: Mapping[str, Any]) -> bool:
        """Whether a group message was aimed at this bot."""
        if raw.get("reply_to_message", {}).get("from", {}).get("is_bot"):
            return True
        return bool(self.bot_username) and f"@{self.bot_username}" in text

    # -- transport -------------------------------------------------------------

    async def _call(
        self, method: str, payload: Mapping[str, Any], *, expect_list: bool = False
    ) -> Any:
        """One Bot API call, with the API's own error shape unwrapped."""
        if self._client is None:
            detail = "telegram adapter is not connected"
            raise AdapterError(detail)
        url = f"{self._api_root}/bot{self._token.reveal()}/{method}"
        response = await self._client.post(url, json=dict(payload))
        try:
            body = response.json()
        except ValueError as exc:
            detail = f"telegram {method} returned {response.status_code} with a non-JSON body"
            raise AdapterError(detail) from exc
        if not body.get("ok", False):
            # The description, never the URL: the URL contains the bot token.
            detail = f"telegram {method} failed: {body.get('description', response.status_code)}"
            raise AdapterError(detail)
        return body.get("result", [] if expect_list else {})
