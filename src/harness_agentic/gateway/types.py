"""What a chat platform looks like once the platform is taken out of it.

Every adapter normalizes into these and denormalizes back out, so nothing above
this line knows whether a message arrived from Telegram's long-poll, a signed
LINE webhook, or a Discord websocket frame. The router, the authorizer, the
session actor, and the delivery layer all read the same shapes.

Two of the fields here exist because of a specific platform lesson:

* ``reply_to`` carries a platform-specific token that expires -- LINE's reply
  token is valid once, for a short window -- so the adapter must be free to
  discard it and fall back to a push API without anything upstream noticing.
* ``Capabilities.edit_messages`` is false for LINE, and that is not a detail.
  Streaming an answer by editing a placeholder is the obvious design and it
  simply cannot work there, so the delivery layer has to be built for both from
  the start rather than retrofitted the first time a message posts nine times.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, TypeAlias


class ChatKind(StrEnum):
    """Who else can read what the agent says."""

    PRIVATE = "private"
    """One human, one agent. The only place a pairing code may be typed."""
    GROUP = "group"
    CHANNEL = "channel"
    THREAD = "thread"
    """A conversation nested inside a channel. Slack's default, and its key."""


class Transport(StrEnum):
    """How events reach the process."""

    POLL = "poll"
    WEBHOOK = "webhook"
    SOCKET = "socket"


@dataclass(frozen=True, slots=True)
class Sender:
    """The human who sent a message, as the platform identifies them."""

    id: str
    display_name: str = ""
    is_admin: bool = False
    """Set by the authorizer from config, never by the platform payload."""

    def label(self) -> str:
        """A name for logs and audit lines."""
        return f"{self.display_name} ({self.id})" if self.display_name else self.id


@dataclass(frozen=True, slots=True)
class Attachment:
    """A file that came in with a message."""

    kind: Literal["image", "audio", "file"]
    media_type: str
    url: str = ""
    data: bytes | None = None
    name: str = ""


@dataclass(frozen=True, slots=True)
class MessageEvent:
    """One inbound message, normalized.

    ``raw`` is kept for adapters that need to answer with platform-specific
    machinery, and for recording fixtures. Nothing above the adapter reads it.
    """

    platform: str
    chat_id: str
    chat_kind: ChatKind
    sender: Sender
    text: str
    received_at: datetime
    message_id: str = ""
    """Used for deduplication. Webhook platforms redeliver on any non-200."""
    thread_id: str = ""
    reply_to: str = ""
    """A single-use reply token, where the platform has one."""
    attachments: tuple[Attachment, ...] = ()
    mentioned: bool = True
    """False in a group where the agent was not addressed; the router ignores it."""
    raw: Mapping[str, Any] = field(default_factory=dict)

    def is_command(self) -> bool:
        """Whether this is a slash command rather than something to answer."""
        return self.text.lstrip().startswith("/")


@dataclass(frozen=True, slots=True)
class DeliveryTarget:
    """Where a reply goes."""

    platform: str
    chat_id: str
    thread_id: str = ""
    reply_to: str = ""


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    """Something to say."""

    text: str
    kind: Literal["answer", "status", "error", "approval"] = "answer"
    """Surfaces that support it render these differently; the rest ignore it."""
    markdown: bool = True
    silent: bool = False
    """Deliver without a notification, for progress updates nobody asked for."""


@dataclass(frozen=True, slots=True)
class SentRef:
    """A handle to a message already posted, for platforms that allow edits."""

    platform: str
    chat_id: str
    message_id: str


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What one platform can actually do.

    The delivery layer reads this instead of branching on platform names, which
    is the same reason transports read ``ChatCompatQuirks`` instead of checking
    which provider they are talking to. A new adapter declares its limits and
    gets correct streaming behaviour with no changes anywhere else.
    """

    max_chars: int = 4000
    edit_messages: bool = False
    typing_indicator: bool = False
    min_edit_interval_s: float = 1.5
    """Platform rate limits punish faster edits than this."""
    markdown: Literal["none", "commonmark", "markdown_v2", "mrkdwn"] = "none"
    threads: bool = False
    max_edits_per_message: int = 40
    """After this many edits, finalize and start a new message."""


@dataclass(frozen=True, slots=True)
class WebhookResponse:
    """What to send back to the platform's delivery service.

    Status matters more than body: a non-2xx tells most platforms to redeliver,
    which is right for "I could not store this" and wrong for "this payload was
    nonsense", because the nonsense will be just as nonsensical next time.
    """

    status: int = 200
    body: bytes = b""
    content_type: str = "text/plain"


WebhookHandler: TypeAlias = Callable[[bytes, Mapping[str, str]], Awaitable[WebhookResponse]]
"""Handles one inbound HTTP request, given its raw body and headers."""


@dataclass(frozen=True, slots=True)
class Route:
    """An HTTP endpoint an adapter needs served.

    The handler takes the raw body and headers rather than a framework request
    object, so the whole webhook path -- signature check included -- is testable
    by calling a function, and no test needs a socket.
    """

    path: str
    handler: WebhookHandler
    methods: tuple[str, ...] = ("POST",)
