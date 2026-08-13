"""Discord, over the gateway websocket.

Fourth adapter and the third transport, which is the point of it: poll and
webhook both let the process stay stateless between messages, and a websocket
does not. There is a session to resume, a heartbeat to keep, a sequence number
to replay from, and a set of close codes where reconnecting is correct and
another set where reconnecting forever is how a bot gets its token revoked.

That protocol lives in :class:`GatewayProtocol` as a pure state machine: frames
in, frames out, no socket. Every branch that matters -- resume versus fresh
identify, a missed heartbeat acknowledgement, an invalid session, a fatal close
code -- is then a table-driven test rather than something only reproducible by
unplugging a network cable at the right moment.

Sending goes over the REST API, not the socket. Discord's gateway is for
receiving; messages are posted with HTTP, which also means the send path is
tested the same way every other adapter's is.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
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
    from harness_agentic.core.secrets import Secret

log = logging.getLogger(__name__)

API_ROOT = "https://discord.com/api/v10"
GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"

GUILD_MESSAGES = 1 << 9
DIRECT_MESSAGES = 1 << 12
MESSAGE_CONTENT = 1 << 15
DEFAULT_INTENTS = GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT
"""``MESSAGE_CONTENT`` is a privileged intent. Without it every message arrives
with an empty ``content`` and the bot looks broken in a way the API never
explains -- so it is requested by default and named here."""

DM_CHANNEL_TYPES = frozenset({1, 3})


class Op(IntEnum):
    """Gateway opcodes, named."""

    DISPATCH = 0
    HEARTBEAT = 1
    IDENTIFY = 2
    RESUME = 6
    RECONNECT = 7
    INVALID_SESSION = 9
    HELLO = 10
    HEARTBEAT_ACK = 11


FATAL_CLOSE_CODES = frozenset({4004, 4010, 4011, 4012, 4013, 4014})
"""Codes where reconnecting cannot help: bad token, bad shard, bad intents.

Retrying these is worse than useless -- repeated authentication failures are
what gets a bot token disabled, so the supervisor is told to stop instead."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One outbound gateway frame."""

    op: Op
    data: Any = None

    def encode(self) -> str:
        """Serialize for the socket."""
        return json.dumps({"op": int(self.op), "d": self.data})


@dataclass
class GatewayProtocol:
    """The Discord gateway handshake, as a pure state machine.

    Holds no socket and performs no I/O: :meth:`on_frame` takes a decoded frame
    and returns the frames to send back. The transport does nothing but move
    bytes and obey :attr:`heartbeat_interval_s`.
    """

    token: Secret
    intents: int = DEFAULT_INTENTS
    session_id: str = ""
    resume_url: str = ""
    seq: int | None = None
    heartbeat_interval_s: float = 0.0
    awaiting_ack: bool = False
    ready: bool = False

    def reset_session(self) -> None:
        """Forget the session, so the next connect identifies afresh."""
        self.session_id = ""
        self.resume_url = ""
        self.seq = None
        self.ready = False

    @property
    def can_resume(self) -> bool:
        """Whether there is a session worth resuming."""
        return bool(self.session_id) and self.seq is not None

    def connect_url(self) -> str:
        """Where to connect: the resume endpoint if there is a session."""
        if self.can_resume and self.resume_url:
            return f"{self.resume_url}/?v=10&encoding=json"
        return GATEWAY_URL

    def on_frame(self, frame: Mapping[str, Any]) -> list[Frame]:  # noqa: PLR0911
        """React to one inbound frame, returning what to send.

        One branch per opcode, each returning its own reply. Folding them
        together would hide which opcodes are handled, and an unhandled
        opcode here is a bot that silently stops receiving.
        """
        if (sequence := frame.get("s")) is not None:
            # Tracked on every dispatch, because a resume replays from it and
            # a heartbeat carries it.
            self.seq = int(sequence)

        match int(frame.get("op", -1)):
            case Op.HELLO:
                interval_ms = float((frame.get("d") or {}).get("heartbeat_interval", 41250))
                self.heartbeat_interval_s = interval_ms / 1000.0
                return [self._resume() if self.can_resume else self._identify()]
            case Op.HEARTBEAT:
                # The gateway may ask for one out of band.
                return [self.heartbeat()]
            case Op.HEARTBEAT_ACK:
                self.awaiting_ack = False
                return []
            case Op.RECONNECT:
                # Asked to move; the session stays valid, so resume.
                return []
            case Op.INVALID_SESSION:
                # ``d: true`` means the session may still be resumed.
                if not frame.get("d", False):
                    self.reset_session()
                return []
            case Op.DISPATCH:
                self._on_dispatch(frame)
                return []
            case _:
                return []

    def _on_dispatch(self, frame: Mapping[str, Any]) -> None:
        """Record what READY tells us about the session."""
        if frame.get("t") != "READY":
            return
        data = frame.get("d") or {}
        self.session_id = str(data.get("session_id", ""))
        self.resume_url = str(data.get("resume_gateway_url", ""))
        self.ready = True

    def heartbeat(self) -> Frame:
        """The next heartbeat, marking that an acknowledgement is owed."""
        self.awaiting_ack = True
        return Frame(Op.HEARTBEAT, self.seq)

    def zombied(self) -> bool:
        """Whether the last heartbeat went unacknowledged.

        A TCP connection can stay open long after the other end has stopped
        listening. Without this check the bot sits on a dead socket looking
        perfectly healthy, which is the failure everyone hits once.
        """
        return self.awaiting_ack

    def _identify(self) -> Frame:
        return Frame(
            Op.IDENTIFY,
            {
                "token": self.token.reveal(),
                "intents": self.intents,
                "properties": {"os": "linux", "browser": "harness-agentic", "device": "harness"},
            },
        )

    def _resume(self) -> Frame:
        return Frame(
            Op.RESUME,
            {"token": self.token.reveal(), "session_id": self.session_id, "seq": self.seq},
        )


def close_is_fatal(code: int | None) -> bool:
    """Whether a close code means stop rather than reconnect."""
    return code is not None and code in FATAL_CLOSE_CODES


class DiscordAdapter(PlatformAdapter):
    """Receives over the gateway websocket, sends over REST."""

    platform = "discord"
    transport = Transport.SOCKET
    capabilities = Capabilities(
        max_chars=2000,
        edit_messages=True,
        typing_indicator=True,
        min_edit_interval_s=1.2,
        markdown="commonmark",
        threads=True,
        max_edits_per_message=30,
    )

    def __init__(
        self,
        token: Secret,
        *,
        intents: int = DEFAULT_INTENTS,
        api_root: str = API_ROOT,
        client: httpx.AsyncClient | None = None,
        bot_user_id: str = "",
        require_mention_in_channels: bool = True,
        socket_factory: SocketFactory | None = None,
    ) -> None:
        """Create an adapter for one bot token."""
        self._token = token
        self._api_root = api_root.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self.bot_user_id = bot_user_id
        self.require_mention_in_channels = require_mention_in_channels
        self.protocol = GatewayProtocol(token=token, intents=intents)
        self._socket_factory = socket_factory

    # -- lifecycle -------------------------------------------------------------

    async def connect(self) -> None:
        """Open the REST client and confirm the token works.

        Checked here rather than on the socket: a bad token closes the gateway
        with 4004 and no explanation, whereas ``/users/@me`` says so plainly at
        startup.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        me = await self._call("GET", "/users/@me")
        self.bot_user_id = self.bot_user_id or str(me.get("id", ""))

    async def disconnect(self) -> None:
        """Close the REST client if this adapter opened it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- inbound ---------------------------------------------------------------

    def poll(self) -> AsyncIterator[MessageEvent]:
        """Yield messages from the gateway connection."""
        return self._run_socket()

    async def _run_socket(self) -> AsyncIterator[MessageEvent]:
        """Connect, handshake, heartbeat, and yield dispatched messages."""
        factory = self._socket_factory or _default_socket_factory
        socket = await factory(self.protocol.connect_url())
        try:
            async for raw in socket.frames(self.protocol):
                event = self.normalize(raw)
                if event is not None:
                    yield event
        finally:
            await socket.close()

    def normalize(self, frame: Mapping[str, Any]) -> MessageEvent | None:
        """Convert a ``MESSAGE_CREATE`` dispatch into a normalized event."""
        if int(frame.get("op", -1)) != Op.DISPATCH or frame.get("t") != "MESSAGE_CREATE":
            return None
        raw = frame.get("d") or {}
        author = raw.get("author") or {}
        if author.get("bot") or (self.bot_user_id and str(author.get("id")) == self.bot_user_id):
            # The agent's own message. Answering it would loop forever.
            return None

        channel = str(raw.get("channel_id") or "")
        is_dm = not raw.get("guild_id")
        text = str(raw.get("content") or "")
        mentioned = (
            is_dm
            or not self.require_mention_in_channels
            or any(str(m.get("id")) == self.bot_user_id for m in raw.get("mentions") or [])
        )
        if mentioned and self.bot_user_id:
            for form in (f"<@{self.bot_user_id}>", f"<@!{self.bot_user_id}>"):
                text = text.replace(form, "")
            text = text.strip()

        return MessageEvent(
            platform=self.platform,
            chat_id=channel,
            chat_kind=ChatKind.PRIVATE if is_dm else ChatKind.CHANNEL,
            sender=Sender(
                id=str(author.get("id") or ""),
                display_name=str(author.get("global_name") or author.get("username") or ""),
            ),
            text=text,
            received_at=_timestamp(raw.get("timestamp")),
            message_id=str(raw.get("id") or ""),
            # A thread in Discord is its own channel, so the channel id is
            # already the right session scope and no thread id is needed.
            mentioned=mentioned,
            raw=frame,
        )

    # -- outbound --------------------------------------------------------------

    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        """Post a message to a channel."""
        result = await self._call(
            "POST",
            f"/channels/{target.chat_id}/messages",
            {"content": message.text, "flags": 4096 if message.silent else 0},
        )
        return SentRef(
            platform=self.platform, chat_id=target.chat_id, message_id=str(result.get("id", ""))
        )

    async def edit(self, ref: SentRef, message: OutboundMessage) -> None:
        """Rewrite a message already posted."""
        await self._call(
            "PATCH",
            f"/channels/{ref.chat_id}/messages/{ref.message_id}",
            {"content": message.text},
        )

    async def typing(self, target: DeliveryTarget) -> None:
        """Show the typing indicator for about ten seconds."""
        await self._call("POST", f"/channels/{target.chat_id}/typing", {})

    async def _call(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        """One REST call, with Discord's error shape unwrapped."""
        if self._client is None:
            detail = "discord adapter is not connected"
            raise AdapterError(detail)
        response = await self._client.request(
            method,
            f"{self._api_root}{path}",
            json=dict(payload) if payload is not None else None,
            headers={"Authorization": f"Bot {self._token.reveal()}"},
        )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            detail = f"discord {method} {path} failed with {response.status_code}"
            raise AdapterError(detail)
        if not response.content:
            return {}
        try:
            body: Mapping[str, Any] = response.json()
        except ValueError:
            return {}
        return body


# -- the socket ------------------------------------------------------------------


class GatewaySocket:
    """What the protocol needs from a websocket. Implemented over ``websockets``.

    Narrow on purpose: a fake that yields recorded frames satisfies this in a
    dozen lines, which is what makes the reconnect and heartbeat paths testable
    without a network.
    """

    def frames(self, protocol: GatewayProtocol) -> AsyncIterator[Mapping[str, Any]]:
        """Run the connection, yielding every inbound frame."""
        raise NotImplementedError  # pragma: no cover - interface only

    async def send(self, frame: Frame) -> None:
        """Send one frame."""
        raise NotImplementedError  # pragma: no cover - interface only

    async def close(self) -> None:
        """Close the connection."""


SocketFactory = Any
"""``Callable[[str], Awaitable[GatewaySocket]]``, kept loose so a test can pass
a plain async function without importing the websocket library."""


@dataclass
class ScriptedSocket(GatewaySocket):
    """A socket that replays recorded frames. Ships for adapter authors.

    The gateway handshake is the part of Discord most likely to be got wrong
    and least likely to be exercised by hand, so the fake that drives it lives
    in the package rather than in the test directory.
    """

    inbound: Sequence[Mapping[str, Any]] = ()
    sent: list[Frame] = field(default_factory=list)
    closed: bool = False

    def frames(self, protocol: GatewayProtocol) -> AsyncIterator[Mapping[str, Any]]:
        """Feed each recorded frame through the protocol, then yield it."""
        return self._replay(protocol)

    async def _replay(self, protocol: GatewayProtocol) -> AsyncIterator[Mapping[str, Any]]:
        for frame in self.inbound:
            for reply in protocol.on_frame(frame):
                await self.send(reply)
            yield frame

    async def send(self, frame: Frame) -> None:
        """Record an outbound frame."""
        self.sent.append(frame)

    async def close(self) -> None:
        """Mark the socket closed."""
        self.closed = True

    def ops(self) -> list[Op]:
        """The opcodes sent, in order. What a handshake test asserts on."""
        return [frame.op for frame in self.sent]


async def _default_socket_factory(url: str) -> GatewaySocket:  # pragma: no cover - needs a socket
    """Open a real gateway connection using the optional ``websockets`` extra."""
    try:
        import websockets
    except ModuleNotFoundError as exc:
        detail = "discord needs the gateway extra: pip install 'harness-agentic[gateway]'"
        raise AdapterError(detail) from exc
    return _WebsocketsSocket(await websockets.connect(url, max_size=None))


class _WebsocketsSocket(GatewaySocket):  # pragma: no cover - needs a socket
    """The real transport: a websocket plus a heartbeat task."""

    def __init__(self, connection: Any) -> None:
        """Wrap an open connection."""
        self._connection = connection
        self._heartbeat: Any = None

    def frames(self, protocol: GatewayProtocol) -> AsyncIterator[Mapping[str, Any]]:
        """Read frames, answering the protocol's replies and keeping the beat."""
        return self._read(protocol)

    async def _read(self, protocol: GatewayProtocol) -> AsyncIterator[Mapping[str, Any]]:
        import asyncio

        async def beat() -> None:
            while protocol.heartbeat_interval_s > 0:
                await asyncio.sleep(protocol.heartbeat_interval_s)
                if protocol.zombied():
                    # The far end stopped acknowledging. Closing forces the
                    # supervisor to reconnect rather than sitting on a socket
                    # that will never deliver anything again.
                    await self._connection.close(code=4000)
                    return
                await self.send(protocol.heartbeat())

        async for message in self._connection:
            frame = json.loads(message)
            for reply in protocol.on_frame(frame):
                await self.send(reply)
            if protocol.heartbeat_interval_s > 0 and self._heartbeat is None:
                self._heartbeat = asyncio.create_task(beat())
            yield frame

        code = getattr(self._connection, "close_code", None)
        if close_is_fatal(code):
            detail = f"discord closed the gateway with {code}; reconnecting will not help"
            raise AdapterError(detail)

    async def send(self, frame: Frame) -> None:
        """Send one frame as JSON."""
        await self._connection.send(frame.encode())

    async def close(self) -> None:
        """Stop the heartbeat and close the connection."""
        if self._heartbeat is not None:
            self._heartbeat.cancel()
        await self._connection.close()


def _timestamp(raw: object) -> datetime:
    """Parse Discord's ISO-8601 timestamp, falling back to now."""
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return datetime.now(UTC)
