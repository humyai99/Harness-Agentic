"""Deciding which conversation a message belongs to.

One function builds every session key, because the moment two places construct
them the two will disagree about a corner -- a forum topic, a thread, a group
where several people are talking -- and the symptom is a user's context silently
splitting in half or, far worse, two users sharing one.

The shape is ``agent:{agent}:{platform}:{kind}:{scope}``. It is greppable in
logs, sorts usefully, and its parts are recoverable, which matters because the
CLI lists sessions by key and an operator needs to read them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from harness_agentic.gateway.types import ChatKind, MessageEvent

SEPARATOR = ":"
_UNSAFE = re.compile(r"[^A-Za-z0-9._@=-]")


@dataclass(frozen=True, slots=True)
class KeyPolicy:
    """How one platform maps conversations onto sessions.

    ``group_per_sender`` is the consequential one. Sharing a session across a
    group is what makes the agent a participant in the conversation rather than
    a private assistant several people happen to be poking, and it is right for
    a Slack thread. It is wrong for a support account where two customers must
    never see each other's context, so it is a decision per platform, taken by
    the operator, not a default anyone inherits by accident.
    """

    agent: str = "main"
    group_per_sender: bool = False
    thread_scoped: bool = True
    """Threads get their own session. Slack's whole model; Telegram forums too."""


def build_session_key(event: MessageEvent, policy: KeyPolicy | None = None) -> str:
    """The session key for one inbound message."""
    rules = policy or KeyPolicy()
    kind = event.chat_kind
    # Each part is sanitized before joining, never after. Sanitizing the joined
    # string would let a chat literally named ``c1#t9`` collide with thread
    # ``t9`` of chat ``c1`` -- two different conversations, one session.
    scope = _safe(event.chat_id)

    if rules.thread_scoped and event.thread_id:
        kind = ChatKind.THREAD
        scope = f"{scope}#{_safe(event.thread_id)}"
    elif kind is not ChatKind.PRIVATE and rules.group_per_sender:
        scope = f"{scope}@{_safe(event.sender.id)}"

    return SEPARATOR.join(("agent", rules.agent, event.platform, kind.value, scope))


@dataclass(frozen=True, slots=True)
class ParsedKey:
    """A session key taken apart again, for display and filtering."""

    agent: str
    platform: str
    kind: str
    scope: str


def parse_session_key(key: str) -> ParsedKey | None:
    """Split a key back into its parts, or ``None`` if it is not one.

    Returning ``None`` rather than raising: keys come from a database that may
    predate a format change, and a listing command should show the odd one out
    rather than crash on it.
    """
    parts = key.split(SEPARATOR, 4)
    expected = 5
    if len(parts) != expected or parts[0] != "agent":
        return None
    return ParsedKey(agent=parts[1], platform=parts[2], kind=parts[3], scope=parts[4])


def _safe(value: str) -> str:
    """Make an identifier safe to embed in a colon-separated key.

    Platform ids are usually opaque and well-behaved, but Slack channel names
    and Telegram usernames are not, and a colon inside one would make the key
    unparseable in a way that only shows up months later.
    """
    cleaned = _UNSAFE.sub("_", value.strip())
    return cleaned or "unknown"
