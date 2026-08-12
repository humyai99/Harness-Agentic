"""The chat gateway: many platforms, one agent, one process.

Nothing in here knows how the agent works, and nothing in the agent knows this
exists. The seam is :class:`~harness_agentic.agent.build.AgentBundle` on one
side and :class:`~harness_agentic.gateway.adapter.PlatformAdapter` on the other,
which is what lets a fix to tool dispatch reach Telegram and LINE at the same
moment it reaches the terminal.
"""

from __future__ import annotations

from harness_agentic.gateway.adapter import PlatformAdapter, QueueAdapter
from harness_agentic.gateway.authz import Authorizer, PlatformAuth
from harness_agentic.gateway.keys import KeyPolicy, build_session_key, parse_session_key
from harness_agentic.gateway.router import RouteOutcome, Router
from harness_agentic.gateway.runner import SessionActor, Turn
from harness_agentic.gateway.service import Gateway
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

__all__ = [
    "Authorizer",
    "Capabilities",
    "ChatKind",
    "DeliveryTarget",
    "Gateway",
    "KeyPolicy",
    "MessageEvent",
    "OutboundMessage",
    "PlatformAdapter",
    "PlatformAuth",
    "QueueAdapter",
    "RouteOutcome",
    "Router",
    "Sender",
    "SentRef",
    "SessionActor",
    "Transport",
    "Turn",
    "build_session_key",
    "parse_session_key",
]
