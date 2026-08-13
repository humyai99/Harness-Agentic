"""Who may talk to the agent.

Five layers, checked in order, and the last one is deny. That ordering is the
whole design: a gateway reachable from a public LINE account or a Telegram bot
username anyone can find is an internet-facing endpoint that spends money and
runs tools, so an unrecognised sender must be refused by default rather than
served by default.

The pairing flow exists because allowlisting by platform id is miserable --
nobody knows their own Telegram numeric id. Instead the operator generates a
code out of band, the user sends it in a *direct message*, and the id is
recorded. Codes are single-use and expire, and pairing is refused in group
chats: a code pasted into a group would enrol whoever read it fastest.
"""

from __future__ import annotations

import hmac
import json
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from harness_agentic.gateway.types import ChatKind, MessageEvent

CODE_TTL = timedelta(hours=24)
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
"""No I/L/O/0/1 -- codes get read aloud and typed on phones."""
CODE_LENGTH = 8


@dataclass(frozen=True, slots=True)
class AuthDecision:
    """Whether one sender may proceed, and why."""

    allowed: bool
    reason: str
    layer: str
    is_admin: bool = False

    def __bool__(self) -> bool:
        """Truthy when allowed, so call sites read naturally."""
        return self.allowed


@dataclass
class PlatformAuth:
    """The operator's rules for one platform."""

    allow_all: bool = False
    """Anyone may talk to the agent here. Warned about loudly at startup."""
    allowed_senders: frozenset[str] = frozenset()
    allowed_chats: frozenset[str] = frozenset()
    """Whole conversations, for a team channel where membership is the control."""
    admins: frozenset[str] = frozenset()
    pairing_enabled: bool = True


@dataclass(frozen=True, slots=True)
class PairingCode:
    """One outstanding invitation."""

    code: str
    platform: str
    issued_at: datetime
    expires_at: datetime
    note: str = ""
    make_admin: bool = False

    def is_live(self, now: datetime) -> bool:
        """Whether the code may still be redeemed."""
        return now < self.expires_at


@dataclass
class Authorizer:
    """The five-layer check, plus the pairing store that feeds layer three."""

    platforms: Mapping[str, PlatformAuth] = field(default_factory=dict)
    global_allow_all: bool = False
    state_path: Path | None = None
    _paired: dict[str, set[str]] = field(default_factory=dict)
    _admins: dict[str, set[str]] = field(default_factory=dict)
    _codes: dict[str, PairingCode] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Load previously paired senders, if a state path was given."""
        if self.state_path and self.state_path.exists():
            self._load()

    # -- the decision ---------------------------------------------------------

    def check(self, event: MessageEvent) -> AuthDecision:
        """Decide whether ``event`` may reach the agent."""
        rules = self.platforms.get(event.platform, PlatformAuth())
        sender = event.sender.id
        admin = sender in self._admins.get(event.platform, set()) or sender in rules.admins

        if rules.allow_all:
            return AuthDecision(
                True, f"{event.platform} is open to everyone", "platform-open", admin
            )
        if sender in rules.allowed_senders:
            return AuthDecision(True, "sender is on the allowlist", "sender-allowlist", admin)
        if event.chat_id in rules.allowed_chats:
            return AuthDecision(True, "chat is on the allowlist", "chat-allowlist", admin)
        if sender in self._paired.get(event.platform, set()):
            return AuthDecision(True, "sender completed pairing", "paired", admin)
        if self.global_allow_all:
            return AuthDecision(True, "the gateway is open to everyone", "global-open", admin)
        return AuthDecision(
            False,
            "not authorized; send the pairing code in a direct message to enrol",
            "default-deny",
        )

    def is_admin(self, platform: str, sender_id: str) -> bool:
        """Whether this sender may run administrative commands."""
        rules = self.platforms.get(platform, PlatformAuth())
        return sender_id in rules.admins or sender_id in self._admins.get(platform, set())

    # -- pairing --------------------------------------------------------------

    def issue_code(
        self,
        platform: str,
        *,
        now: datetime,
        note: str = "",
        make_admin: bool = False,
        ttl: timedelta = CODE_TTL,
    ) -> PairingCode:
        """Mint a single-use pairing code for one platform."""
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        record = PairingCode(
            code=code,
            platform=platform,
            issued_at=now,
            expires_at=now + ttl,
            note=note,
            make_admin=make_admin,
        )
        self._codes[code] = record
        self._save()
        return record

    def redeem(self, event: MessageEvent, code: str, *, now: datetime) -> AuthDecision:
        """Try to pair the sender of ``event`` using ``code``."""
        rules = self.platforms.get(event.platform, PlatformAuth())
        if not rules.pairing_enabled:
            return AuthDecision(False, "pairing is disabled on this platform", "pairing")
        if event.chat_kind is not ChatKind.PRIVATE:
            # A code visible to a group enrols whoever types it first, which is
            # not the person the operator meant to invite.
            return AuthDecision(
                False, "pairing codes may only be sent in a direct message", "pairing"
            )

        candidate = code.strip().upper()
        record = self._match(candidate)
        if record is None:
            return AuthDecision(False, "that pairing code is not valid", "pairing")
        if record.platform != event.platform:
            return AuthDecision(False, "that pairing code is for another platform", "pairing")
        if not record.is_live(now):
            del self._codes[record.code]
            self._save()
            return AuthDecision(False, "that pairing code has expired", "pairing")

        self._paired.setdefault(event.platform, set()).add(event.sender.id)
        if record.make_admin:
            self._admins.setdefault(event.platform, set()).add(event.sender.id)
        del self._codes[record.code]
        self._save()
        return AuthDecision(True, "paired", "pairing", is_admin=record.make_admin)

    def _match(self, candidate: str) -> PairingCode | None:
        """Find a code without leaking which prefix was right.

        Non-ASCII candidates are rejected before any comparison.
        :func:`hmac.compare_digest` raises ``TypeError`` on a ``str`` holding
        non-ASCII characters, and the candidate here is whatever an
        unauthenticated stranger typed after ``/pair`` -- so ``/pair สวัสดี``
        raised out of the router instead of being answered, which on a Thai
        deployment is the ordinary case rather than an edge one. Codes are drawn
        from :data:`CODE_ALPHABET`, so nothing outside ASCII can ever match and
        refusing early loses nothing.

        Still checked against every outstanding code, and without an early
        return, so the time taken does not depend on which one matched.
        """
        if not candidate.isascii():
            return None
        found: PairingCode | None = None
        for code, record in self._codes.items():
            if hmac.compare_digest(code, candidate):
                found = record
        return found

    def revoke(self, platform: str, sender_id: str) -> bool:
        """Un-pair a sender. Returns whether they were paired."""
        paired = self._paired.get(platform, set())
        existed = sender_id in paired
        paired.discard(sender_id)
        self._admins.get(platform, set()).discard(sender_id)
        self._save()
        return existed

    def paired_senders(self, platform: str) -> frozenset[str]:
        """Everyone who has paired on one platform."""
        return frozenset(self._paired.get(platform, set()))

    def pending_codes(self) -> Iterable[PairingCode]:
        """Codes issued and not yet redeemed."""
        return tuple(self._codes.values())

    def warnings(self) -> list[str]:
        """Configuration worth shouting about at startup."""
        notes: list[str] = []
        if self.global_allow_all:
            notes.append(
                "gateway.allow_all is on: anyone who finds this bot can run it. "
                "Use pairing or an allowlist instead."
            )
        notes.extend(
            f"{name}.allow_all is on: every sender on {name} is authorized."
            for name, rules in self.platforms.items()
            if rules.allow_all
        )
        return notes

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        assert self.state_path is not None  # noqa: S101 - guarded by the caller
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._paired = {k: set(v) for k, v in raw.get("paired", {}).items()}
        self._admins = {k: set(v) for k, v in raw.get("admins", {}).items()}
        self._codes = {
            code: PairingCode(
                code=code,
                platform=entry["platform"],
                issued_at=datetime.fromisoformat(entry["issued_at"]),
                expires_at=datetime.fromisoformat(entry["expires_at"]),
                note=entry.get("note", ""),
                make_admin=bool(entry.get("make_admin", False)),
            )
            for code, entry in raw.get("codes", {}).items()
        }

    def _save(self) -> None:
        if self.state_path is None:
            return
        payload = {
            "paired": {k: sorted(v) for k, v in self._paired.items() if v},
            "admins": {k: sorted(v) for k, v in self._admins.items() if v},
            "codes": {
                code: {
                    "platform": record.platform,
                    "issued_at": record.issued_at.isoformat(),
                    "expires_at": record.expires_at.isoformat(),
                    "note": record.note,
                    "make_admin": record.make_admin,
                }
                for code, record in self._codes.items()
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp.chmod(0o600)
        temp.replace(self.state_path)
