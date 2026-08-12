"""Signature verification, authorization, rate limits, and deduplication.

These four are the gateway's whole security surface, and every one of them
fails open if it is subtly wrong -- a signature check that verifies re-serialized
JSON verifies nothing, a deduplicator that forgets too fast lets a deploy run
twice. So they are tested for what they must *refuse*, not only what they allow.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harness_agentic.errors import Unauthorized
from harness_agentic.gateway.authz import Authorizer, PlatformAuth
from harness_agentic.gateway.dedupe import Deduplicator
from harness_agentic.gateway.keys import KeyPolicy, build_session_key, parse_session_key
from harness_agentic.gateway.ratelimit import Limit, RateLimiter
from harness_agentic.gateway.signature import verify_line, verify_slack
from harness_agentic.gateway.types import ChatKind, MessageEvent, Sender
from harness_agentic.providers.credentials import Secret

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def event(
    *,
    text: str = "hello",
    sender: str = "u1",
    chat: str = "c1",
    kind: ChatKind = ChatKind.PRIVATE,
    platform: str = "fake",
    thread: str = "",
) -> MessageEvent:
    return MessageEvent(
        platform=platform,
        chat_id=chat,
        chat_kind=kind,
        sender=Sender(id=sender),
        text=text,
        received_at=NOW,
        thread_id=thread,
    )


# -- signatures ----------------------------------------------------------------


def line_signature(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def test_line_signature_accepts_the_real_thing() -> None:
    body = json.dumps({"events": []}).encode()
    secret = Secret("s3cr3t", source="test")
    verify_line(body, {"X-Line-Signature": line_signature(body, "s3cr3t")}, secret)


def test_line_signature_is_checked_against_raw_bytes() -> None:
    # The signature is over the bytes LINE sent. Re-serializing the parsed JSON
    # produces different bytes -- and a verifier that checks the round-tripped
    # form would accept anything it could parse.
    original = b'{"events":[],  "destination":"x"}'
    secret = Secret("s3cr3t", source="test")
    header = {"X-Line-Signature": line_signature(original, "s3cr3t")}
    reserialized = json.dumps(json.loads(original), separators=(",", ":")).encode()

    verify_line(original, header, secret)
    with pytest.raises(Unauthorized):
        verify_line(reserialized, header, secret)


def test_line_signature_rejects_a_wrong_key_and_a_missing_header() -> None:
    body = b"{}"
    secret = Secret("s3cr3t", source="test")
    with pytest.raises(Unauthorized):
        verify_line(body, {"X-Line-Signature": line_signature(body, "wrong")}, secret)
    with pytest.raises(Unauthorized, match="missing"):
        verify_line(body, {}, secret)


def test_slack_signature_refuses_a_replay() -> None:
    body = b"token=x&team_id=T1"
    secret = Secret("shh", source="test")
    timestamp = "1700000000"
    base = b"v0:" + timestamp.encode() + b":" + body
    signature = "v0=" + hmac.new(b"shh", base, hashlib.sha256).hexdigest()
    headers = {"X-Slack-Signature": signature, "X-Slack-Request-Timestamp": timestamp}

    verify_slack(body, headers, secret, now=1700000010)
    with pytest.raises(Unauthorized, match="refusing a replay"):
        verify_slack(body, headers, secret, now=1700009999)


def test_header_lookup_is_case_insensitive() -> None:
    body = b"{}"
    secret = Secret("s3cr3t", source="test")
    verify_line(body, {"x-line-signature": line_signature(body, "s3cr3t")}, secret)


# -- authorization -------------------------------------------------------------


def test_an_unknown_sender_is_denied_by_default() -> None:
    decision = Authorizer().check(event())
    assert not decision
    assert decision.layer == "default-deny"


def test_each_allow_layer_works() -> None:
    auth = Authorizer(platforms={"fake": PlatformAuth(allowed_senders=frozenset({"u1"}))})
    assert auth.check(event(sender="u1")).layer == "sender-allowlist"
    assert not auth.check(event(sender="u2"))

    chat_auth = Authorizer(platforms={"fake": PlatformAuth(allowed_chats=frozenset({"c1"}))})
    assert chat_auth.check(event(sender="anyone")).layer == "chat-allowlist"

    open_auth = Authorizer(global_allow_all=True)
    assert open_auth.check(event()).layer == "global-open"


def test_pairing_enrols_a_sender_once() -> None:
    auth = Authorizer(platforms={"fake": PlatformAuth()})
    code = auth.issue_code("fake", now=NOW)

    assert auth.redeem(event(), code.code, now=NOW)
    assert auth.check(event()).layer == "paired"
    # Single use: the same code must not enrol a second person.
    assert not auth.redeem(event(sender="u2"), code.code, now=NOW)
    assert not auth.check(event(sender="u2"))


def test_pairing_is_refused_in_a_group() -> None:
    # A code pasted into a group would enrol whoever typed it first, which is
    # not who the operator invited.
    auth = Authorizer(platforms={"fake": PlatformAuth()})
    code = auth.issue_code("fake", now=NOW)
    decision = auth.redeem(event(kind=ChatKind.GROUP), code.code, now=NOW)
    assert not decision
    assert "direct message" in decision.reason


def test_pairing_codes_expire() -> None:
    auth = Authorizer(platforms={"fake": PlatformAuth()})
    code = auth.issue_code("fake", now=NOW, ttl=timedelta(hours=1))
    decision = auth.redeem(event(), code.code, now=NOW + timedelta(hours=2))
    assert not decision
    assert "expired" in decision.reason


def test_a_code_for_another_platform_does_not_cross_over() -> None:
    auth = Authorizer(platforms={"fake": PlatformAuth(), "line": PlatformAuth()})
    code = auth.issue_code("line", now=NOW)
    assert not auth.redeem(event(platform="fake"), code.code, now=NOW)


def test_pairing_survives_a_restart(tmp_path: Path) -> None:
    state = tmp_path / "pairing.json"
    first = Authorizer(platforms={"fake": PlatformAuth()}, state_path=state)
    code = first.issue_code("fake", now=NOW, make_admin=True)
    first.redeem(event(), code.code, now=NOW)

    revived = Authorizer(platforms={"fake": PlatformAuth()}, state_path=state)
    assert revived.check(event()).layer == "paired"
    assert revived.is_admin("fake", "u1")
    assert state.stat().st_mode & 0o077 == 0


def test_a_non_ascii_pairing_attempt_is_refused_not_a_crash() -> None:
    """The bug: ``hmac.compare_digest`` raises ``TypeError`` on non-ASCII text.

    The candidate is whatever an unauthenticated stranger typed after ``/pair``,
    so ``/pair สวัสดี`` raised out of the router instead of being answered -- a
    remote sender crashing the handler with one message, and on a Thai deployment
    that is the ordinary case rather than an edge one. It needs an outstanding
    code to reach the comparison, which is exactly the state an operator is in
    while inviting somebody.
    """
    auth = Authorizer(platforms={"fake": PlatformAuth()})
    auth.issue_code("fake", now=NOW)

    for attempt in ("สวัสดี", "码", "café", "🙂"):
        decision = auth.redeem(event(), attempt, now=NOW)
        assert not decision, f"{attempt!r} should not authorize anybody"
        assert "not valid" in decision.reason

    # And a real code still works afterwards -- nothing was consumed.
    live = auth.pending_codes()
    assert len(tuple(live)) == 1
    assert auth.redeem(event(), next(iter(live)).code, now=NOW)


def test_open_configuration_is_reported() -> None:
    auth = Authorizer(platforms={"line": PlatformAuth(allow_all=True)}, global_allow_all=True)
    warnings = auth.warnings()
    assert len(warnings) == 2
    assert any("line" in w for w in warnings)


# -- rate limiting -------------------------------------------------------------


def test_a_burst_is_allowed_then_refused() -> None:
    limiter = RateLimiter(messages=Limit(capacity=3, per_second=1.0))
    assert all(limiter.check_message("u1", now=0.0) for _ in range(3))
    verdict = limiter.check_message("u1", now=0.0)
    assert not verdict
    assert 0 < verdict.retry_after_s <= 1.0
    assert "Try again in about" in verdict.message()


def test_the_bucket_refills_over_time() -> None:
    limiter = RateLimiter(messages=Limit(capacity=2, per_second=1.0))
    limiter.check_message("u1", now=0.0)
    limiter.check_message("u1", now=0.0)
    assert not limiter.check_message("u1", now=0.0)
    assert limiter.check_message("u1", now=1.5)


def test_senders_have_separate_buckets() -> None:
    limiter = RateLimiter(messages=Limit(capacity=1, per_second=0.1))
    assert limiter.check_message("u1", now=0.0)
    assert not limiter.check_message("u1", now=0.0)
    assert limiter.check_message("u2", now=0.0)


def test_a_refunded_turn_is_not_charged() -> None:
    limiter = RateLimiter(turns=Limit(capacity=1, per_second=0.001))
    assert limiter.check_turn("u1", now=0.0)
    assert not limiter.check_turn("u1", now=0.0)
    limiter.refund_turn("u1", now=0.0)
    assert limiter.check_turn("u1", now=0.0)


def test_exempt_senders_bypass_both_buckets() -> None:
    limiter = RateLimiter(messages=Limit(capacity=1, per_second=0.0), exempt=frozenset({"ops"}))
    assert all(limiter.check_message("ops", now=0.0) for _ in range(50))


def test_refilled_buckets_are_pruned() -> None:
    limiter = RateLimiter(messages=Limit(capacity=2, per_second=1.0))
    limiter.check_message("u1", now=0.0)
    assert limiter.tracked() == 1
    assert limiter.prune(now=100.0) == 1
    assert limiter.tracked() == 0


# -- deduplication -------------------------------------------------------------


def test_a_redelivery_is_recognised() -> None:
    dedupe = Deduplicator()
    assert not dedupe.seen("line", "msg-1", now=0.0)
    assert dedupe.seen("line", "msg-1", now=1.0)
    assert dedupe.duplicates == 1


def test_ids_are_namespaced_by_platform() -> None:
    dedupe = Deduplicator()
    assert not dedupe.seen("line", "1", now=0.0)
    assert not dedupe.seen("telegram", "1", now=0.0)


def test_records_expire_and_are_bounded() -> None:
    dedupe = Deduplicator(ttl_s=10.0, capacity=3)
    dedupe.seen("p", "old", now=0.0)
    assert not dedupe.seen("p", "old", now=100.0)

    fresh = Deduplicator(capacity=2)
    for index in range(4):
        fresh.seen("p", str(index), now=0.0)
    assert len(fresh) == 2


def test_an_event_with_no_id_is_always_new() -> None:
    # Polling transports deduplicate by offset and have nothing to key on;
    # refusing those would drop real messages.
    dedupe = Deduplicator()
    assert not dedupe.seen("telegram", "", now=0.0)
    assert not dedupe.seen("telegram", "", now=0.0)


def test_forgetting_lets_a_failed_accept_retry() -> None:
    dedupe = Deduplicator()
    dedupe.seen("line", "m1", now=0.0)
    dedupe.forget("line", "m1")
    assert not dedupe.seen("line", "m1", now=0.0)


# -- session keys --------------------------------------------------------------


def test_keys_round_trip() -> None:
    key = build_session_key(event())
    assert key == "agent:main:fake:private:c1"
    parsed = parse_session_key(key)
    assert parsed is not None
    assert (parsed.platform, parsed.kind, parsed.scope) == ("fake", "private", "c1")


def test_a_thread_gets_its_own_session() -> None:
    key = build_session_key(event(kind=ChatKind.CHANNEL, thread="t9"))
    assert key == "agent:main:fake:thread:c1#t9"


def test_a_group_can_be_shared_or_split_per_sender() -> None:
    shared = build_session_key(event(kind=ChatKind.GROUP))
    split = build_session_key(event(kind=ChatKind.GROUP), KeyPolicy(group_per_sender=True))
    assert shared != split
    assert split.endswith("c1@u1")


def test_hostile_identifiers_cannot_break_the_key_format() -> None:
    key = build_session_key(event(chat="c:1:evil"))
    assert parse_session_key(key) is not None
    assert key.count(":") == 4


def test_a_key_that_is_not_one_returns_none() -> None:
    assert parse_session_key("garbage") is None
