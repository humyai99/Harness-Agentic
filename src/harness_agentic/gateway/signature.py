"""Proving a webhook came from who it claims.

A webhook endpoint is a URL on the public internet that makes an agent do work.
Without a signature check it is an open command channel for anyone who learns
the path, so verification is not optional and not "add it before production" --
it is the first thing an adapter does with a request body.

Three rules, all of which have burned real systems:

* **Verify against the raw bytes.** Parse afterwards. A body that round-trips
  through ``json.loads``/``json.dumps`` has different bytes and a different
  digest, and the natural fix -- verifying the re-serialized form -- verifies
  nothing.
* **Compare in constant time.** ``==`` on digests leaks the correct prefix.
* **Bound the timestamp.** Without it a captured request replays forever.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping
from typing import TYPE_CHECKING

from harness_agentic.errors import Unauthorized

if TYPE_CHECKING:
    from harness_agentic.core.secrets import Secret

MAX_SKEW_S = 300.0
"""Slack's own recommendation, and a sane default for anyone else."""


def verify_line(body: bytes, headers: Mapping[str, str], secret: Secret) -> None:
    """Check LINE's ``X-Line-Signature``: base64 of HMAC-SHA256 over the body."""
    provided = _header(headers, "x-line-signature")
    if not provided:
        raise Unauthorized("missing X-Line-Signature")
    digest = hmac.new(secret.reveal().encode(), body, hashlib.sha256).digest()
    if not hmac.compare_digest(base64.b64encode(digest).decode(), provided):
        raise Unauthorized("X-Line-Signature does not match the request body")


def verify_slack(
    body: bytes,
    headers: Mapping[str, str],
    secret: Secret,
    *,
    now: float,
    skew_s: float = MAX_SKEW_S,
) -> None:
    """Check Slack's ``v0=`` signature over ``v0:{timestamp}:{body}``."""
    provided = _header(headers, "x-slack-signature")
    timestamp = _header(headers, "x-slack-request-timestamp")
    if not provided or not timestamp:
        raise Unauthorized("missing Slack signature headers")
    try:
        sent_at = float(timestamp)
    except ValueError as exc:
        raise Unauthorized("Slack timestamp header is not a number") from exc
    if abs(now - sent_at) > skew_s:
        detail = f"Slack request is {abs(now - sent_at):.0f}s old; refusing a replay"
        raise Unauthorized(detail)

    basestring = b"v0:" + timestamp.encode() + b":" + body
    expected = "v0=" + hmac.new(secret.reveal().encode(), basestring, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, provided):
        raise Unauthorized("Slack signature does not match the request body")


def verify_secret_token(headers: Mapping[str, str], header: str, secret: Secret) -> None:
    """Check a shared-secret header, as Telegram's webhook mode uses.

    Weaker than a signature -- it proves the sender knows the secret, not that
    the body is untampered -- so it is only right where the platform offers
    nothing better, and only over TLS.
    """
    provided = _header(headers, header)
    if not provided or not hmac.compare_digest(provided, secret.reveal()):
        detail = f"{header} is missing or wrong"
        raise Unauthorized(detail)


def _header(headers: Mapping[str, str], name: str) -> str:
    """Case-insensitive header lookup.

    HTTP header names are case-insensitive and every server normalizes them
    differently; a lookup that assumes one casing works until the deployment
    changes.
    """
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return ""
