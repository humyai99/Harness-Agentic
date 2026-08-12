"""What a credential looks like, defined once.

These shapes were written out three times -- in the skill validator's safety
scan, in the reflection pass's redaction, and in the check that warns an
operator who has pasted a key into the chat. Three copies meant three different
coverages, and the weakest one decided what actually leaked: the operator
warning only matched a key at the *start* of the message, so "here is my key:
sk-…" went to the provider and into the transcript without a word.

A new provider prefix has to be worth adding in one place, or it gets added in
one place and the other two quietly stay behind.

Deliberately shape-based, and deliberately not entropy-based. High-entropy
heuristics fire on git hashes, base64 test fixtures and minified assets, and a
check that cries wolf is a check somebody switches off. What is here is
specific enough to act on.
"""

from __future__ import annotations

import re

CREDENTIAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "<redacted-key>"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"), "<redacted-token>"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "<redacted-token>"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "<redacted-token>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<redacted-key>"),
    (re.compile(r"AIza[0-9A-Za-z_-]{30,}"), "<redacted-key>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "<redacted-key>"),
    # A JWT: three dot-separated segments, the first announcing itself as JSON.
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}"), "<redacted-jwt>"),
)
"""Credential shapes, with what to replace each one with when redacting."""


def looks_like_credential(text: str) -> bool:
    """Whether ``text`` contains something shaped like a credential.

    *Contains*, not starts with. A person pasting a key almost always writes a
    sentence around it, and anchoring at the start missed every one of those --
    which is the entire population of real cases.
    """
    return any(pattern.search(text) for pattern, _ in CREDENTIAL_PATTERNS)


def redact_credentials(text: str) -> str:
    """Replace every credential shape in ``text`` with a marker naming its kind."""
    for pattern, replacement in CREDENTIAL_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
