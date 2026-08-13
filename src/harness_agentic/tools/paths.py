"""Workspace containment and the deny-list.

Two layers, and they answer different questions. Containment asks "is this
inside the workspace?" and is enforced by the execution environment. The
deny-list asks "is this something the agent should never read even when it is
inside the workspace?" -- and that one matters more than it first looks.

A repository legitimately contains ``.env``, ``.git/config`` with a token in a
remote URL, and sometimes a stray key. All of those are inside the workspace,
so containment says yes. But an agent that reads them will paraphrase their
contents into a message, and that message goes to a provider and into a
transcript on disk. The deny-list is what stops that.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Sequence
from pathlib import PurePath, PurePosixPath

from harness_agentic.core.secrets import looks_like_credential
from harness_agentic.errors import PathOutsideWorkspace

DEFAULT_DENY_GLOBS: tuple[str, ...] = (
    # Credentials that live inside ordinary projects.
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/id_rsa",
    "**/id_ed25519",
    "**/.npmrc",
    "**/.pypirc",
    "**/.netrc",
    # A remote URL can carry a token, and hooks are executable.
    "**/.git/config",
    "**/.git/hooks/**",
    # Home-directory credential stores, reachable when the workspace is ~.
    "**/.ssh/**",
    "**/.aws/**",
    "**/.gnupg/**",
    "**/.kube/config",
    "**/.docker/config.json",
    # Our own state: secrets, transcripts, and agent-authored skills.
    "**/.harness/.env",
    "**/.harness/auth.json",
)
"""Paths the agent may not read, even inside the workspace."""

ALWAYS_ALLOW: tuple[str, ...] = (".env.example", "**/.env.example", "**/.env.sample")
"""Templates that exist precisely to be read; they hold no values."""

_KEY_PREFIXES = (
    "sk-",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "xoxr-",
    "AKIA",
    "AIza",
    "ya29.",
)
"""Prefixes worth warning about even when what follows is too short to be real."""
_TOKENS = re.compile(r"[\s,;\"'`()\[\]{}<>=]+")
"""Splits a message into tokens, so ``KEY=sk-…`` is seen and ``task-list`` is not."""


class PathPolicy:
    """Decides which paths a tool may touch."""

    def __init__(
        self,
        *,
        deny_globs: Sequence[str] = DEFAULT_DENY_GLOBS,
        allow_globs: Sequence[str] = ALWAYS_ALLOW,
        extra_deny: Iterable[str] = (),
    ) -> None:
        """Build a policy from deny and allow patterns."""
        self._deny = _expand((*deny_globs, *extra_deny))
        self._allow = _expand(allow_globs)

    def is_denied(self, path: PurePath, *, root: PurePath) -> bool:
        """Whether reading or writing ``path`` is forbidden by policy.

        Matching happens on the workspace-relative path with forward slashes,
        so one pattern set behaves the same on every platform.
        """
        relative = _relative_posix(path, root)
        if relative is None:
            return True
        candidates = (relative, PurePosixPath(relative).name)
        if _matches_any(candidates, self._allow):
            return False
        return _matches_any(candidates, self._deny)

    def check(self, path: PurePath, *, root: PurePath) -> None:
        """Raise unless ``path`` is permitted."""
        if self.is_denied(path, root=root):
            msg = (
                f"{path} is on the deny-list; it may hold credentials. "
                f"Adjust tools.path_policy in config if this is wrong."
            )
            raise PathOutsideWorkspace(msg)


def _expand(patterns: Sequence[str]) -> tuple[str, ...]:
    """Add a bare form for every ``**/`` pattern.

    ``fnmatch`` is not glob: its ``*`` already crosses directory separators, so
    ``**/.git/config`` demands at least one leading component and quietly fails
    to match ``.git/config`` sitting at the workspace root -- which is the case
    that matters most. Every ``**/X`` therefore also matches bare ``X``.
    """
    expanded: list[str] = []
    for pattern in patterns:
        expanded.append(pattern)
        if pattern.startswith("**/"):
            expanded.append(pattern.removeprefix("**/"))
    return tuple(dict.fromkeys(expanded))


def _matches_any(candidates: Sequence[str], patterns: Sequence[str]) -> bool:
    """Whether any candidate matches any pattern, ignoring case.

    Case-folded on every platform. ``fnmatch.fnmatch`` normalizes case using
    ``os.path.normcase``, so it is case-insensitive on macOS and Windows and
    case-*sensitive* on Linux -- which meant ``key.PEM``, ``.ENV`` and
    ``ID_RSA`` were denied on a developer's Mac and readable on the Linux box
    the gateway runs on. A deny-list whose coverage depends on the host
    filesystem is the wrong kind of surprise, and every pattern here is a
    lowercase convention rather than an exact filename.
    """
    folded = [candidate.lower() for candidate in candidates]
    return any(
        fnmatch.fnmatchcase(candidate, pattern.lower())
        for pattern in patterns
        for candidate in folded
    )


def _relative_posix(path: PurePath, root: PurePath) -> str | None:
    """Express ``path`` relative to ``root`` using forward slashes."""
    try:
        return PurePosixPath(*PurePath(path).relative_to(root).parts).as_posix()
    except ValueError:
        return None


def looks_like_secret(text: str) -> bool:
    """Cheap check for a credential-shaped string.

    Used to warn when one appears in *user input*, so the operator is told to
    rotate it rather than having it silently forwarded to a provider. Not a
    replacement for redaction, which runs over tool output and logs.

    This used to test ``startswith`` against a list of prefixes, which meant it
    only fired on a message that was *nothing but* a key. "here is my key: sk-…"
    and "ANTHROPIC_API_KEY=sk-…" both went to the provider and into the
    transcript with no warning -- and a sentence around the key is what a real
    paste looks like. The shapes now come from
    :mod:`harness_agentic.core.secrets`, so there is one definition rather than
    three that drift.

    Deliberately the most eager of the three users of those shapes. Redaction and
    the validator's blocking scan pay for a false positive -- a mangled tool
    result, a refused skill -- so they want the full-length shapes. Here the cost
    of being wrong is one line of yellow text, and the cost of being right and
    silent is a live credential in a transcript. So a known prefix on any *token*
    counts, even when what follows is too short to be a real key: tokenized
    rather than substring-matched, because "task-list" contains "sk-".
    """
    if looks_like_credential(text):
        return True
    return any(token.startswith(_KEY_PREFIXES) for token in _TOKENS.split(text))
