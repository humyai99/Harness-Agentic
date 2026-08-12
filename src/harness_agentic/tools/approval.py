"""Who is allowed to say yes, and where.

This is where the organisation's "recommendation -> human review -> approval ->
execution" chain lives in code. The policy is per *surface*, which is the part
that matters: a developer at a terminal can be prompted, but a cron job at 3am
and a stranger messaging a public LINE account cannot be, so those surfaces
must not be able to escalate.

The defaults encode that. ``cli`` prompts, ``gateway`` allows only what an
allowlist names, ``cron`` denies outright. An unattended surface never gains a
capability the operator did not write down in advance.
"""

from __future__ import annotations

import fnmatch
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from harness_agentic.tools.spec import ApprovalRequest, Danger


class Mode(StrEnum):
    """How one surface handles a consequential call."""

    ALLOW = "allow"
    """Run without asking. Prints a warning banner at startup."""
    PROMPT = "prompt"
    """Ask a human, synchronously."""
    ALLOWLIST = "allowlist"
    """Run only what the allowlist matches; deny the rest."""
    DENY = "deny"
    """Refuse. The refusal is reported to the model as a tool error."""


DEFAULT_MODES: Mapping[str, Mode] = {
    "cli": Mode.PROMPT,
    "gateway": Mode.ALLOWLIST,
    "cron": Mode.DENY,
    "voice": Mode.ALLOWLIST,
    "web": Mode.PROMPT,
}

DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "git status",
    "git diff*",
    "git log*",
    "ls*",
    "cat *",
    "pytest*",
    "ruff*",
    "mypy*",
    "uv run pytest*",
    "uv run ruff*",
    "uv run mypy*",
)
"""Read-mostly commands, safe to run unattended. Deliberately narrow."""

Prompter = Callable[[ApprovalRequest], bool]
"""Asks a human. Returns whether the call may proceed."""


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """The outcome of one policy check."""

    granted: bool
    reason: str
    asked_human: bool = False


@dataclass
class ApprovalPolicy:
    """Decides whether a tool call may run on a given surface."""

    surface: str = "cli"
    modes: Mapping[str, Mode] = field(default_factory=lambda: dict(DEFAULT_MODES))
    allowlist: Sequence[str] = DEFAULT_ALLOWLIST
    prompter: Prompter | None = None
    auto_approve_below: Danger = Danger.NETWORK
    """Calls strictly below this level never need approval."""

    def mode(self) -> Mode:
        """The mode in force for this surface."""
        return self.modes.get(self.surface, Mode.DENY)

    def check(self, request: ApprovalRequest) -> ApprovalDecision:  # noqa: PLR0911
        """Decide whether ``request`` may proceed."""
        if request.danger < self.auto_approve_below:
            return ApprovalDecision(granted=True, reason="below the approval threshold")

        match self.mode():
            case Mode.ALLOW:
                return ApprovalDecision(granted=True, reason="surface set to allow")
            case Mode.DENY:
                return ApprovalDecision(
                    granted=False,
                    reason=(
                        f"{request.tool} needs approval and the {self.surface} surface "
                        f"cannot ask anyone"
                    ),
                )
            case Mode.ALLOWLIST:
                if matches_allowlist(request.summary, self.allowlist):
                    return ApprovalDecision(granted=True, reason="matched the allowlist")
                return ApprovalDecision(
                    granted=False,
                    reason=(
                        f"{request.tool} is not on the {self.surface} allowlist; "
                        f"add a pattern to tools.approval.allowlist to permit it"
                    ),
                )
            case Mode.PROMPT:
                if self.prompter is None:
                    return ApprovalDecision(
                        granted=False, reason="no prompter is attached to this surface"
                    )
                granted = self.prompter(request)
                return ApprovalDecision(
                    granted=granted,
                    reason="approved by operator" if granted else "declined by operator",
                    asked_human=True,
                )

    def for_surface(self, surface: str, *, prompter: Prompter | None = None) -> ApprovalPolicy:
        """Return the same policy bound to a different surface."""
        return ApprovalPolicy(
            surface=surface,
            modes=self.modes,
            allowlist=self.allowlist,
            prompter=prompter if prompter is not None else self.prompter,
            auto_approve_below=self.auto_approve_below,
        )


def matches_allowlist(command: str, patterns: Sequence[str]) -> bool:
    """Whether ``command`` matches an allowlist pattern.

    Anything that chains, redirects, or substitutes is refused before matching.
    ``git status && rm -rf /`` starts with an allowlisted prefix, and a naive
    glob would wave it through -- so a shell metacharacter disqualifies the
    whole command rather than being matched around.
    """
    text = command.strip()
    if not text:
        return False
    if any(token in text for token in ("&&", "||", ";", "|", "`", "$(", ">", "<", "\n")):
        return False
    try:
        # Reject unbalanced quotes, which are their own kind of surprise.
        shlex.split(text)
    except ValueError:
        return False
    return any(fnmatch.fnmatch(text, pattern) for pattern in patterns)


def always_allow() -> ApprovalPolicy:
    """A policy that approves everything. Tests and explicit opt-in only."""
    return ApprovalPolicy(surface="test", modes={"test": Mode.ALLOW})


def always_deny() -> ApprovalPolicy:
    """A policy that refuses everything needing approval."""
    return ApprovalPolicy(surface="test", modes={"test": Mode.DENY})
