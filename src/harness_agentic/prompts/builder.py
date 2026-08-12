"""Assembling the system prompt in tiers.

The tier is the whole point. Prompt caching bills a cached prefix at a fraction
of the normal rate, but only while that prefix is byte-identical from one turn
to the next. Put a timestamp, a working directory, or a todo list near the top
and the cache misses on every single turn -- with no error, no warning, and a
bill several times larger than it should be. It is the most expensive mistake
available in this codebase and the easiest one to make.

So content is sorted into tiers by how often it changes, breakpoints go only at
tier boundaries, and :meth:`PromptBuilder.stable_digest` gives the loop
something to assert against for the life of a session. A golden test then
checks mechanically that nothing volatile leaked upwards, because a rule this
easy to break by accident cannot rely on reviewers noticing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from hashlib import sha256

from harness_agentic.core.types import SystemPrompt, SystemSegment, ToolSchema


class Tier(IntEnum):
    """How often a piece of the prompt changes.

    Lower changes less often, and everything cacheable sorts to the front.
    """

    IDENTITY = 0
    """Who the agent is. Changes when the operator edits SOUL.md."""
    GUIDANCE = 1
    """How to behave, and what the tools are. Changes when tools change."""
    KNOWLEDGE = 2
    """Skills catalog and memory. A frozen snapshot taken at session start."""
    CONTEXT = 3
    """Workspace facts: repo layout, language, conventions. Stable per session."""
    VOLATILE = 4
    """Time, cwd, todo list, budget. Changes every turn. Never cached."""


CACHEABLE_TIERS: frozenset[Tier] = frozenset(
    {Tier.IDENTITY, Tier.GUIDANCE, Tier.KNOWLEDGE, Tier.CONTEXT}
)
"""Tiers that may sit inside a cached prefix."""


@dataclass(frozen=True, slots=True)
class PromptFragment:
    """One labelled piece of the system prompt."""

    key: str
    tier: Tier
    text: str
    order: int = 100
    """Sort position within a tier. Ties break on key, so output is stable."""

    def __post_init__(self) -> None:
        """Reject a fragment that is cacheable in name only."""
        if not self.key:
            msg = "a prompt fragment needs a key"
            raise ValueError(msg)


@dataclass
class PromptBuilder:
    """Sorts fragments into tiers and places cache breakpoints."""

    max_breakpoints: int = 4
    """Anthropic's ceiling. Extra markers are an API error, not a no-op."""

    _fragments: dict[str, PromptFragment] = field(default_factory=dict)

    # -- composition --------------------------------------------------------

    def add(self, fragment: PromptFragment) -> PromptBuilder:
        """Add or replace a fragment, keyed by name."""
        self._fragments[fragment.key] = fragment
        return self

    def add_text(self, key: str, tier: Tier, text: str, *, order: int = 100) -> PromptBuilder:
        """Add a fragment from raw text."""
        return self.add(PromptFragment(key=key, tier=tier, text=text, order=order))

    def extend(self, fragments: Sequence[PromptFragment]) -> PromptBuilder:
        """Add several fragments."""
        for fragment in fragments:
            self.add(fragment)
        return self

    def remove(self, key: str) -> PromptBuilder:
        """Drop a fragment if present."""
        self._fragments.pop(key, None)
        return self

    def fragments(self) -> tuple[PromptFragment, ...]:
        """Every fragment, in render order."""
        return tuple(sorted(self._fragments.values(), key=lambda f: (f.tier, f.order, f.key)))

    # -- rendering ----------------------------------------------------------

    def build(self) -> SystemPrompt:
        """Render the prompt, one segment per tier.

        Fragments are grouped by tier rather than emitted individually so a
        breakpoint always lands on a tier boundary. Emitting one segment per
        fragment would let a breakpoint fall mid-tier, which is how a prefix
        stops being a prefix.
        """
        grouped: dict[Tier, list[str]] = {}
        for fragment in self.fragments():
            if fragment.text.strip():
                grouped.setdefault(fragment.tier, []).append(fragment.text.strip())

        segments: list[SystemSegment] = []
        cacheable_tiers = [t for t in sorted(grouped) if t in CACHEABLE_TIERS]
        last_cacheable = cacheable_tiers[-1] if cacheable_tiers else None
        used = 0

        for tier in sorted(grouped):
            text = "\n\n".join(grouped[tier])
            # Only the final cacheable tier gets a breakpoint by default: one
            # marker at the boundary caches everything before it, and spending
            # the scarce markers on every tier buys nothing.
            mark = (
                tier is last_cacheable and tier in CACHEABLE_TIERS and used < self.max_breakpoints
            )
            if mark:
                used += 1
            segments.append(SystemSegment(text=text, cache_breakpoint=mark))

        return SystemPrompt(tuple(segments))

    def stable_digest(self) -> str:
        """A hash of everything that must not change during a session.

        The loop asserts this is constant across iterations. Prompt stability
        is not a style preference: a mid-conversation edit to the system prompt
        invalidates the cache and can make the model contradict what it already
        said earlier in the same conversation.
        """
        digest = sha256()
        for fragment in self.fragments():
            if fragment.tier in CACHEABLE_TIERS:
                digest.update(fragment.key.encode())
                digest.update(b"\0")
                digest.update(fragment.text.encode())
                digest.update(b"\0")
        return digest.hexdigest()[:16]

    def volatile_text(self) -> str:
        """Everything in the volatile tier, for tests that assert containment."""
        return "\n\n".join(f.text for f in self.fragments() if f.tier is Tier.VOLATILE)


# -- standard fragments -------------------------------------------------------


IDENTITY_DEFAULT = """\
You are an agent operating inside a user's workspace through a set of tools.

Work by doing, not by describing. When a question can be answered by reading a \
file or running a command, do that instead of speculating. Report what you \
actually observed, and say plainly when something failed or when you did not \
check.
"""


TOOL_GUIDANCE = """\
## Tools

Prefer a specific tool over a shell command when one exists: read_file over \
cat, glob_files over find, grep_files over grep. They are cheaper, their \
output is easier to read, and they do not need approval.

Some tools require approval and may be refused. A refusal comes back as a tool \
error -- treat it as a real answer, not a transient failure to retry. Say what \
you were trying to do and why it needed that access.

Call tools in parallel when they are independent. Read three files in one turn \
rather than three turns.
"""


def identity_fragment(text: str = IDENTITY_DEFAULT) -> PromptFragment:
    """The agent's persona, from ``SOUL.md`` or the default."""
    return PromptFragment(key="identity", tier=Tier.IDENTITY, text=text, order=0)


def tool_guidance_fragment(tools: Sequence[ToolSchema]) -> PromptFragment:
    """Guidance on using tools, listing the ones actually on offer.

    Sits in ``GUIDANCE`` rather than ``VOLATILE`` because the tool list is
    fixed for a session -- and if it were volatile, every turn would miss the
    cache.
    """
    if not tools:
        return PromptFragment(key="tools", tier=Tier.GUIDANCE, text="", order=10)
    names = ", ".join(sorted(tool.name for tool in tools))
    return PromptFragment(
        key="tools",
        tier=Tier.GUIDANCE,
        text=f"{TOOL_GUIDANCE}\nAvailable this session: {names}.",
        order=10,
    )


def workspace_fragment(root: str, *, notes: str = "") -> PromptFragment:
    """Facts about the workspace that hold for the whole session."""
    body = f"## Workspace\n\nThe workspace root is `{root}`."
    if notes:
        body += f"\n\n{notes}"
    return PromptFragment(key="workspace", tier=Tier.CONTEXT, text=body, order=10)


def volatile_fragment(*, now: str, cwd: str, extra: str = "") -> PromptFragment:
    """Per-turn facts.

    Everything here changes constantly, which is exactly why it is last. A
    golden test asserts none of it appears above the final cache breakpoint.
    """
    body = f"## Now\n\nCurrent time: {now}\nWorking directory: `{cwd}`"
    if extra:
        body += f"\n{extra}"
    return PromptFragment(key="volatile", tier=Tier.VOLATILE, text=body, order=0)
