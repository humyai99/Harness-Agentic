"""Durable facts the agent carries between sessions.

Two files, because they answer different questions and go stale at different
rates. ``MEMORY.md`` holds facts about the work -- this project builds with
``uv``, staging is ``staging-1``, the tests need a running Postgres.
``USER.md`` holds facts about the person -- they prefer Thai, they want the diff
before the commit, they are on Windows. Splitting them means a new project does
not lose what was learned about the person, and a new colleague does not inherit
someone else's preferences.

**Memory is facts; a skill is a procedure.** That line decides which one a thing
belongs in. Memory is in the prompt on every single turn and is therefore paid
for on every single turn, so what earns a place is what would change the answer
to an arbitrary future question. A procedure, however useful, is loaded on
demand and belongs in a skill.

Two decisions carry the design.

**A hard character limit, enforced on write.** Not a warning, not a nudge --
:meth:`MemoryStore.add` refuses once the file is full and says which entries are
the oldest. Memory that grows without bound is the common failure: it costs
tokens on every turn forever, and nobody ever goes back to prune it, because
nothing ever forces the question. The limit is what turns "remember this too"
into "which of these matters more".

**Injected as a frozen snapshot.** The snapshot is taken once at session start.
A write lands on disk immediately but does not change the prompt until the next
session, because the memory block sits inside the cached prefix and editing it
mid-session invalidates the cache for every remaining turn. The cost of that
choice is that a fact just recorded is not visible in this conversation -- so
the tool says so in its answer rather than leaving the model to wonder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from harness_agentic.errors import HarnessError

MEMORY_FILE = "MEMORY.md"
USER_FILE = "USER.md"
MEMORY_LIMIT = 2_200
"""Characters. Roughly 550 tokens, paid on every turn of every session."""
USER_LIMIT = 1_375
MAX_ENTRY_CHARS = 400
"""One fact per entry. Longer than this is a procedure wearing a fact's clothes,
and it belongs in a skill where it is loaded only when relevant."""


class MemoryFull(HarnessError):
    """A memory file is at its limit and something has to go first."""


@dataclass(frozen=True, slots=True)
class Entry:
    """One remembered fact."""

    text: str
    section: str = ""

    def render(self) -> str:
        """The markdown line this entry is stored as."""
        return f"- {self.text}"


@dataclass
class MemoryFileState:
    """One memory file: its entries, its limit, and where it lives."""

    path: Path
    limit: int
    title: str
    entries: list[Entry] = field(default_factory=list)

    @property
    def used(self) -> int:
        """Characters this file costs in the prompt.

        Zero when nothing is recorded, rather than the length of a heading with
        no content under it: :meth:`MemoryStore.snapshot` omits an empty file
        entirely, so charging for it would report a cost nobody pays and make a
        fresh install look like it had already spent some of its budget.
        """
        if not self.entries:
            return 0
        return len(self.render())

    @property
    def remaining(self) -> int:
        """Characters still available."""
        return max(0, self.limit - self.used)

    def render(self) -> str:
        """The file as markdown, grouped by section.

        Sections keep their first-seen order rather than being sorted. A file a
        person also edits by hand should not reshuffle itself underneath them.
        """
        by_section: dict[str, list[Entry]] = {}
        for entry in self.entries:
            by_section.setdefault(entry.section, []).append(entry)

        lines = [f"# {self.title}", ""]
        for section, group in by_section.items():
            if section:
                lines.append(f"## {section}")
            lines.extend(entry.render() for entry in group)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class MemoryStore:
    """Reads, writes, and bounds the two memory files.

    Writes go straight to disk. The *prompt* uses :meth:`snapshot`, taken once at
    session start -- see the module docstring for why those are different things.
    """

    root: Path
    memory_limit: int = MEMORY_LIMIT
    user_limit: int = USER_LIMIT

    KINDS = ("memory", "user")

    def path_for(self, kind: str) -> Path:
        """Where one kind of memory is stored."""
        self._check_kind(kind)
        return self.root / (MEMORY_FILE if kind == "memory" else USER_FILE)

    def limit_for(self, kind: str) -> int:
        """The character ceiling for one kind."""
        self._check_kind(kind)
        return self.memory_limit if kind == "memory" else self.user_limit

    def load(self, kind: str) -> MemoryFileState:
        """Read one file from disk, or an empty state when it does not exist."""
        path = self.path_for(kind)
        state = MemoryFileState(
            path=path,
            limit=self.limit_for(kind),
            title="Project memory" if kind == "memory" else "About the user",
        )
        if path.is_file():
            state.entries = parse_entries(path.read_text(encoding="utf-8"))
        return state

    # -- writing ---------------------------------------------------------------

    def add(self, kind: str, text: str, *, section: str = "") -> Entry:
        """Record one fact, refusing once the file is full.

        The refusal names the oldest entries, because "make room" is not
        actionable without knowing what is in there -- and the model cannot see
        the file, only this message.
        """
        cleaned = _entry_text(text)
        state = self.load(kind)
        if any(entry.text == cleaned for entry in state.entries):
            # Silently succeeding on a duplicate would let the same fact be
            # re-added every session until it crowds out everything else.
            return Entry(text=cleaned, section=section)

        state.entries.append(Entry(text=cleaned, section=section))
        if state.used > state.limit:
            oldest = ", ".join(repr(e.text[:40]) for e in state.entries[:3])
            detail = (
                f"{state.path.name} is full ({state.limit} characters). Remove or "
                f"replace something first -- the oldest entries are: {oldest}"
            )
            raise MemoryFull(detail)

        self._write(state)
        return state.entries[-1]

    def replace(self, kind: str, old: str, new: str) -> bool:
        """Swap one entry's text. Returns whether it was found.

        Bounded exactly like :meth:`add`: a limit enforced on one write path and
        not the other is not a limit.
        """
        needle = _needle(old)
        cleaned = _entry_text(new)
        state = self.load(kind)
        for index, entry in enumerate(state.entries):
            if needle in entry.text:
                state.entries[index] = Entry(text=cleaned, section=entry.section)
                if state.used > state.limit:
                    detail = f"{state.path.name} would exceed {state.limit} characters"
                    raise MemoryFull(detail)
                self._write(state)
                return True
        return False

    def remove(self, kind: str, text: str) -> bool:
        """Delete the first entry containing ``text``."""
        needle = _needle(text)
        state = self.load(kind)
        for index, entry in enumerate(state.entries):
            if needle in entry.text:
                del state.entries[index]
                self._write(state)
                return True
        return False

    def _write(self, state: MemoryFileState) -> None:
        """Persist one file, at mode 0600.

        Memory holds whatever the agent was told about a person and a project,
        which is not secret in the credential sense and is nobody else's business
        either.
        """
        state.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = state.path.with_name(f".{state.path.name}.tmp")
        temporary.write_text(state.render(), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(state.path)

    def _check_kind(self, kind: str) -> None:
        if kind not in self.KINDS:
            detail = f"unknown memory {kind!r}; use one of {', '.join(self.KINDS)}"
            raise ValueError(detail)

    # -- reading ---------------------------------------------------------------

    def snapshot(self) -> str:
        """The prompt block, taken once at session start.

        Empty when nothing is remembered, so a fresh install does not spend a
        cache-breaking block on two headings and no content.
        """
        parts: list[str] = []
        for kind in self.KINDS:
            state = self.load(kind)
            if state.entries:
                parts.append(state.render().strip())
        if not parts:
            return ""
        body = "\n\n".join(parts)
        return (
            "## What you remember\n\n"
            "Facts carried over from earlier sessions. They were recorded by you or "
            "by the operator, and they may be out of date -- prefer what you can "
            "verify now.\n\n"
            f"{body}"
        )

    def usage(self) -> list[tuple[str, int, int]]:
        """``(kind, used, limit)`` for each file. What ``harn memory`` prints."""
        return [(kind, self.load(kind).used, self.limit_for(kind)) for kind in self.KINDS]


_BULLET = re.compile(r"^\s*[-*]\s+(?P<text>.+?)\s*$")
_HEADING = re.compile(r"^##\s+(?P<title>.+?)\s*$")


def parse_entries(raw: str) -> list[Entry]:
    """Read entries from a memory file.

    Tolerant on purpose: these files are meant to be edited by hand as well as by
    the agent, and a person who adds a paragraph of prose should not have their
    file silently emptied. Anything that is not a bullet is ignored rather than
    treated as an error.
    """
    entries: list[Entry] = []
    section = ""
    for line in raw.splitlines():
        heading = _HEADING.match(line)
        if heading:
            section = heading.group("title")
            continue
        bullet = _BULLET.match(line)
        if bullet:
            entries.append(Entry(text=bullet.group("text"), section=section))
    return entries


def _clean(text: str) -> str:
    """Normalize an entry to a single line.

    Collapsed rather than rejected: a multi-line paste is a normal thing to do,
    and the file format is one fact per bullet.
    """
    return " ".join(text.split()).lstrip("-*").strip()


def _entry_text(text: str) -> str:
    """Clean one fact, refusing what cannot be stored as one.

    Shared by ``add`` and ``replace`` so both write paths carry the same bound.
    """
    cleaned = _clean(text)
    if not cleaned:
        detail = "a memory entry cannot be empty"
        raise ValueError(detail)
    if len(cleaned) > MAX_ENTRY_CHARS:
        detail = (
            f"that entry is {len(cleaned)} characters, over the {MAX_ENTRY_CHARS} "
            f"limit for one fact. If it is a procedure rather than a fact, it "
            f"belongs in a skill, where it costs nothing until it is loaded."
        )
        raise ValueError(detail)
    return cleaned


def _needle(text: str) -> str:
    """Clean the text used to find an entry, refusing one that matches anything.

    ``"" in entry.text`` is true for every entry, so an empty needle would edit
    or delete whatever happens to be first and report success. Callers pass this
    straight from a model, which is exactly where a subtly wrong argument comes
    from.
    """
    cleaned = _clean(text)
    if not cleaned:
        detail = "the text identifying an entry cannot be empty"
        raise ValueError(detail)
    return cleaned
