"""The skill test harness, and the catalog audit.

The audit is the part with no equivalent in the systems this design otherwise
follows, and it is the answer to the question those systems leave open: a
library that writes its own members has no way to notice when a *new* skill's
description starts capturing requests that belonged to an *existing* one. Each
skill looks fine in isolation. Routing degrades anyway, gradually, and nothing
reports it.

So every skill carries routing cases -- prompts it should be chosen for and,
just as importantly, prompts it should not -- and the audit replays all of them
against the whole catalog. Skill A stealing skill B's request is then a test
failure with both names in it, which is the only form in which that problem is
actionable.

Routing cases are cheap: they have no side effects and need no sandbox, so they
run on every proposal. Execution cases run a real agent in a container and are
only for skills that bundle scripts -- and when no sandbox is configured they
are recorded as SKIPPED, never as passed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from harness_agentic.skills.frontmatter import parse_frontmatter
from harness_agentic.skills.model import SkillMeta


class Outcome(StrEnum):
    """How one case ended."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Never reported as passed. A skipped execution case is unknown, not fine."""


class Tier(StrEnum):
    """What kind of case this is."""

    ROUTING = "routing"
    EXECUTION = "execution"


@dataclass(frozen=True, slots=True)
class TestCase:
    """One case from a skill's ``tests/cases.yaml``."""

    id: str
    tier: Tier
    prompt: str
    expect_selected: bool = True
    """Routing only. False means this prompt must *not* choose the skill."""
    expect_output_matches: str = ""
    forbid_output_matches: str = ""
    sandbox: str = ""
    timeout_s: float = 180.0


@dataclass(frozen=True, slots=True)
class CaseResult:
    """What happened to one case."""

    case_id: str
    skill: str
    outcome: Outcome
    detail: str = ""


@dataclass(frozen=True, slots=True)
class TestReport:
    """The results for one skill."""

    skill: str
    results: tuple[CaseResult, ...]

    @property
    def passed(self) -> bool:
        """Whether nothing failed. Skips do not count as failures."""
        return all(r.outcome is not Outcome.FAILED for r in self.results)

    @property
    def skipped(self) -> tuple[CaseResult, ...]:
        """Cases that could not run, so their status is unknown."""
        return tuple(r for r in self.results if r.outcome is Outcome.SKIPPED)

    def summary(self) -> str:
        """A one-line count, naming skips explicitly."""
        counts = {o: sum(1 for r in self.results if r.outcome is o) for o in Outcome}
        parts = [f"{counts[Outcome.PASSED]} passed"]
        if counts[Outcome.FAILED]:
            parts.append(f"{counts[Outcome.FAILED]} failed")
        if counts[Outcome.SKIPPED]:
            parts.append(f"{counts[Outcome.SKIPPED]} skipped (status unknown)")
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class Collision:
    """Two skills competing for the same request."""

    prompt: str
    expected: str
    selected: str
    case_id: str

    def describe(self) -> str:
        """A sentence naming both sides, which is what makes it fixable."""
        return (
            f"{self.expected!r} case {self.case_id!r} ({self.prompt!r}) now routes "
            f"to {self.selected!r} -- their descriptions overlap"
        )


@dataclass(frozen=True, slots=True)
class CollisionReport:
    """Everything the audit found."""

    collisions: tuple[Collision, ...]
    over_triggering: tuple[Collision, ...]
    """Negative cases that selected the skill they were written to avoid."""
    checked: int

    @property
    def clean(self) -> bool:
        """Whether the catalog routes as every skill expects."""
        return not self.collisions and not self.over_triggering

    def summary(self) -> str:
        """A human-readable report."""
        if self.clean:
            return f"{self.checked} routing case(s) checked, no collisions"
        lines = [f"{self.checked} routing case(s) checked"]
        lines += [f"  collision: {c.describe()}" for c in self.collisions]
        lines += [
            f"  over-trigger: {c.expected!r} fired on {c.prompt!r}, "
            f"which its 'Do not use when' excludes"
            for c in self.over_triggering
        ]
        return "\n".join(lines)


Router = Callable[[str, Sequence[SkillMeta]], str | None]
"""Chooses a skill for a prompt. A model in production, a stub in tests."""


def parse_cases(raw: str) -> list[TestCase]:
    """Read a ``tests/cases.yaml``."""
    if not raw.strip():
        return []
    parsed = parse_frontmatter(raw)
    entries = parsed.get("cases")
    if not isinstance(entries, list):
        return []
    cases: list[TestCase] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("prompt"):
            continue
        tier_raw = str(entry.get("tier", "routing"))
        cases.append(
            TestCase(
                id=str(entry.get("id") or f"case-{index + 1}"),
                tier=Tier(tier_raw) if tier_raw in tuple(Tier) else Tier.ROUTING,
                prompt=str(entry["prompt"]),
                expect_selected=bool(entry.get("expect_selected", True)),
                expect_output_matches=str(entry.get("expect_output_matches") or ""),
                forbid_output_matches=str(entry.get("forbid_output_matches") or ""),
                sandbox=str(entry.get("sandbox") or ""),
                timeout_s=float(entry.get("timeout_s", 180.0)),
            )
        )
    return cases


@dataclass
class SkillTestRunner:
    """Runs routing cases, and audits the catalog for collisions."""

    router: Router
    sandbox_available: bool = False
    _cases: dict[str, list[TestCase]] = field(default_factory=dict)

    def register(self, skill: str, cases_yaml: str) -> None:
        """Attach a skill's cases without reading them from disk again."""
        self._cases[skill] = parse_cases(cases_yaml)

    def run(
        self,
        skill: SkillMeta,
        cases_yaml: str,
        *,
        catalog: Sequence[SkillMeta],
        tiers: Sequence[Tier] = (Tier.ROUTING,),
    ) -> TestReport:
        """Run one skill's cases against a catalog that includes it."""
        results: list[CaseResult] = []
        for case in parse_cases(cases_yaml):
            if case.tier not in tiers:
                continue
            if case.tier is Tier.EXECUTION and not self.sandbox_available:
                results.append(
                    CaseResult(
                        case.id,
                        skill.name,
                        Outcome.SKIPPED,
                        "no sandbox configured; this case did not run",
                    )
                )
                continue
            results.append(self._run_routing(skill, case, catalog))
        return TestReport(skill=skill.name, results=tuple(results))

    def _run_routing(
        self, skill: SkillMeta, case: TestCase, catalog: Sequence[SkillMeta]
    ) -> CaseResult:
        """Ask the router which skill this prompt should reach for."""
        chosen = self.router(case.prompt, catalog)
        if case.expect_selected:
            if chosen == skill.name:
                return CaseResult(case.id, skill.name, Outcome.PASSED)
            return CaseResult(
                case.id,
                skill.name,
                Outcome.FAILED,
                f"expected {skill.name!r}, got {chosen!r}",
            )
        if chosen == skill.name:
            return CaseResult(
                case.id,
                skill.name,
                Outcome.FAILED,
                f"{skill.name!r} fired on a prompt it should ignore",
            )
        return CaseResult(case.id, skill.name, Outcome.PASSED)

    def audit(self, catalog: Sequence[SkillMeta]) -> CollisionReport:
        """Replay every registered case against the whole catalog.

        Run this after any change to the library, and in CI. It is the only
        thing that notices a skill quietly capturing another's requests, and
        that failure mode is what turns a large skill library from an asset
        into a liability.
        """
        collisions: list[Collision] = []
        over: list[Collision] = []
        checked = 0

        for name, cases in self._cases.items():
            for case in cases:
                if case.tier is not Tier.ROUTING:
                    continue
                checked += 1
                chosen = self.router(case.prompt, catalog)
                if case.expect_selected and chosen != name:
                    collisions.append(
                        Collision(
                            prompt=case.prompt,
                            expected=name,
                            selected=chosen or "nothing",
                            case_id=case.id,
                        )
                    )
                elif not case.expect_selected and chosen == name:
                    over.append(
                        Collision(
                            prompt=case.prompt,
                            expected=name,
                            selected=name,
                            case_id=case.id,
                        )
                    )

        return CollisionReport(
            collisions=tuple(collisions), over_triggering=tuple(over), checked=checked
        )


def keyword_router(prompt: str, catalog: Sequence[SkillMeta]) -> str | None:
    """A dependency-free stand-in for a model's routing decision.

    Scores descriptions by word overlap. Not what production should use -- the
    real router asks the model -- but it makes the audit runnable in CI without
    a key, and it catches the blunt collisions, which are most of them.
    """
    words = {w for w in _words(prompt) if len(w) > 2}  # noqa: PLR2004
    if not words:
        return None
    best: tuple[float, str] | None = None
    for meta in catalog:
        haystack = _words(f"{meta.name} {meta.description} {' '.join(meta.tags)}")
        overlap = len(words & haystack)
        if not overlap:
            continue
        score = overlap + (0.5 if any(w in meta.name for w in words) else 0.0)
        if best is None or score > best[0]:
            best = (score, meta.name)
    return best[1] if best else None


def _words(text: str) -> set[str]:
    """Lowercase word set."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))
