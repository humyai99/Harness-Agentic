"""Staging, reviewing, and applying skill changes.

There is exactly one path by which a skill reaches disk: a proposal is staged,
validated, tested, and then approved. The agent has no write tool at all -- it
can only propose. One write path means one audit point and one place to enforce
quotas, and it means "the agent edited a skill" is never something that
happened without a record.

``autonomy`` defaults to ``propose``: nothing lands without a human. That is a
security decision rather than a UX one. A prompt store the agent can write is a
*persistent* injection surface -- a poisoned memory costs one session, while a
poisoned skill costs every future session whose request matches its
description.

The quotas exist because the failure mode of a self-writing library is not one
catastrophic skill, it is fifty mediocre ones. A cap on new skills per day and
on revisions per skill per week is what keeps the library something a person
can still review.
"""

from __future__ import annotations

import difflib
import json
import shutil
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness_agentic.constants import harness_home
from harness_agentic.errors import SkillQuotaExceeded
from harness_agentic.skills.validator import (
    CandidateSkill,
    DuplicateVerdict,
    ValidationReport,
    check_duplicate,
    validate,
)

if TYPE_CHECKING:
    from harness_agentic.skills.model import SkillMeta

SECONDS_PER_DAY = 86_400
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY


class Autonomy(StrEnum):
    """How much the agent may do without a human."""

    PROPOSE = "propose"
    """Default. Everything waits for review."""
    AUTO_SAFE = "auto-safe"
    """Auto-approve only clean, script-free, untainted proposals."""
    AUTO = "auto"
    """Approve on validation and tests alone. Trusted single-user machines."""


@dataclass
class Quotas:
    """Limits on how fast the library may change."""

    max_new_per_day: int = 5
    max_pending: int = 20
    """A review queue nobody will read is the same as no review."""
    max_revisions_per_week: int = 3
    revision_cooldown_s: float = SECONDS_PER_DAY
    """No re-revising a skill that was just revised."""


@dataclass(frozen=True, slots=True)
class SkillProposal:
    """A staged change to the library."""

    proposal_id: str
    kind: str
    """``create`` or ``patch``."""
    target_name: str
    rationale: str
    evidence: tuple[str, ...]
    content: str | None = None
    patches: tuple[tuple[str, str], ...] = ()
    """(old_string, new_string) pairs, for a surgical revision."""
    files: tuple[tuple[str, str], ...] = ()
    tests_yaml: str = ""
    version_bump: str = "patch"
    tainted: bool = False
    source_session_id: str = ""
    created_at: float = 0.0

    def patch_problems(self, existing_body: str) -> tuple[str, ...]:
        """Reasons this proposal's patches would not do what they claim.

        ``str.replace`` is silent when it finds nothing, so a patch whose
        ``old_string`` no longer matches used to stage clean, auto-approve, report
        "approved", and change absolutely nothing -- while recording a revision
        event, which put the skill under its cooldown. The trigger that produced
        the patch then kept firing on the same unfixed error and kept being
        refused as too recent. The loop dies for that skill and nothing says so.

        Ambiguity is refused for the same reason. ``replace(old, new, 1)`` takes
        the first occurrence, and "the first one" is not a choice a proposal
        should be making implicitly.
        """
        if self.content is not None:
            if not self.patches:
                return ()
            both = (
                "a proposal carrying full content must not also carry patches; "
                "the patches would be ignored"
            )
            return (both,)
        problems: list[str] = []
        for old, _ in self.patches:
            found = existing_body.count(old)
            if found == 0:
                problems.append(f"the text to replace is not in the current skill: {_excerpt(old)}")
            elif found > 1:
                problems.append(
                    f"the text to replace appears {found} times, so which one is "
                    f"meant is ambiguous: {_excerpt(old)}"
                )
        return tuple(problems)

    def candidate(self, *, existing_body: str = "") -> CandidateSkill:
        """Render this proposal as something the validator can check."""
        content = self.content
        if content is None:
            content = existing_body
            for old, new in self.patches:
                content = content.replace(old, new, 1)
        return CandidateSkill(
            name=self.target_name,
            content=content,
            files=self.files,
            tests_yaml=self.tests_yaml,
        )


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """Everything known about a staged proposal."""

    proposal: SkillProposal
    validation: ValidationReport
    duplicate: DuplicateVerdict
    auto_approvable: bool
    blockers: tuple[str, ...]

    def summary(self) -> str:
        """A reviewer-facing description."""
        lines = [
            f"{self.proposal.kind} {self.proposal.target_name} ({self.proposal.proposal_id})",
            f"  rationale: {self.proposal.rationale}",
        ]
        lines += [f"  evidence: {e}" for e in self.proposal.evidence]
        if self.duplicate.action != "create":
            lines.append(f"  duplicate: {self.duplicate.reason}")
        if not self.validation.clean:
            lines.append("  findings:")
            lines += [f"    {line}" for line in self.validation.summary().splitlines()]
        if self.blockers:
            lines += [f"  BLOCKED: {b}" for b in self.blockers]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """What happened when a proposal was approved."""

    applied: bool
    path: Path | None
    reason: str


def default_store(*, autonomy: Autonomy = Autonomy.PROPOSE) -> ProposalStore:
    """The staging area every surface shares.

    One factory because the two callers must agree on the directories. They did
    not: ``harn skills pending|diff|approve`` built a store to review proposals
    with, and no agent was ever given one -- so ``skill_propose`` was never
    registered and the review queue could only ever be empty. The library's
    headline feature had no entry point on any surface, and the command that
    would have shown that reported "nothing is waiting for review", which is
    what it says when the loop is working and idle.

    Autonomy is a parameter because the review commands must pin it to
    ``PROPOSE`` whatever a session was configured with: a review surface that
    could auto-apply what it was shown would not be one.
    """
    return ProposalStore(
        pending_dir=harness_home() / "skills-pending",
        skills_dir=harness_home() / "skills",
        autonomy=autonomy,
    )


def autonomy_from(value: str) -> Autonomy:
    """Read the configured autonomy, falling back to review on anything odd.

    An unrecognised value means review rather than an error: this setting decides
    whether an agent may write to its own prompt library without a human, and the
    safe reading of a typo is the strict one.
    """
    try:
        return Autonomy(value)
    except ValueError:
        return Autonomy.PROPOSE


class ProposalStore:
    """Holds staged proposals and is the only thing that writes skills."""

    def __init__(
        self,
        *,
        pending_dir: Path,
        skills_dir: Path,
        autonomy: Autonomy = Autonomy.PROPOSE,
        quotas: Quotas | None = None,
    ) -> None:
        """Bind a store to its staging and destination directories."""
        self._pending = pending_dir
        self._skills = skills_dir
        self.autonomy = autonomy
        self.quotas = quotas or Quotas()
        self._pending.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._skills.mkdir(mode=0o700, parents=True, exist_ok=True)

    # -- staging ------------------------------------------------------------

    def stage(self, proposal: SkillProposal, *, existing: Sequence[SkillMeta] = ()) -> ReviewResult:
        """Validate a proposal and hold it for review.

        Duplicate detection can rewrite a ``create`` into a ``patch`` of the
        skill it overlaps -- the mechanical decision, not the model's opinion
        of its own novelty.
        """
        self._enforce_pending_quota()

        duplicate = check_duplicate(proposal.candidate(), existing)
        if duplicate.action == "patch" and proposal.kind == "create" and duplicate.target:
            proposal = _as_patch(proposal, duplicate.target)
        elif duplicate.action == "reject":
            return ReviewResult(
                proposal=proposal,
                validation=validate(proposal.candidate()),
                duplicate=duplicate,
                auto_approvable=False,
                blockers=(f"duplicate of {duplicate.target!r}: {duplicate.reason}",),
            )

        if proposal.kind == "create":
            self._enforce_creation_quota()
        else:
            self._enforce_revision_quota(proposal.target_name)

        existing_body = self._read_body(proposal.target_name)
        validation = validate(proposal.candidate(existing_body=existing_body))

        blockers: list[str] = []
        if validation.blocked:
            blockers.append("validation blocked this proposal")
        # Before auto-approval decides anything: a patch that matches nothing
        # would land as a successful no-op and burn the revision budget.
        blockers.extend(proposal.patch_problems(existing_body))

        auto = self._auto_approvable(proposal, validation)
        self._write_staged(proposal, validation, duplicate)
        return ReviewResult(
            proposal=proposal,
            validation=validation,
            duplicate=duplicate,
            auto_approvable=auto and not blockers,
            blockers=tuple(blockers),
        )

    def _auto_approvable(self, proposal: SkillProposal, validation: ValidationReport) -> bool:
        """Whether this may land without a human, under the current autonomy."""
        if self.autonomy is Autonomy.PROPOSE:
            return False
        if validation.blocked:
            return False
        if proposal.tainted:
            # A session that read untrusted content is forced to review no
            # matter the setting. This is the self-poisoning path and the one
            # place where autonomy does not get a vote.
            return False
        if self.autonomy is Autonomy.AUTO:
            return True
        return validation.clean and not proposal.files

    # -- review -------------------------------------------------------------

    def pending(self) -> list[SkillProposal]:
        """Every staged proposal, oldest first."""
        out: list[SkillProposal] = []
        for directory in sorted(self._pending.iterdir()):
            path = directory / "proposal.json"
            if not path.is_file():
                continue
            try:
                out.append(_proposal_from_json(json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, KeyError, OSError):
                continue
        return sorted(out, key=lambda p: p.created_at)

    def get(self, proposal_id: str) -> SkillProposal | None:
        """One staged proposal."""
        path = self._pending / proposal_id / "proposal.json"
        if not path.is_file():
            return None
        try:
            return _proposal_from_json(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, OSError):
            return None

    def diff(self, proposal_id: str) -> str:
        """A unified diff of what approving this would change."""
        proposal = self.get(proposal_id)
        if proposal is None:
            return f"no proposal {proposal_id!r}"
        before = self._read_body(proposal.target_name)
        after = proposal.candidate(existing_body=before).content
        return (
            "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"{proposal.target_name} (current)",
                    tofile=f"{proposal.target_name} (proposed)",
                )
            )
            or "(no textual change)"
        )

    def reject(self, proposal_id: str, *, reason: str) -> bool:
        """Discard a proposal, recording why."""
        directory = self._pending / proposal_id
        if not directory.is_dir():
            return False
        (directory / "rejected.txt").write_text(reason, encoding="utf-8")
        shutil.rmtree(directory, ignore_errors=True)
        return True

    # -- applying -----------------------------------------------------------

    def approve(self, proposal_id: str, *, actor: str = "operator") -> ApplyResult:
        """Write a proposal to disk. The only write path in the system."""
        proposal = self.get(proposal_id)
        if proposal is None:
            return ApplyResult(False, None, f"no proposal {proposal_id!r}")

        existing_body = self._read_body(proposal.target_name)
        # Re-checked here and not only at staging time, because the skill may have
        # changed since -- and a patch that no longer matches must be refused
        # rather than written as an unchanged file reported as approved.
        if problems := proposal.patch_problems(existing_body):
            return ApplyResult(
                False, None, "this patch would change nothing: " + "; ".join(problems)
            )
        candidate = proposal.candidate(existing_body=existing_body)
        validation = validate(candidate)
        if validation.blocked:
            return ApplyResult(False, None, f"validation still blocks it:\n{validation.summary()}")

        target = self._skills / proposal.target_name
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(candidate.content, encoding="utf-8")
        for relative, content in proposal.files:
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if proposal.tests_yaml:
            (target / "tests").mkdir(exist_ok=True)
            (target / "tests" / "cases.yaml").write_text(proposal.tests_yaml, encoding="utf-8")

        self._record_provenance(target, proposal, actor=actor)
        self._record_quota_event(proposal)
        shutil.rmtree(self._pending / proposal_id, ignore_errors=True)
        return ApplyResult(True, target, f"approved by {actor}")

    def _record_provenance(self, target: Path, proposal: SkillProposal, *, actor: str) -> None:
        """Write the sidecar recording where this version came from."""
        sidecar = target / ".harness"
        sidecar.mkdir(exist_ok=True)
        path = sidecar / "provenance.json"
        history: list[dict[str, Any]] = []
        if path.is_file():
            try:
                history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
            except (ValueError, OSError):
                history = []
        history.append(
            {
                "proposal_id": proposal.proposal_id,
                "kind": proposal.kind,
                "actor": actor,
                "rationale": proposal.rationale,
                "evidence": list(proposal.evidence),
                "session": proposal.source_session_id,
                "tainted": proposal.tainted,
                "at": time.time(),
            }
        )
        path.write_text(
            json.dumps({"lifecycle": "active", "history": history}, indent=2),
            encoding="utf-8",
        )

    # -- quotas -------------------------------------------------------------

    def _quota_path(self) -> Path:
        return self._pending.parent / "skill-quota.json"

    def _quota_state(self) -> dict[str, list[float]]:
        path = self._quota_path()
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
        return {k: [float(t) for t in v] for k, v in raw.items()} if isinstance(raw, dict) else {}

    def _record_quota_event(self, proposal: SkillProposal) -> None:
        state = self._quota_state()
        key = "created" if proposal.kind == "create" else f"revised:{proposal.target_name}"
        state.setdefault(key, []).append(time.time())
        self._quota_path().write_text(json.dumps(state), encoding="utf-8")

    def _enforce_creation_quota(self) -> None:
        recent = _within(self._quota_state().get("created", []), SECONDS_PER_DAY)
        if len(recent) >= self.quotas.max_new_per_day:
            msg = (
                f"{len(recent)} skills already created today "
                f"(limit {self.quotas.max_new_per_day}); revise an existing one instead"
            )
            raise SkillQuotaExceeded(msg)

    def _enforce_revision_quota(self, name: str) -> None:
        events = self._quota_state().get(f"revised:{name}", [])
        recent = _within(events, SECONDS_PER_WEEK)
        if len(recent) >= self.quotas.max_revisions_per_week:
            # A skill being rewritten every other day and still not working is
            # a skill to disable, not one to rewrite again.
            msg = (
                f"{name!r} has been revised {len(recent)} times this week "
                f"(limit {self.quotas.max_revisions_per_week}); it may need "
                f"quarantining rather than another rewrite"
            )
            raise SkillQuotaExceeded(msg)
        if events and time.time() - max(events) < self.quotas.revision_cooldown_s:
            msg = f"{name!r} was revised recently; wait for the cooldown"
            raise SkillQuotaExceeded(msg)

    def _enforce_pending_quota(self) -> None:
        if len(self.pending()) >= self.quotas.max_pending:
            msg = (
                f"{self.quotas.max_pending} proposals are already awaiting review; "
                f"clear the queue before adding more"
            )
            raise SkillQuotaExceeded(msg)

    # -- storage ------------------------------------------------------------

    def _read_body(self, name: str) -> str:
        path = self._skills / name / "SKILL.md"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def _write_staged(
        self,
        proposal: SkillProposal,
        validation: ValidationReport,
        duplicate: DuplicateVerdict,
    ) -> None:
        directory = self._pending / proposal.proposal_id
        directory.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            **_proposal_to_json(proposal),
            "validation": [
                {"code": f.code, "severity": f.severity, "message": f.message}
                for f in validation.findings
            ],
            "duplicate": {
                "action": duplicate.action,
                "target": duplicate.target,
                "score": duplicate.score,
                "reason": duplicate.reason,
            },
        }
        (directory / "proposal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def new_proposal(
    *,
    kind: str,
    target_name: str,
    rationale: str,
    evidence: Sequence[str] = (),
    content: str | None = None,
    patches: Sequence[tuple[str, str]] = (),
    files: Sequence[tuple[str, str]] = (),
    tests_yaml: str = "",
    version_bump: str = "patch",
    tainted: bool = False,
    session_id: str = "",
) -> SkillProposal:
    """Build a proposal with an id and a timestamp."""
    return SkillProposal(
        proposal_id=f"p_{uuid.uuid4().hex[:8]}",
        kind=kind,
        target_name=target_name,
        rationale=rationale,
        evidence=tuple(evidence),
        content=content,
        patches=tuple(patches),
        files=tuple(files),
        tests_yaml=tests_yaml,
        version_bump=version_bump,
        tainted=tainted,
        source_session_id=session_id,
        created_at=time.time(),
    )


def _as_patch(proposal: SkillProposal, target: str) -> SkillProposal:
    """Rewrite a create into a patch of the skill it duplicates."""
    return SkillProposal(
        proposal_id=proposal.proposal_id,
        kind="patch",
        target_name=target,
        rationale=f"{proposal.rationale} (converted from create: overlaps {target!r})",
        evidence=proposal.evidence,
        content=proposal.content,
        patches=proposal.patches,
        files=proposal.files,
        tests_yaml=proposal.tests_yaml,
        version_bump=proposal.version_bump,
        tainted=proposal.tainted,
        source_session_id=proposal.source_session_id,
        created_at=proposal.created_at,
    )


def _within(timestamps: Sequence[float], window_s: float) -> list[float]:
    """Timestamps inside the trailing window."""
    cutoff = time.time() - window_s
    return [t for t in timestamps if t >= cutoff]


def _excerpt(text: str, *, limit: int = 60) -> str:
    """A short single-line quotation, for naming which patch failed."""
    flat = " ".join(text.split())
    return repr(flat if len(flat) <= limit else flat[:limit] + "…")


def _proposal_to_json(proposal: SkillProposal) -> dict[str, Any]:
    """Serialize a proposal."""
    return {
        "proposal_id": proposal.proposal_id,
        "kind": proposal.kind,
        "target_name": proposal.target_name,
        "rationale": proposal.rationale,
        "evidence": list(proposal.evidence),
        "content": proposal.content,
        "patches": [list(p) for p in proposal.patches],
        "files": [list(f) for f in proposal.files],
        "tests_yaml": proposal.tests_yaml,
        "version_bump": proposal.version_bump,
        "tainted": proposal.tainted,
        "source_session_id": proposal.source_session_id,
        "created_at": proposal.created_at,
    }


def _proposal_from_json(raw: dict[str, Any]) -> SkillProposal:
    """Deserialize a proposal."""
    return SkillProposal(
        proposal_id=str(raw["proposal_id"]),
        kind=str(raw.get("kind", "create")),
        target_name=str(raw["target_name"]),
        rationale=str(raw.get("rationale", "")),
        evidence=tuple(str(e) for e in raw.get("evidence") or []),
        content=raw.get("content"),
        patches=tuple((str(p[0]), str(p[1])) for p in raw.get("patches") or []),
        files=tuple((str(f[0]), str(f[1])) for f in raw.get("files") or []),
        tests_yaml=str(raw.get("tests_yaml", "")),
        version_bump=str(raw.get("version_bump", "patch")),
        tainted=bool(raw.get("tainted", False)),
        source_session_id=str(raw.get("source_session_id", "")),
        created_at=float(raw.get("created_at", 0.0)),
    )
