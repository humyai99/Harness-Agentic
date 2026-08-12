"""Gates a proposed skill has to pass.

Structural validation, a safety scan, and duplicate detection. All mechanical:
asking a model whether its own proposal is a duplicate reliably produces "no".

Duplicate detection is the load-bearing one. A library that grows by revision
stays useful; a library that grows by accretion collapses into forty
descriptions that all sort of match, and then nothing routes correctly. So a
proposal that looks like an existing skill is *converted* into a patch of that
skill rather than being rejected or accepted as new.
"""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from harness_agentic.skills.frontmatter import Finding, parse_frontmatter, read_fields
from harness_agentic.skills.model import (
    LINT_DESCRIPTION_CHARS,
    REQUIRED_SECTIONS,
    SkillMeta,
)


class Severity(IntEnum):
    """How badly a finding matters."""

    INFO = 0
    WARN = 1
    CAUTION = 2
    DANGEROUS = 3
    """Never overridable. Not by a flag, not by an operator."""


MAX_BODY_CHARS = 16_000
MAX_SCRIPTS = 10
MAX_DIR_BYTES = 256 * 1024

# Credential shapes. A skill that embeds one has either leaked a real key or is
# trying to smuggle one somewhere.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."),
)

# Instructions aimed at the agent rather than at the task. A skill is reference
# material; text like this is trying to be something else.
_INJECTION_PATTERNS = (
    re.compile(r"ignore (all |any )?(previous|prior|earlier) instructions", re.I),
    re.compile(r"disregard (your|the) (system prompt|guidance|rules)", re.I),
    re.compile(r"you (are|must) now (act as|behave as|become)", re.I),
    re.compile(r"do not (ask|request|seek) (for )?(permission|approval|confirmation)", re.I),
    re.compile(r"(auto[- ]?approve|approve (this|all) automatically)", re.I),
    re.compile(r"without (telling|informing|notifying) the user", re.I),
)

_DANGEROUS_SHELL = (
    re.compile(r"curl[^\n|]*\|\s*(ba)?sh"),
    re.compile(r"wget[^\n|]*\|\s*(ba)?sh"),
    re.compile(r"rm\s+-[rRf]{2,}\s+/(?!\w)"),
    re.compile(r"chmod\s+777"),
    re.compile(r"dd\s+.*of=/dev/"),
    re.compile(r">>\s*~?/?\.(bashrc|zshrc|profile)"),
    re.compile(r"nc\s+.*-e\s"),
    re.compile(r":\(\)\{.*\};:"),
)

_EXFIL_MODULES = frozenset({"socket", "ftplib", "smtplib", "telnetlib"})
_CREDENTIAL_PATHS = (".ssh", ".aws", ".gnupg", ".netrc", "credentials", "id_rsa")


@dataclass(frozen=True, slots=True)
class CandidateSkill:
    """A skill being proposed, before it exists on disk."""

    name: str
    content: str
    """The complete SKILL.md, frontmatter included."""
    files: tuple[tuple[str, str], ...] = ()
    """(relative path, content) for scripts and references."""
    tests_yaml: str = ""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Everything found about a candidate."""

    findings: tuple[Finding, ...]
    severity: Severity

    @property
    def blocked(self) -> bool:
        """Whether this candidate may not proceed under any autonomy setting."""
        return self.severity >= Severity.DANGEROUS or any(f.blocking for f in self.findings)

    @property
    def clean(self) -> bool:
        """Whether nothing at all was found."""
        return not self.findings

    def summary(self) -> str:
        """One line per finding."""
        return "\n".join(f"[{f.severity}] {f.code}: {f.message}" for f in self.findings)


def validate(candidate: CandidateSkill) -> ValidationReport:
    """Run every gate over a candidate."""
    findings: list[Finding] = []
    severity = Severity.INFO

    try:
        raw = parse_frontmatter(candidate.content)
    except Exception as exc:  # any parse failure is one finding, not a crash
        return ValidationReport(
            (Finding("SK000", "error", f"frontmatter did not parse: {exc}"),),
            Severity.WARN,
        )

    _, field_findings = read_fields(raw, dirname=candidate.name)
    findings.extend(field_findings)

    body = candidate.content.split("---", 2)[-1]
    findings.extend(_check_structure(body, candidate))
    secret_findings, secret_severity = _scan_secrets(candidate)
    findings.extend(secret_findings)
    severity = max(severity, secret_severity)

    injection = _scan_injection(body)
    findings.extend(injection)
    if injection:
        severity = max(severity, Severity.DANGEROUS)

    script_findings, script_severity = _scan_scripts(candidate)
    findings.extend(script_findings)
    severity = max(severity, script_severity)

    if findings and severity is Severity.INFO:
        severity = Severity.WARN
    return ValidationReport(tuple(findings), severity)


def _check_structure(body: str, candidate: CandidateSkill) -> list[Finding]:
    """Check the body's shape and size."""
    findings: list[Finding] = []
    headings = {line[3:].strip().lower() for line in body.splitlines() if line.startswith("## ")}
    findings.extend(
        Finding("SK030", "error", f"body is missing a '## {required}' section")
        for required in REQUIRED_SECTIONS
        if required.lower() not in headings
    )
    if "do not use when" not in headings:
        # The negative signal is the under-used half of routing, and its
        # absence is the usual reason a skill starts over-triggering.
        findings.append(
            Finding(
                "SK031",
                "warn",
                "no '## Do not use when' section; without it the skill will "
                "tend to trigger on adjacent tasks",
            )
        )
    if len(body) > MAX_BODY_CHARS:
        findings.append(Finding("SK032", "error", f"body exceeds {MAX_BODY_CHARS} characters"))
    if len(candidate.files) > MAX_SCRIPTS:
        findings.append(Finding("SK033", "error", f"more than {MAX_SCRIPTS} bundled files"))
    if sum(len(c) for _, c in candidate.files) > MAX_DIR_BYTES:
        findings.append(Finding("SK034", "error", "bundled files exceed 256 KiB"))
    if not candidate.tests_yaml.strip():
        findings.append(
            Finding(
                "SK035",
                "error",
                "no tests/cases.yaml; a skill without routing cases cannot be "
                "checked against the rest of the library",
            )
        )
    for path, _ in candidate.files:
        if ".." in path or path.startswith("/"):
            findings.append(Finding("SK036", "error", f"bundled path escapes the skill: {path}"))
    return findings


def _scan_secrets(candidate: CandidateSkill) -> tuple[list[Finding], Severity]:
    """Look for embedded credentials. Always blocking."""
    findings: list[Finding] = []
    blobs = [("SKILL.md", candidate.content), *candidate.files]
    for where, text in blobs:
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(
                    Finding(
                        "GRD100",
                        "error",
                        f"{where} contains something shaped like a credential; "
                        f"declare it in required_env instead and rotate the value",
                    )
                )
                break
    return findings, Severity.DANGEROUS if findings else Severity.INFO


def _scan_injection(body: str) -> list[Finding]:
    """Look for text aimed at the agent rather than the task.

    A skill is loaded into the conversation as reference material. Text telling
    the agent to skip approvals or ignore its guidance is not a skill doing its
    job, whoever wrote it -- and a self-written skill is a *persistent* place
    for that to live, unlike a single poisoned message.
    """
    findings = [
        Finding("GRD200", "error", f"body contains an instruction aimed at the agent: {p.pattern}")
        for p in _INJECTION_PATTERNS
        if p.search(body)
    ]
    if re.search(r"<!--.*?(ignore|instruct|system|approve).*?-->", body, re.S | re.I):
        findings.append(
            Finding("GRD201", "error", "an HTML comment contains instruction-like text")
        )
    return findings


def _scan_scripts(candidate: CandidateSkill) -> tuple[list[Finding], Severity]:
    """Statically inspect bundled executables."""
    findings: list[Finding] = []
    severity = Severity.INFO

    for path, content in candidate.files:
        if path.endswith(".py"):
            found, level = _scan_python(path, content)
        elif path.endswith((".sh", ".bash", ".zsh")):
            found, level = _scan_shell(path, content)
        else:
            continue
        findings.extend(found)
        severity = max(severity, level)
    return findings, severity


def _scan_python(path: str, content: str) -> tuple[list[Finding], Severity]:
    """Inspect a bundled Python script's AST."""
    findings: list[Finding] = []
    severity = Severity.INFO
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return [Finding("GRD300", "warn", f"{path} does not parse: {exc}")], Severity.WARN

    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".")[0] in _EXFIL_MODULES:
                    findings.append(
                        Finding("GRD301", "warn", f"{path} imports {name!r} (network egress)")
                    )
                    severity = max(severity, Severity.CAUTION)
        if isinstance(node, ast.Call):
            target = _dotted(node.func)
            if target in ("exec", "eval"):
                findings.append(Finding("GRD302", "warn", f"{path} calls {target}()"))
                severity = max(severity, Severity.CAUTION)

    lowered = content.lower()
    for marker in _CREDENTIAL_PATHS:
        if marker in lowered:
            findings.append(
                Finding("GRD303", "error", f"{path} references a credential path ({marker})")
            )
            severity = Severity.DANGEROUS
    return findings, severity


def _scan_shell(path: str, content: str) -> tuple[list[Finding], Severity]:
    """Inspect a bundled shell script."""
    findings: list[Finding] = []
    severity = Severity.INFO
    for pattern in _DANGEROUS_SHELL:
        if pattern.search(content):
            findings.append(
                Finding(
                    "GRD400", "error", f"{path} matches a destructive pattern: {pattern.pattern}"
                )
            )
            severity = Severity.DANGEROUS
    try:
        shlex.split(content)
    except ValueError:
        findings.append(Finding("GRD401", "warn", f"{path} has unbalanced quoting"))
        severity = max(severity, Severity.WARN)
    for marker in _CREDENTIAL_PATHS:
        if marker in content:
            findings.append(
                Finding("GRD402", "error", f"{path} references a credential path ({marker})")
            )
            severity = Severity.DANGEROUS
    return findings, severity


def _dotted(node: ast.AST) -> str:
    """Render a dotted call target."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


# -- duplicate detection -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DuplicateVerdict:
    """Whether a proposal is really new."""

    action: str
    """``create``, ``patch``, or ``reject``."""
    target: str | None
    score: float
    reason: str


PATCH_THRESHOLD = 0.55
REJECT_THRESHOLD = 0.85


def check_duplicate(candidate: CandidateSkill, existing: Sequence[SkillMeta]) -> DuplicateVerdict:
    """Decide whether a proposal should become a patch of something that exists.

    Preferring a patch over a new skill is the whole ballgame. Two skills whose
    descriptions both plausibly match a request means neither routes reliably,
    and every additional near-duplicate makes the rest slightly worse.
    """
    for meta in existing:
        if meta.name == candidate.name:
            return DuplicateVerdict(
                "patch", meta.name, 1.0, f"a skill named {meta.name!r} already exists"
            )

    description = _description_of(candidate)
    best: tuple[float, SkillMeta] | None = None
    for meta in existing:
        score = _similarity(description, meta.description)
        if best is None or score > best[0]:
            best = (score, meta)

    if best is None:
        return DuplicateVerdict("create", None, 0.0, "the library is empty")

    score, meta = best
    if score >= REJECT_THRESHOLD:
        return DuplicateVerdict("reject", meta.name, score, f"almost identical to {meta.name!r}")
    if score >= PATCH_THRESHOLD:
        return DuplicateVerdict(
            "patch",
            meta.name,
            score,
            f"overlaps {meta.name!r}; revise that rather than adding a near-duplicate",
        )
    return DuplicateVerdict("create", None, score, "no close match")


def _description_of(candidate: CandidateSkill) -> str:
    """Pull the description out of a candidate's frontmatter."""
    try:
        raw = parse_frontmatter(candidate.content)
    except Exception:  # validation reports the parse failure separately
        return candidate.name
    return str(raw.get("description") or candidate.name)


def _similarity(left: str, right: str) -> float:
    """Jaccard overlap of significant words.

    Crude on purpose. Embeddings would be better and are an optional upgrade,
    but a dependency-free check that catches the obvious cases is worth more
    than a perfect one that is off by default.
    """
    a = _significant(left)
    b = _significant(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_STOPWORDS = frozenset(
    {
        "use",
        "when",
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "to",
        "of",
        "in",
        "on",
        "with",
        "this",
        "that",
        "is",
        "are",
        "be",
        "not",
        "do",
        "does",
        "user",
        "users",
        "it",
        "its",
        "from",
        "by",
        "at",
        "as",
        "if",
        "than",
        "then",
    }
)


def _significant(text: str) -> set[str]:
    """Content words from a description."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}  # noqa: PLR2004


def lint_description(description: str) -> list[Finding]:
    """Check the one field that decides whether a skill is ever reached."""
    findings: list[Finding] = []
    text = description.strip()
    if len(text) > LINT_DESCRIPTION_CHARS:
        findings.append(
            Finding(
                "SK013",
                "warn",
                f"description is {len(text)} characters; over "
                f"{LINT_DESCRIPTION_CHARS} it will be truncated in the catalog",
            )
        )
    if "use when" not in text.lower():
        findings.append(Finding("SK012", "warn", "description does not say when to use the skill"))
    return findings
