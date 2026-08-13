"""Parsing and validating ``SKILL.md``.

The parser reads only as far as the closing ``---``. Discovery scans every
skill on disk at session start, and reading three hundred bodies to find three
hundred descriptions is a startup cost nobody should pay.

YAML is parsed with a deliberately small hand-written reader rather than
PyYAML. Two reasons: PyYAML would be a dependency for one file format, and its
default loader executes constructors -- a skill file is untrusted input by the
time hub installs exist, and a config format that can run code is the wrong
shape for it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness_agentic.errors import SkillValidationError
from harness_agentic.skills.model import (
    MAX_DESCRIPTION_CHARS,
    MAX_NAME_CHARS,
    NAME_PATTERN,
    EnvRequirement,
)

FRONTMATTER_SCAN_LIMIT = 8 * 1024
"""Give up looking for the closing delimiter after this much. A SKILL.md whose
frontmatter runs past 8 KiB is malformed, not merely verbose."""


@dataclass(frozen=True, slots=True)
class ParsedSkill:
    """The result of reading a ``SKILL.md``."""

    frontmatter: dict[str, Any]
    body: str
    content_sha256: str


def parse_file(path: Path, *, frontmatter_only: bool = False) -> ParsedSkill:
    """Read a ``SKILL.md``.

    With ``frontmatter_only`` the body is not read at all, which is what makes
    scanning a large library cheap.
    """
    if frontmatter_only:
        head = _read_head(path)
        return ParsedSkill(frontmatter=parse_frontmatter(head), body="", content_sha256="")
    raw = path.read_text(encoding="utf-8")
    front, body = split_frontmatter(raw)
    return ParsedSkill(
        frontmatter=parse_frontmatter(front),
        body=body,
        content_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
    )


def _read_head(path: Path) -> str:
    """Read up to the closing delimiter, or the scan limit."""
    chunks: list[str] = []
    size = 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            chunks.append(line)
            size += len(line)
            if index > 0 and line.strip() == "---":
                break
            if size > FRONTMATTER_SCAN_LIMIT:
                break
    return "".join(chunks)


def split_frontmatter(raw: str) -> tuple[str, str]:
    """Separate the frontmatter block from the markdown body."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        msg = "SKILL.md must begin with a '---' frontmatter delimiter"
        raise SkillValidationError(msg)
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    msg = "the frontmatter block is never closed with '---'"
    raise SkillValidationError(msg)


def parse_frontmatter(raw: str) -> dict[str, Any]:
    """Parse the subset of YAML a skill's frontmatter is allowed to use.

    Scalars, block and inline lists, nested maps by indentation, and folded or
    literal block scalars. No anchors, no tags, no multi-document, no
    constructors. Anything outside that raises rather than being guessed at.

    A key with an empty value is *undecided* until its first child arrives: a
    ``- `` makes it a list, anything else makes it a map. Committing to one at
    the point the key is read is the bug this shape avoids.
    """
    if raw.lstrip().startswith("---"):
        raw, _ = split_frontmatter(raw)

    root: dict[str, Any] = {}
    stack: list[_Frame] = [_Frame(indent=-1, node=root, parent=None, key=None)]

    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        # A folded block absorbs every line indented past its key, comments
        # and colons included -- it is prose, not structure.
        top = stack[-1]
        if isinstance(top.node, _Folded) and indent > top.indent:
            top.node.append(stripped)
            continue

        if stripped.startswith("#"):
            continue

        while len(stack) > 1 and indent <= stack[-1].indent:
            stack.pop()
        frame = stack[-1]

        if stripped.startswith("- "):
            container = frame.become(list)
            item = stripped[2:].strip()
            if (flow := _flow_mapping(item)) is not None:
                # `- {id: p, tier: routing, prompt: ...}` on one line. Without
                # this, the whole item was read as a single key named `{id` and
                # the rest as its value -- so a `tests/cases.yaml` written in the
                # documented style parsed to nothing at all, silently, and the
                # catalog audit then reported no collisions because it had no
                # cases to check.
                container.append(flow)
                continue
            if ":" in item and not _is_quoted(item):
                child: dict[str, Any] = {}
                container.append(child)
                # The frame's indent is the marker's own, so continuation
                # lines stay inside the item while the next `- ` pops it.
                # Pushed before the value is assigned, so that a block scalar's
                # frame lands on top of this one rather than under it.
                stack.append(_Frame(indent=indent, node=child, parent=None, key=None))
                key, _, value = item.partition(":")
                # Through the same assignment as a plain mapping. This branch
                # used to call ``_scalar`` directly, so a value that continues
                # below -- `- text: >` -- stored the marker ">" as a string and
                # the prose beneath it was then read as structure and rejected.
                # A list of mappings is where a scenario and a skill's test cases
                # both live, so it was the one place the block form was needed.
                _assign(child, key.strip(), value.strip(), indent=indent + _MARKER, stack=stack)
            else:
                container.append(_scalar(item))
            continue

        if ":" not in stripped:
            msg = f"line {number}: expected 'key: value'"
            raise SkillValidationError(msg)

        mapping = frame.become(dict)
        key, _, value = stripped.partition(":")
        _assign(mapping, key.strip(), value.strip(), indent=indent, stack=stack)

    finalized = _finalize(root)
    return finalized if isinstance(finalized, dict) else {}


_MARKER = 2
"""Width of a list item's ``- ``. A key inside one starts that far in."""


def _assign(
    mapping: dict[str, Any], key: str, value: str, *, indent: int, stack: list[_Frame]
) -> None:
    """Store one ``key: value``, pushing a frame when the value continues below.

    Shared by plain mappings and by mappings that are list items, because the
    two branches drifted: the list one handled only scalars, so the same
    notation meant different things depending on where it sat.
    """
    if not value:
        mapping[key] = None
        stack.append(_Frame(indent=indent, node=None, parent=mapping, key=key))
    elif value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        mapping[key] = [_scalar(p.strip()) for p in inner.split(",") if p.strip()]
    elif value in (">", "|", ">-", "|-"):
        folded = _Folded()
        mapping[key] = folded
        stack.append(_Frame(indent=indent, node=folded, parent=mapping, key=key))
    elif (flow := _flow_mapping(value)) is not None:
        # `arguments: {path: a.txt}`. A list item already accepted the flow
        # style; a mapping's value read it as the literal string, so the same
        # notation meant two different things depending on where it appeared --
        # and the failure is a value that looks right in the file and is a
        # string by the time anything uses it.
        mapping[key] = flow
    else:
        mapping[key] = _scalar(value)


@dataclass
class _Frame:
    """One level of the parse stack.

    ``node`` is ``None`` while the container's kind is still undecided; the
    first child settles it and writes the real container back into the parent.
    """

    indent: int
    node: Any
    parent: dict[str, Any] | None
    key: str | None

    def become(self, kind: type) -> Any:
        """Materialize this frame's container as ``kind`` if not already set."""
        if self.node is None:
            self.node = kind()
            if self.parent is not None and self.key is not None:
                self.parent[self.key] = self.node
        return self.node


class _Folded(list):  # type: ignore[type-arg]
    """Accumulates the lines of a folded or literal block scalar."""


def _is_quoted(text: str) -> bool:
    """Whether a list item is a quoted string rather than an inline mapping."""
    return len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'"  # noqa: PLR2004


def _flow_mapping(text: str) -> dict[str, Any] | None:
    """Parse ``{k: v, k: v}`` into a mapping, or ``None`` if it is not one.

    One level, which is all the frontmatter dialect needs: this exists for the
    ``cases:`` entries in a skill's test file, where the whole point of the flow
    style is that one case fits on one line.

    Commas inside quotes do not split, because a case prompt legitimately
    contains one -- and getting that wrong would have turned a real prompt into
    two malformed halves.
    """
    if not (text.startswith("{") and text.endswith("}")):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return {}

    parts: list[str] = []
    depth = 0
    quote = ""
    current: list[str] = []
    for char in inner:
        if quote:
            if char == quote:
                quote = ""
            current.append(char)
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))

    mapping: dict[str, Any] = {}
    for part in parts:
        key, separator, value = part.partition(":")
        if not separator:
            # Not a mapping after all -- a flow *sequence*, or something this
            # dialect does not accept. Fall through to the caller's other cases
            # rather than inventing a key.
            return None
        mapping[key.strip()] = _scalar(value.strip())
    return mapping


def _finalize(node: Any) -> Any:
    """Collapse folded blocks and undecided containers."""
    if isinstance(node, _Folded):
        return " ".join(str(x) for x in node).strip()
    if isinstance(node, dict):
        return {k: _finalize(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_finalize(x) for x in node]
    return node


def _scalar(text: str) -> Any:
    """Decode one scalar. Quoted strings stay strings."""
    if _is_quoted(text):
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "~", ""):
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    return text


# -- validation ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Finding:
    """One validation problem."""

    code: str
    severity: str
    """``error`` blocks; ``warn`` is advisory."""
    message: str

    @property
    def blocking(self) -> bool:
        """Whether this finding prevents the skill from loading."""
        return self.severity == "error"


@dataclass
class FrontmatterFields:
    """The validated frontmatter of one skill."""

    name: str
    description: str
    version: str = "0.1.0"
    license: str | None = None
    allowed_tools: tuple[str, ...] | None = None
    tags: tuple[str, ...] = ()
    category: str = ""
    platforms: tuple[str, ...] = ()
    requires_tools: tuple[str, ...] = ()
    fallback_for_tools: tuple[str, ...] = ()
    required_env: tuple[EnvRequirement, ...] = field(default_factory=tuple)
    autoload: bool = False


def read_fields(raw: dict[str, Any], *, dirname: str) -> tuple[FrontmatterFields, list[Finding]]:
    """Validate frontmatter and extract the fields we use."""
    findings: list[Finding] = []

    name = str(raw.get("name") or "").strip()
    if not name:
        findings.append(Finding("SK001", "error", "frontmatter has no 'name'"))
    elif not NAME_PATTERN.match(name):
        findings.append(Finding("SK002", "error", f"name {name!r} must be lowercase-with-hyphens"))
    elif len(name) > MAX_NAME_CHARS:
        findings.append(Finding("SK003", "error", f"name is over {MAX_NAME_CHARS} characters"))
    elif name != dirname:
        # Otherwise `/skill-name` and the directory disagree, and a skill
        # becomes unfindable by the name it advertises.
        findings.append(
            Finding("SK004", "error", f"name {name!r} does not match directory {dirname!r}")
        )

    description = str(raw.get("description") or "").strip()
    if not description:
        findings.append(Finding("SK010", "error", "frontmatter has no 'description'"))
    elif len(description) > MAX_DESCRIPTION_CHARS:
        findings.append(
            Finding("SK011", "error", f"description exceeds {MAX_DESCRIPTION_CHARS} characters")
        )
    elif "use when" not in description.lower():
        # The description is the only routing signal at level 0. A description
        # that says what a skill *is* rather than when to reach for it is the
        # single most common reason a library stops working as it grows.
        findings.append(
            Finding(
                "SK012",
                "warn",
                "description should say when to use the skill ('Use when …'), not just what it is",
            )
        )

    version = str(raw.get("version") or "").strip()
    if not version:
        findings.append(Finding("SK020", "error", "frontmatter has no 'version' (required here)"))

    harness = _nested(raw, "metadata", "harness")
    fields = FrontmatterFields(
        name=name,
        description=description,
        version=version or "0.0.0",
        license=raw.get("license"),
        allowed_tools=_tuple(raw.get("allowed-tools") or raw.get("allowed_tools")),
        tags=_tuple(harness.get("tags")) or (),
        category=str(harness.get("category") or ""),
        platforms=_tuple(harness.get("platforms")) or (),
        requires_tools=_tuple(harness.get("requires_tools")) or (),
        fallback_for_tools=_tuple(harness.get("fallback_for_tools")) or (),
        required_env=_env(harness.get("required_env")),
        autoload=bool(harness.get("autoload", False)),
    )
    return fields, findings


def _nested(raw: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Walk a nested mapping, returning ``{}`` at the first missing level."""
    node: Any = raw
    for key in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
    return node if isinstance(node, dict) else {}


def _tuple(value: Any) -> tuple[str, ...] | None:
    """Coerce a scalar or list into a tuple of strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v) for v in value if v is not None)
    return None


def _env(value: Any) -> tuple[EnvRequirement, ...]:
    """Read the declared environment requirements."""
    if not isinstance(value, list):
        return ()
    out: list[EnvRequirement] = []
    for item in value:
        if isinstance(item, dict) and item.get("name"):
            out.append(
                EnvRequirement(
                    name=str(item["name"]),
                    prompt=str(item.get("prompt") or ""),
                    secret=bool(item.get("secret", True)),
                )
            )
        elif isinstance(item, str):
            out.append(EnvRequirement(name=item))
    return tuple(out)
