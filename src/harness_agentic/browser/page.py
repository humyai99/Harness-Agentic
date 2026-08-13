"""What an agent needs to know about a web page, and nothing more.

The central decision is **the accessibility tree, not a screenshot**. A model
given a picture has to locate a button by pixel coordinates, which it does badly
and cannot verify; a model given ``button "Submit" [ref=e12]`` clicks by
reference and either it exists or it does not. The tree is also an order of
magnitude cheaper: a page that costs 1,500 tokens as a labelled outline costs
tens of thousands as an image, and the image is the less actionable of the two.

So screenshots are for the cases where appearance genuinely is the question --
"is the chart rendering" -- and everything else runs on the outline.

The other decision is that **element references are opaque and short-lived**. A
CSS selector the model invents is a guess about markup it has not read; a
reference minted when the snapshot was taken either resolves or reports that the
page has changed underneath it. That second outcome is the useful one: a click on
a stale reference is a click on whatever moved into that position, and doing it
silently is how an agent cancels an order it meant to confirm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MAX_SNAPSHOT_CHARS = 24_000
MAX_TEXT_CHARS = 400
INTERACTIVE = frozenset(
    {
        "button", "link", "textbox", "combobox", "checkbox", "radio", "slider",
        "menuitem", "tab", "option", "searchbox", "switch", "spinbutton",
    }
)  # fmt: skip
"""Roles worth giving a reference. A reference on a `paragraph` is noise: there
is nothing to do with it, and it doubles the size of the outline."""

SKIP_ROLES = frozenset({"none", "presentation", "generic", "InlineTextBox", "StaticText"})
"""Wrapper roles that carry no meaning. Their children are kept; they are not."""

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Node:
    """One element in the accessibility outline."""

    role: str
    name: str = ""
    ref: str = ""
    value: str = ""
    checked: bool | None = None
    disabled: bool = False
    children: tuple[Node, ...] = ()

    @property
    def interactive(self) -> bool:
        """Whether the agent can do anything with this node."""
        return self.role in INTERACTIVE and not self.disabled

    def render(self, *, depth: int = 0) -> list[str]:
        """This node and its descendants, as indented lines."""
        indent = "  " * depth
        parts = [self.role]
        if self.name:
            parts.append(f'"{_clip(self.name)}"')
        if self.value:
            parts.append(f"value={_clip(self.value, 80)!r}")
        if self.checked is not None:
            parts.append("checked" if self.checked else "unchecked")
        if self.disabled:
            parts.append("disabled")
        if self.ref:
            parts.append(f"[ref={self.ref}]")
        lines = [f"{indent}{' '.join(parts)}"]
        for child in self.children:
            lines.extend(child.render(depth=depth + 1))
        return lines

    def walk(self) -> list[Node]:
        """This node and every descendant, depth first."""
        found = [self]
        for child in self.children:
            found.extend(child.walk())
        return found


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A page at one moment, with references valid only for this moment."""

    url: str
    title: str
    root: Node
    generation: int
    """Increments on every navigation and every snapshot. A reference from an
    older generation is refused rather than resolved against whatever moved
    into that position."""

    def render(self, *, max_chars: int = MAX_SNAPSHOT_CHARS) -> str:
        """The outline as text, truncated honestly."""
        head = f"# {self.title}\n{self.url}\n"
        body = "\n".join(self.root.render())
        if len(body) > max_chars:
            kept = body[:max_chars].rsplit("\n", 1)[0]
            dropped = body[len(kept) :].count("\n")
            return (
                f"{head}\n{kept}\n"
                f"[the outline was truncated: {dropped} more element(s). "
                f"Narrow the page or use browser_find.]"
            )
        return f"{head}\n{body}"

    def refs(self) -> dict[str, Node]:
        """Every referenced node, by reference."""
        return {node.ref: node for node in self.root.walk() if node.ref}

    def find(self, query: str, *, limit: int = 20) -> list[Node]:
        """Interactive nodes whose role or name matches a query.

        Exists so a large page does not have to be re-read in full to click one
        thing. The alternative -- the model asking for the whole outline again --
        costs the whole outline again.
        """
        needle = query.strip().lower()
        matches = [
            node
            for node in self.root.walk()
            if node.interactive and (needle in node.name.lower() or needle in node.role)
        ]
        return matches[:limit]


class StaleReference(LookupError):
    """A reference from an earlier snapshot was used."""


@dataclass
class Builder:
    """Turns a raw accessibility tree into an outline with references.

    Written against the shape Playwright and CDP both produce -- a nested dict
    with ``role``, ``name`` and ``children`` -- so a driver hands over whatever
    it got and this decides what is worth keeping.
    """

    generation: int = 0
    prefix: str = "e"
    _counter: int = 0

    def build(self, raw: dict[str, Any], *, url: str, title: str) -> Snapshot:
        """Build a snapshot, minting fresh references."""
        self.generation += 1
        self._counter = 0
        root = self._node(raw) or Node(role="document")
        return Snapshot(url=url, title=title, root=root, generation=self.generation)

    def _node(self, raw: dict[str, Any]) -> Node | None:
        """Convert one raw node, dropping wrappers but keeping their children."""
        role = str(raw.get("role") or "generic")
        children = [
            built
            for child in raw.get("children") or []
            if isinstance(child, dict) and (built := self._node(child)) is not None
        ]
        name = _normalize(str(raw.get("name") or ""))

        if role in SKIP_ROLES and not name:
            # A wrapper with one child collapses into it; with several, they are
            # spliced into the parent. Keeping wrappers doubles the outline and
            # tells the model nothing.
            if len(children) == 1:
                return children[0]
            if not children:
                return None
            return Node(role="group", children=tuple(children))

        disabled = bool(raw.get("disabled", False))
        node = Node(
            role=role,
            name=name,
            value=_normalize(str(raw.get("value") or "")),
            checked=raw.get("checked") if isinstance(raw.get("checked"), bool) else None,
            disabled=disabled,
            children=tuple(children),
        )
        if node.interactive:
            self._counter += 1
            node = Node(
                role=node.role,
                name=node.name,
                ref=f"{self.prefix}{self._counter}",
                value=node.value,
                checked=node.checked,
                disabled=node.disabled,
                children=node.children,
            )
        return node


@dataclass
class RefTable:
    """Maps references to whatever the driver needs to act on them.

    Cleared on every snapshot, which is the point: a reference is a promise about
    a page that existed a moment ago, and a click on a stale one lands on
    whatever moved into that position.
    """

    generation: int = 0
    handles: dict[str, Any] = field(default_factory=dict)

    def replace(self, generation: int, handles: dict[str, Any]) -> None:
        """Adopt a new snapshot's references, discarding the old ones."""
        self.generation = generation
        self.handles = dict(handles)

    def resolve(self, ref: str) -> Any:
        """The handle for a reference, or a refusal that says what to do."""
        handle = self.handles.get(ref)
        if handle is None:
            known = ", ".join(sorted(self.handles)[:10]) or "none"
            detail = (
                f"{ref!r} is not on the current page (generation {self.generation}). "
                f"Take a fresh snapshot; known references: {known}"
            )
            raise StaleReference(detail)
        return handle


def _normalize(text: str) -> str:
    """Collapse whitespace in an accessible name."""
    return _WHITESPACE.sub(" ", text).strip()


def _clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """Bound one label so a page of long labels stays readable."""
    return text if len(text) <= limit else text[: limit - 1] + "…"
