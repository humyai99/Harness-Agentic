"""Turning a web page into something worth spending context on.

Raw HTML is mostly not content. A typical article is 200 KB of markup around
6 KB of prose, and handing the markup to a model costs fifty thousand tokens to
convey what four thousand would -- while burying the actual text where the model
is less likely to attend to it.

So: drop the machinery, keep the prose and the structure that carries meaning
(headings, list items, link targets), and collapse the whitespace. Written
against ``html.parser`` from the standard library rather than a readability
library, because this runs on every fetch and a dependency that parses hostile
input is a dependency worth not having.

The output is deliberately markdown-ish, not markdown: it is for a model to
read, not for anything to render.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser

DROP = frozenset(
    {"script", "style", "noscript", "svg", "canvas", "template", "iframe", "object", "embed"}
)
"""Elements whose contents are never prose."""

CHROME = frozenset({"nav", "header", "footer", "aside", "form", "button", "select"})
"""Elements that are usually navigation, not content. Dropped by default, and
that is a trade: a site whose article lives in a ``<header>`` loses it. The
alternative -- keeping every nav link on every page -- costs more, every time."""

BLOCK = frozenset(
    {
        "p", "div", "section", "article", "main", "br", "hr", "tr", "table",
        "ul", "ol", "dl", "blockquote", "pre", "figure", "figcaption",
    }
)  # fmt: skip
HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}

_BLANKS = re.compile(r"\n{3,}")
_SPACES = re.compile("[ \t\u00a0\u200b]+")
"""Ordinary space, tab, no-break space, zero-width space. The last two are
everywhere in real HTML and reach the model as invisible token noise."""


@dataclass(frozen=True, slots=True)
class Extracted:
    """A page reduced to text."""

    title: str
    text: str
    links: tuple[tuple[str, str], ...]
    """``(text, href)`` pairs, in document order. What the agent navigates by."""

    def render(self, *, max_chars: int = 20_000, max_links: int = 40) -> str:
        """Format for a tool result, truncating honestly.

        Says how much was cut. A silently truncated page makes the model
        confidently answer from the first third of an article.
        """
        parts: list[str] = []
        if self.title:
            parts.append(f"# {self.title}")
        body = self.text
        if len(body) > max_chars:
            body = body[:max_chars].rstrip()
            parts.append(body)
            parts.append(f"\n[truncated: {len(self.text) - max_chars} more characters]")
        else:
            parts.append(body)
        if self.links:
            shown = self.links[:max_links]
            listed = "\n".join(f"- {text or href}: {href}" for text, href in shown)
            more = "" if len(self.links) <= max_links else f"\n[{len(self.links) - max_links} more]"
            parts.append(f"\n## Links\n{listed}{more}")
        return "\n\n".join(parts)


class _Reader(HTMLParser):
    """Accumulates prose while skipping markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._link_href = ""
        self._link_text: list[str] = []
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open an element."""
        if self._skip_depth:
            # Nested skipped elements have to be counted, or the first closing
            # tag would turn output back on halfway through a script.
            if tag in DROP or tag in CHROME:
                self._skip_depth += 1
            return
        if tag in DROP or tag in CHROME:
            self._skip_depth = 1
            return

        attributes = dict(attrs)
        match tag:
            case "title":
                self._in_title = True
            case "pre":
                self._in_pre = True
                self.chunks.append("\n```\n")
            case "a":
                self._link_href = (attributes.get("href") or "").strip()
                self._link_text = []
            case "li":
                self.chunks.append("\n- ")
            case "img":
                alt = (attributes.get("alt") or "").strip()
                if alt:
                    self.chunks.append(f"[image: {alt}]")
            case _ if tag in HEADINGS:
                self.chunks.append(f"\n\n{HEADINGS[tag]} ")
            case _ if tag in BLOCK:
                self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Close an element."""
        if self._skip_depth:
            if tag in DROP or tag in CHROME:
                self._skip_depth -= 1
            return
        match tag:
            case "title":
                self._in_title = False
            case "pre":
                self._in_pre = False
                self.chunks.append("\n```\n")
            case "a":
                text = "".join(self._link_text).strip()
                if self._link_href and not self._link_href.startswith(("javascript:", "#")):
                    self.links.append((text, self._link_href))
                self._link_href = ""
                self._link_text = []
            case _ if tag in HEADINGS or tag in BLOCK:
                self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        """Collect text."""
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data.strip()
            return
        if self._link_href:
            self._link_text.append(data)
        self.chunks.append(data if self._in_pre else _SPACES.sub(" ", data))


def extract(html: str, *, base_url: str = "") -> Extracted:
    """Reduce an HTML document to title, prose, and links."""
    reader = _Reader()
    try:
        reader.feed(html)
        reader.close()
    except (AssertionError, ValueError):
        # Malformed markup is the normal case, not an exception. Whatever was
        # parsed before the parser gave up is still worth having.
        log_note = "\n[note: the page was malformed and may be incomplete]"
        reader.chunks.append(log_note)

    text = _tidy("".join(reader.chunks))
    links = tuple((label, _absolute(href, base_url)) for label, href in _dedupe(reader.links))
    return Extracted(title=reader.title, text=text, links=links)


def to_text(html_or_text: str, *, content_type: str = "text/html", base_url: str = "") -> str:
    """Extract if it is HTML, tidy it if it is not."""
    if "html" in content_type.lower():
        return extract(html_or_text, base_url=base_url).render()
    return _tidy(html_or_text)


def _tidy(text: str) -> str:
    """Collapse the whitespace HTML leaves behind, outside code.

    Indentation inside a fence is content. Stripping it turns a Python sample
    into something that will not run, which is worse than useless when the
    reason for fetching the page was to read the sample.
    """
    tidied: list[str] = []
    in_code = False
    for line in unescape(text).splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            tidied.append(line.strip())
            continue
        tidied.append(line.rstrip() if in_code else _SPACES.sub(" ", line).strip())
    return _BLANKS.sub("\n\n", "\n".join(tidied)).strip()


def _dedupe(links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop repeated hrefs, keeping the first label seen for each."""
    seen: dict[str, str] = {}
    for label, href in links:
        if href not in seen:
            seen[href] = label
    return [(label, href) for href, label in seen.items()]


def _absolute(href: str, base_url: str) -> str:
    """Resolve a relative href, if there is a base to resolve against."""
    if not base_url:
        return href
    from urllib.parse import urljoin

    return urljoin(base_url, href)
