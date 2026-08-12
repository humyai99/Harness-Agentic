"""Web search, behind one interface.

Search is the one capability here that cannot be done without a third party, so
it is a provider seam rather than an implementation: an operator who has a Brave
key uses Brave, one who has Tavily uses Tavily, one who has neither does not get
the tool at all. That last case is the important one -- the tool is *absent*
from the catalog rather than present and failing, because a tool that always
returns "not configured" teaches the model to keep trying it.

Results are attacker-influenced. Anyone can rank for a query, and a title or
snippet is a fine place to put an instruction and hope somebody's agent reads it
as one. So search output goes through the same untrusted-content envelope as a
fetched page.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from harness_agentic.errors import ConfigError
from harness_agentic.net.policy import UrlPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from harness_agentic.net.fetch import Fetcher
    from harness_agentic.providers.credentials import Secret, SecretResolver

log = logging.getLogger(__name__)

MAX_SNIPPET_CHARS = 300


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One result."""

    title: str
    url: str
    snippet: str = ""

    def clipped(self, limit: int = MAX_SNIPPET_CHARS) -> SearchHit:
        """The same hit with a snippet short enough to be worth the tokens."""
        if len(self.snippet) <= limit:
            return self
        return SearchHit(self.title, self.url, self.snippet[:limit].rstrip() + "…")


class SearchProvider(Protocol):
    """What :func:`~harness_agentic.tools.builtin.web.install_web_tools` needs."""

    name: str

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Run one search."""
        ...


@dataclass
class BraveSearch:
    """Brave's Search API. A JSON endpoint and a header, nothing more."""

    api_key: Secret
    fetcher: Fetcher
    name: str = "brave"
    endpoint: str = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Search, returning at most ``limit`` hits."""
        from urllib.parse import urlencode

        url = f"{self.endpoint}?{urlencode({'q': query, 'count': limit})}"
        result = self.fetcher.get(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key.reveal(),
            },
        )
        if not result.ok:
            detail = f"brave search returned HTTP {result.status}"
            raise ConfigError(detail)
        web = _section(_json(result.text()), "web")
        entries = _entries(web.get("results"))
        return [
            SearchHit(
                title=str(entry.get("title", "")),
                url=str(entry.get("url", "")),
                snippet=_strip_tags(str(entry.get("description", ""))),
            ).clipped()
            for entry in entries[:limit]
            if entry.get("url")
        ]


@dataclass
class TavilySearch:
    """Tavily's search API, which answers with prose snippets already."""

    api_key: Secret
    fetcher: Fetcher
    name: str = "tavily"
    endpoint: str = "https://api.tavily.com/search"

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Search, returning at most ``limit`` hits."""
        result = self.fetcher.request(
            "POST",
            self.endpoint,
            headers={"Content-Type": "application/json"},
            json_body={
                "api_key": self.api_key.reveal(),
                "query": query,
                "max_results": limit,
            },
        )
        if not result.ok:
            detail = f"tavily search returned HTTP {result.status}"
            raise ConfigError(detail)
        entries = _entries(_json(result.text()).get("results"))
        return [
            SearchHit(
                title=str(entry.get("title", "")),
                url=str(entry.get("url", "")),
                snippet=str(entry.get("content", "")),
            ).clipped()
            for entry in entries[:limit]
            if entry.get("url")
        ]


@dataclass
class StaticSearch:
    """A provider that answers from a table. Ships for tests and demos."""

    hits: dict[str, list[SearchHit]] = field(default_factory=dict)
    name: str = "static"
    queries: list[str] = field(default_factory=list)

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Return whatever was registered for this query."""
        self.queries.append(query)
        return self.hits.get(query, [])[:limit]


PROVIDERS: dict[str, tuple[str, ...]] = {
    "brave": ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"),
    "tavily": ("TAVILY_API_KEY",),
}
"""Provider name to the credential names it accepts, in preference order."""


def from_environment(
    resolver: SecretResolver,
    fetcher: Fetcher,
    *,
    prefer: Sequence[str] = ("brave", "tavily"),
) -> SearchProvider | None:
    """Build whichever search provider is configured, or ``None``.

    ``None`` rather than a stub: the caller omits the tool entirely, and the
    model never learns that asking for a search sometimes produces an apology.
    """
    for name in prefer:
        key = resolver.first(PROVIDERS.get(name, ()))
        if key is None:
            continue
        match name:
            case "brave":
                return BraveSearch(api_key=key, fetcher=fetcher)
            case "tavily":
                return TavilySearch(api_key=key, fetcher=fetcher)
    return None


def search_policy(base: UrlPolicy | None = None) -> UrlPolicy:
    """A policy that reaches the search APIs and nothing else.

    Worth having separately from the fetch policy: the search provider is a
    known host, so it can be allowlisted, and an allowlist is a much stronger
    control than a denylist of internal ranges.
    """
    template = base or UrlPolicy()
    return UrlPolicy(
        allowed_hosts=frozenset({"api.search.brave.com", "api.tavily.com"}),
        max_redirects=0,
        max_bytes=template.max_bytes,
        resolver=template.resolver,
    )


def _section(payload: dict[str, object], key: str) -> dict[str, object]:
    """One nested object from a decoded payload, or an empty one."""
    section = payload.get(key)
    return section if isinstance(section, dict) else {}


def _entries(raw: object) -> list[dict[str, object]]:
    """The result list from a provider payload, ignoring anything malformed."""
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def _json(text: str) -> dict[str, object]:
    """Parse a JSON object, treating anything else as empty."""
    try:
        loaded = json.loads(text)
    except ValueError:
        log.warning("search provider returned a body that was not JSON")
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _strip_tags(text: str) -> str:
    """Remove the ``<strong>`` highlighting search APIs put in snippets."""
    import re

    return re.sub(r"<[^>]+>", "", text)
