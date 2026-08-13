"""Fetching a URL, once the policy has agreed to it.

Every redirect is re-checked. That is the whole reason redirects are followed
manually here rather than by handing ``follow_redirects=True`` to httpx: a
public URL that 302s to ``http://169.254.169.254/`` defeats a check that only
ran on the first hop, and that is the standard way SSRF filters are bypassed.

The response is capped while streaming, not after. A cap applied to an
already-buffered body is not a cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urljoin

import httpx

from harness_agentic.net.policy import UrlPolicy, UrlRefused

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)

USER_AGENT = "harness-agentic/1.0 (+agent)"
DEFAULT_TIMEOUT_S = 20.0
TEXTUAL_TYPES = ("text/", "application/json", "application/xml", "+json", "+xml", "javascript")


@dataclass(frozen=True, slots=True)
class FetchResult:
    """What came back."""

    url: str
    """The final URL, after redirects. Not the one that was asked for."""
    status: int
    content_type: str
    body: bytes
    truncated: bool = False
    redirects: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the server answered successfully."""
        return httpx.codes.OK <= self.status < httpx.codes.MULTIPLE_CHOICES

    @property
    def is_text(self) -> bool:
        """Whether the body is worth decoding as text."""
        kind = self.content_type.split(";")[0].strip().lower()
        return any(marker in kind for marker in TEXTUAL_TYPES)

    def text(self) -> str:
        """The body as text, replacing anything undecodable."""
        return self.body.decode(_charset(self.content_type), errors="replace")


class Fetcher(Protocol):
    """What the web tools need. Implemented for real, and faked in tests."""

    def get(
        self, url: str, *, headers: Mapping[str, str] | None = None, timeout_s: float = ...
    ) -> FetchResult:
        """Fetch a URL, following redirects under the policy."""
        ...

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: object = None,
        timeout_s: float = ...,
    ) -> FetchResult:
        """Make an arbitrary request, under the same policy."""
        ...


@dataclass
class HttpFetcher:
    """The real fetcher: httpx, plus the policy on every hop.

    Synchronous, like the rest of the core. A tool that blocks for twenty
    seconds blocks one worker thread, which is the deal the whole architecture
    makes -- see :mod:`harness_agentic.core.async_bridge`.
    """

    policy: UrlPolicy = field(default_factory=UrlPolicy)
    client: httpx.Client | None = None
    user_agent: str = USER_AGENT

    def __post_init__(self) -> None:
        """Open a client if one was not supplied."""
        if self.client is None:
            self.client = httpx.Client(
                follow_redirects=False,
                headers={"User-Agent": self.user_agent},
                timeout=DEFAULT_TIMEOUT_S,
            )

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> FetchResult:
        """Fetch a URL."""
        return self.request("GET", url, headers=headers, timeout_s=timeout_s)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: object = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> FetchResult:
        """Make a request, re-checking the policy at every redirect."""
        assert self.client is not None  # noqa: S101 - established in __post_init__
        seen: list[str] = []
        current = url
        sent = dict(headers or {})

        for _ in range(self.policy.max_redirects + 1):
            checked = self.policy.check(current)
            response = self.client.request(
                method,
                checked.url,
                headers=sent,
                json=json_body,
                timeout=timeout_s,
                follow_redirects=False,
            )
            location = response.headers.get("location")
            if _is_redirect(response.status_code) and location:
                response.close()
                target = urljoin(checked.url, location)
                seen.append(target)
                # Authorization is not carried across a redirect. A page that
                # bounces to another host must not be handed the credential
                # that was meant for the first one.
                if _host_of(target) != checked.host:
                    sent = {k: v for k, v in sent.items() if k.lower() != "authorization"}
                current = target
                if method.upper() == "POST" and response.status_code in (301, 302, 303):
                    method, json_body = "GET", None
                continue

            body, truncated = self._read_capped(response)
            return FetchResult(
                url=str(response.url),
                status=response.status_code,
                content_type=response.headers.get("content-type", ""),
                body=body,
                truncated=truncated,
                redirects=tuple(seen),
            )

        detail = f"more than {self.policy.max_redirects} redirects starting at {url}"
        raise UrlRefused(detail)

    def _read_capped(self, response: httpx.Response) -> tuple[bytes, bool]:
        """Read the body, stopping at the cap rather than trimming after."""
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            remaining = self.policy.max_bytes - total
            if remaining <= 0:
                response.close()
                return b"".join(chunks), True
            chunks.append(chunk[:remaining])
            total += min(len(chunk), remaining)
        response.close()
        return b"".join(chunks), total >= self.policy.max_bytes

    def close(self) -> None:
        """Close the underlying client."""
        if self.client is not None:
            self.client.close()


@dataclass
class RecordedFetcher:
    """A fetcher that answers from a table. Ships for tests and plugin authors.

    Keyed on exact URL, because a fake that fuzzy-matches will happily answer a
    request the real thing would have refused, and then the test proves
    nothing.
    """

    responses: dict[str, FetchResult] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    sent: list[tuple[str, str, dict[str, str], object, float]] = field(default_factory=list)
    """Every request as ``(method, url, headers, body, timeout)``."""
    policy: UrlPolicy = field(default_factory=UrlPolicy)
    enforce_policy: bool = True
    """On by default: a fake that skips the policy hides exactly the bugs the
    policy exists to catch."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> FetchResult:
        """Answer a GET from the table."""
        return self.request("GET", url, headers=headers, timeout_s=timeout_s)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: object = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> FetchResult:
        """Answer any request from the table.

        Headers, body and timeout are accepted to match :class:`Fetcher` and
        recorded rather than acted on -- a test that wants to assert on them
        reads :attr:`sent`.
        """
        self.sent.append((method.upper(), url, dict(headers or {}), json_body, timeout_s))
        if self.enforce_policy:
            self.policy.check(url)
        self.calls.append((method.upper(), url))
        found = self.responses.get(url)
        if found is None:
            return FetchResult(url=url, status=404, content_type="text/plain", body=b"not found")
        return found

    def add(
        self,
        url: str,
        body: str,
        *,
        status: int = 200,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        """Register a canned response."""
        self.responses[url] = FetchResult(
            url=url, status=status, content_type=content_type, body=body.encode()
        )


def _is_redirect(status: int) -> bool:
    """Whether a status code means "look elsewhere"."""
    return status in (301, 302, 303, 307, 308)


def _host_of(url: str) -> str:
    """The lowercase host of a URL, or the empty string."""
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "").lower()


def _charset(content_type: str) -> str:
    """The charset a Content-Type declares, defaulting to UTF-8."""
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            return value.strip("\"'") or "utf-8"
    return "utf-8"
