"""Driving a browser, behind an interface that Playwright is only one of.

A browser is arbitrary code execution with a network connection, so the same URL
policy that guards ``web_fetch`` guards navigation here. That is not belt and
braces: a page the agent is told to open can redirect to
``http://169.254.169.254/`` exactly as a fetch can, and a browser will follow it
and render the credentials.

Playwright is an optional extra, several hundred megabytes of it, so it is
imported inside the one method that needs it and the toolset hides itself when it
is absent. :class:`FakeDriver` implements the same protocol over a scripted page
tree, which is what lets the tools, the reference lifecycle and the staleness
handling be tested without downloading a browser.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from harness_agentic.browser.page import Builder, RefTable, Snapshot, StaleReference
from harness_agentic.errors import HarnessError
from harness_agentic.net.policy import UrlPolicy, UrlRefused

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 15_000
MAX_SCREENSHOT_BYTES = 2_000_000
DEFAULT_VIEWPORT = (1280, 800)


class BrowserError(HarnessError):
    """The browser could not do what was asked."""


class Driver(Protocol):
    """What the browser tools need."""

    def start(self) -> None:
        """Launch the browser."""
        ...

    def stop(self) -> None:
        """Close it, releasing the process."""
        ...

    def navigate(self, url: str) -> Snapshot:
        """Go to a URL and return the page outline."""
        ...

    def snapshot(self) -> Snapshot:
        """Re-read the current page."""
        ...

    def click(self, ref: str) -> Snapshot:
        """Click a referenced element and return the page after it settles."""
        ...

    def type_text(self, ref: str, text: str, *, submit: bool = False) -> Snapshot:
        """Type into a referenced element."""
        ...

    def screenshot(self, *, full_page: bool = False) -> tuple[str, bytes]:
        """Capture the viewport. Returns ``(media_type, data)``."""
        ...

    def current_url(self) -> str:
        """Where the browser is now."""
        ...


@dataclass
class PlaywrightDriver:
    """The real driver. Chromium under Playwright, one page at a time.

    One page deliberately. Tabs multiply the state the model has to track and
    every question they answer ("open this in a new tab") is answerable by
    navigating and coming back. The reference table makes coming back cheap.
    """

    policy: UrlPolicy = field(default_factory=UrlPolicy)
    headless: bool = True
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    viewport: tuple[int, int] = DEFAULT_VIEWPORT
    user_agent: str = ""
    _playwright: Any = None
    _browser: Any = None
    _page: Any = None
    _builder: Builder = field(default_factory=Builder)
    _refs: RefTable = field(default_factory=RefTable)

    def start(self) -> None:  # pragma: no cover - needs a real browser
        """Launch Chromium, or explain what to install."""
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            detail = (
                "browser automation needs the browser extra: "
                "pip install 'harness-agentic[browser]' && playwright install chromium"
            )
            raise BrowserError(detail) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        context = self._browser.new_context(
            viewport={"width": self.viewport[0], "height": self.viewport[1]},
            user_agent=self.user_agent or None,
        )
        self._page = context.new_page()
        self._page.set_default_timeout(self.timeout_ms)

    def stop(self) -> None:  # pragma: no cover - needs a real browser
        """Close the browser and the driver process."""
        for closer in (self._browser, self._playwright):
            if closer is not None:
                try:
                    closer.close() if hasattr(closer, "close") else closer.stop()
                except Exception:
                    log.debug("browser shutdown raised", exc_info=True)
        self._page = self._browser = self._playwright = None

    def navigate(self, url: str) -> Snapshot:  # pragma: no cover - needs a real browser
        """Check the URL against the policy, then go there.

        The same policy as ``web_fetch``, and for the same reason: a page can
        redirect somewhere internal, and a browser will follow it and render
        whatever it finds.
        """
        checked = self.policy.check(url)
        self._require_page().goto(checked.url, wait_until="domcontentloaded")
        return self.snapshot()

    def snapshot(self) -> Snapshot:  # pragma: no cover - needs a real browser
        """Read the accessibility tree and mint fresh references."""
        page = self._require_page()
        raw = page.accessibility.snapshot(interesting_only=True) or {}
        built = self._builder.build(raw, url=page.url, title=page.title())
        self._refs.replace(built.generation, self._locate(built))
        return built

    def _locate(self, snapshot: Snapshot) -> dict[str, Any]:  # pragma: no cover
        """Map each reference onto a Playwright locator.

        By role and accessible name, which is how the tree described it in the
        first place -- so the locator and the outline cannot disagree about what
        an element is.
        """
        page = self._require_page()
        handles: dict[str, Any] = {}
        for ref, node in snapshot.refs().items():
            locator = page.get_by_role(node.role, name=node.name, exact=True)
            handles[ref] = locator.first if node.name else locator
        return handles

    def click(self, ref: str) -> Snapshot:  # pragma: no cover - needs a real browser
        """Click and re-snapshot."""
        self._refs.resolve(ref).click(timeout=self.timeout_ms)
        self._require_page().wait_for_load_state("domcontentloaded")
        return self.snapshot()

    def type_text(
        self, ref: str, text: str, *, submit: bool = False
    ) -> Snapshot:  # pragma: no cover
        """Fill a field, optionally pressing Enter."""
        locator = self._refs.resolve(ref)
        locator.fill(text, timeout=self.timeout_ms)
        if submit:
            locator.press("Enter")
            self._require_page().wait_for_load_state("domcontentloaded")
        return self.snapshot()

    def screenshot(self, *, full_page: bool = False) -> tuple[str, bytes]:  # pragma: no cover
        """Capture a PNG, bounded."""
        data = self._require_page().screenshot(full_page=full_page, type="png")
        if len(data) > MAX_SCREENSHOT_BYTES:
            detail = f"the screenshot is {len(data)} bytes, over the {MAX_SCREENSHOT_BYTES} cap"
            raise BrowserError(detail)
        return "image/png", data

    def current_url(self) -> str:  # pragma: no cover - needs a real browser
        """Where the browser is now."""
        return str(self._require_page().url)

    def _require_page(self) -> Any:  # pragma: no cover - needs a real browser
        """The open page, or a failure that says to start the browser."""
        if self._page is None:
            detail = "the browser is not running"
            raise BrowserError(detail)
        return self._page


@dataclass
class FakeDriver:
    """A scripted browser. Ships in the package, for the same reason the others do.

    Playwright is several hundred megabytes and needs a downloaded Chromium, so a
    fake that speaks the same protocol is the difference between the tool layer
    being tested on every commit and being tested when somebody remembers.
    """

    pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: UrlPolicy = field(default_factory=UrlPolicy)
    url: str = "about:blank"
    started: bool = False
    visited: list[str] = field(default_factory=list)
    clicked: list[str] = field(default_factory=list)
    typed: list[tuple[str, str, bool]] = field(default_factory=list)
    screenshots: int = 0
    _builder: Builder = field(default_factory=Builder)
    _refs: RefTable = field(default_factory=RefTable)
    _snapshot: Snapshot | None = None

    def add(self, url: str, tree: Mapping[str, Any], *, title: str = "") -> None:
        """Register a page."""
        self.pages[url] = {"tree": dict(tree), "title": title or url}

    def start(self) -> None:
        """Mark the browser open."""
        self.started = True

    def stop(self) -> None:
        """Mark it closed."""
        self.started = False

    def navigate(self, url: str) -> Snapshot:
        """Check the policy, then load a registered page."""
        if not self.started:
            detail = "the browser is not running"
            raise BrowserError(detail)
        # Enforced in the fake too: a fake that skips the policy hides exactly
        # the bugs the policy exists to catch.
        self.policy.check(url)
        page = self.pages.get(url)
        if page is None:
            detail = f"no such page: {url}"
            raise BrowserError(detail)
        self.url = url
        self.visited.append(url)
        return self._build(page)

    def snapshot(self) -> Snapshot:
        """Re-read the current page, minting fresh references."""
        page = self.pages.get(self.url)
        if page is None:
            detail = "nothing is loaded"
            raise BrowserError(detail)
        return self._build(page)

    def _build(self, page: dict[str, Any]) -> Snapshot:
        built = self._builder.build(page["tree"], url=self.url, title=page["title"])
        self._refs.replace(built.generation, {ref: ref for ref in built.refs()})
        self._snapshot = built
        return built

    def click(self, ref: str) -> Snapshot:
        """Record a click, following a link when the page declares a target."""
        self._refs.resolve(ref)
        self.clicked.append(ref)
        target = (self.pages.get(self.url) or {}).get("links", {}).get(ref)
        if target:
            return self.navigate(target)
        return self.snapshot()

    def type_text(self, ref: str, text: str, *, submit: bool = False) -> Snapshot:
        """Record typing."""
        self._refs.resolve(ref)
        self.typed.append((ref, text, submit))
        return self.snapshot()

    def screenshot(self, *, full_page: bool = False) -> tuple[str, bytes]:
        """Return a one-pixel PNG."""
        del full_page
        self.screenshots += 1
        return "image/png", b"\x89PNG\r\n\x1a\n"

    def current_url(self) -> str:
        """Where the fake browser is."""
        return self.url


def stale_hint(exc: StaleReference) -> str:
    """The message a stale reference produces for the model."""
    return f"{exc} (the page changed; nothing was clicked)"


def refused_hint(exc: UrlRefused) -> str:
    """The message a refused navigation produces for the model."""
    return f"refused to open that URL: {exc}"
