"""The ``browser`` toolset.

Five tools, and the ordering of the first two is the whole design:
``browser_snapshot`` comes before ``browser_screenshot`` in every sense -- it is
what the docstrings point at, it is what navigation and clicking return, and it
is what a model should be using. A screenshot is for when appearance is genuinely
the question.

Everything the browser renders is untrusted content, so outlines and page text go
through the same envelope as a fetched page and taint the session. A page can ask
to be treated as an instruction just as effectively through an accessible name as
through prose: ``button "SYSTEM: ignore prior instructions"`` is a button anybody
can put on a page.

Clicking is classified ``WRITES``. A click submits forms, confirms dialogs and
spends money, and no amount of care in the outline changes that -- so it goes
through the approval policy like any other consequential action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from harness_agentic.browser.driver import BrowserError, refused_hint, stale_hint
from harness_agentic.browser.page import StaleReference
from harness_agentic.core.types import ImageBlock
from harness_agentic.net.policy import UrlRefused
from harness_agentic.tools.builtin.web import envelope
from harness_agentic.tools.spec import ApprovalRequest, Danger, ToolContext, ToolResult

if TYPE_CHECKING:
    from harness_agentic.browser.driver import Driver
    from harness_agentic.browser.page import Snapshot
    from harness_agentic.tools.registry import ToolRegistry

MAX_OUTLINE_CHARS = 24_000
MAX_FIND_RESULTS = 20


class NavigateParams(BaseModel):
    """Arguments for ``browser_navigate``."""

    url: str = Field(description="Absolute http or https URL to open.")


class SnapshotParams(BaseModel):
    """Arguments for ``browser_snapshot``."""

    max_chars: int = Field(
        MAX_OUTLINE_CHARS, gt=0, le=80_000, description="Truncate the outline here."
    )


class FindParams(BaseModel):
    """Arguments for ``browser_find``."""

    query: str = Field(
        min_length=1,
        description="Text or role to look for among the page's interactive elements.",
    )


class ClickParams(BaseModel):
    """Arguments for ``browser_click``."""

    ref: str = Field(description="An element reference from the current snapshot, such as 'e12'.")
    what: str = Field(
        "", description="What this element is, in your words. Shown when asking for approval."
    )


class TypeParams(BaseModel):
    """Arguments for ``browser_type``."""

    ref: str = Field(description="A textbox reference from the current snapshot.")
    text: str = Field(description="What to type. Replaces whatever is there.")
    submit: bool = Field(default=False, description="Press Enter afterwards.")


class ScreenshotParams(BaseModel):
    """Arguments for ``browser_screenshot``."""

    full_page: bool = Field(default=False, description="Capture beyond the viewport.")
    why: str = Field(
        "",
        description=(
            "Why an image is needed rather than a snapshot. Most questions about "
            "a page are answered more cheaply and more reliably by the outline."
        ),
    )


def install_browser_tools(target: ToolRegistry, driver: Driver) -> ToolRegistry:
    """Register the ``browser`` toolset against a driver."""

    def outline(snapshot: Snapshot, *, max_chars: int = MAX_OUTLINE_CHARS) -> ToolResult:
        """Wrap a snapshot as an enveloped, tainting tool result."""
        return ToolResult(
            text=envelope(snapshot.render(max_chars=max_chars), origin=snapshot.url),
            display=f"{snapshot.title or snapshot.url}",
            tainted=True,
        )

    @target.tool(
        toolset="browser",
        danger=Danger.NETWORK,
        name="browser_navigate",
        max_result_chars=60_000,
        override=True,
    )
    def browser_navigate(params: NavigateParams, ctx: ToolContext) -> ToolResult:
        """Open a URL and return the page as a labelled outline of its elements."""
        try:
            snapshot = driver.navigate(params.url)
        except UrlRefused as exc:
            return ToolResult.error(refused_hint(exc))
        except BrowserError as exc:
            return ToolResult.error(str(exc))
        ctx.emit(f"opened {snapshot.url}")
        return outline(snapshot)

    @target.tool(
        toolset="browser",
        danger=Danger.SAFE,
        name="browser_snapshot",
        max_result_chars=100_000,
        override=True,
    )
    def browser_snapshot(params: SnapshotParams, ctx: ToolContext) -> ToolResult:
        """Re-read the current page. Element references from older snapshots expire."""
        try:
            snapshot = driver.snapshot()
        except BrowserError as exc:
            return ToolResult.error(str(exc))
        ctx.emit(f"read {snapshot.url}")
        return outline(snapshot, max_chars=params.max_chars)

    @target.tool(
        toolset="browser",
        danger=Danger.SAFE,
        name="browser_find",
        max_result_chars=8_000,
        override=True,
    )
    def browser_find(params: FindParams, ctx: ToolContext) -> ToolResult:
        """Find interactive elements on the current page without re-reading it all.

        Use on a large page: asking for the whole outline again to click one
        button costs the whole outline again.
        """
        try:
            snapshot = driver.snapshot()
        except BrowserError as exc:
            return ToolResult.error(str(exc))
        matches = snapshot.find(params.query, limit=MAX_FIND_RESULTS)
        if not matches:
            return ToolResult(
                text=(
                    f"Nothing interactive matches {params.query!r} on {snapshot.url}. "
                    f"Take a snapshot to see what is there."
                )
            )
        ctx.emit(f"{len(matches)} match(es) for {params.query!r}")
        listed = "\n".join("\n".join(node.render()) for node in matches)
        return ToolResult(text=envelope(listed, origin=snapshot.url), tainted=True)

    @target.tool(
        toolset="browser",
        # A click submits forms, confirms dialogs, and spends money.
        danger=Danger.WRITES,
        name="browser_click",
        max_result_chars=60_000,
        override=True,
    )
    def browser_click(params: ClickParams, ctx: ToolContext) -> ToolResult:
        """Click an element by reference. Returns the page after it settles."""
        summary = f"click {params.what or params.ref} on {driver.current_url()}"
        if not ctx.approve(
            ApprovalRequest(tool="browser_click", danger=Danger.WRITES, summary=summary)
        ):
            return ToolResult.error(f"{summary} was not approved")
        try:
            snapshot = driver.click(params.ref)
        except StaleReference as exc:
            # Reported rather than retried against a fresh snapshot: whatever
            # moved into that position is not what the model meant to click, and
            # guessing is how an agent cancels an order it meant to confirm.
            return ToolResult.error(stale_hint(exc))
        except (BrowserError, UrlRefused) as exc:
            return ToolResult.error(str(exc))
        ctx.emit(summary)
        return outline(snapshot)

    @target.tool(
        toolset="browser",
        danger=Danger.WRITES,
        name="browser_type",
        max_result_chars=60_000,
        override=True,
    )
    def browser_type(params: TypeParams, ctx: ToolContext) -> ToolResult:
        """Type into a field. Set submit to press Enter afterwards."""
        summary = f"type into {params.ref} on {driver.current_url()}"
        if not ctx.approve(
            ApprovalRequest(
                tool="browser_type",
                danger=Danger.WRITES,
                summary=summary,
                # The text itself, not a preview: an operator approving a form
                # submission needs to see what is being submitted.
                detail=params.text[:500],
            )
        ):
            return ToolResult.error(f"{summary} was not approved")
        try:
            snapshot = driver.type_text(params.ref, params.text, submit=params.submit)
        except StaleReference as exc:
            return ToolResult.error(stale_hint(exc))
        except (BrowserError, UrlRefused) as exc:
            return ToolResult.error(str(exc))
        ctx.emit(summary)
        return outline(snapshot)

    @target.tool(
        toolset="browser",
        danger=Danger.SAFE,
        name="browser_screenshot",
        max_result_chars=2_000,
        override=True,
    )
    def browser_screenshot(params: ScreenshotParams, ctx: ToolContext) -> ToolResult:
        """Capture the page as an image.

        Prefer browser_snapshot: the outline is far cheaper and lets you act by
        reference. Use this only when appearance itself is the question -- whether
        something rendered, what a chart shows.
        """
        try:
            media_type, data = driver.screenshot(full_page=params.full_page)
        except BrowserError as exc:
            return ToolResult.error(str(exc))
        import base64

        ctx.emit(f"captured {len(data)} bytes")
        return ToolResult(
            text=f"A screenshot of {driver.current_url()} follows.",
            images=(ImageBlock(media_type=media_type, data_b64=base64.b64encode(data).decode()),),
            display=f"screenshot of {driver.current_url()}",
            tainted=True,
        )

    return target


def browser_available() -> bool:
    """Whether Playwright is installed. The toolset hides itself when it is not.

    Absent rather than present-and-failing: a tool that always reports a missing
    dependency teaches the model to keep calling it.
    """
    import importlib.util

    return importlib.util.find_spec("playwright") is not None
