"""Letting the agent search its own past.

This is what makes compaction survivable. A summary necessarily drops detail;
without a way back to the original text, "we decided this last week" becomes
unanswerable and the agent confidently guesses instead. With it, the summary
only has to be a good index -- the facts are still on disk.

The tool is registered lazily by the surface that has a store, because it needs
one and the registry does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from harness_agentic.tools.spec import Danger, Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from harness_agentic.session.store import SessionStore
    from harness_agentic.tools.registry import ToolRegistry

MAX_RESULTS = 25


class SessionSearchParams(BaseModel):
    """Arguments for ``session_search``."""

    query: str = Field(description="Words to look for in past conversations.")
    this_session_only: bool = Field(
        default=False, description="Restrict the search to the current conversation."
    )
    limit: int = Field(10, gt=0, le=MAX_RESULTS, description="Maximum matches to return.")


def install_session_tools(registry: ToolRegistry, store: SessionStore) -> None:
    """Register the session tools against a live store."""

    def session_search(params: SessionSearchParams, ctx: ToolContext) -> ToolResult:
        hits = store.search(
            params.query,
            session_id=ctx.session_id if params.this_session_only else None,
            limit=params.limit,
        )
        if not hits:
            return ToolResult(text=f"no earlier messages match {params.query!r}")
        lines = [f"{hit.created_at:%Y-%m-%d %H:%M} [{hit.role}] {hit.snippet}" for hit in hits]
        return ToolResult(
            text="\n".join(lines),
            display=f"{len(hits)} match(es) for {params.query!r}",
        )

    registry.register(
        Tool(
            name="session_search",
            description=(
                "Search earlier conversations, including messages that context "
                "compaction has summarized away. Use this when the user refers "
                "to something decided before rather than guessing at it."
            ),
            toolset="core",
            params_model=SessionSearchParams,
            handler=session_search,
            danger=Danger.SAFE,
            source="builtin",
        ),
        override=True,
    )
