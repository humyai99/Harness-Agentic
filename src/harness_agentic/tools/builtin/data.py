"""The ``data`` and ``retrieval`` toolsets.

Three tools, and each is installed only when the operator has supplied the thing
it reads from. There is no ``sql_query`` unless a source was configured, no
``kb_search`` unless a corpus was, and no ``http_request`` unless a fetcher was.
The alternative -- always present, failing with "not configured" -- trains the
model to keep calling a tool that will never work.

``http_request`` is the one to be careful with. Unlike ``web_fetch`` it can POST,
send arbitrary headers, and reach an API that changes something, so it is
classified higher than network and goes through the approval policy on every
surface. It also defaults to an **allowlist**: a fetch tool with a denylist of
internal ranges is defence in depth, but a request tool that can call any API on
the internet with a header of the model's choosing wants a list of the three
hosts it is actually for.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from harness_agentic.data.sql import NotReadOnly
from harness_agentic.net.policy import UrlRefused
from harness_agentic.tools.builtin.web import envelope
from harness_agentic.tools.spec import ApprovalRequest, Danger, ToolContext, ToolResult

if TYPE_CHECKING:
    from harness_agentic.data.kb import Retriever
    from harness_agentic.data.sql import SqlSource
    from harness_agentic.net.fetch import Fetcher
    from harness_agentic.tools.registry import ToolRegistry

MAX_ROWS = 200
MAX_PASSAGES = 10
ALLOWED_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD")
MAX_BODY_CHARS = 20_000


class SqlParams(BaseModel):
    """Arguments for ``sql_query``."""

    sql: str = Field(
        min_length=1,
        description=(
            "One read-only SQL statement. SELECT, WITH, EXPLAIN or VALUES only; "
            "no semicolons, and no statement that writes."
        ),
    )
    limit: int = Field(50, gt=0, le=MAX_ROWS, description="Maximum rows to return.")


class SchemaParams(BaseModel):
    """Arguments for ``sql_schema``."""

    refresh: bool = Field(default=False, description="Re-read the schema from the database.")


class KbParams(BaseModel):
    """Arguments for ``kb_search``."""

    query: str = Field(min_length=1, description="What to look for in the knowledge base.")
    limit: int = Field(5, gt=0, le=MAX_PASSAGES, description="How many passages to return.")


class HttpParams(BaseModel):
    """Arguments for ``http_request``."""

    url: str = Field(description="Absolute http or https URL.")
    method: str = Field("GET", description=f"One of {', '.join(ALLOWED_METHODS)}.")
    body: str = Field("", description="JSON request body. Ignored for GET and HEAD.")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra request headers. Do not put credentials here -- the operator "
            "configures those outside the conversation."
        ),
    )


def install_sql_tools(target: ToolRegistry, source: SqlSource) -> ToolRegistry:
    """Register read-only SQL access against one source."""

    @target.tool(toolset="data", danger=Danger.SAFE, name="sql_schema", override=True)
    def sql_schema(params: SchemaParams, ctx: ToolContext) -> ToolResult:
        """List the tables and columns available to query."""
        del params
        try:
            described = source.describe()
        except Exception as exc:  # a broken connection is a tool error
            return ToolResult.error(f"could not read the schema: {exc}")
        ctx.emit(f"described {source.name}")
        return ToolResult(text=described, display=f"schema of {source.name}")

    @target.tool(
        toolset="data",
        # Reading is safe; the risk is volume and content, and both are handled
        # by the row cap and the column masking rather than by an approval
        # prompt nobody can answer on a chat surface.
        danger=Danger.SAFE,
        name="sql_query",
        max_result_chars=60_000,
        override=True,
    )
    def sql_query(params: SqlParams, ctx: ToolContext) -> ToolResult:
        """Run one read-only SQL query. Writes are refused, not attempted."""
        try:
            rows = source.query(params.sql, limit=params.limit)
        except NotReadOnly as exc:
            # Named precisely, so the model rewrites the query rather than
            # trying the same shape three more ways.
            return ToolResult.error(f"refused: {exc}")
        except Exception as exc:  # a SQL error is information, not a crash
            return ToolResult.error(f"query failed: {exc}")

        ctx.emit(f"{len(rows.rows)} row(s) from {source.name}")
        return ToolResult(
            text=rows.render(),
            display=f"{len(rows.rows)} row(s) from {source.name}",
            data={"columns": list(rows.columns), "masked": list(rows.masked)},
        )

    return target


def install_kb_tools(target: ToolRegistry, retriever: Retriever) -> ToolRegistry:
    """Register knowledge-base search against one retriever."""

    @target.tool(
        toolset="retrieval",
        danger=Danger.SAFE,
        name="kb_search",
        max_result_chars=40_000,
        override=True,
    )
    def kb_search(params: KbParams, ctx: ToolContext) -> ToolResult:
        """Search the knowledge base. Answer from what it returns, not from memory."""
        try:
            passages = retriever.search(params.query, limit=params.limit)
        except Exception as exc:  # an index error is a tool error
            return ToolResult.error(f"search failed: {exc}")
        if not passages:
            # Says what was searched, so the model does not conclude the topic
            # does not exist when the corpus simply does not cover it.
            return ToolResult(
                text=(
                    f"No passages matched {params.query!r}. "
                    f"The knowledge base holds {retriever.count()} passage(s); "
                    f"it may not cover this."
                )
            )
        ctx.emit(f"{len(passages)} passage(s) for {params.query!r}")
        return ToolResult(
            text="\n\n---\n\n".join(passage.render() for passage in passages),
            display=f"{len(passages)} passage(s) for {params.query!r}",
        )

    return target


def install_http_tools(target: ToolRegistry, fetcher: Fetcher) -> ToolRegistry:
    """Register ``http_request`` against a fetcher.

    The fetcher's policy is what makes this safe to expose, so pass one with an
    ``allowed_hosts`` set. A request tool with an open policy is a proxy the
    model controls.
    """

    @target.tool(
        toolset="data",
        # Higher than the fetch tool: this can change something on the other
        # end, so it goes through the approval policy on every surface.
        danger=Danger.WRITES,
        name="http_request",
        max_result_chars=40_000,
        override=True,
    )
    def http_request(params: HttpParams, ctx: ToolContext) -> ToolResult:
        """Call an HTTP API. Only hosts the operator allowlisted are reachable."""
        method = params.method.upper()
        if method not in ALLOWED_METHODS:
            return ToolResult.error(f"{method} is not an allowed method")

        if not ctx.approve(
            ApprovalRequest(
                tool="http_request",
                danger=Danger.WRITES,
                summary=f"{method} {params.url}",
                detail=params.body[:500] or None,
            )
        ):
            return ToolResult.error(f"{method} {params.url} was not approved")

        payload: object = None
        if params.body and method not in ("GET", "HEAD"):
            try:
                payload = json.loads(params.body)
            except ValueError as exc:
                return ToolResult.error(f"the body is not valid JSON: {exc}")

        try:
            result = fetcher.request(
                method, params.url, headers=params.headers or None, json_body=payload
            )
        except UrlRefused as exc:
            return ToolResult.error(f"refused: {exc}")
        except Exception as exc:  # a transport failure is a tool error
            return ToolResult.error(f"request failed: {exc}")

        ctx.emit(f"{method} {result.url} -> {result.status}")
        body = result.text()[:MAX_BODY_CHARS] if result.is_text else f"<{len(result.body)} bytes>"
        return ToolResult(
            text=envelope(f"HTTP {result.status}\n\n{body}", origin=result.url),
            display=f"{method} {result.url} -> {result.status}",
            is_error=not result.ok,
            # A response body is third-party content, exactly like a page.
            tainted=True,
        )

    return target
