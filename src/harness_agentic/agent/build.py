"""Assembling a runnable agent from configuration.

Every surface -- the CLI now, the chat gateway and the web UI later -- builds
its agent through this one function. Wiring duplicated per surface is how they
drift apart, and then a fix applied to the terminal quietly fails to reach the
LINE bot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from harness_agentic.agent.delegate import (
    DelegationLimits,
    Delegator,
    install_delegate_tool,
)
from harness_agentic.agent.runner import AgentRunner, ModelChoice
from harness_agentic.core.cancel import CancelToken
from harness_agentic.core.clock import Clock, SystemClock
from harness_agentic.core.events import EventSink, Notice, ToolProgress, null_sink
from harness_agentic.core.types import Message, TextBlock
from harness_agentic.envs.local import LocalEnvironment
from harness_agentic.mcp.bridge import McpBridge
from harness_agentic.memory.budget import TokenBudget
from harness_agentic.memory.compactor import (
    SUMMARY_PROMPT,
    ContextCompactor,
    transcript_for_summary,
)
from harness_agentic.net.fetch import HttpFetcher
from harness_agentic.net.search import from_environment
from harness_agentic.prompts.builder import (
    PromptBuilder,
    identity_fragment,
    tool_guidance_fragment,
    volatile_fragment,
    workspace_fragment,
)
from harness_agentic.providers.base import CompletionRequest
from harness_agentic.providers.credentials import SecretResolver
from harness_agentic.providers.resolver import TransportResolver
from harness_agentic.session.sqlite_store import SqliteSessionStore
from harness_agentic.tools.approval import ApprovalPolicy
from harness_agentic.tools.builtin import builtin_registry
from harness_agentic.tools.builtin.browser import install_browser_tools
from harness_agentic.tools.builtin.data import (
    install_http_tools,
    install_kb_tools,
    install_sql_tools,
)
from harness_agentic.tools.builtin.session import install_session_tools
from harness_agentic.tools.builtin.web import install_web_tools
from harness_agentic.tools.dispatch import ToolExecutor
from harness_agentic.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from harness_agentic.browser.driver import Driver
    from harness_agentic.data.kb import Retriever
    from harness_agentic.data.sql import SqlSource
    from harness_agentic.envs.base import ExecEnvironment
    from harness_agentic.mcp.stdio import ServerConfig
    from harness_agentic.net.fetch import Fetcher
    from harness_agentic.net.search import SearchProvider
    from harness_agentic.providers.base import ProviderTransport
    from harness_agentic.session.store import SessionStore
    from harness_agentic.tools.registry import ToolRegistry
    from harness_agentic.tools.spec import ApprovalRequest


@dataclass
class RunContext:
    """The concrete :class:`~harness_agentic.tools.spec.ToolContext`."""

    session_id: str
    workspace_root: Path
    cwd: PurePath
    env: ExecEnvironment
    cancel: CancelToken
    surface: str = "cli"
    emit: EventSink = null_sink
    approval: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    _call_id: str = ""

    def emit_progress(self, message: str) -> None:
        """Report progress from inside a running tool."""
        self.emit(ToolProgress(call_id=self._call_id, message=message))

    def approve(self, request: ApprovalRequest) -> bool:
        """Ask the policy. Kept on the context so tools can gate sub-steps."""
        return self.approval.check(request).granted


class _ContextAdapter:
    """Bridges :class:`RunContext` onto the ToolContext protocol."""

    def __init__(self, inner: RunContext) -> None:
        self._inner = inner
        self.session_id = inner.session_id
        self.workspace_root = inner.workspace_root
        self.cwd = inner.cwd
        self.env = inner.env
        self.cancel = inner.cancel
        self.surface = inner.surface

    def emit(self, message: str) -> None:
        """Forward a tool's progress line onto the event stream."""
        self._inner.emit_progress(message)

    def approve(self, request: ApprovalRequest) -> bool:
        """Defer to the approval policy."""
        return self._inner.approve(request)


@dataclass
class AgentBundle:
    """A runner plus the pieces a surface needs to drive and inspect it."""

    runner: AgentRunner
    store: SessionStore
    registry: ToolRegistry
    context: RunContext
    prompts: PromptBuilder
    executor: ToolExecutor
    """Exposed so a surface can read what ran, and whether anything it ran
    brought untrusted content into the conversation."""
    mcp: McpBridge | None = None
    """The connected MCP servers, if any. A surface that owns this bundle owns
    closing it -- an MCP server is a child process, and one left running after
    the agent goes away is a leak the operator finds with `ps`."""


def build_agent(
    *,
    model: str,
    workspace: Path,
    sessions_dir: Path,
    toolsets: Sequence[str] = ("file", "terminal"),
    fetcher: Fetcher | None = None,
    search: SearchProvider | None = None,
    sql_source: SqlSource | None = None,
    retriever: Retriever | None = None,
    http_fetcher: Fetcher | None = None,
    delegation: DelegationLimits | None = None,
    mcp_servers: Sequence[ServerConfig] = (),
    browser: Driver | None = None,
    surface: str = "cli",
    emit: EventSink = null_sink,
    approval: ApprovalPolicy | None = None,
    clock: Clock | None = None,
    fallbacks: Sequence[str] = (),
    transports: dict[str, ProviderTransport] | None = None,
    stream: bool = True,
    max_iterations: int = 40,
) -> AgentBundle:
    """Wire up an agent ready to run turns.

    ``transports`` lets a caller inject one -- which is how the whole stack is
    exercised in tests with ``FakeTransport`` and no API key.

    The ``data`` and ``retrieval`` tools appear only when the thing they read
    from is supplied: no ``sql_query`` without a ``sql_source``, no ``kb_search``
    without a ``retriever``, no ``http_request`` without an ``http_fetcher``.
    That last one is separate from ``fetcher`` on purpose -- ``web_fetch`` wants
    a broad policy with an internal-range denylist, and ``http_request`` wants a
    narrow one with a host allowlist, because it can change what it calls.
    """
    the_clock = clock or SystemClock()
    resolver = TransportResolver(clock=the_clock, overrides=transports)
    chain = [
        ModelChoice(transport=r.transport, model=r.model, provider=r.provider)
        for r in resolver.chain(model, fallbacks)
    ]

    # Forked, not the process-wide registry: the session store and the HTTP
    # fetcher installed below are this agent's, and a second agent in the same
    # process must not inherit them.
    tool_registry = builtin_registry()
    store = SqliteSessionStore(sessions_dir / "state.db")
    # Needs a live store, so it is registered here rather than at import time.
    install_session_tools(tool_registry, store)
    if "web" in toolsets:
        # Only built when asked for. Opening an HTTP client and resolving a
        # search key for an agent that will never fetch anything is waste, and
        # the URL policy is deployment configuration a caller may want to
        # narrow -- so a caller may pass its own fetcher instead.
        install_web_tools(
            tool_registry,
            fetcher or HttpFetcher(),
            search=search or from_environment(SecretResolver(), fetcher or HttpFetcher()),
        )
    if sql_source is not None:
        install_sql_tools(tool_registry, sql_source)
    if retriever is not None:
        install_kb_tools(tool_registry, retriever)
    if http_fetcher is not None:
        install_http_tools(tool_registry, http_fetcher)
    if browser is not None:
        install_browser_tools(tool_registry, browser)

    bridge: McpBridge | None = None
    if mcp_servers:
        # Connected before the toolsets are resolved, because each server
        # contributes its own `mcp:<name>` toolset and the agent has to be able
        # to ask for it.
        bridge = McpBridge(registry=tool_registry)
        bridge.connect_all(mcp_servers)
        for line in bridge.report():
            emit(Notice("warning" if "FAILED" in line else "info", line))

    mcp_toolsets = [f"mcp:{name}" for name in (bridge.servers if bridge else ())]
    limits = delegation or DelegationLimits(allowed_toolsets=(*toolsets, *mcp_toolsets))
    if limits.max_depth > 0:
        # A child gets a fresh agent with the same wiring and one less level of
        # depth. Building it lazily matters: an agent that never delegates must
        # not pay for a second store and a second HTTP client.
        def spawn(task: str, granted: Sequence[str], max_steps: int) -> AgentBundle:
            del task  # the prompt is passed to run_turn, not to the constructor
            return build_agent(
                model=model,
                workspace=workspace,
                sessions_dir=sessions_dir,
                toolsets=granted,
                fetcher=fetcher,
                search=search,
                sql_source=sql_source,
                retriever=retriever,
                http_fetcher=http_fetcher,
                delegation=limits.child(),
                surface=surface,
                emit=emit,
                approval=approval,
                clock=clock,
                fallbacks=fallbacks,
                transports=transports,
                stream=stream,
                max_iterations=max_steps,
            )

        install_delegate_tool(tool_registry, Delegator(factory=spawn, limits=limits))

    # `core` is always on: an agent that cannot reach its own history has to
    # guess at anything compaction summarized away.
    active_toolsets = list(dict.fromkeys(["core", *toolsets, *mcp_toolsets]))
    resolved_tools = tool_registry.resolve(enabled_toolsets=active_toolsets, surface=surface)
    session = store.create(source=surface, cwd=workspace, model=chain[0].model)

    policy = approval or ApprovalPolicy(surface=surface)
    context = RunContext(
        session_id=session.id,
        workspace_root=workspace,
        cwd=PurePath(workspace),
        env=LocalEnvironment(workspace),
        cancel=CancelToken(),
        surface=surface,
        emit=emit,
        approval=policy,
    )

    prompts = (
        PromptBuilder()
        .add(identity_fragment())
        .add(tool_guidance_fragment(tool_registry.schemas(resolved_tools)))
        .add(workspace_fragment(str(workspace)))
        .add(
            volatile_fragment(now=the_clock.now().isoformat(timespec="seconds"), cwd=str(workspace))
        )
    )

    # Summarize with the cheapest model in the chain rather than the primary:
    # compaction happens on the longest conversations, which is exactly when
    # paying top-tier rates to write a summary hurts most.
    summary_choice = chain[-1]

    def summarize(span: Sequence[Message]) -> str:
        request = CompletionRequest(
            model=summary_choice.model,
            messages=(
                Message(
                    role="user",
                    content=(TextBlock(f"{SUMMARY_PROMPT}\n\n{transcript_for_summary(span)}"),),
                    created_at=the_clock.now(),
                ),
            ),
            max_output_tokens=1024,
            stream=False,
        )
        return summary_choice.transport.send(request).message.text()

    executor = ToolExecutor(tool_registry, approval=policy, emit=emit)
    primary = resolver.resolve(model)
    runner = AgentRunner(
        chain=chain,
        registry=tool_registry,
        executor=executor,
        enabled_toolsets=active_toolsets,
        budget=TokenBudget(
            window=primary.info.context_window,
            reserve_output=min(primary.info.max_output_tokens, 8192),
        ),
        compactor=ContextCompactor(summarize),
        prompts=prompts,
        store=store,
        context=_ContextAdapter(context),
        emit=emit,
        clock=the_clock,
        stream=stream,
        max_iterations=max_iterations,
    )
    # The session the caller will run against; surfaces read it off the store.
    context.session_id = session.id
    return AgentBundle(
        runner=runner,
        store=store,
        registry=tool_registry,
        context=context,
        prompts=prompts,
        executor=executor,
        mcp=bridge,
    )
