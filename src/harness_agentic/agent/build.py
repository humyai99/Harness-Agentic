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

from harness_agentic.agent.runner import AgentRunner, ModelChoice
from harness_agentic.core.cancel import CancelToken
from harness_agentic.core.clock import Clock, SystemClock
from harness_agentic.core.events import EventSink, ToolProgress, null_sink
from harness_agentic.envs.local import LocalEnvironment
from harness_agentic.prompts.builder import (
    PromptBuilder,
    identity_fragment,
    tool_guidance_fragment,
    volatile_fragment,
    workspace_fragment,
)
from harness_agentic.providers.catalog import parse_model_ref
from harness_agentic.providers.credentials import SecretResolver, resolve_credentials
from harness_agentic.providers.transports.anthropic import AnthropicTransport
from harness_agentic.session.store import JsonlSessionStore, SessionStore
from harness_agentic.tools.approval import ApprovalPolicy
from harness_agentic.tools.builtin import install_builtins
from harness_agentic.tools.dispatch import ToolExecutor
from harness_agentic.tools.registry import ToolRegistry, registry

if TYPE_CHECKING:
    from harness_agentic.envs.base import ExecEnvironment
    from harness_agentic.providers.base import ProviderTransport
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


def build_transport(
    provider: str, *, resolver: SecretResolver, clock: Clock | None = None
) -> ProviderTransport:
    """Construct the transport for one provider.

    The only place a provider name maps to a class. Adding a provider means
    adding a branch here and an entry in the catalog -- nowhere else.
    """
    credentials = resolve_credentials(provider, resolver)
    if provider == "anthropic":
        return AnthropicTransport(credentials=credentials, clock=clock)
    msg = (
        f"provider {provider!r} has no transport yet; anthropic is available, "
        f"and openai-compatible providers land in M2"
    )
    raise NotImplementedError(msg)


@dataclass
class AgentBundle:
    """A runner plus the pieces a surface needs to drive and inspect it."""

    runner: AgentRunner
    store: SessionStore
    registry: ToolRegistry
    context: RunContext
    prompts: PromptBuilder


def build_agent(
    *,
    model: str,
    workspace: Path,
    sessions_dir: Path,
    toolsets: Sequence[str] = ("file", "terminal"),
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
    """
    the_clock = clock or SystemClock()
    resolver = SecretResolver()
    supplied = transports or {}

    def transport_for(reference: str) -> ModelChoice:
        provider, name = parse_model_ref(reference)
        transport = supplied.get(provider) or build_transport(
            provider, resolver=resolver, clock=the_clock
        )
        return ModelChoice(transport=transport, model=name, provider=provider)

    chain = [transport_for(model), *(transport_for(ref) for ref in fallbacks)]

    tool_registry = install_builtins(registry)
    resolved_tools = tool_registry.resolve(enabled_toolsets=list(toolsets), surface=surface)

    store = JsonlSessionStore(sessions_dir)
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

    executor = ToolExecutor(tool_registry, approval=policy, emit=emit)
    runner = AgentRunner(
        chain=chain,
        registry=tool_registry,
        executor=executor,
        enabled_toolsets=list(toolsets),
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
        runner=runner, store=store, registry=tool_registry, context=context, prompts=prompts
    )
