"""The agent loop.

One class, and it owns the decisions no other layer has enough context to
make: which model to use, what the prompt is, when to retry, when to fall back,
when to stop. Transports know their wire format and nothing else; tools know
their job and nothing else.

Two ordering rules in here are load-bearing and easy to get wrong:

* **Persist before acting.** The user message is written before the request
  goes out, and the assistant message before its tool calls run. A crash then
  loses at most the outcome of one tool, not the reasoning that led to it.
* **Sanitize last.** Repair runs against the history immediately before it is
  encoded, never against what is stored. The store keeps what actually
  happened; the wire gets a version the provider will accept.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from harness_agentic.agent.sanitize import sanitize
from harness_agentic.core.cancel import CancelToken
from harness_agentic.core.clock import Clock, SystemClock
from harness_agentic.core.events import (
    CompactionFinished,
    CompactionStarted,
    EventSink,
    IterationStarted,
    Notice,
    ProviderFallback,
    RetryScheduled,
    TextChunk,
    ThinkingChunk,
    TurnFinished,
    TurnStarted,
    UsageReported,
    null_sink,
)
from harness_agentic.core.stream import (
    StreamAccumulator,
    TextDelta,
    ThinkingDelta,
)
from harness_agentic.core.types import (
    Message,
    ModelResponse,
    Usage,
    tool_result_message,
    user_message,
)
from harness_agentic.errors import (
    AuthError,
    CompactionDeferred,
    ContentFiltered,
    ContextExhausted,
    ContextOverflow,
    Interrupted,
    MalformedResponse,
    ModelUnavailable,
    ProviderExhausted,
    RateLimited,
    TransientProviderError,
)
from harness_agentic.memory.compactor import SUMMARY_MARKER
from harness_agentic.providers.base import CompletionRequest, ProviderTransport

if TYPE_CHECKING:
    from harness_agentic.memory.budget import TokenBudget
    from harness_agentic.memory.compactor import CompactionNote, ContextCompactor
    from harness_agentic.prompts.builder import PromptBuilder
    from harness_agentic.session.store import SessionRecord, SessionStore
    from harness_agentic.tools.dispatch import ToolExecutor
    from harness_agentic.tools.registry import ToolRegistry
    from harness_agentic.tools.spec import ToolContext

ExitReason = Literal[
    "completed",
    "max_iterations",
    "interrupted",
    "content_filter",
    "error",
]

MAX_API_RETRIES = 5
MAX_CONTINUATIONS = 2
"""How many times to nudge a truncated answer to continue before accepting it."""
MAX_TOOL_REISSUES = 2
"""How many times to ask again when the provider claims a tool call but sends none."""
MAX_COMPACTIONS_PER_TURN = 3
"""After this the request still does not fit and pretending otherwise wastes calls."""


@dataclass(frozen=True, slots=True)
class TurnResult:
    """What one user turn produced."""

    final_text: str
    exit_reason: ExitReason
    iterations: int
    usage: Usage
    appended: tuple[Message, ...] = ()
    error: str | None = None
    provider_used: str = ""
    model_used: str = ""


@dataclass
class ModelChoice:
    """One entry in the fallback chain."""

    transport: ProviderTransport
    model: str
    provider: str

    def label(self) -> str:
        """A human-readable ``provider/model``."""
        return f"{self.provider}/{self.model}"


class AgentRunner:
    """Runs turns against a model, dispatching tools until the model stops."""

    def __init__(
        self,
        *,
        chain: Sequence[ModelChoice],
        registry: ToolRegistry,
        executor: ToolExecutor,
        prompts: PromptBuilder,
        store: SessionStore,
        context: ToolContext,
        enabled_toolsets: Sequence[str] | None = None,
        budget: TokenBudget | None = None,
        compactor: ContextCompactor | None = None,
        emit: EventSink = null_sink,
        clock: Clock | None = None,
        max_iterations: int = 40,
        max_output_tokens: int = 8192,
        stream: bool = True,
    ) -> None:
        """Assemble a runner from its collaborators."""
        if not chain:
            msg = "an agent needs at least one model in its fallback chain"
            raise ValueError(msg)
        self._chain = list(chain)
        self._registry = registry
        self._executor = executor
        self._prompts = prompts
        self._store = store
        self._context = context
        self._enabled_toolsets = list(enabled_toolsets) if enabled_toolsets else None
        self._budget = budget
        self._compactor = compactor
        self._emit = emit
        self._clock = clock or SystemClock()
        self._max_iterations = max_iterations
        self._max_output_tokens = max_output_tokens
        self._stream = stream

        self._chain_index = 0
        self._steer: list[str] = []
        self.cancel = CancelToken()

    # -- external control ---------------------------------------------------

    def interrupt(self, reason: str = "interrupted by user") -> None:
        """Ask the current turn to stop at the next checkpoint."""
        self.cancel.cancel(reason)

    def steer(self, text: str) -> None:
        """Inject a user message to be picked up at the next loop head.

        Steering rather than interrupting is what lets someone correct an agent
        mid-task without discarding the work it has already done.
        """
        self._steer.append(text)

    @property
    def current(self) -> ModelChoice:
        """The model currently in use."""
        return self._chain[self._chain_index]

    # -- the loop -----------------------------------------------------------

    def run_turn(  # noqa: PLR0912, PLR0915  -- the loop is one readable state machine
        self, user_input: str, *, session: SessionRecord
    ) -> TurnResult:
        """Run one user turn to completion."""
        self.cancel = CancelToken()
        self._emit(TurnStarted(session_id=session.id, model=self.current.label()))

        history = list(self._store.history(session.id))
        opening = user_message(user_input, now=self._clock.now())
        # Persisted before the request: a crash mid-request must not lose what
        # the user asked for.
        self._store.append(session.id, [opening])
        history.append(opening)

        appended: list[Message] = [opening]
        total = Usage()
        iterations = 0
        api_retries = 0
        continuations = 0
        reissues = 0
        stable_digest: str | None = None
        compactions = 0
        final_text = ""
        exit_reason: ExitReason = "completed"
        error: str | None = None

        with self._store.turn_lease(session.id):
            while True:
                if self.cancel.is_set():
                    exit_reason = "interrupted"
                    break
                if iterations >= self._max_iterations:
                    exit_reason = "max_iterations"
                    break

                for steer in self._drain_steer():
                    message = user_message(steer, now=self._clock.now())
                    self._store.append(session.id, [message])
                    history.append(message)
                    appended.append(message)

                self._emit(IterationStarted(index=iterations))

                # Toolset gating is a security control, not a convenience: an
                # agent answering strangers on a public chat account must not
                # have `terminal` within reach. It has to be applied here, on
                # every iteration, and not just when the prompt was built.
                tools = self._registry.resolve(
                    enabled_toolsets=self._enabled_toolsets, surface=self._surface()
                )
                system = self._prompts.build()
                digest = self._prompts.stable_digest()
                if stable_digest is None:
                    stable_digest = digest
                elif digest != stable_digest:
                    # Prompt stability is not cosmetic: a mid-conversation edit
                    # to the cached prefix both loses the cache and can make the
                    # model contradict what it already said.
                    self._emit(
                        Notice("warning", "the stable prompt changed mid-turn; cache will miss")
                    )
                    stable_digest = digest

                transport = self.current.transport
                repaired, report = sanitize(
                    history, transport.sanitize_rules(), now=self._clock.now()
                )
                if not report.clean:
                    self._emit(Notice("info", f"history repaired: {report.summary()}"))

                # Budget before building the request, not after: an overflow
                # discovered by the provider costs a full round trip, and the
                # estimate is cheap.
                if self._budget is not None and self._compactor is not None:
                    estimate = self._budget.estimate(
                        system, repaired, self._registry.schemas(tools)
                    )
                    if self._budget.over_threshold(estimate):
                        if compactions >= MAX_COMPACTIONS_PER_TURN:
                            exit_reason, error = "error", "context exhausted after compaction"
                            break
                        try:
                            history, note = self._compact(
                                history, session, aggressiveness=compactions
                            )
                            compactions += 1
                            self._emit(
                                CompactionFinished(
                                    replaced_messages=note.messages_before - note.messages_after,
                                    tokens_before=note.tokens_before,
                                    tokens_after=note.tokens_after,
                                )
                            )
                            continue
                        except CompactionDeferred as exc:
                            # Another writer holds the session. Soft: back off
                            # and retry rather than discarding the conversation.
                            self._emit(Notice("info", f"compaction deferred: {exc}"))
                            self._clock.sleep(0.2)
                            continue
                        except ContextExhausted as exc:
                            exit_reason, error = "error", str(exc)
                            break

                request = CompletionRequest(
                    model=self.current.model,
                    messages=tuple(repaired),
                    system=system,
                    tools=self._registry.schemas(tools),
                    max_output_tokens=self._max_output_tokens,
                    stream=self._stream,
                )

                try:
                    response = self._complete(transport, request)
                except Interrupted as exc:
                    exit_reason, error = "interrupted", str(exc)
                    break
                except ContentFiltered as exc:
                    exit_reason, error = "content_filter", str(exc)
                    break
                except (RateLimited, TransientProviderError, MalformedResponse) as exc:
                    if api_retries < MAX_API_RETRIES:
                        api_retries += 1
                        delay = self._backoff(api_retries, exc)
                        self._emit(
                            RetryScheduled(attempt=api_retries, delay_s=delay, reason=str(exc))
                        )
                        self._clock.sleep(delay)
                        continue
                    if self._advance(str(exc)):
                        api_retries = 0
                        continue
                    exit_reason, error = "error", str(exc)
                    break
                except ContextOverflow as exc:
                    # The estimator was wrong in the expensive direction. Bump
                    # it hard and compact rather than repeating the request.
                    if self._budget is not None:
                        self._budget.shrink_from_error()
                    if self._compactor is not None and compactions < MAX_COMPACTIONS_PER_TURN:
                        try:
                            history, note = self._compact(
                                history, session, aggressiveness=compactions
                            )
                            compactions += 1
                            # Same event as the budget-driven path. A silent
                            # compaction leaves the operator wondering why the
                            # agent suddenly forgot the middle of the session.
                            self._emit(
                                CompactionFinished(
                                    replaced_messages=note.messages_before - note.messages_after,
                                    tokens_before=note.tokens_before,
                                    tokens_after=note.tokens_after,
                                )
                            )
                            continue
                        except (ContextExhausted, CompactionDeferred):
                            pass
                    if self._advance(str(exc)):
                        api_retries = 0
                        continue
                    exit_reason, error = "error", str(exc)
                    break
                except (AuthError, ModelUnavailable) as exc:
                    # Nothing to gain from retrying any of these on the same
                    # model: bad credentials stay bad, a missing model stays
                    # missing, and an overflow needs compaction (M3) not a
                    # repeat of the identical request.
                    if self._advance(str(exc)):
                        api_retries = 0
                        continue
                    exit_reason, error = "error", str(exc)
                    break
                except ProviderExhausted as exc:
                    exit_reason, error = "error", str(exc)
                    break

                iterations += 1
                api_retries = 0
                total = total + response.usage
                if self._budget is not None and response.usage.input_tokens:
                    # Every real response teaches the estimator something, and
                    # the error is systematic enough to be worth learning.
                    self._budget.observe(response.usage.input_tokens)
                self._emit(UsageReported(usage=response.usage, cumulative=total))

                self._store.append(session.id, [response.message], usage=response.usage)
                history.append(response.message)
                appended.append(response.message)

                if response.finish_reason == "interrupted":
                    final_text = response.message.text()
                    exit_reason = "interrupted"
                    break

                calls = response.tool_uses()

                if response.finish_reason == "length" and not calls:
                    if continuations >= MAX_CONTINUATIONS:
                        final_text = response.message.text()
                        break
                    continuations += 1
                    nudge = user_message(
                        "Your previous message was cut off. Continue from where it stopped.",
                        now=self._clock.now(),
                    )
                    self._store.append(session.id, [nudge])
                    history.append(nudge)
                    appended.append(nudge)
                    continue

                if response.finish_reason == "tool_calls" and not calls:
                    if reissues >= MAX_TOOL_REISSUES:
                        exit_reason, error = "error", "provider promised a tool call and sent none"
                        break
                    reissues += 1
                    nudge = user_message(
                        "You indicated a tool call but none arrived. Reissue it.",
                        now=self._clock.now(),
                    )
                    self._store.append(session.id, [nudge])
                    history.append(nudge)
                    appended.append(nudge)
                    continue

                if not calls:
                    final_text = response.message.text()
                    exit_reason = "completed"
                    break

                results = self._executor.execute_batch(calls, self._context)
                message = tool_result_message(results, now=self._clock.now())
                self._store.append(session.id, [message])
                history.append(message)
                appended.append(message)

                if self.cancel.is_set():
                    exit_reason = "interrupted"
                    break

        self._store.update(
            session.id,
            model=self.current.model,
            total_usage=session.total_usage + total,
        )
        self._emit(
            TurnFinished(reason=exit_reason, iterations=iterations, usage=total, error=error)
        )
        return TurnResult(
            final_text=final_text,
            exit_reason=exit_reason,
            iterations=iterations,
            usage=total,
            appended=tuple(appended),
            error=error,
            provider_used=self.current.provider,
            model_used=self.current.model,
        )

    def _compact(
        self,
        history: list[Message],
        session: SessionRecord,
        *,
        aggressiveness: int,
    ) -> tuple[list[Message], CompactionNote]:
        """Compact the working history and record it in the store.

        The store keeps the originals hidden rather than deleting them, so the
        compaction stays reversible and full-text search still reaches the
        text that was summarized away.
        """
        assert self._compactor is not None  # noqa: S101  -- guarded by the caller
        self._emit(CompactionStarted(from_seq=0, to_seq=len(history) - 1))
        rebuilt, note = self._compactor.compact(
            history, now=self._clock.now(), aggressiveness=aggressiveness
        )
        recorder = getattr(self._store, "record_compaction", None)
        if callable(recorder):
            summary = next(
                (m for m in rebuilt if SUMMARY_MARKER in m.text()),
                None,
            )
            if summary is not None:
                recorder(session.id, replaced=note.replaced, summary=summary)
        return rebuilt, note

    # -- helpers ------------------------------------------------------------

    def _complete(self, transport: ProviderTransport, request: CompletionRequest) -> ModelResponse:
        """Issue one request, streaming when enabled."""
        if not request.stream:
            response = transport.send(request, cancel=self.cancel)
            transport.validate_response(response)
            return response

        started = time.monotonic()
        accumulator = StreamAccumulator(
            provider=self.current.provider, model=request.model, now=self._clock.now()
        )
        for event in transport.stream(request, cancel=self.cancel):
            accumulator.feed(event)
            if isinstance(event, TextDelta):
                self._emit(TextChunk(event.text))
            elif isinstance(event, ThinkingDelta):
                self._emit(ThinkingChunk(event.text))

        interrupted = self.cancel.is_set()
        response = accumulator.finalize(
            interrupted=interrupted, latency_ms=int((time.monotonic() - started) * 1000)
        )
        if interrupted:
            # Return the partial answer rather than raising, so the caller can
            # persist it before stopping. finalize() has already dropped the
            # blocks that would be illegal to replay, so what comes back is a
            # legal prefix for the next turn.
            return response
        transport.validate_response(response)
        return response

    def _advance(self, reason: str) -> bool:
        """Move to the next model in the chain. False when none is left."""
        if self._chain_index + 1 >= len(self._chain):
            return False
        previous = self.current.label()
        self._chain_index += 1
        self._emit(
            ProviderFallback(from_model=previous, to_model=self.current.label(), reason=reason)
        )
        return True

    def _backoff(self, attempt: int, exc: Exception) -> float:
        """Exponential backoff with full jitter, honouring a provider hint.

        Full jitter rather than a fixed schedule: several sessions hitting the
        same rate limit would otherwise retry in lockstep and rate-limit each
        other again.
        """
        if isinstance(exc, RateLimited) and exc.retry_after_s:
            return min(exc.retry_after_s, 60.0)
        ceiling = min(2.0**attempt, 30.0)
        return random.uniform(0.0, ceiling)  # noqa: S311  -- jitter, not cryptography

    def _drain_steer(self) -> list[str]:
        """Take any steering messages queued since the last iteration."""
        pending, self._steer = self._steer, []
        return pending

    def _surface(self) -> str:
        """Which surface this runner serves. Drives tool and approval gating."""
        return getattr(self._context, "surface", "cli")
