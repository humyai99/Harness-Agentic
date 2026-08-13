"""The provider transport contract.

One transport per *API shape*, not per vendor: a single
``ChatCompletionsTransport`` serves eleven OpenAI-compatible endpoints, and
their differences arrive as :class:`~harness_agentic.providers.catalog.ChatCompatQuirks`.

What lives inside a transport: message conversion, tool conversion, request
assembly, HTTP, SSE framing, and response normalization. Streaming is included
deliberately -- ``content_block_delta`` versus ``choices[0].delta`` versus
Gemini's chunked JSON is irreducibly provider-specific, and hoisting it out
just recreates a provider switch one level up.

What stays outside: retry policy, model fallback, credential discovery, token
budgeting, and compaction. Those belong to the agent loop, which is the only
component with enough context to make them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypeAlias

from harness_agentic.core.cancel import NEVER_CANCELLED, CancelToken
from harness_agentic.core.types import (
    Message,
    ModelResponse,
    SystemPrompt,
    ToolSchema,
)
from harness_agentic.errors import MalformedResponse, ProviderError

if TYPE_CHECKING:
    from harness_agentic.core.secrets import Secret
    from harness_agentic.core.stream import StreamEvent
    from harness_agentic.providers.catalog import TransportFeature

WireRequest: TypeAlias = dict[str, Any]
"""A provider-shaped request body, ready to serialize."""

CredentialSource: TypeAlias = Literal["env", "dotenv", "keyring", "config", "explicit", "none"]
"""Where a credential was found. Reported by `harn doctor`; never the value."""


@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    """Extended-thinking settings, where the model supports them."""

    effort: Literal["off", "low", "medium", "high"] = "off"
    budget_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Credentials:
    """Resolved credentials for one provider.

    ``api_key`` is a :class:`~harness_agentic.core.secrets.Secret` rather than a
    ``str`` so that mypy stops a raw key from reaching a log statement or an
    f-string. It was documented as one and typed as the other, which is the
    worst of both: this dataclass's generated ``__repr__`` printed the key in
    full, and the docstring told every reader it could not.
    """

    base_url: str
    api_key: Secret | None = None
    source: CredentialSource = "none"
    extra_headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """A provider-agnostic request.

    The transport turns this into a wire body; nothing above the transport
    layer ever constructs provider JSON.
    """

    model: str
    messages: tuple[Message, ...]
    system: SystemPrompt = field(default_factory=SystemPrompt)
    tools: tuple[ToolSchema, ...] = ()
    tool_choice: Literal["auto", "any", "none"] = "auto"
    max_output_tokens: int | None = None
    temperature: float | None = None
    stop: tuple[str, ...] = ()
    reasoning: ReasoningConfig | None = None
    stream: bool = True
    extra: Mapping[str, object] = field(default_factory=dict)
    """Escape hatch for a provider parameter we have not modelled yet."""


@dataclass(frozen=True, slots=True)
class SanitizeRules:
    """What one provider will and will not accept in a message history.

    The sanitizer is driven by these rather than by provider names, so adding a
    provider does not mean editing the sanitizer.
    """

    require_alternating_roles: bool = False
    """Consecutive same-role messages must be merged or separated."""
    require_tool_result_pairing: bool = True
    """Every tool call needs a matching result before the next turn."""
    tool_results_in_user_message: bool = False
    """Anthropic puts tool results in a user turn; chat-completions uses `tool`."""
    drop_unsigned_thinking: bool = True
    allow_empty_assistant: bool = False
    max_image_bytes: int | None = None


class ProviderTransport(ABC):
    """Base class for every provider integration."""

    api_mode: ClassVar[str]
    features: ClassVar[frozenset[TransportFeature]]

    def __init__(self, *, credentials: Credentials, timeout_s: float = 600.0) -> None:
        """Store credentials and the per-request timeout."""
        self._credentials = credentials
        self._timeout_s = timeout_s

    @property
    def credentials(self) -> Credentials:
        """The credentials this transport was constructed with."""
        return self._credentials

    # -- conversion: pure, no I/O, therefore fully unit-testable ------------

    @abstractmethod
    def build_request(self, request: CompletionRequest) -> WireRequest:
        """Convert a :class:`CompletionRequest` into a provider wire body."""

    @abstractmethod
    def normalize_response(self, raw: Mapping[str, Any]) -> ModelResponse:
        """Convert a provider reply into a :class:`ModelResponse`."""

    @abstractmethod
    def parse_stream(self, chunks: Iterator[Mapping[str, Any]]) -> Iterator[StreamEvent]:
        """Convert decoded stream frames into normalized events."""

    # -- I/O ----------------------------------------------------------------

    @abstractmethod
    def send(
        self, request: CompletionRequest, *, cancel: CancelToken = NEVER_CANCELLED
    ) -> ModelResponse:
        """Perform one non-streaming completion."""

    @abstractmethod
    def stream(
        self, request: CompletionRequest, *, cancel: CancelToken = NEVER_CANCELLED
    ) -> Iterator[StreamEvent]:
        """Perform one streaming completion, yielding normalized events."""

    def count_tokens(self, request: CompletionRequest) -> int | None:  # noqa: ARG002
        """Ask the provider to count tokens, when it offers an endpoint.

        ``None`` means "no such endpoint"; the caller falls back to estimation.
        """
        return None

    # -- hooks with useful defaults -----------------------------------------

    def sanitize_rules(self) -> SanitizeRules:
        """Describe what this provider accepts. Override where it differs."""
        return SanitizeRules()

    def validate_response(self, response: ModelResponse) -> None:
        """Reject a structurally impossible reply.

        Tolerant of empty content: a refusal or a bare stop legitimately
        carries no blocks, and treating that as malformed turns a normal
        outcome into a retry loop.
        """
        if response.finish_reason == "tool_calls" and not response.tool_uses():
            msg = "provider reported tool_calls but returned no tool call"
            raise MalformedResponse(msg)

    def classify_error(self, exc: Exception) -> ProviderError:
        """Map a provider-specific failure onto our taxonomy.

        Provider-specific on purpose: "context too long" arrives as a 400 with
        a differently worded body from every vendor, and only the transport
        knows which wording is which.
        """
        if isinstance(exc, ProviderError):
            return exc
        return ProviderError(str(exc))

    def close(self) -> None:  # noqa: B027  -- an optional hook, not part of the contract
        """Release any held connection."""

    def __enter__(self) -> Self:
        """Enter a context that closes the transport on exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the transport."""
        self.close()


def supports_all(transport: type[ProviderTransport], required: Sequence[TransportFeature]) -> bool:
    """Whether ``transport`` advertises every feature in ``required``."""
    return all(feature in transport.features for feature in required)
