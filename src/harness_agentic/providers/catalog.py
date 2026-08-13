"""The single table of provider facts.

Context windows, feature support, credential environment variables, default
base URLs, and per-endpoint quirks all live here and nowhere else. Provider
APIs drift constantly; concentrating the facts means drift is a one-file edit
rather than an archaeology exercise across the loop.

The rule this enforces: no module outside this package may branch on a provider
name. Behaviour differences are expressed as :class:`ChatCompatQuirks` fields
and :class:`TransportFeature` flags, which are data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TransportFeature(StrEnum):
    """Capabilities a transport may or may not support."""

    TOOLS = "tools"
    PARALLEL_TOOL_CALLS = "parallel_tool_calls"
    STREAMING = "streaming"
    PROMPT_CACHE = "prompt_cache"
    REASONING = "reasoning"
    IMAGE_INPUT = "image_input"
    TOKEN_COUNT_ENDPOINT = "token_count_endpoint"  # noqa: S105  -- a capability, not a secret
    SYSTEM_ROLE_MESSAGE = "system_role_message"
    """The system prompt goes in the messages array rather than its own field."""


@dataclass(frozen=True, slots=True)
class ChatCompatQuirks:
    """How one OpenAI-compatible endpoint deviates from OpenAI itself.

    One ``ChatCompletionsTransport`` serves OpenAI, OpenRouter, Together, Groq,
    DeepSeek, xAI, Mistral, vLLM, Ollama, LM Studio and SGLang. They differ in
    small, enumerable ways -- so the differences are a value object rather than
    a chain of ``if provider ==`` branches inside the transport.
    """

    supports_tool_choice_required: bool = True
    strips_unknown_fields: bool = False
    """The endpoint 400s on fields it does not recognise instead of ignoring them."""
    max_tokens_field: str = "max_completion_tokens"
    tool_result_role: str = "tool"
    stream_usage_option: bool = True
    """Whether ``stream_options.include_usage`` is understood."""
    double_encoded_tool_args: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    """Routing headers that must NOT be sent to a different vendor's endpoint."""


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Everything needed to reach one provider."""

    id: str
    label: str
    api_mode: str
    base_url: str
    api_key_env: tuple[str, ...] = ()
    base_url_env: tuple[str, ...] = ()
    requires_key: bool = True
    quirks: ChatCompatQuirks = field(default_factory=ChatCompatQuirks)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Per-model facts the loop needs before it can budget a request."""

    provider: str
    model: str
    context_window: int
    max_output_tokens: int
    features: frozenset[TransportFeature]

    def supports(self, feature: TransportFeature) -> bool:
        """Whether this model supports ``feature``."""
        return feature in self.features


_CHAT_BASE: frozenset[TransportFeature] = frozenset(
    {
        TransportFeature.TOOLS,
        TransportFeature.PARALLEL_TOOL_CALLS,
        TransportFeature.STREAMING,
        TransportFeature.SYSTEM_ROLE_MESSAGE,
    }
)

_ANTHROPIC_BASE: frozenset[TransportFeature] = frozenset(
    {
        TransportFeature.TOOLS,
        TransportFeature.PARALLEL_TOOL_CALLS,
        TransportFeature.STREAMING,
        TransportFeature.PROMPT_CACHE,
        TransportFeature.REASONING,
        TransportFeature.IMAGE_INPUT,
        TransportFeature.TOKEN_COUNT_ENDPOINT,
    }
)


PROVIDERS: dict[str, ProviderInfo] = {
    "anthropic": ProviderInfo(
        id="anthropic",
        label="Anthropic",
        api_mode="anthropic_messages",
        base_url="https://api.anthropic.com",
        api_key_env=("ANTHROPIC_API_KEY",),
    ),
    "openai": ProviderInfo(
        id="openai",
        label="OpenAI",
        api_mode="openai_chat",
        base_url="https://api.openai.com/v1",
        api_key_env=("OPENAI_API_KEY",),
    ),
    "openrouter": ProviderInfo(
        id="openrouter",
        label="OpenRouter",
        api_mode="openai_chat",
        base_url="https://openrouter.ai/api/v1",
        api_key_env=("OPENROUTER_API_KEY",),
        quirks=ChatCompatQuirks(
            max_tokens_field="max_tokens",
            extra_headers={"X-Title": "Harness-Agentic"},
        ),
    ),
    "together": ProviderInfo(
        id="together",
        label="Together AI",
        api_mode="openai_chat",
        base_url="https://api.together.xyz/v1",
        api_key_env=("TOGETHER_API_KEY",),
        quirks=ChatCompatQuirks(max_tokens_field="max_tokens"),
    ),
    "groq": ProviderInfo(
        id="groq",
        label="Groq",
        api_mode="openai_chat",
        base_url="https://api.groq.com/openai/v1",
        api_key_env=("GROQ_API_KEY",),
        quirks=ChatCompatQuirks(max_tokens_field="max_tokens"),
    ),
    "ollama": ProviderInfo(
        id="ollama",
        label="Ollama (local)",
        api_mode="openai_chat",
        base_url="http://localhost:11434/v1",
        base_url_env=("OLLAMA_BASE_URL",),
        requires_key=False,
        quirks=ChatCompatQuirks(
            supports_tool_choice_required=False,
            strips_unknown_fields=True,
            max_tokens_field="max_tokens",
            stream_usage_option=False,
        ),
    ),
    "vllm": ProviderInfo(
        id="vllm",
        label="vLLM (self-hosted)",
        api_mode="openai_chat",
        base_url="http://localhost:8000/v1",
        base_url_env=("VLLM_BASE_URL",),
        requires_key=False,
        quirks=ChatCompatQuirks(max_tokens_field="max_tokens"),
    ),
    "gemini": ProviderInfo(
        id="gemini",
        label="Google Gemini",
        api_mode="gemini_generate",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    ),
    "fake": ProviderInfo(
        id="fake",
        label="Scripted (testing)",
        api_mode="fake",
        base_url="",
        requires_key=False,
    ),
}


_MODELS: dict[str, ModelInfo] = {
    info.provider + "/" + info.model: info
    for info in (
        ModelInfo("anthropic", "claude-opus-4-6", 200_000, 32_000, _ANTHROPIC_BASE),
        ModelInfo("anthropic", "claude-sonnet-4-6", 200_000, 64_000, _ANTHROPIC_BASE),
        ModelInfo("anthropic", "claude-haiku-4-5", 200_000, 32_000, _ANTHROPIC_BASE),
        ModelInfo(
            "openai",
            "gpt-5",
            400_000,
            128_000,
            _CHAT_BASE | {TransportFeature.REASONING, TransportFeature.IMAGE_INPUT},
        ),
        ModelInfo("gemini", "gemini-2.5-pro", 1_048_576, 65_536, _ANTHROPIC_BASE),
        ModelInfo("fake", "scripted", 200_000, 8_192, _CHAT_BASE),
    )
}

DEFAULT_CONTEXT_WINDOW = 128_000
"""Assumed window for a model we have no entry for."""

DEFAULT_MAX_OUTPUT = 8_192


def provider_info(provider: str) -> ProviderInfo | None:
    """Look up a provider, or ``None`` if unknown."""
    return PROVIDERS.get(provider)


def model_info(provider: str, model: str) -> ModelInfo:
    """Return facts for one model.

    Unknown models get conservative defaults rather than an error: a
    self-hosted deployment names its models whatever it likes, and refusing to
    run against an unlisted one would make the local-model story unusable.
    """
    known = _MODELS.get(f"{provider}/{model}")
    if known is not None:
        return known
    info = PROVIDERS.get(provider)
    features = _ANTHROPIC_BASE if info and info.api_mode == "anthropic_messages" else _CHAT_BASE
    return ModelInfo(
        provider=provider,
        model=model,
        context_window=DEFAULT_CONTEXT_WINDOW,
        max_output_tokens=DEFAULT_MAX_OUTPUT,
        features=features,
    )


def parse_model_ref(ref: str, *, default_provider: str = "anthropic") -> tuple[str, str]:
    """Split ``"provider/model"`` into its parts.

    Model names may themselves contain slashes -- OpenRouter uses
    ``openrouter/anthropic/claude-sonnet-4-6`` -- so only the first segment is
    treated as the provider, and only when it names a provider we know.
    """
    if "/" not in ref:
        return default_provider, ref
    head, tail = ref.split("/", 1)
    if head in PROVIDERS:
        return head, tail
    return default_provider, ref


def known_models() -> list[str]:
    """Every catalogued ``provider/model`` reference, sorted."""
    return sorted(_MODELS)
