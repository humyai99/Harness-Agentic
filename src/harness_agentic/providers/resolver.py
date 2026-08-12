"""Turning a ``provider/model`` string into a live transport.

The single place a provider name maps to a class. Everywhere else works with
:class:`~harness_agentic.providers.base.ProviderTransport` and the capability
flags in the catalog -- which is the rule that stops a provider switch from
sprouting inside the agent loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness_agentic.errors import CredentialError, ProviderError
from harness_agentic.providers.catalog import (
    ModelInfo,
    model_info,
    parse_model_ref,
    provider_info,
)
from harness_agentic.providers.credentials import SecretResolver, resolve_credentials
from harness_agentic.providers.transports.anthropic import AnthropicTransport
from harness_agentic.providers.transports.chat_completions import ChatCompletionsTransport
from harness_agentic.providers.transports.gemini import GeminiTransport

if TYPE_CHECKING:
    from harness_agentic.core.clock import Clock
    from harness_agentic.providers.base import ProviderTransport


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A model reference resolved to something callable."""

    provider: str
    model: str
    transport: ProviderTransport
    info: ModelInfo

    def label(self) -> str:
        """A human-readable ``provider/model``."""
        return f"{self.provider}/{self.model}"


class TransportResolver:
    """Builds and caches transports, one per provider."""

    def __init__(
        self,
        *,
        secrets: SecretResolver | None = None,
        clock: Clock | None = None,
        overrides: dict[str, ProviderTransport] | None = None,
    ) -> None:
        """Build a resolver, optionally with injected transports for tests."""
        self._secrets = secrets or SecretResolver()
        self._clock = clock
        self._cache: dict[str, ProviderTransport] = dict(overrides or {})

    def resolve(self, reference: str) -> ResolvedModel:
        """Resolve one ``provider/model`` reference."""
        provider, model = parse_model_ref(reference)
        return ResolvedModel(
            provider=provider,
            model=model,
            transport=self.transport_for(provider),
            info=model_info(provider, model),
        )

    def chain(self, primary: str, fallbacks: Sequence[str] = ()) -> list[ResolvedModel]:
        """Resolve a primary model and its fallbacks.

        A fallback that cannot be resolved -- usually a missing credential --
        is skipped rather than fatal. Refusing to start because the *backup*
        has no key would be the wrong trade: the primary still works.
        """
        chain = [self.resolve(primary)]
        for reference in fallbacks:
            try:
                chain.append(self.resolve(reference))
            except (CredentialError, ProviderError, NotImplementedError):
                continue
        return chain

    def transport_for(self, provider: str) -> ProviderTransport:
        """Return the transport for one provider, building it on first use."""
        if existing := self._cache.get(provider):
            return existing
        transport = self._build(provider)
        self._cache[provider] = transport
        return transport

    def _build(self, provider: str) -> ProviderTransport:
        info = provider_info(provider)
        if info is None:
            msg = f"unknown provider {provider!r}"
            raise ProviderError(msg)

        credentials = resolve_credentials(provider, self._secrets)
        match info.api_mode:
            case "anthropic_messages":
                return AnthropicTransport(credentials=credentials, clock=self._clock)
            case "openai_chat":
                return ChatCompletionsTransport(
                    credentials=credentials,
                    provider=provider,
                    quirks=info.quirks,
                    clock=self._clock,
                )
            case "gemini_generate":
                return GeminiTransport(credentials=credentials, clock=self._clock)
            case _:
                msg = f"no transport implements api_mode {info.api_mode!r}"
                raise ProviderError(msg)

    def close(self) -> None:
        """Close every transport this resolver built."""
        for transport in self._cache.values():
            transport.close()
        self._cache.clear()
