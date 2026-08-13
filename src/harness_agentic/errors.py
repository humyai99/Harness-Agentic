"""Exception taxonomy.

The shape of this hierarchy is load-bearing: the agent loop decides whether to
retry, fall back to another model, compact the context, or give up purely by
matching on these types. Provider transports are responsible for mapping their
own HTTP and SDK errors onto them -- ``"context too long"`` arrives as a 400
with a different message shape from every vendor, so classification cannot live
in the loop.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for every error the framework raises deliberately."""


# --------------------------------------------------------------- config -----


class ConfigError(HarnessError):
    """Configuration could not be loaded, parsed, or validated."""


class CredentialError(HarnessError):
    """A required credential is missing, malformed, or unsafely stored."""


class InsecureCredentialFile(CredentialError):
    """A secrets file is readable by users other than its owner.

    Raised at startup rather than warned about: a world-readable ``.env`` is a
    live incident, not a lint finding.
    """


# ------------------------------------------------------------- provider -----


class ProviderError(HarnessError):
    """Base class for anything that went wrong talking to a model provider."""


class TransientProviderError(ProviderError):
    """A retryable failure: connection reset, 5xx, malformed stream frame."""


class RateLimited(TransientProviderError):
    """The provider asked us to slow down."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        """Record the provider's own backoff hint when it supplied one."""
        super().__init__(message)
        self.retry_after_s = retry_after_s


class AuthError(ProviderError):
    """Credentials were rejected. Never retried; falls back to another model."""


class ModelUnavailable(ProviderError):
    """The requested model does not exist or is not enabled for this account."""


class ContextOverflow(ProviderError):
    """The request exceeded the model's context window.

    A first-class recoverable error on purpose. Token estimation is
    systematically approximate, so the loop treats an overflow as a signal to
    compact and recalibrate rather than as a violated assumption.
    """


class ContentFiltered(ProviderError):
    """The provider refused to answer on policy grounds."""


class MalformedResponse(ProviderError):
    """The provider returned something we could not normalize."""


class ProviderExhausted(ProviderError):
    """Every model in the fallback chain failed."""


# ---------------------------------------------------------------- agent -----


class AgentError(HarnessError):
    """Base class for failures originating inside the agent loop."""


class ContextExhausted(AgentError):
    """Compaction ran to its per-turn limit and the request still does not fit."""


class CompactionDeferred(AgentError):
    """Another writer holds the session lease.

    Soft by design. The loop backs off and retries; treating a deferral as a
    failure is how a framework ends up discarding a user's conversation.
    """


class Interrupted(AgentError):
    """The turn was cancelled cooperatively."""


# ----------------------------------------------------------------- tool -----


class ToolError(HarnessError):
    """Base class for tool-layer failures."""


class ToolNotFound(ToolError):
    """No tool is registered under that name."""


class ToolArgumentError(ToolError):
    """Arguments failed schema validation.

    Surfaced back to the model as a tool result rather than raised past the
    loop: a bad argument is something the model can correct on the next turn.
    """


class ApprovalDenied(ToolError):
    """A human or a policy declined to run this tool call."""


class PathOutsideWorkspace(ToolError):
    """A path resolved outside the workspace root after symlink resolution."""


# ---------------------------------------------------------------- skill -----


class SkillError(HarnessError):
    """Base class for skill discovery, validation, and lifecycle failures."""


class SkillValidationError(SkillError):
    """A skill's frontmatter or body failed validation."""


class SkillQuotaExceeded(SkillError):
    """The self-improvement loop hit a configured rate limit."""


# -------------------------------------------------------------- gateway -----


class GatewayError(HarnessError):
    """Base class for chat-gateway failures."""


class AdapterError(GatewayError):
    """A platform adapter failed to start, send, or normalize."""


class Unauthorized(GatewayError):
    """The sender is not permitted to talk to this agent."""
