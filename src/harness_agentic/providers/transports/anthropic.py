"""Anthropic Messages API.

Written before the chat-completions transport because it is the strictest of
the four, and a canonical format that satisfies Anthropic satisfies everyone.
Three of its rules shape the design:

- Tool results go in a **user** turn, not a dedicated ``tool`` role.
- Thinking blocks must be replayed **in their original position** with their
  original signature, or the request is rejected.
- ``cache_control`` marks a prefix, so a breakpoint placed after anything
  volatile silently stops caching working -- no error, just a bill.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar

from harness_agentic.core.cancel import NEVER_CANCELLED, CancelToken
from harness_agentic.core.clock import Clock, SystemClock
from harness_agentic.core.stream import (
    BlockStop,
    StreamAccumulator,
    StreamDone,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ThinkingSignature,
    ToolUseArgsDelta,
    ToolUseStart,
    UsageUpdate,
)
from harness_agentic.core.types import (
    FinishReason,
    ImageBlock,
    Message,
    ModelResponse,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
    Usage,
)
from harness_agentic.errors import MalformedResponse
from harness_agentic.providers.base import (
    CompletionRequest,
    Credentials,
    ProviderTransport,
    SanitizeRules,
    WireRequest,
)
from harness_agentic.providers.catalog import TransportFeature
from harness_agentic.providers.http import build_client, post_json, stream_sse

API_VERSION = "2023-06-01"
MESSAGES_PATH = "/v1/messages"
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"

_STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
    "pause_turn": "stop",
}

_MAX_CACHE_BREAKPOINTS = 4
"""Anthropic's limit. Extra `cache_control` markers are an error, not a no-op."""


class AnthropicTransport(ProviderTransport):
    """Talks to the Anthropic Messages API over plain HTTP."""

    api_mode: ClassVar[str] = "anthropic_messages"
    features: ClassVar[frozenset[TransportFeature]] = frozenset(
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

    def __init__(
        self,
        *,
        credentials: Credentials,
        timeout_s: float = 600.0,
        clock: Clock | None = None,
    ) -> None:
        """Build a transport bound to one set of credentials."""
        super().__init__(credentials=credentials, timeout_s=timeout_s)
        self._clock = clock or SystemClock()
        self._client = build_client(
            base_url=credentials.base_url or "https://api.anthropic.com",
            headers={
                "x-api-key": credentials.api_key or "",
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
                **dict(credentials.extra_headers),
            },
            timeout_s=timeout_s,
        )

    # -- conversion ---------------------------------------------------------

    def build_request(self, request: CompletionRequest) -> WireRequest:
        """Assemble a Messages API body."""
        wire: WireRequest = {
            "model": request.model,
            "messages": self._encode_messages(request.messages),
            "max_tokens": request.max_output_tokens or 4096,
        }
        if system := self._encode_system(request):
            wire["system"] = system
        if request.tools:
            wire["tools"] = self._encode_tools(request.tools)
            wire["tool_choice"] = {
                "auto": {"type": "auto"},
                "any": {"type": "any"},
                "none": {"type": "none"},
            }[request.tool_choice]
        if request.temperature is not None:
            wire["temperature"] = request.temperature
        if request.stop:
            wire["stop_sequences"] = list(request.stop)
        if request.reasoning and request.reasoning.effort != "off":
            wire["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.reasoning.budget_tokens or 4096,
            }
            # Anthropic rejects a temperature alongside extended thinking.
            wire.pop("temperature", None)
        if request.stream:
            wire["stream"] = True
        wire.update(request.extra)
        return wire

    def _encode_system(self, request: CompletionRequest) -> list[dict[str, Any]]:
        """Render the system prompt, honouring cache breakpoints.

        Segments are emitted in order and a breakpoint attaches
        ``cache_control`` to the block it closes, so everything before it is
        the cached prefix.
        """
        blocks: list[dict[str, Any]] = []
        used = 0
        for segment in request.system.segments:
            if not segment.text:
                continue
            block: dict[str, Any] = {"type": "text", "text": segment.text}
            if segment.cache_breakpoint and used < _MAX_CACHE_BREAKPOINTS:
                block["cache_control"] = {"type": "ephemeral"}
                used += 1
            blocks.append(block)
        return blocks

    def _encode_tools(self, tools: Sequence[ToolSchema]) -> list[dict[str, Any]]:
        """Render tool definitions."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.parameters),
            }
            for tool in tools
        ]

    def _encode_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Render the conversation.

        Our ``tool`` role becomes a ``user`` turn carrying ``tool_result``
        blocks, which is the shape Anthropic expects. Consecutive user turns
        that result from this are merged, because the API rejects them.
        """
        encoded: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            role = "user" if message.role in ("user", "tool") else "assistant"
            blocks = [b for b in (self._encode_block(x) for x in message.content) if b]
            if not blocks:
                continue
            if encoded and encoded[-1]["role"] == role:
                encoded[-1]["content"].extend(blocks)
            else:
                encoded.append({"role": role, "content": blocks})
        return encoded

    def _encode_block(self, block: object) -> dict[str, Any] | None:  # noqa: PLR0911
        """Render one content block, or ``None`` if it must not be sent."""
        match block:
            case TextBlock(text=text):
                return {"type": "text", "text": text} if text else None
            case ThinkingBlock(text=text, signature=signature) if signature:
                return {"type": "thinking", "thinking": text, "signature": signature}
            case ThinkingBlock():
                # Unsigned thinking is unreplayable; the sanitizer should have
                # removed it already. Dropping here is the last line of defence.
                return None
            case ToolUseBlock(id=call_id, name=name, arguments=arguments):
                return {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": dict(arguments),
                }
            case ToolResultBlock(tool_use_id=call_id, text=text, is_error=is_error):
                return {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": [{"type": "text", "text": text}],
                    "is_error": is_error,
                }
            case ImageBlock(media_type=media_type, data_b64=data) if data:
                return {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }
            case _:
                return None

    def normalize_response(self, raw: Mapping[str, Any]) -> ModelResponse:
        """Turn a Messages API reply into a :class:`ModelResponse`."""
        content = raw.get("content")
        if not isinstance(content, list):
            msg = "anthropic: response has no content array"
            raise MalformedResponse(msg)

        blocks: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            match item.get("type"):
                case "text":
                    blocks.append(TextBlock(str(item.get("text", ""))))
                case "thinking":
                    blocks.append(
                        ThinkingBlock(
                            str(item.get("thinking", "")),
                            signature=item.get("signature"),
                        )
                    )
                case "redacted_thinking":
                    blocks.append(ThinkingBlock("", signature=item.get("data"), redacted=True))
                case "tool_use":
                    blocks.append(
                        ToolUseBlock(
                            id=str(item.get("id", "")),
                            name=str(item.get("name", "")),
                            arguments=item.get("input") or {},
                        )
                    )

        return ModelResponse(
            message=Message(role="assistant", content=tuple(blocks), created_at=self._clock.now()),
            finish_reason=self.map_stop_reason(raw.get("stop_reason")),
            usage=self._usage(raw.get("usage")),
            model=str(raw.get("model", "")),
            provider="anthropic",
        )

    @staticmethod
    def map_stop_reason(raw: object) -> FinishReason:
        """Map a ``stop_reason`` onto our vocabulary."""
        return _STOP_REASONS.get(str(raw), "stop")

    @staticmethod
    def _usage(raw: object) -> Usage:
        """Read token accounting, including the two cache counters."""
        if not isinstance(raw, dict):
            return Usage()
        return Usage(
            input_tokens=int(raw.get("input_tokens", 0) or 0),
            output_tokens=int(raw.get("output_tokens", 0) or 0),
            cache_read_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(raw.get("cache_creation_input_tokens", 0) or 0),
        )

    def parse_stream(  # noqa: PLR0912  -- one branch per SSE frame type; splitting hides the mapping
        self, chunks: Iterator[Mapping[str, Any]]
    ) -> Iterator[StreamEvent]:
        """Convert Messages API stream frames into normalized events."""
        usage = Usage()
        finish: FinishReason = "stop"
        model = ""

        for frame in chunks:
            match frame.get("type"):
                case "message_start":
                    message = frame.get("message") or {}
                    if isinstance(message, dict):
                        model = str(message.get("model", ""))
                        usage = self._usage(message.get("usage"))
                case "content_block_start":
                    index = int(frame.get("index", 0))
                    block = frame.get("content_block") or {}
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        yield ToolUseStart(
                            index=index,
                            id=str(block.get("id", "")),
                            name=str(block.get("name", "")),
                        )
                case "content_block_delta":
                    index = int(frame.get("index", 0))
                    delta = frame.get("delta") or {}
                    if not isinstance(delta, dict):
                        continue
                    match delta.get("type"):
                        case "text_delta":
                            yield TextDelta(str(delta.get("text", "")))
                        case "thinking_delta":
                            yield ThinkingDelta(str(delta.get("thinking", "")))
                        case "signature_delta":
                            yield ThinkingSignature(
                                index=index, signature=str(delta.get("signature", ""))
                            )
                        case "input_json_delta":
                            yield ToolUseArgsDelta(
                                index=index, fragment=str(delta.get("partial_json", ""))
                            )
                case "content_block_stop":
                    yield BlockStop(index=int(frame.get("index", 0)))
                case "message_delta":
                    delta = frame.get("delta") or {}
                    if isinstance(delta, dict) and "stop_reason" in delta:
                        finish = self.map_stop_reason(delta.get("stop_reason"))
                    # Output tokens only become known here, so they are merged
                    # rather than replacing the counts from message_start.
                    extra = frame.get("usage")
                    if isinstance(extra, dict):
                        usage = Usage(
                            input_tokens=usage.input_tokens,
                            output_tokens=int(extra.get("output_tokens", 0) or 0),
                            cache_read_tokens=usage.cache_read_tokens,
                            cache_write_tokens=usage.cache_write_tokens,
                        )
                case "message_stop":
                    break
                case "error":
                    error = frame.get("error") or {}
                    detail = error.get("message") if isinstance(error, dict) else frame
                    msg = f"anthropic: stream error: {detail}"
                    raise MalformedResponse(msg)

        yield UsageUpdate(usage)
        yield StreamDone(finish_reason=finish, model=model)

    # -- I/O ----------------------------------------------------------------

    def send(
        self, request: CompletionRequest, *, cancel: CancelToken = NEVER_CANCELLED
    ) -> ModelResponse:
        """Perform one non-streaming completion."""
        cancel.raise_if_cancelled()
        started = time.monotonic()
        wire = self.build_request(request)
        wire.pop("stream", None)
        raw = post_json(self._client, MESSAGES_PATH, wire, provider="anthropic")
        response = self.normalize_response(raw)
        latency = int((time.monotonic() - started) * 1000)
        return ModelResponse(
            message=response.message,
            finish_reason=response.finish_reason,
            usage=response.usage,
            model=response.model or request.model,
            provider="anthropic",
            latency_ms=latency,
        )

    def stream(
        self, request: CompletionRequest, *, cancel: CancelToken = NEVER_CANCELLED
    ) -> Iterator[StreamEvent]:
        """Perform one streaming completion."""
        wire = self.build_request(request)
        wire["stream"] = True
        frames = stream_sse(self._client, MESSAGES_PATH, wire, provider="anthropic", cancel=cancel)
        yield from self.parse_stream(frames)

    def count_tokens(self, request: CompletionRequest) -> int | None:
        """Ask Anthropic to count the request exactly.

        Worth the extra call at the point where the loop is deciding whether to
        compact: an estimator that is a few percent low at 200k tokens produces
        an overflow, and an overflow costs a full round trip anyway.
        """
        wire = self.build_request(request)
        for key in ("max_tokens", "stream", "temperature", "stop_sequences"):
            wire.pop(key, None)
        try:
            raw = post_json(self._client, COUNT_TOKENS_PATH, wire, provider="anthropic")
        except Exception:  # estimation must never be the thing that fails a turn
            return None
        value = raw.get("input_tokens")
        return int(value) if isinstance(value, int) else None

    # -- provider rules -----------------------------------------------------

    def sanitize_rules(self) -> SanitizeRules:
        """Anthropic's history requirements."""
        return SanitizeRules(
            require_alternating_roles=True,
            require_tool_result_pairing=True,
            tool_results_in_user_message=True,
            drop_unsigned_thinking=True,
            allow_empty_assistant=False,
        )

    def accumulator(self, model: str) -> StreamAccumulator:
        """Build an accumulator wired to this transport's clock."""
        return StreamAccumulator(provider="anthropic", model=model, now=self._clock.now())

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._client.close()
