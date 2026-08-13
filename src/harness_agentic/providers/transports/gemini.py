"""Google Gemini's generateContent API.

The third shape, and the one that shares least with the other two. Its
vocabulary is different at every level -- ``contents`` not ``messages``,
``parts`` not content blocks, ``model`` not ``assistant``, ``functionCall`` and
``functionResponse`` instead of tool calls and results, and the system prompt
in its own ``systemInstruction`` field.

Which is exactly why it was worth writing early. Two similar transports prove
nothing about an abstraction; a third that shares almost no vocabulary is what
shows whether the canonical format was actually neutral or just
Anthropic-with-extra-steps.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar

from harness_agentic.core.cancel import NEVER_CANCELLED, CancelToken
from harness_agentic.core.clock import Clock, SystemClock
from harness_agentic.core.stream import (
    BlockStop,
    StreamDone,
    StreamEvent,
    TextDelta,
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

_FINISH_REASONS: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "BLOCKLIST": "content_filter",
    "MALFORMED_FUNCTION_CALL": "error",
}

_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"additionalProperties", "$schema", "$defs", "$ref", "default", "examples", "title"}
)
"""Keys Gemini's schema dialect rejects outright rather than ignoring."""


class GeminiTransport(ProviderTransport):
    """Talks to Gemini's generateContent endpoints."""

    api_mode: ClassVar[str] = "gemini_generate"
    features: ClassVar[frozenset[TransportFeature]] = frozenset(
        {
            TransportFeature.TOOLS,
            TransportFeature.PARALLEL_TOOL_CALLS,
            TransportFeature.STREAMING,
            TransportFeature.IMAGE_INPUT,
        }
    )

    def __init__(
        self,
        *,
        credentials: Credentials,
        timeout_s: float = 600.0,
        clock: Clock | None = None,
    ) -> None:
        """Bind a transport to one API key."""
        super().__init__(credentials=credentials, timeout_s=timeout_s)
        self._clock = clock or SystemClock()
        headers = {"content-type": "application/json", **dict(credentials.extra_headers)}
        if credentials.api_key:
            # Header rather than a query parameter: a key in a URL ends up in
            # proxy logs, browser history, and error messages.
            headers["x-goog-api-key"] = credentials.api_key.reveal()
        self._client = build_client(
            base_url=credentials.base_url or "https://generativelanguage.googleapis.com/v1beta",
            headers=headers,
            timeout_s=timeout_s,
        )

    # -- conversion ---------------------------------------------------------

    def build_request(self, request: CompletionRequest) -> WireRequest:
        """Assemble a generateContent body."""
        wire: WireRequest = {"contents": self._encode_contents(request.messages)}
        if system := request.system.rendered():
            wire["systemInstruction"] = {"parts": [{"text": system}]}
        if request.tools:
            wire["tools"] = [{"functionDeclarations": self._encode_tools(request.tools)}]
            wire["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": {"auto": "AUTO", "any": "ANY", "none": "NONE"}[request.tool_choice]
                }
            }
        config: dict[str, Any] = {}
        if request.max_output_tokens:
            config["maxOutputTokens"] = request.max_output_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.stop:
            config["stopSequences"] = list(request.stop)
        if config:
            wire["generationConfig"] = config
        wire.update(request.extra)
        return wire

    def _encode_tools(self, tools: Sequence[ToolSchema]) -> list[dict[str, Any]]:
        """Render function declarations in Gemini's schema dialect."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": _clean_schema(dict(tool.parameters)),
            }
            for tool in tools
        ]

    def _encode_contents(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Render the conversation as ``contents``.

        Gemini calls the assistant ``model`` and has no tool role: results go
        back as ``functionResponse`` parts inside a user turn. Consecutive
        same-role turns are merged, as with Anthropic.

        A ``functionResponse`` is matched to its call by **name**, not by id,
        which nothing else in the stack needs -- so the names are recovered
        from the assistant turns that issued the calls. The sanitizer has
        already dropped any result whose call is missing, so every lookup here
        resolves.
        """
        names = {call.id: call.name for message in messages for call in message.tool_uses()}
        contents: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            parts = [p for p in (self._encode_part(b, names) for b in message.content) if p]
            if not parts:
                continue
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        return contents

    def _encode_part(self, block: object, names: Mapping[str, str]) -> dict[str, Any] | None:
        """Render one block as a ``part``, or ``None`` when it cannot be sent."""
        match block:
            case TextBlock(text=text):
                return {"text": text} if text else None
            case ToolUseBlock(name=name, arguments=arguments):
                return {"functionCall": {"name": name, "args": dict(arguments)}}
            case ToolResultBlock(tool_use_id=call_id, text=text):
                return {
                    "functionResponse": {
                        "name": names.get(call_id, call_id),
                        "response": {"result": text},
                    }
                }
            case ImageBlock(media_type=media_type, data_b64=data) if data:
                return {"inlineData": {"mimeType": media_type, "data": data}}
            case _:
                return None

    def normalize_response(self, raw: Mapping[str, Any]) -> ModelResponse:
        """Convert a generateContent reply into a :class:`ModelResponse`."""
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            # A prompt blocked by safety filters comes back with no candidates
            # at all, which is a legitimate outcome rather than a malformation.
            if feedback := raw.get("promptFeedback"):
                reason = feedback.get("blockReason") if isinstance(feedback, dict) else "blocked"
                return ModelResponse(
                    message=Message(role="assistant", content=(), created_at=self._clock.now()),
                    finish_reason="content_filter",
                    usage=self._usage(raw.get("usageMetadata")),
                    model=str(raw.get("modelVersion", "")),
                    provider="gemini",
                    raw={"blockReason": reason},
                )
            msg = "gemini: response had no candidates"
            raise MalformedResponse(msg)

        candidate = candidates[0]
        content = candidate.get("content") or {}
        blocks: list[Any] = []
        for index, part in enumerate(content.get("parts") or []):
            if not isinstance(part, dict):
                continue
            if text := part.get("text"):
                blocks.append(TextBlock(str(text)))
            if call := part.get("functionCall"):
                blocks.append(
                    ToolUseBlock(
                        # Gemini does not issue call ids, so one is synthesized.
                        # Pairing is positional on their side and by id on ours.
                        id=f"call_{index + 1}",
                        name=str(call.get("name", "")),
                        arguments=call.get("args") or {},
                    )
                )

        return ModelResponse(
            message=Message(role="assistant", content=tuple(blocks), created_at=self._clock.now()),
            finish_reason=self.map_finish_reason(candidate.get("finishReason")),
            usage=self._usage(raw.get("usageMetadata")),
            model=str(raw.get("modelVersion", "")),
            provider="gemini",
        )

    @staticmethod
    def map_finish_reason(raw: object) -> FinishReason:
        """Map a ``finishReason`` onto our vocabulary."""
        return _FINISH_REASONS.get(str(raw), "stop")

    @staticmethod
    def _usage(raw: object) -> Usage:
        """Read token accounting.

        ``promptTokenCount`` counts the whole prompt, including the cached part
        reported in ``cachedContentTokenCount``, so the cached tokens are
        subtracted out to match what
        :class:`~harness_agentic.core.types.Usage` means by ``input_tokens`` --
        the non-cached input. Leaving both in made ``total`` count a cached
        prefix twice.
        """
        if not isinstance(raw, dict):
            return Usage()
        cached = int(raw.get("cachedContentTokenCount", 0) or 0)
        prompt = int(raw.get("promptTokenCount", 0) or 0)
        return Usage(
            input_tokens=max(0, prompt - cached),
            output_tokens=int(raw.get("candidatesTokenCount", 0) or 0),
            cache_read_tokens=cached,
            # Gemini counts thoughts inside candidatesTokenCount, so this is
            # reported for visibility and deliberately not added to `total`.
            reasoning_tokens=int(raw.get("thoughtsTokenCount", 0) or 0),
        )

    def parse_stream(self, chunks: Iterator[Mapping[str, Any]]) -> Iterator[StreamEvent]:
        """Convert streamed candidates into normalized events.

        Gemini streams whole ``parts`` rather than character deltas, so a
        function call arrives complete in one frame -- its start, arguments and
        stop are emitted together.
        """
        usage = Usage()
        finish: FinishReason = "stop"
        model = ""
        call_index = 0

        for frame in chunks:
            if version := frame.get("modelVersion"):
                model = str(version)
            if metadata := frame.get("usageMetadata"):
                usage = self._usage(metadata)

            candidates = frame.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                continue
            candidate = candidates[0]
            if reason := candidate.get("finishReason"):
                finish = self.map_finish_reason(reason)

            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if text := part.get("text"):
                    yield TextDelta(str(text))
                if call := part.get("functionCall"):
                    import json  # noqa: PLC0415  -- only on the tool-call path

                    yield ToolUseStart(
                        index=call_index,
                        id=f"call_{call_index + 1}",
                        name=str(call.get("name", "")),
                    )
                    yield ToolUseArgsDelta(
                        index=call_index, fragment=json.dumps(call.get("args") or {})
                    )
                    yield BlockStop(index=call_index)
                    call_index += 1

        yield UsageUpdate(usage)
        yield StreamDone(finish_reason=finish, model=model)

    # -- I/O ----------------------------------------------------------------

    def send(
        self, request: CompletionRequest, *, cancel: CancelToken = NEVER_CANCELLED
    ) -> ModelResponse:
        """Perform one non-streaming completion."""
        cancel.raise_if_cancelled()
        started = time.monotonic()
        raw = post_json(
            self._client,
            f"/models/{request.model}:generateContent",
            self.build_request(request),
            provider="gemini",
        )
        response = self.normalize_response(raw)
        return ModelResponse(
            message=response.message,
            finish_reason=response.finish_reason,
            usage=response.usage,
            model=response.model or request.model,
            provider="gemini",
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def stream(
        self, request: CompletionRequest, *, cancel: CancelToken = NEVER_CANCELLED
    ) -> Iterator[StreamEvent]:
        """Perform one streaming completion."""
        frames = stream_sse(
            self._client,
            f"/models/{request.model}:streamGenerateContent?alt=sse",
            self.build_request(request),
            provider="gemini",
            cancel=cancel,
        )
        yield from self.parse_stream(frames)

    # -- provider rules -----------------------------------------------------

    def sanitize_rules(self) -> SanitizeRules:
        """Gemini's history requirements, which mirror Anthropic's."""
        return SanitizeRules(
            require_alternating_roles=True,
            require_tool_result_pairing=True,
            tool_results_in_user_message=True,
            drop_unsigned_thinking=True,
            allow_empty_assistant=False,
        )

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._client.close()


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON Schema keywords Gemini rejects.

    Its dialect is a subset, and unknown keys are an error rather than being
    ignored -- so a schema that every other provider accepts will 400 here
    unless it is trimmed first.
    """
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_schema(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned
