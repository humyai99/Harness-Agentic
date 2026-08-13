"""The OpenAI chat-completions shape.

One transport, eleven endpoints: OpenAI, OpenRouter, Together, Groq, DeepSeek,
xAI, Mistral, vLLM, Ollama, LM Studio, SGLang. They differ in small enumerable
ways, and those differences arrive as a
:class:`~harness_agentic.providers.catalog.ChatCompatQuirks` value rather than
as ``if provider ==`` branches -- which is what keeps this file from turning
into the thing every framework's provider layer eventually turns into.

The lossy direction is downhill and that is deliberate. Converting our
block-based history into chat-completions drops thinking blocks (no place to
put them) and flattens structured tool results into text. Both are acceptable
because the *stored* history keeps everything; only the wire copy is reduced.
Going the other way -- storing chat-completions and reconstructing blocks --
would lose the information permanently.
"""

from __future__ import annotations

import json
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
    ThinkingDelta,
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
from harness_agentic.providers.catalog import ChatCompatQuirks, TransportFeature
from harness_agentic.providers.http import build_client, post_json, stream_sse

COMPLETIONS_PATH = "/chat/completions"

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
}


class ChatCompletionsTransport(ProviderTransport):
    """Talks the OpenAI chat-completions protocol to any endpoint that speaks it."""

    api_mode: ClassVar[str] = "openai_chat"
    features: ClassVar[frozenset[TransportFeature]] = frozenset(
        {
            TransportFeature.TOOLS,
            TransportFeature.PARALLEL_TOOL_CALLS,
            TransportFeature.STREAMING,
            TransportFeature.SYSTEM_ROLE_MESSAGE,
            TransportFeature.IMAGE_INPUT,
        }
    )

    def __init__(
        self,
        *,
        credentials: Credentials,
        provider: str = "openai",
        quirks: ChatCompatQuirks | None = None,
        timeout_s: float = 600.0,
        clock: Clock | None = None,
    ) -> None:
        """Bind a transport to one endpoint and its quirks."""
        super().__init__(credentials=credentials, timeout_s=timeout_s)
        self._provider = provider
        self._quirks = quirks or ChatCompatQuirks()
        self._clock = clock or SystemClock()
        headers = {"content-type": "application/json", **dict(credentials.extra_headers)}
        if credentials.api_key:
            headers["authorization"] = f"Bearer {credentials.api_key.reveal()}"
        self._client = build_client(
            base_url=credentials.base_url, headers=headers, timeout_s=timeout_s
        )

    # -- conversion ---------------------------------------------------------

    def build_request(self, request: CompletionRequest) -> WireRequest:
        """Assemble a chat-completions body."""
        wire: WireRequest = {
            "model": request.model,
            "messages": self._encode_messages(request),
        }
        if request.max_output_tokens:
            wire[self._quirks.max_tokens_field] = request.max_output_tokens
        if request.tools:
            wire["tools"] = self._encode_tools(request.tools)
            wire["tool_choice"] = self._encode_tool_choice(request.tool_choice)
        if request.temperature is not None:
            wire["temperature"] = request.temperature
        if request.stop:
            wire["stop"] = list(request.stop)
        if request.stream:
            wire["stream"] = True
            if self._quirks.stream_usage_option:
                # Without this most endpoints omit usage entirely when
                # streaming, and the loop then budgets against nothing.
                wire["stream_options"] = {"include_usage": True}
        wire.update(request.extra)
        return wire

    def _encode_tool_choice(self, choice: str) -> str:
        """Map our vocabulary onto the endpoint's.

        ``any`` is OpenAI's ``required``, which several compatible endpoints
        never implemented -- so it degrades to ``auto`` rather than 400ing.
        """
        if choice == "any":
            return "required" if self._quirks.supports_tool_choice_required else "auto"
        return choice

    def _encode_tools(self, tools: Sequence[ToolSchema]) -> list[dict[str, Any]]:
        """Render tool definitions in the function-calling shape."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
            for tool in tools
        ]

    def _encode_messages(self, request: CompletionRequest) -> list[dict[str, Any]]:
        """Render the conversation, with the system prompt as the first message."""
        encoded: list[dict[str, Any]] = []
        if system := request.system.rendered():
            encoded.append({"role": "system", "content": system})

        for message in request.messages:
            match message.role:
                case "system":
                    encoded.append({"role": "system", "content": message.text()})
                case "tool":
                    # Each result is its own message here, unlike Anthropic
                    # where they share one user turn.
                    encoded.extend(
                        {
                            "role": self._quirks.tool_result_role,
                            "tool_call_id": block.tool_use_id,
                            "content": block.text,
                        }
                        for block in message.tool_results()
                    )
                case "assistant":
                    encoded.append(self._encode_assistant(message))
                case _:
                    encoded.append({"role": "user", "content": self._encode_user(message)})
        return encoded

    def _encode_assistant(self, message: Message) -> dict[str, Any]:
        """Render an assistant turn, hoisting tool calls out of the content."""
        text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        payload: dict[str, Any] = {"role": "assistant", "content": text or None}
        calls = [
            {
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    # Arguments are a JSON *string* here, not an object. Getting
                    # this wrong is the classic chat-completions integration bug.
                    "arguments": json.dumps(dict(block.arguments)),
                },
            }
            for block in message.tool_uses()
        ]
        if calls:
            payload["tool_calls"] = calls
        return payload

    def _encode_user(self, message: Message) -> str | list[dict[str, Any]]:
        """Render a user turn, using the parts form only when images are present."""
        images = [b for b in message.content if isinstance(b, ImageBlock)]
        if not images:
            return message.text()
        parts: list[dict[str, Any]] = []
        if text := message.text():
            parts.append({"type": "text", "text": text})
        parts.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": image.url or f"data:{image.media_type};base64,{image.data_b64}"
                },
            }
            for image in images
        )
        return parts

    def normalize_response(self, raw: Mapping[str, Any]) -> ModelResponse:
        """Convert a chat-completions reply into a :class:`ModelResponse`."""
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            msg = f"{self._provider}: response had no choices"
            raise MalformedResponse(msg)
        choice = choices[0]
        message = choice.get("message") or {}

        blocks: list[Any] = []
        # Several endpoints expose reasoning under a non-standard key. Read it
        # when present, but never replay it -- there is nowhere legal to put it.
        for key in ("reasoning", "reasoning_content"):
            if reasoning := message.get(key):
                blocks.append(ThinkingBlock(str(reasoning)))
                break
        if content := message.get("content"):
            blocks.append(TextBlock(str(content)))
        blocks.extend(self._decode_tool_calls(message.get("tool_calls")))

        return ModelResponse(
            message=Message(role="assistant", content=tuple(blocks), created_at=self._clock.now()),
            finish_reason=self.map_finish_reason(choice.get("finish_reason")),
            usage=self._usage(raw.get("usage")),
            model=str(raw.get("model", "")),
            provider=self._provider,
        )

    def _decode_tool_calls(self, raw: object) -> list[ToolUseBlock]:
        """Parse the ``tool_calls`` array, tolerating malformed arguments."""
        if not isinstance(raw, list):
            return []
        calls: list[ToolUseBlock] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            function = item.get("function") or {}
            arguments_text = str(function.get("arguments") or "")
            calls.append(
                ToolUseBlock(
                    id=str(item.get("id", "")),
                    name=str(function.get("name", "")),
                    arguments=_loads_object(arguments_text),
                    raw_arguments=arguments_text or None,
                )
            )
        return calls

    @staticmethod
    def map_finish_reason(raw: object) -> FinishReason:
        """Map a ``finish_reason`` onto our vocabulary."""
        return _FINISH_REASONS.get(str(raw), "stop")

    @staticmethod
    def _usage(raw: object) -> Usage:
        """Read token accounting, including the cached-prefix count when given.

        ``prompt_tokens`` is the *whole* prompt here, cached portion included,
        which is the opposite of Anthropic's convention -- there ``input_tokens``
        excludes the cache counters. :class:`~harness_agentic.core.types.Usage`
        tracks cache reads "separately from ordinary input", so the cached part is
        subtracted out. Reporting it both ways meant ``Usage.total`` counted a
        cached prefix twice: a 1000-token prompt with 800 cached and 50 out came
        back as 1850 instead of 1050, and the same conversation costed differently
        depending on which provider answered it.
        """
        if not isinstance(raw, dict):
            return Usage()
        details = raw.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
        prompt = int(raw.get("prompt_tokens", 0) or 0)
        return Usage(
            # Clamped: an endpoint claiming to be OpenAI-compatible while
            # reporting these the other way round should not produce a negative.
            input_tokens=max(0, prompt - cached),
            output_tokens=int(raw.get("completion_tokens", 0) or 0),
            cache_read_tokens=cached,
        )

    def parse_stream(  # noqa: PLR0912  -- one branch per delta shape; splitting hides the mapping
        self, chunks: Iterator[Mapping[str, Any]]
    ) -> Iterator[StreamEvent]:
        """Convert stream frames into normalized events.

        Tool calls arrive as sparse deltas identified by ``index``, and the
        ``id`` and ``name`` only appear on the first fragment -- so the started
        event has to be emitted lazily, the first time an index is seen.
        """
        usage = Usage()
        finish: FinishReason = "stop"
        model = ""
        started: set[int] = set()
        open_indices: list[int] = []

        for frame in chunks:
            if raw_model := frame.get("model"):
                model = str(raw_model)
            if raw_usage := frame.get("usage"):
                usage = self._usage(raw_usage)

            choices = frame.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if reason := choice.get("finish_reason"):
                finish = self.map_finish_reason(reason)

            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue

            for key in ("reasoning", "reasoning_content"):
                if fragment := delta.get(key):
                    yield ThinkingDelta(str(fragment))
                    break
            if content := delta.get("content"):
                yield TextDelta(str(content))

            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                index = int(call.get("index", 0))
                function = call.get("function") or {}
                if index not in started:
                    started.add(index)
                    open_indices.append(index)
                    yield ToolUseStart(
                        index=index,
                        id=str(call.get("id") or f"call_{index + 1}"),
                        name=str(function.get("name") or ""),
                    )
                if fragment := function.get("arguments"):
                    yield ToolUseArgsDelta(index=index, fragment=str(fragment))

        for index in open_indices:
            yield BlockStop(index=index)
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
        wire.pop("stream_options", None)
        raw = post_json(self._client, COMPLETIONS_PATH, wire, provider=self._provider)
        response = self.normalize_response(raw)
        return ModelResponse(
            message=response.message,
            finish_reason=response.finish_reason,
            usage=response.usage,
            model=response.model or request.model,
            provider=self._provider,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def stream(
        self, request: CompletionRequest, *, cancel: CancelToken = NEVER_CANCELLED
    ) -> Iterator[StreamEvent]:
        """Perform one streaming completion."""
        wire = self.build_request(request)
        wire["stream"] = True
        frames = stream_sse(
            self._client, COMPLETIONS_PATH, wire, provider=self._provider, cancel=cancel
        )
        yield from self.parse_stream(frames)

    # -- provider rules -----------------------------------------------------

    def sanitize_rules(self) -> SanitizeRules:
        """Chat-completions is looser than Anthropic in useful ways.

        It has a real ``tool`` role, so results do not have to be folded into a
        user turn, and consecutive same-role messages are accepted. Tool-call
        pairing is still mandatory -- an unanswered call is a 400 everywhere.
        """
        return SanitizeRules(
            require_alternating_roles=False,
            require_tool_result_pairing=True,
            tool_results_in_user_message=False,
            drop_unsigned_thinking=True,
            allow_empty_assistant=True,
        )

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._client.close()


def _loads_object(text: str) -> dict[str, Any]:
    """Decode a JSON object, returning ``{}`` for anything else."""
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
