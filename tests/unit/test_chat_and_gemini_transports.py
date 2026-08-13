"""Details the shared contract suite cannot express.

Each of these encodes a specific way one provider differs from the others --
the sort of thing that is obvious once you have hit it in production and
invisible until then.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harness_agentic.core.secrets import Secret
from harness_agentic.core.stream import StreamAccumulator
from harness_agentic.core.types import (
    ImageBlock,
    Message,
    SystemPrompt,
    SystemSegment,
    TextBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
)
from harness_agentic.providers.base import CompletionRequest, Credentials
from harness_agentic.providers.catalog import ChatCompatQuirks
from harness_agentic.providers.transports.chat_completions import ChatCompletionsTransport
from harness_agentic.providers.transports.gemini import GeminiTransport

NOW = datetime(2026, 1, 1, tzinfo=UTC)
CREDS = Credentials(
    base_url="https://example.invalid", api_key=Secret("k", source="test"), source="explicit"
)

SCHEMA = ToolSchema(
    name="read_file",
    description="Read a file.",
    parameters={
        "type": "object",
        "title": "ReadFileParams",
        "properties": {"path": {"type": "string", "default": "a.txt"}},
        "additionalProperties": False,
    },
)


def _msg(role: str, *blocks: object) -> Message:
    return Message(role=role, content=tuple(blocks), created_at=NOW)  # type: ignore[arg-type]


def _req(*messages: Message, **kwargs: object) -> CompletionRequest:
    return CompletionRequest(model="m", messages=messages, **kwargs)  # type: ignore[arg-type]


# -- chat completions ----------------------------------------------------------


@pytest.fixture
def openai() -> ChatCompletionsTransport:
    return ChatCompletionsTransport(credentials=CREDS)


def test_tool_arguments_are_a_json_string_not_an_object(
    openai: ChatCompletionsTransport,
) -> None:
    """The classic chat-completions integration bug, pinned."""
    wire = openai.build_request(
        _req(_msg("assistant", ToolUseBlock(id="c1", name="t", arguments={"a": 1})))
    )
    arguments = wire["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert arguments == '{"a": 1}'


def test_tool_results_become_their_own_messages(openai: ChatCompletionsTransport) -> None:
    """Unlike Anthropic, where they share one user turn."""
    wire = openai.build_request(
        _req(
            _msg("assistant", ToolUseBlock(id="c1", name="t"), ToolUseBlock(id="c2", name="t")),
            _msg(
                "tool",
                ToolResultBlock(tool_use_id="c1", text="one"),
                ToolResultBlock(tool_use_id="c2", text="two"),
            ),
        )
    )
    roles = [m["role"] for m in wire["messages"]]
    assert roles == ["assistant", "tool", "tool"]
    assert wire["messages"][1]["tool_call_id"] == "c1"


def test_the_system_prompt_becomes_the_first_message(
    openai: ChatCompletionsTransport,
) -> None:
    wire = openai.build_request(
        _req(
            _msg("user", TextBlock("hi")),
            system=SystemPrompt((SystemSegment("be helpful"),)),
        )
    )
    assert wire["messages"][0] == {"role": "system", "content": "be helpful"}


def test_images_switch_the_user_turn_to_the_parts_form(
    openai: ChatCompletionsTransport,
) -> None:
    wire = openai.build_request(
        _req(_msg("user", TextBlock("what is this"), ImageBlock("image/png", data_b64="AAA")))
    )
    parts = wire["messages"][0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_text_only_turn_stays_a_plain_string(openai: ChatCompletionsTransport) -> None:
    """The parts form is legal everywhere but several endpoints prefer a string."""
    wire = openai.build_request(_req(_msg("user", TextBlock("hi"))))
    assert wire["messages"][0]["content"] == "hi"


def test_quirks_rename_the_max_tokens_field() -> None:
    ollama = ChatCompletionsTransport(
        credentials=CREDS,
        provider="ollama",
        quirks=ChatCompatQuirks(max_tokens_field="max_tokens"),
    )
    wire = ollama.build_request(_req(_msg("user", TextBlock("hi")), max_output_tokens=100))
    assert wire["max_tokens"] == 100
    assert "max_completion_tokens" not in wire


def test_tool_choice_required_degrades_where_unsupported() -> None:
    """Better a weaker hint than a 400 from an endpoint that never shipped it."""
    ollama = ChatCompletionsTransport(
        credentials=CREDS,
        provider="ollama",
        quirks=ChatCompatQuirks(supports_tool_choice_required=False),
    )
    wire = ollama.build_request(
        _req(_msg("user", TextBlock("hi")), tools=(SCHEMA,), tool_choice="any")
    )
    assert wire["tool_choice"] == "auto"


def test_stream_usage_is_requested_when_supported(openai: ChatCompletionsTransport) -> None:
    """Without it most endpoints omit usage entirely and the loop budgets blind."""
    wire = openai.build_request(_req(_msg("user", TextBlock("hi")), stream=True))
    assert wire["stream_options"] == {"include_usage": True}


def test_stream_usage_is_omitted_where_unsupported() -> None:
    ollama = ChatCompletionsTransport(
        credentials=CREDS, provider="ollama", quirks=ChatCompatQuirks(stream_usage_option=False)
    )
    wire = ollama.build_request(_req(_msg("user", TextBlock("hi")), stream=True))
    assert "stream_options" not in wire


def test_streamed_tool_calls_reassemble_from_sparse_deltas(
    openai: ChatCompletionsTransport,
) -> None:
    """The id and name arrive only on the first fragment, keyed by index."""
    frames = [
        {
            "model": "gpt-5",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "function": {"name": "read_file", "arguments": '{"pa'},
                            }
                        ]
                    }
                }
            ],
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th":"a"}'}}]}}
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ]
    accumulator = StreamAccumulator(provider="openai", model="m", now=NOW)
    for event in openai.parse_stream(iter(frames)):
        accumulator.feed(event)
    response = accumulator.finalize()

    (call,) = response.tool_uses()
    assert call.id == "call_abc"
    assert call.name == "read_file"
    assert call.arguments == {"path": "a"}
    assert response.finish_reason == "tool_calls"
    assert response.usage.input_tokens == 10


def test_cached_prompt_tokens_are_read(openai: ChatCompletionsTransport) -> None:
    response = openai.normalize_response(
        {
            "model": "gpt-5",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 800},
            },
        }
    )
    assert response.usage.cache_read_tokens == 800


def test_reasoning_content_is_captured_but_never_replayed(
    openai: ChatCompletionsTransport,
) -> None:
    """Several endpoints expose reasoning; there is nowhere legal to send it back."""
    response = openai.normalize_response(
        {
            "choices": [
                {
                    "message": {"reasoning_content": "thinking out loud", "content": "answer"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    assert "thinking out loud" in str(response.message.content)

    wire = openai.build_request(_req(response.message))
    assert "thinking out loud" not in str(wire)


# -- gemini --------------------------------------------------------------------


@pytest.fixture
def gemini() -> GeminiTransport:
    return GeminiTransport(credentials=CREDS)


def test_the_assistant_role_is_called_model(gemini: GeminiTransport) -> None:
    wire = gemini.build_request(_req(_msg("assistant", TextBlock("hi"))))
    assert wire["contents"][0]["role"] == "model"


def test_the_system_prompt_gets_its_own_field(gemini: GeminiTransport) -> None:
    wire = gemini.build_request(
        _req(_msg("user", TextBlock("hi")), system=SystemPrompt((SystemSegment("be nice"),)))
    )
    assert wire["systemInstruction"]["parts"][0]["text"] == "be nice"
    assert all(p.get("text") != "be nice" for p in wire["contents"][0]["parts"])


def test_a_function_response_is_matched_by_name_not_id(gemini: GeminiTransport) -> None:
    """Gemini pairs on name, so the name has to be recovered from the call."""
    wire = gemini.build_request(
        _req(
            _msg("user", TextBlock("go")),
            _msg("assistant", ToolUseBlock(id="c1", name="read_file", arguments={})),
            _msg("tool", ToolResultBlock(tool_use_id="c1", text="contents")),
        )
    )
    response_part = wire["contents"][-1]["parts"][0]["functionResponse"]
    assert response_part["name"] == "read_file"
    assert response_part["response"]["result"] == "contents"


def test_unsupported_schema_keys_are_stripped(gemini: GeminiTransport) -> None:
    """Gemini's dialect is a subset and rejects extras rather than ignoring them."""
    wire = gemini.build_request(_req(_msg("user", TextBlock("hi")), tools=(SCHEMA,)))
    parameters = wire["tools"][0]["functionDeclarations"][0]["parameters"]
    assert "additionalProperties" not in parameters
    assert "title" not in parameters
    assert "default" not in parameters["properties"]["path"]
    assert parameters["properties"]["path"]["type"] == "string"


def test_a_blocked_prompt_is_a_content_filter_not_a_malformation(
    gemini: GeminiTransport,
) -> None:
    """No candidates plus promptFeedback is a real outcome, not a broken reply."""
    response = gemini.normalize_response(
        {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 5}}
    )
    assert response.finish_reason == "content_filter"
    assert response.usage.input_tokens == 5


def test_function_calls_arrive_whole_in_one_frame(gemini: GeminiTransport) -> None:
    """Gemini streams parts, not character deltas."""
    frames = [
        {
            "modelVersion": "gemini-2.5-pro",
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "read_file", "args": {"path": "a"}}}]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
        }
    ]
    accumulator = StreamAccumulator(provider="gemini", model="m", now=NOW)
    for event in gemini.parse_stream(iter(frames)):
        accumulator.feed(event)
    response = accumulator.finalize()

    (call,) = response.tool_uses()
    assert call.name == "read_file"
    assert call.arguments == {"path": "a"}
    assert response.usage.output_tokens == 3


def test_the_api_key_travels_in_a_header_not_the_url(gemini: GeminiTransport) -> None:
    """A key in a URL ends up in proxy logs and error messages.

    Asserted against the client the transport actually built. The old version
    checked that the credentials object still held the key, which is true
    whichever way the key is sent and so could not fail.
    """
    assert gemini._client.headers["x-goog-api-key"] == "k"
    assert "k" not in str(gemini._client.base_url)


def test_credentials_do_not_print_the_key(gemini: GeminiTransport) -> None:
    """The field is a Secret so that this holds by construction.

    It was documented as a Secret and typed as ``str``, so the dataclass's
    generated repr put the key in full into any log line, exception, or bug
    report that happened to render it -- and the docstring told every reader
    that could not happen.
    """
    rendered = f"{gemini.credentials!r} {gemini.credentials}"
    assert "k" not in rendered.replace("Secret", "").replace("api_key", "")
    assert "***" in rendered
    # Still reachable where it is genuinely needed, and greppable when it is.
    assert gemini.credentials.api_key is not None
    assert gemini.credentials.api_key.reveal() == "k"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("SOMETHING_NEW", "stop"),
    ],
)
def test_gemini_finish_reason_mapping(raw: str, expected: str) -> None:
    assert GeminiTransport.map_finish_reason(raw) == expected
