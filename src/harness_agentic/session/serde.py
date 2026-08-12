"""Turning messages into JSON and back.

Round-tripping has to be exact. A thinking signature that survives the wire but
not the disk turns ``harn --continue`` into a request the provider rejects, and
the failure surfaces one turn later with nothing pointing at serialization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from harness_agentic.core.types import (
    ContentBlock,
    ImageBlock,
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


def block_to_json(block: ContentBlock) -> dict[str, Any]:
    """Serialize one content block."""
    match block:
        case TextBlock(text=text):
            return {"kind": "text", "text": text}
        case ThinkingBlock(text=text, signature=signature, redacted=redacted):
            return {
                "kind": "thinking",
                "text": text,
                "signature": signature,
                "redacted": redacted,
            }
        case ImageBlock(media_type=media_type, data_b64=data, url=url):
            return {"kind": "image", "media_type": media_type, "data_b64": data, "url": url}
        case ToolUseBlock(id=call_id, name=name, arguments=arguments, raw_arguments=raw):
            return {
                "kind": "tool_use",
                "id": call_id,
                "name": name,
                "arguments": dict(arguments),
                "raw_arguments": raw,
            }
        case ToolResultBlock(
            tool_use_id=call_id, text=text, is_error=is_error, truncated=truncated
        ):
            return {
                "kind": "tool_result",
                "tool_use_id": call_id,
                "text": text,
                "is_error": is_error,
                "truncated": truncated,
            }


def block_from_json(raw: Mapping[str, Any]) -> ContentBlock | None:
    """Deserialize one content block, or ``None`` if the kind is unknown.

    Unknown kinds are skipped rather than raising: a transcript written by a
    newer version must not make an older one refuse to resume.
    """
    match raw.get("kind"):
        case "text":
            return TextBlock(str(raw.get("text", "")))
        case "thinking":
            return ThinkingBlock(
                str(raw.get("text", "")),
                signature=raw.get("signature"),
                redacted=bool(raw.get("redacted", False)),
            )
        case "image":
            return ImageBlock(
                media_type=str(raw.get("media_type", "image/png")),
                data_b64=raw.get("data_b64"),
                url=raw.get("url"),
            )
        case "tool_use":
            return ToolUseBlock(
                id=str(raw.get("id", "")),
                name=str(raw.get("name", "")),
                arguments=raw.get("arguments") or {},
                raw_arguments=raw.get("raw_arguments"),
            )
        case "tool_result":
            return ToolResultBlock(
                tool_use_id=str(raw.get("tool_use_id", "")),
                text=str(raw.get("text", "")),
                is_error=bool(raw.get("is_error", False)),
                truncated=bool(raw.get("truncated", False)),
            )
        case _:
            return None


def message_to_json(message: Message, *, usage: Usage | None = None) -> dict[str, Any]:
    """Serialize a message and its optional usage record."""
    payload: dict[str, Any] = {
        "role": message.role,
        "created_at": message.created_at.isoformat(),
        "content": [block_to_json(b) for b in message.content],
    }
    if message.name:
        payload["name"] = message.name
    if usage is not None:
        payload["usage"] = usage_to_json(usage)
    return payload


def message_from_json(raw: Mapping[str, Any]) -> Message:
    """Deserialize a message."""
    blocks = [b for b in (block_from_json(x) for x in raw.get("content", [])) if b is not None]
    return Message(
        role=_role(raw.get("role")),
        content=tuple(blocks),
        created_at=_timestamp(raw.get("created_at")),
        name=raw.get("name"),
    )


def usage_to_json(usage: Usage) -> dict[str, int]:
    """Serialize token accounting."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def usage_from_json(raw: Mapping[str, Any] | None) -> Usage:
    """Deserialize token accounting."""
    if not raw:
        return Usage()
    return Usage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cache_read_tokens=int(raw.get("cache_read_tokens", 0)),
        cache_write_tokens=int(raw.get("cache_write_tokens", 0)),
        reasoning_tokens=int(raw.get("reasoning_tokens", 0)),
    )


def _role(raw: object) -> Role:
    """Coerce a stored role, defaulting to ``user`` for anything unrecognised."""
    value = str(raw)
    if value in ("system", "user", "assistant", "tool"):
        return value  # type: ignore[return-value]
    return "user"


def _timestamp(raw: object) -> datetime:
    """Parse a stored timestamp, always returning an aware datetime."""
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return datetime.now(UTC)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)


def messages_to_json(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Serialize a history."""
    return [message_to_json(m) for m in messages]


def messages_from_json(raw: Sequence[Mapping[str, Any]]) -> list[Message]:
    """Deserialize a history."""
    return [message_from_json(item) for item in raw]
