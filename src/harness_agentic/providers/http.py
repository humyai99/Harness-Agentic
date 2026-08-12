"""Shared HTTP and server-sent-events plumbing.

The core talks to providers over ``httpx`` and nothing else -- no vendor SDKs.
Three SDKs would mean three release cadences and three conflicting pydantic and
httpx pins, and since every response is normalized anyway their types are dead
weight. The cost is this module: roughly a hundred lines of SSE framing that
all four transports share.

Bedrock and Vertex will be the exception. SigV4 and ADC are enough work that
their transports will be optional extras that do use an SDK, isolated behind
the same :class:`~harness_agentic.providers.base.ProviderTransport` contract.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from http import HTTPStatus
from typing import Any

import httpx

from harness_agentic.core.cancel import NEVER_CANCELLED, CancelToken
from harness_agentic.errors import (
    AuthError,
    ContextOverflow,
    MalformedResponse,
    ModelUnavailable,
    ProviderError,
    RateLimited,
    TransientProviderError,
)

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
    "input length and `max_tokens` exceed",
    "request too large",
)


def build_client(
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout_s: float,
) -> httpx.Client:
    """Create the HTTP client a transport will hold for its lifetime.

    Connection reuse matters more than it looks: a multi-turn agent makes many
    requests to the same host, and a fresh TLS handshake per turn is a visible
    chunk of latency.
    """
    return httpx.Client(
        base_url=base_url,
        headers=dict(headers),
        timeout=httpx.Timeout(timeout_s, connect=15.0),
        follow_redirects=False,
    )


def post_json(
    client: httpx.Client,
    path: str,
    payload: Mapping[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    """POST a JSON body and return the decoded reply, or raise a typed error."""
    try:
        response = client.post(path, json=dict(payload))
    except httpx.TimeoutException as exc:
        msg = f"{provider}: request timed out"
        raise TransientProviderError(msg) from exc
    except httpx.HTTPError as exc:
        msg = f"{provider}: {exc}"
        raise TransientProviderError(msg) from exc

    if response.status_code >= HTTPStatus.BAD_REQUEST:
        raise classify_status(response, provider=provider)
    try:
        body = response.json()
    except ValueError as exc:
        msg = f"{provider}: response was not JSON"
        raise MalformedResponse(msg) from exc
    if not isinstance(body, dict):
        msg = f"{provider}: expected a JSON object"
        raise MalformedResponse(msg)
    return body


def stream_sse(
    client: httpx.Client,
    path: str,
    payload: Mapping[str, Any],
    *,
    provider: str,
    cancel: CancelToken = NEVER_CANCELLED,
) -> Iterator[dict[str, Any]]:
    """POST and yield each decoded SSE ``data:`` frame.

    Cancellation is checked between frames and closes the response, which is
    what actually stops a provider mid-generation -- there is no way to ask it
    to stop, only to hang up.
    """
    try:
        with client.stream("POST", path, json=dict(payload)) as response:
            if response.status_code >= HTTPStatus.BAD_REQUEST:
                response.read()
                raise classify_status(response, provider=provider)
            for line in response.iter_lines():
                if cancel.is_set():
                    return
                frame = _decode_sse_line(line)
                if frame is _DONE:
                    return
                if frame is not None:
                    yield frame
    except httpx.TimeoutException as exc:
        msg = f"{provider}: stream timed out"
        raise TransientProviderError(msg) from exc
    except httpx.HTTPError as exc:
        msg = f"{provider}: {exc}"
        raise TransientProviderError(msg) from exc


_DONE: dict[str, Any] = {}
"""Sentinel for the ``[DONE]`` terminator some providers send."""


def _decode_sse_line(line: str) -> dict[str, Any] | None:
    """Decode one SSE line, or ``None`` when it carries no payload.

    Comment lines, blank keep-alives, and ``event:`` lines are all skipped: the
    event name is redundant because every provider repeats it inside the JSON,
    and relying on it would mean buffering across lines for no gain.
    """
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data:
        return None
    if data == "[DONE]":
        return _DONE
    try:
        parsed = json.loads(data)
    except ValueError:
        # A malformed frame is not worth killing a turn over; the accumulator
        # will notice if the stream ends without a finish reason.
        return None
    return parsed if isinstance(parsed, dict) else None


def classify_status(response: httpx.Response, *, provider: str) -> ProviderError:  # noqa: PLR0911
    """Map an HTTP failure onto the error taxonomy the loop dispatches on.

    Context overflow is picked out by message text rather than status, because
    every vendor reports it as a 400 with differently worded prose. Getting
    this wrong is expensive in a specific way: an overflow misread as a generic
    400 is retried identically five times and fails five times, when the
    correct response is to compact and continue.
    """
    status = response.status_code
    detail = _error_detail(response)
    label = f"{provider}: {status} {detail}" if detail else f"{provider}: HTTP {status}"

    if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
        return AuthError(label)
    if status == HTTPStatus.NOT_FOUND:
        return ModelUnavailable(label)
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        return RateLimited(label, retry_after_s=_retry_after(response))
    if status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
        return ContextOverflow(label)
    if status == HTTPStatus.BAD_REQUEST and any(
        marker in detail.lower() for marker in _OVERFLOW_MARKERS
    ):
        return ContextOverflow(label)
    if status in _RETRYABLE_STATUS:
        return TransientProviderError(label)
    return ProviderError(label)


def _error_detail(response: httpx.Response) -> str:
    """Pull a human-readable message out of an error body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        if isinstance(error, str):
            return error
        message = body.get("message")
        if isinstance(message, str):
            return message
    return str(body)[:500]


def _retry_after(response: httpx.Response) -> float | None:
    """Read the provider's own backoff hint, when it gave one."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
