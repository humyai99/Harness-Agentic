"""Running several platforms in one process, and surviving one of them failing.

The property under test is isolation: a revoked Telegram token must not stop
the LINE account from answering. Everything else in the gateway is correctness;
this is availability, and it is the part an operator notices.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from harness_agentic.errors import AdapterError
from harness_agentic.gateway.adapter import PlatformAdapter, QueueAdapter
from harness_agentic.gateway.platforms.fake import FakeAdapter
from harness_agentic.gateway.service import Gateway
from harness_agentic.gateway.types import (
    Capabilities,
    ChatKind,
    DeliveryTarget,
    MessageEvent,
    OutboundMessage,
    Route,
    Sender,
    SentRef,
    Transport,
    WebhookResponse,
)


class Recorder:
    """Stands in for the router, recording what reached it."""

    def __init__(self) -> None:
        self.events: list[MessageEvent] = []
        self.closed = False
        self.explode = False

    async def handle(self, event: MessageEvent) -> object:
        if self.explode:
            detail = "router blew up"
            raise RuntimeError(detail)
        self.events.append(event)
        return None

    async def aclose(self) -> None:
        self.closed = True

    def status_lines(self) -> tuple[str, ...]:
        return (f"{len(self.events)} event(s) routed",)


class BrokenAdapter(PlatformAdapter):
    """An adapter that cannot start."""

    platform = "broken"
    transport = Transport.POLL
    capabilities = Capabilities()

    async def connect(self) -> None:
        detail = "the token was revoked"
        raise AdapterError(detail)

    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        raise NotImplementedError


class FlakyAdapter(PlatformAdapter):
    """An adapter whose poll loop dies once, then works."""

    platform = "flaky"
    transport = Transport.POLL
    capabilities = Capabilities()

    def __init__(self) -> None:
        self.attempts = 0

    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        raise NotImplementedError

    def poll(self) -> AsyncIterator[MessageEvent]:
        return self._poll()

    async def _poll(self) -> AsyncIterator[MessageEvent]:
        self.attempts += 1
        if self.attempts == 1:
            detail = "connection reset"
            raise ConnectionError(detail)
        while True:  # pragma: no cover - parked until cancelled
            await asyncio.sleep(3600)
            yield  # type: ignore[misc]


class WebhookOnly(PlatformAdapter):
    """A webhook adapter that claims a fixed path."""

    platform = "hooky"
    transport = Transport.WEBHOOK
    capabilities = Capabilities()

    def __init__(self, path: str = "/webhooks/x") -> None:
        self._path = path

    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        raise NotImplementedError

    def webhook_routes(self) -> tuple[Route, ...]:
        async def handler(_body: bytes, _headers: object) -> WebhookResponse:
            return WebhookResponse(status=200)

        return (Route(path=self._path, handler=handler),)  # type: ignore[arg-type]


class QueueingWebhook(QueueAdapter):
    """A webhook adapter shaped like the real ones: queue, then answer 200."""

    platform = "queueing"
    transport = Transport.WEBHOOK
    capabilities = Capabilities()

    async def send(self, target: DeliveryTarget, message: OutboundMessage) -> SentRef:
        raise NotImplementedError

    async def deliver(self, text: str) -> WebhookResponse:
        """What the HTTP handler does once a signature has checked out."""
        self.offer(
            MessageEvent(
                platform=self.platform,
                chat_id="U1",
                chat_kind=ChatKind.PRIVATE,
                sender=Sender(id="U1"),
                text=text,
                received_at=datetime.now(UTC),
                message_id="m1",
            )
        )
        return WebhookResponse(status=200)


async def test_one_platform_failing_to_start_does_not_stop_the_others() -> None:
    working = FakeAdapter()
    gateway = Gateway(router=Recorder(), adapters=[BrokenAdapter(), working])  # type: ignore[arg-type]

    await gateway.start()

    assert gateway.statuses["broken"].state == "failed"
    assert "revoked" in gateway.statuses["broken"].last_error
    assert gateway.statuses["fake"].state == "running"
    assert working.connected
    await gateway.stop()


async def test_events_reach_the_router() -> None:
    adapter = FakeAdapter()
    recorder = Recorder()
    gateway = Gateway(router=recorder, adapters=[adapter])  # type: ignore[arg-type]

    await gateway.start()
    adapter.inject("hello there")
    for _ in range(20):
        await asyncio.sleep(0)
        if recorder.events:
            break
    await gateway.stop()

    assert [e.text for e in recorder.events] == ["hello there"]


async def test_a_router_exception_does_not_end_the_poll_loop() -> None:
    # One malformed message from one user must not stop the platform for
    # everyone else on it.
    adapter = FakeAdapter()
    recorder = Recorder()
    gateway = Gateway(router=recorder, adapters=[adapter])  # type: ignore[arg-type]

    await gateway.start()
    recorder.explode = True
    adapter.inject("poison")
    for _ in range(20):
        await asyncio.sleep(0)
    recorder.explode = False
    adapter.inject("fine")
    for _ in range(20):
        await asyncio.sleep(0)
        if recorder.events:
            break

    assert [e.text for e in recorder.events] == ["fine"]
    # Still running, and never restarted: the exception was absorbed at the
    # event, not escalated into a platform-level failure.
    assert gateway.statuses["fake"].state == "running"
    assert gateway.statuses["fake"].restarts == 0
    await gateway.stop()


async def test_a_webhook_adapter_is_listening_not_polling() -> None:
    gateway = Gateway(router=Recorder(), adapters=[WebhookOnly()])  # type: ignore[arg-type]
    await gateway.start()
    assert gateway.statuses["hooky"].state == "listening"
    assert [r.path for r in gateway.routes()] == ["/webhooks/x"]
    await gateway.stop()


async def test_a_signed_webhook_delivery_reaches_the_router() -> None:
    """The whole point of a webhook adapter, and it was never wired up.

    A webhook handler must answer fast, so it parks the event on a queue and
    returns 200. Nothing drained that queue: only poll and socket adapters got
    a task. So LINE and Slack verified every delivery, answered 200, and never
    replied to a single message -- and because 200 means success, the platform
    did not retry and nothing anywhere reported a problem. The gateway said
    "listening" the entire time.
    """
    adapter = QueueingWebhook()
    router = Recorder()
    gateway = Gateway(router=router, adapters=[adapter])  # type: ignore[arg-type]
    await gateway.start()

    assert (await adapter.deliver("deploy staging")).status == 200
    await _settle()

    assert [e.text for e in router.events] == ["deploy staging"]
    # Still reported as a listener, not a poller -- the label was never wrong.
    assert gateway.statuses["queueing"].state == "listening"
    await gateway.stop()


async def _settle() -> None:
    """Let the drain task pick up whatever was just queued."""
    for _ in range(10):
        await asyncio.sleep(0)


def test_two_adapters_claiming_one_path_is_refused() -> None:
    # Letting the second win means one platform's webhooks reach the other's
    # handler, fail signature verification, and look exactly like an attack.
    gateway = Gateway(
        router=Recorder(),  # type: ignore[arg-type]
        adapters=[WebhookOnly("/hook"), WebhookOnly("/hook")],
    )
    with pytest.raises(AdapterError, match="same webhook path"):
        gateway.routes()


async def test_stopping_closes_everything_once() -> None:
    adapter = FakeAdapter()
    recorder = Recorder()
    gateway = Gateway(router=recorder, adapters=[adapter])  # type: ignore[arg-type]

    await gateway.start()
    await gateway.stop()

    assert recorder.closed
    assert not adapter.connected
    assert gateway.statuses["fake"].state == "stopped"


async def test_run_forever_returns_when_asked_to_stop() -> None:
    adapter = FakeAdapter()
    gateway = Gateway(router=Recorder(), adapters=[adapter])  # type: ignore[arg-type]

    task = asyncio.create_task(gateway.run_forever())
    for _ in range(10):
        await asyncio.sleep(0)
    gateway.request_stop()
    async with asyncio.timeout(5):
        await task

    assert not adapter.connected


async def test_the_status_report_names_every_platform() -> None:
    gateway = Gateway(
        router=Recorder(),  # type: ignore[arg-type]
        adapters=[FakeAdapter(), BrokenAdapter()],
    )
    await gateway.start()
    report = "\n".join(gateway.status_report())
    await gateway.stop()

    assert "fake" in report
    assert "broken: failed" in report
    assert "event(s) routed" in report


def test_an_adapter_describes_its_own_limits() -> None:
    assert "chunked" in FakeAdapter(capabilities=Capabilities()).describe()
    assert "edit" in FakeAdapter().describe()
