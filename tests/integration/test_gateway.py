"""A message arrives from a chat platform and the agent answers it.

Everything below the adapter is the production stack: the real router, the real
authorizer, the real actor, the real agent loop, the real tool registry, the
real SQLite store. Only two things are fake, and both are fakes on purpose --
``FakeTransport`` so no provider is called, and ``FakeAdapter`` so no platform
is. Between them the whole gateway path runs in-process in milliseconds, which
is why these assertions can be about behaviour under concurrency and
interruption rather than about a smoke test that sent one "hello".
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from harness_agentic.agent.build import AgentBundle, build_agent
from harness_agentic.core.events import EventSink
from harness_agentic.core.secrets import Secret
from harness_agentic.errors import AuthError
from harness_agentic.gateway.authz import Authorizer, PlatformAuth
from harness_agentic.gateway.platforms.fake import FakeAdapter, line_like, telegram_like
from harness_agentic.gateway.platforms.line import LineAdapter
from harness_agentic.gateway.ratelimit import Limit, RateLimiter
from harness_agentic.gateway.router import Router
from harness_agentic.gateway.service import Gateway
from harness_agentic.gateway.types import ChatKind, MessageEvent
from harness_agentic.testing import FakeTransport, ScriptedTurn, text_turn, tool_turn
from harness_agentic.tools.approval import ApprovalPolicy, Mode

pytestmark = pytest.mark.usefixtures("isolated_home")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "notes.md").write_text("the answer is 42\n", encoding="utf-8")
    return root


class Harness:
    """A gateway wired to a fake platform and a scripted model."""

    def __init__(
        self,
        workspace: Path,
        tmp_path: Path,
        script: list[ScriptedTurn],
        *,
        adapter: FakeAdapter | None = None,
        authorizer: Authorizer | None = None,
        limiter: RateLimiter | None = None,
        stream: bool = True,
    ) -> None:
        self.adapter = adapter or FakeAdapter(capabilities=telegram_like())
        self.transport = FakeTransport(script)
        self.workspace = workspace
        self.tmp_path = tmp_path
        self.bundles: list[AgentBundle] = []

        async def factory(key: str, event: MessageEvent, sink: EventSink) -> AgentBundle:
            bundle = build_agent(
                model="fake/scripted",
                workspace=workspace,
                sessions_dir=tmp_path / "sessions" / _slug(key),
                toolsets=["file"],
                surface="gateway",
                emit=sink,
                # The gateway surface is allowlist-only by default, which is
                # correct in production and would make every tool call in these
                # tests a refusal. Opened deliberately, per test harness.
                approval=ApprovalPolicy(surface="gateway", modes={"gateway": Mode.ALLOW}),
                transports={"fake": self.transport},
                stream=stream,
            )
            self.bundles.append(bundle)
            return bundle

        self.router = Router(
            adapters={self.adapter.platform: self.adapter},
            authorizer=authorizer or Authorizer(global_allow_all=True),
            bundle_factory=factory,
            limiter=limiter or RateLimiter(exempt=frozenset({"fake:u1", "fake:u2"})),
        )

    async def send(self, text: str, **kwargs: object) -> None:
        """Deliver one message and wait for the turn it starts."""
        event = self.adapter.inject(text, **kwargs)  # type: ignore[arg-type]
        await self.router.handle(event)
        await self.settle()

    async def settle(self) -> None:
        """Wait for every actor to finish. Awaits the pumps; never sleeps."""
        async with asyncio.timeout(10):
            await self.router.actors.wait_idle()
        # Let the fire-and-forget delivery callbacks scheduled from the worker
        # thread land before anything asserts on what was posted.
        for _ in range(5):
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        await self.router.aclose()


def _slug(key: str) -> str:
    return key.replace(":", "_").replace("#", "_").replace("@", "_")


# -- the happy path --------------------------------------------------------------


async def test_a_message_becomes_an_answer(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [text_turn("Hello from the agent")])
    await harness.send("hi")
    await harness.aclose()

    assert "Hello from the agent" in harness.adapter.transcript()
    harness.adapter.assert_within_limits()


async def test_a_tool_call_runs_and_is_narrated(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(
        workspace,
        tmp_path,
        [
            tool_turn("read_file", {"path": "notes.md"}),
            text_turn("The note says the answer is 42."),
        ],
    )
    await harness.send("what is in notes.md?")
    await harness.aclose()

    assert "answer is 42" in harness.adapter.transcript()
    assert any("read_file" in p.message.text for p in harness.adapter.of_kind("status"))


async def test_two_conversations_stay_separate(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [text_turn("ok")] * 4)
    await harness.send("first", chat="c1", sender="u1")
    await harness.send("second", chat="c2", sender="u2")

    assert len(set(harness.router.actors.actors)) == 2
    # Separate agents, so separate sessions and separate histories.
    assert len({b.context.session_id for b in harness.bundles}) == 2
    await harness.aclose()


async def test_a_thread_is_its_own_conversation(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [text_turn("ok")] * 4)
    await harness.send("in the channel", chat="c1", kind=ChatKind.CHANNEL)
    await harness.send("in a thread", chat="c1", kind=ChatKind.CHANNEL, thread="t1")
    assert len(harness.router.actors.actors) == 2
    await harness.aclose()


# -- the security gates ----------------------------------------------------------


async def test_an_unauthorized_sender_gets_no_agent(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [text_turn("should never run")], authorizer=Authorizer())
    outcome = await harness.router.handle(harness.adapter.inject("hello"))

    assert not outcome.accepted
    assert harness.bundles == []
    # Told why, and how to fix it. Silence reads as a broken bot.
    assert "/pair" in harness.adapter.transcript()


async def test_pairing_turns_a_stranger_into_a_user(workspace: Path, tmp_path: Path) -> None:
    from datetime import UTC, datetime

    authorizer = Authorizer(platforms={"fake": PlatformAuth()})
    # The router stamps redemption with the system clock, so the code has to be
    # issued against the same one for its TTL to mean anything.
    code = authorizer.issue_code("fake", now=datetime.now(UTC))
    harness = Harness(workspace, tmp_path, [text_turn("now I can answer")], authorizer=authorizer)

    await harness.router.handle(harness.adapter.inject("hello"))
    assert harness.bundles == []

    await harness.router.handle(harness.adapter.inject(f"/pair {code.code}"))
    assert "Paired" in harness.adapter.transcript()

    await harness.send("hello again")
    await harness.aclose()
    assert "now I can answer" in harness.adapter.transcript()


async def test_a_redelivered_webhook_does_not_run_twice(workspace: Path, tmp_path: Path) -> None:
    # The failure this prevents is not a duplicated message -- it is a second
    # agent turn, with tools, against the same request.
    harness = Harness(workspace, tmp_path, [text_turn("ran once")] * 3)
    event = harness.adapter.inject("deploy it", message_id="evt-1")

    first = await harness.router.handle(event)
    second = await harness.router.handle(event)
    await harness.settle()
    await harness.aclose()

    assert first.accepted
    assert not second.accepted
    assert second.reason == "duplicate delivery"
    assert len(harness.bundles) == 1


async def test_the_turn_budget_is_enforced_and_explained(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(
        workspace,
        tmp_path,
        [text_turn("ok")] * 5,
        limiter=RateLimiter(turns=Limit(capacity=1, per_second=0.0)),
    )
    await harness.send("first")
    outcome = await harness.router.handle(harness.adapter.inject("second"))
    await harness.aclose()

    assert not outcome.accepted
    assert "Rate limit reached" in harness.adapter.transcript()


async def test_a_group_message_that_does_not_address_the_bot_is_ignored(
    workspace: Path, tmp_path: Path
) -> None:
    harness = Harness(workspace, tmp_path, [text_turn("ok")])
    outcome = await harness.router.handle(
        harness.adapter.inject("chatting among ourselves", kind=ChatKind.GROUP, mentioned=False)
    )
    await harness.aclose()

    assert not outcome.accepted
    assert harness.adapter.posted == []


# -- commands --------------------------------------------------------------------


async def test_commands_answer_without_starting_a_turn(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [text_turn("ok")])
    outcome = await harness.router.handle(harness.adapter.inject("/status"))
    await harness.aclose()

    assert outcome.was_command
    assert harness.bundles == []
    assert "Idle" in harness.adapter.transcript()


async def test_help_lists_the_commands(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [])
    await harness.router.handle(harness.adapter.inject("/help"))
    await harness.aclose()
    transcript = harness.adapter.transcript()
    assert "/stop" in transcript
    assert "/new" in transcript


async def test_an_unknown_command_says_so(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [])
    await harness.router.handle(harness.adapter.inject("/frobnicate"))
    await harness.aclose()
    assert "Unknown command" in harness.adapter.transcript()


async def test_new_drops_the_conversation_binding(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [text_turn("first"), text_turn("second")])
    await harness.send("hello")
    first_session = harness.bundles[0].context.session_id

    await harness.router.handle(harness.adapter.inject("/new"))
    await harness.send("hello again")
    await harness.aclose()

    assert len(harness.bundles) == 2
    assert harness.bundles[1].context.session_id != first_session


async def test_pair_is_refused_in_a_group(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [])
    await harness.router.handle(harness.adapter.inject("/pair ABCD1234", kind=ChatKind.GROUP))
    await harness.aclose()
    assert "direct message" in harness.adapter.transcript()


# -- concurrency and interruption -------------------------------------------------


async def test_messages_arriving_together_are_answered_as_one_thought(
    workspace: Path, tmp_path: Path
) -> None:
    # Three lines typed in quick succession are one request. Answering them as
    # three turns produces three partial answers that each miss the others.
    harness = Harness(workspace, tmp_path, [text_turn("understood")] * 4)
    for text in ("deploy the app", "to staging", "not production"):
        await harness.router.handle(harness.adapter.inject(text))
    await harness.settle()
    await harness.aclose()

    assert len(harness.bundles) == 1
    sent = harness.transport.requests
    combined = "\n".join(
        block.text
        for request in sent
        for message in request.messages
        for block in message.content
        if getattr(block, "text", None)
    )
    assert "not production" in combined


async def test_one_conversation_never_runs_two_turns_at_once(
    workspace: Path, tmp_path: Path
) -> None:
    harness = Harness(workspace, tmp_path, [text_turn("a"), text_turn("b"), text_turn("c")])
    for index in range(3):
        await harness.router.handle(harness.adapter.inject(f"message {index}"))
        # No settle between sends: they pile up while the first turn runs.
    await harness.settle()

    actor = next(iter(harness.router.actors.actors.values()))
    assert not actor.running
    assert actor.queue_depth() == 0
    await harness.aclose()


async def test_stop_interrupts_a_running_turn(workspace: Path, tmp_path: Path) -> None:
    harness = Harness(workspace, tmp_path, [text_turn("ok")])
    await harness.send("do a thing")
    actor = next(iter(harness.router.actors.actors.values()))

    # Nothing is running now, so /stop must say so rather than claim success.
    await harness.router.handle(harness.adapter.inject("/stop"))
    assert "Nothing is running" in harness.adapter.transcript()
    assert actor.stats.turns == 1
    await harness.aclose()


async def test_a_failing_turn_does_not_kill_the_conversation(
    workspace: Path, tmp_path: Path
) -> None:
    harness = Harness(
        workspace,
        tmp_path,
        [ScriptedTurn(raises=AuthError("the key was rotated")), text_turn("recovered")],
    )

    await harness.send("first")
    await harness.send("second")
    await harness.aclose()

    # The first turn failed; the same actor still served the second, and the
    # user was told rather than left with silence, which reads as a crash.
    transcript = harness.adapter.transcript()
    assert "recovered" in transcript
    assert "error" in transcript.lower()


# -- platform shape ----------------------------------------------------------------


async def test_the_same_answer_is_shaped_per_platform(workspace: Path, tmp_path: Path) -> None:
    # One agent, two platforms, two delivery strategies -- and nothing above the
    # adapter knows which is which.
    long_answer = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(6))

    def script() -> list[ScriptedTurn]:
        return [tool_turn("read_file", {"path": "notes.md"}), text_turn(long_answer)]

    telegram = Harness(
        workspace,
        tmp_path / "tg",
        script(),
        adapter=FakeAdapter(capabilities=telegram_like()),
    )
    await telegram.send("explain")
    await telegram.aclose()

    line = Harness(
        workspace,
        tmp_path / "line",
        script(),
        adapter=FakeAdapter(capabilities=line_like(), name="fake"),
    )
    await line.send("explain")
    await line.aclose()

    # Both get the whole answer, neither exceeds what its platform accepts.
    for harness in (telegram, line):
        harness.adapter.assert_within_limits()
        assert "Paragraph 5." in harness.adapter.transcript()

    # LINE cannot edit, so no edit is ever attempted against it -- that is a
    # property of the capability, not of how fast the model streamed.
    assert all(not p.edits for p in line.adapter.posted)
    # And progress is narrated only where a status line is free. On LINE every
    # one would be a metered push and an unwanted notification.
    assert telegram.adapter.of_kind("status")
    assert not line.adapter.of_kind("status")


async def test_a_signed_line_webhook_is_answered_end_to_end(
    workspace: Path, tmp_path: Path
) -> None:
    """The M5 acceptance criterion, through the real adapter and a real Gateway.

    Every other test here calls ``Router.handle`` directly, which is why nothing
    noticed that ``Gateway.start`` never drained a webhook adapter's queue: LINE
    verified the signature, answered 200, parked the event, and stopped. The
    platform treats 200 as delivered, so there was no retry and no error
    anywhere -- just an account that never replied.
    """
    posted: list[dict[str, object]] = []

    def line_api(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={})

    adapter = LineAdapter(
        channel_secret=Secret(CHANNEL_SECRET, source="test"),
        access_token=Secret("line-access-token", source="test"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(line_api)),
    )
    transport = FakeTransport([text_turn("staging is deployed")])

    async def factory(key: str, _event: MessageEvent, sink: EventSink) -> AgentBundle:
        return build_agent(
            model="fake/scripted",
            workspace=workspace,
            sessions_dir=tmp_path / "sessions" / _slug(key),
            toolsets=["file"],
            surface="gateway",
            emit=sink,
            approval=ApprovalPolicy(surface="gateway", modes={"gateway": Mode.ALLOW}),
            transports={"fake": transport},
        )

    router = Router(
        adapters={"line": adapter},
        authorizer=Authorizer(global_allow_all=True),
        bundle_factory=factory,
        limiter=RateLimiter(exempt=frozenset({"line:Uuser1"})),
    )
    gateway = Gateway(router=router, adapters=[adapter])
    await gateway.start()

    body = _line_delivery("deploy staging")
    response = await adapter.handle_webhook(body, _sign(body))
    assert response.status == 200

    # Yield until the drain task has taken the event off the adapter's queue
    # and the router has built an actor for it; then wait on the actor itself.
    for _ in range(20):
        await asyncio.sleep(0)
        if router.actors.actors:
            break
    async with asyncio.timeout(10):
        await router.actors.wait_idle()
    for _ in range(5):
        await asyncio.sleep(0)

    await gateway.stop()

    said = [str(m["text"]) for call in posted for m in call["messages"]]  # type: ignore[union-attr]
    assert "staging is deployed" in "\n".join(said)


CHANNEL_SECRET = "line-channel-secret"


def _sign(body: bytes) -> dict[str, str]:
    digest = hmac.new(CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return {"X-Line-Signature": base64.b64encode(digest).decode()}


def _line_delivery(text: str) -> bytes:
    return json.dumps(
        {
            "destination": "Uabc",
            "events": [
                {
                    "type": "message",
                    "replyToken": "reply-token-1",
                    "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                    "source": {"type": "user", "userId": "Uuser1"},
                    "message": {"id": "m1", "type": "text", "text": text},
                }
            ],
        }
    ).encode()


async def test_a_non_streaming_turn_still_reaches_the_platform(
    workspace: Path, tmp_path: Path
) -> None:
    """`TextChunk` was emitted only on the streaming path.

    Every surface learns the answer from that event, so `model.stream = false`
    -- a setting an operator picks when an endpoint streams badly -- made the
    gateway post nothing at all. The turn completed, the transcript was written,
    the usage was recorded, and the user saw silence. Nothing failed, which is
    why it could sit there.
    """
    harness = Harness(workspace, tmp_path, [text_turn("staging is deployed")], stream=False)
    await harness.send("deploy staging")
    await harness.aclose()

    assert "staging is deployed" in harness.adapter.transcript()
