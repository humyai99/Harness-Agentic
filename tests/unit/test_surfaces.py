"""The web API and the voice session -- two more subscribers to one event stream.

The point of both milestones is that neither reimplements the loop. So the
assertions are about translation and about the surface-specific hazards: an
unauthenticated caller reaching an endpoint that runs tools, and a voice session
that keeps talking over somebody who interrupted it.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from urllib.parse import quote

import pytest

from harness_agentic.api.events import Fanout, Frame, Stream, sink_for, to_frame
from harness_agentic.api.server import Api, ApiError, Approvals, AsgiApp, new_token
from harness_agentic.core.cancel import CancelToken
from harness_agentic.core.events import (
    ApprovalRequested,
    Notice,
    TextChunk,
    ThinkingChunk,
    ToolCallStarted,
    TurnFinished,
    TurnStarted,
    UsageReported,
)
from harness_agentic.core.types import Usage
from harness_agentic.tools.spec import ApprovalRequest, Danger
from harness_agentic.voice.session import (
    BargeIn,
    Heard,
    Segmenter,
    State,
    Utterance,
    VoiceSession,
    speakable,
    voice_surfaces,
)

TOKEN = "test-token-value"

# -- SSE translation ---------------------------------------------------------------


def test_a_frame_encodes_as_one_data_line() -> None:
    # A multi-line `data:` field is legal SSE and every hand-rolled client gets
    # it wrong.
    wire = Frame("text", {"text": "line one\nline two"}).encode()
    assert wire.count("data:") == 1
    assert wire.endswith("\n\n")
    assert "line one\\nline two" in wire


def test_text_and_tool_events_translate() -> None:
    frame = to_frame(TextChunk("hello"))
    assert frame is not None
    assert frame.event == "text"

    started = to_frame(ToolCallStarted("c1", "read_file", Danger.SAFE, "app.py"))
    assert started is not None
    assert json.loads(started.encode().split("data: ")[1])["tool"] == "read_file"


def test_thinking_is_dropped_unless_asked_for() -> None:
    # Long, not the answer, and it buries the answer.
    assert to_frame(ThinkingChunk("pondering")) is None
    assert to_frame(ThinkingChunk("pondering"), include_thinking=True) is not None


def test_usage_reports_tokens_and_never_money() -> None:
    # A wrong cost estimate is worse than none.
    frame = to_frame(
        UsageReported(usage=Usage(input_tokens=10, output_tokens=5), cumulative=Usage(100, 50))
    )
    assert frame is not None
    assert frame.data["input"] == 10
    assert not any("cost" in key or "usd" in key for key in frame.data)


def test_every_event_type_translates_or_is_deliberately_dropped() -> None:
    # The match in to_frame is exhaustive, so a new event type is a type error
    # rather than an event that silently never reaches the browser.
    events = [
        TurnStarted("s1", "fake/scripted"),
        TextChunk("x"),
        ToolCallStarted("c", "t", Danger.SAFE, "s"),
        Notice("info", "hello"),
        TurnFinished("completed", iterations=1, usage=Usage()),
    ]
    assert all(to_frame(event) is not None for event in events)


def test_a_stream_drops_the_oldest_and_says_it_did() -> None:
    # A backgrounded tab must not hold the agent's memory hostage, and a
    # subscriber that missed frames needs to know it did.
    stream = Stream(maxlen=3)
    for index in range(6):
        stream.push(Frame("text", {"text": str(index)}))

    drained = "".join(stream.drain())
    assert '"dropped":3' in drained
    assert '"text":"5"' in drained
    assert '"text":"0"' not in drained


def test_draining_clears_the_buffer() -> None:
    stream = Stream()
    stream.push(Frame("text", {"text": "one"}))
    assert stream.drain()
    assert stream.drain() == []


def test_a_closed_stream_accepts_nothing_more() -> None:
    stream = Stream()
    stream.close()
    stream.push(Frame("text", {"text": "late"}))
    assert stream.drain() == []


def test_the_sink_fills_a_stream() -> None:
    stream = Stream()
    sink = sink_for(stream)
    sink(TextChunk("hello"))
    sink(ThinkingChunk("dropped"))
    assert len(stream.drain()) == 1


# -- fanning one session out to several connections ----------------------------------


def test_every_subscriber_gets_every_frame() -> None:
    """The bug: one shared stream per session, and ``drain()`` empties it.

    Two consumers -- two tabs, or the moment during SSE's own reconnect when the
    old connection and the new are both alive -- each took whatever happened to be
    buffered. The answer arrived split between them, with a gap the reader can see
    and nothing in the log.
    """
    fanout = Fanout()
    first, second = fanout.subscribe(), fanout.subscribe()
    fanout.push(Frame("text", {"text": "the whole answer"}))

    assert "the whole answer" in "".join(first.drain())
    assert "the whole answer" in "".join(second.drain())


def test_frames_pushed_before_anyone_subscribes_reach_the_first_subscriber() -> None:
    # POST /turns returns immediately and the page subscribes afterwards, so the
    # opening of an answer is pushed with nobody attached.
    fanout = Fanout()
    fanout.push(Frame("text", {"text": "opening words"}))

    assert "opening words" in "".join(fanout.subscribe().drain())
    # And only the first subscriber gets the backlog; a later tab joins live.
    assert fanout.subscribe().drain() == []


def test_unsubscribing_detaches_exactly_that_subscriber() -> None:
    # Two streams holding the same frames are value-equal unless Stream compares
    # by identity, and `list.remove` would then detach the wrong one.
    fanout = Fanout()
    first, second = fanout.subscribe(), fanout.subscribe()
    fanout.unsubscribe(first)

    assert fanout.subscriber_count() == 1
    fanout.push(Frame("text", {"text": "still listening"}))
    assert first.closed
    assert "still listening" in "".join(second.drain())


def test_a_disconnected_subscriber_stops_being_filled() -> None:
    # Otherwise every reconnect leaves a buffer behind that nobody reads.
    fanout = Fanout()
    stream = fanout.subscribe()
    fanout.unsubscribe(stream)
    fanout.push(Frame("text", {"text": "gone"}))

    assert stream.drain() == []
    assert fanout.subscriber_count() == 0


# -- the ASGI layer ------------------------------------------------------------------


async def test_a_disconnected_stream_detaches_its_subscriber() -> None:
    """Driven through ASGI, because that is the layer that decides this.

    A closed tab used to be noticed only when the next ``send`` happened to
    raise -- up to a heartbeat away on an idle session -- and until then its
    stream stayed attached and kept being filled.
    """
    surface = api()
    inbound: list[dict[str, object]] = [
        {"type": "http.request", "body": b"", "more_body": False},
        {"type": "http.disconnect"},
    ]

    async def receive() -> dict[str, object]:
        return inbound.pop(0) if inbound else {"type": "http.disconnect"}

    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    # A frame produced before anyone was listening, as a real turn does.
    surface.sink("s1")(TextChunk("hello from the agent"))

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/sessions/s1/events",
        "query_string": f"token={TOKEN}".encode(),
        "headers": [],
    }
    await asyncio.wait_for(AsgiApp(surface)(scope, receive, send), timeout=5)

    body = b"".join(bytes(message.get("body", b"")) for message in sent)  # type: ignore[arg-type]
    headers = dict(sent[0]["headers"])  # type: ignore[call-overload]
    assert headers[b"content-type"] == b"text/event-stream"
    assert headers[b"x-accel-buffering"] == b"no", "nginx buffers SSE into oblivion without it"
    assert b"hello from the agent" in body
    assert b"retry:" in body, "the reconnect interval is sent once per stream"
    assert surface.streams["s1"].subscriber_count() == 0, "the subscriber should be detached"


# -- the API -----------------------------------------------------------------------


def api(**kwargs: object) -> Api:
    return Api(token=TOKEN, run_turn=lambda session, prompt: None, **kwargs)  # type: ignore[arg-type]


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_the_api_refuses_to_exist_without_a_token() -> None:
    # Not optional-with-a-warning: this endpoint runs tools.
    with pytest.raises(ApiError, match="requires a token"):
        Api(token="", run_turn=lambda s, p: None)


def test_an_unauthenticated_request_is_refused_without_explanation() -> None:
    # Telling an unauthenticated caller which part of its credential was wrong
    # is telling an attacker.
    response = api().dispatch("GET", "/api/sessions", {}, b"")
    assert response.status == 401
    assert b"unauthorized" in response.body
    assert b"token" not in response.body.lower()


def test_a_wrong_token_is_refused() -> None:
    response = api().dispatch("GET", "/api/sessions", {"Authorization": "Bearer nope"}, b"")
    assert response.status == 401


def test_the_page_itself_needs_no_token() -> None:
    # It contains no data, and requiring a header to fetch HTML means no browser
    # can load it.
    response = api().dispatch("GET", "/", {}, b"")
    assert response.status == 200
    assert response.content_type == "text/html"
    assert b"EventSource" in response.body


def test_the_event_stream_accepts_a_query_token() -> None:
    # EventSource cannot set headers. It is the one concession, and the token is
    # single-purpose and revocable.
    response = api().dispatch("GET", f"/api/sessions/s1/events?token={TOKEN}", {}, b"")
    assert response.status == 200
    assert response.stream is not None
    assert api().dispatch("GET", "/api/sessions/s1/events?token=wrong", {}, b"").status == 401


def test_starting_a_turn_answers_before_the_turn_finishes() -> None:
    # A turn takes minutes and an HTTP request that waits for one times out in
    # every proxy between here and the browser.
    started: list[tuple[str, str]] = []
    surface = Api(token=TOKEN, run_turn=lambda session, prompt: started.append((session, prompt)))
    response = surface.dispatch(
        "POST", "/api/sessions/s1/turns", auth(), json.dumps({"prompt": "hello"}).encode()
    )

    assert response.status == 202
    assert surface.wait_for_turns(timeout=5)
    assert started == [("s1", "hello")]


def test_a_turn_with_no_prompt_is_refused() -> None:
    response = api().dispatch("POST", "/api/sessions/s1/turns", auth(), b'{"prompt": "  "}')
    assert response.status == 400


def test_a_failing_turn_reports_onto_the_stream_rather_than_vanishing() -> None:
    def explode(session: str, prompt: str) -> None:
        detail = "the model is unreachable"
        raise RuntimeError(detail)

    surface = Api(token=TOKEN, run_turn=explode)
    surface.dispatch("POST", "/api/sessions/s1/turns", auth(), b'{"prompt": "hi"}')
    assert surface.wait_for_turns(timeout=5)

    drained = "".join(surface.streams["s1"].subscribe().drain())
    assert "unreachable" in drained
    assert "notice" in drained


def test_an_unknown_endpoint_is_a_404() -> None:
    assert api().dispatch("GET", "/api/nothing", auth(), b"").status == 404


def test_an_escaped_query_token_still_authorizes() -> None:
    """The bug: the query token was compared without percent-decoding.

    The page sends it through ``encodeURIComponent``, so an operator-supplied
    ``HARNESS_WEB_TOKEN`` holding any base64 secret arrives escaped. The result
    was a bare 401 -- correct for an attacker, undebuggable for the person
    holding the right token.
    """
    secret = "a+b/c=d"  # what a base64 secret looks like
    surface = Api(token=secret, run_turn=lambda session, prompt: None)
    escaped = quote(secret, safe="")

    assert (
        surface.dispatch("GET", f"/api/sessions/s1/events?token={escaped}", {}, b"").status == 200
    )
    # And a wrong token is still refused, escaped or not.
    assert surface.dispatch("GET", "/api/sessions/s1/events?token=a%2Bb", {}, b"").status == 401


def test_waiting_for_turns_bounds_the_total_wait_not_each_thread() -> None:
    """The bug: the timeout was handed to every ``join`` in turn.

    So it was a per-thread budget, and a shutdown asked to wait half a second
    with several turns in flight waited several times that.
    """
    release = threading.Event()

    def blocked(session: str, prompt: str) -> None:
        release.wait(timeout=10)

    surface = Api(token=TOKEN, run_turn=blocked)
    for index in range(4):
        surface.dispatch("POST", f"/api/sessions/s{index}/turns", auth(), b'{"prompt": "hi"}')

    began = time.monotonic()
    finished = surface.wait_for_turns(timeout=0.3)
    elapsed = time.monotonic() - began
    release.set()

    assert not finished, "the turns are still blocked, so this should report a timeout"
    assert elapsed < 1.0, f"waited {elapsed:.2f}s for a 0.3s timeout across 4 turns"
    surface.wait_for_turns(timeout=5)


def test_tokens_are_long_enough_to_be_worth_having() -> None:
    minimum = 32
    assert len(new_token()) >= minimum
    assert new_token() != new_token()


# -- approvals from a browser --------------------------------------------------------


def test_a_web_approval_releases_the_waiting_tool() -> None:
    import threading

    surface = api()
    granted: list[bool] = []
    ask = surface.prompter("s1")

    def tool_thread() -> None:
        granted.append(
            ask(ApprovalRequest(tool="terminal", danger=Danger.DESTRUCTIVE, summary="rm -rf build"))
        )

    thread = threading.Thread(target=tool_thread, daemon=True)
    thread.start()

    listed: list[dict[str, object]] = []
    deadline = time.monotonic() + 5
    while not listed and time.monotonic() < deadline:
        listed = surface.approvals.listed()
        time.sleep(0.005)
    assert listed
    assert listed[0]["summary"] == "rm -rf build"

    response = surface.dispatch(
        "POST", f"/api/approvals/{listed[0]['id']}", auth(), b'{"granted": true, "by": "web"}'
    )
    thread.join(timeout=5)

    assert response.status == 200
    assert granted == [True]
    # And the stream shows both halves, so a second tab sees what happened.
    drained = "".join(surface.streams["s1"].subscribe().drain())
    assert "approval_requested" in drained
    assert "approval_resolved" in drained


def test_answering_an_approval_twice_is_a_conflict() -> None:
    surface = api()
    identifier, _ = surface.approvals.open(
        ApprovalRequest(tool="t", danger=Danger.WRITES, summary="s")
    )
    route = f"/api/approvals/{identifier}"
    assert surface.dispatch("POST", route, auth(), b'{"granted":true}').status == 200
    # The second answer has nothing to answer, and saying so beats pretending.
    assert surface.dispatch("POST", route, auth(), b'{"granted":true}').status == 409


def test_an_approval_that_nobody_answers_is_refused_not_left_hanging() -> None:
    # A turn blocked forever holds a worker thread until the process restarts.
    approvals = Approvals()
    _, waiting = approvals.open(ApprovalRequest(tool="t", danger=Danger.WRITES, summary="s"))
    assert waiting.wait(timeout=0.05) is False


def test_a_web_approval_event_carries_the_detail() -> None:
    surface = api()
    frame = to_frame(ApprovalRequested("a1", "terminal", Danger.DESTRUCTIVE, "rm -rf /"))
    assert frame is not None
    assert frame.data["summary"] == "rm -rf /"
    del surface


# -- voice: segmentation ------------------------------------------------------------


def test_streamed_text_is_spoken_a_sentence_at_a_time() -> None:
    # A synthesizer handed a fragment puts the pause in the wrong place.
    segmenter = Segmenter()
    assert segmenter.feed("The deploy ") == []
    ready = segmenter.feed("finished successfully. Everything looks fine now. ")
    assert ready
    assert ready[0].endswith(".")
    assert "deploy" in ready[0]


def test_a_long_run_with_no_sentence_end_is_still_spoken() -> None:
    # Otherwise the agent is silent for the whole paragraph and then talks for
    # two minutes.
    segmenter = Segmenter(max_chars=120)
    ready = segmenter.feed("word " * 60)
    assert ready
    assert all(len(piece) <= 120 for piece in ready)
    # And it cuts at a space, not mid-word.
    assert not ready[0].endswith("wor")


def test_thai_sentence_endings_are_recognised() -> None:
    segmenter = Segmenter(min_chars=10)
    ready = segmenter.feed("ระบบทำงานเรียบร้อยแล้วครับ。 ตรวจสอบอีกครั้งได้เลย。 ")
    assert ready


def test_flush_returns_whatever_is_left() -> None:
    segmenter = Segmenter()
    segmenter.feed("a short tail")
    assert segmenter.flush() == "a short tail"
    assert segmenter.flush() == ""


# -- voice: what is worth saying ------------------------------------------------------


def test_code_urls_and_tables_are_not_read_aloud() -> None:
    spoken = speakable(
        "Run ```python\nprint(1)\n``` then open https://example.test/a/b for **details**"
    )
    assert "a code block" in spoken
    assert "a link" in spoken
    assert "print" not in spoken
    assert "**" not in spoken


def test_a_table_is_summarized_rather_than_read_cell_by_cell() -> None:
    spoken = speakable("| a | b |\n| 1 | 2 |\n| 3 | 4 |")
    assert "a table of 3 rows" in spoken
    assert "|" not in spoken


def test_a_long_path_becomes_its_filename() -> None:
    assert "app.py" in speakable("check src/harness_agentic/agent/app.py")
    assert "harness_agentic" not in speakable("check src/harness_agentic/agent/app.py")


def test_tools_with_no_spoken_form_are_excluded_from_the_voice_surface() -> None:
    # Reading a file listing aloud is not a degraded experience, it is an
    # unusable one. This is what Tool.surfaces was for.
    offered = voice_surfaces(["read_file", "web_search", "terminal", "sql_query"])
    assert offered == ["web_search", "terminal"]


# -- voice: barge-in -------------------------------------------------------------------


class FakeSpeaker:
    """A speaker that records what it said and whether it was cut off."""

    def __init__(self) -> None:
        self.said: list[Utterance] = []
        self.stops = 0
        self._speaking = False

    def speak(self, utterance: Utterance) -> None:
        self.said.append(utterance)
        self._speaking = True

    def stop(self) -> None:
        self.stops += 1
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    def finish(self) -> None:
        self._speaking = False


class FakeListener:
    """A listener that yields nothing; the tests call on_heard directly."""

    def listen(self) -> object:
        return iter(())

    def stop(self) -> None:
        return None


def voice() -> tuple[VoiceSession, FakeSpeaker, list[str], CancelToken]:
    speaker = FakeSpeaker()
    submitted: list[str] = []
    token = CancelToken()
    session = VoiceSession(
        speaker=speaker,
        listener=FakeListener(),  # type: ignore[arg-type]
        submit=submitted.append,
        cancel_token=lambda: token,
    )
    return session, speaker, submitted, token


def test_interrupting_stops_the_speech_and_cancels_the_turn() -> None:
    # In that order. Cancelling first leaves audio playing for however long the
    # cancellation takes to be noticed.
    session, speaker, submitted, token = voice()
    session.handle(TextChunk("This is a long answer that has started playing already. "))
    assert speaker.said

    interrupted = session.on_heard(Heard("actually stop", duration_ms=600))

    assert interrupted
    assert speaker.stops == 1
    assert token.is_set()
    assert token.reason == "the user interrupted"
    # And what they said becomes the next request rather than being discarded.
    assert submitted == ["actually stop"]


def test_a_cough_does_not_interrupt() -> None:
    # Below the threshold the agent is interrupted by its own audio bleeding
    # through the microphone and never finishes a sentence.
    session, speaker, _, token = voice()
    session.handle(TextChunk("A long answer that is currently being spoken aloud. "))

    assert not session.on_heard(Heard("uh", duration_ms=80))
    assert speaker.stops == 0
    assert not token.is_set()
    assert session.barge_in.ignored == 1


def test_an_interim_result_can_interrupt_without_becoming_a_request() -> None:
    # Interim results exist to trigger barge-in early; treating one as a request
    # would submit half a sentence.
    session, speaker, submitted, _ = voice()
    session.handle(TextChunk("Something long enough to still be playing right now. "))

    assert session.on_heard(Heard("no wait", duration_ms=500, final=False))
    assert speaker.stops == 1
    assert submitted == []


def test_speech_while_the_agent_is_thinking_is_queued_not_a_second_turn() -> None:
    session, speaker, submitted, _ = voice()
    session.on_heard(Heard("first question", duration_ms=900))
    assert submitted == ["first question"]
    assert session.state is State.THINKING

    speaker.finish()
    session.on_heard(Heard("and also this", duration_ms=900))
    assert submitted == ["first question"]
    assert session.queued == ["and also this"]

    session.handle(TurnFinished("completed", iterations=1, usage=Usage()))
    assert submitted == ["first question", "and also this"]


def test_queued_speech_is_coalesced_into_one_request() -> None:
    # Three things said while the agent was busy are one thought.
    session, speaker, submitted, _ = voice()
    session.on_heard(Heard("do the thing", duration_ms=900))
    speaker.finish()
    for text in ("on staging", "not production"):
        session.on_heard(Heard(text, duration_ms=900))

    session.handle(TurnFinished("completed", iterations=1, usage=Usage()))
    assert submitted[-1] == "on staging not production"


def test_tool_calls_are_narrated_briefly() -> None:
    # Named, not described. "Reading a file" is useful; reading a path aloud is
    # unbearable.
    session, speaker, _, _ = voice()
    session.handle(ToolCallStarted("c1", "read_file", Danger.SAFE, "src/very/long/path.py"))

    assert speaker.said[0].text == "reading a file"
    assert "path.py" not in speaker.said[0].text


def test_an_unfinished_turn_says_why() -> None:
    session, speaker, _, _ = voice()
    session.handle(TurnFinished("max_iterations", iterations=40, usage=Usage()))
    assert any("narrow" in utterance.text for utterance in speaker.said)


def test_an_error_is_spoken() -> None:
    session, speaker, _, _ = voice()
    session.handle(Notice("error", "the deploy failed"))
    assert speaker.said[0].kind == "error"


def test_the_barge_in_threshold_is_tunable() -> None:
    # The one number in a voice system worth tuning per deployment.
    strict = BargeIn(min_ms=1000)
    assert not strict.should_interrupt(Heard("stop", duration_ms=500), speaking=True)
    assert strict.should_interrupt(Heard("stop", duration_ms=1500), speaking=True)
    # And nothing interrupts when nothing is playing.
    assert not strict.should_interrupt(Heard("hello", duration_ms=5000), speaking=False)
