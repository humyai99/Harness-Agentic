"""Voice, and the part of it that is actually hard.

Not speech recognition -- that is a library call. The hard part is **barge-in**:
the user starts talking while the agent is still speaking, and three things have
to happen at once and in the right order.

1. Stop the speech immediately. Not at the end of the sentence, not at the end of
   the current audio chunk -- immediately, because a system that keeps talking
   over someone who interrupted it is a system nobody uses twice.
2. Cancel the turn that is producing the speech, using the same
   :class:`~harness_agentic.core.cancel.CancelToken` the gateway uses. A cancelled
   voice turn and a cancelled chat turn are the same cancellation.
3. Keep what the user just said. The interruption *is* the next request, and
   discarding it means they have to repeat themselves, which is exactly the
   behaviour that makes people give up on voice assistants.

The second thing this module gets right is knowing what *not* to speak. A voice
surface that reads out a file listing is unusable, so ``Tool.surfaces`` excludes
the long-output tools and :func:`speakable` strips what survives: code fences
become "a code block", URLs become "a link", and tables are summarized rather
than read cell by cell.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from harness_agentic.core.events import (
    AgentEvent,
    Notice,
    TextChunk,
    ToolCallStarted,
    TurnFinished,
)

if TYPE_CHECKING:
    from harness_agentic.core.cancel import CancelToken

SENTENCE_END = re.compile("(?<=[.!?\u3002\uff01\uff1f])\\s+|(?<=[:;])\\s+(?=[A-Z])")
"""Sentence boundaries, including the full-width stop, exclamation and
question marks -- Thai and Chinese text uses them, and a synthesizer handed
a run with no boundary speaks for two minutes without pausing."""
MIN_SPEAK_CHARS = 40
"""Below this, wait for more text. Speaking two words at a time makes the
synthesizer stutter and sounds worse than a slightly later start."""
MAX_UTTERANCE_CHARS = 500
BARGE_IN_MIN_MS = 250
"""Speech shorter than this is a cough, a door, or the agent's own audio through
the microphone -- not an interruption. Acting on it makes the agent
uninterruptible in one direction and impossible to finish a sentence to in the
other."""


class State(StrEnum):
    """What the voice session is doing."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True, slots=True)
class Utterance:
    """One thing to say."""

    text: str
    kind: str = "answer"
    """``answer``, ``status``, or ``error``. A surface may use a different voice."""


class Speaker(Protocol):
    """Text to audio, and the ability to stop mid-word."""

    def speak(self, utterance: Utterance) -> None:
        """Begin speaking. Returns as soon as playback starts."""
        ...

    def stop(self) -> None:
        """Stop immediately, discarding anything queued."""
        ...

    @property
    def speaking(self) -> bool:
        """Whether audio is playing right now."""
        ...


class Listener(Protocol):
    """Audio to text, with voice-activity boundaries."""

    def listen(self) -> Iterator[Heard]:
        """Yield transcripts as they are recognised."""
        ...

    def stop(self) -> None:
        """Stop capturing."""
        ...


@dataclass(frozen=True, slots=True)
class Heard:
    """One recognised stretch of speech."""

    text: str
    duration_ms: int
    final: bool = True
    """Interim results drive barge-in; only final ones become requests."""
    confidence: float = 1.0


@dataclass
class BargeIn:
    """Decides whether speech during playback is a real interruption.

    The threshold is not fussiness. Below it, the agent is interrupted by its own
    audio bleeding through the microphone and never finishes a sentence; above it,
    the user has to shout twice. It is the one number in a voice system worth
    tuning per deployment, so it is a field rather than a constant.
    """

    min_ms: int = BARGE_IN_MIN_MS
    min_chars: int = 2
    interruptions: int = 0
    ignored: int = 0

    def should_interrupt(self, heard: Heard, *, speaking: bool) -> bool:
        """Whether this speech should stop the agent."""
        if not speaking:
            return False
        if heard.duration_ms < self.min_ms or len(heard.text.strip()) < self.min_chars:
            self.ignored += 1
            return False
        self.interruptions += 1
        return True


@dataclass
class Segmenter:
    """Accumulates streamed text and emits whole utterances.

    Sentence-aligned, because a synthesizer handed a fragment puts the pause in
    the wrong place and the result sounds like a badly dubbed film. And bounded,
    because a model that produces one 4,000-character paragraph would otherwise
    be silent for its entire duration and then speak for two minutes.
    """

    min_chars: int = MIN_SPEAK_CHARS
    max_chars: int = MAX_UTTERANCE_CHARS
    buffer: str = ""

    def feed(self, text: str) -> list[str]:
        """Add streamed text, returning whatever is ready to speak."""
        self.buffer += text
        ready: list[str] = []
        while True:
            piece = self._take()
            if piece is None:
                return ready
            ready.append(piece)

    def _take(self) -> str | None:
        """One complete utterance from the buffer, if there is one."""
        if len(self.buffer) < self.min_chars:
            return None
        boundaries = [match.end() for match in SENTENCE_END.finditer(self.buffer)]
        cut = next((at for at in boundaries if at >= self.min_chars), 0)
        if not cut and len(self.buffer) >= self.max_chars:
            # No sentence end in a long run. Cut at a space rather than
            # mid-word, which a synthesizer pronounces as two half-words.
            space = self.buffer.rfind(" ", 0, self.max_chars)
            cut = space + 1 if space > self.min_chars else self.max_chars
        if not cut:
            return None
        piece, self.buffer = self.buffer[:cut], self.buffer[cut:]
        return piece.strip() or None

    def flush(self) -> str:
        """Whatever is left, at the end of a turn."""
        remaining, self.buffer = self.buffer.strip(), ""
        return remaining


@dataclass
class VoiceSession:
    """Drives one voice conversation.

    Holds no audio devices: a :class:`Speaker` and a :class:`Listener` are
    injected, which is what lets the interruption logic -- the part that is
    genuinely hard to get right -- be tested without a microphone.
    """

    speaker: Speaker
    listener: Listener
    submit: Callable[[str], None]
    """Starts a turn. The session does not build agents."""
    cancel_token: Callable[[], CancelToken | None] = lambda: None
    """The running turn's token, or ``None`` when nothing is running."""
    barge_in: BargeIn = field(default_factory=BargeIn)
    segmenter: Segmenter = field(default_factory=Segmenter)
    state: State = State.IDLE
    spoken: list[Utterance] = field(default_factory=list)
    queued: list[str] = field(default_factory=list)
    narrate_tools: bool = True

    # -- outbound ---------------------------------------------------------------

    def handle(self, event: AgentEvent) -> None:
        """React to one agent event by speaking, or not."""
        match event:
            case TextChunk(text=text):
                self.state = State.SPEAKING
                for piece in self.segmenter.feed(text):
                    self._say(Utterance(speakable(piece)))
            case ToolCallStarted(tool=tool) if self.narrate_tools:
                # Named, not described. "Reading a file" is useful; the path is
                # not, and reading a path aloud is unbearable.
                self._say(Utterance(_tool_phrase(tool), kind="status"))
            case Notice(level="error", message=message):
                self._say(Utterance(speakable(message), kind="error"))
            case TurnFinished(reason=reason):
                tail = self.segmenter.flush()
                if tail:
                    self._say(Utterance(speakable(tail)))
                if reason != "completed":
                    self._say(Utterance(_ending_phrase(reason), kind="status"))
                self.state = State.LISTENING
                self._drain_queue()
            case _:
                return

    def _say(self, utterance: Utterance) -> None:
        """Speak one utterance, unless it has nothing left after stripping."""
        if not utterance.text.strip():
            return
        self.spoken.append(utterance)
        self.speaker.speak(utterance)

    # -- inbound ----------------------------------------------------------------

    def on_heard(self, heard: Heard) -> bool:
        """Handle recognised speech. Returns whether it interrupted the agent."""
        interrupting = self.barge_in.should_interrupt(heard, speaking=self.speaker.speaking)
        if interrupting:
            self.interrupt()

        if not heard.final:
            # Interim results exist to trigger barge-in early. Treating one as a
            # request would submit half a sentence.
            return interrupting

        text = heard.text.strip()
        if not text:
            return interrupting
        if self.state is State.THINKING and not interrupting:
            # The agent is working and was not interrupted; hold this until it
            # finishes rather than starting a second turn.
            self.queued.append(text)
            return False
        self._start(text)
        return interrupting

    def interrupt(self) -> None:
        """Stop speaking and cancel the running turn.

        In that order. Cancelling first leaves audio playing for however long the
        cancellation takes to be noticed, which is the exact experience this
        method exists to prevent.
        """
        self.speaker.stop()
        self.segmenter.buffer = ""
        token = self.cancel_token()
        if token is not None:
            token.cancel("the user interrupted")
        self.state = State.LISTENING

    def _start(self, text: str) -> None:
        """Begin a turn on recognised speech."""
        self.state = State.THINKING
        self.segmenter.buffer = ""
        self.submit(text)

    def _drain_queue(self) -> None:
        """Start the next queued request, if any.

        Coalesced into one, for the same reason the chat gateway does it: three
        things said while the agent was busy are one thought, and answering them
        separately produces three partial answers.
        """
        if not self.queued:
            return
        pending, self.queued = " ".join(self.queued), []
        self._start(pending)

    def sink(self) -> Callable[[AgentEvent], None]:
        """This session as an :class:`~harness_agentic.core.events.EventSink`."""
        return self.handle


# -- what is worth saying out loud -------------------------------------------------

_FENCE = re.compile(r"```[\s\S]*?```|`[^`]+`")
_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_#>|]+")
_PATH = re.compile(r"(?:[\w.-]+/){2,}[\w.-]+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*]\s+", re.MULTILINE)


def speakable(text: str) -> str:
    """Reduce text to something worth hearing.

    Every substitution here is a thing that is fine to read and awful to listen
    to. A synthesizer reading a URL character by character, or a markdown table
    cell by cell, is the reason voice assistants get switched off.
    """
    out = _FENCE.sub(" (a code block) ", text)
    out = _URL.sub(" (a link) ", out)
    if _TABLE_ROW.search(out):
        rows = len(_TABLE_ROW.findall(out))
        out = _TABLE_ROW.sub("", out) + f" (a table of {rows} rows) "
    out = _PATH.sub(lambda m: m.group(0).rsplit("/", 1)[-1], out)
    out = _BULLET.sub(" ", out)
    out = _MD.sub("", out)
    return re.sub(r"\s+", " ", out).strip()


def _tool_phrase(tool: str) -> str:
    """A short spoken phrase for a tool call."""
    spoken = {
        "read_file": "reading a file",
        "write_file": "writing a file",
        "grep_files": "searching the code",
        "glob_files": "looking for files",
        "terminal": "running a command",
        "web_fetch": "reading a page",
        "web_search": "searching the web",
        "sql_query": "querying the database",
        "kb_search": "checking the knowledge base",
        "delegate": "handing this to a helper",
    }
    return spoken.get(tool, f"using {tool.replace('_', ' ')}")


def _ending_phrase(reason: str) -> str:
    """What to say when a turn did not finish normally."""
    match reason:
        case "interrupted":
            return "Stopped."
        case "max_iterations":
            return "I stopped after too many steps. Could you narrow that down?"
        case "content_filter":
            return "I can't answer that one."
        case _:
            return "Something went wrong there."


VOICE_UNFRIENDLY: frozenset[str] = frozenset(
    {"read_file", "glob_files", "grep_files", "sql_query", "browser_snapshot", "session_search"}
)
"""Tools whose output has no spoken form.

Excluded from the ``voice`` surface rather than spoken badly: reading a file
listing aloud is not a degraded experience, it is an unusable one. This is what
``Tool.surfaces`` was for, and why it existed from the first milestone."""


def voice_surfaces(names: Sequence[str]) -> list[str]:
    """Filter a tool list down to what makes sense spoken."""
    return [name for name in names if name not in VOICE_UNFRIENDLY]
