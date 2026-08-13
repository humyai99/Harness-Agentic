"""Running jobs on a schedule, with nobody watching.

The unattended surface is the one where a mistake is most expensive and least
visible, so the defaults are the strictest in the framework:

* **Approval mode is DENY.** A cron job cannot be asked for permission, so it
  gets none. Anything consequential has to be on an allowlist the operator wrote
  down in advance, and switching a job to ``allowlist`` mode is a deliberate act.
* **Every run is time-bounded and iteration-bounded.** A job that loops until the
  context window fills, at 3am, every night, is a bill nobody notices for a
  month.
* **Overlap is refused, not queued.** If the previous run of a job is still
  going, this firing is skipped and counted. Queueing means a job that takes
  longer than its interval accumulates copies of itself until the process dies.
* **A missed window does not backfill.** A process that was down for six hours
  must not wake up and run the hourly report six times.

What a job *produces* is deliberately not a chat message. It goes to the session
store and to the job's own log, and a notifier is an explicit hook -- because a
cron job that messages a chat channel every time it finds nothing is a cron job
someone mutes, and then it is not a monitor any more.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from harness_agentic.core.clock import Clock, SystemClock
from harness_agentic.cron.schedule import Schedule, parse
from harness_agentic.tools.approval import ApprovalPolicy, Mode

if TYPE_CHECKING:
    from harness_agentic.agent.build import AgentBundle
    from harness_agentic.agent.runner import TurnResult
    from harness_agentic.session.store import SessionRecord

log = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 12
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_TOOLSETS: tuple[str, ...] = ("core",)
"""What an unattended job gets unless the operator widens it. Not ``terminal``."""


@dataclass(frozen=True, slots=True)
class Job:
    """One scheduled task."""

    name: str
    schedule: Schedule
    prompt: str
    toolsets: tuple[str, ...] = DEFAULT_TOOLSETS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    timeout_s: float = DEFAULT_TIMEOUT_S
    enabled: bool = True
    approval: Mode = Mode.DENY
    """DENY by default. A job that needs to run something consequential gets
    ``ALLOWLIST`` and a written-down list, never ``PROMPT`` -- there is nobody
    to prompt."""
    allowlist: tuple[str, ...] = ()
    notify: str = ""
    """Where to send the result, if anywhere. Empty means the store only."""

    @classmethod
    def create(cls, name: str, expression: str, prompt: str, **kwargs: object) -> Job:
        """Build a job from a cron expression string."""
        return cls(name=name, schedule=parse(expression), prompt=prompt, **kwargs)  # type: ignore[arg-type]

    def policy(self) -> ApprovalPolicy:
        """The approval policy this job runs under."""
        return ApprovalPolicy(
            surface="cron",
            modes={"cron": self.approval},
            allowlist=self.allowlist or (),
        )

    def describe(self) -> str:
        """One readable line for ``harn cron list``."""
        state = "" if self.enabled else " [disabled]"
        return f"{self.name}: {self.schedule.describe()} -> {self.prompt[:60]!r}{state}"


@dataclass
class Run:
    """What one firing did."""

    job: str
    started_at: datetime
    finished_at: datetime | None = None
    outcome: str = "running"
    """``running``, ``ok``, ``error``, ``timeout``, or ``skipped``."""
    detail: str = ""
    iterations: int = 0
    session_id: str = ""

    @property
    def duration_s(self) -> float:
        """How long it took, so far or in total."""
        end = self.finished_at or self.started_at
        return (end - self.started_at).total_seconds()

    def describe(self) -> str:
        """One readable line."""
        return (
            f"{self.started_at:%Y-%m-%d %H:%M} {self.job}: {self.outcome} "
            f"({self.duration_s:.1f}s, {self.iterations} iteration(s))"
            + (f" -- {self.detail}" if self.detail else "")
        )


BundleFactory = Callable[[Job], "AgentBundle"]
"""Builds the agent for one job. Called per firing, so each run starts clean."""

Notifier = Callable[[Job, Run, str], None]
"""Delivers a finished run somewhere. Explicit, and usually absent."""


@dataclass
class CronRunner:
    """Decides what is due and runs it, one job at a time.

    Synchronous, like the rest of the core. A scheduler that runs jobs
    concurrently needs a concurrency limit, a fairness policy and a story about
    shared resources; running them in sequence needs none of that, and a cron
    tick that takes four minutes because two jobs fired together is fine.
    """

    jobs: list[Job] = field(default_factory=list)
    bundle_factory: BundleFactory | None = None
    clock: Clock = field(default_factory=SystemClock)
    notifier: Notifier | None = None
    history: list[Run] = field(default_factory=list)
    _last_fired: dict[str, datetime] = field(default_factory=dict)
    _running: set[str] = field(default_factory=set)

    def add(self, job: Job) -> None:
        """Register a job, replacing any with the same name."""
        self.jobs = [existing for existing in self.jobs if existing.name != job.name]
        self.jobs.append(job)

    def due(self, now: datetime | None = None) -> list[Job]:
        """Which jobs should fire at this minute.

        Only *this* minute. A process that was down for six hours must not wake
        up and run the hourly report six times, so there is no catch-up: the
        scheduler asks "is this minute a firing minute", never "what did I
        miss".
        """
        moment = (now or self.clock.now()).replace(second=0, microsecond=0)
        return [
            job
            for job in self.jobs
            if job.enabled
            and job.schedule.matches(moment)
            and self._last_fired.get(job.name) != moment
        ]

    def tick(self, now: datetime | None = None) -> list[Run]:
        """Run everything due at this minute. Returns what happened."""
        moment = (now or self.clock.now()).replace(second=0, microsecond=0)
        runs: list[Run] = []
        for job in self.due(moment):
            self._last_fired[job.name] = moment
            runs.append(self.run(job, now=moment))
        return runs

    def run(self, job: Job, *, now: datetime | None = None) -> Run:
        """Run one job once, whatever its schedule says."""
        started = now or self.clock.now()
        record = Run(job=job.name, started_at=started)

        if job.name in self._running:
            # Skipped, not queued. A job slower than its own interval would
            # otherwise accumulate copies of itself until the process dies.
            record.outcome = "skipped"
            record.detail = "the previous run has not finished"
            record.finished_at = started
            self.history.append(record)
            log.warning("cron job %s overlapped and was skipped", job.name)
            return record

        if self.bundle_factory is None:
            record.outcome = "error"
            record.detail = "no bundle factory is configured"
            record.finished_at = started
            self.history.append(record)
            return record

        self._running.add(job.name)
        try:
            record = self._execute(job, record)
        finally:
            self._running.discard(job.name)
        self.history.append(record)
        return record

    def _execute(self, job: Job, record: Run) -> Run:
        """Build an agent, run the prompt, and record the outcome."""
        assert self.bundle_factory is not None  # noqa: S101 - checked by the caller
        try:
            bundle = self.bundle_factory(job)
        except Exception as exc:
            record.outcome = "error"
            record.detail = f"could not build the agent: {exc}"
            record.finished_at = self.clock.now()
            return record

        record.session_id = bundle.context.session_id
        deadline = self.clock.now() + timedelta(seconds=job.timeout_s)
        # Cancellation is cooperative, so the deadline is a token the loop
        # checks rather than something that can interrupt a running tool.
        watchdog = _Watchdog(bundle, deadline, self.clock)

        try:
            result: TurnResult = bundle.runner.run_turn(
                job.prompt, session=_require_session(bundle)
            )
        except Exception as exc:
            record.outcome = "error"
            record.detail = f"{type(exc).__name__}: {exc}"
            record.finished_at = self.clock.now()
            log.exception("cron job %s failed", job.name)
            return record
        finally:
            watchdog.stop()

        record.iterations = result.iterations
        record.finished_at = self.clock.now()
        record.outcome = "timeout" if watchdog.fired else _outcome_for(result.exit_reason)
        record.detail = result.error or ""
        if self.notifier is not None and result.final_text.strip():
            try:
                self.notifier(job, record, result.final_text)
            except Exception:
                # A broken notifier must not turn a successful run into a
                # failure. The result is already in the store.
                log.exception("notifier failed for cron job %s", job.name)
        return record

    def next_runs(
        self, *, limit: int = 10, now: datetime | None = None
    ) -> list[tuple[str, datetime]]:
        """The upcoming firings, soonest first. What ``harn cron list`` prints."""
        moment = now or self.clock.now()
        upcoming: list[tuple[str, datetime]] = []
        for job in self.jobs:
            if not job.enabled:
                continue
            when = job.schedule.next_after(moment)
            if when is not None:
                upcoming.append((job.name, when))
        return sorted(upcoming, key=lambda pair: pair[1])[:limit]

    def report(self, *, limit: int = 20) -> list[str]:
        """Recent history, newest last."""
        return [run.describe() for run in self.history[-limit:]]


def _require_session(bundle: AgentBundle) -> SessionRecord:
    """The bundle's session, or a failure that names what is missing."""
    session = bundle.store.get(bundle.context.session_id)
    if session is None:  # pragma: no cover - created moments earlier
        detail = f"session {bundle.context.session_id} vanished before the job ran"
        raise RuntimeError(detail)
    return session


class _Watchdog:
    """Sets the run's cancel token once its deadline passes.

    A thread rather than a signal: signals only arrive on the main thread, and a
    cron tick may well be running on a worker.
    """

    def __init__(self, bundle: AgentBundle, deadline: datetime, clock: Clock) -> None:
        """Start watching a run."""
        self.fired = False
        self._done = threading.Event()
        seconds = max(0.0, (deadline - clock.now()).total_seconds())

        def watch() -> None:
            if not self._done.wait(seconds):
                self.fired = True
                bundle.runner.cancel.cancel(f"the job exceeded {seconds:.0f}s")
                bundle.context.cancel.cancel("timeout")

        self._thread = threading.Thread(target=watch, name="cron-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        self._done.set()


def _outcome_for(exit_reason: str) -> str:
    """Map a turn's exit reason onto a job outcome."""
    match exit_reason:
        case "completed":
            return "ok"
        case "interrupted":
            return "timeout"
        case _:
            return "error"


def load_jobs(entries: Sequence[dict[str, object]]) -> list[Job]:
    """Build jobs from configuration, refusing anything malformed.

    Refusing rather than skipping: a typo in a cron expression means a job that
    silently never runs, and "silently never runs" is the failure mode a
    scheduler must not have.

    Duplicate names are refused for the same reason. A job's name is its identity
    everywhere it matters -- ``_running`` keys overlap detection by it,
    ``_last_fired`` keys the no-backfill rule by it, and ``harn cron run <name>``
    resolves by it. Two jobs sharing one meant the second silently shadowed the
    first at the command line and the two fought over each other's overlap state.
    """
    jobs: list[Job] = []
    seen: set[str] = set()
    for entry in entries:
        name = str(entry.get("name") or "")
        expression = str(entry.get("schedule") or "")
        prompt = str(entry.get("prompt") or "")
        if not (name and expression and prompt):
            detail = f"a cron job needs name, schedule and prompt; got {entry!r}"
            raise ValueError(detail)
        if name in seen:
            detail = (
                f"two cron jobs are both named {name!r}; a name identifies a job, "
                f"so they have to differ"
            )
            raise ValueError(detail)
        seen.add(name)
        jobs.append(
            Job(
                name=name,
                schedule=parse(expression),
                prompt=prompt,
                toolsets=_strings(entry.get("toolsets")) or DEFAULT_TOOLSETS,
                max_iterations=_int(entry.get("max_iterations"), DEFAULT_MAX_ITERATIONS),
                timeout_s=_float(entry.get("timeout_s"), DEFAULT_TIMEOUT_S),
                enabled=bool(entry.get("enabled", True)),
                approval=Mode(str(entry.get("approval") or Mode.DENY.value)),
                allowlist=_strings(entry.get("allowlist")),
                notify=str(entry.get("notify") or ""),
            )
        )
    return jobs


def _strings(raw: object) -> tuple[str, ...]:
    """A tuple of strings from a config value, ignoring anything else."""
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    return ()


def _int(raw: object, default: int) -> int:
    """An integer from a config value, falling back rather than crashing."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(raw: object, default: float) -> float:
    """A float from a config value, falling back rather than crashing."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return default
    try:
        return float(raw)
    except ValueError:
        return default
