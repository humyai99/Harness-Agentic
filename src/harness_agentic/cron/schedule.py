"""When a scheduled job should next run.

A five-field cron expression, parsed with the standard library and nothing else.
The reason not to take a dependency for this is that the hard part is not
parsing -- it is the semantics, and every library disagrees about them in the
same two places:

* **Day-of-month and day-of-week are OR-ed, not AND-ed**, when both are
  restricted. ``0 0 13 * 5`` means "the 13th, and also every Friday", not "Friday
  the 13th". That is what cron does, it surprises everyone, and getting it wrong
  means a job fires eleven times a year instead of sixty.
* **Timezones.** Everything here is computed in UTC. A job scheduled at 02:30
  local time in a zone with daylight saving either runs twice or not at all on
  two days a year, and the failure is silent. So schedules are UTC and the
  operator does the arithmetic once, visibly, instead of the scheduler doing it
  wrong invisibly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

MAX_SEARCH_DAYS = 366 * 4
"""Enough to find 29 February. Beyond it the expression matches nothing."""

FIELD_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))
FIELD_NAMES = ("minute", "hour", "day-of-month", "month", "day-of-week")

ALIASES = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}

MONTHS = {
    name: index
    for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
DAYS = {name: index for index, name in enumerate(("sun", "mon", "tue", "wed", "thu", "fri", "sat"))}

_STEP = re.compile(r"^(?P<range>[^/]+)(?:/(?P<step>\d+))?$")


class BadSchedule(ValueError):
    """A cron expression could not be parsed."""


@dataclass(frozen=True, slots=True)
class Schedule:
    """A parsed cron expression, in UTC."""

    expression: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]
    day_restricted: bool
    weekday_restricted: bool

    def matches(self, moment: datetime) -> bool:
        """Whether this schedule fires at ``moment`` (to the minute)."""
        when = moment.astimezone(UTC)
        if when.minute not in self.minutes or when.hour not in self.hours:
            return False
        if when.month not in self.months:
            return False
        # Python's Monday=0 to cron's Sunday=0.
        weekday = (when.weekday() + 1) % 7
        day_ok = when.day in self.days
        weekday_ok = weekday in self.weekdays

        if self.day_restricted and self.weekday_restricted:
            # OR, not AND. `0 0 13 * 5` is the 13th *and* every Friday.
            return day_ok or weekday_ok
        return day_ok and weekday_ok

    def next_after(self, moment: datetime, *, inclusive: bool = False) -> datetime | None:
        """The next firing strictly after ``moment``, or ``None`` if never.

        Minute-by-minute from the next minute boundary. A closed-form solution
        would be faster and is not worth the bugs: this runs once per job per
        fire, and four years of minutes is a fraction of a second.
        """
        cursor = moment.astimezone(UTC).replace(second=0, microsecond=0)
        if not inclusive:
            cursor += timedelta(minutes=1)
        limit = cursor + timedelta(days=MAX_SEARCH_DAYS)
        while cursor <= limit:
            if self.matches(cursor):
                return cursor
            # Skip a whole day when the date cannot match, rather than walking
            # 1,440 minutes to discover it.
            if not self._date_could_match(cursor):
                cursor = (cursor + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            cursor += timedelta(minutes=1)
        return None

    def _date_could_match(self, when: datetime) -> bool:
        """Whether any minute of this date can fire."""
        if when.month not in self.months:
            return False
        weekday = (when.weekday() + 1) % 7
        if self.day_restricted and self.weekday_restricted:
            return when.day in self.days or weekday in self.weekdays
        return when.day in self.days and weekday in self.weekdays

    def describe(self) -> str:
        """The expression, plus a note when its semantics surprise people."""
        if self.day_restricted and self.weekday_restricted:
            return (
                f"{self.expression} (day-of-month OR day-of-week -- this fires on "
                f"both, which is cron's rule and rarely what is meant)"
            )
        return self.expression


def parse(expression: str) -> Schedule:
    """Parse a five-field cron expression, or an ``@daily``-style alias."""
    text = ALIASES.get(expression.strip().lower(), expression).strip()
    fields = text.split()
    if len(fields) != len(FIELD_RANGES):
        detail = (
            f"a cron expression has {len(FIELD_RANGES)} fields "
            f"({', '.join(FIELD_NAMES)}); got {len(fields)} in {expression!r}"
        )
        raise BadSchedule(detail)

    parsed = [
        _field(raw, low, high, name)
        for raw, (low, high), name in zip(fields, FIELD_RANGES, FIELD_NAMES, strict=True)
    ]
    return Schedule(
        expression=text,
        minutes=parsed[0],
        hours=parsed[1],
        days=parsed[2],
        months=parsed[3],
        weekdays=parsed[4],
        day_restricted=fields[2] not in ("*", "?"),
        weekday_restricted=fields[4] not in ("*", "?"),
    )


def _field(raw: str, low: int, high: int, name: str) -> frozenset[int]:
    """Expand one cron field into the set of values it matches."""
    values: set[int] = set()
    for part in raw.split(","):
        values |= _part(part.strip(), low, high, name)
    if not values:
        detail = f"the {name} field {raw!r} matches nothing"
        raise BadSchedule(detail)
    return frozenset(values)


def _part(part: str, low: int, high: int, name: str) -> set[int]:
    """Expand one comma-separated component."""
    match = _STEP.match(part)
    if match is None:
        detail = f"could not read {part!r} in the {name} field"
        raise BadSchedule(detail)
    body = match.group("range")
    step = int(match.group("step") or 1)
    if step < 1:
        detail = f"a step must be at least 1 in the {name} field"
        raise BadSchedule(detail)

    if body in ("*", "?"):
        return set(range(low, high + 1, step))
    if "-" in body[1:]:
        start_raw, _, end_raw = body.partition("-")
        start = _value(start_raw, low, high, name)
        end = _value(end_raw, low, high, name)
        if start <= end:
            return set(range(start, end + 1, step))
        # A wrapping range: `fri-mon` and `22-2` are both things people write.
        return set(range(start, high + 1, step)) | set(range(low, end + 1, step))

    single = _value(body, low, high, name)
    return set(range(single, high + 1, step)) if step > 1 else {single}


def _value(raw: str, low: int, high: int, name: str) -> int:
    """One numeric or named value, range-checked."""
    text = raw.strip().lower()
    if name == "month" and text in MONTHS:
        return MONTHS[text]
    if name == "day-of-week" and text in DAYS:
        return DAYS[text]
    try:
        number = int(text)
    except ValueError as exc:
        detail = f"{raw!r} is not a valid {name}"
        raise BadSchedule(detail) from exc
    if name == "day-of-week" and number == 7:  # noqa: PLR2004 - both spell Sunday
        return 0
    if not low <= number <= high:
        detail = f"{number} is outside {low}-{high} for {name}"
        raise BadSchedule(detail)
    return number
