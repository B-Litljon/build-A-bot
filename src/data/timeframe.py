"""Broker-agnostic timeframe abstraction.

Replaces direct usage of ``alpaca.data.timeframe.TimeFrame``. Adapters
translate these into vendor types.

Glossary:
    TimeFrameUnit -- the granularity unit: MINUTE, HOUR, DAY, WEEK, MONTH.
    TimeFrame -- an immutable (amount, unit) pair, e.g. 5 MINUTE. Frozen and
        validated on construction (amount must be positive), so it is safe to
        share and to use as a dict key.
    MIN_1 / MIN_5 / HOUR_1 / DAY_1 -- prebuilt constants for the common cases.
        MIN_1 and MIN_5 are the ones the scalper actually uses: it trades off
        1-minute bars while reading 5-minute bars for context.
"""

from dataclasses import dataclass
from enum import Enum


class TimeFrameUnit(str, Enum):
    """Granularity unit for a timeframe."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass(frozen=True)
class TimeFrame:
    """Immutable timeframe definition: *amount* of a *unit*."""

    amount: int
    unit: TimeFrameUnit

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError(f"TimeFrame amount must be positive, got {self.amount}")

    def __str__(self) -> str:
        return f"{self.amount}{self.unit.value.title()}"


# Convenience constants -------------------------------------------------------

MIN_1 = TimeFrame(1, TimeFrameUnit.MINUTE)
MIN_5 = TimeFrame(5, TimeFrameUnit.MINUTE)
HOUR_1 = TimeFrame(1, TimeFrameUnit.HOUR)
DAY_1 = TimeFrame(1, TimeFrameUnit.DAY)
