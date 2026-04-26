"""Broker-agnostic timeframe abstraction.

Replaces direct usage of ``alpaca.data.timeframe.TimeFrame``. Adapters
translate these into vendor types.
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
