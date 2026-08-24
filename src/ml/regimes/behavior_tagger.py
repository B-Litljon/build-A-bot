"""
behavior_tagger.py — deterministic, causal market-behavior labels.

Tags every sealed bar with what the market was *doing* at that bar, along two
independent axes:

    volatility  — is this bar quiet or violent, relative to its own recent past?
    trend       — is price going somewhere, or oscillating in place?

The composite of the two (``trend_high``, ``range_low``, ``mixed_normal``, …)
is the unit the behavior matrix scores candidates over: "which configuration
earns its keep when the market looks like *this*".

WHY THIS IS NOT ``reinforcement_voter.calculate_atr_regimes``
------------------------------------------------------------
That function buckets by ``quantile(0.33)`` / ``quantile(0.67)`` taken over the
**whole** frame, so a bar's label depends on bars that had not happened yet. It
is fine for a post-hoc report and fatal here: a tagger that peeks at the future
makes every downstream per-regime number fiction, and it could never be
reproduced live.

This module instead reuses the estimator the **live** Gate B already runs
(``RiskManager._evaluate_dynamic_gates``): the percentile rank of the current
bar inside a trailing window, with the same window length, the same
finite-filtering, and the same cold-start rule. That keeps the symmetry
contract intact — a tag computed offline over history equals the tag the bot
would compute live at that bar, which ``tests/test_behavior_tagger.py`` pins
directly against the risk manager.

Glossary:
    BehaviorTag -- one bar's verdict: composite label plus the two component
        bands and the raw ranks that produced them. Frozen; ranks are kept so
        a caller can re-bucket without re-scanning history.
    tag_bar -- label ONE bar from trailing windows. The live-shaped entry
        point: hand it the same deques the orchestrator already maintains.
    tag_series -- label a whole history. Defined to be exactly repeated
        tag_bar calls, and tested as such; this equivalence IS the symmetry
        contract.
    trailing_pctile_rank -- fraction of finite values in a trailing window at
        or below its most recent finite value, in [0, 1]. Byte-for-byte the
        computation Gate B does inline.
    DEFAULT_WINDOW -- 260 bars, matching RiskProfile.regime_window. Not
        calendar time: 260 M15 bars is about 2.7 trading days.
    DEFAULT_MIN_SAMPLES -- 60, matching RiskProfile.regime_min_samples. Below
        this the window is COLD and the bar is labelled LABEL_COLD rather than
        guessed at — the live gate bypasses itself here, so no honest tag
        exists.
    LOW_CUT / HIGH_CUT -- 1/3 and 2/3, the rank cut points splitting each axis
        into three bands. Thirds are the analysis convention (they keep cells
        comparably sized); the *estimator* is Gate B's, not this convention's.
    VOL_LOW / VOL_NORMAL / VOL_HIGH -- volatility band names.
    TREND_RANGING / TREND_MIXED / TREND_TRENDING -- trend axis band names.
    LABEL_COLD -- the label for a bar whose window is not yet warm. Excluded
        from the matrix rather than pooled, since it is an absence of evidence.
    _band -- maps a rank to one of three band names given the cut points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Kept in sync with RiskProfile.regime_window / regime_min_samples. The test
# suite asserts they still match rather than trusting these copies.
DEFAULT_WINDOW = 260
DEFAULT_MIN_SAMPLES = 60

LOW_CUT = 1.0 / 3.0
HIGH_CUT = 2.0 / 3.0

VOL_LOW = "low"
VOL_NORMAL = "normal"
VOL_HIGH = "high"

TREND_RANGING = "range"
TREND_MIXED = "mixed"
TREND_TRENDING = "trend"

LABEL_COLD = "cold"


@dataclass(frozen=True)
class BehaviorTag:
    """One bar's behavior verdict. ``label`` is the matrix's cell key."""

    label: str
    vol_band: str
    trend_state: str
    vol_rank: Optional[float]
    trend_rank: Optional[float]
    warm: bool


def trailing_pctile_rank(window: Sequence[float]) -> Optional[float]:
    """
    Percentile rank of the window's latest finite value within that window.

    Mirrors Gate B exactly, including the order of operations: drop non-finite
    values FIRST, then take the last survivor as "current". A trailing NaN
    therefore ranks the last real observation rather than poisoning the bar.

    Returns None when the window holds no finite value at all.
    """
    arr = np.asarray(window, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    current = float(arr[-1])
    return float(np.mean(arr <= current))


def _band(rank: float, low_name: str, mid_name: str, high_name: str,
          low_cut: float, high_cut: float) -> str:
    if rank < low_cut:
        return low_name
    if rank >= high_cut:
        return high_name
    return mid_name


def tag_bar(
    natr_window: Sequence[float],
    trend_window: Sequence[float],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    low_cut: float = LOW_CUT,
    high_cut: float = HIGH_CUT,
) -> BehaviorTag:
    """
    Label a single bar from its two trailing windows.

    ``natr_window`` is volatility history (natr_14) and ``trend_window`` is
    trend-strength history — |ppo|, i.e. momentum magnitude regardless of
    direction, since "trending down hard" and "trending up hard" are the same
    behavior for our purposes.

    Both windows must END at the bar being tagged and contain only bars at or
    before it. Passing a longer window than ``DEFAULT_WINDOW`` is the caller's
    business: live, the deque's maxlen enforces the length; offline,
    :func:`tag_series` slices it.

    Warmth is judged on the volatility window alone, matching Gate B — the
    trend axis degrades to ``mixed`` on its own if it lacks data.
    """
    vol_rank = trailing_pctile_rank(natr_window)
    trend_rank = trailing_pctile_rank(trend_window)

    n_finite = int(np.count_nonzero(np.isfinite(np.asarray(natr_window, dtype=float))))
    warm = n_finite >= min_samples

    if not warm or vol_rank is None:
        return BehaviorTag(
            label=LABEL_COLD,
            vol_band=LABEL_COLD,
            trend_state=LABEL_COLD,
            vol_rank=vol_rank,
            trend_rank=trend_rank,
            warm=False,
        )

    vol_band = _band(vol_rank, VOL_LOW, VOL_NORMAL, VOL_HIGH, low_cut, high_cut)
    if trend_rank is None:
        trend_state = TREND_MIXED
    else:
        trend_state = _band(
            trend_rank, TREND_RANGING, TREND_MIXED, TREND_TRENDING, low_cut, high_cut
        )

    return BehaviorTag(
        label=f"{trend_state}_{vol_band}",
        vol_band=vol_band,
        trend_state=trend_state,
        vol_rank=vol_rank,
        trend_rank=trend_rank,
        warm=True,
    )


def tag_series(
    natr: Sequence[float],
    trend_strength: Sequence[float],
    *,
    window: int = DEFAULT_WINDOW,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    low_cut: float = LOW_CUT,
    high_cut: float = HIGH_CUT,
) -> List[BehaviorTag]:
    """
    Label an entire history, one tag per input bar.

    Defined as repeated :func:`tag_bar` over expanding-then-sliding trailing
    windows, which is precisely what the live bot's bounded deque produces as
    it fills and then rolls. No value at index ``i`` is ever influenced by
    index > ``i``.
    """
    natr_arr = np.asarray(natr, dtype=float)
    trend_arr = np.asarray(trend_strength, dtype=float)
    if natr_arr.shape != trend_arr.shape:
        raise ValueError(
            f"natr and trend_strength must be the same length, "
            f"got {natr_arr.shape} and {trend_arr.shape}"
        )

    tags: List[BehaviorTag] = []
    for i in range(natr_arr.size):
        lo = max(0, i - window + 1)
        tags.append(
            tag_bar(
                natr_arr[lo : i + 1],
                trend_arr[lo : i + 1],
                min_samples=min_samples,
                low_cut=low_cut,
                high_cut=high_cut,
            )
        )
    return tags


def trend_strength_from_ppo(ppo: Sequence[float]) -> np.ndarray:
    """
    Momentum magnitude: |ppo|.

    Direction is deliberately discarded. A hard downtrend and a hard uptrend
    are the same *behavior*, and the model already has signed momentum as a
    feature; conflating the two here would halve every cell's sample size for
    no analytic gain.
    """
    return np.abs(np.asarray(ppo, dtype=float))
