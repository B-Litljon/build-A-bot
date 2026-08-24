"""
behavior_matrix.py — score candidate configurations per market behavior.

Answers one question: *which configuration earns its keep when the market
looks like this?* Rows are behavior tags (``trend_high``, ``range_low``, …
from ``ml/regimes/behavior_tagger``), columns are candidates, cells carry
trade count, win rate, expectancy and profit factor with bootstrap intervals.

WHERE THE NUMBERS COME FROM
---------------------------
Every trade in this matrix is a **Devil-approved out-of-sample trade from the
retrainer's own expanding walk-forward**, captured via
``validate_candidate(oos_ledger=...)``. That choice is deliberate:

* A candidate is a **configuration** (lookback, brackets, thresholds), not a
  trained artifact. Scoring saved artifacts instead would compare models over
  different periods — the exact error the 2026-08-17 five-year review found and
  discarded, since fold windows scale with lookback.
* The walk-forward is the maintained forex scorer. ``replay_test.py`` and
  ``evaluate_performance.py`` are Alpaca-era and hardcode the equities setup
  (root model paths, 0.5x/3.0x brackets, threshold 0.50).
* Reusing it means the chop veto, the labels and the feature pipeline are
  identical to the ones the promotion gate sees. No second, drifting scorer.

COST IS NOT OPTIONAL
--------------------
Gross profit factor is the trap this project has already been caught by: gross
1.373 fell to 1.004 after spread on the shipped model. Every cell is therefore
reported **net** of a per-trade spread toll, and gross is kept beside it only
so the size of the toll stays visible.

Glossary:
    Candidate -- one evaluation configuration: a name, the bracket multiples,
        and the training lookback. What gets ranked.
    CellStats -- one (behavior, candidate) cell: n, win rate, gross and net
        expectancy in R, profit factor, and a bootstrap CI on net expectancy.
    MIN_CELL_TRADES -- 30. Below this a cell is reported but flagged
        uninformative; per the 2026-08-17 power analysis, cells thin out fast
        and a 12-trade cell will happily show PF 3.0 by luck.
    BOOTSTRAP_N -- 2000 resamples for the CI. Enough to stabilise a 95%
        interval without making a full sweep slow.
    tag_frame -- adds ``behavior_label`` to an engineered feature frame,
        per symbol, causally. Must run BEFORE the walk-forward so the label
        rides into the ledger.
    score_ledger -- turns a captured OOS ledger into per-cell statistics.
    net_r -- a trade's result in R after the spread toll. Wins pay
        ``tp_mult/sl_mult`` R, losses cost 1 R, and the toll is subtracted from
        both because the spread is paid on entry regardless of outcome.
    spread_toll_r -- the toll in R units: spread / stop distance. Defaults to
        the measured live figure rather than a guess; see GLOSSARY.md.
    DEFAULT_TOLL_R -- 0.33. The forex profile admits a trade only when the
        spread eats at most 1/spread_k_base = 1/3.0 of the stop, so 0.33 is the
        WORST admissible toll, not the typical one. Override with the measured
        per-instrument alpha when a spread table is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

from ml.regimes.behavior_tagger import (
    LABEL_COLD,
    tag_series,
    trend_strength_from_ppo,
)

logger = logging.getLogger(__name__)

MIN_CELL_TRADES = 30
BOOTSTRAP_N = 2000
DEFAULT_TOLL_R = 0.33


@dataclass(frozen=True)
class Candidate:
    """One configuration to score. Not a trained artifact — a recipe."""

    name: str
    sl_mult: float
    tp_mult: float
    lookback_days: int
    notes: str = ""


@dataclass
class CellStats:
    behavior: str
    candidate: str
    n: int
    win_rate: float
    gross_ev_r: float
    net_ev_r: float
    profit_factor_gross: float
    profit_factor_net: float
    ci_low: float
    ci_high: float
    informative: bool = field(default=False)

    @property
    def significant(self) -> bool:
        """True when the bootstrap CI on net expectancy excludes zero."""
        return not (self.ci_low < 0.0 < self.ci_high)


def tag_frame(
    df: pl.DataFrame,
    *,
    natr_col: str = "natr_14",
    ppo_col: str = "ppo",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """
    Add a causal ``behavior_label`` column, tagged per symbol.

    Tagging per symbol matters: a bar is quiet or violent relative to *its own*
    instrument's recent range, never relative to a pool that mixes GBP_JPY with
    XAU_USD. Rows are sorted by timestamp within each symbol first, because
    the tagger's causality guarantee assumes chronological order.
    """
    for col in (natr_col, ppo_col, symbol_col):
        if col not in df.columns:
            raise ValueError(f"tag_frame requires a '{col}' column")

    frames: List[pl.DataFrame] = []
    for sym in df[symbol_col].unique(maintain_order=True).to_list():
        d = df.filter(pl.col(symbol_col) == sym)
        if "timestamp" in d.columns:
            d = d.sort("timestamp")
        tags = tag_series(
            d[natr_col].to_numpy().astype(float),
            trend_strength_from_ppo(d[ppo_col].to_numpy().astype(float)),
        )
        frames.append(d.with_columns(pl.Series("behavior_label", [t.label for t in tags])))

    return pl.concat(frames)


def net_r(
    macro_win: np.ndarray,
    sl_mult: float,
    tp_mult: float,
    toll_r: float = DEFAULT_TOLL_R,
) -> np.ndarray:
    """
    Per-trade result in R, net of the spread toll.

    A win pays ``tp_mult / sl_mult`` R and a loss costs 1 R — the retrainer's
    own R convention, kept identical so gross numbers here reconcile with the
    gate's. The toll is then subtracted from *every* trade, win or lose,
    because the spread is paid crossing the book on entry.
    """
    payoff = float(tp_mult) / float(sl_mult)
    gross = np.where(macro_win.astype(bool), payoff, -1.0)
    return gross - float(toll_r)


def _profit_factor(r: np.ndarray) -> float:
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses <= 0.0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _bootstrap_ci(
    r: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = 0
) -> tuple[float, float]:
    if r.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n_boot, r.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def score_ledger(
    ledger: pl.DataFrame,
    candidate: Candidate,
    *,
    toll_r: float = DEFAULT_TOLL_R,
    min_cell_trades: int = MIN_CELL_TRADES,
    seed: int = 0,
) -> List[CellStats]:
    """
    Turn one candidate's captured OOS ledger into per-behavior statistics.

    ``cold`` bars are dropped rather than pooled: an unwarmed window is an
    absence of evidence, and folding it into a real cell would contaminate it.
    """
    if "behavior_label" not in ledger.columns:
        raise ValueError(
            "ledger has no 'behavior_label' — call tag_frame() on the feature "
            "frame BEFORE running the walk-forward, so the label rides along"
        )

    warm = ledger.filter(pl.col("behavior_label") != LABEL_COLD)
    if warm.height < ledger.height:
        logger.info(
            "behavior_matrix: dropped %d cold-window trades of %d",
            ledger.height - warm.height,
            ledger.height,
        )

    out: List[CellStats] = []
    for behavior in sorted(warm["behavior_label"].unique().to_list()):
        cell = warm.filter(pl.col("behavior_label") == behavior)
        wins = cell["macro_win"].to_numpy()
        r_net = net_r(wins, candidate.sl_mult, candidate.tp_mult, toll_r)
        r_gross = net_r(wins, candidate.sl_mult, candidate.tp_mult, 0.0)
        lo, hi = _bootstrap_ci(r_net, seed=seed)

        out.append(
            CellStats(
                behavior=behavior,
                candidate=candidate.name,
                n=int(cell.height),
                win_rate=float(wins.mean()) if wins.size else 0.0,
                gross_ev_r=float(r_gross.mean()),
                net_ev_r=float(r_net.mean()),
                profit_factor_gross=_profit_factor(r_gross),
                profit_factor_net=_profit_factor(r_net),
                ci_low=lo,
                ci_high=hi,
                informative=cell.height >= min_cell_trades,
            )
        )
    return out


def to_frame(cells: Sequence[CellStats]) -> pl.DataFrame:
    """Flatten cells into a frame, best net expectancy first."""
    if not cells:
        return pl.DataFrame(
            schema={
                "behavior": pl.Utf8,
                "candidate": pl.Utf8,
                "n": pl.Int64,
                "win_rate": pl.Float64,
                "gross_ev_r": pl.Float64,
                "net_ev_r": pl.Float64,
                "profit_factor_gross": pl.Float64,
                "profit_factor_net": pl.Float64,
                "ci_low": pl.Float64,
                "ci_high": pl.Float64,
                "informative": pl.Boolean,
                "significant": pl.Boolean,
            }
        )
    return pl.DataFrame(
        [
            {
                "behavior": c.behavior,
                "candidate": c.candidate,
                "n": c.n,
                "win_rate": c.win_rate,
                "gross_ev_r": c.gross_ev_r,
                "net_ev_r": c.net_ev_r,
                "profit_factor_gross": c.profit_factor_gross,
                "profit_factor_net": c.profit_factor_net,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
                "informative": c.informative,
                "significant": c.significant,
            }
            for c in cells
        ]
    ).sort("net_ev_r", descending=True)


def recommend(
    cells: Sequence[CellStats], *, require_informative: bool = True
) -> Dict[str, Optional[str]]:
    """
    Best candidate per behavior — the "recommender" half of the tool.

    Returns None for a behavior where no candidate clears the evidence bar,
    which is a real and common answer. Refusing to name a winner on 11 trades
    is the whole point; a recommender that always recommends is a random
    number generator with a nice interface.
    """
    best: Dict[str, Optional[str]] = {}
    by_behavior: Dict[str, List[CellStats]] = {}
    for c in cells:
        by_behavior.setdefault(c.behavior, []).append(c)

    for behavior, group in by_behavior.items():
        eligible = [
            c
            for c in group
            if (c.informative or not require_informative) and c.net_ev_r > 0
        ]
        best[behavior] = (
            max(eligible, key=lambda c: c.net_ev_r).candidate if eligible else None
        )
    return best
