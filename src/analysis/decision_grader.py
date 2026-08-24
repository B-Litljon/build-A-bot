"""
decision_grader.py — grade the LIVE bot's decisions against what price did next.

The bot scores every bar of every instrument and writes the verdict to
``logs/events-*.jsonl`` as an ``ev="bar"`` record: symbol, bar timestamp, close,
Angel probability, Devil probability, outcome. That is roughly 500 decisions a
day. Only about one a week becomes a fill, so judging the model on its fills
throws away 99.97% of the evidence it has already generated.

This module supplies the missing answer key. It refetches the bars that
followed each decision, walks the same ATR bracket the model was trained
against, and joins the result back onto the live decision. Every recorded
decision becomes a labelled win or loss.

WHY THIS BEATS WAITING FOR FILLS
--------------------------------
* It is genuinely out-of-sample — the served artifact was trained to a cutoff,
  and every decision after it is unseen data.
* It measures the whole decision function, not just its tail: whether a 0.35
  would have won, and whether the model's confidence is honest.
* It measures THE ARTIFACT IN PRODUCTION, not a retrained proxy of its config.
  That distinction has bitten this project before.
* It is already accumulating, at ~500 decisions/day, whether or not anyone
  looks.

WHAT IT IS NOT
--------------
Simulated fills assume the price on the bar is obtainable, so results flatter
reality by roughly the spread toll (~0.10R at break-even; see GLOSSARY.md).
Read these as a *relative* measure — which confidence bands and which market
behaviors work — not as a P&L forecast.

Glossary:
    Decision -- one recorded live judgement: symbol, bar timestamp, close,
        angel/devil probabilities, and the bot's verdict.
    load_decisions -- reads ev="bar" records out of the JSONL telemetry.
    grade_decisions -- joins decisions to realized bracket outcomes, returning
        one row per decision with `won` attached.
    calibration_table -- observed win rate per Angel-probability band. The
        honest question: when the model says 0.30, does it win 30% of the
        time? Systematic under-confidence is what score compression looks
        like from the inside.
    threshold_sweep -- what would have happened at each entry bar, net of a
        spread toll. Answers "is the 0.40 bar in the right place" using LIVE
        decisions rather than a backtest.
    LOOKAHEAD_BARS -- 45, matching MAX_HOLD_BARS. A decision within this many
        bars of the data end cannot be graded and is dropped, never guessed.
"""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

LOOKAHEAD_BARS = 45


@dataclass(frozen=True)
class Decision:
    """One live judgement, exactly as the bot recorded it."""

    symbol: str
    bar_ts: str
    close: float
    angel: Optional[float]
    devil: Optional[float]
    outcome: str


def load_decisions(pattern: str = "logs/events-*.jsonl") -> pl.DataFrame:
    """
    Read every ``ev="bar"`` record from the telemetry logs.

    Malformed lines are skipped rather than fatal: the telemetry writer is
    best-effort by design, and a truncated final line during a crash must not
    cost us the other 9,000 records.
    """
    rows: List[dict] = []
    skipped = 0
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    skipped += 1
                    continue
                if rec.get("ev") != "bar":
                    continue
                if rec.get("bar_ts") is None or rec.get("sym") is None:
                    skipped += 1
                    continue
                rows.append(
                    {
                        "symbol": rec["sym"],
                        "bar_ts": str(rec["bar_ts"]),
                        "live_close": rec.get("close"),
                        "angel": rec.get("angel"),
                        "devil": rec.get("devil"),
                        "verdict": rec.get("outcome", "unknown"),
                    }
                )
    if skipped:
        logger.info("decision_grader: skipped %d unusable telemetry lines", skipped)
    if not rows:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "bar_ts": pl.Utf8,
                "live_close": pl.Float64,
                "angel": pl.Float64,
                "devil": pl.Float64,
                "verdict": pl.Utf8,
            }
        )
    return pl.DataFrame(rows).unique(subset=["symbol", "bar_ts"], keep="first")


def _normalise_ts(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Reduce a timestamp column to 'YYYY-MM-DD HH:MM' for joining.

    The telemetry writes bar_ts as a stringified datetime while the bar frame
    carries real datetimes; matching on a normalised minute avoids depending
    on either side's exact formatting or timezone suffix.
    """
    return df.with_columns(
        pl.col(col).cast(pl.Utf8).str.replace("T", " ").str.slice(0, 16).alias("_key")
    )


def grade_decisions(
    decisions: pl.DataFrame,
    graded_bars: pl.DataFrame,
) -> pl.DataFrame:
    """
    Attach realized outcomes to live decisions.

    ``graded_bars`` must carry symbol, timestamp, ``won`` (1/0 from the bracket
    walk) and optionally ``behavior_label``. Decisions with no matching graded
    bar are dropped: those are bars too close to the end of history for the
    45-bar walk to have resolved, and inventing an outcome for them would be
    the same self-deception this module exists to remove.
    """
    if decisions.height == 0 or graded_bars.height == 0:
        return decisions.head(0).with_columns(pl.lit(None, pl.Int8).alias("won"))

    d = _normalise_ts(decisions, "bar_ts")
    keep = ["symbol", "_key", "won"] + (
        ["behavior_label"] if "behavior_label" in graded_bars.columns else []
    )
    g = _normalise_ts(graded_bars, "timestamp").select(keep)

    joined = d.join(g, on=["symbol", "_key"], how="inner").drop("_key")
    if joined.height < decisions.height:
        logger.info(
            "decision_grader: %d of %d decisions graded (%d unresolved — too "
            "close to the end of history for a %d-bar walk)",
            joined.height, decisions.height,
            decisions.height - joined.height, LOOKAHEAD_BARS,
        )
    return joined


def calibration_table(
    graded: pl.DataFrame, bands: Sequence[float] = (0.0, 0.15, 0.20, 0.25, 0.30, 0.40, 1.01)
) -> pl.DataFrame:
    """
    Observed win rate per Angel-probability band.

    A well-calibrated model wins about as often as it claims. Consistently
    winning MORE often than claimed is under-confidence — which is what score
    compression looks like from the inside, and it is fixable by recalibration
    rather than by lowering the entry bar.
    """
    if graded.height == 0:
        return pl.DataFrame(
            schema={"band": pl.Utf8, "n": pl.Int64, "mean_angel": pl.Float64,
                    "win_rate": pl.Float64, "gap": pl.Float64}
        )

    rows = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        cell = graded.filter(
            (pl.col("angel") >= lo) & (pl.col("angel") < hi) & pl.col("angel").is_not_null()
        )
        if cell.height == 0:
            continue
        mean_p = float(cell["angel"].mean())
        wr = float(cell["won"].mean())
        rows.append(
            {
                "band": f"{lo:.2f}-{hi:.2f}" if hi <= 1.0 else f"{lo:.2f}+",
                "n": cell.height,
                "mean_angel": round(mean_p, 4),
                "win_rate": round(wr, 4),
                "gap": round(wr - mean_p, 4),
            }
        )
    return pl.DataFrame(rows)


def threshold_sweep(
    graded: pl.DataFrame,
    sl_mult: float,
    tp_mult: float,
    toll_r: float,
    thresholds: Sequence[float] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
) -> pl.DataFrame:
    """
    What each entry bar would have produced, net of the spread toll.

    This is the live-data version of the July threshold study. It cannot settle
    the question alone — these are simulated fills — but unlike a backtest it
    uses the probabilities the PRODUCTION artifact actually emitted.
    """
    payoff = float(tp_mult) / float(sl_mult)
    rows = []
    for t in thresholds:
        cell = graded.filter(pl.col("angel") >= t)
        if cell.height == 0:
            continue
        won = cell["won"].to_numpy().astype(bool)
        r = np.where(won, payoff, -1.0) - float(toll_r)
        gains, losses = float(r[r > 0].sum()), float(-r[r < 0].sum())
        rows.append(
            {
                "threshold": t,
                "n": int(cell.height),
                "win_rate": round(float(won.mean()), 4),
                "net_ev_r": round(float(r.mean()), 4),
                "net_pf": round(gains / losses, 3) if losses > 0 else float("inf"),
            }
        )
    return pl.DataFrame(rows)


def behavior_breakdown(graded: pl.DataFrame, min_n: int = 30) -> pl.DataFrame:
    """Win rate per market behavior, with thin cells flagged not hidden."""
    if "behavior_label" not in graded.columns or graded.height == 0:
        return pl.DataFrame(schema={"behavior_label": pl.Utf8, "n": pl.Int64,
                                    "win_rate": pl.Float64, "informative": pl.Boolean})
    return (
        graded.group_by("behavior_label")
        .agg(pl.len().alias("n"), pl.col("won").mean().alias("win_rate"))
        .with_columns((pl.col("n") >= min_n).alias("informative"))
        .sort("win_rate", descending=True)
    )
