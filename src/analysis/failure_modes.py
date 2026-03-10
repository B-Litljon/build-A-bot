#!/usr/bin/env python3
"""
Trade Failure Mode Analysis — Diagnoses how losing trades die.

Runs Angel/Devil inference on recent data, simulates ATR-dynamic brackets,
and classifies each trade outcome to guide the next optimization step.

Usage (from project root):
    python -m src.analysis.failure_modes
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import polars as pl

from src.ml.feature_pipeline import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# Bracket parameters (must match retrainer and evaluate_performance)
SL_ATR_MULTIPLIER = 0.5
TP_ATR_MULTIPLIER = 3.0
MAX_HOLD_BARS = 45
FAST_SL_CUTOFF = 3  # bars 1-3 = "fast" SL hit

# Model thresholds (must match LiveOrchestrator / MLStrategy)
ANGEL_THRESHOLD = 0.40
DEVIL_THRESHOLD = 0.50

# Model paths — primary (retrainer output), then fallback (train_model output)
ANGEL_PATH = Path("models/angel_latest.pkl")
DEVIL_PATH = Path("models/devil_latest.pkl")
ALT_ANGEL_PATH = Path("src/ml/models/angel_rf_model.joblib")
ALT_DEVIL_PATH = Path("src/ml/models/devil_rf_model.joblib")

# Data paths
OOS_BARS_PATH = Path("data/oos_bars.parquet")
RAW_DATA_DIR = Path("data/raw")

# Tickers to analyse
TICKERS = ["TSLA", "NVDA", "MARA", "COIN", "SMCI"]


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_data() -> pl.DataFrame:
    """Load the most recent available bar data."""
    if OOS_BARS_PATH.exists():
        logger.info(f"Loading OOS bars from {OOS_BARS_PATH}")
        df = pl.read_parquet(OOS_BARS_PATH)
        logger.info(f"  {len(df):,} rows, columns: {df.columns}")
        return df

    logger.info(f"No OOS bars found at {OOS_BARS_PATH}. Scanning {RAW_DATA_DIR}")
    frames = []
    for path in sorted(RAW_DATA_DIR.glob("*_1min.parquet")):
        symbol = path.stem.replace("_1min", "").upper()
        if symbol in TICKERS:
            df = pl.read_parquet(path)
            if "symbol" not in df.columns:
                df = df.with_columns(pl.lit(symbol).alias("symbol"))
            frames.append(df)
            logger.info(f"  Loaded {len(df):,} bars for {symbol}")

    if not frames:
        raise FileNotFoundError(
            f"No bar data found at {OOS_BARS_PATH} or in {RAW_DATA_DIR}/*.parquet"
        )

    combined = pl.concat(frames, how="vertical_relaxed")
    return combined.sort(["symbol", "timestamp"])


def load_models():
    """Load Angel and Devil models from primary or fallback paths."""
    if ANGEL_PATH.exists() and DEVIL_PATH.exists():
        logger.info(f"Loading models from {ANGEL_PATH} / {DEVIL_PATH}")
        return joblib.load(ANGEL_PATH), joblib.load(DEVIL_PATH)

    if ALT_ANGEL_PATH.exists() and ALT_DEVIL_PATH.exists():
        logger.info(f"Loading models from {ALT_ANGEL_PATH} / {ALT_DEVIL_PATH}")
        return joblib.load(ALT_ANGEL_PATH), joblib.load(ALT_DEVIL_PATH)

    raise FileNotFoundError(
        f"Models not found at {ANGEL_PATH} or {ALT_ANGEL_PATH}. "
        "Run the retrainer first: python -m src.core.retrainer"
    )


# ── Inference ─────────────────────────────────────────────────────────────────


def run_inference(
    df: pl.DataFrame,
    angel_model,
    devil_model,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run Angel -> Devil inference and return:
      - signals_df: rows that passed both thresholds, with angel_prob/devil_prob
      - featured_df: full feature-engineered DataFrame (for bracket simulation)
    """
    fe = FeatureEngineer()
    featured_df = fe.compute_indicators(df)

    # Feature lists come from the models themselves — guaranteed to match
    angel_features: List[str] = list(angel_model.feature_names_in_)
    devil_features: List[str] = list(devil_model.feature_names_in_)
    # devil_features = angel_features + ["angel_prob"]

    # Drop rows missing any base feature
    featured_df = featured_df.drop_nulls(subset=angel_features)
    logger.info(f"Running inference on {len(featured_df):,} bars after null-drop...")

    X_angel = featured_df[angel_features].to_numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        angel_probs = angel_model.predict_proba(X_angel)[:, 1]

    angel_mask = angel_probs >= ANGEL_THRESHOLD
    n_angel = int(angel_mask.sum())
    logger.info(
        f"Angel proposed {n_angel:,} trades "
        f"({n_angel / len(featured_df):.1%} of bars, threshold={ANGEL_THRESHOLD})"
    )

    if n_angel == 0:
        raise ValueError("Angel proposed 0 trades — cannot analyse failure modes.")

    # Devil inference on Angel-proposed rows only
    X_proposed_base = X_angel[angel_mask]
    angel_probs_proposed = angel_probs[angel_mask]

    # Build Devil feature matrix: base features + angel_prob (must match training order)
    X_devil = np.column_stack([X_proposed_base, angel_probs_proposed])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        devil_probs = devil_model.predict_proba(X_devil)[:, 1]

    devil_mask = devil_probs >= DEVIL_THRESHOLD
    n_devil = int(devil_mask.sum())
    logger.info(
        f"Devil approved {n_devil:,} trades "
        f"({n_devil / n_angel:.1%} of Angel proposals, threshold={DEVIL_THRESHOLD})"
    )

    if n_devil == 0:
        raise ValueError("Devil approved 0 trades — cannot analyse failure modes.")

    # Reconstruct which rows of featured_df are approved
    angel_indices = np.where(angel_mask)[0]
    approved_indices = angel_indices[devil_mask]

    signals_df = featured_df[approved_indices.tolist()].with_columns(
        [
            pl.Series("angel_prob", angel_probs_proposed[devil_mask]),
            pl.Series("devil_prob", devil_probs[devil_mask]),
        ]
    )

    return signals_df, featured_df


# ── Bracket Simulation ────────────────────────────────────────────────────────


def simulate_brackets(
    signals_df: pl.DataFrame,
    bars_df: pl.DataFrame,
) -> List[Dict]:
    """
    Simulate ATR-dynamic bracket orders bar-by-bar for each signal row.

    Returns a list of trade result dicts with 5-way exit classification.
    """
    trades: List[Dict] = []

    for symbol in sorted(signals_df["symbol"].unique().to_list()):
        sym_signals = signals_df.filter(pl.col("symbol") == symbol)
        sym_bars = bars_df.filter(pl.col("symbol") == symbol).sort("timestamp")

        bar_timestamps = sym_bars["timestamp"].to_list()
        bar_highs = sym_bars["high"].to_numpy()
        bar_lows = sym_bars["low"].to_numpy()
        bar_closes = sym_bars["close"].to_numpy()

        ts_to_idx = {ts: i for i, ts in enumerate(bar_timestamps)}

        for row in sym_signals.iter_rows(named=True):
            entry_ts = row["timestamp"]
            entry_price = float(row["close"])
            natr_14 = float(row["natr_14"])

            atr_abs = entry_price * natr_14 / 100.0
            if atr_abs <= 0:
                continue

            sl_price = entry_price - SL_ATR_MULTIPLIER * atr_abs
            tp_price = entry_price + TP_ATR_MULTIPLIER * atr_abs

            entry_idx = ts_to_idx.get(entry_ts)
            if entry_idx is None:
                continue

            exit_type = None
            exit_bar = None
            exit_price = None

            for j in range(1, MAX_HOLD_BARS + 1):
                lookahead_idx = entry_idx + j
                if lookahead_idx >= len(bar_closes):
                    break

                # SL checked first — conservative, matches evaluate_performance.py
                if bar_lows[lookahead_idx] <= sl_price:
                    exit_price = sl_price
                    exit_bar = j
                    exit_type = "SL_HIT_FAST" if j <= FAST_SL_CUTOFF else "SL_HIT_SLOW"
                    break

                if bar_highs[lookahead_idx] >= tp_price:
                    exit_price = tp_price
                    exit_bar = j
                    exit_type = "TP_HIT"
                    break

            # Timeout
            if exit_type is None:
                exit_bar = MAX_HOLD_BARS
                timeout_idx = entry_idx + MAX_HOLD_BARS
                exit_price = (
                    float(bar_closes[timeout_idx])
                    if timeout_idx < len(bar_closes)
                    else float(bar_closes[-1])
                )
                exit_type = (
                    "TIMEOUT_WIN" if exit_price >= entry_price else "TIMEOUT_LOSS"
                )

            pnl = exit_price - entry_price
            pnl_r = pnl / atr_abs

            trades.append(
                {
                    "symbol": symbol,
                    "entry_ts": entry_ts,
                    "entry_price": entry_price,
                    "atr_abs": atr_abs,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "exit_type": exit_type,
                    "exit_bar": exit_bar,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_r": pnl_r,
                    "angel_prob": row["angel_prob"],
                    "devil_prob": row["devil_prob"],
                }
            )

    return trades


# ── Analysis Output ───────────────────────────────────────────────────────────


def print_analysis(trades: List[Dict]) -> None:
    """Print the full failure mode analysis to stdout."""
    if not trades:
        logger.error("No trades to analyse!")
        return

    n_total = len(trades)

    counts: Dict[str, int] = {}
    pnls: Dict[str, List[float]] = {}
    bars_by_type: Dict[str, List[int]] = {}

    for t in trades:
        et = t["exit_type"]
        counts[et] = counts.get(et, 0) + 1
        pnls.setdefault(et, []).append(t["pnl_r"])
        bars_by_type.setdefault(et, []).append(t["exit_bar"])

    display_order = [
        "TP_HIT",
        "SL_HIT_FAST",
        "SL_HIT_SLOW",
        "TIMEOUT_LOSS",
        "TIMEOUT_WIN",
    ]
    icons = {
        "TP_HIT": "WIN ",
        "TIMEOUT_WIN": "WIN ",
        "SL_HIT_FAST": "LOSS",
        "SL_HIT_SLOW": "LOSS",
        "TIMEOUT_LOSS": "LOSS",
    }

    n_wins = counts.get("TP_HIT", 0) + counts.get("TIMEOUT_WIN", 0)

    print("\n" + "=" * 70)
    print("TRADE FAILURE MODE ANALYSIS")
    print("=" * 70)
    print(f"\nTotal Trades : {n_total}")
    print(f"Win Rate     : {n_wins}/{n_total} ({n_wins / n_total:.1%})")

    print(f"\n{'─' * 70}")
    print(
        f"{'Exit Type':<22} {'Count':>6} {'Pct':>8} {'Avg PnL (R)':>13} {'Avg Bars':>10}"
    )
    print(f"{'─' * 70}")

    for et in display_order:
        if et not in counts:
            continue
        c = counts[et]
        avg_pnl = float(np.mean(pnls[et]))
        avg_bar = float(np.mean(bars_by_type[et]))
        print(
            f"[{icons[et]}] {et:<18} {c:>6} {c / n_total:>7.1%} "
            f"{avg_pnl:>+12.3f}R {avg_bar:>9.1f}"
        )

    print(f"{'─' * 70}")

    # ── Diagnostic Summary ────────────────────────────────────────────────────
    sl_fast = counts.get("SL_HIT_FAST", 0)
    sl_slow = counts.get("SL_HIT_SLOW", 0)
    to_loss = counts.get("TIMEOUT_LOSS", 0)
    to_win = counts.get("TIMEOUT_WIN", 0)
    tp_hit = counts.get("TP_HIT", 0)
    total_losses = sl_fast + sl_slow + to_loss

    print(f"\n{'=' * 70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'=' * 70}")

    if total_losses > 0:
        print(f"\nOf {total_losses} losing trades:")
        print(
            f"  SL Hit Fast  (bars 1-{FAST_SL_CUTOFF}):  {sl_fast:>4} ({sl_fast / total_losses:.1%})"
        )
        print(
            f"  SL Hit Slow  (bars {FAST_SL_CUTOFF + 1}-{MAX_HOLD_BARS}): {sl_slow:>4} ({sl_slow / total_losses:.1%})"
        )
        print(
            f"  Timeout Loss (bar {MAX_HOLD_BARS}):   {to_loss:>4} ({to_loss / total_losses:.1%})"
        )
    else:
        print("\nNo losses! Nothing to diagnose.")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    if total_losses == 0:
        print("No losses to diagnose!")
    else:
        sl_fast_pct = sl_fast / total_losses
        sl_slow_pct = sl_slow / total_losses
        to_loss_pct = to_loss / total_losses
        total_sl_pct = (sl_fast + sl_slow) / total_losses

        if sl_fast_pct > 0.50:
            print("\nWHIPSAW DOMINANT: >50% of losses are instant SL hits (bars 1-3)")
            print(
                "   Primary fix  : REGIME GATING — filter out choppy/ranging conditions"
            )
            print("   Secondary    : Consider widening SL or adding entry delay")
        elif total_sl_pct > 0.70:
            print("\nSL DOMINANT: >70% of losses hit stop-loss (fast + slow combined)")
            print("   Consider     : Wider SL, tighter TP, or regime-based SL scaling")
        elif to_loss_pct > 0.50:
            print("\nCHOP DOMINANT: >50% of losses are timeouts (no directional move)")
            print(
                "   Primary fix  : BRACKET TUNING — lower TP target (2x ATR instead of 3x)"
            )
            print("   Secondary    : Reduce MAX_HOLD_BARS or add trailing stop")
        else:
            print("\nMIXED: No single dominant failure mode")
            print(
                f"   SL Fast {sl_fast_pct:.0%} | SL Slow {sl_slow_pct:.0%} | Timeout {to_loss_pct:.0%}"
            )
            print("   Consider both regime gating AND bracket tuning")

    # ── Break-even analysis ───────────────────────────────────────────────────
    # Need WR > SL / (SL + TP) = 0.5 / (0.5 + 3.0) = 14.3%
    breakeven_wr = SL_ATR_MULTIPLIER / (SL_ATR_MULTIPLIER + TP_ATR_MULTIPLIER)
    actual_wr = n_wins / n_total
    print(f"\n{'─' * 70}")
    print(f"Break-even WR (at {TP_ATR_MULTIPLIER}:1 R/R): {breakeven_wr:.1%}")
    print(
        f"Actual WR                   : {actual_wr:.1%}  "
        f"({'ABOVE' if actual_wr >= breakeven_wr else 'BELOW'} break-even)"
    )
    wr_gap = actual_wr - breakeven_wr
    print(f"Gap to break-even           : {wr_gap:+.1%}")

    # ── Per-Symbol Breakdown ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PER-SYMBOL BREAKDOWN")
    print(f"{'=' * 70}")
    print(
        f"\n{'Symbol':<10} {'Trades':>7} {'WR':>6} "
        f"{'SL Fast':>8} {'SL Slow':>8} {'TO Loss':>8} {'TP Hit':>7} {'TO Win':>7}"
    )
    print(f"{'─' * 70}")

    for sym in sorted(set(t["symbol"] for t in trades)):
        st = [t for t in trades if t["symbol"] == sym]
        n = len(st)
        if n == 0:
            continue
        sym_wins = sum(1 for t in st if t["exit_type"] in ("TP_HIT", "TIMEOUT_WIN"))
        sym_slf = sum(1 for t in st if t["exit_type"] == "SL_HIT_FAST")
        sym_sls = sum(1 for t in st if t["exit_type"] == "SL_HIT_SLOW")
        sym_tol = sum(1 for t in st if t["exit_type"] == "TIMEOUT_LOSS")
        sym_tp = sum(1 for t in st if t["exit_type"] == "TP_HIT")
        sym_tow = sum(1 for t in st if t["exit_type"] == "TIMEOUT_WIN")
        print(
            f"{sym:<10} {n:>7} {sym_wins / n:>5.0%} "
            f"{sym_slf:>8} {sym_sls:>8} {sym_tol:>8} {sym_tp:>7} {sym_tow:>7}"
        )

    print(f"{'─' * 70}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────


def main() -> int:
    try:
        print("=" * 70)
        print("FAILURE MODE ANALYSIS — How Are Losing Trades Dying?")
        print("=" * 70)

        bars_df = load_data()
        logger.info(f"Loaded {len(bars_df):,} total bars")

        angel_model, devil_model = load_models()
        logger.info(
            f"Angel features ({len(angel_model.feature_names_in_)}): "
            f"{list(angel_model.feature_names_in_)}"
        )
        logger.info(
            f"Devil features ({len(devil_model.feature_names_in_)}): "
            f"{list(devil_model.feature_names_in_)}"
        )

        signals_df, featured_df = run_inference(bars_df, angel_model, devil_model)
        logger.info(f"Generated {len(signals_df):,} approved BUY signals for analysis")

        trades = simulate_brackets(signals_df, featured_df)
        logger.info(f"Simulated {len(trades):,} bracket trades")

        print_analysis(trades)
        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing data or models: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Analysis error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failure mode analysis crashed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
