#!/usr/bin/env python3
"""
Vectorized Bracket Optimization — Sweeps SL/TP/Timeout parameters.

Finds the optimal bracket configuration by simulating all parameter
combinations against real Angel/Devil signals on recent OOS data.

Usage (from project root):
    python -m src.analysis.optimize_brackets
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import polars as pl

from src.ml.feature_pipeline import FeatureEngineer
from src.core.retrainer import fetch_training_data, get_alpaca_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Grid Search Parameters ────────────────────────────────────────────────────
SL_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]
TP_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]
MAX_HOLD_OPTIONS = [10, 15, 20, 30, 45]

# Model thresholds (must match LiveOrchestrator / MLStrategy)
ANGEL_THRESHOLD = 0.40
DEVIL_THRESHOLD = 0.50

# Model paths — primary (retrainer output), then fallback (train_model output)
MODEL_PATHS = [
    (Path("models/angel_latest.pkl"), Path("models/devil_latest.pkl")),
    (
        Path("src/ml/models/angel_rf_model.joblib"),
        Path("src/ml/models/devil_rf_model.joblib"),
    ),
]

# Ticker list (matches retrainer; kept here for reference only)
TICKERS = ["TSLA", "NVDA", "MARA", "COIN", "SMCI"]

# Minimum trades for a configuration to be reported
MIN_TRADES = 10


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BracketConfig:
    sl_mult: float
    tp_mult: float
    max_hold: int


@dataclass
class BracketResult:
    config: BracketConfig
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_r: float
    avg_r: float
    tp_hits: int
    sl_hits: int
    timeout_wins: int
    timeout_losses: int
    breakeven_wr: float


# ── Data / Model Loading ──────────────────────────────────────────────────────


def load_data() -> pl.DataFrame:
    """
    Fetch 60 days of 1-minute bars from Alpaca via the retrainer's pipeline.

    Uses the same fetch_training_data() / get_alpaca_client() functions as
    src/core/retrainer.py to guarantee data consistency with training.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment (.env).
    """
    logger.info("Fetching 60 days of 1-minute bars from Alpaca...")
    client = get_alpaca_client()
    raw_df = fetch_training_data(client)
    logger.info(
        f"Fetched {len(raw_df):,} bars across {raw_df['symbol'].n_unique()} symbols"
    )
    return raw_df


def load_models():
    for angel_path, devil_path in MODEL_PATHS:
        if angel_path.exists() and devil_path.exists():
            logger.info(f"Loading models from {angel_path.parent}/")
            angel_model = joblib.load(angel_path)
            devil_model = joblib.load(devil_path)
            angel_model.n_jobs = (
                1  # Prevent joblib IPC overhead on single-row inference
            )
            devil_model.n_jobs = 1
            return angel_model, devil_model
    raise FileNotFoundError(
        "Models not found. Run the retrainer: python -m src.core.retrainer"
    )


# ── Signal Generation ─────────────────────────────────────────────────────────


def generate_signals(
    bars_df: pl.DataFrame,
    angel_model,
    devil_model,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Run Angel -> Devil inference; return (signals_df, featured_df)."""
    # Safety: ensure timestamp is UTC-aware (fetch_training_data should already do this)
    if "timestamp" in bars_df.columns:
        ts_dtype = bars_df.schema["timestamp"]
        if not hasattr(ts_dtype, "time_zone") or ts_dtype.time_zone is None:
            bars_df = bars_df.with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC")
            )

    fe = FeatureEngineer()
    featured_df = fe.compute_indicators(bars_df)

    angel_features: List[str] = list(angel_model.feature_names_in_)
    featured_df = featured_df.drop_nulls(subset=angel_features)
    logger.info(f"Running inference on {len(featured_df):,} bars...")

    X_angel = featured_df[angel_features].to_numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        angel_probs = angel_model.predict_proba(X_angel)[:, 1]

    angel_mask = angel_probs >= ANGEL_THRESHOLD
    n_angel = int(angel_mask.sum())
    logger.info(
        f"Angel proposed {n_angel:,} trades "
        f"({n_angel / len(featured_df):.1%}, threshold={ANGEL_THRESHOLD})"
    )
    if n_angel == 0:
        raise ValueError("Angel proposed 0 trades.")

    X_proposed = X_angel[angel_mask]
    ap_proposed = angel_probs[angel_mask]
    X_devil = np.column_stack([X_proposed, ap_proposed])

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
        raise ValueError("Devil approved 0 trades.")

    approved_indices = np.where(angel_mask)[0][devil_mask]
    signals_df = featured_df[approved_indices.tolist()].with_columns(
        [
            pl.Series("angel_prob", ap_proposed[devil_mask]),
            pl.Series("devil_prob", devil_probs[devil_mask]),
        ]
    )
    return signals_df, featured_df


# ── Lookahead Precomputation ──────────────────────────────────────────────────


def precompute_lookahead(
    signals_df: pl.DataFrame,
    bars_df: pl.DataFrame,
) -> List[Dict]:
    """
    Build a list of signal dicts, each containing precomputed future bar arrays.

    Done once; reused across all 100 bracket configurations.
    """
    max_bars = max(MAX_HOLD_OPTIONS)
    logger.info(f"Precomputing lookahead arrays (max {max_bars} bars)...")

    signal_data: List[Dict] = []

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

            entry_idx = ts_to_idx.get(entry_ts)
            if entry_idx is None or atr_abs <= 0 or np.isnan(atr_abs):
                continue

            end_idx = min(entry_idx + max_bars + 1, len(bar_closes))
            future_highs = bar_highs[entry_idx + 1 : end_idx]
            future_lows = bar_lows[entry_idx + 1 : end_idx]
            future_closes = bar_closes[entry_idx + 1 : end_idx]

            if len(future_highs) == 0:
                continue

            signal_data.append(
                {
                    "entry_price": entry_price,
                    "atr_abs": atr_abs,
                    "future_highs": future_highs,
                    "future_lows": future_lows,
                    "future_closes": future_closes,
                    "symbol": row["symbol"],
                }
            )

    logger.info(f"Precomputed lookahead for {len(signal_data):,} valid signals")
    return signal_data


# ── Single-Config Simulation ──────────────────────────────────────────────────


def simulate_config(signal_data: List[Dict], config: BracketConfig) -> BracketResult:
    """Simulate one bracket configuration across all precomputed signals (single pass)."""
    sl_mult = config.sl_mult
    tp_mult = config.tp_mult
    max_hold = config.max_hold

    tp_hits = sl_hits = timeout_wins = timeout_losses = 0
    gross_profit = gross_loss = 0.0
    total_r = 0.0

    for sig in signal_data:
        entry_price = sig["entry_price"]
        atr_abs = sig["atr_abs"]
        future_highs = sig["future_highs"]
        future_lows = sig["future_lows"]
        future_closes = sig["future_closes"]

        sl_price = entry_price - sl_mult * atr_abs
        tp_price = entry_price + tp_mult * atr_abs
        risk = sl_mult * atr_abs  # always > 0 (filtered in precompute)

        bars_to_check = min(max_hold, len(future_highs))
        exit_price = None
        is_tp = False
        is_sl = False

        for j in range(bars_to_check):
            if future_lows[j] <= sl_price:
                exit_price = sl_price
                is_sl = True
                break
            if future_highs[j] >= tp_price:
                exit_price = tp_price
                is_tp = True
                break

        if exit_price is None:
            # Timeout — use close of last available bar up to max_hold
            timeout_idx = min(max_hold - 1, len(future_closes) - 1)
            exit_price = float(future_closes[timeout_idx])

        pnl_r = (exit_price - entry_price) / risk
        total_r += pnl_r

        if pnl_r > 0:
            gross_profit += pnl_r
        else:
            gross_loss += abs(pnl_r)

        if is_tp:
            tp_hits += 1
        elif is_sl:
            sl_hits += 1
        elif exit_price >= entry_price:
            timeout_wins += 1
        else:
            timeout_losses += 1

    total_trades = tp_hits + sl_hits + timeout_wins + timeout_losses
    wins = tp_hits + timeout_wins
    losses = sl_hits + timeout_losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    avg_r = total_r / total_trades if total_trades > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    breakeven_wr = 1.0 / (1.0 + tp_mult / sl_mult)

    return BracketResult(
        config=config,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        profit_factor=pf,
        total_r=total_r,
        avg_r=avg_r,
        tp_hits=tp_hits,
        sl_hits=sl_hits,
        timeout_wins=timeout_wins,
        timeout_losses=timeout_losses,
        breakeven_wr=breakeven_wr,
    )


# ── Grid Search ───────────────────────────────────────────────────────────────


def run_grid_search(signal_data: List[Dict]) -> List[BracketResult]:
    configs = [
        BracketConfig(sl, tp, hold)
        for sl in SL_MULTIPLIERS
        for tp in TP_MULTIPLIERS
        for hold in MAX_HOLD_OPTIONS
    ]
    n = len(configs)
    logger.info(f"Running {n} bracket configurations...")

    results = []
    for i, config in enumerate(configs):
        results.append(simulate_config(signal_data, config))
        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i + 1}/{n}")

    logger.info(f"Grid search complete.")
    return results


# ── Output ────────────────────────────────────────────────────────────────────

_HDR = (
    f"{'Rank':<5} {'SL×':>5} {'TP×':>5} {'Hold':>5} {'Trades':>7} "
    f"{'WR':>6} {'BE WR':>6} {'PF':>7} {'Total R':>9} {'Avg R':>7} "
    f"{'TP%':>5} {'SL%':>5} {'TO%':>5}"
)
_SEP = "─" * 92


def _row(rank: int, r: BracketResult) -> str:
    tp_pct = r.tp_hits / r.total_trades if r.total_trades else 0
    sl_pct = r.sl_hits / r.total_trades if r.total_trades else 0
    to_pct = (
        (r.timeout_wins + r.timeout_losses) / r.total_trades if r.total_trades else 0
    )
    margin = r.win_rate - r.breakeven_wr
    flag = "+" if margin > 0.05 else ("~" if margin > 0 else "-")
    return (
        f"[{flag}]{rank:<4} {r.config.sl_mult:>5.1f} {r.config.tp_mult:>5.1f} "
        f"{r.config.max_hold:>5} {r.total_trades:>7} {r.win_rate:>5.1%} "
        f"{r.breakeven_wr:>5.1%} {r.profit_factor:>7.2f} {r.total_r:>+9.1f} "
        f"{r.avg_r:>+7.3f} {tp_pct:>4.0%} {sl_pct:>4.0%} {to_pct:>4.0%}"
    )


def print_results(results: List[BracketResult]) -> None:
    valid = [r for r in results if r.total_trades >= MIN_TRADES]
    if not valid:
        logger.error(f"No configurations had >= {MIN_TRADES} trades!")
        return

    print("\n" + "=" * 92)
    print("BRACKET OPTIMIZATION — GRID SEARCH RESULTS")
    print("=" * 92)
    print(f"Total configurations tested : {len(results)}")
    print(f"Valid (>= {MIN_TRADES} trades)           : {len(valid)}")
    print(f"Flag key: [+] WR > BE+5%   [~] WR > BE   [-] WR < BE")

    # ── Top 10 by Profit Factor ───────────────────────────────────────────────
    by_pf = sorted(valid, key=lambda r: r.profit_factor, reverse=True)
    print(f"\n{_SEP}\nTOP 10 BY PROFIT FACTOR\n{_SEP}")
    print(_HDR)
    print(_SEP)
    for i, r in enumerate(by_pf[:10]):
        print(_row(i + 1, r))

    # ── Top 10 by Total R ─────────────────────────────────────────────────────
    by_r = sorted(valid, key=lambda r: r.total_r, reverse=True)
    print(f"\n{_SEP}\nTOP 10 BY TOTAL R-MULTIPLE\n{_SEP}")
    print(_HDR)
    print(_SEP)
    for i, r in enumerate(by_r[:10]):
        print(_row(i + 1, r))

    # ── Top 10 by Avg R (risk-adjusted) ──────────────────────────────────────
    by_avg = sorted(valid, key=lambda r: r.avg_r, reverse=True)
    print(f"\n{_SEP}\nTOP 10 BY AVG R PER TRADE (RISK-ADJUSTED)\n{_SEP}")
    print(_HDR)
    print(_SEP)
    for i, r in enumerate(by_avg[:10]):
        print(_row(i + 1, r))

    # ── Current config baseline ───────────────────────────────────────────────
    current_matches = [
        r
        for r in results
        if r.config.sl_mult == 0.5
        and r.config.tp_mult == 3.0
        and r.config.max_hold == 45
    ]
    print(f"\n{_SEP}\nCURRENT CONFIG BASELINE  (SL=0.5× TP=3.0× Hold=45)\n{_SEP}")
    if current_matches:
        c = current_matches[0]
        print(_HDR)
        print(_SEP)
        # Find its rank by PF
        pf_rank = next(
            i + 1
            for i, r in enumerate(by_pf)
            if r.config.sl_mult == 0.5
            and r.config.tp_mult == 3.0
            and r.config.max_hold == 45
        )
        print(_row(pf_rank, c) + f"  ← PF rank #{pf_rank}/{len(valid)}")
    else:
        print("  (Not in results — config not in grid)")

    # ── Recommendation ────────────────────────────────────────────────────────
    best_pf = by_pf[0]
    best_r = by_r[0]
    best_avg = by_avg[0]

    print(f"\n{'=' * 92}")
    print("RECOMMENDATION")
    print(f"{'=' * 92}")
    print(
        f"\n  Best Profit Factor : SL={best_pf.config.sl_mult}× TP={best_pf.config.tp_mult}× "
        f"Hold={best_pf.config.max_hold}  "
        f"(PF={best_pf.profit_factor:.3f}  WR={best_pf.win_rate:.1%}  TotalR={best_pf.total_r:+.1f})"
    )
    print(
        f"  Best Total R       : SL={best_r.config.sl_mult}× TP={best_r.config.tp_mult}× "
        f"Hold={best_r.config.max_hold}  "
        f"(PF={best_r.profit_factor:.3f}  WR={best_r.win_rate:.1%}  TotalR={best_r.total_r:+.1f})"
    )
    print(
        f"  Best Avg R/trade   : SL={best_avg.config.sl_mult}× TP={best_avg.config.tp_mult}× "
        f"Hold={best_avg.config.max_hold}  "
        f"(PF={best_avg.profit_factor:.3f}  WR={best_avg.win_rate:.1%}  AvgR={best_avg.avg_r:+.4f})"
    )

    # Check agreement
    candidates = {best_pf.config, best_r.config, best_avg.config}
    if len(candidates) == 1:
        cfg = best_pf.config
        print(
            f"\n  ALL THREE METRICS AGREE — lock in: "
            f"SL={cfg.sl_mult}× TP={cfg.tp_mult}× Hold={cfg.max_hold}"
        )
    elif best_pf.config == best_r.config:
        cfg = best_pf.config
        print(
            f"\n  PF and Total R agree — candidate: "
            f"SL={cfg.sl_mult}× TP={cfg.tp_mult}× Hold={cfg.max_hold}"
        )
        print(
            f"  Avg R disagrees (best avg: SL={best_avg.config.sl_mult}× "
            f"TP={best_avg.config.tp_mult}× Hold={best_avg.config.max_hold})"
        )
    else:
        print(f"\n  Metrics disagree — review Top 3 from each table before choosing.")

    print(f"\n{'=' * 92}")
    print("SYNC REMINDER: after choosing brackets, update these constants:")
    print(
        "  1. src/core/retrainer.py              SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER, MAX_HOLD_BARS"
    )
    print(
        "  2. src/evaluate_performance.py         SL_MULTIPLIER, TP_MULTIPLIER, MAX_HOLD_BARS"
    )
    print(
        "  3. src/execution/live_orchestrator.py  SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER"
    )
    print(
        "  4. src/analysis/failure_modes.py       SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER, MAX_HOLD_BARS"
    )
    print(f"{'=' * 92}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────


def main() -> int:
    try:
        n_configs = len(SL_MULTIPLIERS) * len(TP_MULTIPLIERS) * len(MAX_HOLD_OPTIONS)
        print("=" * 92)
        print("BRACKET OPTIMIZATION — VECTORIZED GRID SEARCH")
        print(
            f"SL: {SL_MULTIPLIERS}  |  TP: {TP_MULTIPLIERS}  |  Hold: {MAX_HOLD_OPTIONS}"
        )
        print(f"Total configurations: {n_configs}")
        print("=" * 92)

        bars_df = load_data()

        # Data coverage confirmation
        n_symbols = bars_df["symbol"].n_unique()
        symbols = sorted(bars_df["symbol"].unique().to_list())
        min_ts = bars_df["timestamp"].min()
        max_ts = bars_df["timestamp"].max()
        try:
            days_covered = (max_ts - min_ts).days
        except Exception:
            days_covered = "?"
        logger.info(
            f"Data coverage: {days_covered} days | {n_symbols} symbols: {symbols}"
        )
        logger.info(f"Total bars: {len(bars_df):,}")

        angel_model, devil_model = load_models()
        logger.info(
            f"Angel features ({len(angel_model.feature_names_in_)}): "
            f"{list(angel_model.feature_names_in_)}"
        )

        signals_df, featured_df = generate_signals(bars_df, angel_model, devil_model)
        logger.info(f"Generated {len(signals_df):,} approved BUY signals")

        signal_data = precompute_lookahead(signals_df, featured_df)
        if len(signal_data) < MIN_TRADES:
            raise ValueError(
                f"Only {len(signal_data)} valid signals (need >= {MIN_TRADES}). "
                "Not enough data for meaningful analysis."
            )

        results = run_grid_search(signal_data)
        print_results(results)
        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing data or models: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Analysis error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Grid search failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
