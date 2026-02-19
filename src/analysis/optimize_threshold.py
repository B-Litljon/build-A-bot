#!/usr/bin/env python3
"""
Threshold Optimization Analysis

Analyzes probability distribution and finds optimal operational threshold
for the Random Forest model on imbalanced data.

Usage:
    python src/analysis/optimize_threshold.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import polars as pl
import joblib
from sklearn.metrics import precision_score, recall_score

# Disable verbose logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "training_data.parquet"
MODEL_PATH = PROJECT_ROOT / "src" / "ml" / "models" / "rf_model.joblib"

# Test set split date
SPLIT_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)

# Feature columns (same as training)
FEATURE_COLS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "sma_50",
    "atr_14",
    "bb_pct_b",
    "price_sma50_ratio",
    "log_return",
    "hour_of_day",
    "vol_rel",
    "dist_sma50",
]

# Risk settings (Scalper Profile)
TP_MULT = 1.005  # 0.5%
SL_MULT = 0.998  # 0.2%
TIMEOUT_BARS = 15

# Thresholds to test
THRESHOLDS = [0.30, 0.35, 0.40, 0.42, 0.45, 0.48, 0.50]


def load_data_and_model():
    """Load training data and trained model."""
    print("Loading data...")
    df = pl.read_parquet(DATA_PATH)

    # Temporal split
    train_df = df.filter(pl.col("timestamp") < SPLIT_DATE)
    test_df = df.filter(pl.col("timestamp") >= SPLIT_DATE)

    print(f"  Train: {len(train_df)} rows")
    print(f"  Test:  {len(test_df)} rows")

    # Load model
    print("\nLoading model...")
    model = joblib.load(MODEL_PATH)
    print(f"  Model: {type(model).__name__}")

    return train_df, test_df, model


def analyze_distribution(test_df, model):
    """Analyze probability distribution and base rate."""
    print("\n" + "=" * 70)
    print("STEP 1: DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Prepare features
    X_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df["target"].to_numpy().astype(np.int8)

    # Base rate
    base_rate = (y_test == 1).sum() / len(y_test)
    print(f"\nBase Rate (Class 1): {base_rate:.2%}")
    print(f"  - Total samples: {len(y_test):,}")
    print(f"  - Class 1: {(y_test == 1).sum():,}")
    print(f"  - Class 0: {(y_test == 0).sum():,}")

    # Generate probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Quantiles
    quantiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    print(f"\nProbability Quantiles (Test Set):")
    print("-" * 40)
    for q in quantiles:
        val = np.percentile(y_proba, q)
        print(f"  {q:>5.1f}th percentile: {val:.4f}")

    print(f"\n  Min probability: {y_proba.min():.4f}")
    print(f"  Max probability: {y_proba.max():.4f}")
    print(f"  Mean probability: {y_proba.mean():.4f}")
    print(f"  Std probability: {y_proba.std():.4f}")

    return X_test, y_test, y_proba


def vectorized_backtest(test_df, y_proba, threshold):
    """Run vectorized backtest for a single threshold."""
    # Get signals (bars where probability >= threshold)
    signals = y_proba >= threshold
    signal_indices = np.where(signals)[0]

    if len(signal_indices) == 0:
        return {
            "threshold": threshold,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "pnl": 0.0,
            "tp_hits": 0,
            "sl_hits": 0,
            "timeouts": 0,
        }

    # Load price data
    closes = test_df["close"].to_numpy()
    highs = test_df["high"].to_numpy()
    lows = test_df["low"].to_numpy()

    trades = []
    capital = 10000.0

    for idx in signal_indices:
        if idx >= len(closes) - 1:
            continue

        entry_price = closes[idx]
        qty = (capital * 0.02) / entry_price
        sl_price = entry_price * SL_MULT
        tp_price = entry_price * TP_MULT

        # Look ahead for exit
        exit_found = False
        for j in range(1, TIMEOUT_BARS + 1):
            if idx + j >= len(closes):
                break

            bar_high = highs[idx + j]
            bar_low = lows[idx + j]
            bar_close = closes[idx + j]

            # Check SL
            if bar_low <= sl_price:
                pnl = (sl_price - entry_price) * qty
                trades.append({"pnl": pnl, "reason": "SL"})
                exit_found = True
                break
            # Check TP
            elif bar_high >= tp_price:
                pnl = (tp_price - entry_price) * qty
                trades.append({"pnl": pnl, "reason": "TP"})
                exit_found = True
                break

        # Timeout exit
        if not exit_found and idx + TIMEOUT_BARS < len(closes):
            exit_price = closes[idx + TIMEOUT_BARS]
            pnl = (exit_price - entry_price) * qty
            trades.append({"pnl": pnl, "reason": "TIMEOUT"})

    # Calculate metrics
    if not trades:
        return {
            "threshold": threshold,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "pnl": 0.0,
            "tp_hits": 0,
            "sl_hits": 0,
            "timeouts": 0,
        }

    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]

    gp = sum(t["pnl"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl"] for t in losses)) if losses else 0

    pf = gp / gl if gl > 0 else float("inf")
    wr = len(wins) / len(trades) if trades else 0

    tp_hits = len([t for t in trades if t["reason"] == "TP"])
    sl_hits = len([t for t in trades if t["reason"] == "SL"])
    timeouts = len([t for t in trades if t["reason"] == "TIMEOUT"])

    return {
        "threshold": threshold,
        "trades": len(trades),
        "win_rate": wr,
        "profit_factor": pf,
        "pnl": total_pnl,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "timeouts": timeouts,
    }


def run_threshold_sweep(test_df, y_proba):
    """Run backtest sweep across all thresholds."""
    print("\n" + "=" * 70)
    print("STEP 2: THRESHOLD SWEEP (Scalper Profile: TP 0.5% / SL 0.2%)")
    print("=" * 70)

    print(f"\nTesting {len(THRESHOLDS)} thresholds...")
    print("-" * 70)

    results = []
    for threshold in THRESHOLDS:
        result = vectorized_backtest(test_df, y_proba, threshold)
        results.append(result)
        status = (
            "✓" if result["trades"] >= 50 and result["profit_factor"] > 1.1 else " "
        )
        print(
            f"  Threshold {threshold:.2f}: {result['trades']:>4} trades, "
            f"PF={result['profit_factor']:.2f}, WR={result['win_rate']:.1%} {status}"
        )

    return results


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 90)
    print(
        f"{'Threshold':>10} {'Trades':>8} {'Win Rate':>10} {'P&F':>8} "
        f"{'PnL $':>10} {'TP':>6} {'SL':>6} {'TO':>5} {'Status':>8}"
    )
    print("-" * 90)

    best_config = None

    for r in results:
        meets_criteria = r["trades"] >= 50 and r["profit_factor"] > 1.1
        status = "✅ PASS" if meets_criteria else "❌ FAIL"

        print(
            f"{r['threshold']:>10.2f} {r['trades']:>8} {r['win_rate']:>9.1%} "
            f"{r['profit_factor']:>8.2f} {r['pnl']:>10.2f} "
            f"{r['tp_hits']:>6} {r['sl_hits']:>6} {r['timeouts']:>5} {status:>8}"
        )

        if meets_criteria:
            if best_config is None or r["profit_factor"] > best_config["profit_factor"]:
                best_config = r

    print("=" * 90)
    print(f"Success Criteria: Trades >= 50 AND Profit Factor > 1.1")
    print("=" * 90)

    return best_config


def print_recommendation(best_config, results):
    """Print final recommendation."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if best_config:
        print(f"\n🏆 OPTIMAL THRESHOLD: {best_config['threshold']:.2f}")
        print(f"\n   Metrics:")
        print(f"     - Trades: {best_config['trades']}")
        print(f"     - Win Rate: {best_config['win_rate']:.1%}")
        print(f"     - Profit Factor: {best_config['profit_factor']:.2f}")
        print(f"     - Total PnL: ${best_config['pnl']:.2f}")
        print(f"     - TP Hits: {best_config['tp_hits']}")
        print(f"     - SL Hits: {best_config['sl_hits']}")
        print(f"     - Timeouts: {best_config['timeouts']}")
        print(f"\n   ✅ This threshold meets both criteria (>50 trades, PF > 1.1)")
        print(f"   🚀 RECOMMENDED FOR PAPER TRADING")
    else:
        print("\n   ⚠️  No threshold met success criteria")

        # Find best attempt
        best_pf = max(
            results, key=lambda x: x["profit_factor"] if x["trades"] > 0 else 0
        )
        best_vol = max(results, key=lambda x: x["trades"])

        print(
            f"\n   Best Profit Factor: {best_pf['threshold']:.2f} "
            f"(PF={best_pf['profit_factor']:.2f}, {best_pf['trades']} trades)"
        )
        print(
            f"   Best Volume: {best_vol['threshold']:.2f} "
            f"({best_vol['trades']} trades, PF={best_vol['profit_factor']:.2f})"
        )

        print("\n   🔧 Recommendations:")
        print("      1. Lower minimum threshold below 0.30")
        print("      2. Adjust risk profile (wider SL or higher TP)")
        print("      3. Retrain model with different class weights")

    print("=" * 70)


def main():
    """Main execution."""
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print(f"Data: {DATA_PATH.name}")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Test Period: 2024-01-01 onwards")
    print(f"Risk Profile: Scalper (TP 0.5%, SL 0.2%, Timeout 15 bars)")
    print("=" * 70)

    # Load data
    train_df, test_df, model = load_data_and_model()

    # Analyze distribution
    X_test, y_test, y_proba = analyze_distribution(test_df, model)

    # Run sweep
    results = run_threshold_sweep(test_df, y_proba)

    # Print table
    best_config = print_results_table(results)

    # Recommendation
    print_recommendation(best_config, results)


if __name__ == "__main__":
    main()
