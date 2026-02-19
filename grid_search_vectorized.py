#!/usr/bin/env python3
"""Vectorized Grid Search - Fast Batch Processing"""

import sys, os, logging

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.abspath("src"))

import polars as pl
import numpy as np
import joblib
from datetime import datetime, timezone
from ml.feature_pipeline import FeatureEngineer

# Load data - Full Test Set (2024)
print("Loading data...")
df = pl.read_parquet("data/raw/SPY_1min.parquet")
df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
test_df = df.filter(pl.col("timestamp") >= start)

print(f"Data loaded: {len(test_df)} bars (2024 Full Year)")
print(f"Pre-computing features for all bars...")

# Pre-compute ALL features once
feature_engineer = FeatureEngineer()
features_df = feature_engineer.compute_indicators(test_df)

# Drop rows with NaN values
feature_names = [
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
features_df = features_df.drop_nulls(subset=feature_names)

print(f"Features computed: {len(features_df)} valid bars")

# Load model
model = joblib.load("src/ml/models/rf_model.joblib")

# Batch predict all probabilities
print("Running batch predictions...")
X = features_df[feature_names].to_numpy()
probabilities = model.predict_proba(X)[:, 1]
features_df = features_df.with_columns(pl.Series("prob", probabilities))

# Filter signals at threshold 0.50
signals_df = features_df.filter(pl.col("prob") >= 0.50)
print(f"Signals at threshold 0.50: {len(signals_df)} entries")

if len(signals_df) == 0:
    print("\n❌ No signals generated at threshold 0.50!")
    sys.exit(0)

# Risk configs
configs = [
    ("Config A (Scalper)", 0.998, 1.005),  # SL 0.2%, TP 0.5%
    ("Config B (Balanced)", 0.995, 1.005),  # SL 0.5%, TP 0.5%
    ("Config C (Swinger)", 0.995, 1.010),  # SL 0.5%, TP 1.0%
]

TIMEOUT_BARS = 15

print("\n" + "=" * 80)
print("Running Grid Search with pre-computed signals...")
print("=" * 80)

results = []

for config_name, sl_mult, tp_mult in configs:
    print(f"\n🔄 Testing {config_name}")
    print(f"   SL: {(1 - sl_mult) * 100:.1f}% | TP: {(tp_mult - 1) * 100:.1f}%")

    trades = []
    capital = 10000.0
    in_trade = False
    entry_price = 0
    entry_bar_idx = 0
    qty = 0
    sl_price = 0
    tp_price = 0

    # Get all bars as list for index-based access
    all_bars = list(test_df.iter_rows(named=True))
    signal_bars = set()
    signal_indices = {}

    # Create signal lookup by timestamp
    for row in signals_df.iter_rows(named=True):
        ts = row["timestamp"]
        signal_bars.add(ts)

    for i, bar in enumerate(all_bars):
        ts = bar["timestamp"]

        # Check if in trade
        if in_trade:
            exit_price = None
            reason = None

            # Check SL
            if bar["low"] <= sl_price:
                exit_price = sl_price
                reason = "SL"
            # Check TP
            elif bar["high"] >= tp_price:
                exit_price = tp_price
                reason = "TP"
            # Check timeout
            elif i - entry_bar_idx >= TIMEOUT_BARS:
                exit_price = bar["close"]
                reason = "TIMEOUT"

            if exit_price:
                pnl = (exit_price - entry_price) * qty
                capital += pnl
                trades.append({"pnl": pnl, "reason": reason})
                in_trade = False

        # Check for new signal (only if not in trade)
        if not in_trade and ts in signal_bars:
            price = bar["close"]
            risk = capital * 0.02
            qty = risk / price
            entry_price = price
            entry_bar_idx = i
            sl_price = price * sl_mult
            tp_price = price * tp_mult
            in_trade = True

    # Calculate metrics
    if trades:
        total_pnl = sum(t["pnl"] for t in trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        gp = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = gp / gl if gl > 0 else float("inf")
        wr = len(wins) / len(trades)
        tp_hits = len([t for t in trades if t["reason"] == "TP"])
        sl_hits = len([t for t in trades if t["reason"] == "SL"])
        timeouts = len([t for t in trades if t["reason"] == "TIMEOUT"])

        results.append(
            {
                "name": config_name,
                "trades": len(trades),
                "win_rate": wr,
                "profit_factor": pf,
                "pnl": total_pnl,
                "tp_hits": tp_hits,
                "sl_hits": sl_hits,
                "timeouts": timeouts,
            }
        )

        print(
            f"  ✅ {len(trades)} trades | WR: {wr:.1%} | PF: {pf:.2f} | PnL: ${total_pnl:.2f}"
        )
        print(f"     TP: {tp_hits} | SL: {sl_hits} | Timeout: {timeouts}")
    else:
        print(f"  ❌ No trades")
        results.append(
            {
                "name": config_name,
                "trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "pnl": 0,
                "tp_hits": 0,
                "sl_hits": 0,
                "timeouts": 0,
            }
        )

# Summary Table
print("\n" + "=" * 80)
print("          GRID SEARCH RESULTS (Threshold 0.50 - 2024 Test Set)")
print("=" * 80)
print(
    f"{'Configuration':<20} {'Trades':>8} {'Win Rate':>10} {'P&F':>8} {'PnL':>12} {'Status':<10}"
)
print("-" * 80)

best_config = None
for r in results:
    status = "✅ PASS" if r["trades"] >= 50 and r["profit_factor"] > 1.2 else "❌ FAIL"
    print(
        f"{r['name']:<20} {r['trades']:>8} {r['win_rate']:>9.1%} {r['profit_factor']:>8.2f} ${r['pnl']:>10.2f} {status:<10}"
    )
    if r["trades"] >= 50 and r["profit_factor"] > 1.2:
        if best_config is None or r["profit_factor"] > best_config["profit_factor"]:
            best_config = r

print("=" * 80)
print(f"\nSuccess Criteria: Profit Factor > 1.2 AND Trades >= 50")
print("=" * 80)

if best_config:
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Profile: {best_config['name']}")
    sl_pct = (1 - (0.998 if "Scalper" in best_config["name"] else 0.995)) * 100
    tp_pct = (
        1.005
        if "Scalper" in best_config["name"] or "Balanced" in best_config["name"]
        else 1.010
    ) - 1
    tp_pct *= 100
    print(f"   Trades: {best_config['trades']}")
    print(f"   Win Rate: {best_config['win_rate']:.1%}")
    print(f"   Profit Factor: {best_config['profit_factor']:.2f}")
    print(f"   Total PnL: ${best_config['pnl']:.2f}")
    print(
        f"   TP Hits: {best_config['tp_hits']} | SL Hits: {best_config['sl_hits']} | Timeouts: {best_config['timeouts']}"
    )
    print(f"\n   ✅ CONFIGURATION ACHIEVES PROFIT FACTOR > 1.2!")
    print(
        f"   🚀 RECOMMENDATION: Proceed with paper trading using {best_config['name']}"
    )
else:
    print("\n   ⚠️  NO CONFIGURATION MET SUCCESS CRITERIA")
    print("\n   📊 Detailed Results:")
    for r in results:
        if r["trades"] > 0:
            print(
                f"      {r['name']}: {r['trades']} trades, PF={r['profit_factor']:.2f}"
            )
            print(
                f"         SL hits: {r['sl_hits']}, TP hits: {r['tp_hits']}, Timeouts: {r['timeouts']}"
            )

            if r["sl_hits"] > r["tp_hits"]:
                print(f"         → Issue: More SL hits than TP hits")
            if r["timeouts"] > r["trades"] * 0.3:
                print(f"         → Issue: High timeout rate")

    print("\n   🔧 Recommendations:")
    print("      1. Try wider TP/SL ratios (e.g., TP 1.5% / SL 0.5% for 3:1 R:R)")
    print("      2. Extend timeout period beyond 15 bars")
    print("      3. Consider retraining model with different target criteria")

print("=" * 80)
