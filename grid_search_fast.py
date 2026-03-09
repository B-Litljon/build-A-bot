#!/usr/bin/env python3
"""Fast Grid Search - 2 Weeks Data"""
import sys, os, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.abspath("src"))

import polars as pl
import numpy as np
from datetime import datetime, timezone
from strategies.concrete_strategies.ml_strategy import MLStrategy
from utils.bar_aggregator import LiveBarAggregator

# Load just 2 weeks of data
df = pl.read_parquet("data/raw/SPY_1min.parquet")
df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end = datetime(2024, 1, 15, tzinfo=timezone.utc)
test_df = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))

print(f"Fast Grid Search: {len(test_df)} bars (Jan 1-15, 2024)")
print(f"Threshold: 0.50 | Timeout: 15 bars")
print("=" * 80)

# Configs
configs = [
    ("A (Scalper)", 0.998, 1.005),   # SL 0.2%, TP 0.5%
    ("B (Balanced)", 0.995, 1.005),  # SL 0.5%, TP 0.5%
    ("C (Swinger)", 0.995, 1.010),   # SL 0.5%, TP 1.0%
]

results = []
TIMEOUT_BARS = 15

for name, sl_mult, tp_mult in configs:
    print(f"\n🔄 Testing Config {name}")
    
    strategy = MLStrategy(model_path="src/ml/models/rf_model.joblib", threshold=0.50)
    lba = LiveBarAggregator(timeframe=1, history_size=400)
    
    capital = 10000.0
    active_order = None
    trades = []
    symbol = "SPY"
    
    for i, row in enumerate(test_df.iter_rows(named=True)):
        # Monitor active order
        if active_order:
            exit_price = None
            reason = None
            
            if row["low"] <= active_order["sl"]:
                exit_price = active_order["sl"]
                reason = "SL"
            elif row["high"] >= active_order["tp"]:
                exit_price = active_order["tp"]
                reason = "TP"
            elif i - active_order["entry_idx"] >= TIMEOUT_BARS:
                exit_price = row["close"]
                reason = "TIMEOUT"
            
            if exit_price:
                pnl = (exit_price - active_order["entry"]) * active_order["qty"]
                capital += pnl
                trades.append({"pnl": pnl, "reason": reason})
                active_order = None
        
        # Add bar to aggregator
        is_new = lba.add_bar({
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        })
        
        # Check for signals on new bar
        if is_new and not active_order:
            hist = lba.history_df
            if len(hist) >= strategy.warmup_period:
                sigs = strategy.analyze({symbol: hist})
                for sig in sigs:
                    if sig.type == "BUY":
                        risk = capital * 0.02
                        qty = risk / sig.price
                        active_order = {
                            "entry": sig.price,
                            "qty": qty,
                            "sl": sig.price * sl_mult,
                            "tp": sig.price * tp_mult,
                            "entry_idx": i,
                        }
                        break
    
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
        
        results.append({
            "name": name,
            "trades": len(trades),
            "wr": wr,
            "pf": pf,
            "pnl": total_pnl,
            "tp": tp_hits,
            "sl": sl_hits,
            "to": timeouts,
        })
        
        print(f"  ✅ {len(trades)} trades | WR: {wr:.1%} | PF: {pf:.2f} | PnL: ${total_pnl:.2f}")
        print(f"     TP: {tp_hits} | SL: {sl_hits} | TO: {timeouts}")
    else:
        print(f"  ❌ No trades")
        results.append({"name": name, "trades": 0, "wr": 0, "pf": 0, "pnl": 0, "tp": 0, "sl": 0, "to": 0})

# Print table
print("\n" + "=" * 80)
print("          GRID SEARCH RESULTS (Threshold 0.50)")
print("=" * 80)
print(f"{'Config':<15} {'Trades':>8} {'Win Rate':>10} {'P&F':>8} {'PnL':>12} {'Status':<10}")
print("-" * 80)

for r in results:
    status = "✅ PASS" if r["trades"] >= 10 and r["pf"] > 1.2 else "❌ FAIL"
    print(f"{r['name']:<15} {r['trades']:>8} {r['wr']:>9.1%} {r['pf']:>8.2f} ${r['pnl']:>10.2f} {status:<10}")

print("=" * 80)
print("Success Criteria: PF > 1.2, Trades >= 10 (scaled for 2-week period)")
print("=" * 80)

# Best config
best = max(results, key=lambda x: x["pf"] if x["trades"] >= 10 else 0)
if best["trades"] >= 10 and best["pf"] > 1.2:
    print(f"\n🏆 BEST: Config {best['name']}")
    print(f"   PF: {best['pf']:.2f} | Trades: {best['trades']} | PnL: ${best['pnl']:.2f}")
else:
    print("\n⚠️  No config met success criteria")
    for r in results:
        if r["trades"] > 0:
            print(f"   {r['name']}: {r['sl']} SL hits, {r['tp']} TP hits")
