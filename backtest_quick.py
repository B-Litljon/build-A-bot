#!/usr/bin/env python3
"""Quick 2-week backtest of MLStrategy - silent version"""

import sys
import os
import logging

# Suppress ALL logging
logging.disable(logging.CRITICAL)
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.insert(0, os.path.abspath("src"))

import polars as pl
from datetime import datetime, timezone
from strategies.concrete_strategies.ml_strategy import MLStrategy
from core.signal import Signal
from utils.bar_aggregator import LiveBarAggregator

# Load data
df = pl.read_parquet("data/raw/SPY_1min.parquet")
df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

# Use only 2 weeks for very quick test
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end = datetime(2024, 1, 15, tzinfo=timezone.utc)
test_df = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))

print(f"Quick Test: {len(test_df)} bars (Jan 1-15, 2024)")

# Setup
strategy = MLStrategy(model_path="src/ml/models/rf_model.joblib", threshold=0.70)
order_params = strategy.get_order_params()


# Simple backtest manager
class BOM:
    def __init__(self, op, cap):
        self.capital = cap
        self.order_params = op
        self.active_orders = {}
        self.trades = []

    def place_order(self, sig, cap):
        risk = cap * self.order_params.risk_percentage
        qty = risk / sig.price
        sl = sig.price * self.order_params.sl_multiplier
        tp = sig.price * self.order_params.tp_multiplier
        oid = f"o{len(self.trades)}"
        self.active_orders[oid] = {
            "symbol": sig.symbol,
            "entry_price": sig.price,
            "quantity": qty,
            "stop_loss": sl,
            "take_profit": tp,
        }
        return oid

    def monitor_orders(self, md):
        for oid, det in list(self.active_orders.items()):
            sym = det["symbol"]
            if sym not in md:
                continue
            bar = md[sym]
            exit_price = None
            if bar["low"] <= det["stop_loss"]:
                exit_price = det["stop_loss"]
                reason = "SL"
            elif bar["high"] >= det["take_profit"]:
                exit_price = det["take_profit"]
                reason = "TP"
            if exit_price:
                pnl = (exit_price - det["entry_price"]) * det["quantity"]
                self.capital += pnl
                self.trades.append({"pnl": pnl, "reason": reason})
                del self.active_orders[oid]


bom = BOM(order_params, 10000.0)
lba = LiveBarAggregator(timeframe=1, history_size=240)
symbol = "SPY"

print("Running...")
for i, row in enumerate(test_df.iter_rows(named=True)):
    if i % 5000 == 0 and i > 0:
        print(f"  Progress: {i}/{len(test_df)} bars, {len(bom.trades)} trades")

    bom.monitor_orders({symbol: row})

    is_new = lba.add_bar(
        {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
    )

    if is_new:
        hist = lba.history_df
        if len(hist) >= strategy.warmup_period:
            sigs = strategy.analyze({symbol: hist})
            for sig in sigs:
                if sig.type == "BUY" and not any(
                    d["symbol"] == symbol for d in bom.active_orders.values()
                ):
                    bom.place_order(sig, bom.capital)

# Results
trades = bom.trades
print(f"\nDone! {len(test_df)} bars processed")

if not trades:
    print("No trades executed.")
else:
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(wins) / len(trades) if trades else 0

    print("\n" + "=" * 40)
    print("   ML STRATEGY BACKTEST (2 WEEKS)")
    print("=" * 40)
    print(f"Symbol: SPY | Threshold: 0.70")
    print(f"Initial: $10,000.00 | Final: ${bom.capital:.2f}")
    print(f"Total PnL: ${total_pnl:.2f} ({total_pnl / 10000:.2%})")
    print(f"Total Trades: {len(trades)} | Win Rate: {win_rate:.1%}")
    print(f"Winners: {len(wins)} | Losers: {len(losses)}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print("=" * 40)
    if profit_factor > 1.5:
        print("✅ HYPOTHESIS CONFIRMED: Profit Factor > 1.5")
        print("🚀 Ready for Paper Trading!")
    else:
        print(f"❌ Profit Factor = {profit_factor:.2f} (need > 1.5)")
