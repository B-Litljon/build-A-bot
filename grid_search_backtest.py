#!/usr/bin/env python3
"""Grid Search Backtest - Threshold 0.50 with 3 Risk Profiles (A/B/C)"""

import sys, os, logging

logging.disable(logging.CRITICAL)
for name in logging.Logger.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).propagate = False
sys.path.insert(0, os.path.abspath("src"))

import polars as pl
from datetime import datetime, timezone
from strategies.concrete_strategies.ml_strategy import MLStrategy
from utils.bar_aggregator import LiveBarAggregator
from core.order_management import OrderParams

# Load data - Test Set (2024 onwards)
df = pl.read_parquet("data/raw/SPY_1min.parquet")
df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
test_df = df.filter(pl.col("timestamp") >= start)

print(f"Grid Search: {len(test_df)} bars (2024 Full Year - Test Set)")
print(f"Threshold: 0.50 (Model Native Optimization Point)")
print("=" * 70)

# Risk Profiles to test (as specified by user)
risk_profiles = [
    ("Config A (Scalper)", 0.998, 1.005),  # SL 0.2%, TP 0.5%
    ("Config B (Balanced)", 0.995, 1.005),  # SL 0.5%, TP 0.5%
    ("Config C (Swinger)", 0.995, 1.010),  # SL 0.5%, TP 1.0%
]

results = []
TIMEOUT_BARS = 15

for profile_name, sl_mult, tp_mult in risk_profiles:
    print(f"\n🔄 Testing {profile_name}")
    print(
        f"   SL Multiplier: {sl_mult} ({(1 - sl_mult) * 100:.1f}%) | TP Multiplier: {tp_mult} ({(tp_mult - 1) * 100:.1f}%)"
    )

    # Create strategy with threshold 0.50
    strategy = MLStrategy(model_path="src/ml/models/rf_model.joblib", threshold=0.50)

    order_params = OrderParams(
        risk_percentage=0.02,
        tp_multiplier=tp_mult,
        sl_multiplier=sl_mult,
        use_trailing_stop=False,
    )

    class BOM:
        def __init__(self, op, cap):
            self.capital = cap
            self.order_params = op
            self.active_orders = {}
            self.trades = []

        def place_order(self, sig, cap, bar_idx):
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
                "entry_bar": bar_idx,
            }
            return oid

        def monitor_orders(self, md, bar_idx):
            for oid, det in list(self.active_orders.items()):
                sym = det["symbol"]
                if sym not in md:
                    continue
                bar = md[sym]
                exit_price = None
                reason = None

                # Check SL
                if bar["low"] <= det["stop_loss"]:
                    exit_price = det["stop_loss"]
                    reason = "SL"
                # Check TP
                elif bar["high"] >= det["take_profit"]:
                    exit_price = det["take_profit"]
                    reason = "TP"
                # Check Timeout
                elif bar_idx - det["entry_bar"] >= TIMEOUT_BARS:
                    exit_price = bar["close"]
                    reason = "TIMEOUT"

                if exit_price:
                    pnl = (exit_price - det["entry_price"]) * det["quantity"]
                    self.capital += pnl
                    self.trades.append({"pnl": pnl, "reason": reason})
                    del self.active_orders[oid]

    bom = BOM(order_params, 10000.0)
    lba = LiveBarAggregator(timeframe=1, history_size=240)
    symbol = "SPY"

    for i, row in enumerate(test_df.iter_rows(named=True)):
        if i % 100000 == 0 and i > 0:
            print(
                f"  {i // 1000}k/{len(test_df) // 1000}k bars, {len(bom.trades)} trades"
            )

        bom.monitor_orders({symbol: row}, i)

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
                        bom.place_order(sig, bom.capital, i)

    # Calculate results
    trades = bom.trades
    if trades:
        total = sum(t["pnl"] for t in trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        tp_hits = len([t for t in trades if t.get("reason") == "TP"])
        sl_hits = len([t for t in trades if t.get("reason") == "SL"])
        timeout_exits = len([t for t in trades if t.get("reason") == "TIMEOUT"])
        gp = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = gp / gl if gl > 0 else float("inf")
        wr = len(wins) / len(trades)

        results.append(
            {
                "profile": profile_name,
                "sl": sl_mult,
                "tp": tp_mult,
                "trades": len(trades),
                "win_rate": wr,
                "profit_factor": pf,
                "pnl": total,
                "tp_hits": tp_hits,
                "sl_hits": sl_hits,
                "timeout_exits": timeout_exits,
            }
        )

        print(
            f"  ✅ {len(trades)} trades | WR: {wr:.1%} | PF: {pf:.2f} | PnL: ${total:.2f}"
        )
        print(f"     TP: {tp_hits} | SL: {sl_hits} | Timeout: {timeout_exits}")
    else:
        print(f"  ❌ No trades")
        results.append(
            {
                "profile": profile_name,
                "sl": sl_mult,
                "tp": tp_mult,
                "trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "pnl": 0,
                "tp_hits": 0,
                "sl_hits": 0,
                "timeout_exits": 0,
            }
        )

# Summary Table
print("\n" + "=" * 80)
print("          GRID SEARCH RESULTS (Threshold 0.50 - Test Set)")
print("=" * 80)
print(
    f"{'Configuration':<20} {'Trades':>8} {'Win Rate':>10} {'P&F':>8} {'PnL':>12} {'Status':<15}"
)
print("-" * 80)

best_config = None
for r in results:
    status = "✅ PASS" if r["trades"] >= 50 and r["profit_factor"] > 1.2 else "❌ FAIL"
    print(
        f"{r['profile']:<20} {r['trades']:>8} {r['win_rate']:>9.1%} {r['profit_factor']:>8.2f} ${r['pnl']:>10.2f} {status:<15}"
    )
    if r["trades"] >= 50 and r["profit_factor"] > 1.2:
        if best_config is None or r["profit_factor"] > best_config["profit_factor"]:
            best_config = r

print("=" * 80)
print(f"\nSuccess Criteria: Profit Factor > 1.2 AND Trades > 50")
print("=" * 80)

if best_config:
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Profile: {best_config['profile']}")
    print(
        f"   SL: {(1 - best_config['sl']) * 100:.1f}% | TP: {(best_config['tp'] - 1) * 100:.1f}%"
    )
    print(f"   Trades: {best_config['trades']}")
    print(f"   Win Rate: {best_config['win_rate']:.1%}")
    print(f"   Profit Factor: {best_config['profit_factor']:.2f}")
    print(f"   Total PnL: ${best_config['pnl']:.2f}")
    print(
        f"   TP: {best_config['tp_hits']} | SL: {best_config['sl_hits']} | Timeout: {best_config['timeout_exits']}"
    )
    print(f"\n   ✅ READY FOR PAPER TRADING!")
else:
    print("\n   ⚠️  No configuration met success criteria")
    print(f"\n   📊 Detailed Breakdown:")
    for r in results:
        print(
            f"      {r['profile']}: PF={r['profit_factor']:.2f}, Trades={r['trades']}"
        )

print("=" * 80)
