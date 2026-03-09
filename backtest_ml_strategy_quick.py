import sys
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List
import polars as pl
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from strategies.concrete_strategies.ml_strategy import MLStrategy
from core.signal import Signal

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("utils.bar_aggregator").setLevel(logging.WARNING)
logging.getLogger("strategies.concrete_strategies.ml_strategy").setLevel(logging.WARNING)

class BacktestOrderManager:
    def __init__(self, order_params, initial_capital):
        self.capital = initial_capital
        self.order_params = order_params
        self.active_orders = {}
        self.trades = []

    def place_order(self, signal, current_capital):
        risk_amount = current_capital * self.order_params.risk_percentage
        qty = risk_amount / signal.price
        stop_loss = signal.price * self.order_params.sl_multiplier
        take_profit = signal.price * self.order_params.tp_multiplier
        order_id = f"order_{len(self.trades)}_{signal.symbol}_{len(self.active_orders)}"
        self.active_orders[order_id] = {
            "symbol": signal.symbol,
            "entry_price": signal.price,
            "quantity": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": None,
        }
        return order_id

    def monitor_orders(self, market_data):
        for order_id, details in list(self.active_orders.items()):
            symbol = details["symbol"]
            if symbol not in market_data:
                continue
            bar = market_data[symbol]
            high = bar["high"]
            low = bar["low"]
            close = bar["close"]
            exit_price = None
            reason = ""
            if low <= details["stop_loss"]:
                exit_price = details["stop_loss"]
                reason = "SL"
            elif high >= details["take_profit"]:
                exit_price = details["take_profit"]
                reason = "TP"
            if exit_price:
                pnl = (exit_price - details["entry_price"]) * details["quantity"]
                self.capital += pnl
                self.trades.append({
                    "symbol": symbol,
                    "entry_price": details["entry_price"],
                    "exit_price": exit_price,
                    "qty": details["quantity"],
                    "pnl": pnl,
                    "reason": reason,
                    "time": bar["timestamp"],
                })
                del self.active_orders[order_id]

async def run_backtest():
    data_path = "data/raw/SPY_1min.parquet"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pl.read_parquet(data_path)
    df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    
    # Use only first 2 months of 2024 for quick test
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
    test_df = df.filter((pl.col("timestamp") >= start_date) & (pl.col("timestamp") < end_date))

    if test_df.is_empty():
        print("Error: No data found in date range.")
        return

    print(f"Quick Backtest: {len(test_df)} bars of SPY (Jan-Feb 2024)")

    strategy = MLStrategy(model_path="src/ml/models/rf_model.joblib", threshold=0.70)
    order_params = strategy.get_order_params()
    backtest_om = BacktestOrderManager(order_params, 10000.0)

    from utils.bar_aggregator import LiveBarAggregator
    lba = LiveBarAggregator(timeframe=1, history_size=400)
    symbol = "SPY"
    total_bars = len(test_df)

    print("Running...")
    for i, row in enumerate(test_df.iter_rows(named=True)):
        if i % 10000 == 0 and i > 0:
            print(f"  Progress: {i}/{total_bars} bars, {len(backtest_om.trades)} trades")
        backtest_om.monitor_orders({symbol: row})
        bar_data = {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
        is_new_bar = lba.add_bar(bar_data)
        if is_new_bar:
            history = lba.history_df
            if len(history) >= strategy.warmup_period:
                signals = strategy.analyze({symbol: history})
                for signal in signals:
                    if signal.type == "BUY":
                        if not any(d["symbol"] == symbol for d in backtest_om.active_orders.values()):
                            backtest_om.place_order(signal, backtest_om.capital)

    trades = backtest_om.trades
    print(f"\nCompleted: {total_bars} bars processed")

    if not trades:
        print("No trades executed. Try lowering the threshold.")
        return

    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(wins) / len(trades) if trades else 0

    print("\n" + "=" * 40)
    print("      ML STRATEGY BACKTEST (QUICK)")
    print("=" * 40)
    print(f"Period:          Jan-Feb 2024")
    print(f"Symbol:          {symbol}")
    print(f"Threshold:       {strategy.threshold}")
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital:   ${backtest_om.capital:.2f}")
    print(f"Total PnL:       ${total_pnl:.2f} ({total_pnl / 10000:.2%})")
    print("-" * 40)
    print(f"Total Trades:    {len(trades)}")
    print(f"Winning Trades:  {len(wins)}")
    print(f"Losing Trades:   {len(losses)}")
    print(f"Win Rate:        {win_rate:.2%}")
    print("-" * 40)
    print(f"Gross Profit:    ${gross_profit:.2f}")
    print(f"Gross Loss:      ${gross_loss:.2f}")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print("=" * 40)

    if profit_factor > 1.5:
        print("\n✅ Hypothesis Confirmed: Profit Factor > 1.5 (Sniper Model)")
        print("🚀 Ready for Paper Trading!")
    else:
        print(f"\n❌ Hypothesis Refuted: Profit Factor = {profit_factor:.2f} (need > 1.5)")

if __name__ == "__main__":
    asyncio.run(run_backtest())
