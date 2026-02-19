import sys
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List
import polars as pl
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from strategies.concrete_strategies.ml_strategy import MLStrategy
from core.signal import Signal

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress noise from bar_aggregator and ml_strategy
logging.getLogger("utils.bar_aggregator").setLevel(logging.WARNING)
logging.getLogger("strategies.concrete_strategies.ml_strategy").setLevel(
    logging.WARNING
)


class BacktestOrderManager:
    def __init__(self, order_params, initial_capital):
        self.capital = initial_capital
        self.order_params = order_params
        self.active_orders = {}
        self.trades = []  # List of (entry_price, exit_price, qty, pnl)

    def place_order(self, signal, current_capital):
        # Using 2% risk per trade as defined in MLStrategy or custom
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
            "entry_time": None,  # Could track time if needed
        }
        return order_id

    def monitor_orders(self, market_data):
        for order_id, details in list(self.active_orders.items()):
            symbol = details["symbol"]
            if symbol not in market_data:
                continue

            # For backtesting, we check if the High/Low of the bar hit our SL/TP
            # But here market_data is just the Close.
            # To be more accurate, we should use High/Low if available.

            bar = market_data[symbol]  # This is the full row/dict
            high = bar["high"]
            low = bar["low"]
            close = bar["close"]

            exit_price = None
            reason = ""

            # Check Stop Loss first (conservative)
            if low <= details["stop_loss"]:
                exit_price = details["stop_loss"]
                reason = "SL"
            # Check Take Profit
            elif high >= details["take_profit"]:
                exit_price = details["take_profit"]
                reason = "TP"

            if exit_price:
                pnl = (exit_price - details["entry_price"]) * details["quantity"]
                self.capital += pnl
                self.trades.append(
                    {
                        "symbol": symbol,
                        "entry_price": details["entry_price"],
                        "exit_price": exit_price,
                        "qty": details["quantity"],
                        "pnl": pnl,
                        "reason": reason,
                        "time": bar["timestamp"],
                    }
                )
                del self.active_orders[order_id]


async def run_backtest():
    # 1. Load Data
    data_path = "data/raw/SPY_1min.parquet"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pl.read_parquet(data_path)

    # Ensure timestamp is datetime and TZ aware for comparison
    df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

    split_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    test_df = df.filter(pl.col("timestamp") >= split_date)

    if test_df.is_empty():
        print("Error: No data found after split date.")
        return

    print(
        f"Backtesting MLStrategy on {len(test_df)} bars of SPY (starting {split_date.date()})"
    )

    # 2. Setup Strategy
    # We use the optimized threshold from training if possible, or 0.60 for 'Sniper'
    strategy = MLStrategy(model_path="src/ml/models/rf_model.joblib", threshold=0.70)
    order_params = strategy.get_order_params()

    backtest_om = BacktestOrderManager(order_params, 10000.0)

    # 3. Simulation Loop
    from utils.bar_aggregator import LiveBarAggregator

    lba = LiveBarAggregator(timeframe=1, history_size=240)

    symbol = "SPY"

    # To speed up, we'll use a progress indicator every 10,000 bars
    total_bars = len(test_df)

    print("Starting simulation...")

    for i, row in enumerate(test_df.iter_rows(named=True)):
        if i % 10000 == 0 and i > 0:
            print(
                f"Processed {i}/{total_bars} bars... Trades so far: {len(backtest_om.trades)}"
            )

        # 1. Check Exits (using High/Low for more accuracy)
        backtest_om.monitor_orders({symbol: row})

        # 2. Add Bar to Aggregator
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
                # strategy.analyze expects Dict[symbol, DataFrame]
                signals = strategy.analyze({symbol: history})
                for signal in signals:
                    if signal.type == "BUY":
                        # Only enter if not already in a position for this symbol
                        if not any(
                            d["symbol"] == symbol
                            for d in backtest_om.active_orders.values()
                        ):
                            backtest_om.place_order(signal, backtest_om.capital)

    # 4. Results
    trades = backtest_om.trades
    print(f"\nSimulation Finished. Processed {total_bars} bars.")

    if not trades:
        print("No trades executed. Try lowering the threshold.")
        return

    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]  # strictly less than 0
    breakeven = [t for t in trades if t["pnl"] == 0]

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(wins) / len(trades) if trades else 0

    print("\n" + "=" * 30)
    print("   ML STRATEGY BACKTEST   ")
    print("=" * 30)
    print(f"Symbol:          {symbol}")
    print(f"Threshold:       {strategy.threshold}")
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital:   ${backtest_om.capital:.2f}")
    print(f"Total PnL:       ${total_pnl:.2f} ({total_pnl / 10000:.2%})")
    print("-" * 30)
    print(f"Total Trades:    {len(trades)}")
    print(f"Winning Trades:  {len(wins)}")
    print(f"Losing Trades:   {len(losses)}")
    print(f"Win Rate:        {win_rate:.2%}")
    print("-" * 30)
    print(f"Gross Profit:    ${gross_profit:.2f}")
    print(f"Gross Loss:      ${gross_loss:.2f}")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print("=" * 30)

    # Hypothesis check
    if profit_factor > 1.5:
        print("\n✅ Hypothesis Confirmed: Profit Factor > 1.5 (Sniper Model)")
    else:
        print("\n❌ Hypothesis Refuted: Profit Factor <= 1.5")


if __name__ == "__main__":
    asyncio.run(run_backtest())
