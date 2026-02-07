import sys
import os
import polars as pl
import numpy as np
import logging

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from strategies.concrete_strategies.rsi_bbands import RSIBBands

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_test():
    print("--- Starting Logic Isolation Test for RSIBBands ---")

    # 1. Initialize Strategy
    strategy = RSIBBands()
    symbol = "TEST_BTC"

    # 2. Generate Synthetic Data
    # We need enough data for BB (20) and RSI (14) to be valid.
    # We will construct a baseline and then inject the specific setup.

    base_price = 100.0
    dates = [f"2025-01-01 12:{i:02d}:00" for i in range(50)]

    # Create baseline prices (flat-ish)
    opens = [base_price] * 50
    highs = [base_price + 1] * 50
    lows = [base_price - 1] * 50
    closes = [base_price] * 50

    # --- INJECT SCENARIO ---

    # Step A: The Crash (Trigger Stage 1) - Row 46 (Index 46)
    # Price drops significantly to push RSI down and break lower BB
    opens[46] = 98.0
    closes[46] = 90.0  # Big drop
    highs[46] = 98.0
    lows[46] = 89.0

    # Step B: The Stabilization (Stage 2 Monitoring) - Row 47 (Index 47)
    # Price recovers slightly, RSI moves up but stays low.
    opens[47] = 90.0
    closes[47] = 92.0
    highs[47] = 92.5
    lows[47] = 89.5

    # Step C: The Setup (Previous Red Candle) - Row 48 (Index 48)
    opens[48] = 94.0
    closes[48] = 93.0  # Red candle
    highs[48] = 94.5
    lows[48] = 92.5

    # Step D: The Trigger (Bullish Engulfing) - Row 49 (Index 49)
    # Green candle that fully engulfs the previous red body (93.0 - 94.0)
    opens[49] = 92.5  # Open lower than previous close
    closes[49] = 94.5 # Close higher than previous open
    highs[49] = 95.0
    lows[49] = 92.0

    # Create DataFrame
    df = pl.DataFrame({
        "timestamp": dates,
        "open": np.array(opens, dtype=np.float64),
        "high": np.array(highs, dtype=np.float64),
        "low": np.array(lows, dtype=np.float64),
        "close": np.array(closes, dtype=np.float64),
        "volume": np.ones(50, dtype=np.float64) * 1000
    })

    # 3. Simulate "Live" Feed
    # We feed the data incrementally to simulate time passing
    # and verify the state machine works (Stage 1 -> Stage 2).

    print(f"\n[Test] Feeding data row by row to simulate live market...")

    # We start feeding from index 45 to ensure indicators are calculated
    for i in range(45, 50):
        # Slice data up to current point (simulate receiving new bar)
        current_data = {symbol: df.slice(0, i+1)}

        print(f"\nProcessing Row {i}...")
        signals = strategy.analyze(current_data)

        if signals:
            print(f"!!! SIGNAL RECEIVED at Row {i} !!!")
            for s in signals:
                print(f"  Type: {s.type} | Symbol: {s.symbol} | Price: {s.price}")

            # Assertion
            if i == 49 and signals[0].type == "BUY":
                print("\n✅ SUCCESS: Buy signal generated correctly on Bullish Engulfing!")
                return

    print("\n❌ FAILURE: No signal was generated.")


if __name__ == "__main__":
    run_test()
