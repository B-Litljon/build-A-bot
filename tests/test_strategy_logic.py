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


def build_synthetic_df(with_bullish_engulfing: bool) -> pl.DataFrame:
    dates = [f"2025-01-01 12:{i:02d}:00" for i in range(50)]

    opens = []
    highs = []
    lows = []
    closes = []

    for i in range(50):
        if i % 2 == 0:
            o, h, l, c = 100.0, 101.5, 99.5, 101.0
        else:
            o, h, l, c = 101.0, 102.0, 100.0, 100.0

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)

    # Controlled downtrend so RSI can reach <= 25 before the crash
    closes[40] = 99.0
    closes[41] = 98.0
    closes[42] = 97.0
    closes[43] = 96.0
    closes[44] = 95.0
    closes[45] = 94.0

    opens[40] = 100.0
    opens[41] = 99.0
    opens[42] = 98.0
    opens[43] = 97.0
    opens[44] = 96.0
    opens[45] = 95.0

    highs[40] = 100.5
    highs[41] = 99.5
    highs[42] = 98.5
    highs[43] = 97.5
    highs[44] = 96.5
    highs[45] = 95.5

    lows[40] = 98.5
    lows[41] = 97.5
    lows[42] = 96.5
    lows[43] = 95.5
    lows[44] = 94.5
    lows[45] = 93.5

    # --- INJECT SCENARIO ---

    # Step A: The Crash (Trigger Stage 1) - Row 46
    # Massive drop to expand bands and trigger oversold RSI
    opens[46] = 94.0
    closes[46] = 85.0  # Big red candle
    highs[46] = 100.0
    lows[46] = 84.0

    # Step B: The Stabilization (Stage 2 Monitoring) - Row 47
    # Price recovers slightly, RSI moves up.
    opens[47] = 85.0
    closes[47] = 87.0
    highs[47] = 87.5
    lows[47] = 84.5

    # Step C: The Setup (Previous Red Candle) - Row 48
    opens[48] = 87.0
    closes[48] = 86.0  # Red candle
    highs[48] = 87.5
    lows[48] = 85.5

    # Step D: Row 49
    if with_bullish_engulfing:
        # Green candle engulfing previous body (86.0 - 87.0)
        opens[49] = 85.5
        closes[49] = 88.0
        highs[49] = 88.5
        lows[49] = 85.0
    else:
        # Non-engulfing green candle (fails engulfing condition by design)
        opens[49] = 86.2
        closes[49] = 87.1
        highs[49] = 87.6
        lows[49] = 85.8

    return pl.DataFrame({
        "timestamp": dates,
        "open": np.array(opens, dtype=np.float64),
        "high": np.array(highs, dtype=np.float64),
        "low": np.array(lows, dtype=np.float64),
        "close": np.array(closes, dtype=np.float64),
        "volume": np.ones(50, dtype=np.float64) * 1000,
    })


def replay_and_collect(strategy: RSIBBands, symbol: str, df: pl.DataFrame):
    all_signals = []

    for i in range(45, 50):
        current_data = {symbol: df.slice(0, i + 1)}
        print(f"\nProcessing Row {i}...")
        signals = strategy.analyze(current_data)
        if signals:
            print(f"!!! SIGNAL RECEIVED at Row {i} !!!")
            for s in signals:
                print(f"  Type: {s.type} | Symbol: {s.symbol} | Price: {s.price}")

            all_signals.extend((i, s) for s in signals)

    return all_signals


def run_test():
    symbol = "TEST_BTC"

    print("--- Test 1: Positive Path (Bullish Engulfing should BUY) ---")
    strategy_positive = RSIBBands()
    df_positive = build_synthetic_df(with_bullish_engulfing=True)
    print("\n[Test] Feeding data row by row...")
    positive_signals = replay_and_collect(strategy_positive, symbol, df_positive)

    buy_at_49 = any(i == 49 and s.type == "BUY" for i, s in positive_signals)
    if buy_at_49:
        print("\n✅ SUCCESS: Buy signal generated correctly on Bullish Engulfing!")
    else:
        print("\n❌ FAILURE: Positive path did not generate expected BUY signal.")
        return

    print("\n--- Test 2: Negative Path (No Engulfing should NOT BUY) ---")
    strategy_negative = RSIBBands()
    df_negative = build_synthetic_df(with_bullish_engulfing=False)
    print("\n[Test] Feeding data row by row...")
    negative_signals = replay_and_collect(strategy_negative, symbol, df_negative)

    has_buy_signal = any(s.type == "BUY" for _, s in negative_signals)
    if not has_buy_signal:
        print("\n✅ SUCCESS: No BUY signal generated without Bullish Engulfing.")
        return

    print("\n❌ FAILURE: Negative path generated an unexpected BUY signal.")


if __name__ == "__main__":
    run_test()
