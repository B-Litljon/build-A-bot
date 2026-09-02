"""
Dual Simple Moving Average (SMA) crossover strategy.

Glossary:
    fast_period -- lookback period for the fast moving average (default 20).
    slow_period -- lookback period for the slow moving average (default 50).
    atr_period -- lookback period for ATR stop-loss distance calculation (default 14).
    bullish cross -- fast SMA crossing above slow SMA from previous bar to current bar.
    bearish cross -- fast SMA crossing below slow SMA from previous bar to current bar.
    warmup_period -- minimum number of bars required before generating signals.
"""

from typing import Any, Optional
import numpy as np
import polars as pl
import talib

from strategies.base import BaseStrategy, Signal


class SMACrossoverStrategy(BaseStrategy):
    """
    Standard dual moving average crossover strategy.
    
    Generates a 'long' signal when the fast SMA crosses above the slow SMA,
    and a 'short' signal when the fast SMA crosses below the slow SMA.
    Uses ATR for stop-loss distance so the RiskManager can size brackets dynamically.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        atr_period: int = 14,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            atr_period=atr_period,
            **kwargs,
        )
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be strictly less than slow_period ({slow_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.warmup_period = slow_period + atr_period + 2

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        self.validate_input(df)

        if len(df) < self.warmup_period:
            return None

        close: np.ndarray = df["close"].to_numpy().astype(float)
        high: np.ndarray = df["high"].to_numpy().astype(float)
        low: np.ndarray = df["low"].to_numpy().astype(float)

        fast_sma = talib.SMA(close, timeperiod=self.fast_period)
        slow_sma = talib.SMA(close, timeperiod=self.slow_period)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)

        if (
            np.isnan(fast_sma[-1])
            or np.isnan(fast_sma[-2])
            or np.isnan(slow_sma[-1])
            or np.isnan(slow_sma[-2])
            or np.isnan(atr[-1])
            or atr[-1] <= 0.0
        ):
            return None

        prev_fast, curr_fast = fast_sma[-2], fast_sma[-1]
        prev_slow, curr_slow = slow_sma[-2], slow_sma[-1]

        direction: Optional[str] = None
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            direction = "long"
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            direction = "short"

        if direction is None:
            return None

        current_price = float(close[-1])
        raw_sl_distance = float(atr[-1])
        symbol = str(df["symbol"].tail(1)[0]) if "symbol" in df.columns else None
        bar_ts = df["timestamp"].tail(1)[0] if "timestamp" in df.columns else None

        return Signal(
            direction=direction,
            entry_price=current_price,
            raw_sl_distance=raw_sl_distance,
            raw_tp_distance=raw_sl_distance,
            metadata={
                "strategy": self.name,
                "symbol": symbol,
                "timestamp": bar_ts,
                "fast_sma": float(curr_fast),
                "slow_sma": float(curr_slow),
                "atr": raw_sl_distance,
            },
        )
