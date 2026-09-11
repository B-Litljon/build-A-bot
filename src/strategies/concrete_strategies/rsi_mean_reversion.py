"""
RSI Mean-Reversion strategy.

Glossary:
    rsi_period -- lookback period for RSI calculation (default 14).
    oversold -- threshold below which price is considered oversold (default 30.0).
    overbought -- threshold above which price is considered overbought (default 70.0).
    atr_period -- lookback period for ATR stop-loss distance calculation (default 14).
    recovery cross -- RSI crossing back above oversold (long) or below overbought (short).
    warmup_period -- minimum number of bars required before generating signals.
"""

from typing import Any, Optional
import numpy as np
import polars as pl
import talib

from strategies.base import BaseStrategy, Signal


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI mean-reversion strategy.
    
    Generates a 'long' signal when RSI recovers back above the oversold boundary,
    and a 'short' signal when RSI drops back below the overbought boundary.
    Ideal for range-bound regimes (e.g. range_low, range_normal).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        atr_period: int = 14,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            atr_period=atr_period,
            **kwargs,
        )
        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) must be strictly less than overbought ({overbought})"
            )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.atr_period = atr_period
        self.warmup_period = max(rsi_period, atr_period) + 10

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        self.validate_input(df)

        if len(df) < self.warmup_period:
            return None

        close: np.ndarray = df["close"].to_numpy().astype(float)
        high: np.ndarray = df["high"].to_numpy().astype(float)
        low: np.ndarray = df["low"].to_numpy().astype(float)

        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)

        if (
            np.isnan(rsi[-1])
            or np.isnan(rsi[-2])
            or np.isnan(atr[-1])
            or atr[-1] <= 0.0
        ):
            return None

        prev_rsi, curr_rsi = rsi[-2], rsi[-1]

        direction: Optional[str] = None
        if prev_rsi <= self.oversold and curr_rsi > self.oversold:
            direction = "long"
        elif prev_rsi >= self.overbought and curr_rsi < self.overbought:
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
                "rsi": float(curr_rsi),
                "prev_rsi": float(prev_rsi),
                "oversold": self.oversold,
                "overbought": self.overbought,
                "atr": raw_sl_distance,
            },
        )
