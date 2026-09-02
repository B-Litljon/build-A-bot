"""
Donchian Channel breakout strategy (Turtle style).

Glossary:
    channel_period -- lookback period for highest high and lowest low (default 20).
    atr_period -- lookback period for ATR stop-loss distance calculation (default 14).
    upper breakout -- current close breaking above the prior N-bar highest high.
    lower breakout -- current close breaking below the prior N-bar lowest low.
    lookahead-free -- channel boundaries are strictly computed over prior bars,
        excluding the bar being evaluated.
"""

from typing import Any, Optional
import numpy as np
import polars as pl
import talib

from strategies.base import BaseStrategy, Signal


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Classic Donchian Channel breakout strategy.
    
    Generates a 'long' signal when the close breaks above the previous N-bar highest high,
    and a 'short' signal when the close breaks below the previous N-bar lowest low.
    Exemplary trend-following strategy for strong directional regimes (e.g. trend_high).
    """

    def __init__(
        self,
        channel_period: int = 20,
        atr_period: int = 14,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            channel_period=channel_period,
            atr_period=atr_period,
            **kwargs,
        )
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.warmup_period = channel_period + atr_period + 2

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        self.validate_input(df)

        if len(df) < self.warmup_period:
            return None

        close: np.ndarray = df["close"].to_numpy().astype(float)
        high: np.ndarray = df["high"].to_numpy().astype(float)
        low: np.ndarray = df["low"].to_numpy().astype(float)

        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        if np.isnan(atr[-1]) or atr[-1] <= 0.0:
            return None

        # Prior N-bar high/low for the bar *before* current bar
        # Current bar is at index -1.
        # Prior N bars for bar -1 are high[-1-channel_period : -1].
        # Prior N bars for bar -2 are high[-2-channel_period : -2].
        p = self.channel_period
        if len(high) < p + 2:
            return None

        channel_high_curr = np.max(high[-1 - p : -1])
        channel_low_curr = np.min(low[-1 - p : -1])

        channel_high_prev = np.max(high[-2 - p : -2])
        channel_low_prev = np.min(low[-2 - p : -2])

        prev_close, curr_close = close[-2], close[-1]

        direction: Optional[str] = None
        # Breakout occurs on the transition: previously at/below channel high, now above it
        if prev_close <= channel_high_prev and curr_close > channel_high_curr:
            direction = "long"
        elif prev_close >= channel_low_prev and curr_close < channel_low_curr:
            direction = "short"

        if direction is None:
            return None

        current_price = float(curr_close)
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
                "channel_high": float(channel_high_curr),
                "channel_low": float(channel_low_curr),
                "atr": raw_sl_distance,
            },
        )
