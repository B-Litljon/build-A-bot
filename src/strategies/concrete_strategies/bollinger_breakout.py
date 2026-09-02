"""
Bollinger Bands breakout strategy.

Glossary:
    bb_period -- lookback period for Bollinger Bands (default 20).
    bb_std -- standard deviation multiplier for bands (default 2.0).
    atr_period -- lookback period for ATR stop-loss distance calculation (default 14).
    upper breakout -- price closing above the upper Bollinger Band.
    lower breakout -- price closing below the lower Bollinger Band.
    bandwidth -- (upper - lower) / middle, normalized band width.
"""

from typing import Any, Optional
import numpy as np
import polars as pl
import talib

from strategies.base import BaseStrategy, Signal


class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands breakout strategy.
    
    Generates a 'long' signal when the close breaks above the upper Bollinger Band,
    and a 'short' signal when the close breaks below the lower Bollinger Band.
    Ideal for trending or volatility-expanding market regimes (e.g. trend_high).
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            bb_period=bb_period,
            bb_std=bb_std,
            atr_period=atr_period,
            **kwargs,
        )
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.warmup_period = max(bb_period, atr_period) + 10

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        self.validate_input(df)

        if len(df) < self.warmup_period:
            return None

        close: np.ndarray = df["close"].to_numpy().astype(float)
        high: np.ndarray = df["high"].to_numpy().astype(float)
        low: np.ndarray = df["low"].to_numpy().astype(float)

        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=talib.MA_Type.SMA,
        )
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)

        if (
            np.isnan(upper[-1])
            or np.isnan(upper[-2])
            or np.isnan(lower[-1])
            or np.isnan(lower[-2])
            or np.isnan(atr[-1])
            or atr[-1] <= 0.0
        ):
            return None

        prev_close, curr_close = close[-2], close[-1]
        prev_upper, curr_upper = upper[-2], upper[-1]
        prev_lower, curr_lower = lower[-2], lower[-1]

        direction: Optional[str] = None
        # Long: price crosses above upper band
        if prev_close <= prev_upper and curr_close > curr_upper:
            direction = "long"
        # Short: price crosses below lower band
        elif prev_close >= prev_lower and curr_close < curr_lower:
            direction = "short"

        if direction is None:
            return None

        current_price = float(curr_close)
        raw_sl_distance = float(atr[-1])
        symbol = str(df["symbol"].tail(1)[0]) if "symbol" in df.columns else None
        bar_ts = df["timestamp"].tail(1)[0] if "timestamp" in df.columns else None
        bandwidth = float((curr_upper - curr_lower) / curr_middle) if (curr_middle := middle[-1]) > 0 else 0.0

        return Signal(
            direction=direction,
            entry_price=current_price,
            raw_sl_distance=raw_sl_distance,
            raw_tp_distance=raw_sl_distance,
            metadata={
                "strategy": self.name,
                "symbol": symbol,
                "timestamp": bar_ts,
                "upper_band": float(curr_upper),
                "lower_band": float(curr_lower),
                "middle_band": float(middle[-1]),
                "bandwidth": bandwidth,
                "atr": raw_sl_distance,
            },
        )
