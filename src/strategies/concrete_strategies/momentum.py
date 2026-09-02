"""
Momentum and trend acceleration strategy.

Glossary:
    fast_period -- fast EMA period for MACD oscillator (default 12).
    slow_period -- slow EMA period for MACD oscillator (default 26).
    signal_period -- signal line smoothing period (default 9).
    trend_ema_period -- baseline trend filter EMA (default 50).
    atr_period -- lookback period for ATR stop-loss distance calculation (default 14).
"""

from typing import Any, Optional
import numpy as np
import polars as pl
import talib

from strategies.base import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """
    Momentum acceleration strategy with a higher-order trend filter.
    
    Generates a 'long' signal when the MACD oscillator crosses above its signal line
    while price trades above the 50-period trend EMA.
    Generates a 'short' signal when MACD crosses below its signal line while price
    trades below the 50-period trend EMA.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        trend_ema_period: int = 50,
        atr_period: int = 14,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            trend_ema_period=trend_ema_period,
            atr_period=atr_period,
            **kwargs,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.trend_ema_period = trend_ema_period
        self.atr_period = atr_period
        self.warmup_period = max(slow_period + signal_period, trend_ema_period) + atr_period + 5

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        self.validate_input(df)

        if len(df) < self.warmup_period:
            return None

        close: np.ndarray = df["close"].to_numpy().astype(float)
        high: np.ndarray = df["high"].to_numpy().astype(float)
        low: np.ndarray = df["low"].to_numpy().astype(float)

        macd, macd_signal, _ = talib.MACD(
            close,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period,
        )
        trend_ema = talib.EMA(close, timeperiod=self.trend_ema_period)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)

        if (
            np.isnan(macd[-1])
            or np.isnan(macd[-2])
            or np.isnan(macd_signal[-1])
            or np.isnan(macd_signal[-2])
            or np.isnan(trend_ema[-1])
            or np.isnan(atr[-1])
            or atr[-1] <= 0.0
        ):
            return None

        prev_macd, curr_macd = macd[-2], macd[-1]
        prev_sig, curr_sig = macd_signal[-2], macd_signal[-1]
        curr_close = close[-1]
        curr_trend = trend_ema[-1]

        direction: Optional[str] = None
        # Bullish momentum cross above signal line, filtered by trend EMA
        if prev_macd <= prev_sig and curr_macd > curr_sig and curr_close > curr_trend:
            direction = "long"
        # Bearish momentum cross below signal line, filtered by trend EMA
        elif prev_macd >= prev_sig and curr_macd < curr_sig and curr_close < curr_trend:
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
                "macd": float(curr_macd),
                "macd_signal": float(curr_sig),
                "trend_ema": float(curr_trend),
                "atr": raw_sl_distance,
            },
        )
