from typing import Dict, List

import numpy as np
import polars as pl
import talib

from core.order_management import OrderParams
from core.signal import Signal
from strategies.strategy import Strategy


class SMACrossover(Strategy):
    def __init__(self, fast_period: int = 10, slow_period: int = 50):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.timeframe = 5
        self.order_params = OrderParams(
            risk_percentage=0.02,
            tp_multiplier=1.5,
            sl_multiplier=0.9,
            use_trailing_stop=False,
        )

    def analyze(self, data: Dict[str, pl.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []

        for symbol, df in data.items():
            closes = df["close"].to_numpy()
            if len(closes) < self.slow_period:
                continue

            fast_sma = talib.SMA(closes, timeperiod=self.fast_period)
            slow_sma = talib.SMA(closes, timeperiod=self.slow_period)

            prev_fast = fast_sma[-2]
            prev_slow = slow_sma[-2]
            curr_fast = fast_sma[-1]
            curr_slow = slow_sma[-1]

            if (
                not np.isnan(prev_fast)
                and not np.isnan(prev_slow)
                and not np.isnan(curr_fast)
                and not np.isnan(curr_slow)
                and prev_fast < prev_slow
                and curr_fast > curr_slow
            ):
                current_price = float(closes[-1])
                signals.append(Signal("BUY", symbol, current_price))

        return signals

    def get_order_params(self) -> OrderParams:
        return self.order_params

    @property
    def warmup_period(self) -> int:
        return self.slow_period
