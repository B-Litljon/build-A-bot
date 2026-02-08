from typing import List, Dict
import logging
from strategies.strategy import Strategy
from core.signal import Signal
from core.order_management import OrderParams
import talib
import polars as pl
import numpy as np


class RSIBBands(Strategy):
    def __init__(
        self,
        bb_period: int = 20,
        bb_std_dev: int = 2,
        rsi_period: int = 14,
        roc_period: int = 9,
        # Configurable Logic Thresholds
        stage1_rsi_threshold: int = 30,
        stage2_rsi_entry: int = 30,
        stage2_rsi_exit: int = 40,
        stage2_min_roc: float = 0.15,
    ):
        super().__init__()
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_period = rsi_period
        self.roc_period = roc_period

        # Store Logic Thresholds
        self.stage1_rsi_threshold = stage1_rsi_threshold
        self.stage2_rsi_entry = stage2_rsi_entry
        self.stage2_rsi_exit = stage2_rsi_exit
        self.stage2_min_roc = stage2_min_roc

        # State tracking per symbol (Dict[str, bool])
        self.stage_one_triggered = {}

        self.order_params = OrderParams(
            risk_percentage=0.02,
            tp_multiplier=1.5,
            sl_multiplier=0.9,
            use_trailing_stop=False,
        )

    @property
    def warmup_period(self) -> int:
        """
        Returns the minimum number of candles required.
        Max of all lookback periods + 1 safety buffer.
        """
        return max(self.bb_period, self.rsi_period, self.roc_period) + 1

    def analyze(self, data: Dict[str, pl.DataFrame]) -> List[Signal]:
        signals = []
        for symbol, df in data.items():
            # Note: The TradingBot now handles the 'warmup_period' check before calling this.

            if symbol not in self.stage_one_triggered:
                self.stage_one_triggered[symbol] = False

            logging.info(f"Analyzing data for {symbol}...")

            # --- TA-Lib Calculations (NumPy Conversion) ---
            close_prices = df["close"].to_numpy()

            upper, middle, lower = talib.BBANDS(
                close_prices,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std_dev,
                nbdevdn=self.bb_std_dev,
                matype=0,
            )
            rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)

            bandwidth = upper - lower
            bandwidth_roc = talib.ROC(bandwidth, timeperiod=self.roc_period)

            # --- Latest Values ---
            current_price = df["close"][-1]
            lower_band = lower[-1]
            rsi_value = rsi[-1]
            bandwidth_roc_value = bandwidth_roc[-1]

            logging.info(
                f"Symbol: {symbol}, Price: {current_price:.2f}, Lower BB: {lower_band:.2f}, RSI: {rsi_value:.2f}, ROC: {bandwidth_roc_value:.2f}"
            )

            # --- Logic Engine ---
            if not self.stage_one_triggered[symbol]:
                # Stage 1: Oversold (Configurable)
                if current_price < lower_band and rsi_value <= self.stage1_rsi_threshold:
                    self.stage_one_triggered[symbol] = True
                    logging.info(
                        f"Stage 1 TRIGGERED for {symbol}: Price < Lower BB & RSI {rsi_value:.2f} <= {self.stage1_rsi_threshold}"
                    )

            elif self.stage_one_triggered[symbol]:
                logging.info(f"Checking Stage 2 for {symbol} (Stage 1 active)...")

                # Stage 2: Recovery (Configurable)
                rsi_in_range = self.stage2_rsi_entry <= rsi_value < self.stage2_rsi_exit
                roc_valid = (not np.isnan(bandwidth_roc_value)) and (bandwidth_roc_value > self.stage2_min_roc)

                if rsi_in_range and roc_valid:
                    if self.is_bullish_engulfing(df):
                        logging.info(f"BUY SIGNAL GENERATED for {symbol} at {current_price:.2f}")
                        signals.append(Signal("BUY", symbol, float(current_price)))
                        self.stage_one_triggered[symbol] = False
                    else:
                        logging.warning(f"Stage 2 met, but NO Bullish Engulfing. Holding...")

                # Reset if RSI goes too high
                elif rsi_value > self.stage2_rsi_exit + 5:
                    logging.info(f"Stage 1 Reset for {symbol}: RSI drifted too high ({rsi_value:.2f}).")
                    self.stage_one_triggered[symbol] = False

        return signals

    def get_order_params(self) -> OrderParams:
        return self.order_params

    def is_bullish_engulfing(self, df: pl.DataFrame) -> bool:
        if len(df) < 2:
            return False
        current_open = df["open"][-1]
        current_close = df["close"][-1]
        prev_open = df["open"][-2]
        prev_close = df["close"][-2]
        is_prev_red = prev_close < prev_open
        is_curr_green = current_close > current_open
        engulfing = (current_close > prev_open) and (current_open < prev_close)
        return is_prev_red and is_curr_green and engulfing
