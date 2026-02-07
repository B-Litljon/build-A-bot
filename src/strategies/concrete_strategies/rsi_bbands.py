from typing import List, Dict
import logging
from strategies.strategy import Strategy
from core.signal import Signal
from core.order_management import OrderParams
import talib
import polars as pl
import numpy as np


class RSIBBands(Strategy):
    def __init__(self, bb_period: int = 20, bb_std_dev: int = 2, rsi_period: int = 14, roc_period: int = 9):
        super().__init__()
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_period = rsi_period
        self.roc_period = roc_period

        # State tracking per symbol (Dict[str, bool]) to prevent cross-contamination
        self.stage_one_triggered = {}

        self.order_params = OrderParams(
            risk_percentage=0.02,
            tp_multiplier=1.5,
            sl_multiplier=0.9,
            use_trailing_stop=False,
        )

    def analyze(self, data: Dict[str, pl.DataFrame]) -> List[Signal]:
        signals = []
        for symbol, df in data.items():
            # Ensure we have enough data
            if len(df) < max(self.bb_period, self.rsi_period, self.roc_period) + 1:
                logging.warning(f"Not enough data for {symbol}. Skipping analysis.")
                continue

            # Initialize state for this symbol if not present
            if symbol not in self.stage_one_triggered:
                self.stage_one_triggered[symbol] = False

            logging.info(f"Analyzing data for {symbol}...")

            # --- Technical Analysis (TA-Lib + Polars) ---
            # TA-Lib expects numpy arrays or Polars Series (which act as arrays)
            # We convert to numpy explicitly to ensure safety across all TA-Lib versions
            close_prices = df["close"].to_numpy()

            upper, middle, lower = talib.BBANDS(
                close_prices,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std_dev,
                nbdevdn=self.bb_std_dev,
                matype=0,
            )
            rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)

            # Bandwidth Calculation (Numpy Array operations)
            bandwidth = upper - lower
            bandwidth_roc = talib.ROC(bandwidth, timeperiod=self.roc_period)

            # --- Get Latest Values (Polars/Numpy Negative Indexing) ---
            current_price = df["close"][-1]  # Polars Series scalar access
            lower_band = lower[-1]  # Numpy array access
            rsi_value = rsi[-1]  # Numpy array access
            bandwidth_roc_value = bandwidth_roc[-1]  # Numpy array access

            logging.info(
                f"Symbol: {symbol}, Price: {current_price:.2f}, Lower BB: {lower_band:.2f}, RSI: {rsi_value:.2f}, ROC: {bandwidth_roc_value:.2f}"
            )

            # --- Logic Engine ---
            if not self.stage_one_triggered[symbol]:
                # Stage 1: Oversold Condition
                if current_price < lower_band and rsi_value <= 25:
                    self.stage_one_triggered[symbol] = True
                    logging.info(
                        f"Stage 1 TRIGGERED for {symbol}: Price ({current_price:.2f}) < Lower BB & RSI ({rsi_value:.2f}) <= 25"
                    )

            elif self.stage_one_triggered[symbol]:
                logging.info(f"Checking Stage 2 for {symbol} (Stage 1 active)...")

                # Stage 2: Recovery & Momentum Confirmation
                # RSI recovers to [30, 35] AND Bandwidth is expanding (ROC > 0.15)
                if 30 <= rsi_value < 35 and not np.isnan(bandwidth_roc_value) and bandwidth_roc_value > 0.15:
                    if self.is_bullish_engulfing(df):
                        logging.info(f"BUY SIGNAL GENERATED for {symbol} at {current_price:.2f}")
                        signals.append(Signal("BUY", symbol, current_price))
                        self.stage_one_triggered[symbol] = False  # Reset trigger after signal
                    else:
                        logging.warning(
                            f"Stage 2 conditions met for {symbol}, but NO Bullish Engulfing pattern. holding..."
                        )

                # Reset logic: If RSI goes too high without buying, or drops back down?
                # Optional: Add logic here to reset if RSI > 40 to avoid stale triggers.
                elif rsi_value > 40:
                    logging.info(
                        f"Stage 1 Reset for {symbol}: RSI drifted too high ({rsi_value:.2f}) without signal."
                    )
                    self.stage_one_triggered[symbol] = False

        return signals

    def get_order_params(self) -> OrderParams:
        return self.order_params

    def is_bullish_engulfing(self, df: pl.DataFrame) -> bool:
        """
        Checks for bullish engulfing pattern using Polars indexing.
        """
        if len(df) < 2:
            return False

        # Polars negative indexing: [-1] is current, [-2] is previous
        current_open = df["open"][-1]
        current_close = df["close"][-1]

        prev_open = df["open"][-2]
        prev_close = df["close"][-2]

        # Logic:
        # 1. Previous candle was red (Close < Open)
        # 2. Current candle is green (Close > Open)
        # 3. Current Body engulfs Previous Body
        is_prev_red = prev_close < prev_open
        is_curr_green = current_close > current_open
        engulfing = (current_close > prev_open) and (current_open < prev_close)

        return is_prev_red and is_curr_green and engulfing
