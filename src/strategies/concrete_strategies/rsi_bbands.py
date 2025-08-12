from typing import List, Dict
from strategies.strategy import Strategy
from core.signal import Signal
from core.order_management import OrderParams
import talib
import polars as pl

class RSIBBands(Strategy):
    def __init__(self, bb_period: int = 20, bb_std_dev: int = 2, rsi_period: int = 14, roc_period: int = 9):
        super().__init__()  
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_period = rsi_period
        self.roc_period = roc_period
        self.stage_one_triggered = False
        self.timeframe = 5
        self.order_params = OrderParams(
            risk_percentage=0.02,
            tp_multiplier=1.5,
            sl_multiplier=0.9,
            use_trailing_stop=False
        )

    def analyze(self, data: Dict[str, pl.DataFrame]) -> List[Signal]:
        signals = []
        for symbol, df in data.items():
            # Calculate indicators 
            upper, middle, lower = talib.BBANDS(df["close"], timeperiod=self.bb_period, nbdevup=self.bb_std_dev, nbdevdn=self.bb_std_dev, matype=0)
            rsi = talib.RSI(df["close"], timeperiod=self.rsi_period)
            bandwidth = upper - lower
            bandwidth_roc = talib.ROC(bandwidth, timeperiod=self.roc_period)

            # Get the last values
            current_price = df["close"].iloc[-1]
            lower_band = lower.iloc[-1]
            rsi_value = rsi.iloc[-1]
            bandwidth_roc_value = bandwidth_roc.iloc[-1]
            
            if not self.stage_one_triggered:
                if current_price < lower_band and rsi_value <= 25:
                    self.stage_one_triggered = True
                    print(f"Stage 1 triggered for {symbol}: Price ({current_price}) below lower band ({lower_band}) and RSI ({rsi_value}) oversold")
            
            elif self.stage_one_triggered:
                if 30 <= rsi_value < 35 and bandwidth_roc_value is not None and bandwidth_roc_value > 0.15:
                    if self.is_bullish_engulfing(df):
                        print(f"Buy signal triggered for {symbol}")
                        signals.append(Signal("BUY", symbol, current_price))
                        self.stage_one_triggered = False  # Reset the trigger
                    else:
                        print(f"Bullish engulfing pattern not detected for {symbol}")
                else:
                    print(f"Stage 2 not triggered for {symbol}: RSI ({rsi_value}) not in range or Bollinger Bands not expanding")

        return signals

    def get_order_params(self) -> OrderParams:
        return self.order_params

    def is_bullish_engulfing(self, df: pl.DataFrame) -> bool:
        if len(df) < 2:
            return False
        current_candle_open = df['open'].iloc[-1]
        current_candle_close = df['close'].iloc[-1]
        previous_candle_open = df['open'].iloc[-2]
        previous_candle_close = df['close'].iloc[-2]

        return current_candle_close > previous_candle_open and current_candle_open < previous_candle_close