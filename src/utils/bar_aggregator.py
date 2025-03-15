import polars as pl
import asyncio
from alpaca.data import StockDataStream
from datetime import datetime 


# so basically what we have is a function that accepts ohclv data as input. (alpaca only has 1m and 1d)
# we pass an argument for the timeframe we want to aggregate to. (eg: 5m, 15m, 1h, 4h)
# so this function will keep receiving the candle data and also create a list.
# when the list reaches the desired length, it will aggregate the data and return it.
# the function will then clear the list and start over.
# this function will actually handle the wss connection, the data stream, and the aggregation.
# the trading bot will only need to call this function and pass the desired timeframe.

stock_stream = StockDataStream(api_key, api_secret)
stock_stream.subscribe_bars(handle_bar_update, symbol)



# connect to the websocket

class BarAggregator:
    def __init__(self, interval_minutes):
        """
        Initialize the BarAggregator.
        :param interval_minutes: Number of minutes for aggregation (e.g., 5 for 5m bars).
        """
        self.interval_minutes = interval_minutes
        self.data = pl.DataFrame()  # Initialize an empty Polars DataFrame

    def add_bar(self, bar):
        """
        Add a new 1-minute bar to the aggregator.
        :param bar: Dictionary containing 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
        """
        new_row = pl.DataFrame([bar])
        self.data = pl.concat([self.data, new_row])

    def aggregate(self):
        """
        Aggregate the stored bars into a single higher timeframe bar.
        :return: Dictionary containing aggregated bar data or None if not enough data.
        """
        if len(self.data) < self.interval_minutes:
            return None  # Not enough data to aggregate yet

        # Aggregate the data
        aggregated_bar = {
            "timestamp": self.data["timestamp"].min(),  # Use the earliest timestamp
            "open": self.data["open"].first(),          # Use the first open price
            "high": self.data["high"].max(),            # Use the highest high price
            "low": self.data["low"].min(),              # Use the lowest low price
            "close": self.data["close"].last(),         # Use the last close price
            "volume": self.data["volume"].sum()         # Sum up all volumes
        }

        # Clear the DataFrame for new incoming bars
        self.data = pl.DataFrame()

        return aggregated_bar







# dev notes:
# the bar aggregator should also handle cases where we request bars from the historical bars first as well to get the bars started. but we'll think about that later when
# the bar aggregator is working properly #