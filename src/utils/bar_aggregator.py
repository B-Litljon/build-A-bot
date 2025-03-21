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

class HierarchicalAggregator:
    def __init__(self, base_interval=1, target_intervals=[5, 15, 30]):
        self.base_interval = base_interval
        self.aggregators = {}
        
        # Initialize aggregators for each timeframe
        for interval in target_intervals:
            self.aggregators[interval] = {
                'buffer': pl.DataFrame(),
                'parent_interval': interval // self._get_parent_interval(interval)
            }

    def _get_parent_interval(self, interval):
        # Determine which lower timeframe to use (e.g., 15m uses 5m bars)
        for parent in sorted(self.aggregators.keys(), reverse=True):
            if interval % parent == 0 and parent < interval:
                return parent
        return self.base_interval  # Default to 1m
        

    def add_bar(self, bar, interval=1):
        """
        Add a bar to the target interval's buffer and cascade upward.
        """
        if interval not in self.aggregators:
            raise ValueError(f"Unsupported interval: {interval}m")
        
        # Append to buffer
        agg = self.aggregators[interval]
        agg['buffer'] = pl.concat([agg['buffer'], pl.DataFrame([bar])])
        
        # Check if ready to aggregate
        required_bars = agg['parent_interval']
        if len(agg['buffer']) >= required_bars:
            # Aggregate
            new_bar = self._aggregate_bars(agg['buffer'], interval)
            
            # Cascade to higher timeframe
            parent_interval = interval * required_bars
            if parent_interval in self.aggregators:
                self.add_bar(new_bar, parent_interval)
            
            # Reset buffer
            agg['buffer'] = pl.DataFrame()

    def _aggregate_bars(self, df: pl.DataFrame, interval: int):
        """
        Aggregate a Polars DataFrame into a higher timeframe bar.
        """
        return {
            "timestamp": df["timestamp"].min(),
            "open": df["open"].first(),
            "high": df["high"].max(),
            "low": df["low"].min(),
            "close": df["close"].last(),
            "volume": df["volume"].sum(),
            "interval": interval
        }





# dev notes:
# the bar aggregator should also handle cases where we request bars from the historical bars first as well to get the bars started. but we'll think about that later when
# the bar aggregator is working properly #