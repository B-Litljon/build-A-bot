import polars as pl
import numpy as np
import asyncio
from datetime import datetime 
from typing import Dict, List, Optional, Any

class LiveBarAggregator:
    """
    This aggregator uses the 'Buffer, Aggregate, and Clear' method to process a 
    live stream of 1-minute bars into a single, higher timeframe.

    Design & How it Works:
    It's built to be fast and memory-efficient. A small, temporary buffer holds 
    the incoming 1-minute bars. Once the buffer has enough bars for the target 
    timeframe (e.g., 5 bars for a 5-minute candle), it performs three steps:
    1. AGGREGATE: The bars in the buffer are combined into a single new, 
        higher-timeframe candle.
    2. UPDATE: This new candle is added to a fixed-size Polars DataFrame that 
        holds the recent history for indicator calculations.
    3. CLEAR: The temporary buffer is emptied, ready for the next set of bars.

    How to Use:
    This class is designed to track one timeframe at a time. If you need to 
    monitor multiple timeframes (e.g., 5-min and 30-min), you must create a 
    separate instance of this class for each one.
    """
    def __init__(self, timeframe: float, history_size: int = 240): 
        """
        Initializes the aggregator.
        
        Args:
        timeframe(int): the number of 1 minute bars to aggregate into a higher timeframe
                        (e.g., 5 for a 5 minute bar/candle. maximum of 240 minutes equalling 4 hours) #change to 24 hours, keep a days worth of bars in memory 
        history_size(int): the maximum number of aggregated bars to keep in memory for indicator calculations
        """
        self.timeframe = timeframe
        self.history_size = history_size
        # --State--
        # the temporary buffer for storing incoming 1m bars.
        self.buffer = []

        self.history_df = pl.DataFrame({
            'timestamp': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }, schema={
            'timestamp': pl.Datetime(time_unit="us"),
            'open': pl.Float64,
            'high': pl.Float64,
            'low': pl.Float64,
            'close': pl.Float64,
            'volume': pl.Int64
        })
    def add_bar(self, new_bar: dict):
        """
        This is the main method to be called for each new 1-minute bar.
        It adds the bar to the buffer and triggers aggregation if the
        buffer becomes full.

        Args:
            new_bar (dict): A dictionary representing the new 1-minute bar.
                            e.g., {'timestamp': ..., 'open': ..., ...}
        
        Returns:
            bool: True if a new aggregated bar was created, False otherwise.
        """
        self.buffer.append(new_bar)
        
        # Check if the buffer is full and ready for aggregation
        if len(self.buffer) == self.timeframe:
            self._aggregate_and_update()
            return True
            
        return False

    def _aggregate_and_update(self):
        """
        A private helper method to perform the aggregation and update the state.
        This is called only when the buffer is full.
        """
        print(f"\n--- Buffer full! Aggregating {self.timeframe} bars... ---")

        # 1. Convert the buffer (a list of dicts) into a Polars DataFrame.
        #    This is a very fast operation.
        chunk_df = pl.DataFrame(self.buffer)

        # 2. Perform the aggregation in a single, expressive command.
        new_agg_bar_df = chunk_df.select([
            pl.col("timestamp").last(),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum()
        ])
        
        # 3. Append the new aggregated bar to our history DataFrame.
        self.history_df = pl.concat([self.history_df, new_agg_bar_df], how="vertical")
        
        # 4. Trim the history to maintain the fixed size.
        #    This keeps memory usage stable.
        self.history_df = self.history_df.tail(self.history_size)
        
        # 5. Clear the buffer to get ready for the next set of bars.
        self.buffer = []

        print("Aggregation complete. History updated.")
        # In a real bot, you would now pass self.history_df to your strategy.


    