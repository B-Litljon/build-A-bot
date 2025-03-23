import polars as pl
import numpy as np
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

#stock_stream = StockDataStream(api_key, api_secret)
#stock_stream.subscribe_bars(handle_bar_update, symbol)



# connect to the websocket

class HierarchyAggregator:
    """
    A class for aggregating bar data (OHLCV) into higher timeframes using a hierarchical approach.

    This class provides functionality to efficiently convert bar data from a base interval (e.g., 1 minute)
    into bars of various higher timeframes (e.g., 5 minutes, 15 minutes, 30 minutes). It employs a
    hierarchical aggregation method to minimize redundant calculations and optimize performance.

    **Class Operation:**

    The `HierarchicalAggregator` maintains internal buffers for each target timeframe. When a new bar
    arrives, it is added to the buffer of the base timeframe. The class then checks if enough bars are
    present in the buffer to form a bar of the next higher timeframe. If so, it aggregates the bars in
    the buffer and generates a new bar for the higher timeframe. This process is repeated, with the
    newly generated bars being used to create even higher timeframe bars, until all target timeframes
    have been processed.

    **Attributes:**

        base_interval (int):
            The base interval of the incoming bar data, representing the smallest timeframe that
            the aggregator receives (e.g., 1 for 1-minute bars).

        aggregators (dict):
            A dictionary used to store the aggregation buffers and related information for each
            target interval.

            * Keys: Target intervals (in minutes) for aggregation (e.g., 5, 15, 30).
            * Values: Dictionaries containing:
                * 'buffer' (polars.DataFrame):
                    A Polars DataFrame used as a temporary storage area for bars waiting to be
                    aggregated into the corresponding target interval.
                * 'parent_interval' (int):
                    An integer representing the number of bars from the parent interval required
                    to create a single bar of the current interval. This is calculated during
                    initialization using the `_get_parent_interval` method.
    """

    def __init__(self, base_interval=1, target_intervals=[5, 15, 30]):
        """
        Initializes the HierarchicalAggregator with the specified base interval and target intervals.

        This constructor sets up the aggregator with the base interval (the smallest unit of time for
        incoming bars) and the target intervals (the higher timeframes to which bars will be
        aggregated). It also initializes the internal `aggregators` dictionary.

        Args:
            base_interval (int, optional):
                The base interval in minutes. Defaults to 1 (e.g., for 1-minute bars).

            target_intervals (list, optional):
                A list of target intervals in minutes to aggregate to.
                Defaults to [5, 15, 30] (e.g., for 5-minute, 15-minute, and 30-minute bars).

        **Initialization Process:**

        1.  The `base_interval` attribute is set.
        2.  The `aggregators` dictionary is initialized.
        3.  For each interval in `target_intervals`, an entry is created in the `aggregators`
            dictionary. Each entry contains:
            * A `buffer` (Polars DataFrame) to hold bars before aggregation.
            * A `parent_interval` value, calculated using `_get_parent_interval`, indicating
                how many bars from the parent interval are needed for aggregation.
        """
        self.base_interval = base_interval
        self.aggregators = {}

        self.base_interval = base_interval
        self.aggregators = {}
        
        # Initialize aggregators for each timeframe
        for interval in target_intervals:
            self.aggregators[interval] = {
                'buffer': pl.DataFrame(),
                'parent_interval': interval // self._get_parent_interval(interval)
            }

    def _get_parent_interval(self, interval):
        """
        Determines the parent interval for a given target interval.

        This method calculates the appropriate parent interval to use when aggregating bars for a
        given target interval. The parent interval is the largest interval smaller than the target
        interval that divides evenly into it. This hierarchical approach optimizes aggregation
        (e.g., 15-minute bars are built from 5-minute bars, not directly from 1-minute bars).

        Args:
            interval (int):
                The target interval (in minutes) for which to determine the parent interval.

        Returns:
            int:
                The parent interval (in minutes). If no suitable parent interval is found
                within the target intervals, the base interval is returned.

        **Logic:**

        1.  The method iterates through the sorted keys of the `aggregators` dictionary (which
            represent the target intervals) in descending order.
        2.  For each `parent` interval, it checks two conditions:
            * If the `interval` is evenly divisible by the `parent` (`interval % parent == 0`).
            * If the `parent` interval is smaller than the `interval` (`parent < interval`).
        3.  If both conditions are met, the `parent` interval is considered a suitable parent,
            and the method returns it.
        4.  If no suitable parent interval is found after checking all target intervals, the
            method returns the `base_interval`.
        """
        for parent in sorted(self.aggregators.keys(),
                           reverse=True):  # Iterate through intervals largest to smallest
            if interval % parent == 0 and parent < interval:
                return parent
        # Determine which lower timeframe to use (e.g., 15m uses 5m bars)
        for parent in sorted(self.aggregators.keys(), reverse=True):
            if interval % parent == 0 and parent < interval:
                return parent
        return self.base_interval  # Default to 1m
        

    def add_bar(self, bar, interval=1):
        """
        Adds a bar to the specified interval's buffer and triggers aggregation if necessary.

        This method is the primary interface for feeding bar data into the HierarchicalAggregator.
        It adds a single bar to the buffer associated with the specified interval. After adding
        the bar, it checks if the buffer contains enough bars to perform aggregation into the next
        higher timeframe. If so, it aggregates the bars and recursively calls itself to propagate
        the aggregated bar to the parent interval.

        Args:
            bar (dict):
                A dictionary representing a single bar of OHLCV data.
                It is expected to contain the following keys:
                * "timestamp" (datetime): The timestamp of the bar.
                * "open" (float): The opening price of the bar.
                * "high" (float): The highest price of the bar.
                * "low" (float): The lowest price of the bar.
                * "close" (float): The closing price of the bar.
                * "volume" (float): The trading volume of the bar.

            interval (int, optional):
                The interval (in minutes) to which the bar should be added.
                Defaults to 1 (i.e., the base interval).

        Raises:
            ValueError:
                If the specified `interval` is not a valid target interval.

        **Process:**

        1.  The method first checks if the given `interval` is a key in the `self.aggregators`
            dictionary. If not, it raises a ValueError, as this indicates an invalid interval.
        2.  It retrieves the aggregator information (including the buffer) for the given
            `interval` from the `self.aggregators` dictionary.
        3.  The incoming `bar` (represented as a dictionary) is converted into a Polars DataFrame
            and concatenated to the existing `buffer` DataFrame.
        4.  The method then checks if the length of the `buffer` DataFrame is greater than or
            equal to `required_bars`. `required_bars` is obtained from the
            `parent_interval` value stored in the `aggregators` dictionary for the current
            interval.
        5.  If there are enough bars in the buffer:
            * The `_aggregate_bars` method is called to aggregate the bars in the buffer
                into a new bar.
            * The `parent_interval` is calculated by multiplying the current `interval`
                with the `required_bars`.
            * If this `parent_interval` exists as a key in the `self.aggregators` dictionary,
                the `add_bar` method is recursively called with the `new_bar` and the
                `parent_interval` to propagate the aggregated bar to the next higher
                timeframe.
            * The `buffer` for the current interval is reset to an empty Polars DataFrame.
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
        Aggregates a Polars DataFrame of bars into a single higher timeframe bar.

        This method takes a Polars DataFrame containing multiple bars of the same interval and
        aggregates them into a single bar representing the next higher timeframe.

        Args:
            df (polars.DataFrame):
                A Polars DataFrame containing the bars to aggregate.
                The DataFrame is expected to have the following columns:
                * "timestamp" (datetime): Timestamps of the bars.
                * "open" (float): Opening prices of the bars.
                * "high" (float): Highest prices of the bars.
                * "low" (float): Lowest prices of the bars.
                * "close" (float): Closing prices of the bars.
                * "volume" (float): Trading volumes of the bars.

            interval (int):
                The interval (in minutes) of the bars being aggregated.

        Returns:
            dict:
                A dictionary representing the aggregated bar, containing the following keys:
                * "timestamp" (datetime): The timestamp of the new bar (start time).
                * "open" (float): The opening price of the new bar (first open).
                * "high" (float): The highest price of the new bar (max high).
                * "low" (float): The lowest price of the new bar (min low).
                * "close" (float): The closing price of the new bar (last close).
                * "volume" (float): The total volume of the new bar (sum of volumes).
                * "interval" (int): The interval of the aggregated bar.

        **Aggregation Logic:**

        1.  The method calculates the 'timestamp' of the aggregated bar by taking the minimum
            timestamp from the input DataFrame (`df`).
        2.  The 'open' price of the aggregated bar is taken from the first row of the input
            DataFrame.
        3.  The 'high' price of the aggregated bar is calculated by taking the maximum 'high'
            price from the input DataFrame.
        4.  The 'low' price of the aggregated bar is calculated by taking the minimum 'low'
            price from the input DataFrame.
        5.  The 'close' price of the aggregated bar is taken from the last row of the input
            DataFrame.
        6.  The 'volume' of the aggregated bar is calculated by summing the 'volume' from all
            rows of the input DataFrame.
        7.  The `interval` of the aggregated bar is simply the `interval` provided as input
            to the method.
        8.  A dictionary containing these calculated values is returned.
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