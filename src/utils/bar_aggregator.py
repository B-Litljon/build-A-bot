from typing import Dict, List, Optional, Any
import polars as pl
import numpy as np
import asyncio
from alpaca.data import StockDataStream
from datetime import datetime

# connect to the websocket

class BarAggregator:
    """
    A class for aggregating bar data (OHLCV) into higher timeframes using a hierarchical approach.

    This class provides functionality to efficiently convert bar data from a base interval (e.g., 1 minute)
    into bars of various higher timeframes (e.g., 5 minutes, 15 minutes, 30 minutes). It employs a
    hierarchical aggregation method to minimize redundant calculations and optimize performance.

    **Class Operation:**

    The `BarAggregator` maintains internal buffers for each target timeframe. When a new bar
    arrives, it is added to the buffer of the base timeframe. The class then checks if enough bars are
    present in the buffer to form a bar of the next higher timeframe. If so, it aggregates the bars in
    the buffer and generates a new bar for the higher timeframe. This process is repeated, with the
    newly generated bars being used to create even higher timeframe bars, until all target timeframes
    have been processed.

    Completed bars for the `target_intervals` specified during initialization can be retrieved
    via the return value of the `add_bar` method.

    **Attributes:**

        base_interval (int):
            The base interval of the incoming bar data, representing the smallest timeframe that
            the aggregator receives (e.g., 1 for 1-minute bars).

        target_intervals (List[int]):
            A list of the target intervals (in minutes) that the aggregator is configured to produce.

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
    # update init to accept timeframe from user instead of hardcoded values
    # error check and log to the user if their timeframe is not useable
    def __init__(self, base_interval: int = 1, target_intervals: List[int] = [5, 15, 30]) -> None:
        """
        Initializes the BarAggregator with the specified base interval and target intervals.

        This constructor sets up the aggregator with the base interval (the smallest unit of time for
        incoming bars) and the target intervals (the higher timeframes to which bars will be
        aggregated). It also initializes the internal `aggregators` dictionary.

        Args:
            base_interval (int, optional):
                The base interval in minutes. Defaults to 1 (e.g., for 1-minute bars).
                Must be a positive integer.

            target_intervals (list, optional):
                A list of target intervals in minutes to aggregate to.
                Defaults to [5, 15, 30] (e.g., for 5-minute, 15-minute, and 30-minute bars).
                All intervals must be positive integers and greater than the base_interval.

        Raises:
            ValueError:
                If base_interval is not a positive integer.
            ValueError:
                If any interval in target_intervals is not a positive integer or not greater than base_interval.
        """
        if not isinstance(base_interval, int) or base_interval <= 0:
            raise ValueError("Base interval must be a positive integer.")

        self.base_interval = base_interval
        self.target_intervals = sorted(target_intervals) # Store target intervals
        self.aggregators: Dict[int, Dict[str, Any]] = {}

        for interval in self.target_intervals:
            if not isinstance(interval, int) or interval <= self.base_interval:
                raise ValueError(f"Target intervals must be positive integers and greater than the base interval. Invalid interval: {interval}")
            self.aggregators[interval] = {
                'buffer': pl.DataFrame(),
                'parent_interval': interval // self._get_parent_interval(interval)
            }

    def _get_parent_interval(self, interval: int) -> int:
        """
        Determines the parent interval for a given target interval.

        This method calculates the appropriate parent interval to use when aggregating bars for a
        given target interval. The parent interval is the largest interval smaller than the target
        interval that divides evenly into it. This hierarchical approach optimizes aggregation
        (e.g., 15-minute bars are built from 5-minute bars, not directly from 1-minute bars).

        Args:
            interval (int):
                The target interval (in minutes) for which to determine the parent interval.
                Must be a positive integer.

        Returns:
            int:
                The parent interval (in minutes). If no suitable parent interval is found
                within the target intervals, the base interval is returned.

        Raises:
            ValueError:
                If the provided interval is not a positive integer.
        """

        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("Interval must be a positive integer.")

        # Check if any existing target intervals are parents
        for parent in sorted(self.aggregators.keys(), reverse=True):
             if interval % parent == 0 and parent < interval:
                return parent

        # If not, use the base interval as the parent
        return self.base_interval


    def add_bar(self, bar: Dict[str, Any], interval: int = 1) -> Optional[Dict[str, Any]]:
        """
        Adds a bar to the specified interval's buffer and triggers aggregation if necessary.

        This method is the primary interface for feeding bar data into the BarAggregator.
        It adds a single bar to the buffer associated with the specified interval. After adding
        the bar, it checks if the buffer contains enough bars to perform aggregation into the next
        higher timeframe. If so, it aggregates the bars and recursively calls itself to propagate
        the aggregated bar to the parent interval.

        If a completed bar for one of the `target_intervals` is generated during the aggregation
        process (including recursive calls), that completed bar dictionary is returned.
        Otherwise, None is returned.

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
                The values for 'open', 'high', 'low', 'close', and 'volume' must be numeric.

            interval (int, optional):
                The interval (in minutes) to which the bar should be added.
                Defaults to 1 (i.e., the base interval). Must be a positive integer.

        Returns:
            Optional[Dict[str, Any]]:
                A dictionary representing the completed bar if the aggregation process
                results in a bar for one of the target intervals. Returns None otherwise.

        Raises:
            ValueError:
                If the specified `interval` is not a valid target interval.
            TypeError:
                If the `bar` argument is not a dictionary.
            ValueError:
                 If the bar dictionary does not contain the required keys or if the values for OHLCV are not numeric.
        """
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("Interval must be a positive integer.")

        if interval not in self.aggregators and interval != self.base_interval:
             raise ValueError(f"Unsupported interval: {interval}m. Must be base interval or a target interval.")


        if not isinstance(bar, dict):
            raise TypeError("Bar must be a dictionary.")

        required_keys = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(key in bar for key in required_keys):
            raise ValueError(f"Bar dictionary must contain the following keys: {required_keys}")

        numeric_keys = ["open", "high", "low", "close", "volume"]
        for key in numeric_keys:
            if not isinstance(bar[key], (int, float)):
                raise ValueError(f"Value for '{key}' must be numeric.")

        # If the incoming bar is the base interval, add it to the buffer for the smallest target interval
        if interval == self.base_interval:
            smallest_target_interval = self.target_intervals[0]
            agg = self.aggregators[smallest_target_interval]
            agg['buffer'] = pl.concat([agg['buffer'], pl.DataFrame([bar])])
            current_interval_to_process = smallest_target_interval # Start processing with the smallest target interval
        elif interval in self.aggregators:
             # If the incoming bar is already a target interval, add it to its buffer
            agg = self.aggregators[interval]
            agg['buffer'] = pl.concat([agg['buffer'], pl.DataFrame([bar])])
            current_interval_to_process = interval # Continue processing with this target interval
        else:
             return None # Should not happen with the checks above, but as a safeguard


        # Check if ready to aggregate
        required_bars = agg['parent_interval']
        if len(agg['buffer']) >= required_bars:
            # Aggregate
            new_bar = self._aggregate_bars(agg['buffer'], current_interval_to_process)

            # Reset buffer
            agg['buffer'] = pl.DataFrame()

            # If the aggregated bar is a target interval, return it
            if current_interval_to_process in self.target_intervals:
                 # Cascade to higher timeframe
                parent_interval = current_interval_to_process * required_bars
                if parent_interval in self.aggregators:
                    return self.add_bar(new_bar, parent_interval)
                else:
                     return new_bar # Return the completed target bar if no higher target interval exists

            # If not a target interval, but successfully aggregated, cascade upwards recursively
            # The recursive call will handle returning a target interval bar if one is completed
            parent_interval = current_interval_to_process * required_bars
            if parent_interval in self.aggregators:
                 return self.add_bar(new_bar, parent_interval)
            else:
                 return None # No higher target interval to cascade to


        return None # No bar completed for a target interval


    def _aggregate_bars(self, df: pl.DataFrame, interval: int) -> Dict[str, Any]:
        """
        Aggregates a Polars DataFrame of bars into a single higher timeframe bar.

        This method takes a Polars DataFrame containing multiple bars of the same interval and
        aggregates them into a single bar representing the next higher timeframe.

        Args:
            df (pl.DataFrame):
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
                 Must be a positive integer.

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

        Raises:
            TypeError:
                If `df` is not a Polars DataFrame.
            ValueError:
                If the DataFrame does not contain the required columns.
            ValueError:
                 If the provided interval is not a positive integer.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame.")

        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("Interval must be a positive integer.")

        return {
            "timestamp": df["timestamp"].min(),
            "open": df["open"].first(),
            "high": df["high"].max(),
            "low": df["low"].min(),
            "close": df["close"].last(),
            "volume": df["volume"].sum(),
            "interval": interval
        }