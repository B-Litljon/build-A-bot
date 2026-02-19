import polars as pl
import time
from typing import Iterator

def simulate_ws_stream(historical_data: pl.DataFrame, speed: float = 1.0) -> Iterator[pl.DataFrame]:
    """
    Simulates a WebSocket connection by iterating through historical data.

    Args:
        historical_data (pl.DataFrame): A DataFrame of historical OHLCV data.
        speed (float): The speed of the simulation (1.0 = real-time).

    Yields:
        pl.DataFrame: A single row of the DataFrame at each iteration.
    """
    for i in range(len(historical_data)):
        yield historical_data[i]
        time.sleep(1 / speed)