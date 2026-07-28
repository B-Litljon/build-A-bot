"""
Replays a historical DataFrame row-by-row to imitate a live WebSocket feed.

STATUS (glossary pass, 2026-07-27): nothing in the repo imports this. The
actual offline-replay path is ``src/replay_test.py``, and the live feeds are
real broker WebSockets in ``src/execution/``. Documented as dead, not removed.

Glossary:
    simulate_ws_stream -- generator that yields one row of an OHLCV DataFrame
        at a time, sleeping between rows so a consumer sees something like a
        real-time feed.
    historical_data -- the OHLCV DataFrame to replay, in chronological order.
    speed -- playback multiplier; 1.0 sleeps one second per row, 60.0 replays
        roughly a minute of bars per second. Note the sleep is per row and does
        not read the bars' own timestamps.
"""

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