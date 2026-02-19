import polars as pl
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional


class LiveBarAggregator:
    """
    Clock-aware aggregator that converts a live stream of 1-minute bars into
    higher-timeframe OHLCV candles using logical, wall-clock-aligned windows.

    Why clock-aware?
    ----------------
    The free Alpaca IEX feed frequently drops bars for 30-60+ minutes on
    lower-volume assets.  A naive "count N bars" approach would silently merge
    bars that span different logical windows (e.g. a 12:31 bar and a 13:02 bar
    into the same "5-minute" candle).  By anchoring every bar to its correct
    clock window we guarantee:

    * Each aggregated candle covers exactly one [window_start, window_start + timeframe) period.
    * Gaps between received bars are detected and filled with synthetic flat
      candles (OHLC = prior close, volume = 0) so that TA-Lib always operates
      on a continuous, evenly-spaced array.

    Public API (unchanged from the previous version):
        aggregator = LiveBarAggregator(timeframe=5, history_size=240)
        did_close = aggregator.add_bar(bar_dict)   # -> bool
        df = aggregator.history_df                  # -> pl.DataFrame

    Instantiate one aggregator per (symbol, timeframe) pair.
    """

    # -- Schema contract: every consumer (strategies, indicators) relies on this --
    _SCHEMA = {
        "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }

    def __init__(self, timeframe: int, history_size: int = 240):
        """
        Args:
            timeframe:    Number of minutes per aggregated candle (e.g. 5 for
                          5-minute bars).  Must be >= 1 and evenly divide 60
                          for clean clock alignment.
            history_size: Maximum number of aggregated candles retained in
                          ``history_df`` for indicator calculations.
        """
        if timeframe < 1:
            raise ValueError(f"timeframe must be >= 1, got {timeframe}")

        self.timeframe: int = int(timeframe)
        self.history_size: int = history_size

        # -- mutable state --
        self.buffer: list[dict] = []
        self.current_window_start: Optional[datetime] = None

        self.history_df: pl.DataFrame = pl.DataFrame(
            {col: [] for col in self._SCHEMA},
            schema=self._SCHEMA,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_bar(self, new_bar: dict) -> bool:
        """
        Ingest a single 1-minute bar dictionary.

        Args:
            new_bar: Must contain at minimum the keys
                     ``timestamp``, ``open``, ``high``, ``low``, ``close``,
                     ``volume``.  ``timestamp`` must be a timezone-aware
                     ``datetime`` in UTC (or convertible to UTC).

        Returns:
            ``True`` if one or more aggregated candle(s) were appended to
            ``history_df`` (i.e. a window closed), ``False`` otherwise.
        """
        bar_ts = self._ensure_utc(new_bar["timestamp"])
        bar_window = self._window_floor(bar_ts)

        # --- First bar ever ---
        if self.current_window_start is None:
            self.current_window_start = bar_window
            self.buffer.append(new_bar)
            logging.debug(
                f"First bar received. Window initialised to {bar_window}."
            )
            return False

        # --- Bar belongs to the current open window ---
        if bar_window == self.current_window_start:
            self.buffer.append(new_bar)
            logging.debug(
                f"Bar at {bar_ts} appended to window {self.current_window_start} "
                f"(buffer size: {len(self.buffer)})."
            )
            return False

        # --- Bar belongs to a strictly later window → close current window ---
        if bar_window > self.current_window_start:
            # 1. Aggregate whatever is in the buffer for the current window.
            self._aggregate_and_update(window_timestamp=self.current_window_start)

            # 2. Forward-fill any missing intermediate windows.
            self._forward_fill_gaps(
                closed_window=self.current_window_start,
                new_window=bar_window,
            )

            # 3. Reset state for the new window.
            self.current_window_start = bar_window
            self.buffer = [new_bar]

            logging.debug(
                f"Window advanced to {bar_window}.  New buffer started."
            )
            return True

        # --- Bar is older than current window (late / out-of-order) ---
        logging.warning(
            f"Dropping stale bar at {bar_ts} (window {bar_window}); "
            f"current window is {self.current_window_start}."
        )
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate_and_update(self, window_timestamp: datetime) -> None:
        """
        Collapse the current buffer into one aggregated candle and append it
        to ``history_df``.

        The candle's timestamp is set to ``window_timestamp`` (the logical
        window start), which is the standard convention for OHLCV data and
        matches Alpaca's own historical-bar timestamps.
        """
        if not self.buffer:
            logging.warning("_aggregate_and_update called with empty buffer.")
            return

        logging.info(
            f"Closing window {window_timestamp}: aggregating "
            f"{len(self.buffer)} bar(s) into a {self.timeframe}-minute candle."
        )

        chunk_df = pl.DataFrame(self.buffer)

        new_candle_df = chunk_df.select(
            pl.lit(window_timestamp).alias("timestamp").cast(
                pl.Datetime(time_unit="us", time_zone="UTC")
            ),
            pl.col("open").first().cast(pl.Float64),
            pl.col("high").max().cast(pl.Float64),
            pl.col("low").min().cast(pl.Float64),
            pl.col("close").last().cast(pl.Float64),
            pl.col("volume").cast(pl.Float64).sum(),
        )

        self._append_to_history(new_candle_df)

        logging.info(
            f"Aggregated candle: {new_candle_df.to_dicts()[0]}"
        )

    def _forward_fill_gaps(
        self,
        closed_window: datetime,
        new_window: datetime,
    ) -> None:
        """
        Inject synthetic flat candles for every missing interval between
        ``closed_window`` and ``new_window`` (exclusive on both ends — the
        closed window was already aggregated, and the new window hasn't
        accumulated bars yet).

        Each synthetic candle: OHLC = last known close, volume = 0.
        This keeps the time-series continuous so TA-Lib indicators remain
        mathematically valid.
        """
        step = timedelta(minutes=self.timeframe)
        gap_start = closed_window + step

        if gap_start >= new_window:
            return  # No missing intervals.

        # Resolve the last close price from history.
        if self.history_df.is_empty():
            logging.warning(
                "Cannot forward-fill gaps: history_df is empty "
                "(no prior close price available)."
            )
            return

        last_close: float = self.history_df["close"][-1]

        synthetic_rows: list[dict] = []
        cursor = gap_start
        while cursor < new_window:
            synthetic_rows.append(
                {
                    "timestamp": cursor,
                    "open": last_close,
                    "high": last_close,
                    "low": last_close,
                    "close": last_close,
                    "volume": 0.0,
                }
            )
            cursor += step

        if synthetic_rows:
            logging.info(
                f"Forward-filling {len(synthetic_rows)} missing "
                f"{self.timeframe}-minute candle(s) between "
                f"{gap_start} and {new_window}."
            )
            fill_df = pl.DataFrame(synthetic_rows, schema=self._SCHEMA)
            self._append_to_history(fill_df)

    def _append_to_history(self, df: pl.DataFrame) -> None:
        """Append rows to ``history_df`` and trim to ``history_size``."""
        self.history_df = pl.concat(
            [self.history_df, df], how="vertical"
        ).tail(self.history_size)

        logging.debug(
            f"history_df now has {len(self.history_df)} row(s) "
            f"(max {self.history_size})."
        )

    def _window_floor(self, ts: datetime) -> datetime:
        """
        Compute the logical window start for a given timestamp.

        Examples (timeframe=5):
            12:34:17 → 12:30:00
            12:30:00 → 12:30:00
            12:29:59 → 12:25:00
        """
        floored_minute = ts.minute - (ts.minute % self.timeframe)
        return ts.replace(minute=floored_minute, second=0, microsecond=0)

    @staticmethod
    def _ensure_utc(ts: datetime) -> datetime:
        """
        Normalise a timestamp to UTC.  Handles:
        * Already-UTC aware datetimes  → returned as-is.
        * Non-UTC aware datetimes      → converted to UTC.
        * Naive datetimes              → assumed UTC and tagged.
        """
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
