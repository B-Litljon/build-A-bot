"""
src/day_trading/targets.py
Target Labeling — Universal Scalper V4.0

Implements the End-of-Day (EOD) Angel/Devil meta-labeling targets.

    Angel Target  — Was the Max Favorable Excursion (MFE) from entry to EOD
                    ≥ 0.6 × daily_atr_abs?
                    (Was the move large enough to be a real intraday trend leg?)
                    ENTRY WINDOW: Only bars within the first 90 minutes of the
                    RTH session (09:30–11:00 ET, session_progress ≤ 0.2308) can
                    receive a positive label.  Bars outside this window are
                    forced to 0 — not null — so they remain as negative training
                    examples and the TA-Lib feature warmup is preserved.

    Devil Target  — Did the trade survive from entry to EOD without the
                    session low breaching  close − 0.75 × daily_atr_abs?
                    (A tighter stop forces the Devil to learn microstructure
                    defence; fixes the 99% degeneracy of the 1.5× stop.)

    Macro Target  — Angel AND Devil both satisfied.
                    Used ONLY for EV calibration and threshold sweep.
                    Never used to train the Devil — identical role to
                    `devil_target_macro` in V3.4 retrainer.py.

────────────────────────────────────────────────────────────────────────────────
MFE CALCULATION — Lookahead bias proof
────────────────────────────────────────────────────────────────────────────────
The MFE for bar i is defined as:

    MFE[i] = max(high[i+1], high[i+2], ..., high[EOD_bar]) − close[i]

where EOD_bar is the last 5-minute bar of the RTH session (nominally
closing at 15:55 ET, i.e. the bar with timestamp ~20:55 UTC).

Implementation choices and their tradeoffs:

  OPTION A — Polars window function (reversed suffix-max):
      Compute a "reverse cumulative max" of `high` within each
      (symbol, trade_date) group by sorting descending and using
      cum_max, then re-sorting ascending.

      Pros  : Fully vectorised; no Python loops; clean Polars expression.
      Cons  : Polars' `cum_max().over()` always accumulates in the
              direction of the sort; reversing requires an explicit sort
              flip.  It is clean but less readable.

  OPTION B — Polars `shift` and `rolling_max` with a fixed window:
      Use `rolling_max(window_size=N, center=False)` on a reversed Series.
      Requires knowing N (max bars to EOD), which varies per bar within
      the session (early bars have up to 77 bars of lookahead; late bars
      have 1).

      Cons  : `rolling_max` is forward-looking only if applied to a
              reversed Series; this is conceptually awkward.

  OPTION C — Polars `group_by` suffix aggregation (CHOSEN):
      For each session, reverse-sort by timestamp, compute a
      cumulative max of `high` and a cumulative min of `low`, then
      re-sort ascending.  This gives for each bar i the max/min of all
      bars from i+1 to EOD without any Python loop.

      The key property: after the sort-descending / cum_max / sort-ascending
      pipeline, row i holds max(high[i], high[i+1], ..., high[EOD]).
      We then subtract high[i] itself via `shift(-1, fill_value=null)` on
      the reversed cum_max to get the "future" max only (not including
      bar i's own high).

      This is fully vectorised and has no Python-level iteration.

  OPTION D — NumPy bar-by-bar within each session (ALSO IMPLEMENTED as fallback):
      Iterate over sessions in Python; within each session, iterate over
      bars.  O(n²) per session but n ≤ 78 per session → negligible.
      Clearer correctness guarantees — used as the reference implementation.

We implement OPTION C for performance with OPTION D as the documented
reference so the correctness of C can be verified.

────────────────────────────────────────────────────────────────────────────────
Lookahead-Free Guarantee for the Vectorised Path (Option C)
────────────────────────────────────────────────────────────────────────────────
The reversed-cum_max approach works as follows for a session with bars 1..N:

  Sort descending → rows are [bar_N, bar_{N-1}, ..., bar_1]
  cum_max("high") → row k holds max(high[bar_N], ..., high[bar_{N-k+1}])
  After re-sorting ascending → row i holds max(high[i], ..., high[N])

  This includes bar i's own `high`.  We want max(high[i+1..N]).
  Solution: shift the reversed-cummax forward by 1 (within the descending
  sort) before re-sorting — equivalently, take the reversed-cummax of
  high[i+1..N] by offsetting the window by one position.

  Concrete implementation:
      (a) Sort each group descending by timestamp.
      (b) Compute `_future_high_max = high.shift(1).cum_max()` on the
          descending series.  `shift(1)` in descending order means "the
          bar that comes after me in chronological order".
      (c) Re-sort ascending.
      (d) Row i now holds max(high[i+1], ..., high[EOD]).
          Bar EOD itself (the last bar, now first in descending sort) gets
          null from shift(1) → cum_max returns null → kept as NaN / dropped
          by clean_data().

This is equivalent to the bar-by-bar NumPy result and contains zero
lookahead: no bar sees any information beyond what was available at its
own timestamp.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import polars as pl

from ml.core.interfaces import BaseTargetGenerator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TARGET PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

#: Angel fires when MFE ≥ ANGEL_MFE_MULT × daily_atr_abs
#: Reduced from 1.0 → 0.6: a 0.6× ATR move from the opening 90-min window
#: is achievable on 15–30% of bars, producing a trainable class balance.
ANGEL_MFE_MULT: float = 0.6

#: Devil survives when session never breaches close − DEVIL_SL_MULT × daily_atr_abs
#: Tightened from 1.5 → 0.75 → 0.4:
#:   0.4× daily ATR is tight enough that 25–40% of entry-window bars get stopped out
#:   intraday, pushing devil_target into the 60–75% survival range.
#:   For SPY (~0.7% daily ATR): stop ≈ 0.28% below entry — breached on ~30% of days.
#:   For TSLA (~2.5% daily ATR): stop ≈ 1.0% below entry — breached on ~35–45% of days.
DEVIL_SL_MULT: float = 0.4

#: Used for EV calculation in the validation gate only
#: EV = win_rate × (TP_MULT / SL_MULT) − (1 − win_rate)
#: R:R = 0.6 / 0.4 = 1.5  break-even win-rate = 0.4 / (0.6 + 0.4) = 40.0%
#: Favourable R:R — wins pay 1.5× what losses cost.
TP_MULT: float = 0.6
SL_MULT: float = 0.4

#: Entry window: only bars within the first 90 minutes of the RTH session
#: (09:30–11:00 ET) can receive angel_target = 1.
#: session_progress at 11:00 ET = (660 − 570) / 390 = 90 / 390 ≈ 0.2308
ENTRY_WINDOW_MAX_PROGRESS: float = 90.0 / 390.0  # ≈ 0.2308


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════


class DayTradeTargets(BaseTargetGenerator):
    """
    Generates three target columns for day-trade model training.

    Output columns:
        angel_target        (Int8 / null) — 1 if MFE ≥ ANGEL_MFE_MULT × daily_atr_abs
        devil_target        (Int8 / null) — 1 if trade survives to EOD without SL
        devil_target_macro  (Int8 / null) — 1 if angel AND devil both satisfied
                                            (EV calibration only; NOT for training)

    The final bar of each session (EOD bar) receives null in all three columns —
    it has no lookahead window.  These nulls are dropped by
    FeaturePipeline.clean_data() along with the TA-Lib warm-up NaN rows.

    Prerequisite columns: high, low, close, daily_atr_abs, trade_date
    Optional column     : symbol (enables per-symbol isolation)
    """

    def __init__(
        self,
        angel_mfe_mult: float = ANGEL_MFE_MULT,
        devil_sl_mult: float = DEVIL_SL_MULT,
    ) -> None:
        self.angel_mfe_mult = angel_mfe_mult
        self.devil_sl_mult = devil_sl_mult

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Dispatch to the vectorised Polars implementation per symbol.

        Processing per symbol prevents any cross-symbol contamination in the
        window functions (e.g., TSLA's last bar being treated as SPY's next bar).
        """
        has_symbol = "symbol" in df.columns

        if has_symbol:
            parts: List[pl.DataFrame] = []
            for sym in df["symbol"].unique().sort().to_list():
                sym_df = df.filter(pl.col("symbol") == sym)
                sym_df = self._label_symbol(sym_df)
                parts.append(sym_df)
            return pl.concat(parts, how="vertical_relaxed")

        return self._label_symbol(df)

    # ─────────────────────────────────────────────────────────────────────────
    def _label_symbol(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute all three targets for a single symbol's full date range.

        Iterates over each unique trade_date (session) and calls
        `_label_session`, which is the vectorised Polars session-level labeler.
        """
        df = df.sort("timestamp")
        sessions: List[pl.DataFrame] = []

        for date in df["trade_date"].unique().sort().to_list():
            session = df.filter(pl.col("trade_date") == date)
            session = self._label_session(session)
            sessions.append(session)

        return pl.concat(sessions, how="vertical_relaxed")

    # ─────────────────────────────────────────────────────────────────────────
    def _label_session(self, session: pl.DataFrame) -> pl.DataFrame:
        """
        Vectorised Polars session-level target labeling.

        Uses the reversed-cum_max / reversed-cum_min approach (Option C)
        described in the module docstring.

        MFE (future max high from bar i+1 to EOD):
        ─────────────────────────────────────────
          1. Sort session descending by timestamp.
          2. `shift(1)` the `high` column — in descending order, shift(1)
             moves bar i's value to bar i-1's slot, so bar i now holds
             bar i+1's high (chronologically).
          3. `cum_max()` on this shifted column — bar i now holds
             max(high[i+1], ..., high[EOD]).
          4. Re-sort ascending.
          5. MFE[i] = future_high_max[i] − close[i]

        Session-low from bar i+1 to EOD (for Devil SL check):
        ──────────────────────────────────────────────────────
          Same pattern using shift(1) + cum_min on a descending sort.

        EOD bar handling:
          The last chronological bar (first in descending sort) gets null
          from shift(1) → cum_max/cum_min propagate null → all three targets
          are null for the EOD bar.  clean_data() drops these rows.
        """
        n = len(session)
        if n == 0:
            return session.with_columns(
                pl.lit(None, dtype=pl.Int8).alias("angel_target"),
                pl.lit(None, dtype=pl.Int8).alias("devil_target"),
                pl.lit(None, dtype=pl.Int8).alias("devil_target_macro"),
            )

        session = session.sort("timestamp")  # ascending — canonical order

        # ── Descending sort to build suffix aggregates ────────────────────────
        session_desc = session.sort("timestamp", descending=True)

        # shift(1) in descending order = "the chronologically next bar"
        session_desc = session_desc.with_columns(
            pl.col("high").shift(1).alias("_high_shifted"),
            pl.col("low").shift(1).alias("_low_shifted"),
        )

        # cum_max of shifted high = max(high[i+1..EOD]) for each bar i
        # cum_min of shifted low  = min(low[i+1..EOD])  for each bar i
        session_desc = session_desc.with_columns(
            pl.col("_high_shifted").cum_max().alias("_future_high_max"),
            pl.col("_low_shifted").cum_min().alias("_future_low_min"),
        )

        # Re-sort ascending to align with close / daily_atr_abs
        session_asc = session_desc.sort("timestamp")

        # Merge the suffix aggregates back into the ascending session frame
        # using a direct column assignment (same row-order after sort).
        session = session.with_columns(
            session_asc["_future_high_max"],
            session_asc["_future_low_min"],
        )

        # ── Angel Target ──────────────────────────────────────────────────────
        #
        # MFE[i] = _future_high_max[i] − close[i]
        #
        # Entry-window mask (v3 — DROP strategy):
        #   Bars outside the first 90 minutes (session_progress > 0.2308) receive
        #   null — not 0.  drop_nulls() in clean_data() then removes them entirely.
        #
        #   Why null not 0:
        #   When non-window bars were forced to 0, the labeled universe was 143K rows
        #   with only 8% positive — diluted by 77% structurally-negative rows that
        #   the model cannot learn meaningful signal from.  With null, only the 23%
        #   of bars that are inside the entry window survive clean_data(), and within
        #   that subset ~35% are angel_target=1, which lands cleanly in the 15–30%
        #   target range.  Devil and Macro are also nulled outside the window so all
        #   three targets drop together — no row survives with a mix of valid/null targets.
        #
        # Decision order (highest priority first):
        #   1. null — EOD bar (no lookahead window)
        #   2. null — outside 09:30–11:00 ET entry window → row is dropped
        #   3. 1    — inside window AND MFE ≥ 0.6 × daily_atr_abs
        #   4. 0    — inside window but MFE < threshold
        session = session.with_columns(
            (pl.col("_future_high_max") - pl.col("close")).alias("_mfe")
        )
        session = session.with_columns(
            pl.when(pl.col("_mfe").is_null())
            .then(pl.lit(None, dtype=pl.Int8))  # EOD bar
            .when(pl.col("session_progress") > ENTRY_WINDOW_MAX_PROGRESS)
            .then(pl.lit(None, dtype=pl.Int8))  # outside window → DROP
            .when(pl.col("_mfe") >= self.angel_mfe_mult * pl.col("daily_atr_abs"))
            .then(pl.lit(1, dtype=pl.Int8))  # MFE target hit
            .otherwise(pl.lit(0, dtype=pl.Int8))  # MFE target missed
            .alias("angel_target")
        )

        # ── Devil Target ──────────────────────────────────────────────────────
        #
        # sl_price[i] = close[i] − devil_sl_mult × daily_atr_abs[i]
        # devil_target[i] = 1    if  _future_low_min[i] > sl_price[i]
        #                           (SL never breached from i+1 to EOD)
        #                 = 0    if  _future_low_min[i] ≤ sl_price[i]
        #                 = null if  _future_low_min is null (EOD bar)
        #                 = null if  outside entry window (aligned with Angel — row is dropped)
        session = session.with_columns(
            (pl.col("close") - self.devil_sl_mult * pl.col("daily_atr_abs")).alias(
                "_sl_price"
            )
        )
        session = session.with_columns(
            pl.when(pl.col("_future_low_min").is_null())
            .then(pl.lit(None, dtype=pl.Int8))  # EOD bar
            .when(pl.col("session_progress") > ENTRY_WINDOW_MAX_PROGRESS)
            .then(pl.lit(None, dtype=pl.Int8))  # outside window → DROP
            .when(pl.col("_future_low_min") > pl.col("_sl_price"))
            .then(pl.lit(1, dtype=pl.Int8))  # survived
            .otherwise(pl.lit(0, dtype=pl.Int8))  # stopped out
            .alias("devil_target")
        )

        # ── Macro Target ──────────────────────────────────────────────────────
        #
        # devil_target_macro[i] = 1  if  angel_target[i] = 1
        #                              AND devil_target[i] = 1
        #                       = 0  otherwise
        #                       = null if either component is null
        session = session.with_columns(
            pl.when(pl.col("angel_target").is_null() | pl.col("devil_target").is_null())
            .then(pl.lit(None, dtype=pl.Int8))
            .when((pl.col("angel_target") == 1) & (pl.col("devil_target") == 1))
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("devil_target_macro")
        )

        # Drop internal staging columns.
        # NOTE: _high_shifted and _low_shifted lived only in session_desc
        # (the descending working copy) and were never merged into session,
        # so they must not be listed here.
        session = session.drop(
            [
                "_future_high_max",
                "_future_low_min",
                "_mfe",
                "_sl_price",
            ]
        )

        return session

    # ─────────────────────────────────────────────────────────────────────────
    # REFERENCE IMPLEMENTATION (NumPy bar-by-bar)
    # ─────────────────────────────────────────────────────────────────────────

    def _label_session_numpy(self, session: pl.DataFrame) -> pl.DataFrame:
        """
        NumPy bar-by-bar reference implementation.

        Produces identical output to `_label_session` but uses explicit
        Python iteration for clarity.  Use this to verify the vectorised
        path during development:

            for date in df["trade_date"].unique():
                sess = df.filter(pl.col("trade_date") == date)
                r1 = targets._label_session(sess)
                r2 = targets._label_session_numpy(sess)
                assert (r1["angel_target"].drop_nulls() ==
                        r2["angel_target"].drop_nulls()).all()

        Complexity: O(n²) per session.  With n ≤ 78 RTH bars, worst case is
        78 × 78 = 6,084 comparisons — negligible at training time.
        """
        session = session.sort("timestamp")
        n = len(session)

        close = session["close"].to_numpy()
        high = session["high"].to_numpy()
        low = session["low"].to_numpy()
        daily_atr = session["daily_atr_abs"].to_numpy()
        sess_prog = session["session_progress"].to_numpy()

        angel_arr = np.full(n, np.nan, dtype=np.float32)
        devil_arr = np.full(n, np.nan, dtype=np.float32)
        macro_arr = np.full(n, np.nan, dtype=np.float32)

        for i in range(n - 1):  # last bar (EOD) stays NaN
            atr = daily_atr[i]
            if np.isnan(atr) or atr <= 0.0:
                continue  # insufficient daily data; clean_data drops it

            entry = close[i]
            sl = entry - self.devil_sl_mult * atr
            in_window = sess_prog[i] <= ENTRY_WINDOW_MAX_PROGRESS

            # Outside entry window: all three targets stay NaN → dropped by clean_data
            if not in_window:
                continue

            # Lookahead window: bars i+1 … n-1 (remainder of session)
            future_highs = high[i + 1 :]
            future_lows = low[i + 1 :]

            # Angel: MFE ≥ 0.6× daily ATR (already in-window, checked above)
            mfe = float(np.max(future_highs)) - entry
            angel_hit = mfe >= self.angel_mfe_mult * atr

            # Devil: did the session-low breach close − 0.4× daily ATR by EOD?
            sl_hit = bool(np.any(future_lows <= sl))
            devil_surv = not sl_hit

            angel_arr[i] = np.float32(1) if angel_hit else np.float32(0)
            devil_arr[i] = np.float32(1) if devil_surv else np.float32(0)
            macro_arr[i] = (
                np.float32(1) if (angel_hit and devil_surv) else np.float32(0)
            )

        return session.with_columns(
            pl.Series("angel_target", angel_arr).cast(pl.Int8),
            pl.Series("devil_target", devil_arr).cast(pl.Int8),
            pl.Series("devil_target_macro", macro_arr).cast(pl.Int8),
        )
