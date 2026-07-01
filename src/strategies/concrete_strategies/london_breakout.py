"""
London-Open volatility breakout on GBP/JPY — deterministic, NO machine learning.

The idea (Gemini's spec, the 2026-06-26 pivot):
  * During the quiet overnight ("Asian") hours, GBP/JPY tends to coil into a
    narrow range. We mark that range's High and Low.
  * When the London session opens, volatility expands and price often breaks
    out of that range and keeps going. We take the *first* break of each side:
      - break above the Asian High  -> go long  at the High
      - break below the Asian Low   -> go short at the Low
  * Fixed 1:2 risk/reward. The stop is either the opposite edge of the Asian
    range or its midpoint (two variants we test side by side). The target is
    twice the stop distance.

This module is intentionally self-contained: it owns its own stop/target and
exit logic for the BACKTEST. It does NOT subclass BaseStrategy / emit live
Signals — live wiring (mapping 1:2 RR onto the RiskManager's ATR multipliers,
per the project's TP-distance ruling) is a separate, later step.

Everything here is pure polars / numpy. No LightGBM, no sklearn, no src/ml.

Session windows are FIXED UTC (confirmed with Brandon 2026-06-26): the windows
do NOT shift with British Summer Time. This is the simplest, fully-reproducible
choice for v1; in summer the window sits ~1h off local London time, which is a
known, documented limitation to revisit if the edge looks real.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

# ── Session windows (fixed UTC, half-open intervals) ──────────────────────────
ASIAN_START_HOUR = 0     # 00:00 UTC inclusive
ASIAN_END_HOUR = 7       # 07:00 UTC exclusive  → Asian range = [00:00, 07:00)
LONDON_START_HOUR = 7    # 07:00 UTC inclusive
LONDON_END_HOUR = 15     # 15:00 UTC exclusive  → London window = [07:00, 15:00)

# GBP/JPY: one pip = 0.01 in price (JPY-quoted pair).
PIP = 0.01

# Reward-to-risk ratio (hardcoded 1:2 per spec).
REWARD_RISK = 2.0

StopVariant = Literal["opposite", "midpoint"]

# Trade record column schema (gross — cost applied later in summarize()).
TRADE_COLUMNS = [
    "date",
    "side",
    "entry_time",
    "entry",
    "stop",
    "target",
    "asian_high",
    "asian_low",
    "exit_time",
    "exit_price",
    "exit_reason",  # "target" | "stop" | "session_end"
    "risk_pips",
    "pnl_pips",     # GROSS, before transaction cost
]


def _with_session_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Attach UTC calendar date + hour, sorted by time."""
    return (
        df.sort("timestamp")
        .with_columns(
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("timestamp").dt.hour().alias("hour"),
        )
    )


def compute_asian_ranges(df: pl.DataFrame) -> pl.DataFrame:
    """
    Per-day Asian-session High/Low (the consolidation range).

    Returns one row per UTC date that has at least one Asian-session bar,
    with columns: date, asian_high, asian_low, asian_bars.
    """
    asian = df.filter(
        (pl.col("hour") >= ASIAN_START_HOUR) & (pl.col("hour") < ASIAN_END_HOUR)
    )
    return (
        asian.group_by("date")
        .agg(
            pl.col("high").max().alias("asian_high"),
            pl.col("low").min().alias("asian_low"),
            pl.len().alias("asian_bars"),
        )
        .sort("date")
    )


def _resolve_trade(
    side: str,
    entry: float,
    stop: float,
    asian_high: float,
    asian_low: float,
    day_date,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    times: list,
    first_idx: int,
    enable_trailing: bool,
    breakeven_multiplier: float,
    time_stop_bars: int,
) -> dict:
    """
    Walk the London bars from the breakout bar onward and resolve the exit.
    Supports trailing stops and time-based exits. No fixed target.
    """
    risk_pips = abs(entry - stop) / PIP
    exit_price = None
    exit_reason = None
    exit_time = None
    
    current_stop = stop
    moved_to_be = False

    for j in range(first_idx, len(highs)):
        hi, lo = highs[j], lows[j]
        
        # 1. Stop check (including trailing stop)
        if side == "long":
            if lo <= current_stop:
                exit_price, exit_reason, exit_time = current_stop, "stop", times[j]
                break
        else:
            if hi >= current_stop:
                exit_price, exit_reason, exit_time = current_stop, "stop", times[j]
                break
                
        # 2. Trailing logic (update for next bar)
        if enable_trailing:
            if side == "long":
                if not moved_to_be and hi >= entry + (breakeven_multiplier * risk_pips * PIP):
                    current_stop = entry
                    moved_to_be = True
                if moved_to_be and j > first_idx:
                    trail_level = lows[j-1]
                    if trail_level > current_stop:
                        current_stop = trail_level
            else:
                if not moved_to_be and lo <= entry - (breakeven_multiplier * risk_pips * PIP):
                    current_stop = entry
                    moved_to_be = True
                if moved_to_be and j > first_idx:
                    trail_level = highs[j-1]
                    if trail_level < current_stop:
                        current_stop = trail_level

        # 3. Time stop check (if we haven't exited, and not in profit after N bars)
        bars_held = j - first_idx
        if time_stop_bars > 0 and bars_held == time_stop_bars:
            if side == "long" and closes[j] <= entry:
                exit_price, exit_reason, exit_time = closes[j], "time_stop", times[j]
                break
            elif side == "short" and closes[j] >= entry:
                exit_price, exit_reason, exit_time = closes[j], "time_stop", times[j]
                break

    if exit_price is None:  # neither hit by session end → close at last London close
        exit_price, exit_reason, exit_time = closes[-1], "session_end", times[-1]

    if side == "long":
        pnl_pips = (exit_price - entry) / PIP
    else:
        pnl_pips = (entry - exit_price) / PIP

    return {
        "date": day_date,
        "side": side,
        "entry_time": times[first_idx],
        "entry": entry,
        "stop": stop, # initial stop for record
        "target": None,
        "asian_high": asian_high,
        "asian_low": asian_low,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "risk_pips": risk_pips,
        "pnl_pips": pnl_pips,
    }


def generate_trades(
    df: pl.DataFrame,
    stop_variant: StopVariant = "opposite",
    enable_trailing: bool = False,
    breakeven_multiplier: float = 1.0,
    time_stop_bars: int = 0,
    max_stop_pips: float = float('inf'),
) -> pl.DataFrame:
    """
    Run the breakout rule over a GBP/JPY OHLCV frame and return GROSS trades.

    Parameters
    ----------
    df : pl.DataFrame
        M15 bars with columns: timestamp (UTC tz-aware Datetime), open, high,
        low, close, volume. Multiple days.
    stop_variant : "opposite" | "midpoint"
        "opposite"  → stop at the far edge of the Asian range (wider, fewer
                      stop-outs). "midpoint" → stop at the range midpoint
                      (tighter risk, more stop-outs).
    enable_trailing : bool
        If True, moves stop to break-even at +1R and then trails previous bar.
    breakeven_multiplier : float
        Multiple of risk at which stop is moved to break-even.
    time_stop_bars : int
        If > 0, closes trade at market if not in profit after this many bars.
    max_stop_pips : float
        Caps the maximum initial stop distance to protect account.

    Returns
    -------
    pl.DataFrame
        One row per trade (up to one long + one short per day), GROSS of cost.
    """
    if stop_variant not in ("opposite", "midpoint"):
        raise ValueError(f"stop_variant must be 'opposite' or 'midpoint', got {stop_variant!r}")

    df = _with_session_columns(df)
    ranges = compute_asian_ranges(df)
    range_by_date = {
        row["date"]: (row["asian_high"], row["asian_low"])
        for row in ranges.iter_rows(named=True)
    }

    london = df.filter(
        (pl.col("hour") >= LONDON_START_HOUR) & (pl.col("hour") < LONDON_END_HOUR)
    )
    if london.is_empty():
        return pl.DataFrame(schema={c: None for c in TRADE_COLUMNS})

    trades: list[dict] = []

    for day in london.partition_by("date", maintain_order=True):
        day_date = day["date"][0]
        rng = range_by_date.get(day_date)
        if rng is None:
            continue  # no Asian range for this day → no setup
        asian_high, asian_low = rng
        if asian_high <= asian_low:
            continue  # degenerate range

        highs = day["high"].to_numpy()
        lows = day["low"].to_numpy()
        closes = day["close"].to_numpy()
        times = day["timestamp"].to_list()

        midpoint = (asian_high + asian_low) / 2.0

        # ── LONG: first bar whose high breaks above the Asian High ──────────
        long_hits = np.nonzero(highs > asian_high)[0]
        if long_hits.size:
            entry = asian_high
            stop = asian_low if stop_variant == "opposite" else midpoint
            
            # Cap stop to max risk
            if (entry - stop) / PIP > max_stop_pips:
                stop = entry - (max_stop_pips * PIP)

            trades.append(
                _resolve_trade(
                    "long", entry, stop, asian_high, asian_low,
                    day_date, highs, lows, closes, times, int(long_hits[0]),
                    enable_trailing, breakeven_multiplier, time_stop_bars
                )
            )

        # ── SHORT: first bar whose low breaks below the Asian Low ───────────
        short_hits = np.nonzero(lows < asian_low)[0]
        if short_hits.size:
            entry = asian_low
            stop = asian_high if stop_variant == "opposite" else midpoint
            
            # Cap stop to max risk
            if (stop - entry) / PIP > max_stop_pips:
                stop = entry + (max_stop_pips * PIP)
                
            trades.append(
                _resolve_trade(
                    "short", entry, stop, asian_high, asian_low,
                    day_date, highs, lows, closes, times, int(short_hits[0]),
                    enable_trailing, breakeven_multiplier, time_stop_bars
                )
            )

    if not trades:
        return pl.DataFrame(schema={c: None for c in TRADE_COLUMNS})

    return pl.DataFrame(trades).sort("entry_time")


def summarize(trades: pl.DataFrame, cost_pips: float = 0.0) -> dict:
    """
    Headline metrics for a set of trades at a given round-trip cost (in pips).

    Cost is subtracted once per trade (entry+exit spread+slippage rolled into
    one number). Drawdown is measured on the cumulative pip equity curve.

    Returns dict: total_trades, win_rate, profit_factor, max_drawdown_pips,
    expectancy_pips, net_pips, cost_pips.
    """
    n = trades.height
    if n == 0:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "max_drawdown_pips": 0.0, "expectancy_pips": 0.0,
            "net_pips": 0.0, "cost_pips": cost_pips,
        }

    net = trades["pnl_pips"].to_numpy() - cost_pips
    wins = net[net > 0]
    losses = net[net < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf")

    equity = np.cumsum(net)
    running_max = np.maximum.accumulate(equity)
    max_drawdown = float((running_max - equity).max())

    return {
        "total_trades": n,
        "win_rate": float(len(wins) / n),
        "profit_factor": profit_factor,
        "max_drawdown_pips": max_drawdown,
        "expectancy_pips": float(net.mean()),
        "net_pips": float(net.sum()),
        "cost_pips": cost_pips,
    }
