#!/usr/bin/env python3
"""
Standalone backtest for the London-Open breakout on GBP/JPY (no ML).

Deliberately isolated from src/ml and src/day_trading. Fetches GBP/JPY M15 from
OANDA (cached to parquet), runs the deterministic breakout rule across both stop
variants at 0 / 3 / 5 pips of cost, splits in-sample vs out-of-sample, and prints
honest headline metrics plus a <$400-account sizing reality check.

Run:
    set -a; . ./.env; set +a
    PYTHONPATH=src:. <venv-python> backtest_london.py

Flags:
    --years N      history to fetch / use (default 3)
    --oos-months M most-recent months held out as out-of-sample (default 12)
    --refresh      ignore the cache and re-fetch from OANDA
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import polars as pl

sys.path.insert(0, os.path.abspath("src"))

from data.oanda_provider import OandaMarketProvider
from strategies.concrete_strategies.london_breakout import (
    PIP,
    generate_trades,
    summarize,
)

SYMBOL = "GBP_JPY"
TIMEFRAME_MIN = 15
COSTS_PIPS = [0.0, 3.0, 5.0]
STOP_VARIANTS = ["opposite", "midpoint"]
CACHE_PATH = "data/raw/GBP_JPY_M15.parquet"


# ── data ──────────────────────────────────────────────────────────────────────
def load_bars(years: int, refresh: bool) -> pl.DataFrame:
    """Load GBP/JPY M15 bars, from cache when possible, else fetch + cache."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years + 5)

    if not refresh and os.path.exists(CACHE_PATH):
        df = pl.read_parquet(CACHE_PATH)
        covered_from = df["timestamp"].min()
        print(
            f"Loaded {df.height:,} cached M15 bars "
            f"({covered_from} → {df['timestamp'].max()}) from {CACHE_PATH}"
        )
        # Top up if the cache starts later than we now want.
        if covered_from is not None and covered_from <= start + timedelta(days=2):
            return df.filter(pl.col("timestamp") >= start)
        print("  cache doesn't reach far enough back — re-fetching.")

    print(f"Fetching {SYMBOL} M15 from OANDA: {start.date()} → {end.date()} ...")
    provider = OandaMarketProvider(environment="practice")
    df = provider.get_historical_bars(SYMBOL, TIMEFRAME_MIN, start, end)
    if df.is_empty():
        sys.exit("ERROR: OANDA returned no bars. Check OANDA_API_KEY / connectivity.")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.write_parquet(CACHE_PATH)
    print(f"Fetched {df.height:,} bars, cached to {CACHE_PATH}")
    return df


def latest_close(provider: OandaMarketProvider, symbol: str) -> float:
    """Most recent M15 close for a symbol (for the sizing reality check)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=3)
    df = provider.get_historical_bars(symbol, TIMEFRAME_MIN, start, end)
    return float(df["close"][-1])


# ── reporting ───────────────────────────────────────────────────────────────
def print_metrics_block(title: str, trades: pl.DataFrame) -> None:
    print(f"\n  {title}  (n={trades.height})")
    if trades.height == 0:
        print("    no trades")
        return
    longs = trades.filter(pl.col("side") == "long").height
    shorts = trades.height - longs
    print(f"    sides: {longs} long / {shorts} short")
    header = f"    {'cost(pips)':>10} {'trades':>7} {'win%':>6} {'PF':>6} {'maxDD(pips)':>12} {'exp(pips)':>10} {'net(pips)':>10}"
    print(header)
    for cost in COSTS_PIPS:
        m = summarize(trades, cost_pips=cost)
        pf = "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
        print(
            f"    {cost:>10.0f} {m['total_trades']:>7} {m['win_rate'] * 100:>5.1f}% "
            f"{pf:>6} {m['max_drawdown_pips']:>12.1f} {m['expectancy_pips']:>10.2f} {m['net_pips']:>10.1f}"
        )


def sizing_reality_check(trades: pl.DataFrame, gbpjpy: float, usdjpy: float) -> None:
    """Can a <$400 account even trade this without one stop-out breaking it?"""
    print("\n" + "=" * 78)
    print("  POSITION-SIZING REALITY CHECK  (<$400 account)")
    print("=" * 78)
    if trades.height == 0:
        print("  no trades to size")
        return

    median_stop = float(trades["risk_pips"].median())
    max_stop = float(trades["risk_pips"].max())
    # USD value of one pip on ONE unit of GBP/JPY = 0.01 JPY / (USD/JPY).
    pip_value_per_unit_usd = PIP / usdjpy
    gbpusd = gbpjpy / usdjpy
    account = 400.0

    print(f"  reference prices: GBP/JPY={gbpjpy:.3f}  USD/JPY={usdjpy:.3f}  (GBP/USD≈{gbpusd:.4f})")
    print(f"  typical stop: {median_stop:.0f} pips (median), {max_stop:.0f} pips (worst)")
    print(f"  {'risk/trade':>12} {'units':>8} {'notional$':>10} {'margin@50:1':>12} {'$/pip':>7}")
    for risk_pct in (0.01, 0.02):
        risk_usd = account * risk_pct
        loss_per_unit = median_stop * pip_value_per_unit_usd
        units = risk_usd / loss_per_unit if loss_per_unit > 0 else 0
        notional_usd = units * gbpusd
        margin = notional_usd / 50.0
        dollar_per_pip = units * pip_value_per_unit_usd
        print(
            f"  {risk_pct * 100:>10.0f}% {units:>8.0f} {notional_usd:>10.0f} "
            f"{margin:>12.2f} {dollar_per_pip:>7.3f}"
        )
    print("  (units sized so ONE median stop-out = the stated % of $400; OANDA allows")
    print("   any whole unit count, so fractional sizing isn't a blocker.)")


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--oos-months", type=int, default=12)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--dynamic", action="store_true", help="Enable dynamic trade management")
    ap.add_argument("--max-stop", type=float, default=60.0, help="Max initial stop in pips")
    ap.add_argument("--time-stop", type=int, default=4, help="Bars to hold before time stop")
    args = ap.parse_args()

    df = load_bars(args.years, args.refresh)
    if df.height == 0:
        sys.exit("ERROR: no bars in requested window.")

    span_start, span_end = df["timestamp"].min(), df["timestamp"].max()
    split = span_end - timedelta(days=30 * args.oos_months)
    print(
        f"\nData: {df.height:,} M15 bars, {span_start.date()} → {span_end.date()}"
        f"\nOut-of-sample split: {split.date()} "
        f"(in-sample before, last ~{args.oos_months}mo held out)\n"
    )

    for variant in STOP_VARIANTS:
        if args.dynamic:
            trades = generate_trades(
                df, stop_variant=variant, 
                enable_trailing=True, breakeven_multiplier=1.0, 
                time_stop_bars=args.time_stop, max_stop_pips=args.max_stop
            )
        else:
            trades = generate_trades(df, stop_variant=variant)
            
        in_sample = trades.filter(pl.col("entry_time") < split)
        oos = trades.filter(pl.col("entry_time") >= split)

        print("=" * 78)
        mode_str = "DYNAMIC MANAGEMENT" if args.dynamic else "STATIC 1:2 TARGET"
        print(f"  STOP VARIANT: {variant.upper()} [{mode_str}]"
              + ("  (stop at far edge of Asian range)" if variant == "opposite"
                 else "  (stop at Asian-range midpoint)"))
        print("=" * 78)
        print_metrics_block("FULL PERIOD", trades)
        print_metrics_block("IN-SAMPLE", in_sample)
        print_metrics_block("OUT-OF-SAMPLE", oos)

    # Sizing check uses the wider 'opposite' variant (worst-case stop width).
    provider = OandaMarketProvider(environment="practice")
    gbpjpy = latest_close(provider, "GBP_JPY")
    usdjpy = latest_close(provider, "USD_JPY")
    
    sizing_trades = generate_trades(
        df, stop_variant="opposite",
        enable_trailing=args.dynamic, breakeven_multiplier=1.0,
        time_stop_bars=args.time_stop if args.dynamic else 0,
        max_stop_pips=args.max_stop if args.dynamic else float('inf')
    )
    sizing_reality_check(sizing_trades, gbpjpy, usdjpy)
    print("\nDone. Read the OUT-OF-SAMPLE blocks at 3 and 5 pips — that's the honest test.")


if __name__ == "__main__":
    main()
