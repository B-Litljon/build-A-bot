#!/usr/bin/env python3
"""
Re-price the live decision ledger under wider bracket geometry, band by band.

WHY: two measurements exist but have never been combined.
  * The 2026-09-14 session measured that a constant wide bracket
    (10.25xATR stop / 2.74xATR target) cuts the spread toll from ~0.25R to
    ~0.04R per trade -- the only lever with a measured expectancy effect --
    yet the arm still measured negative net of cost.
  * The 2026-09-20 decision report shows the model's score has a small real
    edge in the upper-middle of its distribution (top decile +2.2pp over
    base, p=0.03) and no edge at the extreme top, exactly where the live bar
    (0.3833) sits.
This script asks the only question that combines the two: at the wide
geometry, which score band (if any) is positive net of the toll?

CONVENTIONS (deliberately identical to the surviving 2026-08-08 studies):
  * long-only, entry at bar close, SL checked first on a same-bar collision
  * spread charged once, per the instrument's median (soak SPREAD_CALIB)
  * R is risk-normalised: pnl / stop distance. Target hit = +tp/sl R,
    stop hit = -1R, timeout exits at the last bar's close.
  * outcome LABEL for validation: timeout counts as a loss -- this matches
    retrainer._compute_devil_targets_atr, the answer key behind the ledger.

VALIDATION: the static arm (2.0/4.0/45) must reproduce the ledger's own `won`
column. If it does not, the bars, ATR or walk convention differ from the
report's answer key and the wide numbers cannot be trusted; the script says so
and exits non-zero.

Glossary:
    LEDGER -- logs/graded_decisions.parquet: every bar the soak evaluated, with
        the live Angel score and the static-bracket win flag.
    GEOMETRIES -- the arms re-priced per row. `static` is the live control;
        `wide` is the 2026-09-14 constant-wide arm (10.25/2.74/45);
        `wide_192` and `best8x` are robustness references (longer hold, and
        the best random cell of the 60-geometry sweep).
    toll_r -- per-instrument median spread as a fraction of the stop distance,
        i.e. the cost in R units at that geometry.
    net_r -- gross outcome R minus toll_r; the currency the gate judges.
    q -- within-symbol score quintile (1 = lowest).
    top_decile -- within-symbol top 10% of Angel scores.
    certified -- rows where the Angel and Devil both approved
        (verdict agreement) or the Devil vetoed an Angel approval
        (devil_veto); the population the live bar actually certifies.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import talib  # noqa: E402

from data.oanda_provider import OandaMarketProvider  # noqa: E402

LEDGER = ROOT / "logs" / "graded_decisions.parquet"
OUT_DIR = ROOT / "analysis_cache" / "2026-09-20_reprice_band_geometry"

SYMBOLS = ["GBP_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD"]
SPREAD_PCT = {
    "GBP_JPY": 0.01651,
    "AUD_JPY": 0.02014,
    "EUR_JPY": 0.01449,
    "NZD_JPY": 0.03686,
    "GBP_AUD": 0.02672,
    "GBP_NZD": 0.04046,
}
FETCH_START = datetime(2026, 7, 1, tzinfo=timezone.utc)
NVOL = 14

GEOMETRIES = [
    ("static", 2.0, 4.0, 45),
    ("wide", 10.25, 2.74, 45),
    ("wide_192", 10.25, 2.74, 192),
    ("best8x", 8.0, 1.0, 90),
]
BANDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 1.01]
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]


def load_bars(symbol: str, provider_box: list, refresh: bool) -> pl.DataFrame:
    cache = OUT_DIR / f"{symbol}_M15.parquet"
    if cache.exists() and not refresh:
        return pl.read_parquet(cache)
    if provider_box[0] is None:
        provider_box[0] = OandaMarketProvider(environment="practice")
    df = provider_box[0].get_historical_bars(
        symbol, 15, FETCH_START, datetime.now(timezone.utc)
    )
    df = df.unique(subset=["timestamp"], keep="first").sort("timestamp")
    df.write_parquet(cache)
    return df


def walk(close, high, low, atr_abs, i, sl_mult, tp_mult, hold):
    """Returns (label, gross_r). timeout => label 0 (loss), r at market close."""
    entry = close[i]
    sl = entry - sl_mult * atr_abs[i]
    tp = entry + tp_mult * atr_abs[i]
    end = min(i + hold, len(close) - 1)
    for j in range(i + 1, end + 1):
        if low[j] <= sl:
            return 0, -1.0
        if high[j] >= tp:
            return 1, tp_mult / sl_mult
    return 0, (close[end] - entry) / (sl_mult * atr_abs[i])


def stats(s: pl.DataFrame) -> dict:
    if s.height == 0:
        return {
            "n": 0,
            "win": None,
            "gross_r": None,
            "toll_r": None,
            "net_r": None,
            "pf": None,
        }
    r = s["net_r"].to_numpy()
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "n": s.height,
        "win": round(float(s["label"].mean()), 4),
        "gross_r": round(float(s["gross_r"].mean()), 4),
        "toll_r": round(float(s["toll_r"].mean()), 4),
        "net_r": round(float(r.mean()), 4),
        "pf": round(pos / neg, 3) if neg > 0 else float("inf"),
    }


def table(df: pl.DataFrame, by: str) -> str:
    rows = []
    for cell in df[by].unique(maintain_order=True):
        rows.append({by: cell, **stats(df.filter(pl.col(by) == cell))})
    return str(pl.DataFrame(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--refresh", action="store_true", help="ignore the bar cache")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ledger = pl.read_parquet(LEDGER).filter(pl.col("symbol").is_in(SYMBOLS))
    ledger = ledger.with_columns(
        pl.col("bar_ts")
        .cast(pl.Utf8)
        .str.replace("T", " ")
        .str.slice(0, 16)
        .alias("key")
    )
    print(f"ledger: {ledger.height:,} fiat decisions", file=sys.stderr)

    provider_box: list = [None]
    per_row = []
    unmatched = 0
    for sym in SYMBOLS:
        bars = load_bars(sym, provider_box, args.refresh)
        close = bars["close"].to_numpy().astype(float)
        high = bars["high"].to_numpy().astype(float)
        low = bars["low"].to_numpy().astype(float)
        natr = talib.NATR(high, low, close, timeperiod=NVOL)
        atr_abs = natr / 100.0 * close
        keys = bars["timestamp"].dt.strftime("%Y-%m-%d %H:%M").to_list()
        key_to_i = {k: i for i, k in enumerate(keys)}
        sp = SPREAD_PCT[sym]
        for row in ledger.filter(pl.col("symbol") == sym).iter_rows(named=True):
            i = key_to_i.get(row["key"])
            if i is None or i >= len(close) - 1:
                unmatched += 1
                continue
            if not np.isfinite(natr[i]) or natr[i] <= 0:
                unmatched += 1
                continue
            rec = {
                "symbol": sym,
                "key": row["key"],
                "bar_ts": row["bar_ts"],
                "angel": float(row["angel"]),
                "verdict": row["verdict"],
                "ledger_won": int(row["won"]),
                "natr": float(natr[i]),
            }
            for name, sl_mult, tp_mult, hold in GEOMETRIES:
                label, gross = walk(close, high, low, atr_abs, i, sl_mult, tp_mult, hold)
                toll = sp / (sl_mult * rec["natr"])
                rec[f"{name}_label"] = label
                rec[f"{name}_gross"] = gross
                rec[f"{name}_toll"] = toll
                rec[f"{name}_net"] = gross - toll
            per_row.append(rec)

    df = pl.DataFrame(per_row)
    if unmatched:
        print(f"unmatched/invalid rows dropped: {unmatched:,}", file=sys.stderr)

    ok = (
        df.filter(pl.col("static_label") == pl.col("ledger_won")).height / df.height
    )
    print(
        f"\nVALIDATION: static walk reproduces ledger 'won' on "
        f"{ok:.2%} of {df.height:,} rows",
        file=sys.stderr,
    )
    if ok < 0.98:
        print(
            "VERDICT: UNRELIABLE -- static walk disagrees with the ledger; "
            "bars/ATR/convention differ from the report's answer key",
            file=sys.stderr,
        )
        return 2

    df = df.with_columns(
        [
            pl.col("angel").rank().over("symbol").alias("_rank"),
            pl.len().over("symbol").alias("_n"),
        ]
    ).with_columns(
        [
            (pl.col("_rank") / pl.col("_n") * 5).ceil().clip(1, 5).alias("q"),
            (pl.col("_rank") / pl.col("_n") > 0.9).alias("top_decile"),
            pl.col("verdict").is_in(["agreement", "devil_veto"]).alias("certified"),
        ]
    ).drop(["_rank", "_n"])

    out_parquet = OUT_DIR / "reprice_trades.parquet"
    df.write_parquet(out_parquet)

    summary_rows = []
    for name, sl_mult, tp_mult, hold in GEOMETRIES:
        arm = df.select(
            [
                pl.col("angel"),
                pl.col("q"),
                pl.col("top_decile"),
                pl.col("certified"),
                pl.col("verdict"),
                pl.col("symbol"),
                pl.col(f"{name}_label").alias("label"),
                pl.col(f"{name}_gross").alias("gross_r"),
                pl.col(f"{name}_toll").alias("toll_r"),
                pl.col(f"{name}_net").alias("net_r"),
            ]
        ).with_columns(
            pl.col("angel").cut(BANDS, left_closed=True).cast(pl.Utf8).alias("band")
        )
        print(
            f"\n{'=' * 88}\n{name}: sl={sl_mult}x tp={tp_mult}x hold={hold} "
            f"-- net_r = outcome R minus per-instrument spread toll\n{'=' * 88}"
        )
        print("\nBY ANGEL BAND")
        print(table(arm, "band"))
        print("\nBY WITHIN-SYMBOL QUINTILE")
        print(table(arm, "q"))
        parts = [
            arm.filter(pl.col("angel") >= thr).with_columns(pl.lit(thr).alias("thr"))
            for thr in THRESHOLDS
        ]
        print("\nPOOLED THRESHOLDS (angel >= thr)")
        print(table(pl.concat(parts), "thr"))
        print("\nCERTIFIED POPULATION (Angel+Devil agreed, or Devil vetoed an Angel)")
        print(table(arm.filter(pl.col("certified")), "verdict"))

        def cell(mask: pl.Expr) -> dict:
            return stats(arm.filter(mask))

        s, q5 = cell(pl.lit(True)), cell(pl.col("q") == 5)
        t10, cert = cell(pl.col("top_decile")), cell(pl.col("certified"))
        summary_rows.append(
            {
                "arm": name,
                "all_n": s["n"],
                "all_win": s["win"],
                "all_net": s["net_r"],
                "q5_n": q5["n"],
                "q5_win": q5["win"],
                "q5_net": q5["net_r"],
                "top10_n": t10["n"],
                "top10_win": t10["win"],
                "top10_net": t10["net_r"],
                "cert_n": cert["n"],
                "cert_win": cert["win"],
                "cert_net": cert["net_r"],
            }
        )

    print(f"\n{'=' * 88}\nSUMMARY -- net_r by arm and population\n{'=' * 88}")
    print(pl.DataFrame(summary_rows))
    print(
        "\ncolumns: q5_* = within-symbol top quintile; top10_* = top decile; "
        "cert_* = live-certified rows",
        file=sys.stderr,
    )
    print(f"\nwrote {out_parquet}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
