"""
Stop geometry: how does bracket width behave against real M15 price, at scale?

WHY THIS AND NOT THE MODEL REPLAY: the production model finished training
2026-07-02, so the genuinely-unseen window is ~5 weeks and yields FOUR
live-faithful trades. Nothing can be concluded from four. But "does a 1x-ATR
stop get hit by noise before a 2x-ATR target" is a property of PRICE, not of
the model -- so it can be measured on every bar in the cache instead of only
the four the model liked.

This measures the NOISE FLOOR: what a bracket of each width does on an
entry with no edge at all. Two things fall out:
  * the stop-hit rate a given geometry incurs from noise alone
  * the spread toll as a fraction of the risk taken

Then the model's four OOS trades can be read against that backdrop, and any
future retrain has a baseline to beat.

Long-only, matching the live strategy. Entry at bar close, SL-first on a
same-bar collision (conservative, same as the EV study), spread charged once
from the soak's measured medians.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/mnt/storage/mystuf/development/build-A-bot")
CACHE = ROOT / "analysis_cache" / "2026-07-27_m15_threshold_sweep"
OUT = Path(__file__).parent

TRADEABLE = ["GBP_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD"]
SPREAD_PCT = {
    "GBP_JPY": 0.01651, "AUD_JPY": 0.02014, "EUR_JPY": 0.01449,
    "NZD_JPY": 0.03686, "GBP_AUD": 0.02672, "GBP_NZD": 0.04046,
}
MAX_HOLD_BARS = 192
SL_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
TP_RATIOS = [2.0, 1.5, 1.0]
STRIDE = 1          # every bar


def outcomes_for(close, high, low, atr_abs, sl_mult, tp_mult, idxs):
    """Vectorised-per-bar bracket resolution. Returns arrays (r_gross, oc, bars).
    r_gross is the R-multiple BEFORE the spread toll (toll added by caller,
    since it depends only on risk)."""
    n = len(close)
    r = np.empty(len(idxs))
    oc = np.empty(len(idxs), dtype=np.int8)
    bars = np.empty(len(idxs))
    for k, i in enumerate(idxs):
        entry = close[i]
        atr = atr_abs[i]
        sl = entry - sl_mult * atr
        tp = entry + tp_mult * atr
        end = min(i + MAX_HOLD_BARS, n - 1)
        fl = low[i + 1:end + 1]
        fh = high[i + 1:end + 1]
        sl_hits = fl <= sl
        tp_hits = fh >= tp
        s_i = int(np.argmax(sl_hits)) if sl_hits.any() else 10**9
        t_i = int(np.argmax(tp_hits)) if tp_hits.any() else 10**9
        if s_i == 10**9 and t_i == 10**9:
            exit_px = close[end]
            r[k] = (exit_px - entry) / (entry - sl)
            oc[k] = 0
            bars[k] = end - i
        elif s_i <= t_i:               # SL first on a tie -- conservative
            r[k] = -1.0
            oc[k] = -1
            bars[k] = s_i + 1
        else:
            r[k] = tp_mult / sl_mult
            oc[k] = 1
            bars[k] = t_i + 1
    return r, oc, bars


rows = []
detail = {}
for sym in TRADEABLE:
    d = (pl.read_parquet(CACHE / "study_scored.parquet")
         .filter(pl.col("symbol") == sym).sort("timestamp"))
    close = d["close"].to_numpy().astype(float)
    high = d["high"].to_numpy().astype(float)
    low = d["low"].to_numpy().astype(float)
    natr = d["natr_14"].to_numpy().astype(float)
    atr_abs = natr / 100.0 * close
    sp = SPREAD_PCT[sym]

    valid = np.where(np.isfinite(atr_abs) & (atr_abs > 0))[0]
    valid = valid[(valid < len(close) - 2)][::STRIDE]
    print(f"  {sym}: {len(valid):,} entries", file=sys.stderr)

    for sl_mult in SL_MULTS:
        risk_pct = sl_mult * natr          # stop distance in percent of price
        for tp_ratio in TP_RATIOS:
            r_gross, oc, bars = outcomes_for(
                close, high, low, atr_abs, sl_mult, sl_mult * tp_ratio, valid)
            toll_r = sp / risk_pct[valid]  # spread as a fraction of risk
            r_net = r_gross - toll_r
            rows.append({
                "symbol": sym, "sl_mult": sl_mult, "tp_ratio": tp_ratio,
                "n": len(valid),
                "stop_rate": float((oc == -1).mean()),
                "targ_rate": float((oc == 1).mean()),
                "time_rate": float((oc == 0).mean()),
                "mean_r_gross": float(r_gross.mean()),
                "mean_r_net": float(r_net.mean()),
                "toll_r": float(toll_r.mean()),
                "median_bars": float(np.median(bars)),
            })
            if sl_mult in (1.0, 2.0) and tp_ratio == 2.0:
                detail[(sym, sl_mult)] = oc

res = pl.DataFrame(rows)
res.write_parquet(OUT / "stop_geometry.parquet")

print(f"\n{'='*104}")
print("NOISE FLOOR -- every bar as a hypothetical entry, pooled over the 6 tradeable crosses")
print("(no model, no edge: this is what bracket geometry alone does to real M15 price)")
print(f"{'='*104}")
print(f"{'SL':>5} {'TP:SL':>6} {'stop%':>7} {'targ%':>7} {'time%':>7} "
      f"{'grossR':>8} {'toll/R':>7} {'netR':>8} {'medBars':>8} {'break-even WR':>14}")
for sl_mult in SL_MULTS:
    for tp_ratio in TP_RATIOS:
        s = res.filter((pl.col("sl_mult") == sl_mult) & (pl.col("tp_ratio") == tp_ratio))
        w = s["n"].to_numpy().astype(float); w = w / w.sum()
        agg = lambda c: float((s[c].to_numpy() * w).sum())
        be = 1.0 / (1.0 + tp_ratio)     # win rate needed at this payoff, pre-cost
        print(f"{sl_mult:>5.2f} {tp_ratio:>6.1f} {agg('stop_rate'):>6.1%} "
              f"{agg('targ_rate'):>6.1%} {agg('time_rate'):>6.1%} "
              f"{agg('mean_r_gross'):>+8.4f} {agg('toll_r'):>6.1%} "
              f"{agg('mean_r_net'):>+8.4f} {agg('median_bars'):>8.0f} {be:>13.1%}")

print(f"\n{'='*104}")
print("PER-INSTRUMENT spread toll as a fraction of risk (why NZD_JPY and GBP_NZD hurt)")
print(f"{'='*104}")
print(f"{'symbol':>9} {'spread%':>9} " + " ".join(f"{m:>7.2f}x" for m in SL_MULTS))
for sym in TRADEABLE:
    s = res.filter((pl.col("symbol") == sym) & (pl.col("tp_ratio") == 2.0))
    tolls = {r["sl_mult"]: r["toll_r"] for r in s.iter_rows(named=True)}
    print(f"{sym:>9} {SPREAD_PCT[sym]:>8.4f}% "
          + " ".join(f"{tolls[m]:>7.1%}" for m in SL_MULTS))
