"""
Stop-multiple sweep: does widening the stop rescue trades that noise kills?

Re-simulates the SAME candidate bars as the 2026-07-27 threshold EV study,
varying the stop multiple. Reuses that study's cached scored.parquet (per-bar
angel/devil probabilities + OHLC + regime rank), so no model inference and no
OANDA calls are needed -- the probabilities do not depend on the bracket.

What DOES depend on the stop multiple, and is therefore recomputed here:
  * the simulated outcome (obviously)
  * Gate A, the transaction-cost floor: sl_mult x natr >= k x spread. A wider
    stop clears this more easily, so wider stops admit MORE trades. Reusing
    the cached gate_a column (computed at 1.0x) would understate them.

Reported per setting, out-of-sample only:
  * stop-hit / target-hit / timeout rates
  * mean after-cost return per trade, two ways:
      - pct     : fixed position size (what the account did)
      - R       : risk-normalised, pnl / stop-distance. This is the fair
                  comparison. At a fixed dollar risk per trade you shrink
                  size as the stop widens, so R-expectancy is what compounds;
                  raw pct would flatter wide stops purely for risking more.
  * profit factor, and the spread toll as a fraction of the stop distance
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/mnt/storage/mystuf/development/build-A-bot")
CACHE = ROOT / "analysis_cache" / "2026-07-27_m15_threshold_sweep"

TRADEABLE = ["GBP_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD"]
SPREAD_PCT = {
    "XAU_USD": 0.01496, "XAG_USD": 0.05900,
    "GBP_JPY": 0.01651, "AUD_JPY": 0.02014, "EUR_JPY": 0.01449,
    "NZD_JPY": 0.03686, "GBP_AUD": 0.02672, "GBP_NZD": 0.04046,
}
OOS_BOUNDARY = datetime(2026, 7, 3, tzinfo=timezone.utc)
MAX_HOLD_BARS = 192       # 2 days of M15; live has no time exit -- cap the sim
ANGEL_THR, DEVIL_THR = 0.40, 0.48
SPREAD_K = 1.5            # Gate A coefficient (RiskProfile.spread_k_base)
REGIME_P = 0.20           # Gate B floor (RiskProfile.regime_pctile / 100)

SL_MULTS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
TP_RATIOS = [2.0, 1.5, 1.0]   # tp_mult = sl_mult * ratio


def simulate(idx, close, high, low, atr_abs_arr, spread_pct, sl_mult, tp_mult):
    """Long bracket outcome. Mirrors the EV study: entry at bar close,
    SL-first on a same-bar collision (conservative), spread charged once.
    Returns (pnl_pct, r_multiple, outcome) with outcome 1 TP / -1 SL / 0 timeout."""
    entry = close[idx]
    atr = atr_abs_arr[idx]
    sl = entry - sl_mult * atr
    tp = entry + tp_mult * atr
    risk_pct = (entry - sl) / entry * 100.0     # stop distance, in percent
    end = min(idx + MAX_HOLD_BARS, len(close) - 1)
    for j in range(idx + 1, end + 1):
        if low[j] <= sl:
            pnl = (sl - entry) / entry * 100.0 - spread_pct
            return pnl, pnl / risk_pct, -1
        if high[j] >= tp:
            pnl = (tp - entry) / entry * 100.0 - spread_pct
            return pnl, pnl / risk_pct, 1
    pnl = (close[end] - entry) / entry * 100.0 - spread_pct
    return pnl, pnl / risk_pct, 0


study = pl.read_parquet(CACHE / "study_scored.parquet")
print(f"loaded {len(study):,} scored bars, "
      f"{study['timestamp'].min():%Y-%m-%d} -> {study['timestamp'].max():%Y-%m-%d}",
      file=sys.stderr)

rows = []
per_trade = []
for sym in TRADEABLE:
    d = study.filter(pl.col("symbol") == sym).sort("timestamp")
    ts = d["timestamp"].to_list()
    close = d["close"].to_numpy().astype(float)
    high = d["high"].to_numpy().astype(float)
    low = d["low"].to_numpy().astype(float)
    natr = d["natr_14"].to_numpy().astype(float)
    ap = d["angel_prob"].to_numpy()
    dp = d["devil_prob"].to_numpy()
    rank = d["regime_rank"].to_numpy()
    gate_c = d["gate_c"].to_numpy()
    sp = SPREAD_PCT[sym]
    atr_abs = natr / 100.0 * close

    # model verdict + the two stop-independent gates
    base_ok = (ap >= ANGEL_THR) & (dp >= DEVIL_THR) & gate_c
    gate_b = np.isnan(rank) | (rank >= REGIME_P)
    base_ok &= gate_b

    for sl_mult in SL_MULTS:
        gate_a = (sl_mult * natr) >= (SPREAD_K * sp)   # depends on the stop!
        idxs = np.where(base_ok & gate_a)[0]
        idxs = idxs[idxs < len(close) - 1]
        for tp_ratio in TP_RATIOS:
            tp_mult = sl_mult * tp_ratio
            for i in idxs:
                pnl, r, oc = simulate(i, close, high, low, atr_abs, sp,
                                      sl_mult, tp_mult)
                per_trade.append({
                    "symbol": sym, "timestamp": ts[i], "sl_mult": sl_mult,
                    "tp_ratio": tp_ratio, "pnl_pct": pnl, "r": r,
                    "outcome": oc, "risk_pct": sl_mult * natr[i],
                    "spread_pct": sp,
                })

t = pl.DataFrame(per_trade)
t.write_parquet(Path(__file__).parent / "stop_sweep_trades.parquet")


def table(d: pl.DataFrame, label: str):
    print(f"\n{'='*96}\n{label}\n{'='*96}")
    print(f"{'SL':>5} {'TP:SL':>6} {'n':>5} {'stop%':>6} {'targ%':>6} "
          f"{'time%':>6} {'meanR':>8} {'totR':>8} {'mean%':>8} {'PF':>6} "
          f"{'toll/R':>7}")
    for sl_mult in SL_MULTS:
        for tp_ratio in TP_RATIOS:
            s = d.filter((pl.col("sl_mult") == sl_mult)
                         & (pl.col("tp_ratio") == tp_ratio))
            n = len(s)
            if n == 0:
                continue
            oc = s["outcome"].to_numpy()
            r = s["r"].to_numpy()
            p = s["pnl_pct"].to_numpy()
            wins = p[p > 0].sum()
            losses = -p[p < 0].sum()
            pf = wins / losses if losses > 0 else float("inf")
            toll = float((s["spread_pct"] / s["risk_pct"]).mean())
            print(f"{sl_mult:>5.2f} {tp_ratio:>6.1f} {n:>5} "
                  f"{(oc==-1).mean()*100:>5.1f}% {(oc==1).mean()*100:>5.1f}% "
                  f"{(oc==0).mean()*100:>5.1f}% {r.mean():>8.3f} {r.sum():>8.2f} "
                  f"{p.mean():>+8.4f} {pf:>6.2f} {toll:>6.1%}")


oos = t.filter(pl.col("timestamp") >= OOS_BOUNDARY)
ins = t.filter(pl.col("timestamp") < OOS_BOUNDARY)
table(oos, "OUT OF SAMPLE (after 2026-07-03) -- the one that counts")
table(ins, "IN SAMPLE (before 2026-07-03) -- context only, model saw this")
table(t, "POOLED (both) -- larger n, but contaminated by in-sample")
