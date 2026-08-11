"""
The spread toll, measured properly.

stop_geometry.py reported the MEAN of (spread / risk). That statistic is
convex in risk and therefore dominated by quiet bars where the ATR is tiny --
exactly the bars Gate B exists to veto. This recomputes the toll as a
distribution, restricted to the population the live bot actually considers
(Gate B regime floor + Gate C blackout), and shows what Gate A does.

Key structural point checked here: Gate A is `sl_mult * natr >= k * spread`,
which is algebraically `toll <= 1/k`. With k = 1.5 the live system ADMITS any
trade whose spread eats up to 66.7% of the stop distance. Gate A is a toll
cap, and it is set very permissively.
"""
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path("/mnt/storage/mystuf/development/build-A-bot")
CACHE = ROOT / "analysis_cache" / "2026-07-27_m15_threshold_sweep"
TRADEABLE = ["GBP_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD"]
SPREAD_PCT = {
    "GBP_JPY": 0.01651, "AUD_JPY": 0.02014, "EUR_JPY": 0.01449,
    "NZD_JPY": 0.03686, "GBP_AUD": 0.02672, "GBP_NZD": 0.04046,
}
SL_MULTS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
SPREAD_K = 1.5

study = pl.read_parquet(CACHE / "study_scored.parquet")
d = study.filter(
    pl.col("symbol").is_in(TRADEABLE)
    & pl.col("gate_b") & pl.col("gate_c")          # live population
    & pl.col("natr_14").is_finite() & (pl.col("natr_14") > 0)
)
sp = d["symbol"].replace_strict(SPREAD_PCT, return_dtype=pl.Float64).to_numpy()
natr = d["natr_14"].to_numpy().astype(float)
print(f"population: {len(d):,} bars passing Gate B + Gate C\n")

print("SPREAD TOLL as a fraction of the risk taken (Gate B/C population)")
print(f"{'SL':>5} {'passGateA':>10} | {'--- toll among Gate-A survivors ---':^38}")
print(f"{'':>5} {'':>10} | {'p25':>8} {'median':>8} {'p75':>8} {'mean':>8}")
for m in SL_MULTS:
    risk = m * natr
    toll = sp / risk
    passes = risk >= SPREAD_K * sp        # == toll <= 1/1.5 = 66.7%
    t = toll[passes]
    print(f"{m:>5.2f} {passes.mean():>9.1%} | {np.percentile(t,25):>7.1%} "
          f"{np.percentile(t,50):>7.1%} {np.percentile(t,75):>7.1%} {t.mean():>7.1%}")

print(f"\nGate A admits any trade with toll <= 1/k = {1/SPREAD_K:.1%} of risk, by construction.\n")

print("PER-INSTRUMENT median toll among Gate-A survivors")
print(f"{'symbol':>9} " + " ".join(f"{m:>7.2f}x" for m in SL_MULTS) + f"  {'%bars passing A @1.0x':>22}")
for sym in TRADEABLE:
    s = d.filter(pl.col("symbol") == sym)
    nt = s["natr_14"].to_numpy().astype(float)
    ss = SPREAD_PCT[sym]
    meds, pass1 = [], None
    for m in SL_MULTS:
        risk = m * nt
        p = risk >= SPREAD_K * ss
        meds.append(np.percentile((ss / risk)[p], 50) if p.any() else float("nan"))
        if m == 1.0:
            pass1 = p.mean()
    print(f"{sym:>9} " + " ".join(f"{x:>7.1%}" for x in meds) + f"  {pass1:>21.1%}")

# What edge must the model supply to break even, at each width?
print("\nREQUIRED EDGE: mean R the model must generate to break even after the toll")
print("(a 2:1 bracket on driftless price returns ~0.00 R gross; the model must beat the toll)")
print(f"{'SL':>5} {'median toll':>12} {'=> needs win rate':>20}   (2:1 payoff, vs 33.3% at zero cost)")
for m in SL_MULTS:
    risk = m * natr
    toll = sp / risk
    passes = risk >= SPREAD_K * sp
    med = float(np.percentile(toll[passes], 50))
    # win*2 - (1-win)*1 - toll = 0  ->  win = (1 + toll)/3
    need = (1.0 + med) / 3.0
    print(f"{m:>5.2f} {med:>11.1%} {need:>19.1%}")
