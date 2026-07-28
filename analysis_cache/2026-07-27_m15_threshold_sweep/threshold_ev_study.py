"""
Threshold EV study: after-cost profit per Angel threshold + Devil verdict.

Extends threshold_sweep.py (same models, same FeaturePipeline, same provider)
from counting signals to scoring MONEY: every candidate bar gets a simulated
live-faithful long bracket outcome — SL 1.0 x ATR / TP 2.0 x ATR (RiskProfile
forex), entry at bar close, SL-first on same-bar collision (conservative) —
charged the instrument's MEDIAN spread as measured by the 2026-07-13..27 soak
(SPREAD_CALIB, n~850 per instrument, clean non-holiday window). The live
gates are reproduced:

  Gate A (cost):   1.0 x ATR >= 1.5 x median spread   (median-spread approx;
                   live uses the live tick spread. Min-SL floors ignored —
                   they bind at ~1/5 of a typical M15 cross ATR.)
  Gate B (regime): natr_14 percentile rank >= P20 in trailing 260 bars,
                   gate stands down below 60 samples (cold-start bypass).
  Gate C (time):   16:55-17:30 America/New_York rollover blackout.

EV tables are OUT-OF-SAMPLE ONLY: bars after 2026-07-03 00:00 UTC (the
production pair finished training 2026-07-02 08:03 UTC), pooled over the six
broker-tradeable crosses — metals are scored but excluded from EV because
this account cannot trade them (INSTRUMENT_NOT_TRADEABLE, 2026-07-14).

The Devil verdict compares pass-vs-veto outcomes on Angel-approved rows,
in-sample and out-of-sample separately: a real filter shows an outcome gap;
a rubber stamp shows none.

Outputs: stdout tables + study_scored.parquet (per-bar probs, gates,
simulated outcomes) next to this script for later slicing.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

ROOT = Path("/mnt/storage/mystuf/development/build-A-bot")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from data.oanda_provider import OandaMarketProvider  # noqa: E402
from strategies.concrete_strategies.ml_strategy import MLStrategy  # noqa: E402
from execution.risk_manager import RiskProfile  # noqa: E402

OUT_DIR = ROOT / "analysis_cache" / "2026-07-27_m15_threshold_sweep"

SYMBOLS = [
    "XAU_USD", "XAG_USD", "GBP_JPY", "AUD_JPY",
    "EUR_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD",
]
# Metals are broker-dead on this account (INSTRUMENT_NOT_TRADEABLE) — scored
# for completeness, excluded from every EV table.
TRADEABLE = {"GBP_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD"}

# Median spread as % of price, from the soak's shutdown SPREAD_CALIB dump
# (2026-07-27, n≈830-870 per instrument, no holidays in window).
SPREAD_PCT = {
    "XAU_USD": 0.01496, "XAG_USD": 0.05900,
    "GBP_JPY": 0.01651, "AUD_JPY": 0.02014, "EUR_JPY": 0.01449,
    "NZD_JPY": 0.03686, "GBP_AUD": 0.02672, "GBP_NZD": 0.04046,
}

MODEL_DIR = ROOT / "models" / "forex_m15"
FETCH_START = datetime(2026, 4, 20, tzinfo=timezone.utc)
OOS_BOUNDARY = datetime(2026, 7, 3, 0, 0, tzinfo=timezone.utc)
NOW = datetime.now(timezone.utc)
THRESHOLDS = [0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.45]
SIM_FLOOR = 0.25          # simulate outcomes only where a trade is conceivable
MAX_HOLD_BARS = 192       # 2 days of M15; live has no time exit — cap the sim
NY = ZoneInfo("America/New_York")
BLACKOUT = (dtime(16, 55), dtime(17, 30))

devil_threshold = json.loads((MODEL_DIR / "threshold.json").read_text())["devil_threshold"]
profile = RiskProfile.for_asset_class("forex")
assert profile.sl_atr_multiplier == 1.0 and profile.tp_atr_multiplier == 2.0

strategy = MLStrategy(
    asset_class="forex",
    angel_path=MODEL_DIR / "angel_latest.pkl",
    devil_path=MODEL_DIR / "devil_latest.pkl",
    timeframe=15,
    htf_timeframe="1h",
    warmup_period=260,
    regime_window=profile.regime_window,
)
provider = OandaMarketProvider(environment="practice")


def regime_rank(natr: np.ndarray, window: int = 260, min_samples: int = 60) -> np.ndarray:
    """Percentile rank of each bar's NATR within its trailing window
    (inclusive, mirroring the live deque). NaN where the gate stands down."""
    n = len(natr)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        w = natr[lo:i + 1]
        if len(w) < min_samples:
            continue
        out[i] = float((w < natr[i]).mean())
    return out


def simulate(idx, close, high, low, natr_pct, spread_pct):
    """Long bracket outcome for bar idx. Returns (pnl_pct, outcome, bars_held).
    outcome: 1 TP, -1 SL, 0 timeout/end-of-data."""
    entry = close[idx]
    atr_abs = natr_pct[idx] / 100.0 * entry
    sl = entry - profile.sl_atr_multiplier * atr_abs
    tp = entry + profile.tp_atr_multiplier * atr_abs
    end = min(idx + MAX_HOLD_BARS, len(close) - 1)
    for j in range(idx + 1, end + 1):
        if low[j] <= sl:  # SL first on same-bar collision — conservative
            return ((sl - entry) / entry * 100.0 - spread_pct, -1, j - idx)
        if high[j] >= tp:
            return ((tp - entry) / entry * 100.0 - spread_pct, 1, j - idx)
    exit_px = close[end]
    return ((exit_px - entry) / entry * 100.0 - spread_pct, 0, end - idx)


frames = []
for sym in SYMBOLS:
    df = provider.get_historical_bars(sym, 15, FETCH_START, NOW)
    if df is None or len(df) == 0:
        print(f"  !! no bars for {sym}", file=sys.stderr)
        continue
    df = df.with_columns(pl.lit(sym).alias("symbol"))

    feats = strategy.pipeline.run(df, feature_cols=strategy.feature_names)
    if strategy.hmm_models is not None:
        from ml.regimes.hmm_regime import predict_regime_probs
        feats = predict_regime_probs(feats, strategy.hmm_models)

    X = feats[strategy.feature_names].to_pandas()
    angel_p = strategy.angel_trainer.predict_proba(X)[:, 1]
    Xd = X.copy()
    Xd["angel_prob"] = angel_p
    devil_p = strategy.devil_trainer.predict_proba(Xd)[:, 1]

    base = feats.select(["timestamp", "natr_14"]).with_columns(
        pl.Series("angel_prob", angel_p),
        pl.Series("devil_prob", devil_p),
    ).join(
        df.select(["timestamp", "open", "high", "low", "close"]),
        on="timestamp", how="inner",
    ).sort("timestamp")

    ts = base["timestamp"].to_numpy()
    close = base["close"].to_numpy().astype(float)
    high = base["high"].to_numpy().astype(float)
    low = base["low"].to_numpy().astype(float)
    natr = base["natr_14"].to_numpy().astype(float)
    ap = base["angel_prob"].to_numpy()
    sp = SPREAD_PCT[sym]

    rank = regime_rank(natr, profile.regime_window, profile.regime_min_samples)
    gate_a = (profile.sl_atr_multiplier * natr) >= (profile.spread_k_base * sp)
    gate_b = np.isnan(rank) | (rank >= profile.regime_pctile / 100.0)

    ny_times = [
        t.astimezone(NY).time()
        for t in base["timestamp"].to_list()
    ]
    gate_c = np.array(
        [not (BLACKOUT[0] <= t < BLACKOUT[1]) for t in ny_times]
    )

    pnl = np.full(len(base), np.nan)
    outcome = np.full(len(base), np.nan)
    held = np.full(len(base), np.nan)
    for i in np.where(ap >= SIM_FLOOR)[0]:
        if i >= len(close) - 1:
            continue
        pnl[i], outcome[i], held[i] = simulate(i, close, high, low, natr, sp)

    frames.append(base.with_columns(
        pl.lit(sym).alias("symbol"),
        pl.Series("regime_rank", rank),
        pl.Series("gate_a", gate_a),
        pl.Series("gate_b", gate_b),
        pl.Series("gate_c", gate_c),
        pl.Series("pnl_pct", pnl),
        pl.Series("outcome", outcome),
        pl.Series("bars_held", held),
    ))
    print(f"  {sym}: {len(base)} bars scored, "
          f"{int((ap >= SIM_FLOOR).sum())} simulated", file=sys.stderr)

study = pl.concat(frames)
study.write_parquet(OUT_DIR / "study_scored.parquet")

# ── sanity: agree with the sweep session's scores on the overlap? ──────────
prev = pl.read_parquet(OUT_DIR / "scored.parquet")
joined = study.join(prev, on=["symbol", "timestamp"], how="inner",
                    suffix="_prev")
d_angel = (joined["angel_prob"] - joined["angel_prob_prev"]).abs().max()
d_devil = (joined["devil_prob"] - joined["devil_prob_prev"]).abs().max()
print(f"\nSanity vs sweep session ({len(joined):,} overlapping bars): "
      f"max |d angel_prob| = {d_angel:.2e}, max |d devil_prob| = {d_devil:.2e}")

oos = study.filter(
    (pl.col("timestamp") >= OOS_BOUNDARY)
    & pl.col("symbol").is_in(list(TRADEABLE))
    & pl.col("pnl_pct").is_not_null()
    & pl.col("pnl_pct").is_not_nan()  # last bars have no forward data to sim
)
ins = study.filter(
    (pl.col("timestamp") < OOS_BOUNDARY)
    & pl.col("symbol").is_in(list(TRADEABLE))
    & pl.col("pnl_pct").is_not_null()
    & pl.col("pnl_pct").is_not_nan()
)
weeks = (NOW - OOS_BOUNDARY).total_seconds() / (7 * 86400)
print(f"\nOOS window: {OOS_BOUNDARY:%Y-%m-%d} -> now  ({weeks:.2f} calendar wk)"
      f" | tradeable-6 candidate bars: OOS {len(oos):,} / IS {len(ins):,}")
print(f"Devil threshold: {devil_threshold} | timeout rate among sims: "
      f"{float((study['outcome'] == 0).sum()) / max(int(study['pnl_pct'].is_not_null().sum()), 1):.1%}")


def ev_row(d: pl.DataFrame) -> str:
    n = len(d)
    if n == 0:
        return f"{0:>6}      -       -        -       -      -"
    p = d["pnl_pct"].to_numpy()
    wr = float((p > 0).mean())
    gp = p[p > 0].sum()
    gl = -p[p <= 0].sum()
    pf = gp / gl if gl > 0 else float("inf")
    return (f"{n:>6} {wr:>6.1%} {p.mean():>+8.4f} {p.sum():>+8.2f}"
            f" {pf:>7.2f} {n / weeks:>6.1f}")


HDR = f"{'thresh':>7} | {'n':>6} {'WR':>6} {'avg%':>8} {'tot%':>8} {'PF':>7} {'n/wk':>6}"

for label, dset in (("OUT-OF-SAMPLE (tradeable 6)", oos),):
    print(f"\n══ {label} ══")
    for variant, flt in (
        ("Angel only (no Devil, no gates)", lambda d, t: d.filter(pl.col("angel_prob") >= t)),
        ("Angel + Devil", lambda d, t: d.filter(
            (pl.col("angel_prob") >= t) & (pl.col("devil_prob") >= devil_threshold))),
        ("Angel + Devil + Gates A/B/C (live-faithful)", lambda d, t: d.filter(
            (pl.col("angel_prob") >= t) & (pl.col("devil_prob") >= devil_threshold)
            & pl.col("gate_a") & pl.col("gate_b") & pl.col("gate_c"))),
    ):
        print(f"\n-- {variant}")
        print(HDR)
        for t in THRESHOLDS:
            print(f"{t:>7.3f} | {ev_row(flt(dset, t))}")

# ── the marginal band: what a threshold drop actually adds ─────────────────
print("\n══ MARGINAL BAND, OOS, gates applied (the trades a lower threshold ADDS) ══")
print(f"{'band':>16} | {HDR.split('|')[1]}")
gated = oos.filter(pl.col("gate_a") & pl.col("gate_b") & pl.col("gate_c")
                   & (pl.col("devil_prob") >= devil_threshold))
for lo, hi, name in ((0.40, 9.9, ">=0.400 (today)"), (0.35, 0.40, "0.350-0.400"),
                     (0.325, 0.35, "0.325-0.350"), (0.30, 0.325, "0.300-0.325"),
                     (0.25, 0.30, "0.250-0.300")):
    band = gated.filter((pl.col("angel_prob") >= lo) & (pl.col("angel_prob") < hi))
    print(f"{name:>16} | {ev_row(band)}")

# same bands WITHOUT the Devil, since its opinion below 0.40 is extrapolation
print("\n(same bands, Devil ignored)")
gated_nd = oos.filter(pl.col("gate_a") & pl.col("gate_b") & pl.col("gate_c"))
for lo, hi, name in ((0.40, 9.9, ">=0.400 (today)"), (0.35, 0.40, "0.350-0.400"),
                     (0.325, 0.35, "0.325-0.350"), (0.30, 0.325, "0.300-0.325"),
                     (0.25, 0.30, "0.250-0.300")):
    band = gated_nd.filter((pl.col("angel_prob") >= lo) & (pl.col("angel_prob") < hi))
    print(f"{name:>16} | {ev_row(band)}")

# ── Devil verdict: does approval predict outcome? ──────────────────────────
print("\n══ DEVIL VERDICT (angel_prob >= 0.40 rows = its training population) ══")
for label, dset in (("IN-SAMPLE", ins), ("OUT-OF-SAMPLE", oos)):
    approved = dset.filter(pl.col("angel_prob") >= 0.40)
    if len(approved) == 0:
        print(f"{label}: no rows")
        continue
    pas = approved.filter(pl.col("devil_prob") >= devil_threshold)
    veto = approved.filter(pl.col("devil_prob") < devil_threshold)
    pp = pas["pnl_pct"].to_numpy()
    vp = veto["pnl_pct"].to_numpy()
    dp = approved["devil_prob"].to_numpy()
    po = approved["pnl_pct"].to_numpy()
    rank_d = np.argsort(np.argsort(dp)).astype(float)
    rank_p = np.argsort(np.argsort(po)).astype(float)
    corr = (np.corrcoef(rank_d, rank_p)[0, 1] if len(dp) > 2 else float("nan"))
    print(f"\n{label}: n={len(approved)} | pass n={len(pas)} "
          f"({len(pas)/len(approved):.0%}) avg {pp.mean() if len(pp) else float('nan'):+.4f}% "
          f"WR {float((pp > 0).mean()) if len(pp) else float('nan'):.1%}"
          f" | veto n={len(veto)} avg {vp.mean() if len(vp) else float('nan'):+.4f}% "
          f"WR {float((vp > 0).mean()) if len(vp) else float('nan'):.1%}"
          f" | spearman(devil_prob, pnl) = {corr:+.3f}")

# extrapolation region, for reference only
ext = oos.filter((pl.col("angel_prob") >= 0.325) & (pl.col("angel_prob") < 0.40))
if len(ext):
    pas = ext.filter(pl.col("devil_prob") >= devil_threshold)
    veto = ext.filter(pl.col("devil_prob") < devil_threshold)
    print(f"\n[extrapolation 0.325-0.40, OOS, reference only] "
          f"pass n={len(pas)} avg {pas['pnl_pct'].mean():+.4f}% | "
          f"veto n={len(veto)} avg "
          f"{(veto['pnl_pct'].mean() if len(veto) else float('nan')):+.4f}%")

# ── per-instrument at candidate thresholds ─────────────────────────────────
print("\n══ PER-INSTRUMENT, OOS, gates applied, Devil ignored ══")
for t in (0.325, 0.35, 0.40):
    print(f"\nthreshold {t}")
    print(f"{'symbol':>9} | {HDR.split('|')[1]}")
    for sym in sorted(TRADEABLE):
        d = gated_nd.filter((pl.col("symbol") == sym) & (pl.col("angel_prob") >= t))
        print(f"{sym:>9} | {ev_row(d)}")

# angel distribution drift check
for label, dset in (("IS", ins), ("OOS", oos)):
    a = dset["angel_prob"].to_numpy()
    if len(a):
        print(f"\nangel_prob {label} (candidates only, >= {SIM_FLOOR}): "
              f"n={len(a)} p50={np.percentile(a, 50):.3f} "
              f"p90={np.percentile(a, 90):.3f} max={a.max():.3f}")
print("\nstudy_scored.parquet written.")
