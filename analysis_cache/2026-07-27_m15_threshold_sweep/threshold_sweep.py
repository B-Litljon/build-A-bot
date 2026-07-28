"""
Angel-threshold sweep over the live soak window (2026-07-13 → now).

Re-scores the exact M15 basket the soak traded, using the same models,
the same FeaturePipeline, and the same granularity profile as run_oanda.py,
then counts how many bars survive each stage of the funnel at a range of
Angel thresholds.

Funnel stages reproduced:
  1. Angel proposes      angel_prob >= T
  2. Devil confirms      devil_prob >= devil_threshold (0.48, from threshold.json)

Gate A (spread/cost) and Gate B (regime) are NOT reproduced here -- Gate A
needs live tick spreads that were never persisted. Post-Devil counts are
therefore an upper bound on fills.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/home/tha_magick_man/mystuf/development/build-A-bot")
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

SYMBOLS = [
    "XAU_USD", "XAG_USD", "GBP_JPY", "AUD_JPY",
    "EUR_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD",
]
MODEL_DIR = ROOT / "models" / "forex_m15"
SOAK_START = datetime(2026, 7, 13, 18, 15, tzinfo=timezone.utc) - timedelta(hours=7)
NOW = datetime.now(timezone.utc)
THRESHOLDS = [0.20, 0.25, 0.30, 0.325, 0.35, 0.375, 0.40, 0.45]

devil_threshold = json.loads((MODEL_DIR / "threshold.json").read_text())["devil_threshold"]
risk_profile = RiskProfile.for_asset_class("forex")

strategy = MLStrategy(
    asset_class="forex",
    angel_path=MODEL_DIR / "angel_latest.pkl",
    devil_path=MODEL_DIR / "devil_latest.pkl",
    timeframe=15,
    htf_timeframe="1h",
    warmup_period=260,
    regime_window=risk_profile.regime_window,
)

provider = OandaMarketProvider(environment="practice")

# Warm-up head: pull extra history so the first in-window bar has full
# rolling state, then trim back to the soak window before scoring.
fetch_start = SOAK_START - timedelta(days=12)

rows = []
for sym in SYMBOLS:
    df = provider.get_historical_bars(sym, 15, fetch_start, NOW)
    if df is None or len(df) == 0:
        print(f"  !! no bars for {sym}", file=sys.stderr)
        continue
    df = df.with_columns(pl.lit(sym).alias("symbol"))

    feats = strategy.pipeline.run(df, feature_cols=strategy.feature_names)
    if strategy.hmm_models is not None:
        from ml.regimes.hmm_regime import predict_regime_probs
        feats = predict_regime_probs(feats, strategy.hmm_models)

    feats = feats.filter(pl.col("timestamp") >= SOAK_START)
    if len(feats) == 0:
        continue

    X = feats[strategy.feature_names].to_pandas()
    angel_p = strategy.angel_trainer.predict_proba(X)[:, 1]

    Xd = X.copy()
    Xd["angel_prob"] = angel_p
    devil_p = strategy.devil_trainer.predict_proba(Xd)[:, 1]

    rows.append(pl.DataFrame({
        "symbol": [sym] * len(feats),
        "timestamp": feats["timestamp"],
        "angel_prob": angel_p,
        "devil_prob": devil_p,
    }))
    print(f"  {sym}: {len(feats)} bars scored", file=sys.stderr)

scored = pl.concat(rows)
scored.write_parquet("/tmp/claude-1000/-home-tha-magick-man/907ec179-ae3b-41d1-b30e-295205512d93/scratchpad/scored.parquet")

print(f"\nWindow: {scored['timestamp'].min()} → {scored['timestamp'].max()}")
print(f"Total bars scored: {len(scored):,} across {scored['symbol'].n_unique()} symbols")
print(f"Devil threshold: {devil_threshold}")

ap = scored["angel_prob"].to_numpy()
print("\nangel_prob distribution (all symbols):")
for q in [50, 75, 90, 95, 99, 99.9]:
    print(f"  p{q:<5} {np.percentile(ap, q):.3f}")
print(f"  max    {ap.max():.3f}")

print(f"\n{'thresh':>7} {'proposed':>9} {'prop rate':>10} {'devil OK':>9} {'pass rate':>10} {'trades/wk':>10}")
weeks = (NOW - SOAK_START).total_seconds() / (7 * 86400)
for t in THRESHOLDS:
    m = scored.filter(pl.col("angel_prob") >= t)
    n_prop = len(m)
    n_devil = len(m.filter(pl.col("devil_prob") >= devil_threshold))
    pr = 100 * n_prop / len(scored)
    dr = (100 * n_devil / n_prop) if n_prop else 0.0
    print(f"{t:>7.3f} {n_prop:>9} {pr:>9.2f}% {n_devil:>9} {dr:>9.1f}% {n_devil / weeks:>10.1f}")

print("\nPer-symbol Devil-confirmed counts:")
hdr = f"{'symbol':<10}" + "".join(f"{t:>8.3f}" for t in THRESHOLDS)
print(hdr)
for sym in SYMBOLS:
    s = scored.filter(pl.col("symbol") == sym)
    if len(s) == 0:
        continue
    cells = []
    for t in THRESHOLDS:
        n = len(s.filter((pl.col("angel_prob") >= t) & (pl.col("devil_prob") >= devil_threshold)))
        cells.append(f"{n:>8}")
    print(f"{sym:<10}" + "".join(cells))
