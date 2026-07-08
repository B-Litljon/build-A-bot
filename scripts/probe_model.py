#!/usr/bin/env python
"""
Zero-retraining model probe: WHY is the frozen model quiet (or loud)?

Two diagnostics against a frozen model dir, no weights touched:

  1. PSI (Population Stability Index) — per instrument, per feature: has the
     live distribution drifted away from the training distribution recorded
     in the model dir's feature_stats.json sidecar?
         < 0.10 stable | 0.10–0.25 moderate | > 0.25 SEVERE
  2. TreeSHAP contributions — exact per-feature contributions (log-odds) to
     the Angel's recent probabilities, straight from LightGBM
     (pred_contrib=True). Ranks what is actively suppressing angel_prob.

The verdict distinguishes the two explanations for a quiet model:
  * DRIFT: top suppressors are also drifted features → the model is looking
    at inputs unlike anything it trained on; its low confidence is
    mathematical unfamiliarity, not market judgment.
  * HONEST: inputs are in-distribution; the suppressors reflect real market
    conditions → the model sees the regime fine and judges there is no edge.

Usage:
    set -a; source .env; set +a
    PYTHONPATH=src:. python scripts/probe_model.py models/forex_m15 \
        [--bars 100] [--symbols XAU_USD,GBP_NZD] [--env practice]

Read-only: fetches history via the OANDA REST API, loads pkls, prints.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl

logging.basicConfig(level=logging.WARNING)  # keep the report readable
logger = logging.getLogger("probe_model")

# Mirrors run_oanda._GRANULARITY_PROFILES — timeframe → (HTF resample, warmup)
_GRANULARITY_PROFILES = {1: ("5m", 260), 5: ("30m", 300), 15: ("1h", 260)}


def _load_model_dir(model_dir: Path):
    angel = joblib.load(model_dir / "angel_latest.pkl")
    _feats = getattr(angel, "feature_names_in_", None)
    feats = list(_feats) if _feats is not None else []
    if not feats:
        raise SystemExit("Angel model exposes no feature_names_in_")
    meta = json.loads((model_dir / "metadata.json").read_text())
    spread_table = None
    if (model_dir / "spread_alphas.json").exists():
        spread_table = json.loads((model_dir / "spread_alphas.json").read_text())
    return angel, feats, meta, spread_table


def _build_pipeline(htf_timeframe: str, spread_table, regime_window: int):
    from ml.feature_pipeline import FeaturePipeline
    from ml.features.v3_features import (
        V3BaseFeatures,
        V3CostFeatures,
        V3HTFFeatures,
        V3SessionFeatures,
    )

    # Same construction as MLStrategy — symmetry with live inference.
    return FeaturePipeline(
        feature_generators=[
            V3BaseFeatures(),
            V3HTFFeatures(timeframe=htf_timeframe),
            V3SessionFeatures(),
            V3CostFeatures(
                alpha_table=spread_table["alphas"] if spread_table else None,
                default_alpha=(
                    spread_table.get("default_alpha", 0.15) if spread_table else 0.15
                ),
                regime_window=regime_window,
            ),
        ]
    )


def probe_symbol(
    symbol: str,
    provider,
    pipeline,
    angel,
    feature_names: list[str],
    stats: dict | None,
    timeframe_minutes: int,
    warmup: int,
    live_bars: int,
) -> dict | None:
    """Run both diagnostics for one instrument. Returns a result dict."""
    # Enough history for indicator warmup + the live comparison window,
    # padded for weekends/holidays (market-closed gaps).
    span_minutes = timeframe_minutes * (warmup + live_bars) * 2 + 3 * 24 * 60
    end = datetime.now(timezone.utc)
    df = provider.get_historical_bars(
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
        start=end - timedelta(minutes=span_minutes),
        end=end,
    )
    if df.height < warmup + live_bars:
        print(f"  [{symbol}] insufficient history ({df.height} bars) — skipped")
        return None
    df = df.with_columns(pl.lit(symbol).alias("symbol"))

    feats = pipeline.run(df, feature_cols=feature_names)
    live = feats.tail(live_bars)

    # ── SHAP: exact contributions to the Angel's recent log-odds ──
    X = live[feature_names].to_pandas()
    probs = angel.predict_proba(X)[:, 1]
    contrib = angel.booster_.predict(X, pred_contrib=True)  # (n, n_feat+1)
    mean_contrib = contrib[:, :-1].mean(axis=0)  # drop bias column
    ranked = sorted(zip(feature_names, mean_contrib), key=lambda t: t[1])

    # ── PSI vs training sidecar ──
    psi_by_feature: dict[str, float] = {}
    if stats is not None:
        from ml.feature_stats import psi_report

        psi_by_feature = psi_report(live, stats, symbol=symbol)

    return {
        "symbol": symbol,
        "n_live": live.height,
        "prob_median": float(np.median(probs)),
        "prob_max": float(np.max(probs)),
        "suppressors": ranked[:5],           # most negative mean contribution
        "boosters": ranked[-3:][::-1],       # most positive
        "psi": psi_by_feature,
    }


def print_report(res: dict, stats: dict | None) -> list[str]:
    """Pretty-print one symbol's result; return its drifted-feature list."""
    from ml.feature_stats import drift_flags

    sym = res["symbol"]
    print(f"\n─── {sym} ───  angel_prob median={res['prob_median']:.3f} "
          f"max={res['prob_max']:.3f}  (last {res['n_live']} bars)")

    drifted = []
    if stats is not None:
        # Calibrated against the null: what PSI do ordinary same-length
        # TRAINING windows produce? Only beating that null is drift —
        # raw textbook thresholds false-alarm on autocorrelated bars.
        labels = drift_flags(res["psi"], stats, symbol=sym)
        flagged = {f: l for f, l in labels.items() if l != "stable"}
        drifted = [f for f, l in labels.items() if l == "SEVERE"]
        ref = stats.get("per_symbol", {}).get(sym, stats["pooled"])
        if flagged:
            print("  PSI drift (vs null of same-length training windows):")
            for f in sorted(flagged, key=lambda f: -res["psi"][f]):
                null = ref.get(f, {}).get("null_psi", {})
                p99 = null.get("p99")
                ctx = f" (null p99={p99:.2f})" if p99 is not None else ""
                print(f"    {f:<22} {res['psi'][f]:6.3f}  [{flagged[f]}]{ctx}")
        else:
            print("  PSI drift: none — window is normal relative to "
                  "training windows of the same length")

    print("  Suppressing angel_prob (mean SHAP, log-odds):")
    for f, c in res["suppressors"]:
        mark = "  ← DRIFTED" if f in drifted else ""
        print(f"    {f:<22} {c:+.4f}{mark}")
    print("  Supporting:")
    for f, c in res["boosters"]:
        print(f"    {f:<22} {c:+.4f}")

    # Per-symbol verdict
    suppressor_names = [f for f, _ in res["suppressors"]]
    drift_hits = [f for f in suppressor_names if f in drifted]
    if drift_hits:
        print(f"  VERDICT: DRIFT — top suppressor(s) {drift_hits} are out of "
              "the training distribution; low confidence is unfamiliarity.")
    else:
        print("  VERDICT: HONEST — inputs in-distribution; the model sees "
              "this regime fine and finds no edge in it.")
    return drifted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--bars", type=int, default=100,
                    help="live window size (bars) for PSI + SHAP (default 100)")
    ap.add_argument("--symbols", type=str, default="",
                    help="comma list; default = trained basket from metadata")
    ap.add_argument("--env", type=str, default="practice")
    args = ap.parse_args()

    angel, feature_names, meta, spread_table = _load_model_dir(args.model_dir)
    if not hasattr(angel, "booster_"):
        raise SystemExit("Angel is not a LightGBM model — SHAP path needs booster_")

    tf = int(meta.get("timeframe_minutes") or 1)
    htf, warmup = _GRANULARITY_PROFILES[tf]
    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        or list(meta.get("trained_on_symbols") or [])
    )

    from ml.feature_stats import load_feature_stats

    stats = load_feature_stats(args.model_dir)

    from data.oanda_provider import OandaMarketProvider

    provider = OandaMarketProvider(environment=args.env)
    pipeline = _build_pipeline(htf, spread_table, regime_window=260)

    print("=" * 72)
    print(f"MODEL PROBE  {args.model_dir}  ({len(feature_names)} features, "
          f"{tf}m bars, live window {args.bars} bars)")
    if stats is None:
        print("NOTE: no feature_stats.json in model dir — PSI skipped. "
              "Backfill with scripts/generate_feature_stats.py")
    else:
        w = stats.get("window", {})
        print(f"Training stats: n={stats['n_rows']:,} rows"
              + (f", window ending {w.get('end_date')}" if w else ""))
        nw = stats.get("null_window")
        if nw and nw != args.bars:
            print(f"WARNING: --bars {args.bars} != sidecar null_window {nw} — "
                  "PSI calibration assumes same-length windows; prefer "
                  f"--bars {nw}")
    print("=" * 72)

    all_drift: dict[str, int] = {}
    quiet_honest = quiet_drift = 0
    for sym in symbols:
        res = probe_symbol(
            sym, provider, pipeline, angel, feature_names, stats,
            timeframe_minutes=tf, warmup=warmup, live_bars=args.bars,
        )
        if res is None:
            continue
        drifted = print_report(res, stats)
        for f in drifted:
            all_drift[f] = all_drift.get(f, 0) + 1
        if any(f in drifted for f, _ in res["suppressors"]):
            quiet_drift += 1
        else:
            quiet_honest += 1

    print("\n" + "=" * 72)
    print("SUMMARY")
    if all_drift:
        print("  Severely drifted features (× instruments):")
        for f, n in sorted(all_drift.items(), key=lambda t: -t[1]):
            print(f"    {f:<22} ×{n}")
    else:
        print("  No severe drift on any instrument.")
    print(f"  Verdicts: {quiet_drift} instrument(s) DRIFT, "
          f"{quiet_honest} HONEST")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
