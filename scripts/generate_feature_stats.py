#!/usr/bin/env python
"""
Backfill a feature_stats.json sidecar for a model dir trained BEFORE the
retrainer started saving one (2026-07-07).

Rebuilds the model's training frame — same fetch, same feature pipeline, same
chop veto, same cleaning, via the retrainer's own functions — for the model's
recorded window, then computes and saves the distribution stats the drift
probe (scripts/probe_model.py) compares live data against.

The window END DATE matters: stats must describe what the model actually
trained on, so pass the model's training date (e.g. 2026-07-02 for the
current forex_m15 model), not today.

Usage (env vars mirror the original retrain invocation):
    set -a; source .env; set +a
    RETRAIN_END_DATE=2026-07-02 RETRAIN_DAYS_BACK=730 \
    RETRAIN_TIMEFRAME_MINUTES=15 RETRAIN_HTF_TIMEFRAME=1h DATA_SOURCE=oanda \
    PYTHONPATH=src:. python scripts/generate_feature_stats.py models/forex_m15

If the model was trained WITH a spread table, also set RETRAIN_SPREAD_TABLE
so the veto population (and cost_ratio feature) match.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("generate_feature_stats")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    model_dir = Path(sys.argv[1])
    if not (model_dir / "angel_latest.pkl").exists():
        raise SystemExit(f"{model_dir} does not look like a model dir (no angel_latest.pkl)")

    from src.core.retrainer import (
        DAYS_BACK,
        SPREAD_TABLE,
        engineer_features_and_labels,
        fetch_training_data,
        get_asset_config,
    )
    from src.data.factory import get_market_provider
    from src.execution.risk_manager import RiskProfile
    from src.ml.feature_stats import compute_feature_stats, save_feature_stats

    import os

    data_source = os.getenv("DATA_SOURCE", "alpaca").strip().lower()
    cfg = get_asset_config(data_source if data_source != "alpaca" else "forex")
    provider = get_market_provider()

    # Sanity: warn when the sidecar target's metadata disagrees with the env.
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("timeframe_minutes") not in (None, cfg["timeframe_minutes"]):
            logger.warning(
                "⚠️  metadata says timeframe=%s but env resolves %s — stats "
                "will NOT describe this model's training distribution",
                meta.get("timeframe_minutes"), cfg["timeframe_minutes"],
            )
    if not os.getenv("RETRAIN_END_DATE", "").strip():
        logger.warning(
            "⚠️  RETRAIN_END_DATE not set — window ends TODAY, which only "
            "matches a model trained today."
        )

    raw = fetch_training_data(
        provider=provider,
        symbols=cfg["tickers"],
        days_back=DAYS_BACK,
        timeframe_minutes=cfg["timeframe_minutes"],
    )
    df, feature_cols, _ = engineer_features_and_labels(
        raw,
        sl_mult=cfg["sl_mult"],
        tp_mult=cfg["tp_mult"],
        max_hold=cfg["max_hold"],
        survival_bars=cfg["survival_bars"],
        htf_timeframe=cfg["htf_timeframe"],
        risk_profile=RiskProfile.for_asset_class(cfg["asset_class"]),
        alpha_table=SPREAD_TABLE["alphas"] if SPREAD_TABLE else None,
    )

    # Guard: the rebuilt schema must match the model's actual schema.
    import joblib

    angel = joblib.load(model_dir / "angel_latest.pkl")
    _feats = getattr(angel, "feature_names_in_", None)
    model_feats = list(_feats) if _feats is not None else []
    if model_feats and sorted(model_feats) != sorted(feature_cols):
        raise SystemExit(
            f"Rebuilt feature set {feature_cols} != model's schema "
            f"{model_feats}. Check RETRAIN_SPREAD_TABLE / RETRAIN_USE_HMM env."
        )

    stats = compute_feature_stats(df, feature_cols)
    stats["backfilled"] = True
    stats["window"] = {
        "days_back": DAYS_BACK,
        "end_date": os.getenv("RETRAIN_END_DATE", "") or "today",
        "timeframe_minutes": cfg["timeframe_minutes"],
    }
    save_feature_stats(stats, model_dir)
    print(f"Wrote {model_dir / 'feature_stats.json'} "
          f"({len(feature_cols)} features, n={stats['n_rows']:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
