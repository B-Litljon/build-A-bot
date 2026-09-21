from dotenv import load_dotenv
load_dotenv()
"""
End-to-End Verification of Retrainer Producer + MLStrategy Barrier Consumer.

Loads cached 730d M240 / M60 data from data/cache/ab_catboost/, runs the full
retrainer pipeline into an isolated test sidecar directory (models/test_barrier_sidecar),
verifies all artifacts are atomically persisted, then boots MLStrategy with
use_barriers=True and executes live signal generation.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl

# Project root setup
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import core.retrainer as R
from core.retrainer import _data as data_mod  # fetch_training_data lives here post-2026-09-16 split
from ml.barriers.estimator import (
    BARRIER_MAE_FILENAME,
    BARRIER_META_FILENAME,
    BARRIER_MFE_FILENAME,
    BarrierEstimator,
)
from strategies.base import BARRIER_GEOMETRY_KEY
from strategies.concrete_strategies.ml_strategy import MLStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("test_sidecar_retrain")

TEST_DIR = Path("models/test_barrier_sidecar")
CACHE_DIR = Path("data/cache/ab_catboost")


def load_cached_raw(granularity: int = 240, days_back: int = 730) -> pl.DataFrame:
    """Load cached historical data across available symbols."""
    symbols = [
        "AUD_JPY", "EUR_JPY", "GBP_AUD", "GBP_JPY", "GBP_NZD", "NZD_JPY",
        "XAG_USD", "XAU_USD"
    ]
    dfs = []
    for sym in symbols:
        matches = list(CACHE_DIR.glob(f"{sym}_M{granularity}_{days_back}d_*.parquet"))
        if not matches:
            logger.warning(f"No cache found for {sym} M{granularity} {days_back}d")
            continue
        df = pl.read_parquet(matches[0])
        df.columns = [c.lower() for c in df.columns]
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(sym).alias("symbol"))
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No cached data found in data/cache/ab_catboost")
    combined = pl.concat(dfs, how="vertical_relaxed").sort(["symbol", "timestamp"])
    logger.info(f"Loaded {combined.height:,} cached rows across {len(dfs)} symbols")
    return combined


def main():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_cached_raw(granularity=240, days_back=730)

    env_overrides = {
        "DATA_SOURCE": "oanda",
        "RETRAIN_MODEL_DIR": str(TEST_DIR),
        "RETRAIN_TIMEFRAME_MINUTES": "240",
        "RETRAIN_HTF_TIMEFRAME": "1d",
        "RETRAIN_DAYS_BACK": "730",
        "RETRAIN_LEARN_BARRIERS": "1",
        "MODEL_FAMILY": "catboost",
        "BARRIER_FAMILY": "catboost",
        "RETRAIN_HOLDOUT_FRAC": "0.18",
    }

    logger.info("=" * 70)
    logger.info(f"RUNNING RETRAINER MAIN INTO {TEST_DIR} (ISOLATED SIDECAR)")
    logger.info("=" * 70)

    with mock.patch.dict(os.environ, env_overrides), \
         mock.patch.object(data_mod, "fetch_training_data", return_value=raw_data):
        exit_code = R.main()

    logger.info(f"Retrainer completed with exit code: {exit_code}")
    # 0 = promoted, 2 = rejected by gate but valid execution
    assert exit_code in (0, 2), f"Unexpected retrainer exit code: {exit_code}"

    # Check files created
    created_files = sorted([f.name for f in TEST_DIR.iterdir()])
    logger.info(f"Files in {TEST_DIR}: {created_files}")

    if exit_code == 0:
        expected = [
            "angel_latest.pkl",
            "devil_latest.pkl",
            "feature_stats.json",
            "metadata.json",
            "threshold.json",
            BARRIER_MAE_FILENAME,
            BARRIER_MFE_FILENAME,
            BARRIER_META_FILENAME,
        ]
        for fname in expected:
            assert (TEST_DIR / fname).exists(), f"Expected artifact {fname} missing from {TEST_DIR}"

        meta = json.loads((TEST_DIR / "metadata.json").read_text())
        assert meta.get("learned_barriers") is True, "metadata.json learned_barriers must be True"
        logger.info(f"metadata.json learned_barriers confirmed: {meta['learned_barriers']}")
        logger.info(f"barriers meta recorded: {meta.get('barriers')}")

        # Now test MLStrategy boot and execution on real bars
        logger.info("=" * 70)
        logger.info("TESTING MLSTRATEGY BOOT WITH LEARNED BARRIERS ACTIVE")
        logger.info("=" * 70)

        strat = MLStrategy(
            asset_class="forex",
            angel_path=str(TEST_DIR / "angel_latest.pkl"),
            devil_path=str(TEST_DIR / "devil_latest.pkl"),
            use_barriers=True,
            warmup_period=20,
        )

        assert strat.use_barriers is True
        assert strat._barrier_estimator is not None
        logger.info(f"MLStrategy loaded BarrierEstimator backend: {strat._barrier_estimator.backend_}")

        # Slice 200 bars for inference test
        test_slice = raw_data.filter(pl.col("symbol") == "GBP_JPY").sort("timestamp").tail(200)
        signals = strat.generate_signals(test_slice)
        logger.info(f"Generated {len(signals)} signals on GBP_JPY test slice")

        barrier_signals = [s for s in signals if s.metadata and BARRIER_GEOMETRY_KEY in s.metadata]
        logger.info(f"Signals with barrier geometry attached: {len(barrier_signals)} / {len(signals)}")
        if barrier_signals:
            sample_geo = barrier_signals[0].metadata[BARRIER_GEOMETRY_KEY]
            logger.info(f"Sample barrier geometry payload: {json.dumps(sample_geo, indent=2)}")
            assert "sl_price_distance" in sample_geo
            assert "tp_price_distance" in sample_geo
            assert "q_mae" in sample_geo
            assert "q_mfe" in sample_geo
            assert "rr" in sample_geo
            assert sample_geo["sl_price_distance"] > 0
            assert sample_geo["tp_price_distance"] > 0
    else:
        logger.info("Fold gate rejected candidate (status 2). Testing manual fit_and_save_barriers on slice.")
        cfg = {"asset_class": "forex", "model_dir": str(TEST_DIR), "max_hold": 45}
        feats, cols, _ = R.engineer_features_and_labels(raw_data.tail(2000), max_hold=45)
        meta = R.fit_and_save_barriers(feats, cols, cfg)
        assert meta is not None
        assert (TEST_DIR / BARRIER_META_FILENAME).exists()
        logger.info(f"Manual fit_and_save_barriers verified successfully: {meta}")

    logger.info("=" * 70)
    logger.info("✅ SIDECAR RETRAINING RUN & STRATEGY BOOT VERIFIED CLEANLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
