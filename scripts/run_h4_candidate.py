"""
H4 CatBoost Candidate Retraining & Barrier Benchmark.

Executes a full 730-day walk-forward training run on H4 (240m) bars using CatBoost
for both classifiers (Angel/Devil) and learned quantile barrier estimators (MAE/MFE).
Saves artifacts into models/forex_h4_catboost.
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np
import polars as pl

import core.retrainer as R
from execution.risk_manager import RiskProfile
from ml.barriers.estimator import BarrierEstimator, BARRIER_META_FILENAME, static_baseline_loss
from ml.barriers.labels import compute_excursions, pinball_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("h4_candidate")

CACHE_DIR = Path("data/cache/ab_catboost")
OUTPUT_DIR = Path("models/forex_h4_catboost")


def load_h4_data() -> pl.DataFrame:
    symbols = [
        "AUD_JPY", "EUR_JPY", "GBP_AUD", "GBP_JPY", "GBP_NZD", "NZD_JPY",
        "XAG_USD", "XAU_USD"
    ]
    dfs = []
    for sym in symbols:
        matches = list(CACHE_DIR.glob(f"{sym}_M240_730d_*.parquet"))
        if not matches:
            raise FileNotFoundError(f"Missing M240 cache for {sym}")
        df = pl.read_parquet(matches[0])
        df.columns = [c.lower() for c in df.columns]
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(sym).alias("symbol"))
        dfs.append(df)
    combined = pl.concat(dfs, how="vertical_relaxed").sort(["symbol", "timestamp"])
    logger.info(f"Loaded {combined.height:,} H4 rows across {len(symbols)} symbols")
    return combined


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_data = load_h4_data()

    # Configure Retrainer for CatBoost on H4
    R.MODEL_FAMILY = "catboost"
    os.environ["BARRIER_FAMILY"] = "catboost"
    os.environ["DATA_SOURCE"] = "oanda"
    os.environ["RETRAIN_MODEL_DIR"] = str(OUTPUT_DIR)
    os.environ["RETRAIN_TIMEFRAME_MINUTES"] = "240"
    os.environ["RETRAIN_HTF_TIMEFRAME"] = "1d"
    os.environ["RETRAIN_DAYS_BACK"] = "730"

    cfg = R.get_asset_config("oanda")
    cfg["model_dir"] = str(OUTPUT_DIR)
    cfg["timeframe_minutes"] = 240
    cfg["htf_timeframe"] = "1d"

    # Split holdout
    remainder_raw, holdout_raw, holdout_range = R._split_holdout(raw_data, R.HOLDOUT_FRAC)
    logger.info(
        f"Carved holdout: remainder={remainder_raw.height:,} rows, "
        f"holdout={holdout_raw.height:,} rows ({R.HOLDOUT_FRAC:.0%})"
    )

    # Feature & Label Engineering
    logger.info("Engineering features and realized excursion labels on H4...")
    features_df, feature_cols, chop_veto_rate = R.engineer_features_and_labels(
        remainder_raw,
        sl_mult=cfg["sl_mult"],
        angel_mult=cfg["angel_mult"],
        tp_mult=cfg["tp_mult"],
        max_hold=cfg["max_hold"],
        survival_bars=cfg["survival_bars"],
        htf_timeframe=cfg["htf_timeframe"],
        risk_profile=RiskProfile.for_asset_class("forex"),
        alpha_table=None,
    )

    tail_cutoffs = R._tail_cutoff_by_symbol(remainder_raw, cfg["max_hold"])
    features_df, n_purged = R._purge_boundary_tail(features_df, tail_cutoffs)
    logger.info(f"Engineered remainder: {features_df.height:,} rows ({n_purged} boundary tail purged)")

    # Run Walk-Forward Validation Gate
    logger.info("Running CatBoost 3-Fold Walk-Forward Gate on H4...")
    ap, dp = R.get_hyperparameters("forex")
    oos_ledger = []

    (
        report,
        angel_model,
        devil_model,
        angel_feats,
        devil_feats,
        optimal_threshold,
        final_hmm_models,
    ) = R.validate_candidate(
        features_df,
        feature_cols,
        sl_mult=cfg["sl_mult"],
        tp_mult=cfg["tp_mult"],
        n_folds=3,
        angel_params=ap,
        devil_params=dp,
        chop_veto_rate=chop_veto_rate,
        oos_ledger=oos_ledger,
        oos_ledger_cols=tuple(feature_cols) + ("timestamp", "symbol", "close", "mae_natr", "mfe_natr"),
    )

    logger.info("=" * 70)
    logger.info(f"H4 CATBOOST GATE RESULT: passed={report.gate_passed}")
    logger.info(f"  Mean Brier: {report.mean_brier:.4f}")
    logger.info(f"  Mean EV:    {report.mean_ev:+.6f}R")
    logger.info(f"  Pooled PF 95% LB: {report.pooled_pf_lower_bound:.4f} (wins={report.pooled_oos_wins}/trades={report.pooled_oos_trades})")
    logger.info(f"  Fold 3 PF 95% LB: {report.fold3_pf_lower_bound:.4f}")
    logger.info(f"  Optimal Devil Threshold: {optimal_threshold:.4f}")
    for fold in report.fold_metrics:
        logger.info(
            f"  Fold {fold.fold_number}: Brier={fold.brier_score:.4f}, EV={fold.expected_value:+.4f}R, "
            f"WR={fold.win_rate:.1%} ({fold.macro_wins}/{fold.devil_approved_trades})"
        )
    logger.info("=" * 70)

    # Save models and barrier models into OUTPUT_DIR
    R.save_models(angel_model, devil_model, cfg, report=report)
    R.save_threshold(optimal_threshold, cfg, angel_threshold=report.production_angel_threshold or None)
    stats = R.compute_feature_stats(features_df, feature_cols)
    R.save_feature_stats(stats, str(OUTPUT_DIR))

    # Fit and save learned barriers
    barrier_meta = R.fit_and_save_barriers(features_df, feature_cols, cfg)
    logger.info(f"Barrier model fitted and saved: {barrier_meta}")

    # Evaluate barrier pinball loss on OOS ledger if available
    led = pl.concat(oos_ledger) if oos_ledger else pl.DataFrame()
    if led.height:
        logger.info(f"OOS Ledger captured {led.height} trades across folds")
        ok_trades = led.filter(pl.col("mae_natr").is_not_null())
        if ok_trades.height:
            y = ok_trades["mae_natr"].to_numpy().astype(float)
            est = BarrierEstimator.load(OUTPUT_DIR)
            preds = est.predict(ok_trades)
            q_mae = np.array([p.q_mae for p in preds])
            learned_pb = pinball_loss(y, q_mae, tau=0.95)
            static_pb = static_baseline_loss(y, sl_mult=2.0, tau=0.95)
            cov = float((y <= q_mae).mean())
            logger.info("=" * 70)
            logger.info("OOS TRADES BARRIER EVALUATION (tau=0.95):")
            logger.info(f"  Learned Pinball Loss: {learned_pb:.4f}")
            logger.info(f"  Static  Pinball Loss: {static_pb:.4f} (Learned beats static: {learned_pb < static_pb})")
            logger.info(f"  MAE Stop Coverage:    {cov:.1%} (Target >= 93%)")
            logger.info(f"  Median Stop Multiple: {np.median(q_mae):.2f}x ATR vs static 2.00x ATR")
            logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    main()
