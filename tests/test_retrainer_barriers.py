"""Unit and integration tests for learned barrier integration in the retrainer.

Covers:
  - Excursion labeling (mae_natr, mfe_natr, resolvable) inside engineer_features_and_labels
  - Atomic persistence and metadata contract in save_models
  - fit_and_save_barriers behavior (graceful skip on small data, full fit/save)
  - Strategy metadata validation bypass when use_barriers is active
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.retrainer import (
    engineer_features_and_labels,
    fit_and_save_barriers,
    save_models,
)
from ml.barriers.estimator import BarrierEstimator, BARRIER_META_FILENAME, BARRIER_MAE_FILENAME, BARRIER_MFE_FILENAME
from ml.barriers.labels import DEFAULT_HORIZON


def _make_synthetic_ohlc(n_bars: int = 200, symbols: list = None) -> pl.DataFrame:
    if symbols is None:
        symbols = ["EUR_USD"]
    dfs = []
    for sym in symbols:
        rng = np.random.RandomState(42)
        closes = 100.0 + np.cumsum(rng.randn(n_bars) * 0.1)
        highs = closes + rng.uniform(0.05, 0.3, size=n_bars)
        lows = closes - rng.uniform(0.05, 0.3, size=n_bars)
        opens = (highs + lows) / 2.0
        volumes = rng.uniform(100, 1000, size=n_bars)
        # 1-minute interval timestamps
        timestamps = pl.datetime_range(
            start=pl.datetime(2026, 1, 1, 0, 0),
            end=pl.datetime(2026, 1, 1, 0, 0) + pl.duration(minutes=n_bars - 1),
            interval="1m",
            eager=True,
        )
        dfs.append(
            pl.DataFrame({
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "symbol": [sym] * n_bars,
            })
        )
    return pl.concat(dfs)


class TestRetrainerBarrierLabeling(unittest.TestCase):
    def test_engineer_features_attaches_excursion_columns(self):
        raw_df = _make_synthetic_ohlc(n_bars=150, symbols=["EUR_USD", "GBP_USD"])
        horizon = 20
        features_df, feature_cols, _ = engineer_features_and_labels(
            raw_df,
            sl_mult=2.0,
            tp_mult=4.0,
            max_hold=horizon,
            survival_bars=5,
            htf_timeframe="5m",
        )
        for col in ("mae_natr", "mfe_natr", "resolvable"):
            self.assertIn(col, features_df.columns, f"Column {col} missing from engineered frame")

        # Check resolvable counts: each symbol has 150 rows.
        # Boundary tail for each symbol should have unresolvable rows
        eur_df = features_df.filter(pl.col("symbol") == "EUR_USD")
        gbp_df = features_df.filter(pl.col("symbol") == "GBP_USD")
        self.assertGreater(eur_df.height, 0)
        self.assertGreater(gbp_df.height, 0)

        # Resolvable rows have finite mae_natr and mfe_natr
        res_rows = features_df.filter(pl.col("resolvable"))
        self.assertGreater(res_rows.height, 50)
        y_mae = res_rows["mae_natr"].to_numpy()
        y_mfe = res_rows["mfe_natr"].to_numpy()
        self.assertTrue(np.all(np.isfinite(y_mae)))
        self.assertTrue(np.all(np.isfinite(y_mfe)))


class TestRetrainerBarrierSerialization(unittest.TestCase):
    def test_save_models_with_barrier_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test_model"
            cfg = {
                "asset_class": "forex",
                "model_dir": str(model_dir),
                "tickers": ["EUR_USD"],
                "timeframe_minutes": 15,
                "sl_mult": 2.0,
                "tp_mult": 4.0,
                "max_hold": 45,
            }

            # Create a fitted dummy BarrierEstimator
            raw = _make_synthetic_ohlc(n_bars=350)
            feat_df, cols, _ = engineer_features_and_labels(raw, max_hold=45)
            est = BarrierEstimator(feature_cols=["natr_14", "vol_rel"], tau_mae=0.95, tau_mfe=0.50)
            est.fit(feat_df, feat_df)

            save_models({"angel": 1}, {"devil": 2}, cfg, barrier_model=est)

            # Assert all files exist
            for name in ("angel_latest.pkl", "devil_latest.pkl", "metadata.json",
                         BARRIER_MAE_FILENAME, BARRIER_MFE_FILENAME, BARRIER_META_FILENAME):
                self.assertTrue((model_dir / name).exists(), f"{name} missing from {model_dir}")

            # Check metadata.json
            meta = json.loads((model_dir / "metadata.json").read_text())
            self.assertTrue(meta.get("learned_barriers"))
            self.assertIn("barriers", meta)
            self.assertEqual(meta["barriers"]["horizon"], 45)

    def test_save_models_without_barrier_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test_model_no_barrier"
            cfg = {
                "asset_class": "forex",
                "model_dir": str(model_dir),
                "tickers": ["EUR_USD"],
            }
            save_models({"angel": 1}, {"devil": 2}, cfg, barrier_model=None)
            meta = json.loads((model_dir / "metadata.json").read_text())
            self.assertFalse(meta.get("learned_barriers"))
            self.assertNotIn("barriers", meta)


class TestFitAndSaveBarriers(unittest.TestCase):
    def test_skips_when_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test_skip"
            cfg = {"asset_class": "forex", "model_dir": str(model_dir)}
            # Frame with only 10 rows
            tiny_df = pl.DataFrame({
                "mae_natr": [1.0] * 10,
                "mfe_natr": [2.0] * 10,
                "resolvable": [True] * 10,
                "natr_14": [1.5] * 10,
                "vol_rel": [1.0] * 10,
            })
            meta = fit_and_save_barriers(tiny_df, ["natr_14", "vol_rel"], cfg)
            self.assertIsNone(meta)
            self.assertFalse((model_dir / BARRIER_META_FILENAME).exists())

    def test_fits_and_persists_when_sufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test_fit_save"
            model_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "asset_class": "forex",
                "model_dir": str(model_dir),
                "max_hold": 30,
            }
            # Pre-seed metadata.json as retrainer.save_models would
            meta_path = model_dir / "metadata.json"
            meta_path.write_text(json.dumps({"asset_class": "forex", "learned_barriers": False}))

            raw = _make_synthetic_ohlc(n_bars=350)
            feat_df, cols, _ = engineer_features_and_labels(raw, max_hold=30)
            meta = fit_and_save_barriers(feat_df, ["natr_14", "vol_rel"], cfg)
            self.assertIsNotNone(meta)
            self.assertEqual(meta["horizon"], 30)

            # Check that files were created
            for name in (BARRIER_MAE_FILENAME, BARRIER_MFE_FILENAME, BARRIER_META_FILENAME):
                self.assertTrue((model_dir / name).exists())

            # Check metadata was updated
            updated_meta = json.loads(meta_path.read_text())
            self.assertTrue(updated_meta.get("learned_barriers"))
            self.assertIn("barriers", updated_meta)


class TestStrategyLearnedBarrierMetadataValidation(unittest.TestCase):
    def test_bypasses_bracket_mismatch_when_use_barriers_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models" / "forex"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Create mock Angel and Devil models
            import joblib
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(n_estimators=1, random_state=42)
            X = np.zeros((10, 2))
            X[5:, :] = 1.0
            y = np.array([0]*5 + [1]*5)
            clf.fit(X, y)
            clf.feature_names_in_ = np.array(["natr_14", "vol_rel"])

            joblib.dump(clf, model_dir / "angel_latest.pkl")
            joblib.dump(clf, model_dir / "devil_latest.pkl")
            (model_dir / "threshold.json").write_text(json.dumps({"angel_threshold": 0.4, "devil_threshold": 0.2}))

            # Metadata with mismatched brackets (e.g. 10.0x / 20.0x vs forex profile 2.0x / 4.0x)
            meta = {
                "asset_class": "forex",
                "sl_atr_multiplier": 10.0,
                "tp_atr_multiplier": 20.0,
                "learned_barriers": True,
            }
            (model_dir / "metadata.json").write_text(json.dumps(meta))

            # Also create barrier files so use_barriers=True can load
            raw = _make_synthetic_ohlc(n_bars=350)
            feat_df, _, _ = engineer_features_and_labels(raw, max_hold=DEFAULT_HORIZON)
            est = BarrierEstimator(feature_cols=["natr_14", "vol_rel"], tau_mae=0.95, tau_mfe=0.50)
            est.fit(feat_df, feat_df)
            est.save(model_dir, horizon=DEFAULT_HORIZON)

            from strategies.concrete_strategies.ml_strategy import MLStrategy
            strat = MLStrategy(
                asset_class="forex",
                angel_path=str(model_dir / "angel_latest.pkl"),
                devil_path=str(model_dir / "devil_latest.pkl"),
                use_barriers=True,
                warmup_period=10,
            )
            self.assertTrue(strat.use_barriers)
            self.assertIsNotNone(strat._barrier_estimator)

            # Conversely, with use_barriers=False, the mismatch MUST raise RuntimeError
            with self.assertRaises(RuntimeError) as ctx:
                MLStrategy(
                    asset_class="forex",
                    angel_path=str(model_dir / "angel_latest.pkl"),
                    devil_path=str(model_dir / "devil_latest.pkl"),
                    use_barriers=False,
                    warmup_period=10,
                )
            self.assertIn("Bracket mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
