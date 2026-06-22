"""Routing-safety tests for the RETRAIN_MODEL_DIR override.

The override lets us train a side model (e.g. metals-only) without clobbering
the promoted model in models/forex/. These tests pin that contract:
  - default output dir is models/<asset_class> (the production location),
  - RETRAIN_MODEL_DIR redirects it while leaving asset_class unchanged,
  - an actual save lands ONLY in the override dir.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add src to path (matches the other tests in this suite).
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.retrainer import get_asset_config, save_models, save_threshold


class TestRetrainerOutputDir(unittest.TestCase):
    def test_default_dir_is_models_asset_class(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RETRAIN_MODEL_DIR", None)
            cfg = get_asset_config("oanda")
        self.assertEqual(cfg["asset_class"], "forex")
        self.assertEqual(cfg["model_dir"], "models/forex")

    def test_override_redirects_without_changing_asset_class(self):
        with mock.patch.dict(os.environ, {"RETRAIN_MODEL_DIR": "models/forex_metals"}):
            cfg = get_asset_config("oanda")
        # asset_class stays "forex" so every feature/gate/hyperparam path is identical;
        self.assertEqual(cfg["asset_class"], "forex")
        # only the destination changes.
        self.assertEqual(cfg["model_dir"], "models/forex_metals")

    def test_save_lands_only_in_override_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "forex_metals"
            cfg = {
                "asset_class": "forex",
                "model_dir": str(override),
                "tickers": ["XAU_USD", "XAG_USD"],
                "timeframe_minutes": 1,
            }
            # joblib.dump serializes any picklable object, so dummies stand in
            # for the trained LGBM models — we only assert *where* files land.
            save_models({"angel": 1}, {"devil": 2}, cfg)
            save_threshold(0.27, cfg)

            for fname in ("angel_latest.pkl", "devil_latest.pkl",
                          "metadata.json", "threshold.json"):
                self.assertTrue((override / fname).exists(),
                                f"{fname} missing from override dir")

            meta = json.loads((override / "metadata.json").read_text())
            self.assertEqual(meta["asset_class"], "forex")
            self.assertEqual(meta["trained_on_symbols"], ["XAU_USD", "XAG_USD"])


if __name__ == "__main__":
    unittest.main()
