"""
Pin the training/live HTF pairing maps against each other.

run_oanda._GRANULARITY_PROFILES and retrainer._HTF_FOR_TIMEFRAME are two
hand-copied dicts that must stay identical: the live bot derives every htf_*
feature from the pairing in run_oanda, while the retrainer engineers the
training features from the pairing in core.retrainer. A silent drift between
them is train/serve skew in the htf_* columns (htf_rsi_14, htf_vol_rel,
htf_bb_pct_b, htf_trend_agreement) — the worst kind, because the models still
load and the numbers still "work".

Glossary:
    _GRANULARITY_PROFILES -- bar-size minutes -> (HTF resample string, warmup
        bars), owned by the live launcher.
    _HTF_FOR_TIMEFRAME -- bar-size minutes -> HTF resample string, owned by
        the retrainer. The warmup counts are not mirrored here (they live on
        the live side), so the test pins only the resample strings.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.retrainer import _HTF_FOR_TIMEFRAME  # noqa: E402


def _load_run_oanda():
    spec = importlib.util.spec_from_file_location(
        "run_oanda", project_root / "run_oanda.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestHTFPairing(unittest.TestCase):
    def test_live_and_training_htf_maps_identical(self):
        run_oanda = _load_run_oanda()
        profiles = run_oanda._GRANULARITY_PROFILES
        # Every bar size the live launcher knows must pair with the same HTF
        # the retrainer engineered features on.
        for minutes in (1, 5, 15):
            with self.subTest(minutes=minutes):
                self.assertIn(minutes, profiles)
                self.assertIn(minutes, _HTF_FOR_TIMEFRAME)
                self.assertEqual(
                    profiles[minutes][0],
                    _HTF_FOR_TIMEFRAME[minutes],
                    f"HTF pairing diverged for {minutes}m bars — "
                    "live and training htf_* features no longer match",
                )

    def test_retrainer_htf_keys_are_live_known(self):
        """Every HTF the retrainer can produce must be launchable live."""
        run_oanda = _load_run_oanda()
        profiles = run_oanda._GRANULARITY_PROFILES
        for minutes, htf in _HTF_FOR_TIMEFRAME.items():
            with self.subTest(minutes=minutes):
                self.assertIn(minutes, profiles)
                self.assertEqual(profiles[minutes][0], htf)


if __name__ == "__main__":
    unittest.main()
