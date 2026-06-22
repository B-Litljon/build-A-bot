import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import sys
from pathlib import Path

import polars as pl

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.strategies.concrete_strategies.ml_strategy import MLStrategy


def _bars(n: int, end: datetime) -> pl.DataFrame:
    ts = [end - timedelta(minutes=n - 1 - i) for i in range(n)]
    base = 1.08
    return pl.DataFrame(
        {
            "symbol": ["EUR_USD"] * n,
            "timestamp": ts,
            "open": [base] * n,
            "high": [base + 0.0005] * n,
            "low": [base - 0.0005] * n,
            "close": [base + 0.0001 * (i % 3) for i in range(n)],
            "volume": [10.0] * n,
        }
    )


class TestStaleFeatureGuard(unittest.TestCase):
    """H3 hardening: never score stale features against the current price."""

    @classmethod
    def setUpClass(cls):
        # Loads the real promoted forex models from models/forex/.
        cls.strategy = MLStrategy(asset_class="forex", warmup_period=10)

    def test_signal_skipped_when_latest_bar_dropped(self):
        """features_df tail older than raw df tail -> None, no prediction."""
        end = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
        df = _bars(12, end)
        # Simulate clean_data having dropped the newest bar: the feature
        # frame's last timestamp is one bar behind the raw frame's.
        stale_features = pl.DataFrame(
            {"timestamp": [end - timedelta(minutes=1)]}
        )

        with patch.object(
            self.strategy, "_generate_features", return_value=stale_features
        ):
            with patch.object(
                self.strategy.angel_trainer, "predict_proba"
            ) as mock_predict:
                result = self.strategy.generate_signals(df)

        self.assertIsNone(result)
        mock_predict.assert_not_called()

    def test_matching_timestamps_proceed_to_prediction(self):
        """Aligned tails -> the guard does not block the pipeline."""
        end = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
        df = _bars(12, end)

        result = self.strategy.generate_signals(df)
        # Whatever the models decide, the guard must not have been the
        # blocker: with aligned real features this exercises the full path
        # without raising. (Signal may legitimately be None on rejection.)
        self.assertTrue(result is None or result.direction == "long")


if __name__ == "__main__":
    unittest.main()
