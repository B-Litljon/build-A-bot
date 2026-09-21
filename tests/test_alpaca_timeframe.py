"""
Bar-size → Alpaca timeframe mapping.

Regression test for a defect found 2026-09-14 while trying to measure daily crypto:
`AlpacaProvider.get_historical_bars` built `TimeFrame(n, TimeFrameUnit.Minute)` for
every call, and Alpaca rejects minute amounts above 59 — so **H4 (240) and D1 (1440)
requests could not be made at all**. They failed inside the method's broad `except`,
which logs and returns an empty frame, so the caller saw "no data" instead of an error.

The sizes matter: the crypto-expansion recon recommends daily bars precisely because
the round-trip fee stops dominating there. That recommendation was unservable.
"""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from alpaca.data.timeframe import TimeFrameUnit  # noqa: E402

from data.alpaca_provider import _timeframe_for  # noqa: E402


class TestTimeframeFor(unittest.TestCase):
    def test_minute_sizes_below_sixty_pass_through(self):
        for n in (1, 5, 15, 30, 59):
            tf = _timeframe_for(n)
            self.assertEqual((tf.amount, tf.unit_value), (n, TimeFrameUnit.Minute), n)

    def test_hour_sizes_use_the_hour_unit(self):
        for minutes, hours in ((60, 1), (120, 2), (240, 4)):
            tf = _timeframe_for(minutes)
            self.assertEqual((tf.amount, tf.unit_value), (hours, TimeFrameUnit.Hour), minutes)

    def test_daily_size_uses_the_day_unit(self):
        tf = _timeframe_for(1440)
        self.assertEqual((tf.amount, tf.unit_value), (1, TimeFrameUnit.Day))

    def test_multi_day_is_rejected_because_alpaca_allows_day_amount_1_only(self):
        """``alpaca-py`` raises "Day and Week units can only be used with amount 1", so
        a 2-day bar is not expressible; the helper must say so locally."""
        with self.assertRaises(ValueError):
            _timeframe_for(2880)

    def test_the_two_sizes_the_crypto_plan_needs_are_expressible(self):
        """H4 and D1 used to raise ValueError from alpaca-py's own validation."""
        self.assertEqual(_timeframe_for(240).unit_value, TimeFrameUnit.Hour)
        self.assertEqual(_timeframe_for(1440).unit_value, TimeFrameUnit.Day)

    def test_inexpressible_size_raises_here_not_remotely(self):
        with self.assertRaises(ValueError):
            _timeframe_for(90)          # > 1 hour, < 1 day, not a multiple of either
        with self.assertRaises(ValueError):
            _timeframe_for(1500)

    def test_non_positive_raises(self):
        for bad in (0, -15):
            with self.assertRaises(ValueError):
                _timeframe_for(bad)


if __name__ == "__main__":
    unittest.main()
