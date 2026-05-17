import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.data.oanda_provider import OandaMarketProvider


class TestOandaTickHook(unittest.TestCase):
    """Unit tests for OandaMarketProvider raw-tick callback."""

    def _make_provider(self):
        """Return a provider with fake credentials (no real API calls)."""
        return OandaMarketProvider(
            environment="practice",
            api_key="fake_key",
            account_id="123",
            stream_granularity_minutes=1,
        )

    def _price_msg(
        self,
        instrument="EUR_USD",
        time="2024-01-01T00:00:00.000000000Z",
        bid="1.08000",
        ask="1.08005",
    ):
        return {
            "type": "PRICE",
            "instrument": instrument,
            "time": time,
            "bids": [{"price": bid}],
            "asks": [{"price": ask}],
        }

    # ── (a) tick_callback receives correct args ────────────────────────

    def test_tick_callback_receives_correct_args(self):
        """Fake PRICE msg -> on_tick gets (symbol, bid, ask)."""
        provider = self._make_provider()
        tick_cb = MagicMock()
        bar_cb = MagicMock()

        provider.subscribe(["EUR/USD"], bar_cb, tick_callback=tick_cb)
        provider._handle_tick(self._price_msg())

        tick_cb.assert_called_once_with("EUR_USD", 1.08000, 1.08005)

    # ── (b) tick_callback fires BEFORE bar flush on rollover ───────────

    def test_tick_callback_fires_before_bar_flush(self):
        """on_tick fires before _flush_bar when a bar rolls over."""
        provider = self._make_provider()
        tick_cb = MagicMock()
        bar_cb = MagicMock()

        provider.subscribe(["EUR/USD"], bar_cb, tick_callback=tick_cb)

        # Seed an old bar so the next tick triggers a rollover
        provider._tick_bars["EUR_USD"] = {
            "epoch": 0,
            "bar_start": "old",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
        }

        def assert_tick_already_called(*args, **kwargs):
            self.assertEqual(
                tick_cb.call_count,
                1,
                "tick_callback must fire before _flush_bar",
            )

        with patch.object(provider, "_flush_bar", side_effect=assert_tick_already_called):
            provider._handle_tick(self._price_msg())

        tick_cb.assert_called_once()

    # ── (c) bar path unchanged when tick_callback is None ──────────────

    def test_no_tick_callback_bar_path_unchanged(self):
        """Default tick_callback=None — bar aggregation unchanged."""
        provider = self._make_provider()
        bar_cb = MagicMock()

        provider.subscribe(["EUR/USD"], bar_cb)  # no tick_callback
        provider._handle_tick(self._price_msg())

        state = provider._tick_bars.get("EUR_USD")
        self.assertIsNotNone(state)
        self.assertEqual(state["open"], 1.080025)   # mid
        self.assertEqual(state["close"], 1.080025)
        self.assertEqual(state["volume"], 0)

        # Second tick in same epoch — update existing bar
        provider._handle_tick(self._price_msg(bid="1.08010", ask="1.08015"))
        state = provider._tick_bars["EUR_USD"]
        self.assertEqual(state["high"], 1.080125)
        self.assertEqual(state["low"], 1.080025)
        self.assertEqual(state["close"], 1.080125)
        self.assertEqual(state["volume"], 1)

    # ── (d) HEARTBEAT-like msg does not trigger tick_callback ──────────

    def test_missing_bids_asks_no_tick_callback(self):
        """Msg without bids/asks (HEARTBEAT shape) does not call on_tick."""
        provider = self._make_provider()
        tick_cb = MagicMock()
        bar_cb = MagicMock()

        provider.subscribe(["EUR/USD"], bar_cb, tick_callback=tick_cb)
        provider._handle_tick(
            {"type": "HEARTBEAT", "time": "2024-01-01T00:00:00.000000000Z"}
        )

        tick_cb.assert_not_called()
        self.assertEqual(provider._tick_bars, {})

    # ── (e) tick_callback exception logged, bar aggregation continues ──

    def test_tick_callback_exception_logged_continues(self):
        """Watchdog bug must never kill ingestion."""
        provider = self._make_provider()
        tick_cb = MagicMock(side_effect=RuntimeError("boom"))
        bar_cb = MagicMock()

        provider.subscribe(["EUR/USD"], bar_cb, tick_callback=tick_cb)

        with self.assertLogs("src.data.oanda_provider", level="ERROR") as cm:
            provider._handle_tick(self._price_msg())

        tick_cb.assert_called_once()
        self.assertTrue(
            any("tick_callback error for EUR_USD" in rec for rec in cm.output),
            f"Expected ERROR log not found in {cm.output}",
        )

        # Bar state must still be updated despite tick_callback exception
        state = provider._tick_bars.get("EUR_USD")
        self.assertIsNotNone(state)
        self.assertEqual(state["open"], 1.080025)


if __name__ == "__main__":
    unittest.main()
