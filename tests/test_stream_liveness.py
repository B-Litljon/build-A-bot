import asyncio
import time
import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.execution.oanda_scalper_orchestrator import OandaScalperOrchestrator
from src.data.oanda_provider import OandaMarketProvider


class TestStreamLiveness(unittest.TestCase):
    """C3 hardening: stale-stream detection and response."""

    def _make_orchestrator(self):
        provider = MagicMock()
        strategy = MagicMock()
        strategy.warmup_period = 3
        order_manager = MagicMock()

        orch = OandaScalperOrchestrator(
            symbols=["EUR/USD"],
            provider=provider,
            strategy=strategy,
            order_manager=order_manager,
            warmup_period=3,
            flatten_on_exit=False,
            notifier=MagicMock(),
        )
        return orch, provider, order_manager

    def test_stale_stream_flattens_and_reconnects(self):
        """Stale beyond threshold + open position -> flatten + disconnect."""
        orch, provider, order_manager = self._make_orchestrator()
        provider.seconds_since_last_message = 120.0
        orch._positions["EUR_USD"] = {
            "entry": 1.085, "sl": 1.084, "tp": 1.086,
            "units": 1000, "state": "OPEN",
        }

        asyncio.run(orch._check_stream_liveness())

        order_manager.close_position.assert_called_once_with("EUR_USD")
        provider.force_disconnect.assert_called_once()
        self.assertEqual(orch._positions, {})

    def test_stale_stream_no_positions_still_reconnects(self):
        """Stale with no exposure -> disconnect only, no close calls."""
        orch, provider, order_manager = self._make_orchestrator()
        provider.seconds_since_last_message = 120.0

        asyncio.run(orch._check_stream_liveness())

        order_manager.close_position.assert_not_called()
        provider.force_disconnect.assert_called_once()

    def test_fresh_stream_no_action(self):
        """Recent message -> no flatten, no disconnect."""
        orch, provider, order_manager = self._make_orchestrator()
        provider.seconds_since_last_message = 3.0

        asyncio.run(orch._check_stream_liveness())

        order_manager.close_position.assert_not_called()
        provider.force_disconnect.assert_not_called()

    def test_stream_not_running_no_action(self):
        """No stream yet (age None) -> watchdog stays quiet."""
        orch, provider, order_manager = self._make_orchestrator()
        provider.seconds_since_last_message = None

        asyncio.run(orch._check_stream_liveness())

        order_manager.close_position.assert_not_called()
        provider.force_disconnect.assert_not_called()


class TestProviderLivenessState(unittest.TestCase):
    """Provider-side message-age tracking."""

    def _make_provider(self):
        return OandaMarketProvider(
            environment="practice",
            api_key="fake-key",
            account_id="123",
        )

    def test_age_none_before_stream(self):
        provider = self._make_provider()
        self.assertIsNone(provider.seconds_since_last_message)

    def test_age_tracks_monotonic(self):
        provider = self._make_provider()
        provider._last_stream_msg = time.monotonic() - 42.0
        age = provider.seconds_since_last_message
        self.assertIsNotNone(age)
        self.assertGreaterEqual(age, 42.0)
        self.assertLess(age, 45.0)

    def test_force_disconnect_noop_without_stream(self):
        """No active stream request -> force_disconnect is a safe no-op."""
        provider = self._make_provider()
        provider.force_disconnect("test")  # must not raise

    def test_force_disconnect_terminates_active_request(self):
        provider = self._make_provider()
        req = MagicMock()
        provider._active_stream_req = req
        provider.force_disconnect("test reason")
        req.terminate.assert_called_once_with("test reason")

    def test_reset_stop_clears_event(self):
        provider = self._make_provider()
        provider._stop_event.set()
        provider.reset_stop()
        self.assertFalse(provider._stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
