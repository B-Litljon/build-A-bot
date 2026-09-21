import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

"""
Tests for the stream-liveness watchdog -- what happens when the price feed goes
silent.

The scenario these exist for: a TCP connection that half-opens without closing.
No error is raised, the reader simply blocks forever, and stops enforced in
software stop being enforced at all. Silence is therefore treated as failure.

Glossary:
    TestStreamLiveness -- the orchestrator's response.
    test_stale_stream_flattens_and_reconnects -- THE CORE SAFETY BEHAVIOUR:
        a quiet feed means reconnect AND close everything. Holding a position
        you cannot see is worse than being flat.
    test_stale_stream_no_positions_still_reconnects -- reconnect even with
        nothing at risk, so the bot recovers instead of sitting idle.
    test_fresh_stream_no_action / test_stream_not_running_no_action -- no
        false alarms on a healthy feed, and no action before startup.
    TestProviderLivenessState -- the provider's own age tracking.
    test_age_none_before_stream -- age is None (unknown), not 0 (fresh), before
        the stream starts. Zero would read as healthy.
    test_age_tracks_monotonic -- measured on a monotonic clock, so a system
        clock adjustment cannot make the feed look fresh or ancient.
    test_force_disconnect_noop_without_stream -- forcing a disconnect with no
        stream is harmless.
    test_force_disconnect_terminates_active_request -- and does terminate a real
        one.
    test_reset_stop_clears_event -- the stop flag can be cleared so the stream
        can restart after a shutdown signal.
    TestSeamBackfillStatusEmit -- bars recovered through the seam paths
        (_backfill_seam_bar / _catch_up_missed_bars) must refresh
        status.json, else a surviving reconnect reads as stale to the
        watchdog. Including when evaluation itself raises.
"""

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.execution.oanda_forex_orchestrator import OandaForexOrchestrator
from src.data.oanda_provider import OandaMarketProvider


class TestStreamLiveness(unittest.TestCase):
    """C3 hardening: stale-stream detection and response."""

    def setUp(self):
        """Pin the clock OUTSIDE any scheduled pause for the whole class.

        Added 2026-09-15 with the pause gate: these tests assert the incident
        path (flatten + reconnect), which is correct behaviour only while prices
        are expected. Without this pin the suite would pass or fail depending on
        whether it happened to run inside the 5pm-ET rollover or over a weekend.
        Pause suppression is tested explicitly in TestScheduledPauseNoAction.
        """
        self._pause = patch(
            "src.execution.oanda_forex_orchestrator.scheduled_market_pause",
            return_value=None,
        )
        self._pause.start()
        self.addCleanup(self._pause.stop)

    def _make_orchestrator(self):
        provider = MagicMock()
        strategy = MagicMock()
        strategy.warmup_period = 3
        order_manager = MagicMock()

        orch = OandaForexOrchestrator(
            symbols=["EUR/USD"],
            provider=provider,
            strategy=strategy,
            order_manager=order_manager,
            warmup_period=3,
            flatten_on_exit=False,
            notifier=MagicMock(),
        )
        return orch, provider, order_manager

    def _healthy(self, provider):
        """Provider liveness attributes pinned to a healthy stream."""
        provider.stream_down_seconds = None
        provider.seconds_since_last_price = 3.0
        provider.seconds_since_last_message = 3.0

    def _stale_prices(self, provider, age=120.0):
        """Stream up (thread alive) but no PRICE past the threshold."""
        provider.stream_down_seconds = None
        provider.seconds_since_last_price = age
        provider.seconds_since_last_message = age

    def _stream_down(self, provider, age=120.0):
        """Stream thread exited `age` seconds ago (age None everywhere else)."""
        provider.stream_down_seconds = age
        provider.seconds_since_last_price = None
        provider.seconds_since_last_message = None

    def test_stale_stream_flattens_and_reconnects(self):
        """Stale beyond threshold + open position -> flatten + disconnect."""
        orch, provider, order_manager = self._make_orchestrator()
        self._stale_prices(provider)
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
        self._stale_prices(provider)

        asyncio.run(orch._check_stream_liveness())

        order_manager.close_position.assert_not_called()
        provider.force_disconnect.assert_called_once()

    def test_stream_down_with_positions_flattens(self):
        """2026-09-09: a DOWN stream (age None everywhere) with positions
        must flatten. The old code returned on age None — the exact state
        during every real outage — leaving the stop unwatched."""
        orch, provider, order_manager = self._make_orchestrator()
        self._stream_down(provider)
        orch._positions["EUR_USD"] = {
            "entry": 1.085, "sl": 1.084, "tp": 1.086,
            "units": 1000, "state": "OPEN",
        }

        asyncio.run(orch._check_stream_liveness())

        order_manager.close_position.assert_called_once_with("EUR_USD")
        provider.force_disconnect.assert_called_once()
        self.assertEqual(orch._positions, {})

    def test_price_silence_while_heartbeats_flow_flattens(self):
        """2026-09-09: heartbeats used to count as liveness, so 'connection
        alive, prices missing' could never flatten. Prices are what the
        stops run on."""
        orch, provider, order_manager = self._make_orchestrator()
        provider.stream_down_seconds = None
        provider.seconds_since_last_price = 120.0   # prices stopped...
        provider.seconds_since_last_message = 1.0   # ...heartbeats continue
        orch._positions["EUR_USD"] = {
            "entry": 1.085, "sl": 1.084, "tp": 1.086,
            "units": 1000, "state": "OPEN",
        }

        asyncio.run(orch._check_stream_liveness())

        order_manager.close_position.assert_called_once_with("EUR_USD")
        provider.force_disconnect.assert_called_once()

    def test_fresh_stream_no_action(self):
        """Recent message -> no flatten, no disconnect."""
        orch, provider, order_manager = self._make_orchestrator()
        self._healthy(provider)

        asyncio.run(orch._check_stream_liveness())

        order_manager.close_position.assert_not_called()
        provider.force_disconnect.assert_not_called()

    def test_stream_not_running_no_action(self):
        """No stream yet (all ages None, down_seconds None) -> watchdog quiet."""
        orch, provider, order_manager = self._make_orchestrator()
        provider.stream_down_seconds = None
        provider.seconds_since_last_price = None
        provider.seconds_since_last_message = None

        asyncio.run(orch._check_stream_liveness())

        order_manager.close_position.assert_not_called()
        provider.force_disconnect.assert_not_called()


class TestSeamBackfillStatusEmit(unittest.TestCase):
    """2026-09-10 regression: bars recovered via the seam paths score while
    the stream is DOWN, so they are the very bars that must refresh
    status.json -- without an emit there, a surviving reconnect looks stale
    to soak_watchdog.sh and the soak gets restarted mid-recovery."""

    def _make_orchestrator(self):
        provider = MagicMock()
        provider._stream_gran = 15
        strategy = MagicMock()
        strategy.warmup_period = 3
        strategy.generate_signals.return_value = None
        order_manager = MagicMock()

        orch = OandaForexOrchestrator(
            symbols=["EUR/USD"],
            provider=provider,
            strategy=strategy,
            order_manager=order_manager,
            warmup_period=3,
            flatten_on_exit=False,
            notifier=MagicMock(),
        )
        return orch, provider

    def _prime_buffer(self, orch, n=5):
        """Fill EUR_USD's buffer with sealed historical bars ending in the
        past, mirroring what _prime_history leaves behind."""
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        for i in range(n):
            ts = now - _dt.timedelta(minutes=15 * (n - i))
            orch._bar_buffers["EUR_USD"] = orch._bar_buffers.get(
                "EUR_USD", []
            ) + [
                {
                    "symbol": "EUR_USD",
                    "timestamp": ts,
                    "open": 1.08, "high": 1.081, "low": 1.079,
                    "close": 1.0805, "volume": 100,
                    "complete": True,
                }
            ]

    def _rest_frame_with(self, ts):
        """A polars frame shaped like get_historical_bars' output,
        containing exactly the bar the backfill is looking for."""
        import polars as pl
        return pl.DataFrame(
            {
                "timestamp": [ts],
                "open": [1.0805], "high": [1.0815], "low": [1.0795],
                "close": [1.081], "volume": [120],
                "complete": [True],
            }
        )

    def test_backfill_seam_bar_emits_status(self):
        orch, provider = self._make_orchestrator()
        self._prime_buffer(orch)
        orch._seam_crossed["EUR_USD"] = False
        import datetime as _dt
        ts = _dt.datetime.now(_dt.timezone.utc)
        provider.get_historical_bars = MagicMock(
            return_value=self._rest_frame_with(ts)
        )

        with patch("src.execution.oanda_forex_orchestrator.events"
                   ) as mock_events:
            mock_events.write_status = MagicMock()
            asyncio.run(orch._backfill_seam_bar("EUR_USD", ts))

        mock_events.write_status.assert_called()

    def test_backfill_seam_bar_emits_status_even_when_eval_raises(self):
        """Telemetry must refresh even if evaluation blew up -- the watchdog
        cares that bars are being PROCESSED, not that they traded."""
        orch, provider = self._make_orchestrator()
        self._prime_buffer(orch)
        orch._seam_crossed["EUR_USD"] = False
        orch._evaluate_and_trade = MagicMock(
            side_effect=RuntimeError("boom")
        )
        import datetime as _dt
        ts = _dt.datetime.now(_dt.timezone.utc)
        provider.get_historical_bars = MagicMock(
            return_value=self._rest_frame_with(ts)
        )

        with patch("src.execution.oanda_forex_orchestrator.events"
                   ) as mock_events:
            mock_events.write_status = MagicMock()
            asyncio.run(orch._backfill_seam_bar("EUR_USD", ts))

        mock_events.write_status.assert_called()

    def test_catch_up_missed_bars_emits_status(self):
        orch, provider = self._make_orchestrator()
        self._prime_buffer(orch)
        with patch("src.execution.oanda_forex_orchestrator.events"
                   ) as mock_events:
            mock_events.write_status = MagicMock()
            asyncio.run(orch._catch_up_missed_bars())

        mock_events.write_status.assert_called()


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


class TestScheduledPauseNoAction(unittest.TestCase):
    """A quiet feed during a SCHEDULED pause takes no action at all.

    Written for the 2026-09-11..13 weekend: the watchdog read the closed forex
    market as a dead feed and produced 12,674 CRITICAL lines, 14 alert
    incidents, and 12 futile reconnects, with a price clock reaching 14.2 hours
    of "silence". With a position open the same path calls _flatten_all(), and
    positions ARE held across the daily rollover — so this was a spurious-exit
    bug wearing a safety feature's clothes.
    """

    def _make_orchestrator(self):
        provider = MagicMock()
        strategy = MagicMock()
        strategy.warmup_period = 3
        orch = OandaForexOrchestrator(
            symbols=["EUR/USD"],
            provider=provider,
            strategy=strategy,
            order_manager=MagicMock(),
            warmup_period=3,
            flatten_on_exit=False,
            notifier=MagicMock(),
        )
        return orch, provider

    def _quiet_with_position(self, provider, age=900.0):
        provider.stream_down_seconds = None
        provider.seconds_since_last_price = age
        provider.seconds_since_last_message = age

    def _run_during(self, pause_name):
        orch, provider = self._make_orchestrator()
        self._quiet_with_position(provider)
        orch._positions["EUR_USD"] = {
            "entry": 1.085, "sl": 1.084, "tp": 1.086,
            "units": 1000, "state": "OPEN",
        }
        with patch(
            "src.execution.oanda_forex_orchestrator.scheduled_market_pause",
            return_value=pause_name,
        ), patch.object(orch, "_flatten_all") as flatten:
            asyncio.run(orch._check_stream_liveness())
        return orch, provider, flatten

    def test_weekend_quiet_does_not_flatten_or_reconnect(self):
        orch, provider, flatten = self._run_during("weekend closure")
        flatten.assert_not_called()
        provider.force_disconnect.assert_not_called()
        self.assertIn("EUR_USD", orch._positions)

    def test_rollover_quiet_does_not_flatten_or_reconnect(self):
        orch, provider, flatten = self._run_during("daily rollover")
        flatten.assert_not_called()
        provider.force_disconnect.assert_not_called()
        self.assertIn("EUR_USD", orch._positions)

    def test_no_discord_alert_during_a_pause(self):
        orch, provider, _ = self._run_during("weekend closure")
        orch._notifier.send_system_message.assert_not_called()

    def test_pause_arms_the_watchdog_for_the_next_real_incident(self):
        """The guard must be re-armed so the first genuine outage still alerts."""
        orch, provider, _ = self._run_during("daily rollover")
        self.assertFalse(orch._liveness_alert_fired)

    def test_pause_logged_once_not_every_probe(self):
        """One INFO line per pause, not one per 10-second probe."""
        orch, provider = self._make_orchestrator()
        self._quiet_with_position(provider)
        with patch(
            "src.execution.oanda_forex_orchestrator.scheduled_market_pause",
            return_value="weekend closure",
        ), patch("src.execution.oanda_forex_orchestrator.logger") as log:
            for _ in range(5):
                asyncio.run(orch._check_stream_liveness())
        self.assertEqual(log.info.call_count, 1)
        log.critical.assert_not_called()

    def test_real_outage_after_the_pause_still_flattens(self):
        """No pause → the original safety behaviour is untouched."""
        orch, provider = self._make_orchestrator()
        self._quiet_with_position(provider)
        orch._positions["EUR_USD"] = {
            "entry": 1.085, "sl": 1.084, "tp": 1.086,
            "units": 1000, "state": "OPEN",
        }
        with patch(
            "src.execution.oanda_forex_orchestrator.scheduled_market_pause",
            return_value=None,
        ):
            asyncio.run(orch._check_stream_liveness())
        provider.force_disconnect.assert_called_once()
        self.assertEqual(orch._positions, {})
