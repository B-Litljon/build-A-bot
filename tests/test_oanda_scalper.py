import asyncio
import unittest
import warnings
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, ANY
import sys
from pathlib import Path

"""
Tests for OandaScalperOrchestrator -- the live forex bot's control flow.

The largest test file here, and it concentrates on the failure paths rather than
the happy path, because with software-enforced stops the dangerous states are
"we think we're flat but aren't" and "we tried to close and it didn't work".

Glossary:
    FakeSignal -- a minimal stand-in for strategies.base.Signal, so tests can
        drive the orchestrator without loading real models.
    test_rapid_breach_ticks_close_once -- quotes arrive far faster than a close
        completes; a burst of breaching ticks must produce ONE close, not one
        per tick.
    test_close_not_called_synchronously_in_tick -- the tick callback runs on the
        provider's stream thread and must return in microseconds, so the close
        must be dispatched to the event loop rather than executed inline.
    test_watchdog_close_failure_retries_then_parks -- when closing keeps
        failing, the position is "parked" rather than forgotten. Forgetting it
        would mean an open position nothing is watching.
    test_no_entry_on_close_failed_position -- a symbol in that parked state must
        not be re-entered.
    test_boot_reconcile_* -- on startup the broker is asked what is actually
        open: an orphan left by a crashed process is flattened, a flat account
        is a no-op, a TRANSIENT sync failure is retried (a 2026-07-28 blip
        cost 15min of downtime via the watchdog's crash-loop brake), and a
        persistent one ABORTS rather than proceeding blind.
    test_reversal_records_authoritative_units -- after flipping direction the
        BROKER's reported size is recorded, not the size that was requested.
    test_failed_flip_restores_old_position_state -- a failed reversal must roll
        local state back, so it still reflects what is really held.
    test_tick_watchdog_ignores_reversing_position -- mid-flip the stop/target
        are meaningless and must not fire.
    test_seam_catchup_* -- after a (re)prime, the newest sealed bar must be
        scored exactly once IF fresh: stale bars are skipped (weekend gap),
        already-scored bars are skipped (repeat reconnects inside one bar must
        not double-trade), warmup still applies, an exception inside the
        evaluation must not escape (it would kill the reconnect loop), and
        max_age=0 disables the feature.
    test_stream_bar_marks_scored -- a bar scored by the normal stream path is
        thereby ineligible for catch-up; both paths share one dedup record.
    test_seam_backfill_* -- the bar in flight when the stream dies is dropped
        as partial but HAS sealed, so its complete version is re-fetched from
        REST and scored: the happy path, the REST-lag retry, giving up
        quietly when REST never publishes it, tolerating a fetch exception
        (this runs inside the bar callback), and refusing to append out of
        order or rescore.
    test_reconnect_delay_* -- backoff grows, stays inside [cap/2, cap], and
        never exceeds the configured maximum.
"""

# Suppress unawaited-coroutine RuntimeWarning when mocking asyncio.run_coroutine_threadsafe
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import polars as pl

from src.execution.oanda_scalper_orchestrator import OandaScalperOrchestrator
from src.execution.oanda_order_manager import OrderCloseError


class FakeSignal:
    """Minimal stand-in for strategies.base.Signal."""

    def __init__(self, direction="long", entry_price=1.08500,
                 raw_sl_distance=0.00050, raw_tp_distance=0.00150):
        self.direction = direction
        self.entry_price = entry_price
        self.raw_sl_distance = raw_sl_distance
        self.raw_tp_distance = raw_tp_distance
        self.metadata = {}


class TestOandaScalperOrchestrator(unittest.TestCase):
    """Mocked unit tests for the V5 scalper orchestrator."""

    def _make_orchestrator(self, **overrides):
        """Build an orchestrator with all dependencies mocked."""
        provider = MagicMock()
        # Real attribute, not an auto-mock: seam catch-up does arithmetic on
        # the provider's granularity (minutes per bar).
        provider._stream_gran = 15
        strategy = MagicMock()
        strategy.warmup_period = 3
        order_manager = MagicMock()
        risk_manager = MagicMock()

        orch = OandaScalperOrchestrator(
            symbols=["EUR/USD"],
            provider=provider,
            strategy=strategy,
            order_manager=order_manager,
            risk_manager=risk_manager,
            units_per_trade=1000,
            warmup_period=3,
            flatten_on_exit=False,
            # Always inject a mock notifier: earlier tests in the suite load
            # .env into os.environ (LiveOrchestrator calls load_dotenv), so a
            # real NotificationManager here would post to the LIVE webhook.
            notifier=MagicMock(),
            **overrides,
        )
        return orch, provider, strategy, order_manager, risk_manager

    # ── (a) bar signal → submit_target_position with correct signed target ──

    def test_bar_signal_long(self):
        """Long signal -> submit_target_position called with +1000."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="long")
        risk_manager.calculate_bracket.return_value = (0.00050, 0.00150)
        order_manager.submit_target_position.return_value = {
            "filled": 1000,
            "avg_price": 1.08500,
            "closed_units": 0,
            "opened_units": 1000,
            "position_units": 1000,
            "position_avg_price": 1.08500,
        }

        # Seed enough bars to pass warmup
        for i in range(3):
            asyncio.run(
                orch._on_bar(self._bar_dict(timestamp=i, close=1.08000 + i * 0.001))
            )

        order_manager.submit_target_position.assert_called_once_with("EUR_USD", 1000)
        self.assertEqual(orch._positions["EUR_USD"]["units"], 1000)
        self.assertEqual(orch._positions["EUR_USD"]["state"], "OPEN")

    def test_bar_signal_short(self):
        """Short signal -> submit_target_position called with -1000."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="short")
        risk_manager.calculate_bracket.return_value = (0.00050, 0.00150)
        order_manager.submit_target_position.return_value = {
            "filled": 1000,
            "avg_price": 1.08500,
            "closed_units": 0,
            "opened_units": 1000,
            "position_units": -1000,
            "position_avg_price": 1.08500,
        }

        for i in range(3):
            asyncio.run(
                orch._on_bar(self._bar_dict(timestamp=i, close=1.08000 + i * 0.001))
            )

        order_manager.submit_target_position.assert_called_once_with("EUR_USD", -1000)
        self.assertEqual(orch._positions["EUR_USD"]["units"], -1000)

    # ── (b) 5 rapid breach ticks -> close dispatched EXACTLY ONCE ──

    @patch("asyncio.run_coroutine_threadsafe")
    def test_rapid_breach_ticks_close_once(self, mock_run_coro):
        """5 rapid ticks on breached position -> PENDING_CLOSE guard fires once."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        orch._loop = mock_loop

        # Seed an open long position
        orch._positions["EUR_USD"] = {
            "entry": 1.08500,
            "sl": 1.08400,
            "tp": 1.08600,
            "units": 1000,
            "state": "OPEN",
        }

        # 5 rapid ticks, all breaching SL (bid <= 1.08400)
        for _ in range(5):
            orch._on_tick("EUR_USD", 1.08350, 1.08355)

        # run_coroutine_threadsafe must be called exactly once
        self.assertEqual(mock_run_coro.call_count, 1)
        # The coroutine scheduled is _watchdog_close
        scheduled_coro = mock_run_coro.call_args[0][0]
        self.assertEqual(scheduled_coro.cr_code.co_name, "_watchdog_close")
        # The loop passed is our mock
        self.assertEqual(mock_run_coro.call_args[0][1], mock_loop)

        # Position state must be PENDING_CLOSE
        self.assertEqual(orch._positions["EUR_USD"]["state"], "PENDING_CLOSE")
        # close_position must NOT have been called synchronously
        order_manager.close_position.assert_not_called()

    # ── (c) close_position NOT called synchronously inside on_tick ──

    def test_close_not_called_synchronously_in_tick(self):
        """on_tick never calls close_position directly; only dispatches async."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        orch._loop = mock_loop

        orch._positions["EUR_USD"] = {
            "entry": 1.08500,
            "sl": 1.08400,
            "tp": 1.08600,
            "units": 1000,
            "state": "OPEN",
        }

        with patch("asyncio.run_coroutine_threadsafe") as mock_run_coro:
            orch._on_tick("EUR_USD", 1.08350, 1.08355)

        order_manager.close_position.assert_not_called()
        mock_run_coro.assert_called_once()

    # ── (d) watchdog close failure handling (C1 hardening) ────────────

    def test_watchdog_close_success_pops_position(self):
        """Successful close -> position removed from tracking."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        order_manager.close_position.return_value = True
        orch._positions["EUR_USD"] = {
            "entry": 1.08500, "sl": 1.08400, "tp": 1.08600,
            "units": 1000, "state": "PENDING_CLOSE",
        }

        asyncio.run(orch._watchdog_close("EUR_USD"))

        self.assertNotIn("EUR_USD", orch._positions)
        order_manager.close_position.assert_called_once_with("EUR_USD")

    def test_watchdog_close_failure_retries_then_parks(self):
        """All close attempts fail -> position retained as CLOSE_FAILED."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        orch._close_max_attempts = 2  # keep backoff short for the test
        order_manager.close_position.side_effect = OrderCloseError("boom")
        orch._positions["EUR_USD"] = {
            "entry": 1.08500, "sl": 1.08400, "tp": 1.08600,
            "units": 1000, "state": "PENDING_CLOSE",
        }

        asyncio.run(orch._watchdog_close("EUR_USD"))

        self.assertEqual(order_manager.close_position.call_count, 2)
        self.assertIn("EUR_USD", orch._positions)
        self.assertEqual(orch._positions["EUR_USD"]["state"], "CLOSE_FAILED")

    def test_watchdog_close_recovers_on_retry(self):
        """First attempt fails, second succeeds -> position popped."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        orch._close_max_attempts = 3
        order_manager.close_position.side_effect = [
            OrderCloseError("transient"), True,
        ]
        orch._positions["EUR_USD"] = {
            "entry": 1.08500, "sl": 1.08400, "tp": 1.08600,
            "units": 1000, "state": "PENDING_CLOSE",
        }

        asyncio.run(orch._watchdog_close("EUR_USD"))

        self.assertEqual(order_manager.close_position.call_count, 2)
        self.assertNotIn("EUR_USD", orch._positions)

    def test_no_entry_on_close_failed_position(self):
        """A CLOSE_FAILED position blocks new signals on that symbol."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="long")
        risk_manager.calculate_bracket.return_value = (0.00050, 0.00150)
        orch._positions["EUR_USD"] = {
            "entry": 1.08500, "sl": 1.08400, "tp": 1.08600,
            "units": -1000, "state": "CLOSE_FAILED",
        }

        for i in range(3):
            asyncio.run(
                orch._on_bar(self._bar_dict(timestamp=i, close=1.08000 + i * 0.001))
            )

        order_manager.submit_target_position.assert_not_called()

    # ── (e) boot reconciliation (C2 hardening) ─────────────────────────

    def test_boot_reconcile_flattens_orphan(self):
        """Non-zero broker position at boot -> flattened."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        order_manager.sync_position.return_value = True
        order_manager.get_net_position.return_value = 500
        order_manager.get_average_entry_price.return_value = 1.09000

        asyncio.run(orch._reconcile_on_boot())

        order_manager.close_position.assert_called_once_with("EUR_USD")

    def test_boot_reconcile_noop_when_flat(self):
        """Flat at broker -> no close attempted."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        order_manager.sync_position.return_value = True
        order_manager.get_net_position.return_value = 0

        asyncio.run(orch._reconcile_on_boot())

        order_manager.close_position.assert_not_called()

    def test_boot_reconcile_aborts_on_sync_failure(self):
        """Unverifiable broker state -> startup refuses to proceed."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        orch._reconcile_retry_delay = 0.0
        order_manager.sync_position.return_value = False

        with self.assertRaises(RuntimeError):
            asyncio.run(orch._reconcile_on_boot())

        self.assertEqual(
            order_manager.sync_position.call_count, orch._reconcile_max_attempts
        )
        order_manager.close_position.assert_not_called()

    def test_boot_reconcile_retries_transient_sync_failure(self):
        """A blip (e.g. the 2026-07-28 401) must not abort startup: one
        transient error costs 15min of downtime via the crash-loop brake."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        orch._reconcile_retry_delay = 0.0
        order_manager.sync_position.side_effect = [False, True]
        order_manager.get_net_position.return_value = 0

        asyncio.run(orch._reconcile_on_boot())  # must not raise

        self.assertEqual(order_manager.sync_position.call_count, 2)
        order_manager.close_position.assert_not_called()

    # ── (f) reversal accounting (H1 hardening) ─────────────────────────

    def test_reversal_records_authoritative_units(self):
        """Flip long->short: recorded units = resulting net, not raw fill."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="short")
        risk_manager.calculate_bracket.return_value = (0.00050, 0.00150)
        orch._positions["EUR_USD"] = {
            "entry": 1.08000, "sl": 1.07900, "tp": 1.08300,
            "units": 1000, "state": "OPEN",
        }
        # Reversal: order trades 2000 units total; resulting net is -1000.
        order_manager.submit_target_position.return_value = {
            "filled": 2000,
            "avg_price": 1.08450,
            "closed_units": 1000,
            "opened_units": 1000,
            "position_units": -1000,
            "position_avg_price": 1.08440,
        }

        for i in range(3):
            asyncio.run(
                orch._on_bar(self._bar_dict(timestamp=i, close=1.08000 + i * 0.001))
            )

        self.assertEqual(orch._positions["EUR_USD"]["units"], -1000)
        self.assertEqual(orch._positions["EUR_USD"]["entry"], 1.08440)

    # ── (g) entry/watchdog race (H2 hardening) ─────────────────────────

    def test_tick_watchdog_ignores_reversing_position(self):
        """Ticks must not dispatch a close while a flip is in flight."""
        orch, _, _, _, _ = self._make_orchestrator()
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        orch._loop = mock_loop
        orch._positions["EUR_USD"] = {
            "entry": 1.08500, "sl": 1.08400, "tp": 1.08600,
            "units": 1000, "state": "REVERSING",
        }

        with patch("asyncio.run_coroutine_threadsafe") as mock_run_coro:
            orch._on_tick("EUR_USD", 1.08350, 1.08355)  # breaches old SL

        mock_run_coro.assert_not_called()
        self.assertEqual(orch._positions["EUR_USD"]["state"], "REVERSING")

    def test_failed_flip_restores_old_position_state(self):
        """Submit failure during a flip re-arms the watchdog (state OPEN)."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="short")
        risk_manager.calculate_bracket.return_value = (0.00050, 0.00150)
        order_manager.submit_target_position.side_effect = RuntimeError("api down")
        orch._positions["EUR_USD"] = {
            "entry": 1.08000, "sl": 1.07900, "tp": 1.08300,
            "units": 1000, "state": "OPEN",
        }

        for i in range(3):
            asyncio.run(
                orch._on_bar(self._bar_dict(timestamp=i, close=1.08000 + i * 0.001))
            )

        self.assertEqual(orch._positions["EUR_USD"]["state"], "OPEN")

    def test_flip_to_flat_pops_record(self):
        """Resulting net of zero after a fill clears the tracked position."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="short")
        risk_manager.calculate_bracket.return_value = (0.00050, 0.00150)
        orch._positions["EUR_USD"] = {
            "entry": 1.08000, "sl": 1.07900, "tp": 1.08300,
            "units": 1000, "state": "OPEN",
        }
        order_manager.submit_target_position.return_value = {
            "filled": 1000,
            "avg_price": 1.08450,
            "closed_units": 1000,
            "opened_units": 0,
            "position_units": 0,
            "position_avg_price": 0.0,
        }

        for i in range(3):
            asyncio.run(
                orch._on_bar(self._bar_dict(timestamp=i, close=1.08000 + i * 0.001))
            )

        self.assertNotIn("EUR_USD", orch._positions)

    # ── (h) seam catch-up: bars sealed during stream outages ───────────

    def test_seam_catchup_scores_fresh_missed_bar(self):
        """A fresh bar sealed during the outage is scored and traded once."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="long")
        risk_manager.calculate_bracket.return_value = (0.00050, 0.00150)
        order_manager.submit_target_position.return_value = {
            "filled": 1000,
            "avg_price": 1.08500,
            "closed_units": 0,
            "opened_units": 1000,
            "position_units": 1000,
            "position_avg_price": 1.08500,
        }
        bars = self._dt_bars(3)
        orch._bar_buffers["EUR_USD"] = list(bars)

        asyncio.run(orch._catch_up_missed_bars())

        order_manager.submit_target_position.assert_called_once_with("EUR_USD", 1000)
        # Catch-up must evaluate the buffer as-is, never re-append the bar
        self.assertEqual(len(orch._bar_buffers["EUR_USD"]), 3)
        self.assertEqual(
            orch._last_scored_ts["EUR_USD"], bars[-1]["timestamp"]
        )

    def test_seam_catchup_skips_stale_bar(self):
        """A bar sealed hours ago (weekend/long outage) must not be traded."""
        orch, _, strategy, order_manager, _ = self._make_orchestrator()
        orch._bar_buffers["EUR_USD"] = self._dt_bars(
            3, newest_sealed_secs_ago=7200
        )

        asyncio.run(orch._catch_up_missed_bars())

        strategy.generate_signals.assert_not_called()
        order_manager.submit_target_position.assert_not_called()
        self.assertIsNone(orch._last_scored_ts["EUR_USD"])

    def test_seam_catchup_skips_already_scored(self):
        """Repeat reconnects inside one bar period score the bar ONCE."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="long")
        risk_manager.calculate_bracket.return_value = (0.00050, 0.00150)
        order_manager.submit_target_position.return_value = {
            "filled": 1000,
            "avg_price": 1.08500,
            "closed_units": 0,
            "opened_units": 1000,
            "position_units": 1000,
            "position_avg_price": 1.08500,
        }
        orch._bar_buffers["EUR_USD"] = self._dt_bars(3)

        asyncio.run(orch._catch_up_missed_bars())
        asyncio.run(orch._catch_up_missed_bars())  # second reconnect, same bar

        # Dedup happens BEFORE inference, not via the position guard
        self.assertEqual(strategy.generate_signals.call_count, 1)
        order_manager.submit_target_position.assert_called_once()

    def test_seam_catchup_respects_warmup(self):
        """A short (post-failed-prime) buffer must not be scored."""
        orch, _, strategy, order_manager, _ = self._make_orchestrator()
        orch._bar_buffers["EUR_USD"] = self._dt_bars(2)  # warmup is 3

        asyncio.run(orch._catch_up_missed_bars())

        strategy.generate_signals.assert_not_called()
        order_manager.submit_target_position.assert_not_called()

    def test_seam_catchup_disabled_by_zero_max_age(self):
        """SEAM_CATCHUP_MAX_AGE_SECONDS=0 turns the feature off."""
        orch, _, strategy, order_manager, _ = self._make_orchestrator()
        orch._seam_catchup_max_age = 0.0
        orch._bar_buffers["EUR_USD"] = self._dt_bars(3)

        asyncio.run(orch._catch_up_missed_bars())

        strategy.generate_signals.assert_not_called()
        order_manager.submit_target_position.assert_not_called()

    def test_seam_catchup_swallows_evaluation_errors(self):
        """An exception mid-evaluation must not escape (it would kill the
        reconnect loop and leave the bot permanently disconnected)."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="long")
        risk_manager.calculate_bracket.side_effect = RuntimeError("boom")
        orch._bar_buffers["EUR_USD"] = self._dt_bars(3)

        asyncio.run(orch._catch_up_missed_bars())  # must not raise

        order_manager.submit_target_position.assert_not_called()
        # Marked as scored anyway: a failed evaluation must not retry into
        # a possible duplicate order on the next reconnect
        self.assertIsNotNone(orch._last_scored_ts["EUR_USD"])

    def test_stream_bar_marks_scored(self):
        """A bar scored by the stream path is ineligible for catch-up."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = None  # no trade, just score
        bars = self._dt_bars(3)
        for bar in bars:
            asyncio.run(orch._on_bar(bar))
        self.assertEqual(strategy.generate_signals.call_count, 1)
        self.assertEqual(orch._last_scored_ts["EUR_USD"], bars[-1]["timestamp"])

        asyncio.run(orch._catch_up_missed_bars())

        # Catch-up found nothing new to score
        self.assertEqual(strategy.generate_signals.call_count, 1)

    # ── (i) seam backfill: the bar in flight when the stream died ──────

    def _armed_for_drop(self, orch, bars):
        """Prime the seam state so the next _on_bar hits the drop branch."""
        orch._bar_buffers["EUR_USD"] = list(bars[:-1])
        orch._last_hist_ts["EUR_USD"] = bars[-2]["timestamp"]
        orch._seam_crossed["EUR_USD"] = False

    @staticmethod
    def _rest_frame(bar):
        """A get_historical_bars-shaped frame containing exactly one bar."""
        return pl.DataFrame(
            {
                "timestamp": [bar["timestamp"]],
                "open": [bar["open"]],
                "high": [bar["high"]],
                "low": [bar["low"]],
                "close": [bar["close"]],
                "volume": [bar["volume"]],
            }
        )

    def test_seam_backfill_scores_dropped_bar(self):
        """The dropped partial bar is re-fetched from REST and traded."""
        orch, provider, strategy, order_manager, risk_manager = self._make_orchestrator()
        strategy.generate_signals.return_value = FakeSignal(direction="long")
        risk_manager.calculate_bracket.return_value = (0.00050, 0.00150)
        order_manager.submit_target_position.return_value = {
            "filled": 1000, "avg_price": 1.08500, "closed_units": 0,
            "opened_units": 1000, "position_units": 1000,
            "position_avg_price": 1.08500,
        }
        bars = self._dt_bars(4)
        self._armed_for_drop(orch, bars)
        # REST holds the COMPLETE version of the bar the stream only has
        # partially — different close, so we can prove which one was used.
        complete = {**bars[-1], "close": 1.09999}
        provider.get_historical_bars.return_value = self._rest_frame(complete)

        asyncio.run(orch._on_bar(bars[-1]))

        order_manager.submit_target_position.assert_called_once_with("EUR_USD", 1000)
        buf = orch._bar_buffers["EUR_USD"]
        self.assertEqual(len(buf), 4)
        self.assertEqual(buf[-1]["close"], 1.09999)  # REST copy, not the partial
        self.assertEqual(buf[-1]["symbol"], "EUR_USD")
        self.assertEqual(orch._last_scored_ts["EUR_USD"], bars[-1]["timestamp"])

    def test_seam_backfill_retries_when_rest_lags_the_seal(self):
        """REST publishes the sealed candle a beat late -> retry succeeds."""
        orch, provider, strategy, order_manager, risk_manager = self._make_orchestrator()
        orch._seam_backfill_retry_delay = 0.0
        strategy.generate_signals.return_value = None
        bars = self._dt_bars(4)
        self._armed_for_drop(orch, bars)
        provider.get_historical_bars.side_effect = [
            pl.DataFrame({"timestamp": [], "open": [], "high": [],
                          "low": [], "close": [], "volume": []}),
            self._rest_frame(bars[-1]),
        ]

        asyncio.run(orch._on_bar(bars[-1]))

        self.assertEqual(provider.get_historical_bars.call_count, 2)
        strategy.generate_signals.assert_called_once()

    def test_seam_backfill_gives_up_without_evaluating(self):
        """REST never publishes it -> no evaluation, no crash, not marked."""
        orch, provider, strategy, order_manager, _ = self._make_orchestrator()
        orch._seam_backfill_retry_delay = 0.0
        bars = self._dt_bars(4)
        self._armed_for_drop(orch, bars)
        provider.get_historical_bars.return_value = pl.DataFrame(
            {"timestamp": [], "open": [], "high": [],
             "low": [], "close": [], "volume": []}
        )

        asyncio.run(orch._on_bar(bars[-1]))

        strategy.generate_signals.assert_not_called()
        order_manager.submit_target_position.assert_not_called()
        self.assertIsNone(orch._last_scored_ts["EUR_USD"])
        self.assertEqual(len(orch._bar_buffers["EUR_USD"]), 3)

    def test_seam_backfill_survives_fetch_exception(self):
        """A REST failure must not escape the bar callback."""
        orch, provider, strategy, _, _ = self._make_orchestrator()
        orch._seam_backfill_retry_delay = 0.0
        bars = self._dt_bars(4)
        self._armed_for_drop(orch, bars)
        provider.get_historical_bars.side_effect = RuntimeError("api down")

        asyncio.run(orch._on_bar(bars[-1]))  # must not raise

        strategy.generate_signals.assert_not_called()

    def test_seam_backfill_skips_already_scored_bar(self):
        """If the bar was already scored, no fetch and no rescore."""
        orch, provider, strategy, _, _ = self._make_orchestrator()
        bars = self._dt_bars(4)
        self._armed_for_drop(orch, bars)
        orch._last_scored_ts["EUR_USD"] = bars[-1]["timestamp"]

        asyncio.run(orch._on_bar(bars[-1]))

        provider.get_historical_bars.assert_not_called()
        strategy.generate_signals.assert_not_called()

    def test_seam_backfill_disabled_by_zero_attempts(self):
        """SEAM_BACKFILL_ATTEMPTS=0 restores the plain drop."""
        orch, provider, strategy, _, _ = self._make_orchestrator()
        orch._seam_backfill_attempts = 0
        bars = self._dt_bars(4)
        self._armed_for_drop(orch, bars)

        asyncio.run(orch._on_bar(bars[-1]))

        provider.get_historical_bars.assert_not_called()
        strategy.generate_signals.assert_not_called()

    # ── (j) reconnect backoff ──────────────────────────────────────────

    def test_reconnect_delay_grows_and_stays_in_jitter_band(self):
        """Backoff doubles per consecutive failure, jittered within [cap/2, cap]."""
        orch, _, _, _, _ = self._make_orchestrator()
        for attempt, ceiling in ((0, 5.0), (1, 10.0), (2, 20.0), (3, 40.0)):
            for _ in range(20):
                d = orch._reconnect_delay(attempt)
                self.assertGreaterEqual(d, ceiling / 2)
                self.assertLessEqual(d, ceiling)

    def test_reconnect_delay_capped(self):
        """Deep backoff never exceeds the cap (which sits under the 60s
        liveness-watchdog flatten threshold)."""
        orch, _, _, _, _ = self._make_orchestrator()
        for _ in range(20):
            d = orch._reconnect_delay(20)
            self.assertLessEqual(d, orch._reconnect_max_delay)
            self.assertGreaterEqual(d, orch._reconnect_max_delay / 2)

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dt_bars(n, gran_min=15, newest_sealed_secs_ago=30, symbol="EUR_USD"):
        """n sealed bars, the newest sealed newest_sealed_secs_ago ago."""
        newest_open = (
            datetime.now(timezone.utc)
            - timedelta(seconds=newest_sealed_secs_ago)
            - timedelta(minutes=gran_min)
        )
        bars = []
        for i in range(n):
            ts = newest_open - timedelta(minutes=gran_min * (n - 1 - i))
            bars.append(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": 1.08000,
                    "high": 1.08100,
                    "low": 1.07900,
                    "close": 1.08050,
                    "volume": 1.0,
                }
            )
        return bars

    @staticmethod
    def _bar_dict(
        symbol="EUR_USD",
        timestamp=0,
        open_p=1.08000,
        high=1.08100,
        low=1.07900,
        close=1.08050,
        volume=1.0,
    ):
        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }


if __name__ == "__main__":
    unittest.main()
