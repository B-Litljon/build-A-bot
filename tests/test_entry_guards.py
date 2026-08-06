import asyncio
import os
import time
import unittest
import warnings
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

"""
Tests for the scalper's two entry guards -- post-exit cooldown and the
correlated-exposure cap.

Both exist because of one morning: on 2026-07-30 the bot opened long GBP_JPY,
AUD_JPY and NZD_JPY on the SAME bar (three short-JPY bets at 3x size, all
stopped when the yen moved), then re-entered NZD_JPY six minutes after its own
stop, into the same falling move, for a second loss. Nothing in the code
noticed either fact. These tests pin the guards that now do.

Glossary:
    _make_orchestrator -- multi-symbol orchestrator with mocked dependencies;
        the JPY crosses are the instruments the incident actually involved.
    _arm -- seeds bar buffers and canned strategy/risk/order responses so a
        call to _evaluate_and_trade reaches the submit path.
    test_cooldown_blocks_reentry_on_next_bar -- the NZD_JPY re-entry case.
    test_cooldown_expires -- the block is temporary, not permanent.
    test_cooldown_disabled_by_zero -- the guard has an off switch.
    test_watchdog_close_starts_cooldown -- a real exit arms it; without this
        the timer never starts and the guard is decorative.
    test_cooldown_does_not_block_flip -- reversing a position that is still
        open is a different act from re-entering a closed one.
    test_third_correlated_position_blocked -- the 3x short-JPY case.
    test_second_correlated_position_allowed -- the cap is 2, not 1.
    test_opposite_leg_does_not_count -- long JPY and short JPY are not the
        same exposure and must not be summed.
    test_exposure_cap_disabled -- off switch.
    test_same_bar_entries_cannot_both_pass -- the race the reservation
        exists for: two entries evaluated concurrently, neither yet filled.
    test_reservation_released_on_submit_failure -- a failed order must not
        leave a phantom reservation blocking the instrument forever.
    test_unparseable_symbol_is_uncapped -- a naming surprise degrades to
        "no cap", never to an exception on the entry path.
    test_env_overrides_read_at_construction -- the knobs are real.
"""

warnings.filterwarnings("ignore", category=RuntimeWarning)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.execution.oanda_scalper_orchestrator import OandaScalperOrchestrator


class FakeSignal:
    """Minimal stand-in for strategies.base.Signal."""

    def __init__(self, direction="long", entry_price=100.0,
                 raw_sl_distance=0.10, raw_tp_distance=0.30):
        self.direction = direction
        self.entry_price = entry_price
        self.raw_sl_distance = raw_sl_distance
        self.raw_tp_distance = raw_tp_distance
        self.metadata = {}


def _bar(symbol, i):
    return {
        "symbol": symbol,
        "timestamp": i,
        "open": 100.0,
        "high": 100.1,
        "low": 99.9,
        "close": 100.0 + i * 0.01,
        "volume": 1.0,
    }


class TestEntryGuards(unittest.TestCase):
    """Cooldown + correlated-exposure cap on the OANDA scalper path."""

    def _make_orchestrator(self, symbols=None, **overrides):
        provider = MagicMock()
        provider._stream_gran = 15  # minutes per bar; the cooldown default
        strategy = MagicMock()
        strategy.warmup_period = 3
        order_manager = MagicMock()
        risk_manager = MagicMock()

        orch = OandaScalperOrchestrator(
            symbols=symbols or ["GBP_JPY", "AUD_JPY", "NZD_JPY"],
            provider=provider,
            strategy=strategy,
            order_manager=order_manager,
            risk_manager=risk_manager,
            units_per_trade=1000,
            warmup_period=3,
            flatten_on_exit=False,
            notifier=MagicMock(),
            **overrides,
        )
        return orch, provider, strategy, order_manager, risk_manager

    @staticmethod
    def _arm(orch, strategy, order_manager, risk_manager,
             direction="long", position_units=1000):
        """Make any _evaluate_and_trade call reach (and pass) the submit."""
        for sym in orch._bar_buffers:
            orch._bar_buffers[sym] = [_bar(sym, i) for i in range(3)]
        strategy.generate_signals.return_value = FakeSignal(direction=direction)
        risk_manager.calculate_bracket.return_value = (0.10, 0.30)
        order_manager.submit_target_position.return_value = {
            "filled": abs(position_units),
            "avg_price": 100.0,
            "closed_units": 0,
            "opened_units": position_units,
            "position_units": position_units,
            "position_avg_price": 100.0,
        }

    @staticmethod
    def _open(orch, symbol, units=1000):
        orch._positions[symbol] = {
            "entry": 100.0, "sl": 99.0, "tp": 101.0,
            "units": units, "state": "OPEN",
        }

    # ── post-exit cooldown ──────────────────────────────────────────────

    def test_cooldown_blocks_reentry_on_next_bar(self):
        """The 2026-07-30 NZD_JPY case: stopped out, re-signalled 6 min later."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)

        # Exited 6 minutes ago; one 15m bar has not yet passed.
        orch._last_exit_ts["NZD_JPY"] = time.monotonic() - 360

        asyncio.run(orch._evaluate_and_trade("NZD_JPY"))

        order_manager.submit_target_position.assert_not_called()
        self.assertNotIn("NZD_JPY", orch._positions)
        self.assertEqual(orch._cooldown_rejections, 1)

    def test_cooldown_expires(self):
        """Past the window, the same signal trades normally."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)

        orch._last_exit_ts["NZD_JPY"] = time.monotonic() - 1200  # > 15 min

        asyncio.run(orch._evaluate_and_trade("NZD_JPY"))

        order_manager.submit_target_position.assert_called_once_with(
            "NZD_JPY", 1000
        )
        self.assertEqual(orch._cooldown_rejections, 0)

    def test_cooldown_disabled_by_zero(self):
        """OANDA_REENTRY_COOLDOWN_SECONDS=0 restores the old behaviour."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)
        orch._reentry_cooldown = 0

        orch._last_exit_ts["NZD_JPY"] = time.monotonic()  # just now

        asyncio.run(orch._evaluate_and_trade("NZD_JPY"))

        order_manager.submit_target_position.assert_called_once()

    def test_watchdog_close_starts_cooldown(self):
        """A completed close is what arms the timer."""
        orch, _, _, order_manager, _ = self._make_orchestrator()
        self._open(orch, "NZD_JPY")

        asyncio.run(orch._watchdog_close("NZD_JPY"))

        order_manager.close_position.assert_called_once_with("NZD_JPY")
        self.assertNotIn("NZD_JPY", orch._positions)
        self.assertGreater(orch._cooldown_remaining("NZD_JPY"), 0)

    def test_cooldown_does_not_block_flip(self):
        """A still-open position may still be reversed during a cooldown."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(
            orch, strategy, order_manager, risk_manager,
            direction="short", position_units=-1000,
        )
        self._open(orch, "NZD_JPY", units=1000)  # long, open
        orch._last_exit_ts["NZD_JPY"] = time.monotonic()  # cooling down

        asyncio.run(orch._evaluate_and_trade("NZD_JPY"))

        order_manager.submit_target_position.assert_called_once_with(
            "NZD_JPY", -1000
        )
        self.assertEqual(orch._cooldown_rejections, 0)

    # ── correlated-exposure cap ─────────────────────────────────────────

    def test_third_correlated_position_blocked(self):
        """Two longs already short JPY; a third breaches the cap of 2."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)
        self._open(orch, "GBP_JPY", units=1000)
        self._open(orch, "AUD_JPY", units=1000)

        asyncio.run(orch._evaluate_and_trade("NZD_JPY"))

        order_manager.submit_target_position.assert_not_called()
        self.assertNotIn("NZD_JPY", orch._positions)
        self.assertEqual(orch._exposure_rejections, 1)
        self.assertEqual(orch._pending_entries, {})

    def test_second_correlated_position_allowed(self):
        """The default cap is 2 — one correlated pair is still permitted."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)
        self._open(orch, "GBP_JPY", units=1000)

        asyncio.run(orch._evaluate_and_trade("AUD_JPY"))

        order_manager.submit_target_position.assert_called_once_with(
            "AUD_JPY", 1000
        )
        self.assertEqual(orch._exposure_rejections, 0)

    def test_opposite_leg_does_not_count(self):
        """Short JPY and long JPY are opposite exposures, not two of a kind."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)
        orch._max_per_currency = 1
        self._open(orch, "GBP_JPY", units=-1000)  # short GBP_JPY = LONG JPY

        # Long AUD_JPY is SHORT JPY: a different leg, so the cap of 1 on
        # "short JPY" is untouched.
        asyncio.run(orch._evaluate_and_trade("AUD_JPY"))

        order_manager.submit_target_position.assert_called_once_with(
            "AUD_JPY", 1000
        )

    def test_exposure_cap_disabled(self):
        """OANDA_MAX_PER_CURRENCY=0 restores the old behaviour."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)
        orch._max_per_currency = 0
        self._open(orch, "GBP_JPY", units=1000)
        self._open(orch, "AUD_JPY", units=1000)

        asyncio.run(orch._evaluate_and_trade("NZD_JPY"))

        order_manager.submit_target_position.assert_called_once()

    def test_same_bar_entries_cannot_both_pass(self):
        """
        Two entries evaluated concurrently, neither filled yet.

        This is the race the reservation exists for: without it both see a
        world with zero JPY exposure and both pass a cap of 1.
        """
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)
        orch._max_per_currency = 1

        def slow_submit(symbol, units):
            time.sleep(0.05)  # fill is in flight while the other evaluates
            return {
                "filled": 1000, "avg_price": 100.0, "closed_units": 0,
                "opened_units": 1000, "position_units": 1000,
                "position_avg_price": 100.0,
            }

        order_manager.submit_target_position.side_effect = slow_submit

        async def both():
            await asyncio.gather(
                orch._evaluate_and_trade("GBP_JPY"),
                orch._evaluate_and_trade("AUD_JPY"),
            )

        asyncio.run(both())

        self.assertEqual(order_manager.submit_target_position.call_count, 1)
        self.assertEqual(orch._exposure_rejections, 1)
        self.assertEqual(orch._pending_entries, {})

    def test_reservation_released_on_submit_failure(self):
        """A rejected order must not leave the instrument blocked forever."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)
        order_manager.submit_target_position.side_effect = RuntimeError("boom")

        asyncio.run(orch._evaluate_and_trade("GBP_JPY"))

        self.assertEqual(orch._pending_entries, {})
        self.assertNotIn("GBP_JPY", orch._positions)

    def test_zero_fill_releases_reservation(self):
        """Broker rejection (zero fill) likewise releases the reservation."""
        orch, _, strategy, order_manager, risk_manager = self._make_orchestrator()
        self._arm(orch, strategy, order_manager, risk_manager)
        order_manager.submit_target_position.return_value = {
            "filled": 0, "avg_price": 0.0, "closed_units": 0,
            "opened_units": 0, "position_units": 0, "position_avg_price": 0.0,
        }

        asyncio.run(orch._evaluate_and_trade("GBP_JPY"))

        self.assertEqual(orch._pending_entries, {})

    def test_unparseable_symbol_is_uncapped(self):
        """A symbol the cap cannot decompose degrades to uncapped, not a crash."""
        orch, _, _, _, _ = self._make_orchestrator()
        self.assertEqual(orch._currency_legs("WEIRD", 1000), {})
        with orch._positions_lock:
            self.assertIsNone(orch._exposure_conflict("WEIRD", 1000))

    def test_currency_legs_signs(self):
        """Long GBP_JPY is +GBP/-JPY; short is the mirror."""
        orch, _, _, _, _ = self._make_orchestrator()
        self.assertEqual(
            orch._currency_legs("GBP_JPY", 1000), {"GBP": 1, "JPY": -1}
        )
        self.assertEqual(
            orch._currency_legs("GBP_JPY", -1000), {"GBP": -1, "JPY": 1}
        )

    def test_env_overrides_read_at_construction(self):
        """Both knobs come from the environment."""
        with patch.dict(
            os.environ,
            {
                "OANDA_REENTRY_COOLDOWN_SECONDS": "42",
                "OANDA_MAX_PER_CURRENCY": "3",
            },
        ):
            orch, _, _, _, _ = self._make_orchestrator()
        self.assertEqual(orch._reentry_cooldown, 42.0)
        self.assertEqual(orch._max_per_currency, 3)


if __name__ == "__main__":
    unittest.main()
