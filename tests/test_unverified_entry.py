import asyncio
import unittest
import warnings
from unittest.mock import MagicMock
import sys
from pathlib import Path

"""
Tests for the UNKNOWN entry outcome -- parked, then reconciled.

An order can go out, fail ambiguously (timeout / 5xx / reset), and leave the
broker unreadable. The position may or may not exist. Before this path the
caller read that as a plain zero fill and dropped it, which is how a live
position ends up with nothing enforcing its stop -- stops here are software,
so an unrecorded position is an unwatched one.

The record is parked in ENTRY_UNRECONCILED instead: visible to the exposure
caps, ignored by the tick stop-monitor (it may not exist), and settled against
the broker by the reconciler on the liveness loop.

Glossary:
    _make_orchestrator -- orchestrator with mocked dependencies.
    _arm_unverified -- makes the submit return the unverified result shape.
    test_unverified_submit_parks_instead_of_dropping -- the core case.
    test_clean_miss_does_not_park -- a definitive reject must NOT park, or a
        position that does not exist would block the symbol.
    test_parked_entry_is_not_stop_monitored -- the tick monitor must ignore a
        position whose existence is unproven.
    test_reconcile_flat_drops_the_record -- broker says nothing was taken.
    test_reconcile_open_arms_the_stop -- broker says it is open; the bracket
        is rebuilt from the stored DISTANCES around the real fill price.
    test_reconcile_short_orientation -- a short's stop sits above entry.
    test_reconcile_sync_failure_keeps_it_parked -- an unreadable broker is
        never resolved by assumption.
"""

warnings.filterwarnings("ignore", category=RuntimeWarning)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.execution.oanda_forex_orchestrator import OandaForexOrchestrator


class FakeSignal:
    def __init__(self, direction="long", entry_price=100.0,
                 raw_sl_distance=0.10, raw_tp_distance=0.30):
        self.direction = direction
        self.entry_price = entry_price
        self.raw_sl_distance = raw_sl_distance
        self.raw_tp_distance = raw_tp_distance
        self.metadata = {}


def _bar(symbol, i):
    return {
        "symbol": symbol, "timestamp": i, "open": 100.0, "high": 100.1,
        "low": 99.9, "close": 100.0 + i * 0.01, "volume": 1.0,
    }


UNVERIFIED_RESULT = {
    "filled": 0,
    "avg_price": 0.0,
    "closed_units": 0,
    "opened_units": 0,
    "position_units": 0,
    "position_avg_price": 0.0,
    "unverified": True,
}

CLEAN_MISS_RESULT = dict(UNVERIFIED_RESULT, unverified=False)


class TestUnverifiedEntry(unittest.TestCase):

    def _make_orchestrator(self, symbols=None, **overrides):
        provider = MagicMock()
        provider._stream_gran = 15
        strategy = MagicMock()
        strategy.warmup_period = 3
        order_manager = MagicMock()
        risk_manager = MagicMock()
        orch = OandaForexOrchestrator(
            symbols=symbols or ["GBP_JPY"],
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
    def _arm(orch, strategy, order_manager, risk_manager, result,
             direction="long"):
        for sym in orch._bar_buffers:
            orch._bar_buffers[sym] = [_bar(sym, i) for i in range(3)]
        strategy.generate_signals.return_value = FakeSignal(direction=direction)
        risk_manager.calculate_bracket.return_value = (0.10, 0.30)
        order_manager.submit_target_position.return_value = result

    @staticmethod
    def _park(orch, symbol="GBP_JPY", units=1000, direction="long"):
        orch._positions[symbol] = {
            "entry": 100.0, "sl": None, "tp": None, "units": units,
            "state": "ENTRY_UNRECONCILED", "sl_dist": 0.10, "tp_dist": 0.30,
            "dir": direction,
        }

    # ── parking ─────────────────────────────────────────────────────────

    def test_unverified_submit_parks_instead_of_dropping(self):
        """An unknown outcome must stay visible, not vanish as a zero fill."""
        orch, _, strategy, om, rm = self._make_orchestrator()
        self._arm(orch, strategy, om, rm, UNVERIFIED_RESULT)

        asyncio.run(orch._evaluate_and_trade("GBP_JPY"))

        pos = orch._positions.get("GBP_JPY")
        self.assertIsNotNone(pos, "unverified entry was dropped")
        self.assertEqual(pos["state"], "ENTRY_UNRECONCILED")
        self.assertEqual(pos["units"], 1000)
        # Distances kept so the bracket can be rebuilt on reconcile.
        self.assertEqual(pos["sl_dist"], 0.10)
        self.assertEqual(pos["tp_dist"], 0.30)
        # The in-flight reservation handed over to _positions.
        self.assertNotIn("GBP_JPY", orch._pending_entries)

    def test_clean_miss_does_not_park(self):
        """A definitive reject took no position; parking would block nothing."""
        orch, _, strategy, om, rm = self._make_orchestrator()
        self._arm(orch, strategy, om, rm, CLEAN_MISS_RESULT)

        asyncio.run(orch._evaluate_and_trade("GBP_JPY"))

        self.assertNotIn("GBP_JPY", orch._positions)
        self.assertNotIn("GBP_JPY", orch._pending_entries)

    def test_parked_entry_is_not_stop_monitored(self):
        """
        The tick monitor must ignore a parked record: running a stop against a
        position that may not exist would close something we do not hold.
        """
        orch, _, _, om, _ = self._make_orchestrator()
        self._park(orch)

        # A price far through where the stop would be, if it were armed.
        orch._on_tick("GBP_JPY", bid=50.0, ask=50.1)

        om.close_position.assert_not_called()
        self.assertEqual(
            orch._positions["GBP_JPY"]["state"], "ENTRY_UNRECONCILED"
        )

    # ── reconciling ─────────────────────────────────────────────────────

    def test_reconcile_flat_drops_the_record(self):
        """Broker is flat: nothing was taken, so the record goes away."""
        orch, _, _, om, _ = self._make_orchestrator()
        self._park(orch)
        om.sync_position.return_value = True
        om.get_net_position.return_value = 0

        asyncio.run(orch._reconcile_unverified_entries())

        self.assertNotIn("GBP_JPY", orch._positions)

    def test_reconcile_open_arms_the_stop(self):
        """
        Broker holds it: the record becomes OPEN so the tick monitor takes
        over, with the bracket rebuilt from the stored distances around the
        price actually filled (98.0), not the stale signal price (100.0).
        """
        orch, _, _, om, _ = self._make_orchestrator()
        self._park(orch)
        om.sync_position.return_value = True
        om.get_net_position.return_value = 1000
        om.get_average_entry_price.return_value = 98.0

        asyncio.run(orch._reconcile_unverified_entries())

        pos = orch._positions["GBP_JPY"]
        self.assertEqual(pos["state"], "OPEN")
        self.assertEqual(pos["units"], 1000)
        self.assertAlmostEqual(pos["entry"], 98.0)
        self.assertAlmostEqual(pos["sl"], 97.90)   # 98.0 - 0.10
        self.assertAlmostEqual(pos["tp"], 98.30)   # 98.0 + 0.30

    def test_reconcile_short_orientation(self):
        """A short's stop sits ABOVE entry and its target below."""
        orch, _, _, om, _ = self._make_orchestrator()
        self._park(orch, units=-1000, direction="short")
        om.sync_position.return_value = True
        om.get_net_position.return_value = -1000
        om.get_average_entry_price.return_value = 98.0

        asyncio.run(orch._reconcile_unverified_entries())

        pos = orch._positions["GBP_JPY"]
        self.assertEqual(pos["state"], "OPEN")
        self.assertAlmostEqual(pos["sl"], 98.10)
        self.assertAlmostEqual(pos["tp"], 97.70)

    def test_reconcile_sync_failure_keeps_it_parked(self):
        """An unreadable broker is never resolved by assumption."""
        orch, _, _, om, _ = self._make_orchestrator()
        self._park(orch)
        om.sync_position.return_value = False

        asyncio.run(orch._reconcile_unverified_entries())

        self.assertEqual(
            orch._positions["GBP_JPY"]["state"], "ENTRY_UNRECONCILED"
        )


if __name__ == "__main__":
    unittest.main()
