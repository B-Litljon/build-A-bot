import asyncio
import unittest
import warnings
from unittest.mock import MagicMock, patch, ANY
import sys
from pathlib import Path

# Suppress unawaited-coroutine RuntimeWarning when mocking asyncio.run_coroutine_threadsafe
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.execution.oanda_scalper_orchestrator import OandaScalperOrchestrator


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

    # ── helpers ────────────────────────────────────────────────────────

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
