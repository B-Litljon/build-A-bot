"""State machine tests for LiveOrchestrator._on_trade_update.

These tests exercise the state transitions in `_on_trade_update` without
spinning up Alpaca clients, models, or websocket streams. SymbolContext and
LiveOrchestrator are constructed via ``__new__`` to bypass their heavy
``__init__`` chains; only the attributes the handler reads are populated.
"""

import asyncio
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock

# Match the import path used by tests/verify_warmup.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from execution.live_orchestrator import (  # noqa: E402
    LiveOrchestrator,
    SymbolContext,
    SymbolState,
)


def _make_ctx(symbol: str = "BTC/USD", state: SymbolState = SymbolState.FLAT) -> SymbolContext:
    """Build a SymbolContext without invoking the heavy __init__ chain."""
    ctx = SymbolContext.__new__(SymbolContext)
    ctx.symbol = symbol
    ctx.is_crypto = "/" in symbol
    ctx.state = state
    ctx.lock = asyncio.Lock()
    ctx.last_client_order_id = None
    ctx.entry_price = None
    ctx.entry_qty = None
    ctx.sl_price = None
    ctx.tp_price = None
    ctx._cooling_task = None
    ctx.aggregator = MagicMock()
    return ctx


def _make_orch(ctx: SymbolContext) -> LiveOrchestrator:
    """Build a LiveOrchestrator with only the attributes _on_trade_update uses."""
    orch = LiveOrchestrator.__new__(LiveOrchestrator)
    orch._contexts = {ctx.symbol: ctx}
    orch._enter_cooling = AsyncMock()
    orch._log_activity = MagicMock()
    orch._notifier = MagicMock()
    return orch


def _make_event(
    event_type: str,
    side: str,
    symbol: str = "BTC/USD",
    filled_avg_price: float = 100.0,
    filled_qty: float = 1.0,
    client_order_id: str = "test_order_1",
) -> types.SimpleNamespace:
    """Build a TradeUpdate-shaped object the handler can introspect."""
    order = types.SimpleNamespace(
        symbol=symbol,
        side=side,
        client_order_id=client_order_id,
        filled_avg_price=filled_avg_price,
        filled_qty=filled_qty,
    )
    return types.SimpleNamespace(event=event_type, order=order)


class TestOnTradeUpdate(unittest.IsolatedAsyncioTestCase):
    """State machine assertions for _on_trade_update."""

    async def test_sell_fill_pending_exit_enters_cooling(self):
        """Finding #1.1: SELL fill while PENDING_EXIT must transition to COOLING."""
        ctx = _make_ctx(state=SymbolState.PENDING_EXIT)
        orch = _make_orch(ctx)

        await orch._on_trade_update(_make_event("fill", "SELL"))

        orch._enter_cooling.assert_awaited_once_with(ctx)

    async def test_sell_fill_in_trade_enters_cooling(self):
        """SELL fill while IN_TRADE (bracket TP/SL hit) must transition to COOLING."""
        ctx = _make_ctx(state=SymbolState.IN_TRADE)
        orch = _make_orch(ctx)

        await orch._on_trade_update(_make_event("fill", "SELL"))

        orch._enter_cooling.assert_awaited_once_with(ctx)

    async def test_sell_partial_fill_does_not_enter_cooling(self):
        """SELL partial_fill must NOT transition to COOLING — wait for terminal fill."""
        ctx = _make_ctx(state=SymbolState.IN_TRADE)
        orch = _make_orch(ctx)

        await orch._on_trade_update(_make_event("partial_fill", "SELL"))

        orch._enter_cooling.assert_not_awaited()
        self.assertEqual(ctx.state, SymbolState.IN_TRADE)

    async def test_sell_fill_then_partial_does_not_double_cool(self):
        """A late partial_fill arriving after the order cooled must not re-cool.

        Exercises the actual state gate: _enter_cooling is mocked, so we
        manually transition to COOLING to simulate its effect, then fire a
        stray partial_fill and assert call_count is still 1.
        """
        ctx = _make_ctx(state=SymbolState.IN_TRADE)
        orch = _make_orch(ctx)

        await orch._on_trade_update(_make_event("fill", "SELL"))
        self.assertEqual(orch._enter_cooling.call_count, 1)

        ctx.state = SymbolState.COOLING

        await orch._on_trade_update(_make_event("partial_fill", "SELL"))
        self.assertEqual(orch._enter_cooling.call_count, 1)

    async def test_buy_fill_pending_enters_in_trade(self):
        """No regression: BUY fill while PENDING must transition to IN_TRADE."""
        ctx = _make_ctx(state=SymbolState.PENDING)
        orch = _make_orch(ctx)

        await orch._on_trade_update(
            _make_event("fill", "BUY", filled_avg_price=125.50, filled_qty=2.0)
        )

        self.assertEqual(ctx.state, SymbolState.IN_TRADE)
        self.assertEqual(ctx.entry_price, 125.50)
        self.assertEqual(ctx.entry_qty, 2.0)
        orch._enter_cooling.assert_not_awaited()

    async def test_rejected_clears_client_order_id(self):
        """After a rejected order, last_client_order_id must clear so a retry signal isn't deduplicated."""
        ctx = _make_ctx(state=SymbolState.PENDING)
        ctx.last_client_order_id = "test_order_1"
        orch = _make_orch(ctx)

        await orch._on_trade_update(_make_event("rejected", "BUY"))

        self.assertEqual(ctx.state, SymbolState.FLAT)
        self.assertIsNone(ctx.last_client_order_id)


if __name__ == "__main__":
    unittest.main()
