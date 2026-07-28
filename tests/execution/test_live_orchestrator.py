"""State machine tests for LiveOrchestrator._on_trade_update.

These tests exercise the state transitions in `_on_trade_update` without
spinning up Alpaca clients, models, or websocket streams. SymbolContext and
LiveOrchestrator are constructed via ``__new__`` to bypass their heavy
``__init__`` chains; only the attributes the handler reads are populated.

Three suites live here: the original state-machine tests, the thread-ownership
regression tests added 2026-07-26 with the concurrency fix, and the
trade-persistence regression tests added 2026-07-27.

Glossary:
    __new__ construction -- SymbolContext and LiveOrchestrator are built without
        running __init__, so no Alpaca client, model or websocket is needed.
        Only the attributes under test are populated.
    TestThreadOwnership -- pins the ownership contract: the event loop is the
        SOLE writer of SymbolContext state.
    __setattr__ instrumentation -- the technique. SymbolContext.__setattr__ is
        patched to record (attribute, thread id) for every write, then the test
        drives a real sealed-bar -> inference -> signal -> order cycle and
        asserts every recorded write happened on the loop thread. This catches
        a whole CLASS of bug rather than one instance.
    vacuity guard -- the test also asserts all eight expected fields were
        actually written. Without it, a test that accidentally exercised
        nothing would still pass.
    negative proof -- the instrumentation was validated by temporarily
        injecting a thread-side write and confirming both tests FAILED, then
        removing it. An assertion never observed failing is not evidence.
    _make_history_df / _make_features_df / _make_inference_orch -- builders for
        synthetic bars, a one-row feature frame, and an orchestrator with
        mocked models and broker.
    natr_14 = 0.05 -- chosen so the volatility kill switch passes AND the
        minimum-stop floor fires, exercising both paths in one cycle.
    TestTradePersistence -- pins the ownership rule that whoever writes
        authoritative position state persists it. Uses the REAL _save_state
        against a real file in a temp cwd (STATE_FILE is a relative path), so
        assertions are about the file the live bot would actually write rather
        than about a mock having been called.
    fill_lands_during_submission -- the fast-fill simulation: asyncio.to_thread
        is patched so the BUY fill is processed on the loop while
        _handle_signal is still awaiting the submission, which is the real
        ordering that lets a stale requested-qty clobber the filled qty.
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import polars as pl

# Match the import path used by tests/verify_warmup.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from execution.live_orchestrator import (  # noqa: E402
    MIN_HISTORY_BARS,
    STATE_FILE,
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
    ctx.last_price = None
    ctx.last_atr = None
    ctx.last_conviction = None
    ctx.htf_cache = None
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


def _make_history_df(n: int, end_ts: datetime) -> pl.DataFrame:
    """Flat synthetic OHLCV history ending at end_ts, one bar per minute."""
    start = end_ts - timedelta(minutes=n - 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(n)],
            "open": [100.0] * n,
            "high": [100.5] * n,
            "low": [99.5] * n,
            "close": [100.0] * n,
            "volume": [10.0] * n,
        }
    )


def _make_features_df(ts: datetime) -> pl.DataFrame:
    """One synthetic feature row with every column _run_inference consumes.

    natr_14=0.05 is chosen so the ATR kill switch passes AND the SL distance
    (0.5×ATR = 0.025% of price) lands below the MIN_SL_PCT floor (0.15%),
    forcing _submit_entry_order down the SL-floor-adjustment path.
    """
    return pl.DataFrame(
        {
            "timestamp": [ts],
            "close": [100.0],
            "rsi_14": [50.0],
            "ppo": [0.1],
            "natr_14": [0.05],
            "bb_pct_b": [0.5],
            "bb_width_pct": [1.0],
            "price_sma50_ratio": [1.0],
            "log_return": [0.001],
            "hour_of_day": [12.0],
            "dist_sma50": [0.1],
            "vol_rel": [1.0],
            "htf_rsi_14": [55.0],
            "htf_trend_agreement": [1.0],
            "htf_vol_rel": [1.0],
            "htf_bb_pct_b": [0.5],
            "range_coil_10": [0.5],
            "bar_body_pct": [0.4],
            "bar_upper_wick_pct": [0.3],
            "bar_lower_wick_pct": [0.3],
        }
    )


def _make_inference_orch(
    ctx: SymbolContext,
    features_df: pl.DataFrame,
    angel_prob: float = 0.8,
    devil_prob: float = 0.8,
) -> LiveOrchestrator:
    """Orchestrator with everything the bar→inference→order cycle touches
    mocked at the I/O boundary (models, feature pipeline, Alpaca REST)."""
    orch = LiveOrchestrator.__new__(LiveOrchestrator)
    orch._contexts = {ctx.symbol: ctx}
    orch._crypto_set = {ctx.symbol}
    orch._stock_set = set()
    orch._market_open = True
    orch._devil_threshold = 0.5
    orch._activity_log = deque(maxlen=5)
    orch._save_state = MagicMock()
    orch._notifier = MagicMock()

    orch._feature_engineer = MagicMock()
    orch._feature_engineer.run.return_value = features_df

    orch._strategy = MagicMock()
    orch._strategy.angel_model.predict_proba.return_value = np.array(
        [[1.0 - angel_prob, angel_prob]]
    )
    orch._strategy.devil_model.predict_proba.return_value = np.array(
        [[1.0 - devil_prob, devil_prob]]
    )

    orch._trading_client = MagicMock()
    orch._trading_client.get_account.return_value = types.SimpleNamespace(
        equity="100000", buying_power="200000", cash="10000"
    )
    orch._trading_client.submit_order.return_value = types.SimpleNamespace(
        id="order-1"
    )
    return orch


class TestThreadOwnership(unittest.IsolatedAsyncioTestCase):
    """Regression tests for the SymbolContext ownership contract.

    Worker threads (_run_inference, _submit_entry_order) must never write to
    SymbolContext — every mutation must happen on the event-loop thread.
    SymbolContext.__setattr__ is instrumented to record the writing thread
    through a full sealed-bar → inference → signal → order-submit cycle.
    """

    def _make_bar(self, end_ts: datetime) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            symbol="BTC/USD",
            timestamp=end_ts,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=10.0,
        )

    def _wire_aggregator(self, ctx: SymbolContext, end_ts: datetime) -> None:
        ctx.aggregator.add_bar.return_value = True  # every bar seals
        ctx.aggregator.history_df = _make_history_df(MIN_HISTORY_BARS, end_ts)

    async def test_full_signal_cycle_writes_only_on_loop_thread(self):
        end_ts = datetime(2026, 7, 26, 12, 3, tzinfo=timezone.utc)
        ctx = _make_ctx()
        self._wire_aggregator(ctx, end_ts)
        orch = _make_inference_orch(ctx, _make_features_df(end_ts))

        writes = []

        def recording_setattr(obj, name, value):
            writes.append((name, threading.get_ident()))
            object.__setattr__(obj, name, value)

        loop_thread = threading.get_ident()
        with patch.object(SymbolContext, "__setattr__", recording_setattr):
            await orch._on_bar(self._make_bar(end_ts))

        # Core assertion: no SymbolContext write from a worker thread.
        off_loop = [(name, tid) for name, tid in writes if tid != loop_thread]
        self.assertEqual(
            off_loop,
            [],
            f"SymbolContext mutated from non-event-loop thread(s): {off_loop}",
        )

        # Vacuity guard: the cycle must actually have exercised every field
        # the old code used to write from threads (plus the loop-side ones).
        written = {name for name, _ in writes}
        for field in (
            "last_price",
            "state",
            "htf_cache",
            "last_atr",
            "last_conviction",
            "sl_price",
            "tp_price",
            "entry_qty",
        ):
            self.assertIn(field, written, f"cycle never wrote ctx.{field}")

        # End-state sanity: order submitted, floored SL applied by the loop.
        orch._trading_client.submit_order.assert_called_once()
        self.assertEqual(ctx.state, SymbolState.PENDING)
        self.assertAlmostEqual(ctx.sl_price, 100.0 - 100.0 * 0.0015, places=6)
        self.assertEqual(ctx.entry_qty, 95.0)  # capped by cash*0.95/price
        self.assertIsNotNone(ctx.htf_cache)
        # No persist on this path: the symbol is still PENDING and
        # _save_state serializes only IN_TRADE symbols, so a call here could
        # never record the trade. Persistence is owned by the fill handler
        # (see TestTradePersistence). This previously asserted
        # assert_called_once, which pinned the no-op save as if it worked.
        orch._save_state.assert_not_called()

    async def test_htf_cache_propagates_on_angel_reject(self):
        """A cold-path HTF recompute must reach ctx even when the Angel
        rejects the bar — otherwise the warm-path optimization silently dies."""
        end_ts = datetime(2026, 7, 26, 12, 3, tzinfo=timezone.utc)
        ctx = _make_ctx()
        self._wire_aggregator(ctx, end_ts)
        orch = _make_inference_orch(
            ctx, _make_features_df(end_ts), angel_prob=0.1
        )

        writes = []

        def recording_setattr(obj, name, value):
            writes.append((name, threading.get_ident()))
            object.__setattr__(obj, name, value)

        loop_thread = threading.get_ident()
        with patch.object(SymbolContext, "__setattr__", recording_setattr):
            await orch._on_bar(self._make_bar(end_ts))

        off_loop = [(name, tid) for name, tid in writes if tid != loop_thread]
        self.assertEqual(off_loop, [])

        # No trade — but the cold-path cache refresh still landed.
        orch._trading_client.submit_order.assert_not_called()
        self.assertEqual(ctx.state, SymbolState.FLAT)
        self.assertIsNotNone(ctx.htf_cache)
        self.assertEqual(ctx.htf_cache.htf_rsi_14, 55.0)
        self.assertEqual(ctx.last_atr, 0.05)
        self.assertIsNone(ctx.last_conviction)


class TestTradePersistence(unittest.IsolatedAsyncioTestCase):
    """Regression tests for the trade-persistence seam (2026-07-27).

    _save_state serializes only IN_TRADE symbols, and _load_state uses that
    file on startup to re-inject SL/TP for positions Alpaca still shows open.
    For crypto the software watchdog is the ONLY exit mechanism, so a position
    missing from the file restarts unprotected.

    These use the REAL _save_state against a real file in a temp working
    directory, so they assert on the artifact the live bot would actually
    write rather than on a mock having been called.
    """

    def setUp(self) -> None:
        # STATE_FILE is a relative path, so chdir redirects the real write.
        self._tmp = tempfile.TemporaryDirectory()
        self._prev_cwd = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self) -> None:
        os.chdir(self._prev_cwd)
        self._tmp.cleanup()

    @staticmethod
    def _saved() -> dict:
        """Parse active_trades.json; {} when it was never written."""
        path = Path(STATE_FILE)
        if not path.exists():
            return {}
        with open(path) as fh:
            return json.load(fh)

    @staticmethod
    def _bar(end_ts: datetime) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            symbol="BTC/USD",
            timestamp=end_ts,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=10.0,
        )

    def _persisting_orch(self, ctx, features_df=None):
        """Orchestrator with the REAL _save_state bound to it."""
        if features_df is None:
            orch = _make_orch(ctx)
        else:
            orch = _make_inference_orch(ctx, features_df)
            orch._enter_cooling = AsyncMock()
        orch._save_state = types.MethodType(LiveOrchestrator._save_state, orch)
        return orch

    async def test_buy_fill_persists_trade_with_authoritative_qty(self):
        """Symptom 1: a filled trade must reach active_trades.json.

        Before the fix, _handle_signal's save ran while the symbol was still
        PENDING (which _save_state filters out) and the fill branch never
        saved at all — so a crash between the fill and the next unrelated
        save left an open position invisible to _load_state.
        """
        end_ts = datetime(2026, 7, 26, 12, 3, tzinfo=timezone.utc)
        ctx = _make_ctx()
        ctx.aggregator.add_bar.return_value = True
        ctx.aggregator.history_df = _make_history_df(MIN_HISTORY_BARS, end_ts)
        orch = self._persisting_orch(ctx, _make_features_df(end_ts))

        await orch._on_bar(self._bar(end_ts))

        # Order submitted, but still PENDING — correctly not yet persisted.
        orch._trading_client.submit_order.assert_called_once()
        self.assertEqual(ctx.state, SymbolState.PENDING)
        self.assertEqual(self._saved(), {})

        # filled_qty deliberately differs from the submitted 95.0.
        await orch._on_trade_update(
            _make_event("fill", "BUY", filled_qty=93.5, filled_avg_price=100.25)
        )

        self.assertEqual(ctx.state, SymbolState.IN_TRADE)
        saved = self._saved()
        self.assertIn("BTC/USD", saved, "filled trade never reached the state file")
        self.assertEqual(saved["BTC/USD"]["qty"], 93.5)
        # The floored SL must persist too — it is what the watchdog monitors.
        self.assertAlmostEqual(
            saved["BTC/USD"]["sl_price"], 100.0 - 100.0 * 0.0015, places=6
        )

    async def test_fast_fill_during_await_is_not_clobbered(self):
        """Symptom 2: ctx.lock is released across the submission await, so the
        fill can be processed first. The post-await seed must not overwrite
        the authoritative filled_qty with the requested qty."""
        end_ts = datetime(2026, 7, 26, 12, 3, tzinfo=timezone.utc)
        ctx = _make_ctx()
        ctx.aggregator.add_bar.return_value = True
        ctx.aggregator.history_df = _make_history_df(MIN_HISTORY_BARS, end_ts)
        orch = self._persisting_orch(ctx, _make_features_df(end_ts))

        fill = _make_event("fill", "BUY", filled_qty=93.5, filled_avg_price=100.25)

        async def fill_lands_during_submission(fn, *args, **kwargs):
            result = fn(*args, **kwargs)
            if fn == orch._submit_entry_order:
                # The fill is handled on the loop before _handle_signal resumes.
                await orch._on_trade_update(fill)
            return result

        with patch("asyncio.to_thread", fill_lands_during_submission):
            await orch._on_bar(self._bar(end_ts))

        self.assertEqual(ctx.state, SymbolState.IN_TRADE)
        self.assertEqual(
            ctx.entry_qty, 93.5, "requested qty clobbered the authoritative fill"
        )
        self.assertEqual(self._saved()["BTC/USD"]["qty"], 93.5)

    async def test_partial_then_terminal_fill_records_final_qty(self):
        """Symptom 3: the BUY branch required PENDING, so a partial_fill
        flipped state to IN_TRADE and the terminal fill was then skipped
        entirely — the final cumulative qty and average price were lost."""
        ctx = _make_ctx(state=SymbolState.PENDING)
        ctx.sl_price = 99.0
        ctx.tp_price = 105.0
        orch = self._persisting_orch(ctx)

        await orch._on_trade_update(
            _make_event("partial_fill", "BUY", filled_qty=40.0, filled_avg_price=100.0)
        )
        self.assertEqual(ctx.state, SymbolState.IN_TRADE)
        self.assertEqual(ctx.entry_qty, 40.0)
        self.assertEqual(self._saved()["BTC/USD"]["qty"], 40.0)

        # Alpaca reports filled_qty / filled_avg_price CUMULATIVELY, so the
        # terminal event carries the final totals, not just the increment.
        await orch._on_trade_update(
            _make_event("fill", "BUY", filled_qty=100.0, filled_avg_price=101.5)
        )
        self.assertEqual(ctx.state, SymbolState.IN_TRADE)
        self.assertEqual(ctx.entry_qty, 100.0, "terminal fill was skipped")
        self.assertEqual(ctx.entry_price, 101.5)
        self.assertEqual(self._saved()["BTC/USD"]["qty"], 100.0)

    async def test_cancel_removes_entry_from_state_file(self):
        """Requirement 2: the entry must leave the file the moment the symbol
        leaves IN_TRADE, or _load_state would re-inject SL/TP on restart for a
        position that no longer exists."""
        ctx = _make_ctx(state=SymbolState.IN_TRADE)
        ctx.sl_price = 99.0
        ctx.tp_price = 105.0
        ctx.entry_qty = 10.0
        orch = self._persisting_orch(ctx)

        orch._save_state()
        self.assertIn("BTC/USD", self._saved())

        await orch._on_trade_update(_make_event("canceled", "BUY"))

        self.assertEqual(ctx.state, SymbolState.FLAT)
        self.assertNotIn(
            "BTC/USD", self._saved(), "canceled trade left stale entry in state file"
        )


if __name__ == "__main__":
    unittest.main()
