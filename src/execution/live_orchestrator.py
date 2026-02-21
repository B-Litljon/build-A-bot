"""
Universal Scalper V3.1 — Live Forward-Testing Orchestrator (Phase 5)
=====================================================================

Architecture: asyncio.to_thread concurrency model
--------------------------------------------------
  StockDataStream (async WebSocket)
       │ raw bar events per symbol
       ▼
  LiveBarAggregator.add_bar(tick)          ← in-event-loop, microseconds
       │ returns True when a 1-min bar seals
       ▼
  asyncio.to_thread(_run_inference)        ← CPU-bound RF offloaded to thread pool
       │ Signal | None
       ▼
  [Volatility Kill Switch]                 ← natr_14 vs ATR_KILL_SWITCH_THRESHOLD
       │ passes
       ▼
  [SymbolState guard — asyncio.Lock]       ← FLAT required; → PENDING
       │
       ▼
  asyncio.to_thread(_submit_bracket)       ← REST call offloaded to thread pool
       │
  TradeUpdateStream (async WebSocket)
       │ fill / cancel / closed events
       ▼
  _on_trade_update → SymbolState machine   ← PENDING→IN_TRADE→COOLING(5m)→FLAT

ATR Kill Switch Threshold: 0.5204 (from drift_report.json, High-regime trigger)
Cooling-off period: 5 minutes after any bracket closes
Schema failure policy: catch → Discord alert → drop bar → symbol stays FLAT
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import traceback
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Alpaca imports — live streams + REST
# ---------------------------------------------------------------------------
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.stream import TradingStream

# ---------------------------------------------------------------------------
# Internal imports — compose, don't reimplement
# ---------------------------------------------------------------------------
# NOTE: sys.path is extended by the entry-point (src/main.py or equivalent)
# so bare module names resolve correctly inside the src/ tree.
from core.notification_manager import NotificationManager
from core.signal import Signal, SignalType
from ml.feature_pipeline import FeatureEngineer
from strategies.concrete_strategies.ml_strategy import MLStrategy
from utils.bar_aggregator import LiveBarAggregator

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# From drift_report.json — High-regime calibration_gap threshold that
# triggers the safety switch.  Do NOT change without re-running
# src/analysis/reinforcement_voter.py on fresh OOS data.
ATR_KILL_SWITCH_THRESHOLD: float = 0.5204  # natr_14 percentage units

# Angel/Devil classification thresholds (matching training configuration)
ANGEL_THRESHOLD: float = 0.40
DEVIL_THRESHOLD: float = 0.50

# Bracket ATR multipliers (matching OOS backtest that produced +39.03R)
SL_ATR_MULTIPLIER: float = 1.5
TP_ATR_MULTIPLIER: float = 3.0

# Cooldown after any bracket resolves before re-entry is allowed (seconds)
COOLING_SECONDS: int = 5 * 60  # 5 minutes

# Minimum history rows required before inference is attempted.
# Must be >= SMA-50 period + safety margin.
MIN_HISTORY_BARS: int = 60

# Rolling window retained by each LiveBarAggregator (memory cap)
HISTORY_SIZE: int = 120

# Capital and risk settings (paper trading defaults; override via env)
ACCOUNT_RISK_PER_TRADE: float = 0.02  # 2% of account equity per trade

# Market hours clock TTL (seconds) — avoids hammering the REST API
CLOCK_CACHE_TTL: float = 30.0


# ---------------------------------------------------------------------------
# Symbol state machine
# ---------------------------------------------------------------------------


class SymbolState(Enum):
    """Lifecycle states for a single symbol's trading slot."""

    FLAT = auto()  # No position; eligible for new signals
    PENDING = auto()  # Order submitted; awaiting fill confirmation
    IN_TRADE = auto()  # Position filled; bracket orders active
    COOLING = auto()  # Bracket resolved; cooling-off timer running


# ---------------------------------------------------------------------------
# Per-symbol runtime context
# ---------------------------------------------------------------------------


class SymbolContext:
    """
    Holds all mutable state for a single tracked symbol.

    Thread-safety note: `state` is read/written only from the asyncio event
    loop (the lock is an asyncio.Lock).  Inference runs in a thread pool but
    only *reads* the immutable `aggregator.history_df` snapshot; it never
    mutates SymbolContext directly.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol: str = symbol
        self.aggregator: LiveBarAggregator = LiveBarAggregator(
            timeframe=1, history_size=HISTORY_SIZE
        )
        self.state: SymbolState = SymbolState.FLAT
        self.lock: asyncio.Lock = asyncio.Lock()

        # Deduplication — client_order_id for the most recent bracket entry
        self.last_client_order_id: Optional[str] = None

        # Filled entry price used for bracket SL/TP calculation
        self.entry_price: Optional[float] = None
        self.entry_qty: Optional[float] = None

        # Handle to the cooling timer so we can cancel it on shutdown
        self._cooling_task: Optional[asyncio.Task] = None

    def __repr__(self) -> str:  # pragma: no cover
        return f"SymbolContext(symbol={self.symbol!r}, state={self.state.name})"


# ---------------------------------------------------------------------------
# Live Orchestrator
# ---------------------------------------------------------------------------


class LiveOrchestrator:
    """
    Async daemon that drives live paper-trading for the Universal Scalper.

    Usage (from an async entry-point)::

        orchestrator = LiveOrchestrator(symbols=["TSLA", "NVDA", "MARA"])
        await orchestrator.run()

    Graceful shutdown is handled automatically on SIGTERM / SIGINT.
    """

    def __init__(
        self,
        symbols: List[str],
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        angel_model_path: str = "src/ml/models/angel_rf_model.joblib",
        devil_model_path: str = "src/ml/models/devil_rf_model.joblib",
    ) -> None:
        load_dotenv()

        self._api_key: str = api_key or os.environ["ALPACA_API_KEY"]
        self._secret_key: str = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self._paper: bool = paper
        self._symbols: List[str] = [s.upper() for s in symbols]

        # -- Core components (stateless / shared across symbols) --
        self._strategy = MLStrategy(
            angel_path=angel_model_path,
            devil_path=devil_model_path,
            angel_threshold=ANGEL_THRESHOLD,
            devil_threshold=DEVIL_THRESHOLD,
            warmup_period=MIN_HISTORY_BARS,
        )
        self._feature_engineer = FeatureEngineer()
        self._notifier = NotificationManager()
        self._trading_client = TradingClient(
            self._api_key, self._secret_key, paper=self._paper
        )

        # -- Per-symbol state --
        self._contexts: Dict[str, SymbolContext] = {
            sym: SymbolContext(sym) for sym in self._symbols
        }

        # -- Alpaca WebSocket clients (created fresh per run()) call) --
        self._data_stream: Optional[StockDataStream] = None
        self._trade_stream: Optional[TradingStream] = None

        # -- Market hours cache --
        self._market_open: bool = False
        self._clock_last_checked: float = 0.0

        # -- Shutdown coordination --
        self._shutdown_event: asyncio.Event = asyncio.Event()

        logger.info(
            "LiveOrchestrator initialised | symbols=%s | paper=%s | "
            "ATR_kill=%.4f | cooling=%ds",
            self._symbols,
            paper,
            ATR_KILL_SWITCH_THRESHOLD,
            COOLING_SECONDS,
        )

    # -----------------------------------------------------------------------
    # Public entry-point
    # -----------------------------------------------------------------------

    async def run(self) -> None:
        """
        Start the orchestrator.  Blocks until SIGTERM/SIGINT is received or
        an unrecoverable error terminates both WebSocket streams.
        """
        loop = asyncio.get_running_loop()

        # Register OS signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._request_shutdown)

        self._notifier.send_system_message(
            f"[LiveOrchestrator] Starting paper-trading daemon | "
            f"symbols={self._symbols} | "
            f"ATR_kill_switch={ATR_KILL_SWITCH_THRESHOLD}"
        )

        # Verify market clock once before subscribing
        await self._refresh_market_clock()

        # Pre-load historical bars so SMA-50 and all indicators are ready
        # at market open.  Must run after credentials are validated and
        # before WebSocket subscriptions are registered.
        await self._warmup_aggregator()

        # Build WebSocket clients
        self._data_stream = StockDataStream(
            self._api_key, self._secret_key, feed=DataFeed.IEX
        )
        self._trade_stream = TradingStream(
            self._api_key, self._secret_key, paper=self._paper
        )

        # Subscribe handlers
        self._data_stream.subscribe_bars(self._on_bar, *self._symbols)
        self._trade_stream.subscribe_trade_updates(self._on_trade_update)

        logger.info("WebSocket subscriptions registered — starting streams.")

        try:
            # Run both streams concurrently; either can raise on disconnect
            await asyncio.gather(
                self._run_data_stream(),
                self._run_trade_stream(),
                self._clock_refresh_loop(),
            )
        except asyncio.CancelledError:
            logger.info("Orchestrator tasks cancelled — shutting down cleanly.")
        except Exception as exc:
            logger.critical(
                "Unrecoverable error in orchestrator gather: %s", exc, exc_info=True
            )
            self._notifier.send_system_message(
                f"[LiveOrchestrator] CRITICAL: unrecoverable error — {exc}"
            )
        finally:
            await self._shutdown()

    # -----------------------------------------------------------------------
    # Stream runners (thin wrappers so gather() can cancel them)
    # -----------------------------------------------------------------------

    async def _run_data_stream(self) -> None:
        """Runs StockDataStream until shutdown is requested."""
        try:
            await self._data_stream._run_forever()  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("StockDataStream error: %s", exc, exc_info=True)
            self._request_shutdown()

    async def _run_trade_stream(self) -> None:
        """Runs TradingStream until shutdown is requested."""
        try:
            await self._trade_stream._run_forever()  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("TradingStream error: %s", exc, exc_info=True)
            self._request_shutdown()

    async def _clock_refresh_loop(self) -> None:
        """Periodically refreshes market-hours status in the background."""
        while not self._shutdown_event.is_set():
            await self._refresh_market_clock()
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=CLOCK_CACHE_TTL
                )
            except asyncio.TimeoutError:
                pass  # Normal — loop again

    # -----------------------------------------------------------------------
    # Primary bar handler (called by StockDataStream in the event loop)
    # -----------------------------------------------------------------------

    async def _on_bar(self, bar) -> None:
        """
        Ingests a raw 1-minute bar from the Alpaca WebSocket.

        This handler MUST be non-blocking.  All CPU-bound work is offloaded
        via asyncio.to_thread() so the event loop is never stalled.

        bar attributes (alpaca-py Bar object):
            symbol, timestamp, open, high, low, close, volume
        """
        symbol: str = bar.symbol.upper()

        if symbol not in self._contexts:
            logger.debug("Received bar for untracked symbol %s — ignoring.", symbol)
            return

        # Market hours gate — no inference outside regular session
        if not self._market_open:
            logger.debug("[%s] Market closed — skipping bar.", symbol)
            return

        ctx = self._contexts[symbol]

        # Convert Alpaca bar object → dict that LiveBarAggregator expects
        bar_dict = {
            "timestamp": self._ensure_utc(bar.timestamp),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }

        # Feed to aggregator (in-loop, pure Python — microseconds)
        bar_sealed = ctx.aggregator.add_bar(bar_dict)

        if not bar_sealed:
            return  # Still accumulating sub-bars; nothing to do yet

        # A complete 1-minute bar just closed — kick off inference off-thread
        history_snapshot: pl.DataFrame = ctx.aggregator.history_df.clone()

        if len(history_snapshot) < MIN_HISTORY_BARS:
            logger.debug(
                "[%s] Warming up (%d/%d bars).",
                symbol,
                len(history_snapshot),
                MIN_HISTORY_BARS,
            )
            return

        # Offload CPU-bound inference — does NOT block the event loop
        signal_result: Optional[Signal] = await asyncio.to_thread(
            self._run_inference, symbol, history_snapshot
        )

        if signal_result is None:
            return

        # All execution checks happen back in the event loop (state machine)
        await self._handle_signal(ctx, signal_result)

    # -----------------------------------------------------------------------
    # Inference (runs in thread pool — must be pure / no asyncio calls)
    # -----------------------------------------------------------------------

    def _run_inference(self, symbol: str, history_df: pl.DataFrame) -> Optional[Signal]:
        """
        Execute Angel → Devil two-stage inference on the latest bar.

        Called via asyncio.to_thread; must NOT call any asyncio primitives.

        Returns a Signal on joint Angel+Devil approval, or None.
        Any exception is caught, logged, and reported to Discord (via a
        thread-safe requests.post call inside NotificationManager).
        """
        try:
            # ----------------------------------------------------------------
            # Schema validation — catch mismatches before they corrupt numpy
            # ----------------------------------------------------------------
            required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
            actual_cols = set(history_df.columns)
            missing = required_cols - actual_cols
            if missing:
                raise ValueError(
                    f"[{symbol}] Schema mismatch: missing columns {missing}"
                )

            # ----------------------------------------------------------------
            # Feature engineering (TA-Lib; releases GIL on C extensions)
            # ----------------------------------------------------------------
            features_df = self._feature_engineer.compute_indicators(history_df)

            ml_feature_names = [
                "rsi_14",
                "ppo",
                "natr_14",
                "bb_pct_b",
                "bb_width_pct",
                "price_sma50_ratio",
                "log_return",
                "hour_of_day",
                "dist_sma50",
                "vol_rel",
            ]

            # Drop any rows that still have NaN (warmup artefacts from TA-Lib)
            features_df = features_df.drop_nulls(subset=ml_feature_names)

            if len(features_df) == 0:
                logger.debug("[%s] All rows null after feature drop — skip.", symbol)
                return None

            latest_row = features_df.tail(1)

            # ----------------------------------------------------------------
            # Volatility Kill Switch — checked BEFORE inference to avoid
            # wasted CPU if the regime is already disqualified.
            # ----------------------------------------------------------------
            natr_value: float = float(latest_row["natr_14"][0])
            if natr_value > ATR_KILL_SWITCH_THRESHOLD:
                logger.info(
                    "[%s] ATR Kill Switch ACTIVE | natr_14=%.4f > threshold=%.4f",
                    symbol,
                    natr_value,
                    ATR_KILL_SWITCH_THRESHOLD,
                )
                return None

            # ----------------------------------------------------------------
            # Current price + ATR for bracket sizing
            # ----------------------------------------------------------------
            current_price: float = float(latest_row["close"][0])

            # TA-Lib NATR is a percentage; convert to absolute ATR for bracket math
            # natr_14 = (ATR / close) * 100  →  ATR = (natr_14 / 100) * close
            atr_abs: float = (natr_value / 100.0) * current_price

            # ----------------------------------------------------------------
            # Stage 1 — The Angel (direction, high recall)
            # ----------------------------------------------------------------
            feature_matrix = latest_row.select(ml_feature_names).to_numpy()

            angel_prob: float = float(
                self._strategy.angel_model.predict_proba(feature_matrix)[0, 1]
            )

            if angel_prob < ANGEL_THRESHOLD:
                logger.debug(
                    "[%s] Angel REJECT | prob=%.4f < threshold=%.4f",
                    symbol,
                    angel_prob,
                    ANGEL_THRESHOLD,
                )
                return None

            # ----------------------------------------------------------------
            # Stage 2 — The Devil (conviction, high precision)
            # Meta-feature: original features + angel_prob appended
            # ----------------------------------------------------------------
            import pandas as pd  # local import — pandas only needed here

            meta_df = pd.DataFrame(feature_matrix, columns=ml_feature_names)
            meta_df["angel_prob"] = angel_prob

            devil_prob: float = float(
                self._strategy.devil_model.predict_proba(meta_df)[0, 1]
            )

            if devil_prob < DEVIL_THRESHOLD:
                logger.debug(
                    "[%s] Devil VETO | angel=%.4f devil=%.4f < threshold=%.4f",
                    symbol,
                    angel_prob,
                    devil_prob,
                    DEVIL_THRESHOLD,
                )
                return None

            # ----------------------------------------------------------------
            # Both agreed — build Signal with ATR bracket data in metadata
            # ----------------------------------------------------------------
            bar_timestamp: datetime = latest_row["timestamp"][0]

            logger.info(
                "[%s] SIGNAL | price=%.2f | angel=%.4f | devil=%.4f | "
                "natr=%.4f | atr_abs=%.4f",
                symbol,
                current_price,
                angel_prob,
                devil_prob,
                natr_value,
                atr_abs,
            )

            return Signal(
                symbol=symbol,
                type=SignalType.BUY,
                price=current_price,
                confidence=devil_prob,
                timestamp=bar_timestamp,
                metadata={
                    "angel_prob": angel_prob,
                    "devil_prob": devil_prob,
                    "natr_14": natr_value,
                    "atr_abs": atr_abs,
                    "sl_price": round(current_price - SL_ATR_MULTIPLIER * atr_abs, 4),
                    "tp_price": round(current_price + TP_ATR_MULTIPLIER * atr_abs, 4),
                },
            )

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[%s] Inference error — dropping bar.\n%s", symbol, tb)
            # Thread-safe Discord notification (requests.post internally)
            self._notifier.send_system_message(
                f"[LiveOrchestrator][{symbol}] Inference error — bar dropped.\n"
                f"```{str(exc)[:800]}```"
            )
            return None

    # -----------------------------------------------------------------------
    # Signal handler (back in the event loop after inference returns)
    # -----------------------------------------------------------------------

    async def _handle_signal(self, ctx: SymbolContext, sig: Signal) -> None:
        """
        Gate the signal through the SymbolState machine, then submit the
        bracket order off-thread if all checks pass.

        Layer 1 dedup: SymbolState lock  (must be FLAT)
        Layer 2 dedup: Alpaca TradeUpdateStream drives state transitions
        Layer 3 dedup: client_order_id = symbol + bar_timestamp (unique per bar)
        """
        symbol = ctx.symbol

        async with ctx.lock:
            if ctx.state != SymbolState.FLAT:
                logger.debug(
                    "[%s] Signal ignored — state=%s (not FLAT).",
                    symbol,
                    ctx.state.name,
                )
                return

            # Build unique, idempotent client order id
            ts_iso = sig.timestamp.strftime("%Y%m%dT%H%M%S")
            client_order_id = f"{symbol}_{ts_iso}"

            if client_order_id == ctx.last_client_order_id:
                logger.warning(
                    "[%s] Duplicate client_order_id=%s — skipping.",
                    symbol,
                    client_order_id,
                )
                return

            # Transition: FLAT → PENDING (inside the lock)
            ctx.state = SymbolState.PENDING
            ctx.last_client_order_id = client_order_id

        # Lock released — submit the bracket order off-thread
        logger.info("[%s] State → PENDING | submitting bracket order.", symbol)

        success = await asyncio.to_thread(
            self._submit_bracket_order,
            sig,
            client_order_id,
        )

        if not success:
            # REST call failed — roll back to FLAT so the next bar can retry
            async with ctx.lock:
                ctx.state = SymbolState.FLAT
                ctx.last_client_order_id = None
            logger.warning("[%s] Bracket submission failed — state → FLAT.", symbol)

    # -----------------------------------------------------------------------
    # Order submission (runs in thread pool)
    # -----------------------------------------------------------------------

    def _submit_bracket_order(self, sig: Signal, client_order_id: str) -> bool:
        """
        Submit a bracket (market entry + OTO take-profit + stop-loss) order
        to Alpaca.

        Called via asyncio.to_thread; must NOT call any asyncio primitives.

        Returns True on success, False on any exception.
        """
        symbol = sig.symbol
        sl_price: float = sig.metadata["sl_price"]
        tp_price: float = sig.metadata["tp_price"]

        try:
            # ----------------------------------------------------------------
            # Position sizing: 2% of account equity at risk
            # The risk distance in dollars determines share count.
            # ----------------------------------------------------------------
            account = self._trading_client.get_account()
            equity: float = float(account.equity)
            risk_dollars: float = equity * ACCOUNT_RISK_PER_TRADE
            risk_per_share: float = sig.price - sl_price

            if risk_per_share <= 0:
                logger.error(
                    "[%s] Invalid risk_per_share=%.4f (SL above entry?) — abort.",
                    symbol,
                    risk_per_share,
                )
                return False

            qty: float = risk_dollars / risk_per_share
            qty = max(round(qty, 4), 0.0001)  # Alpaca fractional shares floor

            logger.info(
                "[%s] Bracket order | equity=%.2f | risk=$%.2f | "
                "qty=%.4f | entry~%.2f | SL=%.4f | TP=%.4f | id=%s",
                symbol,
                equity,
                risk_dollars,
                qty,
                sig.price,
                sl_price,
                tp_price,
                client_order_id,
            )

            # ----------------------------------------------------------------
            # Bracket order: market entry with OTO TP + SL legs
            # ----------------------------------------------------------------
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
                order_class="bracket",
                take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(sl_price, 2)),
            )

            order = self._trading_client.submit_order(order_request)
            order_id = str(getattr(order, "id", "unknown"))

            logger.info(
                "[%s] Bracket submitted | alpaca_order_id=%s | client_id=%s",
                symbol,
                order_id,
                client_order_id,
            )

            # Notify Discord
            self._notifier.send_trade_alert(sig, action="ENTRY")

            return True

        except Exception as exc:
            logger.error(
                "[%s] Bracket order submission failed: %s", symbol, exc, exc_info=True
            )
            self._notifier.send_system_message(
                f"[LiveOrchestrator][{symbol}] Bracket order FAILED: {exc}"
            )
            return False

    # -----------------------------------------------------------------------
    # Trade update handler (Alpaca order lifecycle WebSocket)
    # -----------------------------------------------------------------------

    async def _on_trade_update(self, data) -> None:
        """
        Drives SymbolState transitions from Alpaca order lifecycle events.

        This is the authoritative source of truth for position state —
        never rely on polling or timeouts alone.

        Relevant event types (alpaca-py):
            'fill'              — entry order filled → IN_TRADE
            'partial_fill'      — partially filled (treat as fill for state)
            'canceled'          — order canceled   → FLAT
            'expired'           — order expired    → FLAT
            'rejected'          — order rejected   → FLAT
            'replaced'          — order replaced   → no state change
            'order_cancel_rejected' — cancel failed (ignore)
            'trade_update' w/ position closed via bracket TP/SL → COOLING
        """
        try:
            event_type: str = getattr(data, "event", "")
            order = getattr(data, "order", None)

            if order is None:
                return

            symbol: str = getattr(order, "symbol", "").upper()
            client_order_id: str = getattr(order, "client_order_id", "") or ""

            if symbol not in self._contexts:
                return  # Not a symbol we're managing

            ctx = self._contexts[symbol]

            logger.debug(
                "[%s] Trade update | event=%s | client_id=%s | state=%s",
                symbol,
                event_type,
                client_order_id,
                ctx.state.name,
            )

            # ----------------------------------------------------------------
            # Fill → IN_TRADE
            # ----------------------------------------------------------------
            if event_type in ("fill", "partial_fill"):
                async with ctx.lock:
                    if ctx.state == SymbolState.PENDING:
                        ctx.state = SymbolState.IN_TRADE
                        ctx.entry_price = float(
                            getattr(order, "filled_avg_price", 0.0) or 0.0
                        )
                        ctx.entry_qty = float(getattr(order, "filled_qty", 0.0) or 0.0)
                        logger.info(
                            "[%s] State → IN_TRADE | filled_avg_price=%.4f | qty=%.4f",
                            symbol,
                            ctx.entry_price,
                            ctx.entry_qty,
                        )

            # ----------------------------------------------------------------
            # Cancel / Expire / Reject → FLAT
            # ----------------------------------------------------------------
            elif event_type in ("canceled", "expired", "rejected"):
                async with ctx.lock:
                    if ctx.state in (SymbolState.PENDING, SymbolState.IN_TRADE):
                        ctx.state = SymbolState.FLAT
                        ctx.entry_price = None
                        ctx.entry_qty = None
                        logger.info("[%s] State → FLAT | reason=%s", symbol, event_type)

            # ----------------------------------------------------------------
            # Bracket TP/SL hit — bracket child order fills mean the position
            # closed.  Alpaca sends a 'fill' event on the *child* order with
            # the parent bracket's client_order_id carrying the symbol context.
            # We detect this by checking state == IN_TRADE on a fill event for
            # an order whose side is SELL.
            # ----------------------------------------------------------------
            elif event_type == "fill":
                order_side: str = str(getattr(order, "side", "")).upper()
                async with ctx.lock:
                    if ctx.state == SymbolState.IN_TRADE and order_side == "SELL":
                        await self._enter_cooling(ctx)

        except Exception as exc:
            logger.error("Error in _on_trade_update: %s", exc, exc_info=True)

    # -----------------------------------------------------------------------
    # Cooling-off logic
    # -----------------------------------------------------------------------

    async def _enter_cooling(self, ctx: SymbolContext) -> None:
        """
        Transition symbol to COOLING state and schedule reset to FLAT
        after COOLING_SECONDS.  Cancels any prior cooling timer.

        Must be called with ctx.lock held by the caller.
        """
        symbol = ctx.symbol

        # Cancel any existing cooling task (shouldn't happen, but safe)
        if ctx._cooling_task and not ctx._cooling_task.done():
            ctx._cooling_task.cancel()

        ctx.state = SymbolState.COOLING
        ctx.entry_price = None
        ctx.entry_qty = None

        logger.info("[%s] State → COOLING | reset in %ds.", symbol, COOLING_SECONDS)
        self._notifier.send_system_message(
            f"[{symbol}] Bracket resolved — cooling off for "
            f"{COOLING_SECONDS // 60}m before next entry."
        )

        # Schedule the FLAT reset
        ctx._cooling_task = asyncio.ensure_future(self._reset_after_cooling(ctx))

    async def _reset_after_cooling(self, ctx: SymbolContext) -> None:
        """Waits for the cooling period, then returns the symbol to FLAT."""
        try:
            await asyncio.sleep(COOLING_SECONDS)
            async with ctx.lock:
                if ctx.state == SymbolState.COOLING:
                    ctx.state = SymbolState.FLAT
                    logger.info("[%s] Cooling complete — State → FLAT.", ctx.symbol)
        except asyncio.CancelledError:
            logger.debug("[%s] Cooling timer cancelled.", ctx.symbol)

    # -----------------------------------------------------------------------
    # REST API warm-up — pre-fills aggregator rolling windows on boot
    # -----------------------------------------------------------------------

    async def _warmup_aggregator(self) -> None:
        """
        Pre-loads each symbol's LiveBarAggregator with the last 60 one-minute
        bars from the Alpaca REST API so that the SMA-50 (and all other
        indicators) are ready the moment the first live WebSocket bar arrives.

        On weekends / outside market hours Alpaca returns the most recent
        available data (typically Friday's session), which is exactly what we
        want for warming the rolling window.

        Design notes
        ------------
        * Uses StockHistoricalDataClient (separate credential path from
          TradingClient; paper flag does not apply to market data).
        * Requests limit=60 bars per symbol to guarantee SMA-50 coverage plus
          a 10-bar safety margin.
        * The pandas MultiIndex DataFrame returned by the SDK is converted to a
          strict Polars DataFrame matching _SCHEMA before injection, ensuring
          zero type ambiguity downstream.
        * Bars are fed one-by-one through add_bar() so the aggregator's own
          window-sealing and gap-filling logic runs exactly as it would on live
          data.  Each new bar seals the previous window, so after 60 injections
          the history_df will contain ~59 closed candles — sufficient for all
          TA-Lib indicators.
        """
        logger.info("Starting aggregator warm-up via REST API for %s …", self._symbols)

        # Initialise a historical data client (not paper-specific)
        hist_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        end_time = datetime.now(timezone.utc)

        request = StockBarsRequest(
            symbol_or_symbols=self._symbols,
            timeframe=TimeFrame.Minute,
            limit=60,
            end=end_time,
            adjustment=Adjustment.RAW,
        )

        try:
            bar_set = await asyncio.to_thread(hist_client.get_stock_bars, request)
        except Exception as exc:
            logger.warning(
                "Aggregator warm-up REST request failed — bot will start cold: %s",
                exc,
            )
            return

        # bar_set.df is a pandas DataFrame with a (symbol, timestamp) MultiIndex.
        # We iterate per symbol so each aggregator receives its own bars in order.
        try:
            raw_pd = bar_set.df
        except Exception as exc:
            logger.warning("Could not extract DataFrame from bar_set: %s", exc)
            return

        if raw_pd is None or raw_pd.empty:
            logger.warning("Warm-up returned no data — bot will start cold.")
            return

        for symbol in self._symbols:
            ctx = self._contexts[symbol]

            # Slice this symbol's rows out of the MultiIndex
            try:
                sym_pd = raw_pd.loc[symbol].copy()
            except KeyError:
                logger.warning("[%s] No warm-up data returned — skipping.", symbol)
                continue

            if sym_pd.empty:
                logger.warning("[%s] Empty warm-up slice — skipping.", symbol)
                continue

            # Reset index so 'timestamp' becomes a plain column
            sym_pd = sym_pd.reset_index()

            # ------------------------------------------------------------------
            # Strict Polars conversion — matches LiveBarAggregator._SCHEMA:
            #   timestamp  → Datetime(time_unit="us", time_zone="UTC")
            #   open/high/low/close/volume → Float64
            # ------------------------------------------------------------------
            try:
                pl_schema = {
                    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }

                # Select only the columns we need (drop vwap, trade_count, etc.)
                cols_needed = list(pl_schema.keys())
                sym_pd = sym_pd[cols_needed]

                pl_df = pl.from_pandas(sym_pd).cast(
                    {
                        "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
                        "open": pl.Float64,
                        "high": pl.Float64,
                        "low": pl.Float64,
                        "close": pl.Float64,
                        "volume": pl.Float64,
                    }
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Polars conversion failed during warm-up — skipping: %s",
                    symbol,
                    exc,
                )
                continue

            # ------------------------------------------------------------------
            # Inject bars one-by-one so the aggregator's window logic fires
            # correctly for each transition (each new bar seals the prior window)
            # ------------------------------------------------------------------
            bars_injected = 0
            for row in pl_df.iter_rows(named=True):
                bar_dict = {
                    "timestamp": row["timestamp"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
                ctx.aggregator.add_bar(bar_dict)
                bars_injected += 1

            history_len = len(ctx.aggregator.history_df)
            logger.info(
                "[+] Warmed up %s with %d historical bars → %d candles in history.",
                symbol,
                bars_injected,
                history_len,
            )

        logger.info("Aggregator warm-up complete.")

    # -----------------------------------------------------------------------
    # Market hours
    # -----------------------------------------------------------------------

    async def _refresh_market_clock(self) -> None:
        """
        Updates self._market_open by querying the Alpaca market clock.
        The REST call is offloaded to avoid blocking the event loop.
        Result is cached for CLOCK_CACHE_TTL seconds.
        """
        now = asyncio.get_event_loop().time()
        if now - self._clock_last_checked < CLOCK_CACHE_TTL:
            return  # Cache still valid

        try:
            clock = await asyncio.to_thread(self._trading_client.get_clock)
            self._market_open = bool(clock.is_open)
            self._clock_last_checked = now
            logger.info(
                "Market clock refreshed | is_open=%s | next_open=%s | next_close=%s",
                self._market_open,
                getattr(clock, "next_open", "N/A"),
                getattr(clock, "next_close", "N/A"),
            )
        except Exception as exc:
            logger.warning("Failed to refresh market clock: %s", exc)
            # On failure, retain prior cached state — do not flip to closed

    # -----------------------------------------------------------------------
    # Graceful shutdown
    # -----------------------------------------------------------------------

    def _request_shutdown(self) -> None:
        """Signal-handler callback — sets the shutdown event."""
        logger.info("Shutdown signal received.")
        self._shutdown_event.set()

    async def _shutdown(self) -> None:
        """
        Graceful teardown:
        1. Cancel all cooling timers.
        2. Close all open bracket orders (cancel, not market-close).
        3. Stop WebSocket streams.
        4. Notify Discord.
        """
        logger.info("Initiating graceful shutdown...")

        # 1. Cancel cooling timers
        for ctx in self._contexts.values():
            if ctx._cooling_task and not ctx._cooling_task.done():
                ctx._cooling_task.cancel()

        # 2. Cancel open Alpaca orders
        try:
            cancel_result = await asyncio.to_thread(self._trading_client.cancel_orders)
            logger.info("Cancelled open orders: %s", cancel_result)
        except Exception as exc:
            logger.error("Failed to cancel open orders on shutdown: %s", exc)

        # 3. Close streams
        for stream, name in [
            (self._data_stream, "StockDataStream"),
            (self._trade_stream, "TradingStream"),
        ]:
            if stream is not None:
                try:
                    await stream.stop_ws()  # type: ignore[attr-defined]
                    logger.info("%s stopped.", name)
                except Exception as exc:
                    logger.warning("Error stopping %s: %s", name, exc)

        # 4. Final Discord notification
        self._notifier.send_system_message(
            "[LiveOrchestrator] Daemon shut down cleanly."
        )
        logger.info("Shutdown complete.")

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @staticmethod
    def _ensure_utc(ts: datetime) -> datetime:
        """Normalise any datetime to UTC-aware."""
        if ts is None:
            return datetime.now(tz=timezone.utc)
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Default paper-trading basket — matches OOS evaluation symbols.
# Importable by the root launcher (run_live.py) for env-var override.
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS: List[str] = ["TSLA", "NVDA", "MARA", "COIN", "SMCI"]


# ---------------------------------------------------------------------------
# Convenience async entry-point (called by top-level main.py or CLI)
# ---------------------------------------------------------------------------


async def main() -> None:
    """
    Standalone entry-point for the live orchestrator daemon.

    Reads credentials from environment / .env file.
    Override DEFAULT_SYMBOLS to change the live basket.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    orchestrator = LiveOrchestrator(
        symbols=DEFAULT_SYMBOLS,
        paper=True,
    )
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
