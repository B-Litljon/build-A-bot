"""FactoryOrchestrator — Central execution routing engine for the SDK.

Wires the Universal Data Pipe, Strategy Brain, and Risk Shield into a
cohesive, executable daemon. Backend-focused — no UI components.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

import polars as pl

# Phase 1 & Phase 2 SDK imports
from src.data.feed import MarketDataFeed
from src.strategies.base import BaseStrategy, Signal
from src.execution.risk_manager import RiskManager, AssetClass, RiskValidatedSignal
from src.utils.bar_aggregator import LiveBarAggregator


logger = logging.getLogger(__name__)


class SymbolState(Enum):
    """Lifecycle states for a single symbol's trading slot."""

    FLAT = auto()
    PENDING = auto()
    IN_TRADE = auto()
    PENDING_EXIT = auto()
    COOLING = auto()


@dataclass
class SymbolContext:
    """Mutable runtime state for a single tracked symbol."""

    symbol: str
    is_crypto: bool
    aggregator: LiveBarAggregator = field(
        default_factory=lambda: LiveBarAggregator(timeframe=1, history_size=400)
    )
    state: SymbolState = SymbolState.FLAT
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Execution tracking
    entry_price: Optional[float] = None
    entry_qty: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    last_client_order_id: Optional[str] = None

    # Latest market data
    last_price: Optional[float] = None

    # Cooling timer handle
    _cooling_task: Optional[asyncio.Task] = None


class FactoryOrchestrator:
    """
    Central routing engine that wires data feed, strategy, and risk management.

    Architecture:
        - Injected dependencies (feed, strategy, risk_manager)
        - Per-symbol SymbolContext state tracking
        - Async event loop with unified watchdog
        - No UI/dashboard — strictly backend execution
    """

    # Configuration
    MIN_HISTORY_BARS: int = 260
    COOLING_SECONDS: int = 300  # 5 minutes
    WATCHDOG_INTERVAL: float = 1.0  # seconds

    def __init__(
        self,
        feed: MarketDataFeed,
        strategy: BaseStrategy,
        risk_manager: RiskManager,
        symbols: List[str],
        paper: bool = True,
    ) -> None:
        """
        Initialize the orchestrator with injected dependencies.

        Args:
            feed: MarketDataFeed instance for WebSocket data
            strategy: BaseStrategy instance for signal generation
            risk_manager: RiskManager instance for execution safety
            symbols: List of symbols to trade (crypto contains '/', equities do not)
            paper: Whether to use paper trading
        """
        self._feed = feed
        self._strategy = strategy
        self._risk_manager = risk_manager
        self._symbols: List[str] = [s.upper() for s in symbols]
        self._paper: bool = paper

        # Asset classification
        self._crypto_symbols: Set[str] = {s for s in self._symbols if "/" in s}
        self._stock_symbols: Set[str] = {s for s in self._symbols if "/" not in s}

        # Per-symbol contexts
        self._contexts: Dict[str, SymbolContext] = {
            sym: SymbolContext(symbol=sym, is_crypto=(sym in self._crypto_symbols))
            for sym in self._symbols
        }

        # Shutdown coordination
        self._shutdown_event: asyncio.Event = asyncio.Event()

        logger.info(
            "FactoryOrchestrator initialized | symbols=%d | crypto=%d | equity=%d | paper=%s",
            len(self._symbols),
            len(self._crypto_symbols),
            len(self._stock_symbols),
            paper,
        )

    async def run(self) -> None:
        """
        Main entry point. Starts the feed stream and watchdog loop.
        Blocks until shutdown signal received.
        """
        loop = asyncio.get_running_loop()

        # Register signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._request_shutdown)

        logger.info("Starting FactoryOrchestrator event loop...")

        # Gather core tasks
        tasks: List[asyncio.Task] = [
            asyncio.create_task(self._feed.run(self._on_bar)),
            asyncio.create_task(self._universal_watchdog_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator tasks cancelled — shutting down.")
        except Exception as exc:
            logger.critical("Unrecoverable error: %s", exc, exc_info=True)
        finally:
            await self._shutdown()

    async def _on_bar(self, bar_dict: Dict[str, Any]) -> None:
        """
        Callback invoked by MarketDataFeed for each new bar.

        Args:
            bar_dict: Dictionary with keys: symbol, timestamp, open, high, low, close, volume
        """
        symbol: str = bar_dict.get("symbol", "").upper()

        if symbol not in self._contexts:
            logger.debug("Received bar for untracked symbol %s — ignoring.", symbol)
            return

        ctx = self._contexts[symbol]

        # Update latest price for watchdog
        ctx.last_price = float(bar_dict["close"])

        # Push to aggregator
        bar_sealed = ctx.aggregator.add_bar(
            {
                "timestamp": bar_dict["timestamp"],
                "open": float(bar_dict["open"]),
                "high": float(bar_dict["high"]),
                "low": float(bar_dict["low"]),
                "close": float(bar_dict["close"]),
                "volume": float(bar_dict["volume"]),
            }
        )

        if not bar_sealed:
            return  # Bar still accumulating

        # Complete bar sealed — check history depth
        history_df: pl.DataFrame = ctx.aggregator.history_df.clone()

        if len(history_df) < self.MIN_HISTORY_BARS:
            logger.debug(
                "[%s] Warming up (%d/%d bars)",
                symbol,
                len(history_df),
                self.MIN_HISTORY_BARS,
            )
            return

        # Generate signal via strategy
        try:
            signal: Optional[Signal] = await asyncio.to_thread(
                self._strategy.generate_signals, history_df
            )
        except Exception as exc:
            logger.error("[%s] Strategy error: %s", symbol, exc)
            return

        if signal is None:
            return

        # Execute the signal
        await self._execute_signal(ctx, signal)

    async def _execute_signal(self, ctx: SymbolContext, signal: Signal) -> None:
        """
        Validate signal through risk manager and submit order.

        Args:
            ctx: SymbolContext for the symbol
            signal: Signal from the strategy
        """
        symbol = ctx.symbol

        async with ctx.lock:
            if ctx.state != SymbolState.FLAT:
                logger.debug(
                    "[%s] Signal ignored — state=%s (not FLAT)",
                    symbol,
                    ctx.state.name,
                )
                return

            # Transition to PENDING
            ctx.state = SymbolState.PENDING

        # Validate through risk manager
        asset_class = AssetClass.CRYPTO if ctx.is_crypto else AssetClass.EQUITY

        validated: RiskValidatedSignal = self._risk_manager.validate_signal(
            direction=signal.direction,
            entry_price=signal.entry_price,
            raw_sl_distance=signal.raw_sl_distance,
            raw_tp_distance=signal.raw_tp_distance,
            asset_class=asset_class,
        )

        if not validated.is_valid:
            logger.warning(
                "[%s] Risk manager rejected signal: %s",
                symbol,
                validated.rejection_reason,
            )
            async with ctx.lock:
                ctx.state = SymbolState.FLAT
            return

        # Calculate absolute SL/TP prices
        if validated.direction == "long":
            sl_price = validated.entry_price - validated.actual_sl_distance
            tp_price = validated.entry_price + validated.actual_tp_distance
        else:  # short
            sl_price = validated.entry_price + validated.actual_sl_distance
            tp_price = validated.entry_price - validated.actual_tp_distance

        # Submit market order (offloaded to thread)
        success = await asyncio.to_thread(
            self._submit_market_order,
            symbol,
            validated,
            sl_price,
            tp_price,
        )

        if success:
            # Update context with active trade
            async with ctx.lock:
                ctx.state = SymbolState.IN_TRADE
                ctx.sl_price = sl_price
                ctx.tp_price = tp_price

            logger.info(
                "[%s] Trade active | SL=%.4f | TP=%.4f",
                symbol,
                sl_price,
                tp_price,
            )
        else:
            async with ctx.lock:
                ctx.state = SymbolState.FLAT
            logger.warning("[%s] Order submission failed — state reset to FLAT", symbol)

    def _submit_market_order(
        self,
        symbol: str,
        validated: RiskValidatedSignal,
        sl_price: float,
        tp_price: float,
    ) -> bool:
        """
        Submit fractional market order to Alpaca.

        Args:
            symbol: Trading symbol
            validated: Risk-validated signal with sizing and TIF
            sl_price: Stop-loss price level
            tp_price: Take-profit price level

        Returns:
            True on successful submission
        """
        try:
            # Import here to avoid dependency at module load
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, OrderType

            # Get trading client (assume credentials from env)
            import os

            trading_client = TradingClient(
                os.environ.get("ALPACA_API_KEY", ""),
                os.environ.get("ALPACA_SECRET_KEY", ""),
                paper=self._paper,
            )

            # Build order request
            order_request = MarketOrderRequest(
                symbol=symbol,
                notional=10.0,  # TODO: Dynamic sizing from risk manager
                side=OrderSide.BUY if validated.direction == "long" else OrderSide.SELL,
                time_in_force=validated.time_in_force.value,
                type=OrderType.MARKET,
            )

            order = trading_client.submit_order(order_request)
            order_id = str(getattr(order, "id", "unknown"))

            logger.info(
                "[%s] Market order submitted | id=%s | tif=%s",
                symbol,
                order_id,
                validated.time_in_force.value,
            )

            return True

        except Exception as exc:
            logger.error("[%s] Order submission failed: %s", symbol, exc)
            return False

    async def _universal_watchdog_loop(self) -> None:
        """
        Background coroutine polling IN_TRADE positions once per second.
        Triggers market sell when price breaches stored SL or TP levels.

        Ported from legacy live_orchestrator.py universal watchdog.
        """
        logger.info(
            "Universal watchdog started — polling every %.1fs", self.WATCHDOG_INTERVAL
        )

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.WATCHDOG_INTERVAL)
            except asyncio.CancelledError:
                break

            for ctx in self._contexts.values():
                # Only monitor active trades
                if ctx.state != SymbolState.IN_TRADE:
                    continue

                # Skip if missing required data
                if (
                    ctx.last_price is None
                    or ctx.tp_price is None
                    or ctx.sl_price is None
                ):
                    continue

                # Check breach conditions
                tp_hit = ctx.last_price >= ctx.tp_price
                sl_hit = ctx.last_price <= ctx.sl_price

                if not (tp_hit or sl_hit):
                    continue

                reason = "TP" if tp_hit else "SL"

                # Acquire lock and verify state hasn't changed
                async with ctx.lock:
                    if ctx.state != SymbolState.IN_TRADE:
                        continue  # Another coroutine already transitioned
                    ctx.state = SymbolState.PENDING_EXIT
                    exit_qty = ctx.entry_qty

                logger.info(
                    "[%s] Watchdog %s breach | price=%.4f | tp=%.4f | sl=%.4f",
                    ctx.symbol,
                    reason,
                    ctx.last_price,
                    ctx.tp_price,
                    ctx.sl_price,
                )

                # Fire-and-forget exit order
                asyncio.create_task(
                    asyncio.to_thread(self._submit_exit_order, ctx.symbol, exit_qty)
                )

        logger.info("Universal watchdog stopped.")

    def _submit_exit_order(self, symbol: str, qty: Optional[float]) -> bool:
        """
        Submit market SELL to close position.

        Args:
            symbol: Trading symbol
            qty: Quantity to sell (None for position close)

        Returns:
            True on successful submission
        """
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

            import os

            trading_client = TradingClient(
                os.environ.get("ALPACA_API_KEY", ""),
                os.environ.get("ALPACA_SECRET_KEY", ""),
                paper=self._paper,
            )

            if qty is None or qty <= 0:
                # Close entire position
                position = trading_client.get_open_position(symbol.replace("/", ""))
                qty = float(getattr(position, "qty", 0)) if position else 0

            if qty <= 0:
                logger.error("[%s] Cannot exit — invalid qty=%s", symbol, qty)
                return False

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                type=OrderType.MARKET,
            )

            order = trading_client.submit_order(order_request)
            order_id = str(getattr(order, "id", "unknown"))

            logger.info(
                "[%s] Exit order submitted | qty=%.4f | id=%s",
                symbol,
                qty,
                order_id,
            )

            # Schedule transition to COOLING after fill confirmation
            ctx = self._contexts.get(symbol)
            if ctx:
                asyncio.create_task(self._enter_cooling(ctx))

            return True

        except Exception as exc:
            logger.error("[%s] Exit order failed: %s", symbol, exc)
            return False

    async def _enter_cooling(self, ctx: SymbolContext) -> None:
        """
        Transition symbol to COOLING state, then reset to FLAT after cooldown.

        Args:
            ctx: SymbolContext to transition
        """
        async with ctx.lock:
            if ctx._cooling_task and not ctx._cooling_task.done():
                ctx._cooling_task.cancel()

            ctx.state = SymbolState.COOLING
            ctx.entry_price = None
            ctx.entry_qty = None
            ctx.sl_price = None
            ctx.tp_price = None

        logger.info(
            "[%s] State -> COOLING | reset in %ds", ctx.symbol, self.COOLING_SECONDS
        )

        ctx._cooling_task = asyncio.create_task(self._reset_after_cooling(ctx))

    async def _reset_after_cooling(self, ctx: SymbolContext) -> None:
        """Wait for cooling period then return symbol to FLAT state."""
        try:
            await asyncio.sleep(self.COOLING_SECONDS)
            async with ctx.lock:
                if ctx.state == SymbolState.COOLING:
                    ctx.state = SymbolState.FLAT
                    logger.info("[%s] Cooling complete — State -> FLAT", ctx.symbol)
        except asyncio.CancelledError:
            logger.debug("[%s] Cooling timer cancelled", ctx.symbol)

    def _request_shutdown(self) -> None:
        """Signal handler — sets shutdown event."""
        logger.info("Shutdown signal received")
        self._shutdown_event.set()

    async def _shutdown(self) -> None:
        """Graceful teardown."""
        logger.info("Initiating graceful shutdown...")

        # Cancel cooling timers
        for ctx in self._contexts.values():
            if ctx._cooling_task and not ctx._cooling_task.done():
                ctx._cooling_task.cancel()

        # Cancel open orders
        try:
            from alpaca.trading.client import TradingClient
            import os

            trading_client = TradingClient(
                os.environ.get("ALPACA_API_KEY", ""),
                os.environ.get("ALPACA_SECRET_KEY", ""),
                paper=self._paper,
            )
            await asyncio.to_thread(trading_client.cancel_orders)
            logger.info("Open orders cancelled")
        except Exception as exc:
            logger.error("Failed to cancel orders: %s", exc)

        logger.info("Shutdown complete")
