import asyncio
import logging
import signal
from datetime import datetime, timezone
from typing import List, Dict, Optional

import polars as pl
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from data.feed import MarketDataFeed
from execution.risk_manager import RiskManager
from strategies.concrete_strategies.ml_strategy import MLStrategy
from utils.bar_aggregator import LiveBarAggregator

logger = logging.getLogger(__name__)

class FactoryOrchestrator:
    """
    The Router: Async event loop wiring Feed, Strategy, and Risk Manager.
    Includes a Universal Watchdog for software SL/TP enforcement.
    """

    def __init__(
        self,
        symbols: List[str],
        api_key: str,
        secret_key: str,
        strategy: MLStrategy,
        risk_manager: RiskManager,
        feed: MarketDataFeed,
        paper: bool = True
    ):
        self.symbols = symbols
        self.feed = feed
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)

        self.aggregators = {
            s: LiveBarAggregator(timeframe=1, history_size=400) for s in symbols
        }
        self.active_positions = {} # symbol -> {sl, tp, qty}
        self._shutdown_event = asyncio.Event()

    async def run(self):
        """Main lifecycle entry point."""
        # 1. Graceful Shutdown signals
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, lambda: self._shutdown_event.set())

        # 2. THE WARM-UP PHASE
        # Solve data starvation: pull 300m history BEFORE websocket
        warmup_data = await self.feed.warmup_history(self.symbols, lookback_minutes=300)

        for symbol, df in warmup_data.items():
            agg = self.aggregators[symbol]
            logger.info(f"Injecting {len(df)} historical bars for {symbol}...")
            for row in df.iter_rows(named=True):
                agg.add_bar(row)
            logger.info(f"Aggregator for {symbol} primed. History size: {len(agg.history_df)}")

        # 3. Start Data Pipe
        feed_task = asyncio.create_task(self.feed.subscribe(self.symbols, self._on_tick))

        # 4. Universal Watchdog (1s poll)
        watchdog_task = asyncio.create_task(self._watchdog_loop())

        logger.info("FactoryOrchestrator active. Waiting for ticks...")

        await self._shutdown_event.wait()

        logger.info("Shutdown signaled. Cleaning up...")
        feed_task.cancel()
        watchdog_task.cancel()
        await self.feed.stop()

    async def _on_tick(self, tick: dict):
        """Routes a single bar event into the aggregator and strategy."""
        symbol = tick["symbol"]
        agg = self.aggregators[symbol]

        # logical clock alignment and aggregation
        if not agg.add_bar(tick):
            return # Wait for bar to seal

        # Bar sealed (1m candle complete). Snapshot for inference.
        history = agg.history_df.clone()

        # Offload CPU-bound ML inference to thread
        signal_result = await asyncio.to_thread(self.strategy.analyze, {symbol: history})

        # analyze returns (List[Signal], highest_prob)
        signals, _ = signal_result
        if not signals:
            return

        # Execute if FLAT
        if symbol not in self.active_positions:
            await self._execute_buy(signals[0])

    async def _execute_buy(self, sig):
        """Calculates risk, submits order, and enters watchdog state."""
        symbol = sig.symbol

        # Get account for sizing
        account = await asyncio.to_thread(self.trading_client.get_account)
        equity = float(account.equity)
        buying_power = float(account.buying_power)

        # Retrieve ATR for bracket calculation (assumes strategy includes it in metadata)
        # In current MLStrategy, it's calculated during feature extraction.
        # For brevity, let's assume metadata contains 'atr_abs'
        atr = sig.metadata.get("atr_abs", sig.price * 0.001)

        sl, tp = self.risk_manager.calculate_bracket(sig.price, atr)
        qty = self.risk_manager.calculate_quantity(equity, buying_power, sig.price, sl)

        if qty <= 0:
            return

        logger.info(f"Executing BUY for {symbol} | Qty: {qty} | SL: {sl} | TP: {tp}")

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )

        try:
            order = await asyncio.to_thread(self.trading_client.submit_order, req)
            self.active_positions[symbol] = {
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "order_id": order.id
            }
        except Exception as e:
            logger.error(f"Entry failed for {symbol}: {e}")

    async def _watchdog_loop(self):
        """Polls active symbols and enforces SL/TP targets."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(1)

            for symbol, pos in list(self.active_positions.items()):
                # Get last known price from aggregator
                last_bar = self.aggregators[symbol].history_df.tail(1)
                if last_bar.is_empty():
                    continue

                price = last_bar["close"][0]

                if price <= pos["sl"] or price >= pos["tp"]:
                    reason = "SL" if price <= pos["sl"] else "TP"
                    logger.warning(f"WATCHDOG: {reason} hit for {symbol} at {price}")
                    await self._execute_sell(symbol, pos["qty"])

    async def _execute_sell(self, symbol: str, qty: float):
        """Market close of a position."""
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        try:
            await asyncio.to_thread(self.trading_client.submit_order, req)
            del self.active_positions[symbol]
            logger.info(f"Position closed for {symbol}")
        except Exception as e:
            logger.error(f"Exit failed for {symbol}: {e}")
