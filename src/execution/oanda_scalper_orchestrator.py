"""
OANDA Scalper Orchestrator — V5 Forex Pivot.

Lean async loop wiring OandaMarketProvider → MLStrategy → OandaOrderManager
with an embedded software SL/TP watchdog.

Design constraints:
- One asyncio loop owns: bar callback (ML path), tick callback dispatch
  (watchdog), graceful shutdown.
- Tick callback runs synchronously on the provider's blocking stream thread;
  it must return in <50 µs and do NO blocking I/O.
- Software SL/TP only — never pass native brackets to the broker.
"""

import asyncio
import logging
import os
import signal as sig
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import polars as pl

from core.notification_manager import NotificationManager
from data.oanda_provider import OandaMarketProvider, _to_oanda_symbol
from execution.oanda_order_manager import OandaOrderManager
from execution.risk_manager import RiskManager
from strategies.concrete_strategies.ml_strategy import MLStrategy

logger = logging.getLogger(__name__)


class OandaScalperOrchestrator:
    """
    Async scalper orchestrator for OANDA v20 forex.

    Parameters
    ----------
    symbols : list[str]
        Instruments to trade (e.g. ``["EUR/USD", "GBP/USD"]``).
    provider : OandaMarketProvider
        Streaming market-data adapter.
    strategy : MLStrategy
        Angel/Devil meta-labeling strategy.
    order_manager : OandaOrderManager
        Net-position state manager + order submission.
    risk_manager : RiskManager, optional
        Broker-agnostic bracket calculator (``calculate_bracket``).
    units_per_trade : int
        Absolute unit size per signal (default 1 000).
    warmup_period : int, optional
        Overrides strategy warmup; defaults to ``strategy.warmup_period``.
    flatten_on_exit : bool
        If True (default), close all positions on SIGINT/SIGTERM.
    """

    def __init__(
        self,
        symbols: List[str],
        provider: OandaMarketProvider,
        strategy: MLStrategy,
        order_manager: OandaOrderManager,
        risk_manager: Optional[RiskManager] = None,
        units_per_trade: int = 1000,
        warmup_period: Optional[int] = None,
        flatten_on_exit: bool = True,
    ):
        self._symbols = symbols
        self._provider = provider
        self._strategy = strategy
        self._order_manager = order_manager
        self._risk_manager = risk_manager
        self._units_per_trade = units_per_trade
        self._flatten_on_exit = flatten_on_exit

        self._warmup = warmup_period or strategy.warmup_period
        self._max_bars = self._warmup * 2

        # Rolling bar buffers: normalized symbol -> list of bar dicts
        self._bar_buffers: Dict[str, List[dict]] = {
            _to_oanda_symbol(s): [] for s in symbols
        }

        # History seam state: normalized symbol -> last historical timestamp
        self._last_hist_ts: Dict[str, Optional[datetime]] = {
            _to_oanda_symbol(s): None for s in symbols
        }

        # History seam state: normalized symbol -> seam crossed flag
        self._seam_crossed: Dict[str, bool] = {
            _to_oanda_symbol(s): False for s in symbols
        }

        # Position state: symbol -> {entry, sl, tp, units, state}
        self._positions: Dict[str, dict] = {}
        self._positions_lock = threading.Lock()

        # asyncio loop reference (set in run())
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = asyncio.Event()
        self._stream_task: Optional[asyncio.Task] = None

        # Discord webhook (silently no-ops when DISCORD_WEBHOOK_URL unset)
        self._notifier = NotificationManager()

    # ── tick callback (runs on provider's blocking stream thread) ─────

    def _on_tick(self, symbol: str, bid: float, ask: float) -> None:
        """
        Synchronous tick hook.

        Callee must return in <50 µs and perform NO blocking I/O.
        """
        with self._positions_lock:
            pos = self._positions.get(symbol)

        if not pos or pos.get("state") != "OPEN":
            return

        sl = pos["sl"]
        tp = pos["tp"]
        units = pos["units"]

        breached = False
        if units > 0:  # long
            if bid <= sl or bid >= tp:
                breached = True
        elif units < 0:  # short
            if ask >= sl or ask <= tp:
                breached = True

        if not breached:
            return

        # Idempotent guard — set state to PENDING_CLOSE under lock
        with self._positions_lock:
            current = self._positions.get(symbol)
            if not current or current.get("state") != "OPEN":
                return
            current["state"] = "PENDING_CLOSE"

        # Dispatch close OFF the stream thread onto the asyncio loop
        loop = self._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._watchdog_close(symbol), loop
            )
        else:
            logger.error(
                "[%s] Watchdog breach but event loop not running — close skipped",
                symbol,
            )

    async def _watchdog_close(self, symbol: str) -> None:
        """
        Coroutine running on the asyncio loop.

        Wraps the blocking ``close_position`` HTTP call in an executor so
        the event loop never stalls.
        """
        # Snapshot the position before we close+pop so we can describe it
        # in the Discord alert.
        with self._positions_lock:
            pos_snapshot = self._positions.get(symbol, {}).copy()

        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self._order_manager.close_position, symbol
            )
            logger.info("[%s] Watchdog close completed", symbol)
        except Exception as e:
            logger.error(
                "[%s] Watchdog close failed: %s", symbol, e, exc_info=True
            )
        finally:
            with self._positions_lock:
                self._positions.pop(symbol, None)

        if pos_snapshot:
            units = pos_snapshot.get("units", 0)
            direction = "long" if units > 0 else "short"
            self._notifier.send_oanda_trade_alert(
                symbol=symbol,
                direction=direction,
                action="WATCHDOG_CLOSE",
                price=pos_snapshot.get("entry", 0.0),
                units=units,
                reason="SL or TP breach detected by tick watchdog",
            )

    # ── bar callback (runs on the asyncio loop) ───────────────────────

    async def _on_bar(self, bar: dict) -> None:
        """Process a completed bar: update buffer, generate signal, trade."""
        symbol = bar["symbol"]

        # ── history/stream seam: drop overlap and partial seam bar ──
        last_ts = self._last_hist_ts.get(symbol)
        if last_ts is not None and not self._seam_crossed.get(symbol, False):
            if bar["timestamp"] <= last_ts:
                return  # Drop overlap
            else:
                self._seam_crossed[symbol] = True
                logger.info(
                    "[%s] Dropping partial seam bar at %s; stream is now clean",
                    symbol,
                    bar["timestamp"],
                )
                return  # Drop the first bar > last_ts (the partial seam bar)

        # ── update rolling buffer ──
        buf = self._bar_buffers.get(symbol)
        if buf is None:
            self._bar_buffers[symbol] = [bar]
        else:
            buf.append(bar)
            if len(buf) > self._max_bars:
                buf.pop(0)

        n_bars = len(self._bar_buffers[symbol])
        if n_bars < self._warmup:
            logger.debug(
                "[%s] Warm-up (%d / %d bars) — skipping signal",
                symbol,
                n_bars,
                self._warmup,
            )
            return

        # ── generate signal ──
        df = pl.DataFrame(self._bar_buffers[symbol])
        try:
            signal = self._strategy.generate_signals(df)
        except Exception as e:
            logger.error(
                "[%s] generate_signals failed: %s", symbol, e, exc_info=True
            )
            return

        if signal is None:
            return

        # ── guard: do not trade while a close is pending ──
        with self._positions_lock:
            existing = self._positions.get(symbol)
            if existing and existing.get("state") == "PENDING_CLOSE":
                logger.info(
                    "[%s] Signal generated but position PENDING_CLOSE — skipping",
                    symbol,
                )
                return

        # ── calculate SL/TP bracket ──
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None
        if self._risk_manager is not None:
            bracket = self._risk_manager.calculate_bracket(
                signal.entry_price, signal.raw_sl_distance, symbol=symbol
            )
            if bracket:
                sl_dist, tp_dist = bracket
                if signal.direction == "long":
                    sl_price = signal.entry_price - sl_dist
                    tp_price = signal.entry_price + tp_dist
                else:  # short
                    sl_price = signal.entry_price + sl_dist
                    tp_price = signal.entry_price - tp_dist

        if sl_price is None or tp_price is None:
            logger.warning(
                "[%s] Bracket calculation rejected trade (A3 chop filter)",
                symbol,
            )
            return

        # ── derive signed target units ──
        target_units = (
            self._units_per_trade
            if signal.direction == "long"
            else -self._units_per_trade
        )

        # ── guard: skip same-direction re-entry ──
        with self._positions_lock:
            existing = self._positions.get(symbol)
        if existing:
            if existing["units"] > 0 and target_units > 0:
                logger.info("[%s] Already long — skipping re-entry", symbol)
                return
            if existing["units"] < 0 and target_units < 0:
                logger.info("[%s] Already short — skipping re-entry", symbol)
                return

        # ── submit order (blocking HTTP → executor) ──
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                self._order_manager.submit_target_position,
                symbol,
                target_units,
            )
        except Exception as e:
            logger.error(
                "[%s] submit_target_position failed: %s",
                symbol,
                e,
                exc_info=True,
            )
            return

        filled = result.get("filled", 0)
        if filled == 0:
            logger.warning(
                "[%s] Order rejected / zero fill — not recording position",
                symbol,
            )
            return

        # ── record position state ──
        avg_price = result.get("avg_price", signal.entry_price)
        actual_units = filled if target_units > 0 else -filled
        with self._positions_lock:
            self._positions[symbol] = {
                "entry": avg_price,
                "sl": sl_price,
                "tp": tp_price,
                "units": actual_units,
                "state": "OPEN",
            }

        logger.info(
            "[%s] Position opened | units=%d entry=%.5f sl=%.5f tp=%.5f",
            symbol,
            actual_units,
            avg_price,
            sl_price,
            tp_price,
        )

        # Discord trade alert (no-op if webhook unset)
        meta = signal.metadata or {}
        self._notifier.send_oanda_trade_alert(
            symbol=symbol,
            direction=signal.direction,
            action="ENTRY",
            price=avg_price,
            units=actual_units,
            sl_price=sl_price,
            tp_price=tp_price,
            angel_prob=meta.get("angel_prob"),
            devil_prob=meta.get("devil_prob"),
            timestamp=str(meta.get("timestamp")) if meta.get("timestamp") else None,
        )

    # ── lifecycle ─────────────────────────────────────────────────────

    async def _prime_history(self) -> None:
        """Prime bar buffers with historical REST data to bypass cold warm-up."""
        for symbol in self._symbols:
            norm_sym = _to_oanda_symbol(symbol)
            gran_min = getattr(self._provider, "_stream_gran", 1)
            start = datetime.now(timezone.utc) - timedelta(days=5)
            end = datetime.now(timezone.utc)

            try:
                df = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self._provider.get_historical_bars,
                    symbol,
                    gran_min,
                    start,
                    end,
                )
            except Exception as e:
                logger.warning(
                    "[%s] Historical bars fetch failed: %s", norm_sym, e
                )
                continue

            if df.is_empty():
                logger.warning("[%s] Historical bars returned empty", norm_sym)
                continue

            df = df.tail(self._warmup + 5)

            hist_bars = []
            for row in df.iter_rows(named=True):
                assert row["timestamp"].tzinfo is not None
                hist_bars.append({**row, "symbol": norm_sym})

            self._bar_buffers[norm_sym].extend(hist_bars)
            self._last_hist_ts[norm_sym] = df.select("timestamp").row(-1)[0]
            logger.info(
                "[%s] Primed %d historical bars (tail) up to %s",
                norm_sym,
                len(hist_bars),
                self._last_hist_ts[norm_sym],
            )

    async def _stream_with_retry(self) -> None:
        """Run the pricing stream with reconnect-on-disconnect."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.to_thread(self._provider.run_stream)
            except Exception as e:
                logger.error("OandaMarketProvider stream disconnected: %s", e)
            else:
                if not self._shutdown_event.is_set():
                    logger.warning(
                        "Stream returned without shutdown signal; treating as disconnect"
                    )

            if self._shutdown_event.is_set():
                break

            logger.warning("Stream reconnect in 5s; re-priming history on resume")
            await asyncio.sleep(5)

            # Reset seam state so re-prime + new stream dedup cleanly
            for sym in list(self._bar_buffers.keys()):
                self._bar_buffers[sym] = []
                self._last_hist_ts[sym] = None
                self._seam_crossed[sym] = False

            await self._prime_history()

            # Defensive: clear provider stop event in case it was set
            self._provider._stop_event.clear()

            logger.info("Stream reconnect: priming complete, resuming stream")

    async def run(self) -> None:
        """Start the orchestrator loop."""
        self._loop = asyncio.get_running_loop()

        # Graceful shutdown on SIGINT / SIGTERM
        for s in (sig.SIGINT, sig.SIGTERM):
            try:
                self._loop.add_signal_handler(
                    s, lambda: self._shutdown_event.set()
                )
            except (NotImplementedError, ValueError):
                pass  # Windows or handler already registered

        self._provider.subscribe(
            self._symbols,
            self._on_bar,
            tick_callback=self._on_tick,
        )

        await self._prime_history()

        # Run the pricing stream with reconnect-on-disconnect wrapper
        self._stream_task = asyncio.create_task(self._stream_with_retry())

        logger.info(
            "OandaScalperOrchestrator started | symbols=%s warmup=%d",
            self._symbols,
            self._warmup,
        )

        await self._shutdown_event.wait()
        await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown: stop stream, flatten if configured."""
        logger.info("OandaScalperOrchestrator shutting down...")
        self._provider.stop_stream()

        if self._stream_task and not self._stream_task.done():
            try:
                await asyncio.wait_for(self._stream_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass

        if self._flatten_on_exit:
            await self._flatten_all()

        logger.info("OandaScalperOrchestrator shutdown complete.")

    async def _flatten_all(self) -> None:
        """Close all open positions on exit."""
        with self._positions_lock:
            symbols = list(self._positions.keys())

        if not symbols:
            return

        logger.info("Flattening %d position(s) on exit", len(symbols))

        tasks = [
            asyncio.get_running_loop().run_in_executor(
                None, self._order_manager.close_position, sym
            )
            for sym in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(
                    "[%s] Flatten close failed: %s", sym, result, exc_info=True
                )
            else:
                logger.info("[%s] Flattened on exit", sym)

        with self._positions_lock:
            self._positions.clear()
