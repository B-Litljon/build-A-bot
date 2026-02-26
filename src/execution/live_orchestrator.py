"""
Universal Scalper V3.1 — Live Forward-Testing Orchestrator (Phase 6 — Dual-Stream)
====================================================================================

Production-ready dual-stream orchestrator that concurrently trades both
**equities** (via StockDataStream / IEX) and **crypto** (via CryptoDataStream)
within a single asyncio event loop.

Architecture: asyncio.to_thread concurrency + Rich Dashboard
--------------------------------------------------------------
  StockDataStream (async WebSocket, IEX feed)
  CryptoDataStream (async WebSocket)
       │ raw bar events per symbol (both route to the same handler)
       ▼
  LiveBarAggregator.add_bar(tick)          ← in-event-loop, microseconds
       │ returns True when a 1-min bar seals
       ▼
  [Smart Clock Gate]                       ← equities blocked outside RTH;
       │                                     crypto always passes
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
  TradingStream (async WebSocket)
       │ fill / cancel / closed events
       ▼
  _on_trade_update → SymbolState machine   ← PENDING→IN_TRADE→COOLING(5m)→FLAT

Dashboard Features:
  - Live Header with Bot Name, Environment, and Clock
  - Symbol Status Table with Asset class column (Equity/Crypto)
  - Activity Console (last 5 signals/events)
  - Progress bars for warm-up operations

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
import time
import traceback
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import deque

import warnings

import polars as pl
from dotenv import load_dotenv

# Silence sklearn feature-name warnings that fire when a DataFrame column
# order differs from training time — not actionable during live inference.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Rich imports — Professional CLI Dashboard
# ---------------------------------------------------------------------------
from rich.console import Console
from rich.logging import RichHandler
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.layout import Layout
from rich.text import Text
from rich.align import Align

# ---------------------------------------------------------------------------
# Alpaca imports — live streams + REST (dual-stream: stock + crypto)
# ---------------------------------------------------------------------------
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
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

# From drift_report.json — High-regime calibration_gap threshold that
# triggers the safety switch.  Do NOT change without re-running
# src/analysis/reinforcement_voter.py on fresh OOS data.
ATR_KILL_SWITCH_THRESHOLD: float = 0.5204  # natr_14 percentage units

# Angel/Devil classification thresholds — restored to standard after the
# crypto-only stress-test sprint.
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

# Dashboard update interval (seconds)
DASHBOARD_REFRESH_INTERVAL: float = 1.0

# Max activity log entries to display
MAX_ACTIVITY_LOG: int = 5


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

    def __init__(self, symbol: str, is_crypto: bool) -> None:
        self.symbol: str = symbol
        self.is_crypto: bool = is_crypto
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

        # Dashboard tracking
        self.last_price: Optional[float] = None
        self.last_atr: Optional[float] = None
        self.last_conviction: Optional[float] = None

    def __repr__(self) -> str:  # pragma: no cover
        asset = "CRYPTO" if self.is_crypto else "EQUITY"
        return f"SymbolContext({self.symbol!r}, {asset}, state={self.state.name})"


# ---------------------------------------------------------------------------
# Activity Log Entry
# ---------------------------------------------------------------------------


class ActivityEntry:
    """Represents a single activity/event for the dashboard."""

    def __init__(
        self, timestamp: datetime, symbol: str, message: str, level: str = "info"
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.message = message
        self.level = level  # info, success, warning, error


# ---------------------------------------------------------------------------
# Live Orchestrator
# ---------------------------------------------------------------------------


class LiveOrchestrator:
    """
    Async daemon that drives live paper-trading for the Universal Scalper.

    Supports dual-stream operation: equities via StockDataStream (IEX feed)
    and crypto via CryptoDataStream, both multiplexed into a single asyncio
    event loop with a unified Rich dashboard.

    Usage (from an async entry-point)::

        orchestrator = LiveOrchestrator(
            symbols=["TSLA", "NVDA", "BTC/USD", "ETH/USD"]
        )
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
        daemon_mode: bool = False,
    ) -> None:
        # Configure logging before anything else so the very first logger call
        # uses the correct handler (Rich vs plain StreamHandler).
        # logging.basicConfig is a no-op if handlers are already attached, so
        # calling this here is safe even when an entry-point also calls it.
        self._daemon_mode: bool = daemon_mode
        setup_logging(daemon_mode=daemon_mode)

        load_dotenv()

        self._api_key: str = api_key or os.environ["ALPACA_API_KEY"]
        self._secret_key: str = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self._paper: bool = paper
        self._symbols: List[str] = [s.upper() for s in symbols]

        # -- Asset-class routing --
        # Crypto symbols contain '/' (e.g. "BTC/USD"); equities do not.
        self._crypto_symbols: List[str] = [s for s in self._symbols if "/" in s]
        self._stock_symbols: List[str] = [s for s in self._symbols if "/" not in s]
        self._crypto_set: Set[str] = set(self._crypto_symbols)
        self._stock_set: Set[str] = set(self._stock_symbols)

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
            sym: SymbolContext(sym, is_crypto=(sym in self._crypto_set))
            for sym in self._symbols
        }

        # -- Alpaca WebSocket clients (created fresh per run() call) --
        self._crypto_stream: Optional[CryptoDataStream] = None
        self._stock_stream: Optional[StockDataStream] = None
        self._trade_stream: Optional[TradingStream] = None

        # -- Market hours cache --
        self._market_open: bool = False
        self._clock_last_checked: float = 0.0

        # -- Shutdown coordination --
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # -- Dashboard state --
        # In daemon mode we never write to a Rich Console; create it only for
        # interactive runs so no ANSI/VT sequences leak into journald.
        self._console: Optional[Console] = None if daemon_mode else Console()
        self._activity_log: deque = deque(maxlen=MAX_ACTIVITY_LOG)
        self._start_time: datetime = datetime.now(timezone.utc)
        self._dashboard_running: bool = False

        logger.info(
            "LiveOrchestrator initialised | stocks=%s | crypto=%s | paper=%s | "
            "ATR_kill=%.4f | cooling=%ds",
            self._stock_symbols,
            self._crypto_symbols,
            paper,
            ATR_KILL_SWITCH_THRESHOLD,
            COOLING_SECONDS,
        )

    # -----------------------------------------------------------------------
    # Asset-class helpers
    # -----------------------------------------------------------------------

    def _is_crypto(self, symbol: str) -> bool:
        """Return True if symbol is a crypto pair (contains '/')."""
        return symbol in self._crypto_set

    # -----------------------------------------------------------------------
    # Dashboard Generation
    # -----------------------------------------------------------------------

    def _generate_dashboard(self) -> Layout:
        """Generate the Rich dashboard layout."""
        layout = Layout()

        header = self._create_header()
        symbol_table = self._create_symbol_table()
        activity_panel = self._create_activity_panel()

        layout.split_column(
            Layout(header, size=3),
            Layout(symbol_table),
            Layout(activity_panel, size=12),
        )

        return layout

    def _create_header(self) -> Panel:
        """Create the dashboard header with bot info and clock."""
        now = datetime.now(timezone.utc)
        runtime = now - self._start_time
        runtime_str = (
            f"{runtime.days}d "
            f"{runtime.seconds // 3600:02d}:"
            f"{(runtime.seconds % 3600) // 60:02d}:"
            f"{runtime.seconds % 60:02d}"
        )

        env_color = "green" if self._paper else "red"
        env_text = "PAPER" if self._paper else "LIVE"
        mkt_color = "green" if self._market_open else "red"
        mkt_text = "OPEN" if self._market_open else "CLOSED"

        header_text = Text()
        header_text.append("Universal Scalper v3.1", style="bold white")
        header_text.append("  |  ", style="dim")
        header_text.append(f"{env_text}", style=f"bold {env_color}")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Mkt: {mkt_text}", style=f"bold {mkt_color}")
        header_text.append("  |  ", style="dim")
        header_text.append(f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC", style="cyan")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Up: {runtime_str}", style="dim")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Equities: {len(self._stock_symbols)}", style="blue")
        header_text.append("  ", style="dim")
        header_text.append(f"Crypto: {len(self._crypto_symbols)}", style="yellow")

        return Panel(
            Align.center(header_text),
            border_style="blue",
            title="[bold blue]Mission Control[/bold blue]",
            title_align="center",
        )

    def _create_symbol_table(self) -> Panel:
        """Create the symbol status table."""
        table = Table(
            title="Symbol Status",
            header_style="bold magenta",
            border_style="blue",
            expand=True,
        )

        table.add_column("Symbol", justify="left", style="cyan", no_wrap=True)
        table.add_column("Asset", justify="center")
        table.add_column("Last Price", justify="right", style="green")
        table.add_column("ATR (natr_14)", justify="right", style="yellow")
        table.add_column("State", justify="center")
        table.add_column("Last Conviction", justify="right", style="magenta")

        for symbol in self._symbols:
            ctx = self._contexts[symbol]

            # Asset class badge
            if ctx.is_crypto:
                asset_str = "[yellow]CRYPTO[/yellow]"
            else:
                asset_str = "[blue]EQUITY[/blue]"

            # State styling
            state_colors = {
                SymbolState.FLAT: "dim",
                SymbolState.PENDING: "yellow",
                SymbolState.IN_TRADE: "green",
                SymbolState.COOLING: "red",
            }
            state_style = state_colors.get(ctx.state, "white")

            price_str = f"${ctx.last_price:,.2f}" if ctx.last_price else "---"
            atr_str = f"{ctx.last_atr:.4f}" if ctx.last_atr else "---"
            conviction_str = (
                f"{ctx.last_conviction:.4f}" if ctx.last_conviction else "---"
            )

            table.add_row(
                symbol,
                asset_str,
                price_str,
                atr_str,
                f"[{state_style}]{ctx.state.name}[/{state_style}]",
                conviction_str,
            )

        return Panel(table, border_style="blue")

    def _create_activity_panel(self) -> Panel:
        """Create the activity console panel."""
        if not self._activity_log:
            content = Text("Waiting for activity...", style="dim italic")
        else:
            lines = []
            for entry in self._activity_log:
                ts_str = entry.timestamp.strftime("%H:%M:%S")
                level_colors = {
                    "info": "blue",
                    "success": "green",
                    "warning": "yellow",
                    "error": "red",
                }
                color = level_colors.get(entry.level, "white")
                lines.append(
                    f"[{ts_str}] [{color}]{entry.symbol}[/{color}]: {entry.message}"
                )
            content = Text("\n".join(lines))

        return Panel(
            content,
            title="[bold yellow]Activity Console[/bold yellow]",
            title_align="left",
            border_style="yellow",
        )

    def _log_activity(self, symbol: str, message: str, level: str = "info"):
        """Add an entry to the activity log."""
        entry = ActivityEntry(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            message=message,
            level=level,
        )
        self._activity_log.append(entry)

    # -----------------------------------------------------------------------
    # Public entry-point
    # -----------------------------------------------------------------------

    async def run(self) -> None:
        """
        Start the orchestrator.  Blocks until SIGTERM/SIGINT is received or
        an unrecoverable error terminates the WebSocket streams.
        """
        loop = asyncio.get_running_loop()

        # Register OS signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._request_shutdown)

        self._notifier.send_system_message(
            f"[LiveOrchestrator] Starting dual-stream daemon | "
            f"stocks={self._stock_symbols} | crypto={self._crypto_symbols} | "
            f"ATR_kill_switch={ATR_KILL_SWITCH_THRESHOLD}"
        )

        # Fetch real market clock before subscribing
        await self._refresh_market_clock()

        # Pre-load historical bars so SMA-50 and all indicators are ready
        await self._warmup_aggregator()

        # Build WebSocket clients
        stream_tasks: List = []

        if self._crypto_symbols:
            self._crypto_stream = CryptoDataStream(self._api_key, self._secret_key)
            self._crypto_stream.subscribe_bars(self._on_bar, *self._crypto_symbols)
            stream_tasks.append(self._run_crypto_stream())

        if self._stock_symbols:
            self._stock_stream = StockDataStream(
                self._api_key, self._secret_key, feed=DataFeed.IEX
            )
            self._stock_stream.subscribe_bars(self._on_bar, *self._stock_symbols)
            stream_tasks.append(self._run_stock_stream())

        self._trade_stream = TradingStream(
            self._api_key, self._secret_key, paper=self._paper
        )
        self._trade_stream.subscribe_trade_updates(self._on_trade_update)
        stream_tasks.append(self._run_trade_stream())

        # Always run the clock refresh loop
        stream_tasks.append(self._clock_refresh_loop())

        logger.info(
            "WebSocket subscriptions registered | "
            "streams=%d (crypto=%s, stock=%s, trade=1) — starting.",
            len(stream_tasks),
            bool(self._crypto_symbols),
            bool(self._stock_symbols),
        )

        # Start dashboard
        self._dashboard_running = True
        dashboard_task = asyncio.create_task(self._dashboard_update_loop())

        try:
            await asyncio.gather(*stream_tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator tasks cancelled — shutting down cleanly.")
        except Exception as exc:
            logger.critical(
                "Unrecoverable error in orchestrator gather: %s",
                exc,
                exc_info=True,
            )
            self._notifier.send_system_message(
                f"[LiveOrchestrator] CRITICAL: unrecoverable error — {exc}"
            )
        finally:
            self._dashboard_running = False
            dashboard_task.cancel()
            try:
                await dashboard_task
            except asyncio.CancelledError:
                pass
            await self._shutdown()

    async def _dashboard_update_loop(self) -> None:
        """
        Drive the live display or, in daemon mode, simply idle until shutdown.

        Interactive mode:
            Renders the Rich Live dashboard, refreshing once per second.

        Daemon mode:
            Skips every Rich UI call — no ANSI sequences, no alternate screen
            buffer.  Periodically logs a heartbeat at DEBUG level so operators
            can confirm the loop is alive without flooding journald.
        """
        if self._daemon_mode:
            # Headless: hold the coroutine alive until shutdown is signalled,
            # emitting an occasional heartbeat so the task is never invisible.
            heartbeat_interval: float = 60.0  # once per minute at DEBUG level
            elapsed: float = 0.0
            while self._dashboard_running and not self._shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=DASHBOARD_REFRESH_INTERVAL,
                    )
                except asyncio.TimeoutError:
                    elapsed += DASHBOARD_REFRESH_INTERVAL
                    if elapsed >= heartbeat_interval:
                        logger.debug(
                            "Daemon heartbeat | uptime=%.0fs | symbols=%d",
                            (
                                datetime.now(timezone.utc) - self._start_time
                            ).total_seconds(),
                            len(self._symbols),
                        )
                        elapsed = 0.0
            return

        # Interactive mode — Rich Live dashboard
        with Live(
            self._generate_dashboard(),
            console=self._console,
            refresh_per_second=1,
            screen=False,
        ) as live:
            while self._dashboard_running and not self._shutdown_event.is_set():
                live.update(self._generate_dashboard())
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=DASHBOARD_REFRESH_INTERVAL,
                    )
                except asyncio.TimeoutError:
                    pass

    # -----------------------------------------------------------------------
    # Stream runners (thin wrappers so gather() can cancel them)
    # -----------------------------------------------------------------------

    async def _run_crypto_stream(self) -> None:
        """Runs CryptoDataStream until shutdown is requested."""
        try:
            await self._crypto_stream._run_forever()  # type: ignore[union-attr]
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("CryptoDataStream error: %s", exc, exc_info=True)
            self._log_activity("SYSTEM", f"CryptoDataStream error: {exc}", "error")
            self._request_shutdown()

    async def _run_stock_stream(self) -> None:
        """Runs StockDataStream until shutdown is requested."""
        try:
            await self._stock_stream._run_forever()  # type: ignore[union-attr]
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("StockDataStream error: %s", exc, exc_info=True)
            self._log_activity("SYSTEM", f"StockDataStream error: {exc}", "error")
            self._request_shutdown()

    async def _run_trade_stream(self) -> None:
        """Runs TradingStream until shutdown is requested."""
        try:
            await self._trade_stream._run_forever()  # type: ignore[union-attr]
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("TradingStream error: %s", exc, exc_info=True)
            self._log_activity("SYSTEM", f"TradingStream error: {exc}", "error")
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
    # Primary bar handler (shared callback for both streams)
    # -----------------------------------------------------------------------

    async def _on_bar(self, bar) -> None:
        """
        Ingests a raw 1-minute bar from either StockDataStream or
        CryptoDataStream.

        This handler MUST be non-blocking.  All CPU-bound work is offloaded
        via asyncio.to_thread() so the event loop is never stalled.

        Smart Clock Gate:
          - Crypto symbols always proceed (24/7 market).
          - Stock symbols are gated by self._market_open — bars received
            outside regular trading hours are silently dropped.
        """
        symbol: str = bar.symbol.upper()

        if symbol not in self._contexts:
            logger.debug("Received bar for untracked symbol %s — ignoring.", symbol)
            return

        ctx = self._contexts[symbol]

        # -- Smart Clock Gate -----------------------------------------------
        # Crypto: always proceed (24/7).
        # Equities: require the equity market to be open.
        if not ctx.is_crypto and not self._market_open:
            logger.debug("[%s] Equity market closed — skipping bar.", symbol)
            return

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
        Execute Angel -> Devil two-stage inference on the latest bar.

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

            # Drop any rows containing Nulls, NaNs, or Infinities to prevent model crashes
            features_df = features_df.filter(
                pl.all_horizontal(pl.col(ml_feature_names).is_finite())
            )

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

            # TA-Lib NATR is a percentage; convert to absolute ATR
            # natr_14 = (ATR / close) * 100  ->  ATR = (natr_14 / 100) * close
            atr_abs: float = (natr_value / 100.0) * current_price

            # Update dashboard tracking
            ctx = self._contexts[symbol]
            ctx.last_price = current_price
            ctx.last_atr = natr_value

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

            # Update dashboard conviction
            ctx.last_conviction = devil_prob

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
            # Replace '/' in crypto symbols for a valid order ID
            safe_sym = symbol.replace("/", "-")
            client_order_id = f"{safe_sym}_{ts_iso}"

            if client_order_id == ctx.last_client_order_id:
                logger.warning(
                    "[%s] Duplicate client_order_id=%s — skipping.",
                    symbol,
                    client_order_id,
                )
                return

            # Transition: FLAT -> PENDING (inside the lock)
            ctx.state = SymbolState.PENDING
            ctx.last_client_order_id = client_order_id

        # Lock released — submit the bracket order off-thread
        logger.info("[%s] State -> PENDING | submitting bracket order.", symbol)
        self._log_activity(
            symbol,
            f"Signal detected (conviction: {sig.confidence:.4f})",
            "info",
        )

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
            logger.warning("[%s] Bracket submission failed — state -> FLAT.", symbol)
            self._log_activity(symbol, "Bracket submission failed", "error")

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
            qty = max(round(qty, 4), 0.0001)  # Alpaca fractional floor

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
            # Crypto uses GTC; equities use DAY.
            # ----------------------------------------------------------------
            is_crypto = self._is_crypto(symbol)
            tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=tif,
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
            self._log_activity(
                symbol,
                f"Bracket submitted | Qty: {qty:.4f} | SL: ${sl_price:.2f} "
                f"| TP: ${tp_price:.2f}",
                "success",
            )

            # Notify Discord
            self._notifier.send_trade_alert(sig, action="ENTRY")

            return True

        except Exception as exc:
            logger.error(
                "[%s] Bracket order submission failed: %s",
                symbol,
                exc,
                exc_info=True,
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
            # Fill -> IN_TRADE
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
                            "[%s] State -> IN_TRADE | filled=%.4f | qty=%.4f",
                            symbol,
                            ctx.entry_price,
                            ctx.entry_qty,
                        )
                        self._log_activity(
                            symbol,
                            f"Filled @ ${ctx.entry_price:.2f} | "
                            f"Qty: {ctx.entry_qty:.4f}",
                            "success",
                        )

            # ----------------------------------------------------------------
            # Cancel / Expire / Reject -> FLAT
            # ----------------------------------------------------------------
            elif event_type in ("canceled", "expired", "rejected"):
                async with ctx.lock:
                    if ctx.state in (
                        SymbolState.PENDING,
                        SymbolState.IN_TRADE,
                    ):
                        ctx.state = SymbolState.FLAT
                        ctx.entry_price = None
                        ctx.entry_qty = None
                        logger.info(
                            "[%s] State -> FLAT | reason=%s",
                            symbol,
                            event_type,
                        )
                        self._log_activity(symbol, f"Order {event_type}", "warning")

            # ----------------------------------------------------------------
            # Bracket TP/SL hit — child SELL fill while IN_TRADE -> COOLING
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

        if ctx._cooling_task and not ctx._cooling_task.done():
            ctx._cooling_task.cancel()

        ctx.state = SymbolState.COOLING
        ctx.entry_price = None
        ctx.entry_qty = None

        logger.info("[%s] State -> COOLING | reset in %ds.", symbol, COOLING_SECONDS)
        self._log_activity(
            symbol,
            f"Bracket resolved — cooling for {COOLING_SECONDS // 60}m",
            "warning",
        )
        self._notifier.send_system_message(
            f"[{symbol}] Bracket resolved — cooling off for "
            f"{COOLING_SECONDS // 60}m before next entry."
        )

        ctx._cooling_task = asyncio.ensure_future(self._reset_after_cooling(ctx))

    async def _reset_after_cooling(self, ctx: SymbolContext) -> None:
        """Waits for the cooling period, then returns the symbol to FLAT."""
        try:
            await asyncio.sleep(COOLING_SECONDS)
            async with ctx.lock:
                if ctx.state == SymbolState.COOLING:
                    ctx.state = SymbolState.FLAT
                    logger.info("[%s] Cooling complete — State -> FLAT.", ctx.symbol)
                    self._log_activity(
                        ctx.symbol,
                        "Cooling complete — ready for signals",
                        "info",
                    )
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

        Uses CryptoHistoricalDataClient for crypto symbols and
        StockHistoricalDataClient for equity symbols.  Both result sets are
        cast to the identical Polars schema before injection.

        Daemon mode:
            Bypasses the Rich Progress context entirely.  Progress is reported
            via logger.info() — one log line per symbol completion — so journald
            receives clean, parseable text without ANSI escape sequences.
        """
        logger.info(
            "Starting aggregator warm-up | stocks=%s | crypto=%s",
            self._stock_symbols,
            self._crypto_symbols,
        )

        end_time = datetime.now(timezone.utc)

        if self._daemon_mode:
            # ------------------------------------------------------------------
            # Headless path — no Rich Progress; raw logger.info() for status.
            # ------------------------------------------------------------------
            crypto_pd = None
            if self._crypto_symbols:
                logger.info(
                    "Warm-up: fetching crypto history (%d symbols)...",
                    len(self._crypto_symbols),
                )
                crypto_pd = await self._fetch_crypto_history(end_time, progress=None)

            stock_pd = None
            if self._stock_symbols:
                logger.info(
                    "Warm-up: fetching stock history (%d symbols)...",
                    len(self._stock_symbols),
                )
                stock_pd = await self._fetch_stock_history(end_time, progress=None)

            for idx, symbol in enumerate(self._symbols, start=1):
                ctx = self._contexts[symbol]
                raw_pd = crypto_pd if ctx.is_crypto else stock_pd

                if raw_pd is None or raw_pd.empty:
                    logger.warning("[%s] No warm-up data available — skipping.", symbol)
                    continue

                try:
                    sym_pd = raw_pd.loc[symbol].copy()
                except KeyError:
                    logger.warning("[%s] No warm-up data returned — skipping.", symbol)
                    continue

                if sym_pd.empty:
                    logger.warning("[%s] Empty warm-up slice — skipping.", symbol)
                    continue

                sym_pd = sym_pd.reset_index()

                try:
                    pl_cast_schema = {
                        "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
                        "open": pl.Float64,
                        "high": pl.Float64,
                        "low": pl.Float64,
                        "close": pl.Float64,
                        "volume": pl.Float64,
                    }
                    sym_pd = sym_pd[list(pl_cast_schema.keys())]
                    pl_df = pl.from_pandas(sym_pd).cast(pl_cast_schema)
                except Exception as exc:
                    logger.warning(
                        "[%s] Polars conversion failed — skipping: %s", symbol, exc
                    )
                    continue

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
                    "[+] Warmed %s (%d/%d) | %d bars injected -> %d candles ready.",
                    symbol,
                    idx,
                    len(self._symbols),
                    bars_injected,
                    history_len,
                )

            logger.info("Aggregator warm-up complete.")
            return

        # ----------------------------------------------------------------------
        # Interactive path — Rich Progress bar display
        # ----------------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=self._console,
        ) as progress:
            overall_task = progress.add_task(
                f"[cyan]Warming up {len(self._symbols)} symbols...",
                total=len(self._symbols),
            )

            # ==============================================================
            # Phase 1: Fetch crypto historical bars
            # ==============================================================
            crypto_pd = None
            if self._crypto_symbols:
                crypto_pd = await self._fetch_crypto_history(end_time, progress)

            # ==============================================================
            # Phase 2: Fetch stock historical bars
            # ==============================================================
            stock_pd = None
            if self._stock_symbols:
                stock_pd = await self._fetch_stock_history(end_time, progress)

            # ==============================================================
            # Phase 3: Inject bars into aggregators
            # ==============================================================
            for symbol in self._symbols:
                ctx = self._contexts[symbol]

                # Pick the right DataFrame source
                if ctx.is_crypto:
                    raw_pd = crypto_pd
                else:
                    raw_pd = stock_pd

                if raw_pd is None or raw_pd.empty:
                    logger.warning("[%s] No warm-up data available — skipping.", symbol)
                    progress.advance(overall_task)
                    continue

                # Slice this symbol's rows out of the MultiIndex
                try:
                    sym_pd = raw_pd.loc[symbol].copy()
                except KeyError:
                    logger.warning("[%s] No warm-up data returned — skipping.", symbol)
                    progress.advance(overall_task)
                    continue

                if sym_pd.empty:
                    logger.warning("[%s] Empty warm-up slice — skipping.", symbol)
                    progress.advance(overall_task)
                    continue

                # Reset index so 'timestamp' becomes a plain column
                sym_pd = sym_pd.reset_index()

                # Progress bar for this symbol's bars
                bar_task = progress.add_task(
                    f"[green]Loading {symbol}...",
                    total=len(sym_pd),
                )

                # --------------------------------------------------------------
                # Strict Polars conversion — matches LiveBarAggregator._SCHEMA
                # --------------------------------------------------------------
                try:
                    pl_cast_schema = {
                        "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
                        "open": pl.Float64,
                        "high": pl.Float64,
                        "low": pl.Float64,
                        "close": pl.Float64,
                        "volume": pl.Float64,
                    }

                    cols_needed = list(pl_cast_schema.keys())
                    sym_pd = sym_pd[cols_needed]

                    pl_df = pl.from_pandas(sym_pd).cast(pl_cast_schema)
                except Exception as exc:
                    logger.warning(
                        "[%s] Polars conversion failed — skipping: %s",
                        symbol,
                        exc,
                    )
                    progress.advance(overall_task)
                    continue

                # --------------------------------------------------------------
                # Inject bars one-by-one so the aggregator's window logic fires
                # correctly for each transition
                # --------------------------------------------------------------
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
                    progress.advance(bar_task)

                history_len = len(ctx.aggregator.history_df)
                logger.info(
                    "[+] Warmed %s | %d bars injected -> %d candles ready.",
                    symbol,
                    bars_injected,
                    history_len,
                )

                progress.remove_task(bar_task)
                progress.advance(overall_task)

        logger.info("Aggregator warm-up complete.")

    async def _fetch_crypto_history(
        self, end_time, progress: Optional[Progress] = None
    ) -> object:
        """Fetch historical 1-min bars for all crypto symbols."""
        hist_client = CryptoHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        request = CryptoBarsRequest(
            symbol_or_symbols=self._crypto_symbols,
            timeframe=TimeFrame.Minute,
            limit=60,
            end=end_time,
        )

        try:
            bar_set = await asyncio.to_thread(hist_client.get_crypto_bars, request)
            raw_pd = bar_set.df
            if raw_pd is not None and not raw_pd.empty:
                logger.info("Crypto warm-up fetched %d rows.", len(raw_pd))
                return raw_pd
            logger.warning("Crypto warm-up returned no data.")
            return None
        except Exception as exc:
            logger.warning("Crypto warm-up REST failed — crypto starts cold: %s", exc)
            return None

    async def _fetch_stock_history(
        self, end_time, progress: Optional[Progress] = None
    ) -> object:
        """Fetch historical 1-min bars for all stock symbols."""
        hist_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        request = StockBarsRequest(
            symbol_or_symbols=self._stock_symbols,
            timeframe=TimeFrame.Minute,
            limit=60,
            end=end_time,
            adjustment=Adjustment.SPLIT,
        )

        try:
            bar_set = await asyncio.to_thread(hist_client.get_stock_bars, request)
            raw_pd = bar_set.df
            if raw_pd is not None and not raw_pd.empty:
                logger.info("Stock warm-up fetched %d rows.", len(raw_pd))
                return raw_pd
            logger.warning("Stock warm-up returned no data.")
            return None
        except Exception as exc:
            logger.warning("Stock warm-up REST failed — equities start cold: %s", exc)
            return None

    # -----------------------------------------------------------------------
    # Market hours — real clock for equities, always-open for crypto
    # -----------------------------------------------------------------------

    async def _refresh_market_clock(self) -> None:
        """
        Query the Alpaca market clock via REST to determine whether the US
        equity market is currently open.

        Crypto symbols ignore this flag entirely (they are gated in _on_bar
        by checking ctx.is_crypto).  This method only controls
        self._market_open which governs equity bar acceptance.

        The call is rate-limited by CLOCK_CACHE_TTL (30s default) and is
        offloaded to a thread pool so it never blocks the event loop.
        """
        now_mono = time.monotonic()

        # Rate-limit: skip if we checked recently
        if (now_mono - self._clock_last_checked) < CLOCK_CACHE_TTL:
            return

        try:
            clock = await asyncio.to_thread(self._trading_client.get_clock)
            was_open = self._market_open
            self._market_open = bool(clock.is_open)
            self._clock_last_checked = now_mono

            if self._market_open != was_open:
                status = "OPEN" if self._market_open else "CLOSED"
                logger.info("Equity market status changed -> %s", status)
                self._log_activity(
                    "SYSTEM",
                    f"Equity market is now {status}",
                    "info" if self._market_open else "warning",
                )

        except Exception as exc:
            logger.warning(
                "Failed to refresh market clock: %s — retaining "
                "previous state (open=%s).",
                exc,
                self._market_open,
            )

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

        # 3. Close all streams
        streams = []
        if self._crypto_stream is not None:
            streams.append((self._crypto_stream, "CryptoDataStream"))
        if self._stock_stream is not None:
            streams.append((self._stock_stream, "StockDataStream"))
        if self._trade_stream is not None:
            streams.append((self._trade_stream, "TradingStream"))

        for stream, name in streams:
            try:
                await stream.stop_ws()  # type: ignore[union-attr]
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
# Default combined trading basket — equities + crypto
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS: List[str] = [
    "TSLA",
    "NVDA",
    "MARA",
    "COIN",
    "SMCI",
    "BTC/USD",
    "ETH/USD",
]


# ---------------------------------------------------------------------------
# Configure Rich logging
# ---------------------------------------------------------------------------


def setup_logging(daemon_mode: bool = False) -> None:
    """
    Configure logging for the orchestrator.

    daemon_mode=False (default / interactive):
        Uses RichHandler for a colourised, formatted terminal display.

    daemon_mode=True (systemd / headless):
        Uses a plain StreamHandler(sys.stdout) with a timestamped format
        that journald can parse cleanly.  Rich escape sequences and the
        alternate-screen buffer are never touched so journalctl -f is
        always human-readable.
    """
    if daemon_mode:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logging.basicConfig(
            level=logging.INFO,
            handlers=[handler],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convenience async entry-point (called by top-level main.py or CLI)
# ---------------------------------------------------------------------------


async def main() -> None:
    """
    Standalone entry-point for the live orchestrator daemon.

    Reads credentials from environment / .env file.
    Override DEFAULT_SYMBOLS to change the live basket.
    """
    setup_logging()

    orchestrator = LiveOrchestrator(
        symbols=DEFAULT_SYMBOLS,
        paper=True,
    )
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
