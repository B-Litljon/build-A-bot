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
import functools
import logging
import os
import signal as sig
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional

import numpy as np
import polars as pl
import talib

from core.notification_manager import NotificationManager
from data.oanda_provider import OandaMarketProvider, _to_oanda_symbol
from execution.oanda_order_manager import OandaOrderManager, OrderCloseError
from execution.risk_manager import GATE_REGIME, GATE_SPREAD, RiskManager
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
        notifier: Optional[NotificationManager] = None,
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

        # Discord webhook (silently no-ops when DISCORD_WEBHOOK_URL unset).
        # Injectable so tests can pass a mock — the real one posts to the
        # production webhook whenever the env var is set.
        self._notifier = notifier if notifier is not None else NotificationManager()

        # Chop-filter telemetry: track how often each gate of the dynamic
        # hybrid floor vetoes a Devil-approved signal. Split per-gate so soak
        # logs reveal which gate binds (cost vs regime).
        self._devil_approved_total: int = 0
        self._spread_gate_rejections: int = 0
        self._regime_gate_rejections: int = 0
        self._a3_chop_rejections: int = 0  # combined (spread + regime)

        # ── Dynamic hybrid floor: per-symbol regime + spread state ──
        # Stateful, drift-free NATR: a running Wilder ATR seeded once at boot
        # from priming, then advanced O(1) per closed bar — no per-bar vector
        # recompute over a shifting window. The deque feeds the regime gate and
        # the volatility-scaled spread proxy in calculate_bracket().
        self._natr_period = 14  # matches v3_features._NATR_PERIOD
        try:
            self._regime_window = int(self._risk_manager.profile.regime_window)
        except (AttributeError, TypeError, ValueError):
            self._regime_window = 260  # no/mocked risk manager
        self._regime_natr: Dict[str, Deque[float]] = {
            _to_oanda_symbol(s): deque(maxlen=self._regime_window) for s in symbols
        }
        self._wilder_atr: Dict[str, Optional[float]] = {
            _to_oanda_symbol(s): None for s in symbols
        }
        self._regime_prev_close: Dict[str, Optional[float]] = {
            _to_oanda_symbol(s): None for s in symbols
        }
        # Live spread capture (written lock-free from the tick thread).
        self._latest_spread: Dict[str, float] = {}
        self._latest_spread_ts: Dict[str, float] = {}
        self._spread_stale_seconds = float(
            os.getenv("RISK_SPREAD_STALE_SECONDS", "5")
        )

        # ── Spread calibration sink (soak → empirical spread_atr_alpha) ──
        # The live cost gate uses the real bid-ask spread, but the *training*
        # / stale-fallback proxy is alpha·baseline_ATR with a placeholder
        # alpha=0.15. To calibrate it from reality we sample (spread_pct,
        # baseline_natr) once per CLOSED bar — off the <50µs tick path — and
        # periodically log the per-instrument empirical alpha =
        # median(spread_pct)/median(baseline_natr). spread_pct and the regime
        # NATR share units (pct of price), so the ratio is dimensionless and
        # directly comparable to RiskProfile.spread_atr_alpha. Bounded deques
        # keep memory flat over multi-day soaks (recency-weighted, which is
        # what we want).
        self._spread_calib_maxlen = int(os.getenv("SPREAD_CALIB_MAXLEN", "20000"))
        self._spread_calib_interval = int(
            os.getenv("SPREAD_CALIB_INTERVAL_BARS", "60")
        )
        self._spread_pct_samples: Dict[str, Deque[float]] = {
            _to_oanda_symbol(s): deque(maxlen=self._spread_calib_maxlen)
            for s in symbols
        }
        self._baseline_natr_samples: Dict[str, Deque[float]] = {
            _to_oanda_symbol(s): deque(maxlen=self._spread_calib_maxlen)
            for s in symbols
        }
        self._spread_calib_bars = 0

        # Watchdog close retry policy (C1 hardening)
        self._close_max_attempts = int(os.getenv("OANDA_CLOSE_MAX_ATTEMPTS", "5"))

        # Stream liveness policy (C3 hardening). The provider's read
        # timeout is the primary stall defense; this watchdog is the
        # backstop that also flattens exposure if the stream goes quiet.
        self._stream_stale_seconds = float(
            os.getenv("OANDA_STREAM_STALE_SECONDS", "60")
        )
        self._liveness_task: Optional[asyncio.Task] = None

    # ── notifications ─────────────────────────────────────────────────

    def _notify(self, fn, **kwargs) -> None:
        """
        Fire-and-forget a blocking notifier call off the event loop.

        Discord posts are synchronous ``requests.post`` calls with a 5s
        timeout; running them inline on the loop would stall bar
        processing and queued watchdog closes for every symbol.
        """
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.run_in_executor(None, functools.partial(fn, **kwargs))
        else:
            try:
                fn(**kwargs)
            except Exception as e:
                logger.error("Notification failed: %s", e)

    # ── tick callback (runs on provider's blocking stream thread) ─────

    def _on_tick(self, symbol: str, bid: float, ask: float) -> None:
        """
        Synchronous tick hook.

        Callee must return in <50 µs and perform NO blocking I/O.
        """
        # Capture the live spread for the cost gate. Plain dict assignment is
        # atomic under the GIL — no lock, no blocking, sub-µs.
        self._latest_spread[symbol] = ask - bid
        self._latest_spread_ts[symbol] = time.monotonic()

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

    # ── dynamic hybrid floor: regime + spread helpers ────────────────

    def _seed_regime(self, norm_sym: str, df: "pl.DataFrame") -> None:
        """
        Seed the per-symbol NATR deque + running Wilder ATR from primed bars.

        One talib.NATR call over the priming frame (boot only — not per bar);
        the running ATR is then advanced incrementally in ``_update_regime`` so
        the series matches a continuous stream (no per-bar reseed drift).
        """
        if df.height < self._natr_period + 1:
            return
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        natr = talib.NATR(high, low, close, timeperiod=self._natr_period)
        valid = natr[np.isfinite(natr)]
        if len(valid) == 0:
            return
        dq = self._regime_natr[norm_sym]
        dq.clear()
        for v in valid[-self._regime_window:]:
            dq.append(float(v))
        last_close = float(close[-1])
        # Reconstruct the Wilder ATR state from the last NATR (= 100·ATR/close).
        self._wilder_atr[norm_sym] = float(valid[-1]) * last_close / 100.0
        self._regime_prev_close[norm_sym] = last_close
        logger.info(
            "[%s] Seeded regime NATR deque (%d/%d) + Wilder ATR state",
            norm_sym, len(dq), self._regime_window,
        )

    def _update_regime(self, norm_sym: str, bar: dict) -> None:
        """Advance the Wilder ATR by one closed bar (O(1)) and append NATR."""
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        prev_close = self._regime_prev_close.get(norm_sym)
        if prev_close is None:
            prev_close = close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        prev_atr = self._wilder_atr.get(norm_sym)
        n = self._natr_period
        atr = tr if prev_atr is None else (prev_atr * (n - 1) + tr) / n
        self._wilder_atr[norm_sym] = atr
        self._regime_prev_close[norm_sym] = close
        if close > 0.0:
            self._regime_natr[norm_sym].append(100.0 * atr / close)

    def _get_spread(self, norm_sym: str) -> "tuple[Optional[float], bool]":
        """Return (latest_spread, is_fresh) for the cost gate."""
        ts = self._latest_spread_ts.get(norm_sym)
        sp = self._latest_spread.get(norm_sym)
        if ts is None or sp is None:
            return None, False
        fresh = (time.monotonic() - ts) <= self._spread_stale_seconds
        return sp, fresh

    def _sample_spread_calibration(self, norm_sym: str, close: float) -> None:
        """
        Record one (spread_pct, baseline_natr) sample for alpha calibration.

        Called once per CLOSED bar (not per tick), so it never touches the
        <50µs tick budget. Freshness is irrelevant here — the latest spread
        observed during the bar is a fine once-a-minute sample. Both quantities
        are in pct-of-price, so ``median(spread_pct)/median(baseline_natr)``
        gives the empirical, dimensionless ``spread_atr_alpha``.
        """
        sp, _fresh = self._get_spread(norm_sym)
        if sp is None or close <= 0.0:
            return
        regime = self._regime_natr.get(norm_sym)
        if not regime:
            return
        baseline_natr = float(np.median(regime))
        if baseline_natr <= 0.0:
            return
        self._spread_pct_samples[norm_sym].append(100.0 * sp / close)
        self._baseline_natr_samples[norm_sym].append(baseline_natr)

    def _log_spread_calibration(self) -> None:
        """
        Emit per-instrument empirical ``spread_atr_alpha`` from the soak so far.

        Logged periodically (every ``SPREAD_CALIB_INTERVAL_BARS`` bars) and once
        at shutdown. Grep ``SPREAD_CALIB`` in the soak log to read the converging
        per-instrument alpha; the median over a US-session window is the value
        to plug into ``RISK_SPREAD_ATR_ALPHA`` (or per-instrument overrides).
        """
        for norm_sym, spreads in self._spread_pct_samples.items():
            if len(spreads) < 30:  # too few for a stable median yet
                continue
            baselines = self._baseline_natr_samples[norm_sym]
            med_spread = float(np.median(spreads))
            med_base = float(np.median(baselines))
            alpha = med_spread / med_base if med_base > 0.0 else float("nan")
            p25, p75 = (float(x) for x in np.percentile(spreads, [25, 75]))
            logger.info(
                "SPREAD_CALIB %s | n=%d med_spread_pct=%.5f [p25=%.5f p75=%.5f] "
                "med_baseline_natr=%.5f alpha_emp=%.4f",
                norm_sym, len(spreads), med_spread, p25, p75, med_base, alpha,
            )

    async def _watchdog_close(self, symbol: str) -> None:
        """
        Coroutine running on the asyncio loop.

        Wraps the blocking ``close_position`` HTTP call in an executor so
        the event loop never stalls. A failed close is retried with
        exponential backoff; the position is only dropped from tracking
        once the broker confirms (or reports already-flat). If every
        attempt fails the position is parked in ``CLOSE_FAILED`` state —
        still visible to ``_flatten_all`` on exit — and a manual-
        intervention alert is sent.
        """
        # Snapshot the position before we close+pop so we can describe it
        # in the Discord alert.
        with self._positions_lock:
            pos_snapshot = self._positions.get(symbol, {}).copy()

        units = pos_snapshot.get("units", 0)
        direction = "long" if units > 0 else "short"
        loop = asyncio.get_running_loop()

        last_error: Optional[Exception] = None
        for attempt in range(1, self._close_max_attempts + 1):
            try:
                await loop.run_in_executor(
                    None, self._order_manager.close_position, symbol
                )
            except Exception as e:  # OrderCloseError or executor failure
                last_error = e
                logger.error(
                    "[%s] Watchdog close attempt %d/%d failed: %s",
                    symbol,
                    attempt,
                    self._close_max_attempts,
                    e,
                )
                if attempt < self._close_max_attempts:
                    await asyncio.sleep(2 ** (attempt - 1))
                continue

            # Success (close submitted, or broker already flat)
            with self._positions_lock:
                self._positions.pop(symbol, None)
            logger.info(
                "[%s] Watchdog close completed (attempt %d)", symbol, attempt
            )
            if pos_snapshot:
                self._notify(
                    self._notifier.send_oanda_trade_alert,
                    symbol=symbol,
                    direction=direction,
                    action="WATCHDOG_CLOSE",
                    price=pos_snapshot.get("entry", 0.0),
                    units=units,
                    reason="SL or TP breach detected by tick watchdog",
                )
            return

        # All attempts exhausted — keep the position tracked so the exit
        # flatten still sees it, and demand a human.
        with self._positions_lock:
            current = self._positions.get(symbol)
            if current is not None:
                current["state"] = "CLOSE_FAILED"
        logger.critical(
            "[%s] Watchdog close FAILED after %d attempts — position still "
            "open at broker with no automated exit. Last error: %s",
            symbol,
            self._close_max_attempts,
            last_error,
        )
        self._notify(
            self._notifier.send_oanda_trade_alert,
            symbol=symbol,
            direction=direction,
            action="CLOSE_FAILED",
            price=pos_snapshot.get("entry", 0.0),
            units=units,
            reason=(
                f"MANUAL INTERVENTION REQUIRED: watchdog close failed "
                f"{self._close_max_attempts} times ({last_error}). Position "
                f"remains open at broker without SL/TP enforcement."
            ),
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

        # ── advance the stateful regime NATR for this closed bar (O(1)) ──
        self._update_regime(symbol, bar)

        # ── sample spread for empirical alpha calibration (off tick path) ──
        self._sample_spread_calibration(symbol, bar["close"])
        self._spread_calib_bars += 1
        if self._spread_calib_bars % self._spread_calib_interval == 0:
            self._log_spread_calibration()

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

        # Devil approved this signal. Track it for A3 chop-filter telemetry.
        self._devil_approved_total += 1

        # ── guard: do not trade unless any existing position is cleanly OPEN ──
        # (PENDING_CLOSE = watchdog exit in flight; CLOSE_FAILED = stuck
        # position awaiting manual intervention — never trade on top of it.)
        with self._positions_lock:
            existing = self._positions.get(symbol)
            if existing and existing.get("state") != "OPEN":
                logger.info(
                    "[%s] Signal generated but position state=%s — skipping",
                    symbol,
                    existing.get("state"),
                )
                return

        # ── calculate SL/TP bracket ──
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None
        if self._risk_manager is not None:
            spread, spread_fresh = self._get_spread(symbol)
            bracket = self._risk_manager.calculate_bracket(
                signal.entry_price,
                signal.raw_sl_distance,
                symbol=symbol,
                spread=spread,
                spread_fresh=spread_fresh,
                regime_series=self._regime_natr.get(symbol),
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
            gate = getattr(self._risk_manager, "last_veto_gate", GATE_SPREAD)
            if gate == GATE_REGIME:
                self._regime_gate_rejections += 1
            else:  # GATE_SPREAD or GATE_STATIC (cost-side floors)
                self._spread_gate_rejections += 1
            self._a3_chop_rejections += 1
            ratio = 100.0 * self._a3_chop_rejections / max(self._devil_approved_total, 1)
            logger.warning(
                "[%s] Bracket rejected (%s gate) | spread=%d regime=%d "
                "(%d / %d Devil-approved vetoed, %.1f%%)",
                symbol,
                gate,
                self._spread_gate_rejections,
                self._regime_gate_rejections,
                self._a3_chop_rejections,
                self._devil_approved_total,
                ratio,
            )
            return

        # ── derive signed target units ──
        target_units = (
            self._units_per_trade
            if signal.direction == "long"
            else -self._units_per_trade
        )

        # ── guard: same-direction re-entry / mark reversal in flight ──
        # Re-checked under the lock immediately before submit: the position
        # may have gone PENDING_CLOSE since the earlier guard (tick watchdog
        # races bar processing). On a flip, mark the old position REVERSING
        # so the tick watchdog cannot fire on its stale SL/TP mid-submit.
        reversing = False
        with self._positions_lock:
            existing = self._positions.get(symbol)
            if existing:
                if existing.get("state") != "OPEN":
                    logger.info(
                        "[%s] Position state changed to %s before submit — "
                        "skipping entry",
                        symbol,
                        existing.get("state"),
                    )
                    return
                if existing["units"] > 0 and target_units > 0:
                    logger.info("[%s] Already long — skipping re-entry", symbol)
                    return
                if existing["units"] < 0 and target_units < 0:
                    logger.info("[%s] Already short — skipping re-entry", symbol)
                    return
                existing["state"] = "REVERSING"
                reversing = True

        def _restore_open() -> None:
            """Re-arm the watchdog on the old position after a failed flip."""
            if not reversing:
                return
            with self._positions_lock:
                current = self._positions.get(symbol)
                if current is not None and current.get("state") == "REVERSING":
                    current["state"] = "OPEN"

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
            _restore_open()
            return

        filled = result.get("filled", 0)
        if filled == 0:
            logger.warning(
                "[%s] Order rejected / zero fill — not recording position",
                symbol,
            )
            _restore_open()
            return

        # ── record position state ──
        # Use the order manager's authoritative resulting net position, not
        # the raw fill size: on a reversal the fill includes the closing leg
        # (2× the position) and avg_price blends both legs.
        avg_price = result.get("position_avg_price") or signal.entry_price
        actual_units = result.get("position_units", 0)
        if actual_units == 0:
            logger.warning(
                "[%s] Fill reported but resulting net position is flat — "
                "not recording position",
                symbol,
            )
            # Broker is flat: drop any old record (a REVERSING leftover
            # would block future entries and watchdog alike).
            with self._positions_lock:
                self._positions.pop(symbol, None)
            return
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

        # Discord trade alert (no-op if webhook unset; posted off-loop)
        meta = signal.metadata or {}
        self._notify(
            self._notifier.send_oanda_trade_alert,
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

    async def _reconcile_on_boot(self) -> None:
        """
        Reconcile local state with the broker before trading starts.

        Policy (2026-06-09 ruling): any position found at OANDA on boot is
        an orphan (crash recovery, failed watchdog close from a prior run)
        with no known SL/TP — flatten it. If broker state cannot even be
        *verified*, abort startup: never trade blind.
        """
        loop = asyncio.get_running_loop()
        for symbol in self._symbols:
            norm_sym = _to_oanda_symbol(symbol)

            ok = await loop.run_in_executor(
                None, self._order_manager.sync_position, norm_sym
            )
            if not ok:
                raise RuntimeError(
                    f"[{norm_sym}] Boot reconciliation failed: could not "
                    "verify broker position state — refusing to start."
                )

            net = self._order_manager.get_net_position(norm_sym)
            if net == 0:
                continue

            entry = self._order_manager.get_average_entry_price(norm_sym)
            logger.warning(
                "[%s] Boot reconciliation: orphaned position at broker "
                "(net=%d, avg=%.5f) — flattening",
                norm_sym,
                net,
                entry,
            )
            try:
                await loop.run_in_executor(
                    None, self._order_manager.close_position, norm_sym
                )
            except OrderCloseError as e:
                raise RuntimeError(
                    f"[{norm_sym}] Boot reconciliation: failed to flatten "
                    f"orphaned position (net={net}): {e}"
                ) from e

            self._notify(
                self._notifier.send_oanda_trade_alert,
                symbol=norm_sym,
                direction="long" if net > 0 else "short",
                action="BOOT_FLATTEN",
                price=entry,
                units=net,
                reason=(
                    "Orphaned position found at broker during startup "
                    "reconciliation — flattened (no known SL/TP)."
                ),
            )

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

            # Keep enough tail to both warm the strategy and fully seed the
            # regime NATR deque (window + Wilder warmup), in BAR count.
            keep = max(self._warmup, self._regime_window) + self._natr_period + 5
            df = df.tail(keep)

            hist_bars = []
            for row in df.iter_rows(named=True):
                if row["timestamp"].tzinfo is None:
                    raise ValueError(
                        f"[{norm_sym}] Historical bar at {row['timestamp']} "
                        "is timezone-naive — seam dedup would misbehave"
                    )
                hist_bars.append({**row, "symbol": norm_sym})

            self._bar_buffers[norm_sym].extend(hist_bars)
            self._seed_regime(norm_sym, df)
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
            self._provider.reset_stop()

            logger.info("Stream reconnect: priming complete, resuming stream")

    async def _check_stream_liveness(self) -> None:
        """
        One liveness probe: if the stream has delivered nothing (not even
        heartbeats) for longer than the stale threshold, flatten exposure
        and force a reconnect.

        REST and the pricing stream are separate connections, so the
        flatten very likely still works even when the stream is wedged.
        """
        age = self._provider.seconds_since_last_message
        if age is None or age <= self._stream_stale_seconds:
            return

        logger.critical(
            "Pricing stream stale: no message for %.0fs (threshold %.0fs) — "
            "flattening exposure and forcing reconnect",
            age,
            self._stream_stale_seconds,
        )
        self._notify(
            self._notifier.send_system_message,
            message=(
                f"🚨 Pricing stream stale for {age:.0f}s — flattening open "
                "positions and forcing a reconnect (SL/TP enforcement is "
                "software-only and was blind during the stall)."
            ),
        )

        with self._positions_lock:
            has_positions = bool(self._positions)
        if has_positions:
            await self._flatten_all()

        self._provider.force_disconnect("liveness watchdog: stream stale")

    async def _liveness_watchdog(self) -> None:
        """Periodic stream-liveness checks until shutdown."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=10)
                return  # shutdown signalled
            except asyncio.TimeoutError:
                pass
            try:
                await self._check_stream_liveness()
            except Exception as e:
                logger.error("Liveness check failed: %s", e, exc_info=True)

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

        # Verify broker state before anything else — flattens orphans,
        # raises if state can't be verified.
        await self._reconcile_on_boot()

        self._provider.subscribe(
            self._symbols,
            self._on_bar,
            tick_callback=self._on_tick,
        )

        await self._prime_history()

        # Run the pricing stream with reconnect-on-disconnect wrapper
        self._stream_task = asyncio.create_task(self._stream_with_retry())

        # Backstop liveness watchdog (C3): flatten + reconnect on stall
        self._liveness_task = asyncio.create_task(self._liveness_watchdog())

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

        # Final spread-calibration dump (durable even if flatten hangs below).
        self._log_spread_calibration()

        if self._liveness_task and not self._liveness_task.done():
            self._liveness_task.cancel()
            try:
                await self._liveness_task
            except asyncio.CancelledError:
                pass

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

        failed: List[str] = []
        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                failed.append(sym)
                logger.critical(
                    "[%s] Flatten close failed on exit — position may "
                    "remain open at broker: %s",
                    sym,
                    result,
                )
            else:
                logger.info("[%s] Flattened on exit", sym)

        if failed:
            # Synchronous on purpose: we are shutting down and the loop may
            # not outlive a fire-and-forget executor job.
            try:
                self._notifier.send_system_message(
                    "🚨 MANUAL INTERVENTION REQUIRED: exit flatten failed "
                    f"for {', '.join(failed)} — verify positions at OANDA."
                )
            except Exception as e:
                logger.error("Failed to send flatten-failure alert: %s", e)

        with self._positions_lock:
            self._positions.clear()
