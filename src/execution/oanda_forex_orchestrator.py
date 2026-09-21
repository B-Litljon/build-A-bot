"""
OANDA Forex Orchestrator — V5 Forex Pivot.

Lean async loop wiring OandaMarketProvider → MLStrategy → OandaOrderManager
with an embedded software SL/TP watchdog.

Design constraints:
- One asyncio loop owns: bar callback (ML path), tick callback dispatch
  (watchdog), graceful shutdown.
- Tick callback runs synchronously on the provider's blocking stream thread;
  it must return in <50 µs and do NO blocking I/O.
- Software SL/TP only — never pass native brackets to the broker.

⚠️ THIS IS THE ORCHESTRATOR CURRENTLY RUNNING LIVE (the M15 practice-account
soak, relaunched automatically by soak_watchdog.sh). Treat edits accordingly.

Two clocks run here at once, and most of the design follows from that:
  * the SLOW path -- a bar seals, features are computed, the model is asked,
    and a trade may open. Runs every bar (e.g. every 15 minutes).
  * the FAST path -- every incoming quote is checked against the open
    position's stop and target. Runs thousands of times per bar, on the
    provider's stream thread, which is why it must stay microsecond-cheap.

See GLOSSARY.md (angel/devil, chop veto, NATR, spread, bracket, sealed bar,
heartbeat, watchdog).

Glossary:
    units_per_trade -- 1000, a FIXED position size. Note this path does not use
        RiskManager.calculate_quantity; only the bracket logic is shared.
    _warmup -- bars required before trading, taken from the strategy.
    _max_bars -- warmup * 2, the per-symbol buffer cap, so memory stays flat
        over a multi-day soak.
    _bar_buffers -- rolling recent bars per symbol, the input to inference.
    _positions -- symbol -> {entry, sl, tp, units, state, breach_price?,
        hit_level?}. Guarded by _positions_lock because the tick thread reads
        it while the loop writes. breach_price and hit_level are set by
        _on_tick at breach and consumed by _watchdog_close for BOTH the Discord
        alert and the "exit" event record -- the latter is what makes a fill a
        labelled outcome (which bracket hit) rather than just "a trade closed".
    _positions_lock -- a threading.Lock, not an asyncio one, precisely because
        a non-async thread touches this state.
    flatten_on_exit -- whether shutdown closes everything. Default True: a
        stopped bot must not leave unmonitored positions open, since stops are
        enforced in software by THIS process.

    ── history seam ──
    _last_hist_ts / _seam_crossed -- warm-up history and the live stream
        overlap in time. These track where the fetched history ended so the
        first live bars are not double-counted or scored twice. The "seam" is
        that junction between replayed and live data.
    _prime_history -- fetches the warm-up bars at boot.
    _drop_untradeable_symbols -- at boot, removes configured symbols the
        account may not trade (XAU_USD/XAG_USD here), so a metals signal is
        never spent on an order OANDA will reject. Fail-OPEN: only a
        successful instrument lookup may drop anything, and dropping
        everything aborts startup rather than running a bot that cannot
        place an order.
    _last_scored_ts -- per symbol, the newest bar timestamp that actually went
        through signal evaluation (streamed or catch-up). Deliberately NOT
        reset on reconnect: it is the dedup that stops repeated re-primes
        inside one bar period from scoring the same bar twice (= duplicate
        orders).
    _catch_up_missed_bars -- after every prime, scores the newest primed bar
        if it sealed while the stream was down and is still fresh. Before this
        existed, bars sealing during an outage were fetched into the buffer
        but never evaluated (~6 of 15 would-be signals died in these gaps
        during the 2026-07 soak). Emits _emit_status after scoring each
        symbol (finally: even a failed evaluation refreshes telemetry).
    _seam_catchup_max_age -- SEAM_CATCHUP_MAX_AGE_SECONDS; how stale a missed
        bar may be and still be scored. -1 (default) = one bar period,
        0 disables catch-up.
    _backfill_seam_bar -- the OTHER half of the reconnect gap: the bar in
        flight when the stream died is dropped (the stream's copy is
        incomplete), but it has sealed by then, so its complete version is
        re-fetched from REST and scored. Costs one evaluation per symbol per
        reconnect if absent -- measured at 5/symbol in 16h on 2026-07-28.
        Emits _emit_status after scoring so status.json stays fresh during
        stream outages (the watchdog would otherwise see stale telemetry and
        restart a recovering soak).
    _seam_backfill_attempts / _seam_backfill_retry_delay --
        SEAM_BACKFILL_ATTEMPTS (3, 0 disables) and SEAM_BACKFILL_RETRY_DELAY
        (2s). REST can lag the bar seal by a second or two.
    _reconnect_delay / _reconnect_base_delay / _reconnect_max_delay /
        _reconnect_healthy_seconds -- jittered capped exponential backoff
        (5s base, 60s cap, reset after 120s of healthy streaming) so a
        flapping endpoint is not hammered on a fixed cadence. The cap sits
        below the liveness watchdog's 60s flatten threshold.
    _evaluate_and_trade -- the decision tail (inference → guards → bracket →
        order) shared by the stream path and the catch-up path.
    _reconcile_on_boot -- asks the broker what is actually open before trading.
        A restart must adopt reality rather than assume it is flat -- otherwise
        a position left by a crashed process would run with nothing watching
        its stop. Unverifiable state still refuses to start, but only after
        _reconcile_max_attempts tries (OANDA_RECONCILE_MAX_ATTEMPTS / _RETRY_DELAY):
        one transient error used to cost 15 minutes of downtime via the cron
        watchdog's crash-loop brake.

    ── the fast path ──
    _on_tick -- runs on the provider's stream thread for EVERY quote. Records
        the spread and checks stop/target. Must return in under 50
        microseconds and do no blocking I/O; anything slower stalls the feed.
        On breach it stores breach_price (the live bid/ask that triggered)
        and hit_level ("TP"/"SL") in the position dict so _watchdog_close
        can relay them accurately to the Discord close notification.
    _latest_spread / _latest_spread_ts -- most recent bid-ask gap per symbol,
        written lock-free from that thread (single writer, so safe).
    _spread_stale_seconds -- 5 (RISK_SPREAD_STALE_SECONDS). Older than this and
        the cost gate falls back to its volatility-scaled estimate rather than
        trusting a stale number.
    _watchdog_close -- the actual software exit. Retries up to
        _close_max_attempts (5, OANDA_CLOSE_MAX_ATTEMPTS) because a failed
        close leaves a live position with no protection.

    ── unknown entry outcomes (added 2026-08-29) ──
    ENTRY_UNRECONCILED -- position state meaning "an order went out, failed
        ambiguously, and the broker could not be re-read, so we do not know
        whether this position exists". Counted by the exposure caps (so the
        bot stays conservative) but IGNORED by _on_tick, because running a
        software stop against a position that may not exist could close
        something the account does not hold.
    _park_unverified_entry -- records that state instead of discarding the
        submit as a zero fill. Dropping it is how a live position ends up
        with nothing enforcing its stop.
    _reconcile_unverified_entries -- settles every parked record against the
        broker, on the liveness loop (not a fire-and-forget task, which could
        be GC'd or swallow an exception). Flat -> drop the record; open ->
        promote to OPEN with the bracket rebuilt from the stored sl_dist /
        tp_dist around the price actually filled. A failed sync leaves the
        record parked for the next pass, never resolved by assumption.
    sl_dist / tp_dist (position keys) -- the approved bracket DISTANCES, kept
        on a parked record so the bracket can be rebuilt later. The distances
        are what the cost gate approved, so re-anchoring them on the real
        fill price does not reopen that gate.
    barrier= (calculate_bracket kwarg) -- learned geometry handed through from
        the signal's metadata (strategies.base.BARRIER_GEOMETRY_KEY). This
        orchestrator is a PASS-THROUGH: it does not decide whether geometry is
        learned or static, it only stops dropping the payload on the floor
        before the RiskManager sees it. The entry log line records which source
        produced the bracket, because a soak running fixed 1000-unit positions
        needs to know when the stop it is being handed came out several-fold
        wider than the constant.
    _risk_sizing -- RISK_SIZING_ENABLED: when True and geometry is learned
        (barrier), scales trade_units inversely with stop-loss width via
        RiskManager.calculate_forex_units to preserve constant dollar risk.
        Defaults to OFF (0).

    ── entry guards (added 2026-07-30 after the first multi-fill day) ──
    _reentry_cooldown -- OANDA_REENTRY_COOLDOWN_SECONDS: how long after an
        exit a symbol may not be re-entered. -1 (default) = one bar period,
        0 disables. Blocks OPENING only; flipping an already-open position is
        untouched.
    _last_exit_ts -- symbol -> time.monotonic() of its last close.
    _mark_exit / _cooldown_remaining -- start the cooldown / seconds left.
    _max_per_currency -- OANDA_MAX_PER_CURRENCY, default 2: how many open
        positions may share the SAME signed currency leg. Long GBP_JPY +
        long AUD_JPY are two short-JPY positions; a third would breach a cap
        of 2. 0 or negative disables.
    _currency_legs -- splits SYMBOL + signed units into {currency: ±1}, the
        unit the cap counts in. Unparseable symbol -> {} = uncapped.
    _exposure_conflict -- the cap check; returns a human-readable reason or
        None. Must be called holding _positions_lock.
    _pending_entries -- symbol -> signed units for an entry that passed the
        cap but whose fill has not returned. Counted as held, so two entries
        on the same bar cannot both pass. Released on every path out.
    _cooldown_rejections / _exposure_rejections -- how often each guard has
        blocked an entry, for the soak log.

    ── volatility tracking ──
    _wilder_atr / _regime_natr / _regime_prev_close -- a running volatility
        estimate advanced once per sealed bar in constant time, rather than
        recomputing over a window every bar. Seeded once at boot from the
        priming history.
    _natr_period -- 14, deliberately matching v3_features._NATR_PERIOD so the
        live gate and the model's features measure volatility identically.
    _regime_window -- how many bars of volatility history the regime gate
        sees; read from the RiskProfile so the two cannot disagree.

    ── chop-gate telemetry ──
    _devil_approved_total -- how many signals cleared BOTH model stages.
    _spread_gate_rejections / _regime_gate_rejections / _time_gate_rejections
        -- of those, how many each gate then vetoed. Split per gate so a soak
        log answers "why didn't it trade?" with a specific constraint rather
        than a shrug.
    _a3_chop_rejections -- the combined count.

    ── spread calibration (SPREAD_CALIB) ──
    Purpose: the assumed trading cost (alpha = 0.15) was a placeholder. This
    samples the REAL spread once per sealed bar -- off the fast path -- so the
    assumption can be replaced with measured per-instrument values.
    _spread_pct_samples / _baseline_natr_samples -- bounded deques of the two
        quantities, both expressed as a percent of price so their ratio is
        dimensionless and directly comparable to spread_atr_alpha.
    _spread_calib_maxlen -- 20000 (SPREAD_CALIB_MAXLEN); bounded so memory is
        flat and the estimate stays recency-weighted.
    _spread_calib_interval -- log every 60 bars (SPREAD_CALIB_INTERVAL_BARS).
    _log_spread_calibration -- emits the empirical alpha =
        median(spread_pct) / median(baseline_natr) per instrument. These are
        the numbers scripts/bake_spread_alphas.py turns into a table.

    ── liveness ──
    _stream_stale_seconds -- 60 (OANDA_STREAM_STALE_SECONDS). OANDA heartbeats
        every ~5s, so a minute of silence means the feed is dead.
    _check_stream_liveness / _liveness_watchdog -- the backstop: force a
        reconnect and FLATTEN exposure if the feed goes quiet. Critical because
        stops are software-enforced -- a dead feed means an unwatched position,
        so the safe response is to hold nothing. Gated since 2026-09-15 on
        risk_manager.scheduled_market_pause: silence during the daily 5pm-ET
        rollover or the Friday-close-to-Sunday-reopen weekend is EXPECTED, and
        responding to it flattens positions that are legitimately held across
        the rollover. During a pause this returns early (one INFO line, watchdog
        still re-armed); it never suppresses a genuine outage outside a pause.
    _liveness_pause_logged -- which scheduled pause has already been logged, so
        a 48-hour weekend is one INFO line instead of ~12,674 CRITICALs. None
        when not in a pause.
    _stream_with_retry -- reconnect loop around the blocking stream.
    _notify -- fires Discord posts off the event loop; they are blocking HTTP
        calls with a 5s timeout and must never block the loop.
    shutdown / _flatten_all -- graceful stop: stop the stream, optionally close
        everything, then exit.
"""

import asyncio
import functools
import logging
import os
import random
import signal as sig
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional

import numpy as np
import polars as pl
import talib

from core import events
from core.notification_manager import NotificationManager
from data.oanda_provider import OandaMarketProvider, _to_oanda_symbol
from execution.oanda_order_manager import OandaOrderManager, OrderCloseError
from execution.risk_manager import (
    GATE_REGIME,
    GATE_SPREAD,
    GATE_TIME,
    GEOMETRY_BARRIER,
    GEOMETRY_STATIC,
    RiskManager,
    scheduled_market_pause,
)
from strategies.base import BARRIER_GEOMETRY_KEY
from strategies.concrete_strategies.ml_strategy import MLStrategy

logger = logging.getLogger(__name__)

ENV_RISK_SIZING_ENABLED = "RISK_SIZING_ENABLED"


def _risk_sizing_requested(explicit: Optional[bool] = None) -> bool:
    """Resolve the risk sizing switch: explicit argument first, then
    RISK_SIZING_ENABLED. Anything unset/0/false/no/off means OFF."""
    if explicit is not None:
        return bool(explicit)
    raw = os.getenv(ENV_RISK_SIZING_ENABLED, "0").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


class OandaForexOrchestrator:
    """
    Async orchestrator for OANDA v20 forex.

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
        risk_sizing: Optional[bool] = None,
    ):
        self._symbols = symbols
        self._provider = provider
        self._strategy = strategy
        self._order_manager = order_manager
        self._risk_manager = risk_manager
        self._units_per_trade = units_per_trade
        self._flatten_on_exit = flatten_on_exit
        self._risk_sizing = _risk_sizing_requested(risk_sizing)

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

        # Newest bar timestamp that went through signal evaluation, per
        # symbol. NOT reset on reconnect — this is the dedup that keeps
        # repeated re-primes inside one bar period from scoring (and
        # potentially trading) the same bar twice.
        self._last_scored_ts: Dict[str, Optional[datetime]] = {
            _to_oanda_symbol(s): None for s in symbols
        }

        # Seam catch-up staleness bound, in seconds. Bars that sealed while
        # the stream was down are only scored if at most this old. -1 means
        # "one granularity period" (resolved at catch-up time, when the
        # provider's granularity is known); 0 disables catch-up entirely.
        self._seam_catchup_max_age = float(
            os.getenv("SEAM_CATCHUP_MAX_AGE_SECONDS", "-1")
        )

        # Seam backfill: how hard to chase the complete version of a dropped
        # in-flight bar from REST (0 disables), and how long to wait between
        # tries when REST has not published the sealed candle yet.
        self._seam_backfill_attempts = int(os.getenv("SEAM_BACKFILL_ATTEMPTS", "3"))
        self._seam_backfill_retry_delay = float(
            os.getenv("SEAM_BACKFILL_RETRY_DELAY", "2")
        )

        # Reconnect backoff. A flat retry hammers a struggling endpoint:
        # 2026-07-28 saw six reconnects in two minutes, each re-priming all
        # eight instruments, while OANDA's stream edge went silent for 20s
        # at a time. Capped below the liveness watchdog's flatten threshold
        # so a backoff never leaves a position unwatched longer than the
        # stall response already allows.
        self._reconnect_base_delay = float(
            os.getenv("OANDA_RECONNECT_BASE_DELAY", "5")
        )
        self._reconnect_max_delay = float(
            os.getenv("OANDA_RECONNECT_MAX_DELAY", "60")
        )
        self._reconnect_healthy_seconds = float(
            os.getenv("OANDA_RECONNECT_HEALTHY_SECONDS", "120")
        )

        # History prime retry (2026-09-09). The provider returns EMPTY for
        # any REST error, and an empty prime after a reconnect clears the
        # buffers — silently trading nothing for a whole warm-up window.
        # Bounded retries with backoff turn a transient blip into a repair.
        self._prime_attempts = int(os.getenv("OANDA_PRIME_ATTEMPTS", "3"))
        self._prime_backoff = float(os.getenv("OANDA_PRIME_BACKOFF", "2"))

        # Position state: symbol -> {entry, sl, tp, units, state}
        self._positions: Dict[str, dict] = {}
        self._positions_lock = threading.Lock()

        # ── post-exit cooldown ──
        # On 2026-07-30 NZD_JPY was stopped out at 13:53 UTC and re-entered
        # LONG at 14:00 on the next bar, into the same falling move, for a
        # second loss. A fresh stop is evidence the read was wrong; the next
        # bar is too soon to re-litigate it. -1 = one bar period (resolved
        # from the provider's granularity), 0 disables.
        self._reentry_cooldown = float(
            os.getenv("OANDA_REENTRY_COOLDOWN_SECONDS", "-1")
        )
        # symbol -> time.monotonic() when the position was closed. Monotonic
        # so an NTP step cannot retroactively expire or extend a cooldown.
        self._last_exit_ts: Dict[str, float] = {}

        # ── correlated-exposure cap ──
        # Also 2026-07-30: three simultaneous longs on GBP_JPY, AUD_JPY and
        # NZD_JPY are one short-JPY bet at 3x size, and the yen leg moved
        # against all three at once. Counts SIGNED currency legs across open
        # and in-flight positions; 0 or negative disables.
        self._max_per_currency = int(os.getenv("OANDA_MAX_PER_CURRENCY", "2"))
        # symbol -> signed target units for an entry that has passed the cap
        # check but whose fill has not returned yet. Guarded by
        # _positions_lock. Without this, two entries evaluated on the same
        # bar both see the pre-trade world and both pass a cap of 2.
        self._pending_entries: Dict[str, int] = {}

        # Entry-guard telemetry (mirrors the chop-gate counters).
        self._cooldown_rejections: int = 0
        self._exposure_rejections: int = 0

        # Process start, reported in the status snapshot so a dashboard can
        # show uptime without shelling out to ps.
        self._started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

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
        self._time_gate_rejections: int = 0
        self._a3_chop_rejections: int = 0  # combined (spread + regime + time)

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

        # Boot reconciliation retry policy. Refusing to start when broker
        # state is unverifiable is deliberate, but a single transient error
        # should not trigger it — with the cron watchdog's crash-loop brake,
        # one blip costs 15 minutes of downtime.
        self._reconcile_max_attempts = int(
            os.getenv("OANDA_RECONCILE_MAX_ATTEMPTS", "3")
        )
        self._reconcile_retry_delay = float(
            os.getenv("OANDA_RECONCILE_RETRY_DELAY", "3")
        )

        # Stream liveness policy (C3 hardening). The provider's read
        # timeout is the primary stall defense; this watchdog is the
        # backstop that also flattens exposure if the stream goes quiet.
        self._stream_stale_seconds = float(
            os.getenv("OANDA_STREAM_STALE_SECONDS", "60")
        )
        self._liveness_task: Optional[asyncio.Task] = None
        # One-shot alert per liveness incident (the probe runs every 10s and
        # an hours-long outage must not spam Discord). Re-armed when healthy.
        self._liveness_alert_fired = False
        # Which scheduled pause we have already logged about (2026-09-15), so a
        # 35-minute rollover or a 48-hour weekend logs one INFO line each rather
        # than 12,674 CRITICALs. None = not currently in a pause.
        self._liveness_pause_logged: Optional[str] = None

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

        # Determine breach price and which bracket was hit
        breached: bool = False
        breach_price: float = 0.0
        hit_level: str = ""
        if units > 0:  # long
            if bid <= sl:
                breached = True
                breach_price = bid
                hit_level = "SL"
            elif bid >= tp:
                breached = True
                breach_price = bid
                hit_level = "TP"
        elif units < 0:  # short
            if ask >= sl:
                breached = True
                breach_price = ask
                hit_level = "SL"
            elif ask <= tp:
                breached = True
                breach_price = ask
                hit_level = "TP"

        if not breached:
            return

        # Idempotent guard — set state to PENDING_CLOSE under lock;
        # also store breach metadata so _watchdog_close can relay them
        # accurately to the Discord notification.
        with self._positions_lock:
            current = self._positions.get(symbol)
            if not current or current.get("state") != "OPEN":
                return
            current["state"] = "PENDING_CLOSE"
            current["breach_price"] = breach_price
            current["hit_level"] = hit_level

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
            events.emit(
                "calib",
                sym=norm_sym,
                n=len(spreads),
                alpha_emp=round(alpha, 4),
                med_spread_pct=round(med_spread, 5),
                med_baseline_natr=round(med_base, 5),
            )

    # ── entry guards (cooldown + correlated exposure) ──────────────────

    # ── telemetry ──────────────────────────────────────────────────────

    def _emit_status(self) -> None:
        """
        Publish the 'right now' snapshot for the dashboard.

        Called once per bar and on every entry/exit. Serialises state that
        already exists — no new bookkeeping — so this stays a pure read of the
        orchestrator plus a queue put.
        """
        try:
            with self._positions_lock:
                positions = {
                    sym: dict(pos) for sym, pos in self._positions.items()
                }
                pending = dict(self._pending_entries)
            events.write_status(
                {
                    "pid": os.getpid(),
                    "started": self._started_at,
                    "granularity": getattr(self._provider, "_stream_gran", None),
                    # _symbols, not _bar_buffers: the buffers are keyed from
                    # the CONFIGURED basket at construction, so after
                    # _drop_untradeable_symbols they still hold instruments
                    # this run will never trade. Report what is actually
                    # traded, or a dashboard shows phantom coverage.
                    "symbols": list(self._symbols),
                    "positions": positions,
                    "pending_entries": pending,
                    "counters": {
                        "devil_approved": self._devil_approved_total,
                        "spread_veto": self._spread_gate_rejections,
                        "regime_veto": self._regime_gate_rejections,
                        "time_veto": self._time_gate_rejections,
                        "cooldown_blocks": self._cooldown_rejections,
                        "exposure_blocks": self._exposure_rejections,
                    },
                    "last_bar": {
                        sym: str(ts) if ts else None
                        for sym, ts in self._last_scored_ts.items()
                    },
                    "config": {
                        "units": self._units_per_trade,
                        "risk_sizing": self._risk_sizing,
                        "cooldown_s": self._reentry_cooldown,
                        "max_per_ccy": self._max_per_currency,
                        "angel_thr": getattr(self._strategy, "angel_threshold", None),
                        "devil_thr": getattr(self._strategy, "devil_threshold", None),
                    },
                }
            )
        except Exception:
            # Telemetry must never reach the trading path.
            pass

    def _release_pending(self, symbol: str) -> None:
        """Drop ``symbol``'s in-flight exposure reservation. Takes the lock."""
        with self._positions_lock:
            self._pending_entries.pop(symbol, None)

    def _mark_exit(self, symbol: str) -> None:
        """Start ``symbol``'s re-entry cooldown. Call on every real exit."""
        self._last_exit_ts[symbol] = time.monotonic()

    def _cooldown_remaining(self, symbol: str) -> float:
        """
        Seconds left before ``symbol`` may be re-entered; 0.0 if clear.

        The default (-1) resolves to one bar period, so an instrument that
        just exited sits out the next bar rather than re-entering on it.
        """
        window = self._reentry_cooldown
        if window < 0:
            window = float(getattr(self._provider, "_stream_gran", 1)) * 60.0
        if window <= 0:
            return 0.0
        last = self._last_exit_ts.get(symbol)
        if last is None:
            return 0.0
        return max(0.0, window - (time.monotonic() - last))

    @staticmethod
    def _currency_legs(symbol: str, units: int) -> Dict[str, int]:
        """
        Decompose a signed position into its two signed currency legs.

        Long GBP_JPY is "+GBP, -JPY"; short is the mirror. XAU_USD decomposes
        the same way (XAU against USD), which is what makes a metals position
        and a fiat cross comparable. Returns {} for an unparseable symbol so a
        naming surprise degrades to "uncapped", never to a crash on the
        entry path.
        """
        parts = symbol.split("_")
        if len(parts) != 2 or not all(parts):
            return {}
        base, quote = parts
        sign = 1 if units > 0 else -1
        return {base: sign, quote: -sign}

    def _exposure_conflict(self, symbol: str, target_units: int) -> Optional[str]:
        """
        Reason string if opening ``target_units`` would breach the cap, else None.

        MUST be called with ``_positions_lock`` held: it reads ``_positions``
        and ``_pending_entries`` together, and the caller reserves its own
        slot in ``_pending_entries`` under the same acquisition.
        """
        if self._max_per_currency <= 0:
            return None
        want = self._currency_legs(symbol, target_units)
        if not want:
            return None

        # Existing signed legs, counting in-flight entries as already held.
        # Keyed by (currency, sign) so "two shorts of JPY" and "one long,
        # one short" are counted as the different things they are.
        held: Dict[tuple, int] = {}
        others: Dict[tuple, List[str]] = {}
        existing_units = {
            sym: pos.get("units", 0)
            for sym, pos in self._positions.items()
            if sym != symbol and pos.get("units", 0)
        }
        existing_units.update(
            {sym: u for sym, u in self._pending_entries.items() if sym != symbol}
        )
        for sym, units in existing_units.items():
            for ccy, sign in self._currency_legs(sym, units).items():
                held[(ccy, sign)] = held.get((ccy, sign), 0) + 1
                others.setdefault((ccy, sign), []).append(sym)

        for ccy, sign in want.items():
            count = held.get((ccy, sign), 0)
            if count + 1 > self._max_per_currency:
                side = "long" if sign > 0 else "short"
                return (
                    f"{side} {ccy} exposure would reach {count + 1} positions "
                    f"(cap {self._max_per_currency}); already held via "
                    f"{', '.join(others.get((ccy, sign), []))}"
                )
        return None

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
                closed = await loop.run_in_executor(
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

            # close_position now returns True only when the broker has been
            # VERIFIED flat (2026-09-09). False means "verified still open"
            # (partial fill) — retry rather than popping the record.
            if not closed:
                logger.warning(
                    "[%s] Watchdog close attempt %d/%d: broker verified "
                    "still open (partial fill?) — retrying",
                    symbol,
                    attempt,
                    self._close_max_attempts,
                )
                if attempt < self._close_max_attempts:
                    await asyncio.sleep(2 ** (attempt - 1))
                continue

            # Success (close submitted, or broker already flat)
            with self._positions_lock:
                self._positions.pop(symbol, None)
            self._mark_exit(symbol)
            logger.info(
                "[%s] Watchdog close completed (attempt %d)", symbol, attempt
            )
            events.emit(
                "exit",
                sym=symbol,
                units=units,
                dir=direction,
                entry=pos_snapshot.get("entry"),
                sl=pos_snapshot.get("sl"),
                tp=pos_snapshot.get("tp"),
                exit_price=pos_snapshot.get("breach_price"),
                hit_level=pos_snapshot.get("hit_level"),
                reason="watchdog",
                attempt=attempt,
            )
            self._emit_status()
            if pos_snapshot:
                self._notify(
                    self._notifier.send_oanda_trade_alert,
                    symbol=symbol,
                    direction=direction,
                    action="WATCHDOG_CLOSE",
                    price=pos_snapshot.get("entry", 0.0),
                    units=units,
                    sl_price=pos_snapshot.get("sl"),
                    tp_price=pos_snapshot.get("tp"),
                    close_price=pos_snapshot.get("breach_price"),
                    hit_level=pos_snapshot.get("hit_level"),
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
            sl_price=pos_snapshot.get("sl"),
            tp_price=pos_snapshot.get("tp"),
            reason=(
                f"MANUAL INTERVENTION REQUIRED: watchdog close failed "
                f"{self._close_max_attempts} times ({last_error}). Position "
                f"remains open at broker without SL/TP enforcement."
            ),
        )

    def _park_unverified_entry(
        self,
        symbol: str,
        signal,
        target_units: int,
        sl_dist: float,
        tp_dist: float,
    ) -> None:
        """
        Record an entry whose outcome is UNKNOWN, so it cannot be forgotten.

        The record is parked in ``ENTRY_UNRECONCILED``, which ``_on_tick``
        deliberately ignores — we must not run a software stop against a
        position that may not exist. It still carries ``units``, so the
        exposure caps stay conservative while the outcome is unknown, and it
        replaces the in-flight reservation in one lock acquisition so the
        symbol is never briefly invisible to a concurrent cap check.

        ``_reconcile_unverified_entries`` resolves it on the liveness loop.
        """
        with self._positions_lock:
            self._positions[symbol] = {
                "entry": signal.entry_price,
                "sl": None,
                "tp": None,
                "units": target_units,
                "state": "ENTRY_UNRECONCILED",
                "sl_dist": sl_dist,
                "tp_dist": tp_dist,
                "dir": signal.direction,
            }
            self._pending_entries.pop(symbol, None)

        logger.critical(
            "[%s] Entry outcome UNKNOWN (order sent, broker unreadable) — "
            "parked as ENTRY_UNRECONCILED pending reconciliation. No stop is "
            "being enforced until the broker confirms what is open.",
            symbol,
        )
        events.emit(
            "entry_unreconciled",
            sym=symbol,
            dir=signal.direction,
            units=target_units,
            entry=signal.entry_price,
        )
        self._emit_status()
        self._notify(
            self._notifier.send_system_message,
            message=(
                f"🚨 {symbol}: entry order sent but its outcome could not be "
                f"verified with the broker. Parked as ENTRY_UNRECONCILED "
                f"({target_units:+d} units). SL/TP are NOT being enforced "
                f"until it reconciles — check the account if this persists."
            ),
        )

    async def _reconcile_unverified_entries(self) -> None:
        """
        Settle every ``ENTRY_UNRECONCILED`` record against the broker.

        Runs on the liveness loop rather than a fire-and-forget task, so it
        cannot be garbage-collected mid-flight and cannot swallow an
        exception unnoticed. A sync failure is left parked for the next pass —
        never resolved by assumption.
        """
        with self._positions_lock:
            parked = [
                (sym, rec.copy())
                for sym, rec in self._positions.items()
                if rec.get("state") == "ENTRY_UNRECONCILED"
            ]
        if not parked:
            return

        loop = asyncio.get_running_loop()
        for symbol, rec in parked:
            try:
                synced = await loop.run_in_executor(
                    None, self._order_manager.sync_position, symbol
                )
            except Exception as e:
                logger.error(
                    "[%s] Reconcile of parked entry raised: %s",
                    symbol, e, exc_info=True,
                )
                continue
            if not synced:
                logger.warning(
                    "[%s] Parked entry still unreconciled — broker unreadable; "
                    "retrying next pass",
                    symbol,
                )
                continue

            net = self._order_manager.get_net_position(symbol)

            if net == 0:
                with self._positions_lock:
                    current = self._positions.get(symbol)
                    if (
                        current is not None
                        and current.get("state") == "ENTRY_UNRECONCILED"
                    ):
                        self._positions.pop(symbol, None)
                self._mark_exit(symbol)
                logger.info(
                    "[%s] Parked entry reconciled: broker is flat, nothing was "
                    "taken — record dropped",
                    symbol,
                )
                self._emit_status()
                continue

            # A position really is open. Anchor its bracket on the price we
            # actually got: the signal price may be stale by however long the
            # failure took, and a stop measured from it could sit on the
            # wrong side of the market. The DISTANCES are what the cost gate
            # approved, and they are preserved exactly.
            avg = self._order_manager.get_average_entry_price(symbol)
            sl_dist = rec.get("sl_dist")
            tp_dist = rec.get("tp_dist")
            if not avg or sl_dist is None or tp_dist is None:
                logger.critical(
                    "[%s] Parked entry is OPEN at the broker (%d units) but "
                    "its bracket cannot be rebuilt (entry=%s sl_dist=%s "
                    "tp_dist=%s) — MANUAL INTERVENTION REQUIRED",
                    symbol, net, avg, sl_dist, tp_dist,
                )
                continue

            if net > 0:
                sl_price = avg - sl_dist
                tp_price = avg + tp_dist
            else:
                sl_price = avg + sl_dist
                tp_price = avg - tp_dist

            with self._positions_lock:
                current = self._positions.get(symbol)
                if (
                    current is None
                    or current.get("state") != "ENTRY_UNRECONCILED"
                ):
                    continue
                current.update(
                    {
                        "entry": avg,
                        "sl": sl_price,
                        "tp": tp_price,
                        "units": net,
                        "state": "OPEN",
                    }
                )

            logger.critical(
                "[%s] Parked entry reconciled: broker holds %d units — armed "
                "at entry=%.5f sl=%.5f tp=%.5f; the stop monitor now owns it",
                symbol, net, avg, sl_price, tp_price,
            )
            events.emit(
                "entry",
                sym=symbol,
                dir=rec.get("dir"),
                units=net,
                entry=avg,
                sl=sl_price,
                tp=tp_price,
                reason="reconciled",
            )
            self._emit_status()
            self._notify(
                self._notifier.send_oanda_trade_alert,
                symbol=symbol,
                direction=rec.get("dir"),
                action="ENTRY",
                price=avg,
                units=net,
                sl_price=sl_price,
                tp_price=tp_price,
                reason="reconciled after an unverified submit",
            )

    # ── bar callback (runs on the asyncio loop) ───────────────────────

    async def _on_bar(self, bar: dict) -> None:
        """Process a completed bar: update buffer, generate signal, trade."""
        # Shutdown guard: stop_stream() flushes the part-built bar after
        # _flatten_all has snapshotted the positions it will close. Accepting
        # that bar here could submit a NEW entry that resolves after the
        # flatten and exits the process unwatched. Never evaluate during
        # shutdown.
        if self._shutdown_event.is_set():
            return

        symbol = bar["symbol"]

        # Ignore bars for instruments outside the configured basket: the
        # regime-NATR state is keyed by normalized basket symbol
        # (_to_oanda_symbol), and feeding an unknown instrument in would
        # raise KeyError inside the unawaited loop future, silently dropping
        # the bar (and wedging nothing visibly). The basket itself is stored
        # raw (may be slash-form from the constructor), so test membership
        # against the normalized keys actually used everywhere else.
        if _to_oanda_symbol(symbol) not in self._regime_natr:
            return

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
                # The stream's copy of this bar is incomplete (it missed the
                # ticks from before the reconnect), but it HAS sealed — REST
                # holds the complete version. Fetch and score that instead of
                # losing the bar outright.
                await self._backfill_seam_bar(symbol, bar["timestamp"])
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

        # Record this bar as scored BEFORE evaluating — even a None signal
        # counts as "seen", so the seam catch-up can never rescore it.
        self._last_scored_ts[symbol] = bar["timestamp"]

        await self._evaluate_and_trade(symbol)
        self._emit_status()

    async def _backfill_seam_bar(self, symbol: str, ts: "datetime") -> None:
        """
        Re-fetch a dropped partial seam bar from REST and score it.

        The bar in flight when a disconnect happens is lost twice over: the
        re-prime cannot see it (it had not sealed yet) and the stream's own
        copy is incomplete, so ``_on_bar`` drops it. It has nonetheless
        SEALED by the time that drop fires — this runs at the bar boundary —
        so REST holds a complete version. Without this, every reconnect
        silently costs one evaluation per symbol (measured: 5 per symbol in
        the first 16h of the 2026-07-28 soak, ~8% of bars).

        Retries a few times because REST can lag the seal by a second or two;
        ``get_historical_bars`` filters out still-forming candles, so a miss
        is an empty frame rather than a partial bar. Failure is logged and
        dropped — never raised — because this runs inside the bar callback.
        """
        if self._seam_backfill_attempts <= 0:
            return

        last = self._last_scored_ts.get(symbol)
        if last is not None and ts <= last:
            return  # already scored by another path

        gran_min = getattr(self._provider, "_stream_gran", 1)
        loop = asyncio.get_running_loop()
        start = ts - timedelta(seconds=1)
        end = ts + timedelta(minutes=gran_min)

        for attempt in range(1, self._seam_backfill_attempts + 1):
            df = None
            try:
                df = await loop.run_in_executor(
                    None,
                    self._provider.get_historical_bars,
                    symbol,
                    gran_min,
                    start,
                    end,
                )
            except Exception as e:
                logger.warning(
                    "[%s] Seam backfill fetch failed (attempt %d/%d): %s",
                    symbol, attempt, self._seam_backfill_attempts, e,
                )

            row = None
            if df is not None and not df.is_empty():
                match = df.filter(pl.col("timestamp") == ts)
                if match.height:
                    row = match.row(0, named=True)

            if row is None:
                if attempt < self._seam_backfill_attempts:
                    await asyncio.sleep(self._seam_backfill_retry_delay)
                continue

            if row["timestamp"].tzinfo is None:
                logger.warning(
                    "[%s] Seam backfill got a timezone-naive bar at %s — "
                    "skipping (seam dedup would misbehave)",
                    symbol, row["timestamp"],
                )
                return

            buf = self._bar_buffers.get(symbol)
            if not buf:
                return
            # Re-check ordering after the await: a later bar must never be
            # overtaken by this one, or the buffer stops being monotonic.
            newest = buf[-1].get("timestamp")
            if newest is not None and ts <= newest:
                return
            if len(buf) < self._warmup:
                return

            buf.append({**row, "symbol": symbol})
            if len(buf) > self._max_bars:
                buf.pop(0)

            self._update_regime(symbol, buf[-1])
            self._sample_spread_calibration(symbol, buf[-1]["close"])
            self._spread_calib_bars += 1

            logger.info(
                "SEAM_BACKFILL [%s] recovered the in-flight bar %s from REST "
                "(attempt %d) — scoring it now",
                symbol, ts, attempt,
            )
            events.emit(
                "stream", kind="seam_backfill", sym=symbol,
                bar_ts=str(ts), attempt=attempt,
            )
            # Mark before evaluating: a failure here must not be retried into
            # a possible duplicate order on the next reconnect.
            self._last_scored_ts[symbol] = ts
            try:
                await self._evaluate_and_trade(symbol)
            except Exception as e:
                logger.error(
                    "[%s] Seam backfill evaluation failed: %s",
                    symbol, e, exc_info=True,
                )
            # Status must stay fresh across reconnects: the watchdog reads
            # status.json mtime, and a stream outage that this backfill
            # survived would otherwise look stale and trigger a restart.
            self._emit_status()
            return

        logger.warning(
            "[%s] Seam backfill gave up after %d attempts — the in-flight "
            "bar at %s is lost (one missed evaluation, no position risk)",
            symbol, self._seam_backfill_attempts, ts,
        )

    async def _evaluate_and_trade(self, symbol: str) -> None:
        """
        Decision tail shared by the stream path and the seam catch-up.

        Reads the CURRENT bar buffer for ``symbol`` and runs: strategy
        inference → position-state guards → bracket gates → order submit →
        position record. Acts on the newest buffer row; callers own freshness
        and dedup (``_last_scored_ts``).
        """
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

        # ── guard: post-exit cooldown ──
        # Only applies to OPENING a position. If one is already open this is
        # a flip, which is the model changing its mind about a live trade
        # rather than re-entering a closed one; the direction guards below
        # own that case.
        if existing is None:
            remaining = self._cooldown_remaining(symbol)
            if remaining > 0:
                self._cooldown_rejections += 1
                logger.info(
                    "[%s] Entry blocked by post-exit cooldown — %.0fs left "
                    "(%d blocked so far)",
                    symbol,
                    remaining,
                    self._cooldown_rejections,
                )
                events.emit(
                    "guard_block",
                    sym=symbol,
                    guard="cooldown",
                    remaining_s=round(remaining, 1),
                    total=self._cooldown_rejections,
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
                timestamp=datetime.now(timezone.utc),
                # Learned geometry, when the strategy attached it: per-bar
                # NATR quantile multiples that replace the static profile
                # constants. Absent on every path whose strategy does not
                # produce it (the whole rule-based library), in which case
                # this is None and nothing changes.
                barrier=(signal.metadata or {}).get(BARRIER_GEOMETRY_KEY),
            )
            if bracket:
                sl_dist, tp_dist = bracket
                if signal.direction == "long":
                    sl_price = signal.entry_price - sl_dist
                    tp_price = signal.entry_price + tp_dist
                else:  # short
                    sl_price = signal.entry_price + sl_dist
                    tp_price = signal.entry_price - tp_dist
                logger.info(
                    "[%s] bracket %.6f / %.6f (geometry=%s)",
                    symbol,
                    sl_dist,
                    tp_dist,
                    getattr(self._risk_manager, "last_geometry_source", "static"),
                )

        if sl_price is None or tp_price is None:
            gate = getattr(self._risk_manager, "last_veto_gate", GATE_SPREAD)
            if gate == GATE_REGIME:
                self._regime_gate_rejections += 1
            elif gate == GATE_TIME:
                self._time_gate_rejections += 1
            else:  # GATE_SPREAD or GATE_STATIC (cost-side floors)
                self._spread_gate_rejections += 1
            self._a3_chop_rejections += 1
            ratio = 100.0 * self._a3_chop_rejections / max(self._devil_approved_total, 1)
            logger.warning(
                "[%s] Bracket rejected (%s gate) | spread=%d regime=%d time=%d "
                "(%d / %d Devil-approved vetoed, %.1f%%)",
                symbol,
                gate,
                self._spread_gate_rejections,
                self._regime_gate_rejections,
                self._time_gate_rejections,
                self._a3_chop_rejections,
                self._devil_approved_total,
                ratio,
            )
            events.emit(
                "gate_veto",
                sym=symbol,
                gate=gate,
                spread=self._spread_gate_rejections,
                regime=self._regime_gate_rejections,
                time=self._time_gate_rejections,
                devil_approved=self._devil_approved_total,
            )
            return

        # ── derive signed target units ──
        trade_units = self._units_per_trade
        if (
            self._risk_sizing
            and self._risk_manager is not None
            and getattr(self._risk_manager, "last_geometry_source", GEOMETRY_STATIC) == GEOMETRY_BARRIER
        ):
            static_sl = getattr(
                self._risk_manager,
                "last_static_sl_dist",
                getattr(signal, "raw_sl_distance", getattr(signal, "raw_atr", 0.001)) * getattr(self._risk_manager.profile, "sl_atr_multiplier", 1.0),
            )
            trade_units = self._risk_manager.calculate_forex_units(
                base_units=self._units_per_trade,
                static_sl_distance=static_sl,
                actual_sl_distance=sl_dist,
            )
            logger.info(
                "[%s] Risk-sized units: %d -> %d (static_sl=%.5f actual_sl=%.5f ratio=%.2fx)",
                symbol,
                self._units_per_trade,
                trade_units,
                static_sl,
                sl_dist,
                static_sl / sl_dist if sl_dist > 0 else 1.0,
            )

        target_units = (
            trade_units
            if signal.direction == "long"
            else -trade_units
        )

        # ── guard: same-direction re-entry / mark reversal in flight ──
        # Re-checked under the lock immediately before submit: the position
        # may have gone PENDING_CLOSE since the earlier guard (tick watchdog
        # races bar processing). On a flip, mark the old position REVERSING
        # so the tick watchdog cannot fire on its stale SL/TP mid-submit.
        reversing = False
        with self._positions_lock:
            # Post-exit cooldown re-check (2026-09-09): the earlier cooldown
            # decision read an `existing` snapshot from the first lock block.
            # If a stop-out landed in between (the tick watchdog races bar
            # processing), the position record is now gone but `existing` was
            # non-None, so the earlier check skipped the cooldown — and the
            # symbol could re-enter on the very bar that stopped it out, the
            # exact 2026-07-30 NZD_JPY double-loss this guard exists for.
            existing_now = self._positions.get(symbol)
            if existing_now is None and existing is not None:
                remaining = self._cooldown_remaining(symbol)
                if remaining > 0:
                    self._cooldown_rejections += 1
                    logger.info(
                        "[%s] Entry blocked by post-exit cooldown (stop-out "
                        "landed mid-evaluation) — %.0fs left (%d blocked so far)",
                        symbol,
                        remaining,
                        self._cooldown_rejections,
                    )
                    events.emit(
                        "guard_block",
                        sym=symbol,
                        guard="cooldown",
                        remaining_s=round(remaining, 1),
                        total=self._cooldown_rejections,
                    )
                    return

            # Correlated-exposure cap, checked and RESERVED under one lock
            # acquisition so two same-bar entries cannot both pass it.
            conflict = self._exposure_conflict(symbol, target_units)
            if conflict is not None:
                self._exposure_rejections += 1
                logger.warning(
                    "[%s] Entry blocked by exposure cap: %s (%d blocked so far)",
                    symbol,
                    conflict,
                    self._exposure_rejections,
                )
                # emit() is a non-blocking queue put, so it is safe to call
                # while holding the positions lock.
                events.emit(
                    "guard_block",
                    sym=symbol,
                    guard="exposure",
                    detail=conflict,
                    total=self._exposure_rejections,
                )
                return
            self._pending_entries[symbol] = target_units

            existing = self._positions.get(symbol)
            if existing:
                # Each bail-out below releases the reservation inline: the
                # lock is held here and is NOT reentrant, so it cannot be
                # released through a helper that takes it.
                if existing.get("state") != "OPEN":
                    logger.info(
                        "[%s] Position state changed to %s before submit — "
                        "skipping entry",
                        symbol,
                        existing.get("state"),
                    )
                    self._pending_entries.pop(symbol, None)
                    return
                if existing["units"] > 0 and target_units > 0:
                    logger.info("[%s] Already long — skipping re-entry", symbol)
                    self._pending_entries.pop(symbol, None)
                    return
                if existing["units"] < 0 and target_units < 0:
                    logger.info("[%s] Already short — skipping re-entry", symbol)
                    self._pending_entries.pop(symbol, None)
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
        # From here the exposure reservation is live and MUST be released on
        # every path out — including the ones that record a position, since
        # _positions then carries the exposure itself.
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
            self._release_pending(symbol)
            return

        filled = result.get("filled", 0)
        if filled == 0:
            if result.get("unverified"):
                # An order went out, failed ambiguously, and the broker could
                # not be re-read. It may be live. Treating it as a clean miss
                # would leave a position nothing watches, so park it and let
                # the reconciler settle it against the broker.
                self._park_unverified_entry(
                    symbol, signal, target_units, sl_dist, tp_dist,
                )
                return
            logger.warning(
                "[%s] Order rejected / zero fill — not recording position",
                symbol,
            )
            _restore_open()
            self._release_pending(symbol)
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
                self._pending_entries.pop(symbol, None)
            return
        # Units verification (2026-09-09): the submit expressed a DELTA from
        # the cached net, and a flatten racing this entry (close between the
        # delta computation and the fill) leaves the broker at a different
        # net than requested — e.g. +2× units on a flip. Recording that as a
        # normal OPEN position would mis-track size and mis-reserve the
        # exposure cap. Park it; the reconciler settles against the broker.
        if actual_units != target_units:
            logger.critical(
                "[%s] Resulting net (%d) != requested target (%d) — a close "
                "raced this entry. Parking ENTRY_UNRECONCILED for the "
                "reconciler instead of recording a wrong-sized position.",
                symbol,
                actual_units,
                target_units,
            )
            self._park_unverified_entry(
                symbol, signal, target_units, sl_dist, tp_dist,
            )
            return

        # ── re-anchor brackets on the REAL fill (2026-09-09) ────────────
        # The bracket was computed from the signal's bar-close price. On
        # catch-up/backfill entries the fill can be a full bar's move away,
        # so anchoring the stop to a price that never traded can put it on
        # the wrong side of the market — the exact case
        # _reconcile_unverified_entries already fixes for parked entries.
        # Preserve the approved DISTANCES; re-center them on the fill.
        if avg_price and sl_dist is not None and tp_dist is not None:
            if signal.direction == "long":
                sl_price = avg_price - sl_dist
                tp_price = avg_price + tp_dist
            else:
                sl_price = avg_price + sl_dist
                tp_price = avg_price - tp_dist

        # Record the position and hand the exposure over to _positions in ONE
        # lock acquisition, so the symbol is never briefly invisible to a
        # concurrent cap check.
        with self._positions_lock:
            self._positions[symbol] = {
                "entry": avg_price,
                "sl": sl_price,
                "tp": tp_price,
                "units": actual_units,
                "state": "OPEN",
            }
            self._pending_entries.pop(symbol, None)

        logger.info(
            "[%s] Position opened | units=%d entry=%.5f sl=%.5f tp=%.5f",
            symbol,
            actual_units,
            avg_price,
            sl_price,
            tp_price,
        )

        meta = signal.metadata or {}
        events.emit(
            "entry",
            sym=symbol,
            dir=signal.direction,
            units=actual_units,
            entry=avg_price,
            sl=sl_price,
            tp=tp_price,
            angel=meta.get("angel_prob"),
            devil=meta.get("devil_prob"),
        )
        self._emit_status()

        # Discord trade alert (no-op if webhook unset; posted off-loop)
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

        That refusal is retried before it becomes fatal. On 2026-07-28 a
        single transient 401 on one instrument aborted startup, and the
        cron watchdog's crash-loop brake then kept the bot down for 15
        minutes over a blip that had cleared seconds later. Retrying costs
        nothing and does not weaken the guard: startup still refuses if the
        broker cannot be reached after every attempt.
        """
        loop = asyncio.get_running_loop()
        for symbol in self._symbols:
            norm_sym = _to_oanda_symbol(symbol)

            ok = False
            for attempt in range(1, self._reconcile_max_attempts + 1):
                ok = await loop.run_in_executor(
                    None, self._order_manager.sync_position, norm_sym
                )
                if ok:
                    break
                if attempt < self._reconcile_max_attempts:
                    delay = self._reconcile_retry_delay * attempt
                    logger.warning(
                        "[%s] Boot reconciliation attempt %d/%d could not "
                        "verify broker state; retrying in %.1fs",
                        norm_sym, attempt, self._reconcile_max_attempts, delay,
                    )
                    await asyncio.sleep(delay)
            if not ok:
                raise RuntimeError(
                    f"[{norm_sym}] Boot reconciliation failed after "
                    f"{self._reconcile_max_attempts} attempts: could not "
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
                closed = await loop.run_in_executor(
                    None, self._order_manager.close_position, norm_sym
                )
                if not closed:
                    # Verified-still-open (partial fill): same as failure —
                    # refuse to run unwatched.
                    raise OrderCloseError(
                        f"close_position({norm_sym}) verified still open"
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

    async def _drop_untradeable_symbols(self) -> None:
        """
        Remove configured symbols this account is not permitted to trade.

        Motivation (2026-08-06): XAU_USD and XAG_USD are in the trained basket
        but not on this account's instrument list. The model kept proposing
        them, the order was submitted, and OANDA rejected it with
        INSTRUMENT_NOT_TRADEABLE — three times in four days. Each one is a
        signal spent on an order that could never fill, plus a stack trace in
        the log that looks like a real fault.

        Fail-open by design. If the account cannot be queried we keep every
        symbol: eating an occasional rejection is far cheaper than a transient
        API blip silently muting the whole basket. Only a *successful* lookup
        that positively excludes a symbol may drop it.
        """
        get = getattr(self._provider, "get_tradeable_instruments", None)
        if get is None:
            return

        tradeable = await asyncio.get_running_loop().run_in_executor(None, get)
        if not tradeable:
            logger.warning(
                "Could not determine tradeable instruments — keeping all %d "
                "configured symbol(s). Untradeable ones will be rejected at "
                "order time.",
                len(self._symbols),
            )
            return

        keep = [s for s in self._symbols if _to_oanda_symbol(s) in tradeable]
        dropped = [s for s in self._symbols if s not in keep]

        if not keep:
            raise RuntimeError(
                "None of the configured symbols are tradeable on this "
                f"account: {self._symbols}. Refusing to start a bot that "
                "cannot place a single order."
            )

        if dropped:
            logger.warning(
                "UNTRADEABLE — dropping %s from this run; the account cannot "
                "trade %s. Trading %d of %d configured symbols.",
                dropped,
                "them" if len(dropped) > 1 else "it",
                len(keep),
                len(self._symbols),
            )
            events.emit("untradeable_dropped", dropped=dropped, kept=keep)

        self._symbols = keep

    async def _prime_history(self) -> None:
        """Prime bar buffers with historical REST data to bypass cold warm-up.

        Never raises: a failed/empty prime for one symbol degrades to a
        critical log (and catch-up can still repair a partial buffer), because
        an exception here propagates through _stream_with_retry and kills the
        reconnect loop permanently — a blind zombie the watchdog cannot see.
        """
        for symbol in self._symbols:
            norm_sym = _to_oanda_symbol(symbol)
            gran_min = getattr(self._provider, "_stream_gran", 1)
            start = datetime.now(timezone.utc) - timedelta(days=5)
            end = datetime.now(timezone.utc)

            df = None
            last_err: Optional[Exception] = None
            for attempt in range(self._prime_attempts):
                try:
                    df = await asyncio.get_running_loop().run_in_executor(
                        None,
                        self._provider.get_historical_bars,
                        symbol,
                        gran_min,
                        start,
                        end,
                    )
                    if df is not None and not df.is_empty():
                        break
                    last_err = None  # empty-but-not-raising is its own case
                except Exception as e:
                    last_err = e
                    logger.warning(
                        "[%s] Historical bars fetch failed (attempt %d/%d): %s",
                        norm_sym, attempt + 1, self._prime_attempts, e,
                    )
                if attempt + 1 < self._prime_attempts:
                    await asyncio.sleep(self._prime_backoff * (attempt + 1))

            if df is None or df.is_empty():
                # The provider returns EMPTY for API errors too, so this is
                # indistinguishable from "no bars" — but after a reconnect
                # the buffers were just cleared, so an empty prime means the
                # symbol will trade nothing for the whole warm-up window
                # while every liveness signal stays green. Log like it matters.
                logger.critical(
                    "[%s] PRIME FAILED after %d attempts (last error: %s) — "
                    "buffer empty; symbol will be silent for ~%d bars "
                    "(2.75 days at M15) unless catch-up repairs it",
                    norm_sym, self._prime_attempts, last_err, self._warmup,
                )
                continue

            # Keep enough tail to both warm the strategy and fully seed the
            # regime NATR deque (window + Wilder warmup), in BAR count.
            keep = max(self._warmup, self._regime_window) + self._natr_period + 5
            df = df.tail(keep)

            try:
                hist_bars = []
                for row in df.iter_rows(named=True):
                    if row["timestamp"].tzinfo is None:
                        # Skip the symbol rather than raise: a raise here
                        # escapes _stream_with_retry (which calls this
                        # unguarded) and kills the reconnect loop for good.
                        logger.critical(
                            "[%s] PRIME ABORTED: historical bar at %s is "
                            "timezone-naive — seam dedup would misbehave. "
                            "Skipping symbol.",
                            norm_sym, row["timestamp"],
                        )
                        hist_bars = []
                        df = df.clear()
                        break
                    hist_bars.append({**row, "symbol": norm_sym})
            except Exception as e:
                logger.critical(
                    "[%s] PRIME ABORTED while parsing bars: %s", norm_sym, e
                )
                continue

            if df.is_empty():
                continue

            self._bar_buffers[norm_sym].extend(hist_bars)
            self._seed_regime(norm_sym, df)
            self._last_hist_ts[norm_sym] = df.select("timestamp").row(-1)[0]
            logger.info(
                "[%s] Primed %d historical bars (tail) up to %s",
                norm_sym,
                len(hist_bars),
                self._last_hist_ts[norm_sym],
            )

    async def _catch_up_missed_bars(self) -> None:
        """
        Score the newest primed bar per symbol if it was never evaluated.

        Bars that seal while the stream is down (disconnect, restart, broker
        maintenance) are fetched by ``_prime_history`` into the buffer, but
        the stream never replays them — before this existed their signals
        were silently lost (~6 of 15 would-be signals died in these gaps
        during the 2026-07 soak). Runs after every prime, BEFORE the stream
        (re)starts, so it cannot race ``_on_bar``.

        Only the newest sealed bar is considered: older missed bars are
        decisions whose moment has passed. ``_last_scored_ts`` dedups across
        repeated reconnects inside one bar period; the age bound skips
        weekend/maintenance gaps. At boot this may re-enter a position that
        ``_reconcile_on_boot`` just flattened — deliberate: the signal is
        still live, and a restart should re-adopt it with fresh brackets.
        """
        max_age = self._seam_catchup_max_age
        gran_min = getattr(self._provider, "_stream_gran", 1)
        if max_age < 0:
            max_age = float(gran_min) * 60.0
        if max_age == 0:
            return
        now = datetime.now(timezone.utc)
        for norm_sym, buf in self._bar_buffers.items():
            try:
                if not buf or len(buf) < self._warmup:
                    continue
                bar = buf[-1]
                ts = bar.get("timestamp")
                if not isinstance(ts, datetime) or ts.tzinfo is None:
                    logger.warning(
                        "[%s] Seam catch-up skipped: newest primed bar has "
                        "no usable timestamp (%r)",
                        norm_sym,
                        ts,
                    )
                    continue
                sealed_at = ts + timedelta(minutes=gran_min)
                age = (now - sealed_at).total_seconds()
                if age < 0:
                    continue  # not sealed yet (defensive; prime filters these)
                if age > max_age:
                    logger.info(
                        "[%s] Seam catch-up: newest bar %s sealed %.0fs ago "
                        "(> %.0fs) — too stale to act on, skipping",
                        norm_sym,
                        ts,
                        age,
                        max_age,
                    )
                    continue
                last = self._last_scored_ts.get(norm_sym)
                if last is not None and ts <= last:
                    continue  # already scored (repeat reconnect in one bar)
                logger.info(
                    "SEAM_CATCHUP [%s] bar %s sealed %.0fs ago while the "
                    "stream was down — scoring it now",
                    norm_sym,
                    ts,
                    age,
                )
                events.emit(
                    "stream", kind="seam_catchup", sym=norm_sym,
                    bar_ts=str(ts), age_s=round(age, 1),
                )
                # Mark BEFORE evaluating: a failed evaluation must not retry
                # on the next reconnect — duplicate-order risk outranks one
                # lost signal.
                self._last_scored_ts[norm_sym] = ts
                await self._evaluate_and_trade(norm_sym)
            except Exception as e:
                logger.error(
                    "[%s] Seam catch-up failed: %s", norm_sym, e, exc_info=True
                )
            finally:
                # Same liveness argument as _backfill_seam_bar: bars scored
                # while the stream is down must keep status.json fresh, or
                # the watchdog restarts a healthy soak mid-recovery.
                self._emit_status()

    def _reconnect_delay(self, attempt: int) -> float:
        """
        Jittered, capped exponential backoff for reconnects (0-based attempt).

        Jitter spreads retries so a flapping endpoint is not hammered on a
        fixed cadence; the result is always in [cap/2, cap] and never exceeds
        ``_reconnect_max_delay``.
        """
        ceiling = min(
            self._reconnect_base_delay * (2 ** attempt), self._reconnect_max_delay
        )
        return ceiling * (0.5 + 0.5 * random.random())

    async def _stream_with_retry(self) -> None:
        """Run the pricing stream with reconnect-on-disconnect."""
        attempt = 0
        while not self._shutdown_event.is_set():
            started = time.monotonic()
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

            # A stream that ran healthily before dying is a fresh incident,
            # not an escalating one — start the backoff over.
            if time.monotonic() - started >= self._reconnect_healthy_seconds:
                attempt = 0

            delay = self._reconnect_delay(attempt)
            attempt += 1
            logger.warning(
                "Stream reconnect in %.1fs (consecutive failure %d); "
                "re-priming history on resume",
                delay,
                attempt,
            )
            events.emit(
                "stream", kind="disconnect", delay_s=round(delay, 1), attempt=attempt
            )
            await asyncio.sleep(delay)

            # Reset seam state so re-prime + new stream dedup cleanly
            for sym in list(self._bar_buffers.keys()):
                self._bar_buffers[sym] = []
                self._last_hist_ts[sym] = None
                self._seam_crossed[sym] = False

            await self._prime_history()

            # Score anything that sealed while we were dark (fresh bars only)
            await self._catch_up_missed_bars()

            # Defensive: clear provider stop event in case it was set
            self._provider.reset_stop()

            logger.info("Stream reconnect: priming complete, resuming stream")

    async def _check_stream_liveness(self) -> None:
        """
        One liveness probe with two failure windows (2026-09-09):

        1. Stream DOWN — the stream thread has exited. The old code returned
           on ``age is None``, which is exactly the state during every real
           outage (run_stream's finally nulls the age), so the flatten
           backstop was unreachable precisely when it mattered: disconnect +
           REST down meant a forever backoff loop with no ticks, no flatten,
           no alert, and a watchdog that saw a live process. If positions
           exist and the stream has been down past the threshold, flatten.
        2. Stream up but PRICE-silent — heartbeats used to count as liveness,
           so "connection alive, prices missing" could never flatten. The
           stops run on prices; measure price silence.

        REST and the pricing stream are separate connections, so the
        flatten very likely still works even when the stream is wedged.
        """
        # ── Scheduled pauses are not outages (2026-09-15) ──
        # Forex stops ticking for ~35 minutes at the daily 5pm-ET rollover and
        # from Friday 17:00 ET to Sunday 17:00 ET over the weekend. Price silence
        # during those windows is EXPECTED, and responding to it is not free: the
        # forced reconnect restores nothing (there is nothing to restore) and,
        # with a position open, the flatten below closes a live trade at the
        # moment spreads blow out tenfold. Measured cost of not gating this: on
        # 2026-09-11..13 the weekend produced 12,674 CRITICAL lines, 14 alert
        # incidents and a price clock reading 51,072s (14.2h) of "silence".
        #
        # Nothing is skipped permanently: the first probe after the reopen runs
        # this same path with prices flowing, so a stream that genuinely died
        # during the pause is caught within one threshold of the reopen.
        pause = scheduled_market_pause()
        if pause is not None:
            if self._liveness_pause_logged != pause:
                logger.info(
                    "Feed quiet during the scheduled %s — no action: prices are not "
                    "expected until the market reopens. Watchdog stays armed for a "
                    "real outage after that.",
                    pause,
                )
                self._liveness_pause_logged = pause
            # Re-arm, so the first GENUINE outage after the pause still alerts.
            self._liveness_alert_fired = False
            return
        self._liveness_pause_logged = None

        down_secs = self._provider.stream_down_seconds
        price_age = self._provider.seconds_since_last_price
        message_age = self._provider.seconds_since_last_message

        with self._positions_lock:
            has_positions = bool(self._positions)

        trigger: Optional[str] = None
        if down_secs is not None and down_secs > self._stream_stale_seconds:
            trigger = (
                f"Pricing stream DOWN for {down_secs:.0f}s (threshold "
                f"{self._stream_stale_seconds:.0f}s)"
            )
        elif price_age is not None and price_age > self._stream_stale_seconds:
            trigger = (
                f"No PRICE for {price_age:.0f}s (threshold "
                f"{self._stream_stale_seconds:.0f}s) — stream alive but silent"
            )
        elif message_age is not None and message_age > self._stream_stale_seconds:
            # Legacy window (pre-price-tracking providers): total silence
            # while the thread is up.
            trigger = (
                f"Pricing stream stale: no message for {message_age:.0f}s "
                f"(threshold {self._stream_stale_seconds:.0f}s)"
            )

        if trigger is None:
            # Healthy: re-arm the one-shot alert for the next incident.
            self._liveness_alert_fired = False
            return

        logger.critical(
            "%s — %s", trigger,
            "flattening exposure and forcing reconnect"
            if has_positions else "no positions held; forcing reconnect",
        )
        if not self._liveness_alert_fired:
            # One alert per incident — the liveness loop ticks every 10s and
            # an hours-long outage must not spam the notification channel.
            self._liveness_alert_fired = True
            self._notify(
                self._notifier.send_system_message,
                message=(
                    f"🚨 {trigger} — "
                    + (
                        "flattening open positions (SL/TP enforcement is "
                        "software-only and was blind during the outage)."
                        if has_positions
                        else "no positions held; forcing a reconnect."
                    )
                ),
            )

        if has_positions:
            await self._flatten_all()

        self._provider.force_disconnect("liveness watchdog: stream stale")

    async def _retry_failed_closes(self) -> None:
        """Retry CLOSE_FAILED positions every liveness pass (2026-09-09).

        A failed stop-exit used to be terminal for the whole run: the
        position stayed open at the broker with no SL/TP enforcement for the
        remaining life of the process (days, on a soak) while the bot kept
        trading other symbols. One ~30s OANDA blip during a breach was
        enough. Now every 10s liveness pass re-attempts the verified close;
        success pops the record and marks the exit like any other close.
        """
        with self._positions_lock:
            failed_syms = [
                s for s, p in self._positions.items()
                if p.get("state") == "CLOSE_FAILED"
            ]
        for sym in failed_syms:
            try:
                closed = await asyncio.get_running_loop().run_in_executor(
                    None, self._order_manager.close_position, sym
                )
            except Exception as e:
                logger.error(
                    "[%s] CLOSE_FAILED retry attempt failed: %s", sym, e
                )
                continue
            if closed:
                with self._positions_lock:
                    snap = self._positions.pop(sym, {})
                self._mark_exit(sym)
                logger.info(
                    "[%s] CLOSE_FAILED position recovered — broker verified flat",
                    sym,
                )
                units = snap.get("units", 0)
                events.emit(
                    "exit",
                    sym=sym,
                    units=units,
                    dir="long" if units > 0 else "short",
                    entry=snap.get("entry"),
                    sl=snap.get("sl"),
                    tp=snap.get("tp"),
                    exit_price=None,
                    hit_level=None,
                    reason="close_failed_retry",
                )

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
            try:
                await self._reconcile_unverified_entries()
            except Exception as e:
                logger.error(
                    "Unverified-entry reconcile failed: %s", e, exc_info=True
                )
            try:
                await self._retry_failed_closes()
            except Exception as e:
                logger.error("CLOSE_FAILED retry pass failed: %s", e, exc_info=True)

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

        # Drop instruments the account cannot trade. Runs AFTER reconciliation
        # so orphan checks still cover every configured symbol.
        await self._drop_untradeable_symbols()

        events.emit(
            "boot",
            pid=os.getpid(),
            symbols=self._symbols,
            granularity=getattr(self._provider, "_stream_gran", None),
            units=self._units_per_trade,
            risk_sizing=self._risk_sizing,
            warmup=self._warmup,
            cooldown_s=self._reentry_cooldown,
            max_per_ccy=self._max_per_currency,
            angel_thr=getattr(self._strategy, "angel_threshold", None),
            devil_thr=getattr(self._strategy, "devil_threshold", None),
        )
        self._emit_status()

        self._provider.subscribe(
            self._symbols,
            self._on_bar,
            tick_callback=self._on_tick,
        )

        await self._prime_history()

        # A signal bar that sealed just before this process started (crash
        # relaunch, watchdog restart) is in the primed history but would
        # otherwise never be scored — catch it while it is still fresh.
        await self._catch_up_missed_bars()

        # Run the pricing stream with reconnect-on-disconnect wrapper
        self._stream_task = asyncio.create_task(self._stream_with_retry())

        # Backstop liveness watchdog (C3): flatten + reconnect on stall
        self._liveness_task = asyncio.create_task(self._liveness_watchdog())

        logger.info(
            "OandaForexOrchestrator started | symbols=%s warmup=%d",
            self._symbols,
            self._warmup,
        )

        await self._shutdown_event.wait()
        await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown: stop stream, flatten if configured."""
        logger.info("OandaForexOrchestrator shutting down...")

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
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception as e:
                # The stream task died with an exception (e.g. an unguarded
                # prime failure in an older build). Do NOT let it propagate:
                # flatten below is the last safety net for open positions,
                # and skipping it on a teardown error is how a stopped bot
                # leaves a live position unwatched.
                logger.error(
                    "Stream task raised during shutdown (%s) — continuing "
                    "to flatten",
                    e,
                )
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except (asyncio.CancelledError, Exception):
                    pass

        if self._flatten_on_exit:
            await self._flatten_all()

        logger.info("OandaForexOrchestrator shutdown complete.")

    async def _flatten_all(self) -> None:
        """Close all open positions on exit.

        2026-09-09: only records whose close was VERIFIED flat are cleared.
        Failures (and verified-still-open partials) are parked CLOSE_FAILED
        instead of deleted — the old unconditional ``_positions.clear()``
        deleted the tracking record of a position that was still open at the
        broker, leaving nothing watching it while the bot kept trading (or,
        on shutdown, nothing at all). ENTRY_UNRECONCILED records stay parked
        for the reconciler, and symbols with an in-flight entry are skipped:
        closing the old position mid-delta could double the fill.
        """
        with self._positions_lock:
            pending = set(self._pending_entries.keys())
            targets = [
                (sym, dict(p))
                for sym, p in self._positions.items()
                if p.get("state") != "ENTRY_UNRECONCILED" and sym not in pending
            ]

        if not targets:
            return

        logger.info("Flattening %d position(s) on exit", len(targets))

        tasks = [
            asyncio.get_running_loop().run_in_executor(
                None, self._order_manager.close_position, sym
            )
            for sym, _snap in targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed: List[str] = []
        for (sym, snap), result in zip(targets, results):
            if isinstance(result, Exception) or result is False:
                failed.append(sym)
                # Park, never clear: the position may still be open at the
                # broker and its record is the only thing tracking it.
                with self._positions_lock:
                    current = self._positions.get(sym)
                    if current is not None:
                        current["state"] = "CLOSE_FAILED"
                logger.critical(
                    "[%s] Flatten close failed on exit — position may "
                    "remain open at broker (record parked CLOSE_FAILED): %s",
                    sym,
                    result,
                )
            else:
                # Also reached by the liveness watchdog mid-run, where the
                # process keeps trading: a flatten is a real exit and starts
                # the cooldown like any other.
                with self._positions_lock:
                    self._positions.pop(sym, None)
                self._mark_exit(sym)
                logger.info("[%s] Flattened on exit", sym)
                flat_units = snap.get("units", 0)
                events.emit(
                    "exit",
                    sym=sym,
                    units=flat_units,
                    dir="long" if flat_units > 0 else "short",
                    entry=snap.get("entry"),
                    sl=snap.get("sl"),
                    tp=snap.get("tp"),
                    exit_price=None,
                    hit_level=None,
                    reason="flatten",
                )

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
