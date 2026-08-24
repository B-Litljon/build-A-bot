"""
The Shield -- bracket sizing, position sizing, and the "chop filter" that
refuses trades not worth taking.

Two jobs, both applied AFTER a strategy has already decided it wants to trade:

1. Turn a raw volatility number into a stop and target distance.
2. Veto the trade entirely if conditions make it unwinnable -- if the cost of
   trading eats the expected move, if the market is too quiet to move at all,
   or if it is a time of day when spreads blow out.

The veto is the important part. A model can be perfectly correct about
direction and still lose money on an instrument whose trading cost exceeds the
move being predicted, which is exactly what this filter exists to prevent.

SYMMETRY CONTRACT: ``coupled_keff`` and the gate logic here are mirrored by
``retrainer._compute_chop_veto_mask`` so the model only ever trains on bars the
live bot would actually trade. Change a gate here and the training side must
change with it, or the model learns from setups it will never be given.

See GLOSSARY.md (chop veto, NATR, bracket, spread, pip).

Glossary:
    RiskProfile -- all the tunable numbers, per asset class. Built by
        for_asset_class(), which also reads the environment overrides.
    sl_atr_multiplier / tp_atr_multiplier -- bracket width in units of recent
        volatility. Defaults 0.5 / 3.0 (a 6:1 payoff) for equities; the FOREX
        profile overrides these to 2.0 / 4.0 (2:1). Always check which profile
        is live before assuming a ratio. The forex pair was 1.0 / 2.0 until
        2026-08-08; it was doubled because the spread was eating a median 40%
        of a 1x-ATR stop. Widening does NOT improve the odds of the bracket
        (measured stop-hit rate is ~66.6% at every width, against 66.7% for a
        fair 2:1) -- it dilutes a fixed cost over more risk, cutting the toll
        to ~22% and the break-even win rate from 46.7% to 40.6%. See
        llm_reports/recons/2026-08-08_stop-width-and-the-spread-toll.md.
    risk_per_trade -- 0.02, i.e. risk 2% of account equity per trade. This,
        not a fixed size, is what determines position size.
    max_notional_cap -- 100000, a hard ceiling on position value regardless of
        what the risk maths asks for.
    round_precision -- decimal places for bracket prices; 4 for equities, 5 for
        forex (which quotes finer).
    min_sl_pct -- 0.0015 (0.15%), the equities stop floor.
    min_sl_pips -- 2.0, the forex stop floor in pips.
    min_sl_pct_metals -- 0.0001, a PERCENT floor for gold/silver, because a
        forex "pip" of 0.0001 is meaningless on an instrument priced near 2700
        and would never trigger.
    _METAL_BASES -- {XAU, XAG, XPT, XPD}; how a metal is recognised.
    _get_forex_pip_size -- 0.01 for anything quoted in JPY, 0.0001 otherwise.

    ── The three gates ──
    last_veto_gate -- which gate rejected the most recent call, or GATE_NONE.
        The orchestrator reads this to log WHICH constraint is binding.
    GATE_SPREAD / GATE_REGIME / GATE_TIME / GATE_STATIC / GATE_NONE -- the
        identifiers for that telemetry.
    Gate A (cost, GATE_SPREAD) -- rejects when the stop distance is smaller
        than the cost of trading: sl_dist < k_eff * spread. Trading through a
        toll bigger than the move you are targeting loses on average even when
        the direction is right.
    Gate B (regime, GATE_REGIME) -- rejects when current volatility sits in the
        bottom regime_pctile (20%) of its own recent window. A market too quiet
        to move cannot reach the target before the hold limit expires.
    Gate C (time, GATE_TIME) -- rejects everything inside a daily blackout
        window, default 16:55-17:30 New York (_DEFAULT_BLACKOUT_ET). This is
        the daily rollover, when spreads briefly blow out roughly tenfold.
        Checked FIRST and independent of volatility, because the spread is
        toxic then regardless of conditions. Anchored to New York local time,
        not UTC, so it tracks daylight saving instead of drifting an hour
        twice a year.

    spread_k_base -- 3.0 for forex; the base safety multiple on cost in Gate A.
        Algebraically a TOLL CAP: the gate is sl_dist >= k * spread, i.e. it
        admits a trade only when the spread eats at most 1/k of the stop
        distance. k=3.0 caps the toll at 33%. It was 1.5 (a 67% cap) until
        2026-08-08, which admitted trades that could not pay for themselves.
    spread_k_coupling -- 0.0 by default, i.e. DECOUPLED and flat. Non-zero
        makes the multiplier scale with volatility.
    spread_k_coupling_mode -- "tighten" (more cost discipline as volatility
        rises) or "loosen" (less). Two competing theses, left switchable so a
        soak can decide between them.
    coupled_keff -- computes the effective multiplier. Scales only ABOVE the
        median volatility, and is clipped at >= 1.0 so a passing trade's cost
        can never exceed its own stop distance. Shared verbatim with the
        training side; that sharing is the symmetry contract.
    regime_pctile -- 20.0; Gate B's cut-off percentile.
    regime_window -- 260 BARS (not calendar time) of volatility history.
    regime_min_samples -- 60. Below this the window is "cold": Gate B is
        bypassed entirely and Gate A runs decoupled, so a just-started bot does
        not veto everything on a half-filled buffer.
    spread_atr_alpha -- 0.15, the assumed cost as a fraction of baseline
        volatility, used when no fresh live spread is available.
    _alpha_overrides -- per-instrument measured costs from the model's
        spread_alphas.json. Used ONLY in the stale-spread fallback; a fresh
        live spread always wins.
    spread_fresh -- whether the passed spread is recent enough to trust. When
        false the volatility-scaled proxy is used instead.

    ── Kill switches (all environment variables) ──
    RISK_CHOP_FILTER_ENABLED -- master off switch for ALL gating.
    RISK_SPREAD_GATE_ENABLED / RISK_REGIME_GATE_ENABLED /
        RISK_TIME_GATE_ENABLED -- per-gate switches. Every flag treats
        0/false/no/off as disabled and anything else (including unset) as
        enabled, so gates are ON by default.

    ── Sizing ──
    calculate_bracket -- returns (sl_distance, tp_distance) as DISTANCES, or
        None when a gate vetoes. Two paths: the dynamic hybrid gates when a
        regime_series is supplied (live forex), otherwise the legacy static
        floor (equities / cold start).
    calculate_quantity -- position size from risk_per_trade, then capped by
        max_notional_cap and by available buying power (95% of it). Returns
        the smallest of the three.
    is_crypto -- routes sizing to the cash balance instead of buying power,
        because Alpaca reports crypto funds in the cash field.
    $50 minimum notional -- anything smaller returns 0.0 and the trade is
        skipped, to avoid pointless dust positions.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)

# Environment variables for tuning the chop filter without code changes.
# The dynamic hybrid floor (cost gate + regime gate) is now simulated
# symmetrically in retrainer.py target generation, closing the historical
# training/inference asymmetry. These knobs tune (or disable) each gate per
# environment. See llm_reports/architecture/2026-06-14_dynamic-chop-floor.md.
ENV_FOREX_MIN_SL_PIPS = "RISK_FOREX_MIN_SL_PIPS"
ENV_EQUITIES_MIN_SL_PCT = "RISK_EQUITIES_MIN_SL_PCT"
ENV_METALS_MIN_SL_PCT = "RISK_METALS_MIN_SL_PCT"
ENV_CHOP_FILTER_ENABLED = "RISK_CHOP_FILTER_ENABLED"  # master kill (all gates)

# Dynamic hybrid floor (Option 4) — coupled cost + regime gates.
ENV_SPREAD_K = "RISK_SPREAD_K"                       # base spread multiplier
ENV_SPREAD_K_COUPLING = "RISK_SPREAD_K_COUPLING"     # vol-coupling strength
ENV_COUPLING_MODE = "RISK_COUPLING_MODE"             # "tighten" | "loosen"
ENV_REGIME_PCTILE = "RISK_REGIME_PCTILE"             # Gate B percentile P
ENV_REGIME_WINDOW = "RISK_REGIME_WINDOW"             # rolling window (bars)
ENV_REGIME_MIN_SAMPLES = "RISK_REGIME_MIN_SAMPLES"   # cold-start threshold
ENV_SPREAD_ATR_ALPHA = "RISK_SPREAD_ATR_ALPHA"       # proxy spread / baseline ATR
ENV_SPREAD_GATE_ENABLED = "RISK_SPREAD_GATE_ENABLED"
ENV_REGIME_GATE_ENABLED = "RISK_REGIME_GATE_ENABLED"

# Gate C — time-of-day blackout (e.g. the 5pm-NY daily rollover, when spreads
# blow out ~10x and the model's signals are un-tradeable). Window is in
# America/New_York local time so it tracks the rollover across DST.
ENV_TIME_GATE_ENABLED = "RISK_TIME_GATE_ENABLED"
ENV_BLACKOUT_ET = "RISK_BLACKOUT_ET"   # "HH:MM-HH:MM" in America/New_York

# Coupling modes (the two competing financial theses, soak-selectable).
COUPLING_TIGHTEN = "tighten"  # Claude: more cost discipline as vol expands
COUPLING_LOOSEN = "loosen"    # Gemini: relax cost discipline in high-momentum runs

# Veto-gate identifiers for split telemetry.
GATE_NONE = "none"
GATE_SPREAD = "spread"  # Gate A — transaction-cost floor
GATE_REGIME = "regime"  # Gate B — low-volatility regime floor
GATE_STATIC = "static"  # legacy static floor (equities / no regime context)
GATE_TIME = "time"      # Gate C — time-of-day blackout (e.g. NY 5pm rollover)

# Gate C blackout is anchored to America/New_York so it tracks the 5pm rollover
# across DST (≈21:00 UTC in summer, ≈22:00 UTC in winter). A fixed UTC hour
# would silently drift an hour every DST change.
_DEFAULT_BLACKOUT_ET = "16:55-17:30"
try:
    _NY_TZ: "Optional[ZoneInfo]" = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover — tzdata missing
    _NY_TZ = None


def _parse_blackout_et(spec: str) -> "Optional[Tuple[dtime, dtime]]":
    """Parse ``"HH:MM-HH:MM"`` (America/New_York) → (start, end); None if bad."""
    try:
        start_s, end_s = spec.split("-")
        sh, sm = (int(x) for x in start_s.strip().split(":"))
        eh, em = (int(x) for x in end_s.strip().split(":"))
        return dtime(sh, sm), dtime(eh, em)
    except Exception:
        logger.warning("Invalid RISK_BLACKOUT_ET=%r; Gate C disabled", spec)
        return None


# Metals quoted like forex pairs (XAU_USD etc.) where a 0.0001 "pip" is
# meaningless — the floor for these uses a percent of price instead.
_METAL_BASES = {"XAU", "XAG", "XPT", "XPD"}


def _chop_filter_enabled() -> bool:
    """RISK_CHOP_FILTER_ENABLED toggles the floor. Unset / 1 / true enables."""
    raw = os.getenv(ENV_CHOP_FILTER_ENABLED, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _flag_enabled(env_name: str, default: bool = True) -> bool:
    raw = os.getenv(env_name, "1" if default else "0").strip().lower()
    return raw not in ("0", "false", "no", "off")


def coupled_keff(spread_k_base, spread_k_coupling, mode, pctile_rank):
    """
    Volatility-coupled spread multiplier ``k_eff``. Shared verbatim by live
    execution (``RiskManager.calculate_bracket``) and the training pipeline
    (``retrainer._compute_chop_veto_mask``) so the cost gate is symmetric by
    construction. Accepts scalars or numpy arrays for ``pctile_rank``.

        scale = max(0, (pctile_rank - 0.5) / 0.5)   # couples only above median
        tighten:  k_eff = base * (1 + coupling * scale)
        loosen:   k_eff = base * (1 - coupling * scale)

    Clipped to ``k_eff >= 1.0`` so a passing trade's spread can never exceed
    its stop-loss distance (cost <= sl_dist).
    """
    scale = np.maximum(0.0, (pctile_rank - 0.5) / 0.5)
    if mode == COUPLING_LOOSEN:
        k_eff = spread_k_base * (1.0 - spread_k_coupling * scale)
    else:  # tighten (default)
        k_eff = spread_k_base * (1.0 + spread_k_coupling * scale)
    return np.maximum(1.0, k_eff)


@dataclass
class RiskProfile:
    sl_atr_multiplier: float = 0.5
    tp_atr_multiplier: float = 3.0
    min_sl_pct: float = 0.0015  # 0.15% absolute floor
    min_sl_pips: float = 2.0     # Default Forex pip floor (2.0 pips)
    # Metals floor: % of price. Default 0.01% matches the relative scale of
    # the 2-pip floor on the JPY crosses (2 pips on GBP/JPY ≈ 0.01% of price),
    # so the chop filter has comparable bite across the trained basket.
    min_sl_pct_metals: float = 0.0001
    risk_per_trade: float = 0.02 # 2% of account
    max_notional_cap: float = 100000.0
    round_precision: int = 4

    # ── Dynamic hybrid floor (Option 4): coupled cost + regime gates ──
    # Used when calculate_bracket() is given a regime_series (live forex);
    # falls back to the static floors above otherwise (equities / cold start).
    spread_k_base: float = 1.5            # Gate A: sl_dist >= k_eff * spread
    spread_k_coupling: float = 0.0        # 0.0 = decoupled flat k (safe default)
    spread_k_coupling_mode: str = COUPLING_TIGHTEN
    regime_pctile: float = 20.0           # Gate B: veto bottom P% of vol
    regime_window: int = 260              # rolling window, BARS (not calendar)
    regime_min_samples: int = 60          # cold-start: below this, Gate B bypassed
    spread_atr_alpha: float = 0.15        # proxy spread = alpha * baseline ATR

    # ── Gate C (time-of-day blackout) ── America/New_York window. None = no
    # window parsed → gate no-ops. Populated by for_asset_class() from env.
    blackout_start: Optional[dtime] = None
    blackout_end: Optional[dtime] = None

    @classmethod
    def for_asset_class(cls, asset_class: str) -> "RiskProfile":
        if asset_class == "forex":
            bo = _parse_blackout_et(os.getenv(ENV_BLACKOUT_ET, _DEFAULT_BLACKOUT_ET))
            return cls(
                sl_atr_multiplier=2.0,
                tp_atr_multiplier=4.0,
                min_sl_pips=float(os.getenv(ENV_FOREX_MIN_SL_PIPS, "2.0")),
                min_sl_pct_metals=float(os.getenv(ENV_METALS_MIN_SL_PCT, "0.0001")),
                round_precision=5,
                spread_k_base=float(os.getenv(ENV_SPREAD_K, "3.0")),
                spread_k_coupling=float(os.getenv(ENV_SPREAD_K_COUPLING, "0.0")),
                spread_k_coupling_mode=os.getenv(
                    ENV_COUPLING_MODE, COUPLING_TIGHTEN
                ).strip().lower(),
                regime_pctile=float(os.getenv(ENV_REGIME_PCTILE, "20.0")),
                regime_window=int(os.getenv(ENV_REGIME_WINDOW, "260")),
                regime_min_samples=int(os.getenv(ENV_REGIME_MIN_SAMPLES, "60")),
                spread_atr_alpha=float(os.getenv(ENV_SPREAD_ATR_ALPHA, "0.15")),
                blackout_start=bo[0] if bo else None,
                blackout_end=bo[1] if bo else None,
            )
        return cls(
            min_sl_pct=float(os.getenv(ENV_EQUITIES_MIN_SL_PCT, "0.0015")),
        )

    @property
    def spread_gate_enabled(self) -> bool:
        return _flag_enabled(ENV_SPREAD_GATE_ENABLED)

    @property
    def regime_gate_enabled(self) -> bool:
        return _flag_enabled(ENV_REGIME_GATE_ENABLED)

    @property
    def time_gate_enabled(self) -> bool:
        return _flag_enabled(ENV_TIME_GATE_ENABLED)

class RiskManager:
    """
    The Shield: Enforces institutional-grade safety nets and dynamic sizing.
    """
    def __init__(
        self,
        profile: RiskProfile = RiskProfile(),
        alpha_overrides: Optional[dict] = None,
    ):
        self.profile = profile
        # Per-instrument spread alphas ({symbol: alpha_emp}, from the model
        # dir's spread_alphas.json). Used ONLY in Gate A's stale-spread proxy
        # branch — a fresh live tick spread always wins. Symbols not listed
        # fall back to the flat profile.spread_atr_alpha.
        self._alpha_overrides: dict = alpha_overrides or {}
        # Which gate vetoed the most recent calculate_bracket() call (read by
        # the orchestrator for split telemetry). GATE_NONE when it passed.
        self.last_veto_gate: str = GATE_NONE

    def calculate_bracket(
        self,
        entry_price: float,
        raw_atr: float,
        symbol: Optional[str] = None,
        spread: Optional[float] = None,
        spread_fresh: bool = False,
        regime_series: Optional[Sequence[float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Apply multipliers and the chop filter to raw ATR volatility.

        Returns (sl_distance, tp_distance), or None if a gate vetoes the trade.
        ``self.last_veto_gate`` is set to the gate that fired.

        Two paths:
          * Dynamic hybrid (when ``regime_series`` is provided — live forex):
            Gate A (cost) + Gate B (regime), coupled via the vol percentile.
          * Static floor (no regime context — equities / cold start): the
            legacy pip / percent floor, preserved for backward compatibility.

        ``regime_series`` holds recent NATR scalars (percent) for ``symbol``,
        newest last. ``spread`` is the live absolute bid-ask spread; when not
        ``spread_fresh`` a volatility-scaled proxy is used instead.
        """
        self.last_veto_gate = GATE_NONE
        sl_dist = raw_atr * self.profile.sl_atr_multiplier
        tp_dist = raw_atr * self.profile.tp_atr_multiplier

        def _bracket() -> Tuple[float, float]:
            return (
                round(sl_dist, self.profile.round_precision),
                round(tp_dist, self.profile.round_precision),
            )

        # Master kill switch — no gating at all.
        if not _chop_filter_enabled():
            return _bracket()

        # Dynamic hybrid floor (live forex passes a regime series).
        if regime_series is not None and len(regime_series) > 0:
            gate = self._evaluate_dynamic_gates(
                entry_price, sl_dist, symbol, spread, spread_fresh,
                regime_series, timestamp,
            )
            if gate != GATE_NONE:
                self.last_veto_gate = gate
                return None
            return _bracket()

        # Legacy static floor (equities / no regime context).
        floor = self._static_floor(entry_price, symbol)
        if sl_dist < floor:
            self.last_veto_gate = GATE_STATIC
            logger.info(
                "[%s] static floor veto: sl_dist=%.6f < floor=%.6f (shortfall=%.6f)",
                symbol or "unknown", sl_dist, floor, floor - sl_dist,
            )
            return None
        return _bracket()

    def _static_floor(self, entry_price: float, symbol: Optional[str]) -> float:
        """Legacy pip / percent floor (used when no regime context is given)."""
        if symbol and self._is_metal_symbol(symbol):
            # Pip-based floors are meaningless for XAU/XAG (a 0.0001 "pip"
            # on gold at ~$2,700 never fires) — use percent-of-price.
            return entry_price * self.profile.min_sl_pct_metals
        if symbol and self._is_forex_symbol(symbol):
            return self.profile.min_sl_pips * self._get_forex_pip_size(symbol)
        return entry_price * self.profile.min_sl_pct

    def _evaluate_dynamic_gates(
        self,
        entry_price: float,
        sl_dist: float,
        symbol: Optional[str],
        spread: Optional[float],
        spread_fresh: bool,
        regime_series: Sequence[float],
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Coupled hybrid floor. Returns the gate that vetoes (GATE_REGIME /
        GATE_SPREAD) or GATE_NONE. Gate B is checked first; both are
        independently kill-switchable.
        """
        p = self.profile

        # ── Gate C: time-of-day blackout (e.g. NY 5pm rollover blowout) ──
        # Checked first and independent of vol warmth — the spread is toxic
        # regardless of regime, so we never want to trade this window.
        if p.time_gate_enabled and timestamp is not None and self._in_blackout(timestamp):
            logger.info(
                "[%s] Gate C (time) veto: signal inside NY blackout %s–%s ET",
                symbol or "unknown", p.blackout_start, p.blackout_end,
            )
            return GATE_TIME

        arr = np.asarray(regime_series, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = len(arr)
        if n == 0:
            return GATE_NONE
        current = float(arr[-1])

        # Percentile rank of the current bar's vol within its window.
        # Neutral (0.5) until the window is warm so the gates don't fire on a
        # cold deque (Gate B bypassed; Gate A runs decoupled at k_base).
        warm = n >= p.regime_min_samples
        pctile_rank = float(np.mean(arr <= current)) if warm else 0.5

        # ── Gate B: low-volatility regime ──
        if warm and p.regime_gate_enabled and pctile_rank < (p.regime_pctile / 100.0):
            logger.info(
                "[%s] Gate B (regime) veto: natr=%.5f rank=%.2f < P%.0f%% (window=%d)",
                symbol or "unknown", current, pctile_rank, p.regime_pctile, n,
            )
            return GATE_REGIME

        # ── Gate A: transaction-cost floor (coupled spread multiplier) ──
        if p.spread_gate_enabled:
            k_eff = float(
                coupled_keff(
                    p.spread_k_base, p.spread_k_coupling,
                    p.spread_k_coupling_mode, pctile_rank,
                )
            )
            if spread is not None and spread_fresh and spread > 0.0:
                spread_proxy, src = float(spread), "live"
            else:
                # Volatility-scaled proxy (matches the training-side proxy):
                # baseline ATR (median of window) converted from NATR% to price.
                # Per-instrument measured alpha when the model dir shipped a
                # spread table; flat profile value otherwise.
                alpha = self._alpha_overrides.get(symbol, p.spread_atr_alpha)
                baseline_atr_abs = float(np.median(arr)) * entry_price / 100.0
                spread_proxy = alpha * baseline_atr_abs
                src = (
                    f"proxy/alpha={alpha:.3f}"
                    if symbol in self._alpha_overrides
                    else "proxy"
                )
            floor = k_eff * spread_proxy
            if sl_dist < floor:
                logger.info(
                    "[%s] Gate A (cost/%s) veto: sl_dist=%.6f < k_eff(%.2f)·spread(%.6f)=%.6f",
                    symbol or "unknown", src, sl_dist, k_eff, spread_proxy, floor,
                )
                return GATE_SPREAD

        return GATE_NONE

    def _in_blackout(self, timestamp: datetime) -> bool:
        """True if ``timestamp`` is inside the America/New_York blackout window.

        DST-correct: the window is defined in NY local time, so it tracks the
        5pm rollover whether that is 21:00 UTC (summer) or 22:00 UTC (winter).
        Naive timestamps are assumed UTC.
        """
        p = self.profile
        if p.blackout_start is None or p.blackout_end is None or _NY_TZ is None:
            return False
        ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        ny = ts.astimezone(_NY_TZ).time()
        start, end = p.blackout_start, p.blackout_end
        if start <= end:
            return start <= ny < end
        return ny >= start or ny < end  # window wraps midnight

    def _is_forex_symbol(self, symbol: str) -> bool:
        clean = symbol.replace("_", "").replace("/", "").upper()
        return len(clean) == 6 and clean.isalpha()

    def _is_metal_symbol(self, symbol: str) -> bool:
        clean = symbol.replace("_", "").replace("/", "").upper()
        return len(clean) == 6 and clean[:3] in _METAL_BASES

    def _get_forex_pip_size(self, symbol: str) -> float:
        clean = symbol.replace("_", "").replace("/", "").upper()
        quote = clean[-3:]
        if quote == "JPY":
            return 0.01
        return 0.0001

    def calculate_quantity(
        self,
        equity: float,
        buying_power: float,
        entry_price: float,
        sl_price: float,
        cash: float = 0.0,
        is_crypto: bool = False,
    ) -> float:
        """
        Calculates fractional position size based on risk-per-trade.

        For crypto, uses cash * 0.95 as the buying-power cap (Alpaca reports
        crypto available funds in the cash field, not buying_power).
        Returns 0.0 if the resulting notional is below the $50 zombie-trade floor.
        """
        # 05192026: shouldn't apply to forex trades
        risk_dollars = equity * self.profile.risk_per_trade
        risk_per_share = entry_price - sl_price

        if risk_per_share <= 0:
            return 0.0

        risk_qty = risk_dollars / risk_per_share
        notional_qty = self.profile.max_notional_cap / entry_price
        bp_source = cash if is_crypto else buying_power
        bp_qty = (bp_source * 0.95) / entry_price

        final_qty = min(risk_qty, notional_qty, bp_qty)

        if final_qty < risk_qty:
            logger.warning(
                f"Quantity scaled down from {risk_qty:.4f} to {final_qty:.4f} to meet notional/bp limits."
            )

        # $50 minimum notional — prevents zombie fractional-share trades
        if final_qty * entry_price < 50.0:
            return 0.0

        return max(round(final_qty, 4), 0.0001)
