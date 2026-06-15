import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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

# Coupling modes (the two competing financial theses, soak-selectable).
COUPLING_TIGHTEN = "tighten"  # Claude: more cost discipline as vol expands
COUPLING_LOOSEN = "loosen"    # Gemini: relax cost discipline in high-momentum runs

# Veto-gate identifiers for split telemetry.
GATE_NONE = "none"
GATE_SPREAD = "spread"  # Gate A — transaction-cost floor
GATE_REGIME = "regime"  # Gate B — low-volatility regime floor
GATE_STATIC = "static"  # legacy static floor (equities / no regime context)

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

    @classmethod
    def for_asset_class(cls, asset_class: str) -> "RiskProfile":
        if asset_class == "forex":
            return cls(
                sl_atr_multiplier=1.0,
                tp_atr_multiplier=2.0,
                min_sl_pips=float(os.getenv(ENV_FOREX_MIN_SL_PIPS, "2.0")),
                min_sl_pct_metals=float(os.getenv(ENV_METALS_MIN_SL_PCT, "0.0001")),
                round_precision=5,
                spread_k_base=float(os.getenv(ENV_SPREAD_K, "1.5")),
                spread_k_coupling=float(os.getenv(ENV_SPREAD_K_COUPLING, "0.0")),
                spread_k_coupling_mode=os.getenv(
                    ENV_COUPLING_MODE, COUPLING_TIGHTEN
                ).strip().lower(),
                regime_pctile=float(os.getenv(ENV_REGIME_PCTILE, "20.0")),
                regime_window=int(os.getenv(ENV_REGIME_WINDOW, "260")),
                regime_min_samples=int(os.getenv(ENV_REGIME_MIN_SAMPLES, "60")),
                spread_atr_alpha=float(os.getenv(ENV_SPREAD_ATR_ALPHA, "0.15")),
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

class RiskManager:
    """
    The Shield: Enforces institutional-grade safety nets and dynamic sizing.
    """
    def __init__(self, profile: RiskProfile = RiskProfile()):
        self.profile = profile
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
                entry_price, sl_dist, symbol, spread, spread_fresh, regime_series
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
    ) -> str:
        """
        Coupled hybrid floor. Returns the gate that vetoes (GATE_REGIME /
        GATE_SPREAD) or GATE_NONE. Gate B is checked first; both are
        independently kill-switchable.
        """
        p = self.profile
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
                baseline_atr_abs = float(np.median(arr)) * entry_price / 100.0
                spread_proxy, src = p.spread_atr_alpha * baseline_atr_abs, "proxy"
            floor = k_eff * spread_proxy
            if sl_dist < floor:
                logger.info(
                    "[%s] Gate A (cost/%s) veto: sl_dist=%.6f < k_eff(%.2f)·spread(%.6f)=%.6f",
                    symbol or "unknown", src, sl_dist, k_eff, spread_proxy, floor,
                )
                return GATE_SPREAD

        return GATE_NONE

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
