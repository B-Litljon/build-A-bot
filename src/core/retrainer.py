"""
The Cure V2 - Validated Model Retraining Pipeline.

Triggered when feedback_loop.py detects critical model drift.
Fetches fresh data, engineers ATR-dynamic labels, runs a 3-fold
walk-forward validation gate, and only promotes models that prove
profitability across multiple chronological market regimes.

Usage:
    python -m src.core.retrainer

Exit Codes:
    0 = Models promoted successfully
    1 = Execution error (data fetch failed, config error, etc.)
    2 = Models rejected by validation gate (production weights retained)

    NOTE: run_pipeline.sh checks feedback_loop.py's exit code (2 = trigger
    retraining), NOT this retrainer's exit code. Exit code 2 from the
    retrainer means "tried but rejected" and does not cause an infinite loop.

Environment Variables:
    ALPACA_API_KEY: Alpaca API key
    ALPACA_SECRET_KEY: Alpaca API secret
    DISCORD_WEBHOOK_URL: Discord webhook for retraining reports (optional)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit

from src.data.factory import get_market_provider
from src.data.market_provider import MarketDataProvider
from src.execution.risk_manager import (
    RiskProfile,
    _chop_filter_enabled,
    coupled_keff,
)
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.features.v3_features import V3BaseFeatures, V3HTFFeatures, V3SessionFeatures
from src.ml.regimes.hmm_regime import (
    HMM_OUTPUT_COLS,
    fit_regime_models,
    predict_regime_probs,
    save_hmm_models,
)
from src.core.notification_manager import NotificationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DAYS_BACK = int(os.getenv("RETRAIN_DAYS_BACK", "60"))
# Forex basket pivoted 2026-05-23 from G7 majors (failed integrity gate, NO
# SIGNAL separation) to a volatility-first basket: two liquid metals plus
# three JPY/AUD-crossing pairs known for wide intraday ranges. XPT/XPD
# skipped — too illiquid on OANDA for scalping.
_DEFAULT_TICKERS_BY_CLASS = {
    "forex": [
        "XAU_USD", "XAG_USD",                        # liquid metals
        "GBP_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY",  # JPY-cross volatility
        "GBP_AUD", "GBP_NZD",                        # commonwealth crosses
    ],
    "equities": ["TSLA", "NVDA", "MARA", "COIN", "SMCI"],
}

def _asset_class_for_source(data_source: str) -> str:
    if data_source == "oanda":
        return "forex"
    return "equities"

def get_asset_config(data_source: str) -> dict:
    """Get dynamic configurations for retraining based on the data source."""
    asset_class = _asset_class_for_source(data_source)
    profile = RiskProfile.for_asset_class(asset_class)
    default_tickers = _DEFAULT_TICKERS_BY_CLASS[asset_class]
    
    # Asset-class specific overrides
    if asset_class == "forex":
        max_hold = 45
        timeframe = 1
        htf_timeframe = "5m"
    else:
        max_hold = 45
        timeframe = 1
        htf_timeframe = "5m"
        
    return {
        "asset_class": asset_class,
        "tickers": [t.strip() for t in os.getenv("RETRAIN_SYMBOLS", ",".join(default_tickers)).split(",") if t.strip()],
        "sl_mult": profile.sl_atr_multiplier,
        "tp_mult": profile.tp_atr_multiplier,
        "max_hold": int(os.getenv("RETRAIN_MAX_HOLD", str(max_hold))),
        "survival_bars": int(os.getenv("RETRAIN_SURVIVAL", "5")),
        "timeframe_minutes": int(os.getenv("RETRAIN_TIMEFRAME_MINUTES", str(timeframe))),
        "htf_timeframe": os.getenv("RETRAIN_HTF_TIMEFRAME", htf_timeframe),
    }

# Model Hyperparameters
def get_hyperparameters(asset_class: str) -> Tuple[dict, dict]:
    """
    Get hyperparameter configurations for Angel and Devil LightGBM models.

    Translated from the prior RandomForestClassifier dicts on 2026-05-23 as part
    of the LightGBM-pilot experiment:
      n_estimators 100 → 200 (paired with learning_rate=0.05 — boosting needs
        more rounds to reach RF-equivalent capacity).
      max_depth 10/8 retained as a *cap* on tree depth; num_leaves chosen below
        the 2^max_depth ceiling to keep effective complexity in RF's vicinity.
      min_samples_leaf 20 → min_child_samples 20 (direct equivalent).
      subsample/colsample_bytree 0.8 = LightGBM-native generalization knobs
        (RF gets the same effect for free via bootstrap sampling).
      verbose=-1 silences LightGBM's per-iter chatter so the gate diagnostic
        log stays readable.
    """
    angel_params = {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 10,
        "num_leaves": 63,
        "min_child_samples": 20 if asset_class == "forex" else 50,
        "class_weight": None if asset_class == "forex" else "balanced",
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    devil_params = {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 31,
        "min_child_samples": 20 if asset_class == "forex" else 50,
        "class_weight": None,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    return angel_params, devil_params

# Fallback constants for backward compatibility
ANGEL_PARAMS, DEVIL_PARAMS = get_hyperparameters("equities")


# ═══════════════════════════════════════════════════════════════════════════════
# ATR BRACKET PARAMETERS (must match evaluate_performance.py and LiveOrchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

SL_ATR_MULTIPLIER = 0.5
TP_ATR_MULTIPLIER = 3.0
MAX_HOLD_BARS = 45
SURVIVAL_BARS = 5  # Phase 5.5: Devil survival window (bars)

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE THRESHOLDS (must match MLStrategy)
# ═══════════════════════════════════════════════════════════════════════════════

ANGEL_THRESHOLD = 0.40
# Legacy: used as a fallback. In validate_candidate(), the Devil threshold
# is dynamically selected per-fold via _find_optimal_threshold().
DEVIL_THRESHOLD = 0.50

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION GATE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

BRIER_THRESHOLD = 0.30  # Phase 5.5: raised from 0.25 — survival target base rate
# ~45% shifts naive-classifier Brier to ~0.25, so 0.25
# was a false-rejection boundary. 0.30 rejects only
# genuinely uncalibrated models.
EV_THRESHOLD = 0.0005  # Min acceptable Expected Value
PROFIT_FACTOR_THRESHOLD = 1.2  # Min acceptable Profit Factor
# Sample-size floor for Fold 3 OOS PF: small trade counts under high R:R
# (TP=3.0×ATR / SL=0.5×ATR ⇒ 6:1 payoff) produce false-positive gate passes
# from a handful of lucky wins. Empirically the 2026-05-23 Gemini run "passed"
# with 14–16 OOS trades while the Devil's own separation diagnostic read
# "NO SIGNAL." 100 is the minimum sample for any honest PF claim.
MIN_OOS_TRADES_FOR_PF = 100

# Toggle for the HMM regime-feature experiment. When enabled, a per-symbol
# 3-state GaussianHMM is fit on each fold's training window (no leakage) and
# its posterior state probabilities are appended as 3 additional features.
# Default off so a plain LightGBM swap can be evaluated without confounds.
USE_HMM_FEATURES = os.getenv("RETRAIN_USE_HMM", "0").strip() == "1"

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMNS (must match MLStrategy.feature_names and FeaturePipeline output)
# ═══════════════════════════════════════════════════════════════════════════════

# Base features produced by FeaturePipeline. The HMM regime features (when
# enabled) are appended downstream in validate_candidate() because their
# fitting must respect each fold's temporal boundary.
BASE_FEATURE_COLS: List[str] = [
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
    # V3.3: Multi-timeframe (5m) features
    "htf_rsi_14",
    "htf_trend_agreement",
    "htf_vol_rel",
    "htf_bb_pct_b",
    # V3.4 Phase 5: Microstructure features (stop-hunt defense)
    "range_coil_10",
    "bar_body_pct",
    "bar_upper_wick_pct",
    "bar_lower_wick_pct",
    # V3.5 (2026-05-23): UTC session-activity indicators — tame G7 + XAU
    # failed at M1 with the 18-feature vector; sessions condition the model
    # on activity regime (London/NY overlap = volatility sweet spot).
    "session_asia",
    "session_london",
    "session_ny",
    "session_overlap",
]

FEATURE_COLS: List[str] = (
    BASE_FEATURE_COLS + HMM_OUTPUT_COLS if USE_HMM_FEATURES else BASE_FEATURE_COLS
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward CV fold."""

    fold_number: int
    train_size: int
    val_size: int
    brier_score: float
    expected_value: float
    angel_proposed_trades: int
    devil_approved_trades: int
    win_rate: float


@dataclass
class ValidationReport:
    """Aggregated validation report across all walk-forward folds."""

    fold_metrics: List[FoldMetrics]
    mean_brier: float
    mean_ev: float
    final_profit_factor: float
    final_win_rate: float
    final_total_trades: int
    gate_passed: bool
    rejection_reasons: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# ALPACA DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_training_data(
    provider: MarketDataProvider,
    symbols: List[str],
    days_back: int = DAYS_BACK,
    timeframe_minutes: int = 1,
) -> pl.DataFrame:
    """
    Fetch historical 1-minute bars for training using the unified provider.

    Args:
        provider: MarketDataProvider instance
        symbols: List of symbols to fetch
        days_back: Number of days to fetch (default: 60)

    Returns:
        Polars DataFrame with OHLCV data for all symbols
    """
    logger.info("=" * 70)
    logger.info("FETCHING TRAINING DATA")
    logger.info("=" * 70)

    # Use timezone-aware UTC datetime. RETRAIN_END_DATE (YYYY-MM-DD) lets
    # us shift the window backward for soak-readiness reruns — the audit
    # report flagged single-window results as a deployment risk.
    end_override = os.getenv("RETRAIN_END_DATE", "").strip()
    if end_override:
        end_date = datetime.strptime(end_override, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        logger.info(f"RETRAIN_END_DATE override active: {end_date.date()}")
    else:
        end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Timeframe: 1-minute bars")

    all_frames: List[pl.DataFrame] = []

    for ticker in symbols:
        try:
            # Fetch bars using generic provider
            df = provider.get_historical_bars(
                symbol=ticker,
                timeframe_minutes=timeframe_minutes,
                start=start_date,
                end=end_date,
            )

            if df is None or df.is_empty():
                logger.warning(f"No data returned for {ticker}")
                continue

            # Ensure column names are lowercase
            df.columns = [col.lower() for col in df.columns]

            # Add symbol column if not present
            if "symbol" not in df.columns:
                df = df.with_columns(pl.lit(ticker).alias("symbol"))

            all_frames.append(df)
            logger.info(f"Fetched {len(df):,} bars for {ticker}")

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            continue

    if not all_frames:
        raise ValueError("No data fetched for any symbol")

    # Combine all symbols
    combined = pl.concat(all_frames, how="vertical_relaxed")
    combined = combined.sort(["symbol", "timestamp"])

    logger.info(f"Combined dataset: {len(combined):,} total rows")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# ATR-DYNAMIC DEVIL TARGET (Bar-by-Bar Bracket Simulation)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_devil_targets_atr(
    df: pl.DataFrame,
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    max_hold: int = MAX_HOLD_BARS,
) -> np.ndarray:
    """
    Compute Devil targets using dynamic ATR brackets with bar-by-bar resolution.

    For each bar i, simulates a bracket order:
        SL = close[i] - sl_mult * ATR_abs[i]
        TP = close[i] + tp_mult * ATR_abs[i]

    Then walks forward up to max_hold bars checking:
        - If low[j] <= SL → loss (0)
        - If high[j] >= TP → win (1)
        - SL is checked FIRST (conservative, matches evaluate_performance.py)
        - If neither hit in max_hold bars → loss (0, timeout)

    This avoids rolling max/min which does not respect the temporal ordering
    of SL vs TP hits.  Complexity: O(n × max_hold).  At 60 days × 5 tickers
    × ~390 bars/day ≈ 117k rows × 15 bars = ~1.75M iterations — runs in
    under 2 seconds on modern hardware.

    Args:
        df: DataFrame containing 'close', 'high', 'low', 'natr_14' columns.
        sl_mult: ATR multiplier for stop-loss (default: SL_ATR_MULTIPLIER).
        tp_mult: ATR multiplier for take-profit (default: TP_ATR_MULTIPLIER).
        max_hold: Maximum bars to hold before timeout (default: MAX_HOLD_BARS).

    Returns:
        NumPy array of int8 (0 = loss/timeout, 1 = win), same length as df.
        NaN/invalid entries at the tail are set to 0.
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    natr = df["natr_14"].to_numpy()
    symbol = df["symbol"].to_numpy() if "symbol" in df.columns else np.array([""] * len(close))
    n = len(close)
    targets = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        atr_abs = close[i] * natr[i] / 100.0
        if np.isnan(atr_abs) or atr_abs <= 0:
            continue

        sl_price = close[i] - sl_mult * atr_abs
        tp_price = close[i] + tp_mult * atr_abs

        for j in range(i + 1, min(i + max_hold + 1, n)):
            if symbol[j] != symbol[i]:
                break
            # SL checked first (conservative — matches evaluate_performance.py)
            if low[j] <= sl_price:
                targets[i] = 0
                break
            if high[j] >= tp_price:
                targets[i] = 1
                break
        # If loop completes without break → timeout → 0 (already default)

    return targets


def _compute_devil_survival_target(
    df: pl.DataFrame,
    sl_mult: float = SL_ATR_MULTIPLIER,
    survival_bars: int = SURVIVAL_BARS,
) -> np.ndarray:
    """
    Compute Devil survival targets: whether price survives the SL for the
    next `survival_bars` bars after each row.

    Phase 5.5 — Temporal Realignment:
        The Devil's 1m microstructure features (wick toxicity, range
        compression) operate at a 1–5 minute horizon.  Asking the Devil
        to predict 45-bar macro outcomes (the old devil_target) creates
        an unlearnable temporal gap.  Asking it to predict 5-bar SL
        survival aligns the learning objective with the feature horizon.

    Survival definition:
        target[i] = 1  if  low[j] > SL_price  for ALL j in [i+1, i+SURVIVAL_BARS]
        target[i] = 0  if  low[j] <= SL_price  for ANY j in that window

    SL price is computed identically to the live bracket:
        SL = close[i] - sl_mult * ATR_abs[i]
        ATR_abs = close[i] * natr_14[i] / 100.0

    Args:
        df:             DataFrame with 'close', 'low', 'natr_14' columns.
        sl_mult:        ATR multiplier for stop-loss (default: SL_ATR_MULTIPLIER).
        survival_bars:  Number of bars to check for SL breach (default: SURVIVAL_BARS).

    Returns:
        NumPy int8 array of length len(df).
        1 = survived (no SL breach in window), 0 = stopped out.
        Last `survival_bars` rows are always 0 (insufficient lookahead).
    """
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    natr = df["natr_14"].to_numpy()
    symbol = df["symbol"].to_numpy() if "symbol" in df.columns else np.array([""] * len(close))
    n = len(close)
    targets = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        # Insufficient lookahead safety: check if symbol changes before survival window completes
        if i + survival_bars >= n or symbol[i + survival_bars] != symbol[i]:
            continue  # leaves targets[i] = 0 (default)

        atr_abs = close[i] * natr[i] / 100.0
        if np.isnan(atr_abs) or atr_abs <= 0:
            continue

        sl_price = close[i] - sl_mult * atr_abs
        survived = True

        for j in range(i + 1, min(i + survival_bars + 1, n)):
            if symbol[j] != symbol[i]:
                survived = False
                break
            if low[j] <= sl_price:
                survived = False
                break

        targets[i] = np.int8(1) if survived else np.int8(0)

    return targets


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID CHOP VETO (symmetric with live RiskManager.calculate_bracket)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_chop_veto_mask(
    df: pl.DataFrame, profile: RiskProfile, sl_mult: float
) -> np.ndarray:
    """
    Vectorized hybrid chop veto, mirroring ``RiskManager._evaluate_dynamic_gates``
    so the model trains only on the live-tradeable population.

    For each row, using the trailing ``regime_window`` of ``natr_14`` per symbol:
      * pctile_rank = fraction of the window <= the current bar's NATR
      * Gate B (regime): veto if ``pctile_rank < regime_pctile/100``
      * Gate A (cost): veto if ``sl_mult·natr < k_eff · alpha · baseline_natr``
        (the live inequality ``sl_dist < k_eff·spread_proxy`` with the
        volatility-scaled proxy; ``close`` cancels on both sides). The spread
        proxy scales with each era's *baseline* (median-window) volatility —
        not a static historical constant — so it is era-robust.

    Returns a boolean array (True = veto/drop) aligned to ``df`` rows. Rows are
    dropped only as trade *entries*; the bracket walk in the target functions
    still sees the full contiguous price path (so this must run AFTER target
    generation, not before).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n_total = df.height
    veto = np.zeros(n_total, dtype=bool)
    if not _chop_filter_enabled() or n_total == 0:
        return veto

    w = int(profile.regime_window)
    mins = int(profile.regime_min_samples)
    p_thresh = profile.regime_pctile / 100.0
    alpha = profile.spread_atr_alpha

    symbols = df["symbol"].to_numpy() if "symbol" in df.columns else np.zeros(n_total)
    natr_all = df["natr_14"].to_numpy().astype(float)

    for sym in np.unique(symbols):
        idx = np.where(symbols == sym)[0]  # contiguous, time-ordered per symbol
        natr = natr_all[idx]
        m = len(natr)
        # rank_actual feeds Gate B (regime); rank_eff feeds the coupling and is
        # held neutral (0.5) until the window is warm — exactly as the live gate
        # holds pctile_rank=0.5 below regime_min_samples.
        rank_actual = np.full(m, 0.5)
        rank_eff = np.full(m, 0.5)
        baseline = np.full(m, np.nan)  # expanding/rolling median of the window

        # Full-window region (vectorized): rows i >= w-1 (always warm: w >= mins).
        if m >= w:
            sw = sliding_window_view(natr, w)  # (m-w+1, w) → rows w-1 .. m-1
            last = sw[:, -1]
            fr = (sw <= last[:, None]).mean(axis=1)
            rank_actual[w - 1:] = fr
            rank_eff[w - 1:] = fr
            baseline[w - 1:] = np.median(sw, axis=1)

        # Expanding region (all earlier rows): baseline is always computable, so
        # Gate A (cost) runs from the first bar; Gate B only once warm.
        for i in range(0, min(w - 1, m)):
            win = natr[: i + 1]
            rank_actual[i] = float(np.mean(win <= natr[i]))
            baseline[i] = float(np.median(win))
            if (i + 1) >= mins:  # warm → real rank couples; else stay neutral 0.5
                rank_eff[i] = rank_actual[i]

        warm = (np.arange(m) + 1) >= mins

        gate_b = np.zeros(m, dtype=bool)
        if profile.regime_gate_enabled:
            gate_b = warm & (rank_actual < p_thresh)

        gate_a = np.zeros(m, dtype=bool)
        if profile.spread_gate_enabled:
            k_eff = coupled_keff(
                profile.spread_k_base, profile.spread_k_coupling,
                profile.spread_k_coupling_mode, rank_eff,
            )
            with np.errstate(invalid="ignore"):
                gate_a = (sl_mult * natr) < (k_eff * alpha * baseline)
            gate_a &= np.isfinite(baseline)

        veto[idx] = gate_a | gate_b

    return veto


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & LABEL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def engineer_features_and_labels(
    df: pl.DataFrame,
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    max_hold: int = MAX_HOLD_BARS,
    survival_bars: int = SURVIVAL_BARS,
    htf_timeframe: str = "5m",
    risk_profile: Optional[RiskProfile] = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Engineer technical features and generate ATR-dynamic target labels.

    Delegates feature computation to FeaturePipeline to
    guarantee zero training/inference skew with the production MLStrategy.

    Features (produced by FeaturePipeline, matches MLStrategy.feature_names):
        rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct,
        price_sma50_ratio, log_return, hour_of_day, dist_sma50, vol_rel

    Targets:
        angel_target: 1 if close 3 bars ahead > close + sl_mult × ATR (ATR-relative)
        devil_target: 1 if TP (tp_mult × ATR) hit before SL (sl_mult × ATR) in ≤max_hold bars

    Args:
        df: Raw OHLCV DataFrame with columns:
            open, high, low, close, volume, symbol, timestamp
        sl_mult: Stop-loss ATR multiplier
        tp_mult: Take-profit ATR multiplier
        max_hold: Maximum hold bars
        survival_bars: Number of survival bars for devil training
        htf_timeframe: Higher timeframe representation for HTFFeatures

    Returns:
        Tuple of (features DataFrame with targets, feature column names)
    """
    logger.info("=" * 70)
    logger.info("ENGINEERING FEATURES & LABELS")
    logger.info("=" * 70)

    # ═══════════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS via FeaturePipeline (prevents training/inference skew)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("Computing indicators via FeaturePipeline (zero-skew pipeline)...")
    pipeline = FeaturePipeline(
        feature_generators=[
            V3BaseFeatures(),
            V3HTFFeatures(timeframe=htf_timeframe),
            V3SessionFeatures(),
        ]
    )
    for gen in pipeline.feature_generators:
        df = gen.generate(df)
    logger.info(
        "Applied indicators: RSI, PPO, NATR, BBANDS, SMA50, log_return, "
        "hour_of_day, vol_rel"
    )

    # ═══════════════════════════════════════════════════════════════════
    # ANGEL TARGET: ATR-relative 3-bar momentum
    # 1 if close 3 bars ahead > close + sl_mult × ATR_abs
    # natr_14 is a percentage: ATR_abs = close * natr_14 / 100
    # ═══════════════════════════════════════════════════════════════════
    df = df.with_columns(
        (
            pl.col("close").shift(-3).over("symbol")
            > pl.col("close") + sl_mult * (pl.col("close") * pl.col("natr_14") / 100.0)
        )
        .cast(pl.Int8)
        .alias("angel_target")
    )
    logger.info(f"Generated angel_target (ATR-relative 3-bar momentum with sl_mult={sl_mult})")

    # ═══════════════════════════════════════════════════════════════════
    # DEVIL TARGETS (Phase 5.5 — Two-Target Architecture)
    #
    # devil_target_macro  — max_hold-bar bracket outcome (TP hit before SL).
    #   Used ONLY during threshold calibration (_find_optimal_threshold).
    #   Computes realized EV on Devil-approved trades using the actual
    #   asymmetric R:R payload (sl_mult × SL / tp_mult × TP).
    #
    # devil_target        — survival_bars-bar SL survival.
    #   Used to TRAIN the Devil. Aligns the learning objective with the
    #   1m microstructure feature horizon.
    #   1 = price did NOT breach SL in the next survival_bars bars (survived)
    #   0 = price breached SL within survival_bars bars (stopped out immediately)
    # ═══════════════════════════════════════════════════════════════════

    # -- Macro target (max_hold-bar) — evaluation only -----------------------
    logger.info(
        f"Computing devil_target_macro via ATR bracket simulation "
        f"(SL={sl_mult}×ATR, TP={tp_mult}×ATR, "
        f"max_hold={max_hold} bars)..."
    )
    devil_targets_macro = _compute_devil_targets_atr(df, sl_mult=sl_mult, tp_mult=tp_mult, max_hold=max_hold)
    df = df.with_columns(pl.Series("devil_target_macro", devil_targets_macro))
    logger.info(
        f"Generated devil_target_macro ({max_hold}-bar bracket): "
        f"{int(devil_targets_macro.sum())} wins / {len(devil_targets_macro)} total "
        f"({devil_targets_macro.mean():.1%} macro win rate)"
    )

    # -- Survival target (survival_bars-bar) — Devil training ----------------------
    logger.info(
        f"Computing devil_target via {survival_bars}-bar SL survival "
        f"(SL={sl_mult}×ATR)..."
    )
    devil_targets_survival = _compute_devil_survival_target(df, sl_mult=sl_mult, survival_bars=survival_bars)
    df = df.with_columns(pl.Series("devil_target", devil_targets_survival))
    logger.info(
        f"Generated devil_target ({survival_bars}-bar survival): "
        f"{int(devil_targets_survival.sum())} survived / {len(devil_targets_survival)} total "
        f"({devil_targets_survival.mean():.1%} survival rate)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # HYBRID CHOP VETO — drop untradeable entry rows (symmetric with live)
    # Runs AFTER target generation so the bracket walk saw the full price
    # path; we only remove bars we would never ENTER on. Realigns the
    # training population (and thus Profit Factor) with the live filter.
    # ═══════════════════════════════════════════════════════════════════
    if risk_profile is not None:
        veto_mask = _compute_chop_veto_mask(df, risk_profile, sl_mult)
        n_veto = int(veto_mask.sum())
        if n_veto > 0:
            df = df.filter(~pl.Series(veto_mask))
        logger.info(
            f"Hybrid chop veto dropped {n_veto:,} untradeable rows "
            f"(mode={risk_profile.spread_k_coupling_mode}, "
            f"k_base={risk_profile.spread_k_base}, coupling={risk_profile.spread_k_coupling}, "
            f"P{risk_profile.regime_pctile:.0f}, alpha={risk_profile.spread_atr_alpha})"
        )

    # ═══════════════════════════════════════════════════════════════════
    # CLEANUP: Drop NaN/null rows (uses FeaturePipeline.clean_data)
    # ═══════════════════════════════════════════════════════════════════
    initial_count = len(df)
    # Clean on BASE features only — HMM regime probs (when enabled) are
    # appended later inside validate_candidate so each fold fits its own HMM.
    df = FeaturePipeline.clean_data(
        df, feature_cols=BASE_FEATURE_COLS + ["angel_target", "devil_target"]
    )
    dropped_count = initial_count - len(df)

    logger.info(
        f"Dropped {dropped_count:,} rows with nulls ({dropped_count / initial_count:.1%})"
    )
    logger.info(f"Final dataset: {len(df):,} rows")
    logger.info(f"Base feature columns ({len(BASE_FEATURE_COLS)}): {BASE_FEATURE_COLS}")
    if USE_HMM_FEATURES:
        logger.info(f"HMM regime features ENABLED — will be appended per-fold: {HMM_OUTPUT_COLS}")

    return df, BASE_FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-DECAY WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════


def generate_time_decay_weights(
    n_samples: int, decay_factor: float = 0.95
) -> np.ndarray:
    """
    Generate time-decay sample weights.

    More recent samples get higher weights to prevent catastrophic forgetting.
    Weights decay exponentially from 1.0 (most recent) to 0.1 (oldest).

    Args:
        n_samples: Number of samples in dataset
        decay_factor: Decay rate per time step (default: 0.95)

    Returns:
        NumPy array of sample weights
    """
    # Generate exponential decay from oldest to newest
    weights = np.power(decay_factor, np.arange(n_samples))

    # Reverse so newest has highest weight
    weights = weights[::-1]

    # Normalize to range [0.1, 1.0]
    weights = 0.1 + 0.9 * (weights - weights.min()) / (weights.max() - weights.min())

    return weights


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def refit_models(
    df: pl.DataFrame,
    feature_cols: List[str],
    angel_params: Optional[dict] = None,
    devil_params: Optional[dict] = None,
) -> Tuple["lgb.LGBMClassifier", "lgb.LGBMClassifier", List[str], List[str]]:
    """
    Refit Angel and Devil models with time-decay weighting.

    Implements proper Meta-Labeling architecture:
    1. Train Angel on base features
    2. Generate Angel's probabilities as meta-features
    3. Train Devil on base features + angel_prob

    Args:
        df: Feature-engineered DataFrame with 'angel_target' and 'devil_target'
        feature_cols: List of base feature column names
        angel_params: Hyperparameters for Angel classifier
        devil_params: Hyperparameters for Devil classifier

    Returns:
        Tuple of (Angel model, Devil model, angel_features, devil_features)
    """
    logger.info("=" * 70)
    logger.info("REFITTING MODELS (META-LABELING)")
    logger.info("=" * 70)

    # Resolve parameter configs
    a_params = angel_params if angel_params is not None else ANGEL_PARAMS
    d_params = devil_params if devil_params is not None else DEVIL_PARAMS

    # Extract base features and targets
    X_base = df[feature_cols].to_numpy()
    y_angel = df["angel_target"].to_numpy()
    y_devil = df["devil_target"].to_numpy()

    logger.info(f"Training samples: {len(X_base):,}")
    logger.info(f"Base features: {feature_cols}")
    logger.info(
        f"Angel target distribution: 0={np.sum(y_angel == 0)}, 1={np.sum(y_angel == 1)}"
    )
    logger.info(
        f"Devil target distribution: 0={np.sum(y_devil == 0)}, 1={np.sum(y_devil == 1)}"
    )

    # Generate time-decay weights
    sample_weights = generate_time_decay_weights(len(X_base))
    logger.info(
        f"Time-decay weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}"
    )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Train the Angel (Primary Model - Direction)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 1/4] Training Angel model (Direction)...")
    angel_model = lgb.LGBMClassifier(**a_params)
    df_base = pl.DataFrame(X_base, schema=feature_cols).to_pandas()
    angel_model.fit(df_base, y_angel, sample_weight=sample_weights)
    logger.info(f"✓ Angel model trained on {len(feature_cols)} features")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Generate Out-Of-Fold Meta-Features (Angel's Probabilities)
    # ═══════════════════════════════════════════════════════════════════
    logger.info(
        "\n[Step 2/4] Generating OOF meta-features (temporal cross-validation)..."
    )

    # CRITICAL FIX: Use TimeSeriesSplit with a manual fold loop to generate
    # Angel probabilities via out-of-fold predictions. This prevents the Devil
    # from training on the Angel's inflated in-sample confidence.
    #
    # Why manual loop instead of cross_val_predict:
    #   cross_val_predict requires that every sample appears in exactly one
    #   test fold (a strict partition). TimeSeriesSplit's first ~1/n_splits
    #   of samples are never in any test fold (always train-only). sklearn
    #   raises "cross_val_predict only works for partitions" in this case.
    #   The manual loop handles the train-only head by filling those rows
    #   from a model trained on just that first-fold window — still OOF
    #   for the rows that follow (zero leakage into the majority of data).
    #
    # Why TimeSeriesSplit: respects chronological ordering — each fold only
    # trains on past bars. KFold would let the Angel see future bars.
    #
    # n_splits=5: 5 expanding folds. Early folds → noisier Angel probs,
    # which is realistic (production Angel also starts uncertain).

    tss = TimeSeriesSplit(n_splits=5)
    angel_probs_oof = np.full(len(X_base), np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        for fold_train_idx, fold_val_idx in tss.split(X_base):
            fold_weights = sample_weights[fold_train_idx]
            fold_angel = lgb.LGBMClassifier(**a_params)
            fold_angel.fit(
                X_base[fold_train_idx],
                y_angel[fold_train_idx],
                sample_weight=fold_weights,
            )
            angel_probs_oof[fold_val_idx] = fold_angel.predict_proba(
                X_base[fold_val_idx]
            )[:, 1]

        # Fill train-only head (first ~1/n_splits rows never appear in val)
        # using a model trained solely on that window — no leakage from future.
        head_missing = np.isnan(angel_probs_oof)
        if head_missing.sum() > 0:
            first_train_idx, _ = next(iter(tss.split(X_base)))
            head_angel = lgb.LGBMClassifier(**a_params)
            head_angel.fit(
                X_base[first_train_idx],
                y_angel[first_train_idx],
                sample_weight=sample_weights[first_train_idx],
            )
            angel_probs_oof[head_missing] = head_angel.predict_proba(
                X_base[head_missing]
            )[:, 1]
            logger.info(
                f"  Head fill: {head_missing.sum()} train-only rows scored by "
                f"Fold-1 Angel (no leakage from future)"
            )

    # Add OOF angel_prob as a new column to the DataFrame
    df = df.with_columns(pl.Series("angel_prob", angel_probs_oof))

    logger.info(f"✓ Generated {len(angel_probs_oof):,} OOF Angel probabilities")
    logger.info(
        f"  OOF Angel prob range:  [{angel_probs_oof.min():.3f}, {angel_probs_oof.max():.3f}]"
    )
    logger.info(f"  OOF Angel prob median: {np.median(angel_probs_oof):.3f}")

    # Compare OOF vs in-sample to confirm leakage was present
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        angel_probs_insample = angel_model.predict_proba(X_base)[:, 1]

    logger.info(
        f"  In-sample Angel prob median: {np.median(angel_probs_insample):.3f} "
        f"(should be much higher than OOF — confirms leakage was present)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Train the Devil (Meta Model - Conviction)
    # Phase 5.5: Train ONLY on Angel-approved subpopulation.
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 3/4] Training Devil model (Conviction with meta-features)...")

    devil_features = feature_cols + ["angel_prob"]
    X_devil_full = df[devil_features].to_numpy()

    # Phase 5.5 — Population Fix:
    # The Devil is deployed exclusively on rows where angel_prob >= ANGEL_THRESHOLD.
    # Training on the full global population (all ~117k rows) violates meta-labeling
    # semantics: the Devil learns to discriminate across all market conditions, not
    # within the Angel-approved subset where it actually operates.
    # Solution: filter training data to only Angel-approved rows using the OOF
    # angel_probs (already computed in Step 2 — zero leakage).
    angel_approved_mask = angel_probs_oof >= ANGEL_THRESHOLD
    n_approved = int(angel_approved_mask.sum())
    n_total = len(X_devil_full)
    logger.info(
        f"Phase 5.5: Devil training filtered to Angel-approved subpopulation: "
        f"{n_approved:,} / {n_total:,} rows ({n_approved / n_total:.1%})"
    )

    X_devil = X_devil_full[angel_approved_mask]
    y_devil_train = y_devil[angel_approved_mask]
    devil_weights = sample_weights[angel_approved_mask]

    logger.info(
        f"Devil survival target distribution (approved rows): "
        f"survived={np.sum(y_devil_train == 1):,} | "
        f"stopped={np.sum(y_devil_train == 0):,} | "
        f"rate={np.mean(y_devil_train):.1%}"
    )
    logger.info(f"Devil feature space: {devil_features}")

    devil_model = lgb.LGBMClassifier(**d_params)
    df_devil = pl.DataFrame(X_devil, schema=devil_features).to_pandas()
    devil_model.fit(df_devil, y_devil_train, sample_weight=devil_weights)
    logger.info(
        f"✓ Devil model trained on {len(devil_features)} features "
        f"(Angel-approved subpopulation, n={n_approved:,})"
    )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Validation & Summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 4/4] Model validation...")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        angel_acc = angel_model.score(X_base, y_angel, sample_weight=sample_weights)
        devil_acc = devil_model.score(
            X_devil, y_devil_train, sample_weight=devil_weights
        )

    logger.info(f"\n{'=' * 70}")
    logger.info("META-LABELING TRAINING COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Angel training accuracy: {angel_acc:.3f} (recall-focused)")
    logger.info(f"Devil training accuracy: {devil_acc:.3f} (precision-focused)")
    logger.info(f"Devil can now veto Angel when angel_prob is misleading")

    return angel_model, devil_model, feature_cols, devil_features


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC THRESHOLD SELECTION
# ═══════════════════════════════════════════════════════════════════════════════


def _find_optimal_threshold(
    devil_probs: np.ndarray,
    survival_targets: np.ndarray,
    macro_targets: np.ndarray,
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    min_trades: int = 5,
) -> Tuple[float, float]:
    """
    Sweep thresholds to find the one that maximizes Expected Value.

    Phase 5.5 — Two-Target EV Calibration:
        The Devil is trained on `survival_targets` (5-bar SL survival).
        EV calibration must use `macro_targets` (45-bar bracket outcome) to
        reflect the actual asymmetric R:R payload delivered by the live system.

        Separating these two concerns is critical:
        - Using survival_targets for EV would compute "expected value of not
          getting stopped in 5 bars" — meaningless for bracket sizing.
        - Using macro_targets for training would reintroduce the temporal
          mismatch that caused the Devil to flatline.

    For each candidate threshold:
        1. Filter to approved trades: devil_prob >= threshold
        2. Compute realized win rate from MACRO outcomes on approved trades
        3. Compute EV = win_rate * (tp_mult / sl_mult) - (1 - win_rate)

    Args:
        devil_probs:      Array of Devil's predicted probabilities (survival).
        survival_targets: 5-bar SL survival ground truth (Devil's training target).
                          Passed for signature consistency; not used in EV sweep.
        macro_targets:    45-bar bracket outcome ground truth (0/1).
                          Used to compute realized win rate and EV.
        sl_mult:          Stop-loss ATR multiplier.
        tp_mult:          Take-profit ATR multiplier.
        min_trades:       Minimum approved trades for a threshold to be valid.

    Returns:
        Tuple of (optimal_threshold, best_ev)
    """
    thresholds = np.arange(0.10, 0.66, 0.02)  # 0.10, 0.12, ..., 0.64
    best_threshold = 0.20  # fallback default
    best_ev = -float("inf")

    for t in thresholds:
        mask = devil_probs >= t
        n_approved = int(mask.sum())

        if n_approved < min_trades:
            continue

        # EV is computed from MACRO outcomes (45-bar bracket), not survival.
        # This correctly prices the asymmetric R:R of the live bracket system.
        approved_macro = macro_targets[mask]
        win_rate = float(approved_macro.mean())

        # EV in R-multiples: wins pay (tp_mult / sl_mult) R, losses pay -1R
        rr_ratio = tp_mult / sl_mult
        ev = win_rate * rr_ratio - (1.0 - win_rate)

        if ev > best_ev:
            best_ev = ev
            best_threshold = float(t)

    return best_threshold, float(best_ev)


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION GATE
# ═══════════════════════════════════════════════════════════════════════════════


def validate_candidate(
    df: pl.DataFrame,
    feature_cols: List[str],
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    n_folds: int = 3,
    angel_params: Optional[dict] = None,
    devil_params: Optional[dict] = None,
) -> Tuple[
    ValidationReport,
    "lgb.LGBMClassifier",
    "lgb.LGBMClassifier",
    List[str],
    List[str],
    float,
    Optional[dict],
]:
    """
    Run expanding-window walk-forward cross-validation and apply the promotion gate.

    Splits the 60-day dataset into 3 expanding folds by calendar date (not row
    index) so that all symbols' data for a given date range stays in the same
    fold.  For each fold, trains Angel + Devil on the training window and
    evaluates strictly out-of-sample on the validation window.

    Fold schedule (calendar days from the earliest timestamp in df):
        Fold 1: Train days  0–29, Validate days 30–39
        Fold 2: Train days  0–39, Validate days 40–49
        Fold 3: Train days  0–49, Validate days 50–59  ← Profit Factor gate

    TEMPORAL BOUNDARY:
        The Profit Factor gate uses the Fold 3 model (trained on days 0–49)
        evaluated on the Fold 3 val set (days 50–59). This is strictly OOS.
        The final full-data model (all 60 days) is ONLY trained AFTER the gate
        passes. Training on all 60 days and then evaluating on a subset would
        be data leakage.

    Promotion thresholds:
        Mean Brier Score   ≤ 0.25   (across all folds)
        Mean EV            ≥ 0.0005 (across all folds)
        Profit Factor      ≥ 1.20   (Fold 3 val set only)

    Dynamic threshold:
        With class_weight=None, Devil probabilities reflect the true ~20% base
        rate. Per-fold, _find_optimal_threshold() sweeps 0.10–0.44 and selects
        the threshold maximizing EV. The Fold 3 threshold is returned as the
        production threshold.

    Args:
        df: Full 60-day feature-engineered DataFrame (output of
            engineer_features_and_labels).
        feature_cols: List of base feature column names.
        sl_mult: Stop-loss ATR multiplier
        tp_mult: Take-profit ATR multiplier
        n_folds: Number of expanding folds (default: 3).

    Returns:
        Tuple of:
            - ValidationReport (gate decision + per-fold metrics)
            - angel_model (full-data if gate passed, Fold 3 if rejected)
            - devil_model (full-data if gate passed, Fold 3 if rejected)
            - angel_feature_names
            - devil_feature_names
            - production_threshold (optimal Devil threshold from Fold 3)
    """
    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION (3 EXPANDING FOLDS)")
    logger.info("=" * 70)

    # If the HMM regime experiment is on, the active feature space is the
    # base features plus the 3 HMM_OUTPUT_COLS. Each fold fits its own HMM
    # on its training window only — no leakage from val into train.
    if USE_HMM_FEATURES:
        feature_cols = list(feature_cols) + list(HMM_OUTPUT_COLS)
        logger.info(
            "HMM regime features ENABLED — feature space expanded to %d cols: %s",
            len(feature_cols), feature_cols[-len(HMM_OUTPUT_COLS):],
        )

    # ───────────────────────────────────────────────────────────────────
    # Build date-based fold boundaries
    # ───────────────────────────────────────────────────────────────────
    min_date = df["timestamp"].min()

    # fold_configs: (train_end_days, val_end_days) — exclusive upper bounds.
    # Fractions chosen to reproduce the legacy 60-day schedule exactly:
    #   (30,40),(40,50),(50,60) at DAYS_BACK=60.
    # For larger windows the same expanding-train / fixed-fraction-val shape
    # scales (e.g. DAYS_BACK=180 → (90,120),(120,150),(150,180)). Without
    # this scaling the val sets stayed pinned to days 30–60 and 67% of any
    # extended window was silently discarded.
    fold_configs = [
        (DAYS_BACK // 2,     DAYS_BACK * 2 // 3),   # train 0–½,  val ½–⅔
        (DAYS_BACK * 2 // 3, DAYS_BACK * 5 // 6),   # train 0–⅔,  val ⅔–⅚
        (DAYS_BACK * 5 // 6, DAYS_BACK),            # train 0–⅚,  val ⅚–1
    ]

    fold_metrics: List[FoldMetrics] = []

    # Placeholders for Fold 3 outputs (used for PF gate and fallback)
    fold3_angel: Optional["lgb.LGBMClassifier"] = None
    fold3_devil: Optional["lgb.LGBMClassifier"] = None
    fold3_angel_feats: Optional[List[str]] = None
    fold3_devil_feats: Optional[List[str]] = None
    # Production HMM dict (fit on full data after the gate passes; persisted
    # alongside Angel/Devil and consumed at inference by MLStrategy).
    final_hmm_models: Optional[dict] = None
    profit_factor: float = 0.0
    final_win_rate: float = 0.0
    final_total_trades: int = 0
    production_threshold: float = 0.20  # fallback; overwritten by Fold 3
    # Fold n_folds-1 (the "calibration" fold) sweeps for an optimal threshold;
    # that threshold is frozen and applied to Fold n_folds for strict OOS gate
    # evaluation. Without this freeze, the threshold sweep on Fold 3 would
    # leak validation info into the production parameter — historically
    # surfaced as PF=3.3 on 16 trades with separation gap = -0.0092.
    calibration_threshold: Optional[float] = None

    for fold_idx, (train_end_day, val_end_day) in enumerate(fold_configs):
        fold_number = fold_idx + 1

        # Compute cutoff timestamps
        train_cutoff = min_date + timedelta(days=train_end_day)
        val_cutoff = min_date + timedelta(days=val_end_day)

        train_df = df.filter(pl.col("timestamp") < train_cutoff)
        val_df = df.filter(
            (pl.col("timestamp") >= train_cutoff) & (pl.col("timestamp") < val_cutoff)
        )

        logger.info(
            f"\n[Fold {fold_number}/{n_folds}] "
            f"Train: {len(train_df):,} rows | "
            f"Val: {len(val_df):,} rows"
        )

        # HMM regime augmentation — fit per-fold on train only, score both.
        # This ordering preserves the temporal boundary: nothing val-side ever
        # influences the HMM parameters.
        if USE_HMM_FEATURES and len(train_df) > 0 and len(val_df) > 0:
            fold_hmm_models = fit_regime_models(train_df)
            train_df = predict_regime_probs(train_df, fold_hmm_models)
            val_df = predict_regime_probs(val_df, fold_hmm_models)

        if len(train_df) == 0 or len(val_df) == 0:
            logger.warning(
                f"[Fold {fold_number}] Empty split — skipping. "
                f"Train={len(train_df)}, Val={len(val_df)}"
            )
            fold_metrics.append(
                FoldMetrics(
                    fold_number=fold_number,
                    train_size=len(train_df),
                    val_size=len(val_df),
                    brier_score=1.0,
                    expected_value=-1.0,
                    angel_proposed_trades=0,
                    devil_approved_trades=0,
                    win_rate=0.0,
                )
            )
            continue

        # ─────────────────────────────────────────────────────────────
        # Train on this fold's training window
        # ─────────────────────────────────────────────────────────────
        angel_model, devil_model, angel_feats, devil_feats = refit_models(
            train_df, feature_cols, angel_params=angel_params, devil_params=devil_params
        )

        # ─────────────────────────────────────────────────────────────
        # Score on validation window
        # ─────────────────────────────────────────────────────────────
        X_val_base = val_df[feature_cols].to_numpy()
        y_val_angel = val_df["angel_target"].to_numpy()
        y_val_devil = val_df["devil_target"].to_numpy()  # survival (5-bar)
        y_val_devil_macro = val_df["devil_target_macro"].to_numpy()  # macro (45-bar)

        # Stage 1: Angel inference
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            angel_probs_val = angel_model.predict_proba(X_val_base)[:, 1]

        signal_mask = angel_probs_val >= ANGEL_THRESHOLD
        n_angel_proposed = int(signal_mask.sum())
        logger.info(
            f"[Fold {fold_number}] Angel proposed {n_angel_proposed} trades "
            f"({n_angel_proposed / len(val_df):.1%} of val rows)"
        )

        if n_angel_proposed == 0:
            logger.warning(
                f"[Fold {fold_number}] Zero Angel-proposed trades — "
                f"setting worst-case metrics"
            )
            fold_metrics.append(
                FoldMetrics(
                    fold_number=fold_number,
                    train_size=len(train_df),
                    val_size=len(val_df),
                    brier_score=1.0,
                    expected_value=-1.0,
                    angel_proposed_trades=0,
                    devil_approved_trades=0,
                    win_rate=0.0,
                )
            )
            if fold_number == n_folds:
                fold3_angel, fold3_devil = angel_model, devil_model
                fold3_angel_feats, fold3_devil_feats = angel_feats, devil_feats
            continue

        # Stage 2: Devil inference on Angel-proposed rows
        proposed_base_feats = X_val_base[signal_mask]
        proposed_angel_probs = angel_probs_val[signal_mask]
        proposed_devil_targets = y_val_devil[signal_mask]  # survival — for Brier
        proposed_devil_targets_macro = y_val_devil_macro[signal_mask]  # macro — for EV

        meta_df = pl.DataFrame(proposed_base_feats, schema=feature_cols).with_columns(
            pl.Series("angel_prob", proposed_angel_probs)
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            devil_proba_full = devil_model.predict_proba(meta_df)

        # Defensive: if the Devil's training set was single-class (common on
        # tiny single-symbol windows where Angel approves <10 rows that all
        # survive or all stop), predict_proba returns shape (n, 1) and the
        # `[:, 1]` indexer raises IndexError. Treat the constant as 1.0 if
        # the only class seen was 1 (survived ⇒ no veto), else 0.0 (every
        # proposal vetoed).
        if devil_proba_full.shape[1] == 1:
            only_class = int(devil_model.classes_[0])
            const_prob = 1.0 if only_class == 1 else 0.0
            devil_probs_val = np.full(len(meta_df), const_prob, dtype=np.float64)
            logger.warning(
                f"[Fold {fold_number}] Devil trained on single-class data "
                f"(class={only_class}); using constant probability "
                f"{const_prob:.1f}. This fold's metrics are degenerate."
            )
        else:
            devil_probs_val = devil_proba_full[:, 1]

        # ═══════════════════════════════════════════════════════════════════
        # DEVIL DIAGNOSTIC: Probability Distribution Analysis
        # ═══════════════════════════════════════════════════════════════════
        if len(devil_probs_val) > 0:
            logger.info(f"\n{'─' * 60}")
            logger.info(f"DEVIL DIAGNOSTIC — Fold {fold_number}")
            logger.info(f"{'─' * 60}")

            # 1. Global Distribution
            logger.info(f"  Probability Distribution (n={len(devil_probs_val)}):")
            logger.info(f"    Min:    {np.min(devil_probs_val):.4f}")
            logger.info(f"    P25:    {np.percentile(devil_probs_val, 25):.4f}")
            logger.info(f"    Median: {np.median(devil_probs_val):.4f}")
            logger.info(f"    P75:    {np.percentile(devil_probs_val, 75):.4f}")
            logger.info(f"    Max:    {np.max(devil_probs_val):.4f}")

            # 2. Threshold Density
            n_total = len(devil_probs_val)
            logger.info(f"  Threshold Density:")
            logger.info(
                f"    Above 0.50: {(devil_probs_val >= 0.50).sum():>4d} / {n_total} ({(devil_probs_val >= 0.50).mean():.1%})"
            )
            logger.info(
                f"    Above 0.55: {(devil_probs_val >= 0.55).sum():>4d} / {n_total} ({(devil_probs_val >= 0.55).mean():.1%})"
            )
            logger.info(
                f"    Above 0.60: {(devil_probs_val >= 0.60).sum():>4d} / {n_total} ({(devil_probs_val >= 0.60).mean():.1%})"
            )
            logger.info(
                f"    Above 0.65: {(devil_probs_val >= 0.65).sum():>4d} / {n_total} ({(devil_probs_val >= 0.65).mean():.1%})"
            )
            logger.info(
                f"    Above 0.70: {(devil_probs_val >= 0.70).sum():>4d} / {n_total} ({(devil_probs_val >= 0.70).mean():.1%})"
            )

            # 3. Separation Check: Does the Devil actually distinguish wins from losses?
            # proposed_devil_targets is y_val_devil[signal_mask] — ground truth for Angel-proposed rows
            wins_mask = proposed_devil_targets == 1
            losses_mask = proposed_devil_targets == 0

            if wins_mask.sum() > 0 and losses_mask.sum() > 0:
                mean_prob_wins = devil_probs_val[wins_mask].mean()
                mean_prob_losses = devil_probs_val[losses_mask].mean()
                separation = mean_prob_wins - mean_prob_losses

                logger.info(f"  Separation Check:")
                logger.info(
                    f"    Mean prob (Actual Wins):   {mean_prob_wins:.4f}  (n={wins_mask.sum()})"
                )
                logger.info(
                    f"    Mean prob (Actual Losses): {mean_prob_losses:.4f}  (n={losses_mask.sum()})"
                )
                logger.info(f"    Separation Gap:            {separation:+.4f}")

                if separation > 0.05:
                    logger.info(
                        f"    Verdict: SIGNAL DETECTED -- Devil can distinguish (gap > 0.05)"
                    )
                elif separation > 0.02:
                    logger.info(
                        f"    Verdict: WEAK SIGNAL -- marginal separation (0.02 < gap < 0.05)"
                    )
                else:
                    logger.info(
                        f"    Verdict: NO SIGNAL -- Devil cannot distinguish wins from losses"
                    )
            else:
                logger.info(
                    f"  Separation Check: SKIPPED (wins={wins_mask.sum()}, losses={losses_mask.sum()})"
                )

            logger.info(f"{'─' * 60}\n")

        # ─────────────────────────────────────────────────────────────
        # Threshold selection — split strategy per fold:
        #   Fold 1 .. n_folds-1 : sweep for EV-maximising threshold (in-fold)
        #   Fold n_folds-1      : freeze that threshold as `calibration_threshold`
        #   Fold n_folds        : use the FROZEN calibration_threshold (no sweep)
        #
        # Why: sweeping the threshold on the same fold whose metrics gate
        # promotion leaks val data into the production parameter. Freezing
        # the threshold on the penultimate fold restores OOS purity.
        # ─────────────────────────────────────────────────────────────
        if fold_number < n_folds:
            optimal_threshold, fold_ev_at_threshold = _find_optimal_threshold(
                devil_probs=devil_probs_val,
                survival_targets=proposed_devil_targets,
                macro_targets=proposed_devil_targets_macro,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
            )
            logger.info(
                f"  [Fold {fold_number}] Swept threshold: {optimal_threshold:.2f} "
                f"(EV at threshold: {fold_ev_at_threshold:+.4f})"
            )
            if fold_number == n_folds - 1:
                calibration_threshold = optimal_threshold
                logger.info(
                    f"  [Fold {fold_number}] FROZEN as calibration_threshold "
                    f"for Fold {n_folds} strict-OOS evaluation"
                )
        else:
            # Fold n_folds: strict OOS — use frozen threshold from Fold n_folds-1
            if calibration_threshold is None:
                optimal_threshold = 0.50
                logger.warning(
                    f"  [Fold {fold_number}] No calibration_threshold available "
                    f"(Fold {n_folds - 1} had zero approvals) — "
                    f"falling back to {optimal_threshold:.2f}"
                )
            else:
                optimal_threshold = calibration_threshold
                logger.info(
                    f"  [Fold {fold_number}] Using FROZEN calibration_threshold: "
                    f"{optimal_threshold:.2f} (strict OOS — no threshold leakage)"
                )
            production_threshold = optimal_threshold
            logger.info(
                f"  Production threshold: {production_threshold:.2f}"
            )

        approved_mask = devil_probs_val >= optimal_threshold
        n_devil_approved = int(approved_mask.sum())
        logger.info(
            f"[Fold {fold_number}] Devil approved {n_devil_approved} trades "
            f"({n_devil_approved / max(n_angel_proposed, 1):.1%} of Angel proposals)"
        )

        if n_devil_approved == 0:
            logger.warning(
                f"[Fold {fold_number}] Zero Devil-approved trades — "
                f"setting worst-case metrics"
            )
            fold_metrics.append(
                FoldMetrics(
                    fold_number=fold_number,
                    train_size=len(train_df),
                    val_size=len(val_df),
                    brier_score=1.0,
                    expected_value=-1.0,
                    angel_proposed_trades=n_angel_proposed,
                    devil_approved_trades=0,
                    win_rate=0.0,
                )
            )
            if fold_number == n_folds:
                fold3_angel, fold3_devil = angel_model, devil_model
                fold3_angel_feats, fold3_devil_feats = angel_feats, devil_feats
            continue

        # ─────────────────────────────────────────────────────────────
        # Compute fold metrics on Devil-approved trades
        # ─────────────────────────────────────────────────────────────
        approved_devil_probs = devil_probs_val[approved_mask]
        approved_targets = proposed_devil_targets[approved_mask]

        brier = float(brier_score_loss(approved_targets, approved_devil_probs))
        win_rate = float(approved_targets.mean()) if len(approved_targets) > 0 else 0.0

        # EV using ATR R-multiple: wins = +tp_mult R, losses = -sl_mult R
        # Where R = 1 unit of sl_mult ATR
        # EV = win_rate * (tp_mult / sl_mult) - (1 - win_rate) * 1
        ev = float(
            win_rate * (tp_mult / sl_mult) - (1.0 - win_rate)
        )

        logger.info(
            f"[Fold {fold_number}] "
            f"Brier={brier:.4f} | EV={ev:.6f} | WR={win_rate:.1%} | "
            f"Trades={n_devil_approved}"
        )

        fm = FoldMetrics(
            fold_number=fold_number,
            train_size=len(train_df),
            val_size=len(val_df),
            brier_score=brier,
            expected_value=ev,
            angel_proposed_trades=n_angel_proposed,
            devil_approved_trades=n_devil_approved,
            win_rate=win_rate,
        )
        fold_metrics.append(fm)

        # ─────────────────────────────────────────────────────────────
        # Fold 3 — retain model refs + compute Profit Factor gate
        # CRITICAL: PF is computed from Fold 3 model on Fold 3 val set.
        # These are the same approved_targets already computed above —
        # no additional training or data access needed.
        # ─────────────────────────────────────────────────────────────
        if fold_number == n_folds:
            fold3_angel = angel_model
            fold3_devil = devil_model
            fold3_angel_feats = angel_feats
            fold3_devil_feats = devil_feats

            # Phase 5.5: Profit Factor and win rate computed from MACRO
            # outcomes (45-bar bracket) on Devil-approved trades.
            # approved_targets (survival) is used for Brier only — the PF
            # gate must reflect actual bracket R:R, not survival rate.
            approved_macro_targets = proposed_devil_targets_macro[approved_mask]
            macro_wins = int(approved_macro_targets.sum())
            macro_losses = n_devil_approved - macro_wins
            gross_profit = macro_wins * tp_mult
            gross_loss = macro_losses * sl_mult
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )
            final_win_rate = (
                float(approved_macro_targets.mean()) if n_devil_approved > 0 else 0.0
            )
            final_total_trades = n_devil_approved

            logger.info(
                f"[Fold {fold_number}] Profit Factor (macro) = "
                f"{gross_profit:.2f} / {gross_loss:.2f} = {profit_factor:.4f} "
                f"| Macro WR={final_win_rate:.1%} | Survival WR={win_rate:.1%}"
            )

    # ───────────────────────────────────────────────────────────────────
    # Aggregate metrics and apply gate
    # ───────────────────────────────────────────────────────────────────
    mean_brier = float(np.mean([fm.brier_score for fm in fold_metrics]))
    mean_ev = float(np.mean([fm.expected_value for fm in fold_metrics]))

    rejection_reasons: List[str] = []
    if mean_brier > BRIER_THRESHOLD:
        rejection_reasons.append(
            f"Brier {mean_brier:.4f} > {BRIER_THRESHOLD} threshold"
        )
    if mean_ev < EV_THRESHOLD:
        rejection_reasons.append(f"EV {mean_ev:.6f} < {EV_THRESHOLD} threshold")
    if profit_factor < PROFIT_FACTOR_THRESHOLD:
        rejection_reasons.append(
            f"Profit Factor {profit_factor:.4f} < {PROFIT_FACTOR_THRESHOLD} threshold"
        )
    if final_total_trades < MIN_OOS_TRADES_FOR_PF:
        rejection_reasons.append(
            f"Fold {n_folds} OOS trades {final_total_trades} < "
            f"{MIN_OOS_TRADES_FOR_PF} minimum — sample too small to trust "
            f"PF={profit_factor:.4f}"
        )

    gate_passed = len(rejection_reasons) == 0

    logger.info("=" * 70)
    logger.info("VALIDATION GATE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Mean Brier Score : {mean_brier:.4f} (threshold ≤ {BRIER_THRESHOLD})")
    logger.info(f"Mean EV          : {mean_ev:.6f} (threshold ≥ {EV_THRESHOLD})")
    logger.info(
        f"Profit Factor    : {profit_factor:.4f} "
        f"(threshold ≥ {PROFIT_FACTOR_THRESHOLD}, Fold {n_folds} OOS)"
    )
    logger.info(
        f"OOS Trades       : {final_total_trades} "
        f"(threshold ≥ {MIN_OOS_TRADES_FOR_PF}, Fold {n_folds} sample-size floor)"
    )
    logger.info(f"Gate Result      : {'PASSED ✅' if gate_passed else 'FAILED 🚫'}")

    # ───────────────────────────────────────────────────────────────────
    # If gate passed — train final production model on ALL 60 days.
    # This is the REWARD for passing: maximum information for production.
    # If gate failed — keep Fold 3 models as placeholders (NOT saved).
    # ───────────────────────────────────────────────────────────────────
    if gate_passed:
        logger.info("Gate passed — training final production model on full dataset")
        # Fit a fresh HMM on the full dataset for production inference. This
        # is the dict that gets persisted alongside Angel/Devil; MLStrategy
        # loads it and applies it to live bars.
        if USE_HMM_FEATURES:
            logger.info("Fitting production HMM on full retraining window...")
            final_hmm_models = fit_regime_models(df)
            df = predict_regime_probs(df, final_hmm_models)
        final_angel, final_devil, final_angel_feats, final_devil_feats = refit_models(
            df, feature_cols, angel_params=angel_params, devil_params=devil_params
        )
    else:
        logger.info(
            "Gate failed — skipping full-data training. Production weights retained."
        )
        final_angel = fold3_angel
        final_devil = fold3_devil
        final_angel_feats = fold3_angel_feats
        final_devil_feats = fold3_devil_feats

    report = ValidationReport(
        fold_metrics=fold_metrics,
        mean_brier=mean_brier,
        mean_ev=mean_ev,
        final_profit_factor=profit_factor,
        final_win_rate=final_win_rate,
        final_total_trades=final_total_trades,
        gate_passed=gate_passed,
        rejection_reasons=rejection_reasons,
    )

    return (
        report,
        final_angel,
        final_devil,
        final_angel_feats,
        final_devil_feats,
        production_threshold,
        final_hmm_models,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GATE DECISION & MODEL PROMOTION
# ═══════════════════════════════════════════════════════════════════════════════


def promote_or_reject(
    report: ValidationReport,
    angel_model: "lgb.LGBMClassifier",
    devil_model: "lgb.LGBMClassifier",
    threshold: float = 0.20,
    asset_config: dict = None,
    hmm_models: Optional[dict] = None,
) -> bool:
    """
    Promote or reject candidate models based on the validation report.

    If gate passed: the models passed in were trained on the full 60-day
    dataset (after CV validation confirmed generalizability). Saves them
    atomically via save_models(), then saves the optimal threshold via
    save_threshold().

    If gate failed: the models passed in are Fold 3 models (not saved).
    Production weights are retained, rejection alert is sent to Discord.

    Args:
        report: ValidationReport from validate_candidate()
        angel_model: Final model (full-data if gate passed, Fold 3 if failed)
        devil_model: Final model (full-data if gate passed, Fold 3 if failed)
        threshold: Optimal Devil threshold from Fold 3 (default: 0.20 fallback)
        asset_config: Asset configuration dictionary.

    Returns:
        True if models were promoted, False if rejected.
    """
    notifier = NotificationManager()

    if report.gate_passed:
        logger.info("=" * 70)
        logger.info("✅ VALIDATION GATE PASSED — PROMOTING MODELS")
        logger.info("=" * 70)

        if asset_config is None:
            asset_config = {}

        save_models(angel_model, devil_model, asset_config)
        save_threshold(threshold, asset_config)
        if hmm_models is not None:
            asset_class = asset_config.get("asset_class", "equities")
            hmm_path = Path("models") / asset_class / "hmm_latest.pkl"
            save_hmm_models(hmm_models, hmm_path)

        notifier.send_retraining_report(report, promoted=True)
        return True
    else:
        logger.warning("=" * 70)
        logger.warning("🚫 VALIDATION GATE FAILED — MODELS REJECTED")
        logger.warning("=" * 70)
        for reason in report.rejection_reasons:
            logger.warning(f"  Rejection: {reason}")

        notifier.send_retraining_report(report, promoted=False)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SERIALIZATION (ATOMIC)
# ═══════════════════════════════════════════════════════════════════════════════


def save_models(
    angel_model: "lgb.LGBMClassifier",
    devil_model: "lgb.LGBMClassifier",
    asset_config: dict,
) -> None:
    """
    Serialize models to disk using joblib with POSIX atomic writes.

    Uses temporary files and os.replace() to ensure zero-downtime atomic swaps,
    preventing the live hot-reloader from reading partially written files.

    Args:
        angel_model: Trained Angel model
        devil_model: Trained Devil model
        asset_config: Asset configuration dictionary.
    """
    logger.info("=" * 70)
    logger.info("SERIALIZING MODELS (ATOMIC)")
    logger.info("=" * 70)

    asset_class = asset_config.get("asset_class", "equities")
    model_dir = Path("models") / asset_class
    model_dir.mkdir(parents=True, exist_ok=True)

    angel_path = model_dir / "angel_latest.pkl"
    devil_path = model_dir / "devil_latest.pkl"
    angel_temp = model_dir / "angel_temp.pkl"
    devil_temp = model_dir / "devil_temp.pkl"

    # ═══════════════════════════════════════════════════════════════════
    # ATOMIC WRITE: Angel Model
    # ═══════════════════════════════════════════════════════════════════
    try:
        joblib.dump(angel_model, angel_temp)
        angel_size = angel_temp.stat().st_size / (1024 * 1024)
        os.replace(angel_temp, angel_path)
        logger.info(f"[ATOMIC] Angel model saved: {angel_path} ({angel_size:.1f} MB)")

    except Exception as e:
        logger.error(f"[ATOMIC] Failed to save Angel model: {e}")
        if angel_temp.exists():
            angel_temp.unlink()
        raise

    # ═══════════════════════════════════════════════════════════════════
    # ATOMIC WRITE: Devil Model
    # ═══════════════════════════════════════════════════════════════════
    try:
        joblib.dump(devil_model, devil_temp)
        devil_size = devil_temp.stat().st_size / (1024 * 1024)
        os.replace(devil_temp, devil_path)
        logger.info(f"[ATOMIC] Devil model saved: {devil_path} ({devil_size:.1f} MB)")

    except Exception as e:
        logger.error(f"[ATOMIC] Failed to save Devil model: {e}")
        if devil_temp.exists():
            devil_temp.unlink()
        raise

    # ═══════════════════════════════════════════════════════════════════
    # ATOMIC WRITE: Metadata sidecar
    # ═══════════════════════════════════════════════════════════════════
    metadata_path = model_dir / "metadata.json"
    metadata_temp = model_dir / "metadata_temp.json"
    metadata = {
        "asset_class": asset_class,
        "timeframe_minutes": asset_config.get("timeframe_minutes", 1),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on_symbols": asset_config.get("tickers", []),
        "data_source": os.getenv("DATA_SOURCE", "alpaca").strip().lower(),
    }
    with open(metadata_temp, "w") as f:
        json.dump(metadata, f, indent=2)
    os.replace(metadata_temp, metadata_path)
    logger.info(f"[ATOMIC] Metadata saved: {metadata_path}")

    logger.info(
        "[ATOMIC] Model serialization complete — live bot can hot-reload safely"
    )


def save_threshold(threshold: float, asset_config: dict) -> None:
    """
    Save the optimal Devil threshold to disk as a JSON sidecar file.

    Written atomically alongside the model .pkl files.  The LiveOrchestrator
    and MLStrategy read this on startup and via hot-reload so the live bot
    always uses the threshold that maximises EV on the most recent data.

    Args:
        threshold: The optimal Devil probability threshold (e.g., 0.28)
        asset_config: Asset configuration dictionary.
    """
    asset_class = asset_config.get("asset_class", "equities")
    model_dir = Path("models") / asset_class
    model_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = model_dir / "threshold.json"

    data = {
        "devil_threshold": round(threshold, 4),
        "updated_at": datetime.now().isoformat(),
    }

    # Atomic write — same pattern as model serialisation
    temp_path = model_dir / "threshold_temp.json"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, threshold_path)

    logger.info(
        f"[ATOMIC] Threshold saved: {threshold_path} (devil_threshold={threshold:.4f})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    """
    Main entry point for the validated model retraining pipeline.

    Exit codes:
        0 = Models promoted successfully
        1 = Execution error
        2 = Models rejected by validation gate (production weights retained)

    NOTE: run_pipeline.sh only checks feedback_loop.py's exit code to decide
    whether to trigger retraining. The retrainer's exit code 2 ("tried but
    rejected") is logged for observability but does NOT cause an infinite loop.
    """
    try:
        logger.info(
            "╔══════════════════════════════════════════════════════════════════╗"
        )
        logger.info(
            "║              THE CURE V2 - VALIDATED MODEL RETRAINER             ║"
        )
        logger.info(
            "╚══════════════════════════════════════════════════════════════════╝"
        )

        # ─── Phase 1: Initialize provider + load asset config ──────────────
        data_source = os.getenv("DATA_SOURCE", "alpaca").strip().lower()
        asset_config = get_asset_config(data_source)
        provider = get_market_provider()
        
        # Get asset-class-aware hyperparameters
        asset_class = asset_config.get("asset_class", "equities")
        angel_params, devil_params = get_hyperparameters(asset_class)
        
        logger.info(
            f"Provider initialized: {provider.__class__.__name__} "
            f"| Asset config: {asset_config['tickers']} "
            f"| SL={asset_config['sl_mult']}× TP={asset_config['tp_mult']}× "
            f"max_hold={asset_config['max_hold']} survival={asset_config['survival_bars']} "
            f"| Hyperparams: Angel Leaf={angel_params['min_child_samples']}/{angel_params['class_weight']} "
            f"Devil Leaf={devil_params['min_child_samples']}/{devil_params['class_weight']}"
        )

        # ─── Phase 2: Fetch data ────────────────────────────────────────────
        raw_data = fetch_training_data(
            provider=provider,
            symbols=asset_config["tickers"],
            days_back=DAYS_BACK,
            timeframe_minutes=asset_config["timeframe_minutes"]
        )

        # ─── Phase 3: Engineer features with ATR-dynamic labels ────────────
        features_df, feature_cols = engineer_features_and_labels(
            raw_data,
            sl_mult=asset_config["sl_mult"],
            tp_mult=asset_config["tp_mult"],
            max_hold=asset_config["max_hold"],
            survival_bars=asset_config["survival_bars"],
            htf_timeframe=asset_config.get("htf_timeframe", "5m"),
            # Same RiskProfile path that sources sl_mult/tp_mult → the chop
            # veto simulated here is identical to the live execution gate.
            risk_profile=RiskProfile.for_asset_class(asset_config["asset_class"]),
        )

        # ─── Phase 4: Walk-forward validation (3-fold expanding window) ────
        # TEMPORAL BOUNDARY: CV folds and the Profit Factor gate are evaluated
        # strictly OOS (Fold 3 model evaluated on days 51–60). The full-data
        # production model is only trained AFTER the gate passes. Running the
        # full-data model against any data used in its training would be data
        # leakage.
        logger.info("=" * 70)
        logger.info("WALK-FORWARD VALIDATION (3 EXPANDING FOLDS)")
        logger.info("=" * 70)

        (
            report,
            angel_model,
            devil_model,
            angel_feats,
            devil_feats,
            optimal_threshold,
            final_hmm_models,
        ) = validate_candidate(
            features_df,
            feature_cols,
            sl_mult=asset_config["sl_mult"],
            tp_mult=asset_config["tp_mult"],
            n_folds=3,
            angel_params=angel_params,
            devil_params=devil_params,
        )
        logger.info(f"Optimal Devil threshold (from Fold 3): {optimal_threshold:.4f}")

        # ─── Phase 5: Gate decision ─────────────────────────────────────────
        # If gate passed: angel_model/devil_model are trained on full 60 days
        # If gate failed: they are Fold 3 models (will NOT be saved)
        promoted = promote_or_reject(
            report,
            angel_model,
            devil_model,
            optimal_threshold,
            asset_config,
            hmm_models=final_hmm_models,
        )

        if promoted:
            asset_class = asset_config.get("asset_class", "equities")
            logger.info("=" * 70)
            logger.info(f"✅ MODELS PROMOTED ({asset_class}) — Ready for next market open")
            logger.info(f"  Models saved in: models/{asset_class}/")
            logger.info("=" * 70)
            return 0
        else:
            logger.warning("=" * 70)
            logger.warning("🚫 MODELS REJECTED — Production weights retained")
            logger.warning("Manual review recommended.")
            logger.warning("=" * 70)
            return 2  # 2 = retrained but rejected; production weights intact

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
