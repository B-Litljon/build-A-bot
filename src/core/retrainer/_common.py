"""Env configuration, classifier factory, feature columns, spread-cost table, and the shared
import machinery that every other retrainer submodule inherits via re-export.
Everything below was split out of the monolithic core/retrainer.py on 2026-09-16.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Ensure project root and src/ are on sys.path, and remove src/core if present to avoid shadowing stdlib signal
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
_src_dir = str(Path(__file__).resolve().parent.parent.parent)
_pkg_dir = str(Path(__file__).resolve().parent)
_core_dir = str(Path(__file__).resolve().parent.parent)
while _core_dir in sys.path:
    sys.path.remove(_core_dir)
while _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
for p in (_src_dir, _project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import beta as _scipy_beta
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit

try:
    from src.data.factory import get_market_provider
    from src.data.market_provider import MarketDataProvider
    from src.execution.risk_manager import (
        RiskProfile,
        _chop_filter_enabled,
        coupled_keff,
    )
    from src.ml.feature_pipeline import FeaturePipeline
    from src.ml.feature_stats import compute_feature_stats, save_feature_stats
    from src.ml.features.v3_features import (
        V3BaseFeatures,
        V3CostFeatures,
        V3HTFFeatures,
        V3SessionFeatures,
    )
    from src.ml.regimes.hmm_regime import (
        HMM_OUTPUT_COLS,
        fit_regime_models,
        predict_regime_probs,
        save_hmm_models,
    )
    from src.ml.barriers.estimator import (
        BarrierEstimator,
        DEFAULT_TAU_MAE,
        DEFAULT_TAU_MFE,
        DEFAULT_RR_FLOOR,
    )
    from src.ml.barriers.labels import DEFAULT_HORIZON, compute_excursions
    from src.core.notification_manager import NotificationManager
except ImportError:
    from data.factory import get_market_provider
    from data.market_provider import MarketDataProvider
    from execution.risk_manager import (
        RiskProfile,
        _chop_filter_enabled,
        coupled_keff,
    )
    from ml.feature_pipeline import FeaturePipeline
    from ml.feature_stats import compute_feature_stats, save_feature_stats
    from ml.features.v3_features import (
        V3BaseFeatures,
        V3CostFeatures,
        V3HTFFeatures,
        V3SessionFeatures,
    )
    from ml.regimes.hmm_regime import (
        HMM_OUTPUT_COLS,
        fit_regime_models,
        predict_regime_probs,
        save_hmm_models,
    )
    from ml.barriers.estimator import (
        BarrierEstimator,
        DEFAULT_TAU_MAE,
        DEFAULT_TAU_MFE,
        DEFAULT_RR_FLOOR,
    )
    from ml.barriers.labels import DEFAULT_HORIZON, compute_excursions
    from core.notification_manager import NotificationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DAYS_BACK = int(os.getenv("RETRAIN_DAYS_BACK", "60"))

# Learned quantile barriers (Phase 3): when enabled (default 1), fits and
# persists BarrierEstimator (barriers_mae.pkl, barriers_mfe.pkl,
# barriers_meta.json) alongside the Angel and Devil classifiers.
RETRAIN_LEARN_BARRIERS = os.getenv("RETRAIN_LEARN_BARRIERS", "1").strip() == "1"

# Which label the Devil is trained AND scored on.
ENV_DEVIL_LABEL = "RETRAIN_DEVIL_LABEL"
DEVIL_LABEL_DEFAULT = "survival"


def devil_label_col() -> str:
    """
    The column the Devil is trained and Brier-scored against.

    ``survival`` (the default, and what has always shipped) is ``devil_target``:
    "did price avoid a ``sl_mult`` x ATR stop for ``survival_bars`` (5) bars".
    ``macro`` is ``devil_target_macro``: "did TP hit before SL inside
    ``max_hold`` (45) bars at the served bracket" — the outcome the live path
    actually bets on.

    Why the switch exists, measured 2026-09-14 (honest OOF probabilities,
    chronological 5-fold, 227k rows): a Devil trained on the survival label
    scores AUC **0.4722** against the MACRO outcome — below chance, and **0.4564**
    in the held-out final window — while the same model trained on the macro
    label scores **0.5839** / **0.5806**. The two scores are *anti-correlated*
    (−0.166), i.e. the shipping stage is currently scoring close to the opposite
    of what predicts the bracket. Same defect class as the gate-EV fix of
    2026-09-09 (item 20 of the 2026-09-08 audit), one level down: the metric was
    corrected, the model's target was not.

    Default is ``survival`` so a retrain changes nothing until someone opts in —
    this is a training-semantics change and it belongs to a deliberate retrain
    batch, not to import order. Read from the environment at CALL time rather
    than bound at import for exactly that reason (a module-level constant read at
    import is what silently ignored ``RETRAIN_DAYS_BACK`` in the H4 candidate
    script). An unrecognised value falls back to the default and warns.
    """
    raw = os.getenv(ENV_DEVIL_LABEL, DEVIL_LABEL_DEFAULT).strip().lower()
    if raw in ("macro", "devil_target_macro", "bracket", "45"):
        return "devil_target_macro"
    if raw in ("survival", "devil_target", "survival_bars", "5", ""):
        return "devil_target"
    logger.warning(
        "[DEVIL] %s=%r is not a recognised label — using %r. "
        "Accepted: survival (devil_target), macro (devil_target_macro).",
        ENV_DEVIL_LABEL, raw, DEVIL_LABEL_DEFAULT,
    )
    return "devil_target"

# Instruments this OANDA account cannot trade. They stay in the TRAINING basket
# — the volatility-first basket is load-bearing (a shrink was tried and
# rejected 2026-07-02, and measurement on 2026-08-23 put the crosses at gross PF
# 1.742 when metals were in training vs 1.461 without) — but their trades are
# excluded from the PROMOTION GATE's metrics, because the gate is supposed to
# predict live results and no live trade in these can ever happen.
#
# Why it matters (2026-08-23): pooling them INVERTS the verdict. Pooled gate PF
# read 1.245 with metals vs 1.500 without, while the same two configs scored on
# the six tradeable crosses read 1.742 vs 1.461 — the opposite ordering. The
# pooled figure was diluted by trades that do not exist.
#
# Live already refuses them at boot via
# OandaForexOrchestrator._drop_untradeable_symbols (2026-08-06); this closes
# the same gap on the offline side. Empty string scores everything (old
# behaviour) — set RETRAIN_UNTRADEABLE_SYMBOLS="" to restore it.
# Behavior labels to drop as trade entries, e.g. "trend_high". OFF by default:
# a model trained with this MUST be served behind a matching live gate, and no
# such gate exists yet. See engineer_features_and_labels.
BEHAVIOR_VETO_LABELS = frozenset(
    s.strip() for s in os.getenv("RETRAIN_BEHAVIOR_VETO", "").split(",") if s.strip()
)

UNTRADEABLE_SYMBOLS = frozenset(
    s.strip().upper()
    for s in os.getenv("RETRAIN_UNTRADEABLE_SYMBOLS", "XAU_USD,XAG_USD").split(",")
    if s.strip()
)
# Forex basket pivoted 2026-05-23 from G7 majors (failed integrity gate, NO
# SIGNAL separation) to a volatility-first basket: two liquid metals plus
# three JPY/AUD-crossing pairs known for wide intraday ranges. XPT/XPD
# skipped — too illiquid on OANDA for scalping.
# The Angel's label is a DIRECTION question ("does price run in 45 minutes"),
# and it is deliberately NOT the execution stop multiple. The two answer
# different questions over different horizons: the Angel looks 3 bars ahead,
# while the trade itself has max_hold (45) bars to resolve. Welding them
# together means widening the stop silently doubles the move the Angel must
# predict -- which on 2026-08-08 collapsed pooled proposals from ~300 to 49 and
# failed the gate on sample size alone. These defaults preserve each asset
# class's historical Angel behaviour; override with RETRAIN_ANGEL_ATR_MULT.
_ANGEL_ATR_MULT_BY_CLASS = {"forex": 1.0, "equities": 0.5}

_DEFAULT_TICKERS_BY_CLASS = {
    "forex": [
        "XAU_USD", "XAG_USD",                        # liquid metals
        "GBP_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY",  # JPY-cross volatility
        "GBP_AUD", "GBP_NZD",                        # commonwealth crosses
    ],
    "equities": ["TSLA", "NVDA", "MARA", "COIN", "SMCI"],
}

# Higher-timeframe context per bar size. MUST mirror run_oanda.py's
# _GRANULARITY_PROFILES: the live bot derives htf features from this pairing,
# so training with a different one is train/serve skew in the htf_* columns
# (htf_rsi_14, htf_vol_rel, htf_bb_pct_b, htf_trend_agreement).
_HTF_FOR_TIMEFRAME = {1: "5m", 5: "30m", 15: "1h"}


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
        
    tf_minutes = int(os.getenv("RETRAIN_TIMEFRAME_MINUTES", str(timeframe)))

    return {
        "asset_class": asset_class,
        # Output directory for the trained model. Defaults to models/<asset_class>
        # (the production location). Override with RETRAIN_MODEL_DIR to train a
        # SIDE model (e.g. a metals-only candidate) WITHOUT clobbering the
        # promoted model — asset_class stays "forex" so every feature/gate/
        # hyperparameter path is identical; only the save destination changes.
        "model_dir": (os.getenv("RETRAIN_MODEL_DIR", "").strip() or f"models/{asset_class}"),
        "tickers": [t.strip() for t in os.getenv("RETRAIN_SYMBOLS", ",".join(default_tickers)).split(",") if t.strip()],
        "sl_mult": profile.sl_atr_multiplier,
        "tp_mult": profile.tp_atr_multiplier,
        # Deliberately NOT sl_mult -- see _ANGEL_ATR_MULT_BY_CLASS.
        "angel_mult": float(
            os.getenv("RETRAIN_ANGEL_ATR_MULT", "").strip()
            or _ANGEL_ATR_MULT_BY_CLASS[asset_class]
        ),
        "max_hold": int(os.getenv("RETRAIN_MAX_HOLD", str(max_hold))),
        "survival_bars": int(os.getenv("RETRAIN_SURVIVAL", "5")),
        "timeframe_minutes": tf_minutes,
        # MUST match the live bot's pairing for this bar size, or the htf_*
        # features are computed one way in training and another at inference.
        # run_oanda.py's _GRANULARITY_PROFILES is the authority; the old flat
        # "5m" default was only ever correct for M1, so an M15 retrain that
        # forgot RETRAIN_HTF_TIMEFRAME silently shipped skew.
        "htf_timeframe": os.getenv("RETRAIN_HTF_TIMEFRAME", "").strip()
        or _HTF_FOR_TIMEFRAME.get(tf_minutes, htf_timeframe),
    }

# Model Hyperparameters
def get_hyperparameters(asset_class: str) -> Tuple[dict, dict]:
    """
    Get hyperparameter configurations for Angel and Devil LightGBM models.

    Updated 2026-08-29: Constrained 100-tree / 15-leaf capacity setup with
    min_child_samples=80. Collapses the in-sample vs out-of-sample memorization
    gap from 0.109 to 0.014 on untouched holdout while improving generalization.
    """
    angel_params = {
        "objective": "binary",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 15,
        "min_child_samples": 80 if asset_class == "forex" else 50,
        "class_weight": None if asset_class == "forex" else "balanced",
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        # Fully reproducible sweeps across the cached parquet datasets:
        # random_state already seeds bagging/feature-fraction; deterministic +
        # force_row_wise remove the multithreaded histogram float-ordering
        # jitter that can otherwise nudge a borderline threshold-grid pick.
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": -1,
        "verbose": -1,
    }

    devil_params = {
        "objective": "binary",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 15,
        "min_child_samples": 80 if asset_class == "forex" else 50,
        "class_weight": None,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        # Fully reproducible sweeps across the cached parquet datasets:
        # random_state already seeds bagging/feature-fraction; deterministic +
        # force_row_wise remove the multithreaded histogram float-ordering
        # jitter that can otherwise nudge a borderline threshold-grid pick.
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": -1,
        "verbose": -1,
    }

    return angel_params, devil_params

# Fallback constants for backward compatibility
ANGEL_PARAMS, DEVIL_PARAMS = get_hyperparameters("equities")


# ─── Trainer family seam (Stage 2 CatBoost experiment) ──────────────────────
# MODEL_FAMILY selects the estimator class behind refit_models/validate_candidate.
# LightGBM is the incumbent; the CatBoost arm is the Stage-2 A/B candidate ordered
# boosting with monotone constraints where LightGBM's objective would not take
# them. The interface both sides satisfy is identical: fit(X, y,
# sample_weight=w), predict_proba(X)[:,1]. The served artifact hot-reload path in
# MLStrategy unpickles whatever object lands there, so a promoted CatBoost model
# only requires catboost importable at inference time (it is, in the venv).
#
# The CatBoost arm translates LightGBM's parameter vocabulary rather than
# duplicating a parallel param block: one variable changed, the family. Anything
# CatBoost has no analogue for is dropped explicitly below so the mapping is
# auditable (never silently lost).
MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "lightgbm").strip().lower()

# Ordered boosting must never see subsample < 1.0 — Bayesian bootstrap is the
# only mode Ordered supports, and subsample is not meaningful there. These keys
# have no CatBoost analogue and are dropped in translation, by name, so the
# audit trail says what changed.
_CATBOOST_DROPPED_KEYS = {
    "objective",          # CatBoost infers Logloss for binary labels
    "num_leaves",         # oblivious trees use symmetric depth, no leaf cap
    "class_weight",       # forex path already None; equities "balanced" is
                          # a data-cook, not the variable under test
    "deterministic",      # CatBoost reproducibility comes from random_seed
    "force_row_wise",     # LightGBM histogram-ordering knob, N/A
    "subsample_freq",     # LightGBM bagging schedule, N/A
    "n_jobs",             # translated to thread_count
}


def _catboost_params(params: dict, feature_cols: List[str]) -> dict:
    """
    Translate a LightGBM Angel/Devil param dict into CatBoost's vocabulary.

    Mappings:
      n_estimators      -> iterations
      num_leaves/max_depth -> depth (oblivious trees take one depth)
      min_child_samples -> min_data_in_leaf
      subsample         -> dropped on Ordered boosting (Bayesian bootstrap
                           governs averaging; subsample is invalid there)
      colsample_bytree  -> rsm
      random_state      -> random_seed
      verbose/n_jobs    -> verbose/thread_count

    Monotone constraints (the point of the change): cost_ratio is
    monotone-DECREASING on acceptance probability — higher spread cost must
    never buy a higher score. Constraints are intersected with
    ``feature_cols``: cost_ratio exists only when the spread table is on, and
    CatBoost hard-errors on a constraint naming a column that is not present
    ("Unknown feature name: cost_ratio").
    """
    iterations = params.get("n_estimators", 100)
    depth = min(params.get("max_depth", 6), 16)
    l2 = None  # CatBoost default l2_leaf_reg=3 is fine; not an LGBM param.
    _MONO_BY_NAME = {"cost_ratio": -1}
    monotone = {c: _MONO_BY_NAME[c] for c in feature_cols if c in _MONO_BY_NAME}
    out = {
        "iterations": iterations,
        "learning_rate": params.get("learning_rate", 0.05),
        "depth": depth,
        "min_data_in_leaf": params.get("min_child_samples", 50),
        "rsm": params.get("colsample_bytree", 0.8),
        "boosting_type": "Ordered",
        "random_seed": params.get("random_state", 42),
        "monotone_constraints": monotone,
        "thread_count": -1,
        "verbose": 0,
        "allow_writing_files": False,
    }
    if l2 is not None:
        out["l2_leaf_reg"] = l2
    # subsample is valid on Plain only; Ordered forbids it. Dropped by name.
    dropped = sorted(
        (set(params) | {"subsample"}) & (_CATBOOST_DROPPED_KEYS | {"subsample"})
    )
    if dropped:
        logger.info(f"CatBoost param translation dropped LGBM-only keys: {dropped}")
    return out


def make_classifier(params: dict, feature_cols: List[str]):
    """Return an unfitted Angel/Devil classifier of the active MODEL_FAMILY."""
    if MODEL_FAMILY == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**_catboost_params(params, feature_cols))
    return lgb.LGBMClassifier(**params)


# ═══════════════════════════════════════════════════════════════════════════════
# ATR BRACKET PARAMETERS (must match evaluate_performance.py)
# ═══════════════════════════════════════════════════════════════════════════════

SL_ATR_MULTIPLIER = 0.5
TP_ATR_MULTIPLIER = 3.0
MAX_HOLD_BARS = 45
SURVIVAL_BARS = 5  # Phase 5.5: Devil survival window (bars)

# Round-trip spread applied inside the Devil label walks (spread-adjusted
# brackets, 2026-09-27). Units match RiskManager's Gate A proxy
# (alpha * baseline_atr_abs, a fraction of ATR) and V3CostFeatures'
# cost_ratio (alpha * baseline_natr / natr_14) — alpha is dimensionless in
# NATR units of ATR, so spread_price = alpha * atr_abs. Both Devil target
# generators shift both bracket edges up by it; None-table runs stay
# bit-identical to the historical frictionless labels.
DEFAULT_SPREAD_ALPHA = 0.15

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE THRESHOLDS (must match MLStrategy)
# ═══════════════════════════════════════════════════════════════════════════════

# The Angel bar is imported, not defined here: core.thresholds is the single
# source of truth (ANGEL_THRESHOLD env var overrides at process start). The
# Devil trains only on Angel-approved rows (Phase 5.5) and the bracket
# optimizer fits on the same population — one value, or the stages drift.
from src.core.thresholds import ANGEL_THRESHOLD  # noqa: E402

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
MIN_OOS_TRADES_FOR_PF = 100  # legacy Fold-3-only floor (superseded below)

# Absolute sanity backstop on pooled fold OOS trades — NOT the evidence bar.
# 2026-08-29: the old flat floor (300) was the fold gate's de-facto evidence
# standard, and it was mis-calibrated for every architecture: the shipped
# 200x63 config cleared it 1 time in 3 (238/230/227 vs ~232), while every
# capacity-reduced config landed at 18-39% of it — a cliff that rejected
# configs whose per-trade evidence was strong. The evidential judgement now
# lives in the Clopper-Pearson PF lower bound applied to the pooled fold
# trades (same instrument as the artifact holdout gate — see
# _holdout_pf_lower_bound); this constant is only the backstop that keeps a
# handful of lucky wins from ever reaching that instrument, scaled down by
# the chop filter's row-drop rate as before:
#     effective_floor = BASELINE_POOLED_OOS_TRADES * (1 - chop_veto_rate)
# The ARTIFACT HOLDOUT never borrows this floor: its sample-size discipline
# is the confidence bound (HOLDOUT_PF_CONFIDENCE / _holdout_pf_lower_bound).
BASELINE_POOLED_OOS_TRADES = int(os.getenv("RETRAIN_POOLED_TRADE_FLOOR", "30"))

# Minimum Angel proposals the dynamic Angel threshold must yield on its
# training frame. The Devil trains ONLY on the Angel-approved subpopulation
# (Phase 5.5), so this floor is what guarantees the second stage a learnable
# population: 300 approved rows lets the auto-scaled Devil min_child_samples
# (n//10) still form real leaves. Only reachable when the ANGEL_THRESHOLD env
# var is NOT pinning a fixed bar (see _find_optimal_angel_threshold).
MIN_ANGEL_PROPOSALS = int(os.getenv("RETRAIN_MIN_ANGEL_PROPOSALS", "300"))

# Devil min_child_samples override. LightGBM needs >= 2x min_child_samples
# rows to split a node; the Angel-side value (80) applied to a Devil
# population of dozens-to-hundreds collapses the Devil to a constant
# (separation gap 0.0000, 100% approval — measured 2026-08-29). Unset, the
# Devil's value auto-scales to its approved population inside refit_models
# (capped at the configured value, floored at 5). Set to pin a fixed value.
DEVIL_MIN_CHILD_FIXED = os.getenv("RETRAIN_DEVIL_MIN_CHILD", "").strip()

# Fixed-vs-dynamic Angel threshold mode. core.thresholds reads the
# ANGEL_THRESHOLD env var with a 0.40 default; when the var is explicitly
# set we treat that as a deliberate FIXED bar and skip the OOF calibration
# (old behaviour, bit-for-bit reproducible). Unset: the threshold is
# calibrated per refit from out-of-fold probabilities and pinned into the
# artifact's threshold.json, which MLStrategy already prefers over the
# constant — train/serve symmetry holds by construction.
_FIXED_ANGEL_THRESHOLD = os.getenv("ANGEL_THRESHOLD", "").strip() != ""

# Toggle for the HMM regime-feature experiment. When enabled, a per-symbol
# 3-state GaussianHMM is fit on each fold's training window (no leakage) and
# its posterior state probabilities are appended as 3 additional features.
# Default off so a plain LightGBM swap can be evaluated without confounds.
USE_HMM_FEATURES = os.getenv("RETRAIN_USE_HMM", "0").strip() == "1"

# Holdout fraction for the artifact-level gate. Carved from the chronologically
# last slice of the raw window BEFORE feature engineering, so no part of the
# pipeline can leak holdout information into the served model. 0 disables the
# holdout and preserves the legacy full-data retrain behaviour.
HOLDOUT_FRAC = float(os.getenv("RETRAIN_HOLDOUT_FRAC", "0.18"))

# One-sided confidence level for the holdout profit-factor gate. The verdict
# gates on the Clopper-Pearson LOWER bound of the macro win rate mapped
# through PF, not the point estimate (audit 2026-08-24: the same config
# passed at PF 1.444 and failed at 0.982 an hour apart). For independent
# trades a break-even artifact passes with probability at most 1-confidence
# at any trade count; the 45-bar macro walks overlap, so treat the bound as
# conservative rather than a literal coverage guarantee -- either way it is
# strictly stronger than the point estimate it replaced. The sample-size
# floor, generalised from a cliff to a continuous evidential bar.
# Deliberately NOT env-tunable: the promotion gate is not a knob.
HOLDOUT_PF_CONFIDENCE = 0.95

# Per-instrument spread-cost table (2026-07-07 cost-awareness experiment).
# Points at a JSON baked by scripts/bake_spread_alphas.py from live
# SPREAD_CALIB measurements. When set:
#   * the chop-veto Gate A uses each instrument's measured alpha instead of
#     the flat profile.spread_atr_alpha (0.15) — fixes a real train/live
#     asymmetry where training kept e.g. GBP_NZD (alpha ~0.90) setups that
#     live always vetoes;
#   * V3CostFeatures appends a ``cost_ratio`` feature so the model can SEE
#     cost and learn to suppress conviction on untradeable instruments;
#   * on gate pass the table is copied into model_dir/spread_alphas.json so
#     model + cost assumptions travel together (live loads it from there).
# Unset → prod retrains are bit-identical to before this experiment.
_SPREAD_TABLE_PATH = os.getenv("RETRAIN_SPREAD_TABLE", "").strip()


def _load_spread_table(path: str) -> Optional[dict]:
    """Parse a bake_spread_alphas.py JSON table; raise on malformed input."""
    with open(path, "r") as fh:
        table = json.load(fh)
    if "alphas" not in table or not isinstance(table["alphas"], dict):
        raise ValueError(f"Spread table {path} has no 'alphas' mapping")
    return table


SPREAD_TABLE: Optional[dict] = (
    _load_spread_table(_SPREAD_TABLE_PATH) if _SPREAD_TABLE_PATH else None
)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMNS (must match MLStrategy.feature_names and FeaturePipeline output)
# ═══════════════════════════════════════════════════════════════════════════════

# Base features produced by FeaturePipeline. The HMM regime features (when
# enabled) are appended downstream in validate_candidate() because their
# fitting must respect each fold's temporal boundary.
# 5 dead/memorized features dropped on 2026-08-29 (htf_rsi_14, htf_vol_rel,
# bar_body_pct, bar_upper_wick_pct, bar_lower_wick_pct).
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
    # V3.3: Higher-timeframe context
    "htf_trend_agreement",
    "htf_bb_pct_b",
    # V3.4 Phase 5: Microstructure features (volatility compression)
    "range_coil_10",
    # V3.5 (2026-05-23): UTC session-activity indicators
    "session_asia",
    "session_london",
    "session_ny",
    "session_overlap",
]

FEATURE_COLS: List[str] = (
    BASE_FEATURE_COLS
    + (["cost_ratio"] if SPREAD_TABLE else [])
    + (HMM_OUTPUT_COLS if USE_HMM_FEATURES else [])
)
