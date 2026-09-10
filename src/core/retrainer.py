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

This is the training half of the system. Nothing here runs during live trading;
it fetches history, builds labels, trains the two models, tests them on data
they never saw, and only overwrites the live model files if that test passes.
Domain terms (angel/devil, ATR/NATR, bracket, chop veto, walk-forward, OOS,
Brier score, profit factor) are defined in GLOSSARY.md.

Artifacts written on promotion, into ``model_dir`` (default models/<asset_class>/):
    angel_latest.pkl, devil_latest.pkl, metadata.json, threshold.json,
    feature_stats.json, and spread_alphas.json when the cost experiment is on.
All are written to a temp name then os.replace()d, so the live bot's
hot-reloader can never read a half-written file.

Glossary:
    DAYS_BACK -- how many days of history to train on (default 60, override
        RETRAIN_DAYS_BACK).
    _DEFAULT_TICKERS_BY_CLASS -- the default instrument basket per asset class.
        The forex basket is volatility-first (two metals plus JPY/commonwealth
        crosses) because the calm G7 majors failed the gate outright. Override
        with RETRAIN_SYMBOLS.
    get_asset_config -- turns DATA_SOURCE ("oanda" -> forex, else equities)
        into one dict of every knob the run needs: basket, bracket multipliers,
        hold limit, bar size, and where to save.
    model_dir -- save destination. Defaults to the production models/<class>/;
        set RETRAIN_MODEL_DIR to train a side candidate without overwriting the
        promoted model, while keeping every other setting identical.
    get_hyperparameters -- returns (angel_params, devil_params) for LightGBM.
        Both are deterministic by construction (fixed seed, force_row_wise) so
        two runs on the same data give the same model.

    SL_ATR_MULTIPLIER / TP_ATR_MULTIPLIER -- bracket width in units of the
        instrument's own recent volatility: stop at 0.5x, target at 3.0x (a
        6:1 payoff). ⚠️ These are the EQUITIES/default values and are only
        function defaults here -- get_asset_config() takes the real numbers
        from RiskProfile.for_asset_class(), and the FOREX profile overrides
        them to 2.0x / 4.0x (a 2:1 payoff, widened 2026-08-08). Check the
        profile, not these constants, when reasoning about a forex run.
        Either way the value must match the live orchestrator's, or the model
        is trained on trades the bot would never take.
    _HTF_FOR_TIMEFRAME -- bar size -> higher-timeframe context, mirroring
        run_oanda._GRANULARITY_PROFILES. Training with a different pairing than
        the live bot skews every htf_* feature. Override:
        RETRAIN_HTF_TIMEFRAME.
    UNTRADEABLE_SYMBOLS -- instruments this account cannot trade (XAU_USD,
        XAG_USD; override RETRAIN_UNTRADEABLE_SYMBOLS). They stay in TRAINING
        and are excluded from the GATE's metrics only. Pooling them inverts the
        verdict, so this is a correctness fix, not a tidy-up.
    BEHAVIOR_VETO_LABELS -- behavior tags dropped as trade entries
        (RETRAIN_BEHAVIOR_VETO, e.g. "trend_high"). EMPTY by default. A model
        trained with this set REQUIRES a matching live gate or it is train/serve
        skew; the labels are recorded in metadata.json as "behavior_veto".
    _tradeable_scoring_mask -- narrows an approval mask to tradeable
        instruments. Runs on validation approvals only; training is untouched.
    validate_candidate(oos_ledger=...) -- opt-in per-trade capture of the
        walk-forward's Devil-approved OOS trades, for offline analysis such as
        the behavior matrix. None in every production path; when a list is
        passed it is appended to and NOTHING about the gate changes.
    _capture_oos_ledger -- builds one fold's ledger frame from masks the fold
        already computed. Observational only; cannot affect promotion.
    oos_ledger_cols -- which columns to carry from the validation frame into
        the ledger. Missing columns are skipped, so a caller can request
        optional ones (e.g. behavior_label) without knowing the schema.
    MAX_HOLD_BARS -- 45. A trade that reaches neither level within 45 bars is
        labelled a loss (timeout), because capital was tied up for nothing.
    SURVIVAL_BARS -- 5. The horizon for the Devil's survival label, below.
    ANGEL_THRESHOLD -- imported from core.thresholds (0.40 unless the env var
        overrides at process start). Minimum Angel probability for a bar to count as a
        proposed trade. 2026-08-29: when the env var is NOT set this is now a
        FALLBACK only — the live value is calibrated per refit from OOF
        probabilities (_find_optimal_angel_threshold) and pinned into the
        artifact's threshold.json, which MLStrategy prefers over the
        constant. Setting the env var selects fixed mode (old behaviour).
    _FIXED_ANGEL_THRESHOLD -- True when ANGEL_THRESHOLD is explicitly set in
        the environment; selects the fixed-bar mode above.
    MIN_ANGEL_PROPOSALS -- 300 (RETRAIN_MIN_ANGEL_PROPOSALS). Floor on
        proposals the dynamic Angel threshold must yield, guaranteeing the
        Devil a learnable training population.
    DEVIL_MIN_CHILD_FIXED -- RETRAIN_DEVIL_MIN_CHILD, unset by default.
        Pins the Devil's min_child_samples; unset, it auto-scales to a tenth
        of the Angel-approved population (capped at the Angel-side value,
        floored at 5) because a split needs >= 2x min_child rows and the
        Angel-side 80 collapses a Devil trained on dozens of rows into a
        constant (gate matrix, 2026-08-29).
    _find_optimal_angel_threshold -- sweeps quantiles of the OOF Angel score
        distribution for the EV-maximising proposal bar subject to
        MIN_ANGEL_PROPOSALS. Expressing the bar in the model's own score
        units fixes the proposal-starvation that a fixed 0.40 caused under
        score compression.
    DEVIL_THRESHOLD -- 0.50, fallback only. The real threshold is chosen per
        run by _find_optimal_threshold() and saved to threshold.json.

    BRIER_THRESHOLD -- 0.30 gate ceiling on probability calibration. Raised
        from 0.25 deliberately: the survival label's ~45% base rate puts a
        do-nothing classifier near 0.25, so the old value rejected honest
        models.
    EV_THRESHOLD -- 0.0005, minimum average return per trade to promote.
    PROFIT_FACTOR_THRESHOLD -- 1.2, minimum gross-win / gross-loss ratio.
    MIN_OOS_TRADES_FOR_PF -- 100, the legacy single-fold sample-size floor,
        superseded by the pooled floor below but still referenced.
    BASELINE_POOLED_OOS_TRADES -- 30 (was 300 until 2026-08-29). Absolute
        backstop on pooled fold OOS trades, scaled down by the chop veto's
        drop rate (effective_floor = 30 x (1 - chop_veto_rate)). This is NOT
        the evidence bar: that is the Clopper-Pearson PF lower bound computed
        on the pooled (wins, trades) — the same instrument as the artifact
        holdout gate — applied at two scales (Fold 3 alone = recency; all
        folds pooled = evidence). The old flat 300 floor was a cliff the
        shipped config itself cleared only once in three pins while
        rejecting strong-evidence low-frequency configs. FOLD gate only --
        the artifact holdout never borrows it (see HOLDOUT_PF_CONFIDENCE).
    pooled_pf_lower_bound / fold3_pf_lower_bound -- ValidationReport fields
        carrying the two CP bounds the fold gate judged; recomputed from the
        raw (wins, trades) evidence via _holdout_pf_lower_bound.
    FoldMetrics.macro_wins -- per-fold macro (45-bar bracket) wins among
        scored approvals; pooled across folds for the CP instrument.
    production_angel_threshold -- ValidationReport field: the Angel bar the
        returned models were trained with (calibrated or env-fixed). main()
        threads it to the holdout scoring, threshold.json, and metadata.json.
    HOLDOUT_FRAC -- fraction of the chronological window carved off before any
        feature engineering (default 0.18; set RETRAIN_HOLDOUT_FRAC=0 to
        disable). The served model trains on the remainder only and is then
        judged on this untouched slice.
    HOLDOUT_PF_CONFIDENCE -- 0.95. One-sided confidence level for the holdout
        profit-factor gate. The verdict gates on the Clopper-Pearson LOWER
        bound of the macro win rate mapped through PF, not the point
        estimate, because a PF on 55-82 trades flips with the clock (audit
        2026-08-24). Exact for independent trades; overlapping 45-bar walks
        make it conservative in practice. Deliberately not env-tunable.
    _holdout_pf_lower_bound -- one-sided lower confidence bound
        (Clopper-Pearson) on holdout PF from (wins, trades) and the bracket
        multiples. CP rather than Wilson because Wilson under-covers below
        ~40 trades, exactly the regime that used to flip.
    _holdout_verdict -- applies the holdout bars (CI-bound PF, Brier, EV) to
        an _evaluate_holdout score dict. NaN metrics fail loudly rather than
        passing vacuously.
    _tail_cutoff_by_symbol -- per-symbol timestamp where the unresolvable
        tail begins: the last max_hold bars per symbol, whose macro walk runs
        off the end of the frame and reads "timeout -> loss" regardless of
        the true outcome. Derived from the RAW series, applied after
        engineering.
    _purge_boundary_tail -- drops engineered rows at/after their symbol's
        tail cutoff; returns the frame and how many rows went.
    _score_artifact_holdout -- engineers the holdout slice (same parameters
        as the remainder, no cross-boundary access), purges its unresolvable
        tail, and scores the artifact with the frozen production threshold.
        Used by main() for both the pass-verdict and the fold-fail diagnostic.
    _split_holdout -- carves the chronologically last slice from raw fetched
        bars. Returns (remainder, holdout, date_range). All training and fold
        validation use the remainder; the holdout is engineered separately after
        the fold gate and scored with the frozen production threshold.
    _evaluate_holdout -- scores the final served artifact on the holdout using
        the production threshold from validate_candidate. Restricts scoring to
        tradeable instruments, mirroring the fold gate. No parameter choice on
        the holdout. The score dict carries wins/losses so the confidence
        bound can be recomputed from the raw evidence.
    HoldoutMetrics -- the artifact holdout scorecard: used flag, fraction, date
        range, Brier/EV/win rate/PF/trade counts, bypass_reason when the
        holdout was disabled or empty, plus wins, the PF lower confidence
        bound, purged tail-row count, and diagnostic_only (fold gate already
        failed; verdict not applied).
    USE_HMM_FEATURES -- off by default (RETRAIN_USE_HMM=1 to enable). Adds
        3 hidden-regime probability features, fit per fold to avoid leakage.
    _SPREAD_TABLE_PATH / SPREAD_TABLE -- optional per-instrument trading-cost
        table (RETRAIN_SPREAD_TABLE). When set, the cost gate uses each
        instrument's measured cost instead of one flat assumption, a
        ``cost_ratio`` feature is added so the model can see cost directly, and
        the table is copied next to the model on promotion. Unset, runs are
        bit-identical to before that experiment.

    BASE_FEATURE_COLS -- the 17 always-on model inputs (5 dead/memorized
        features dropped 2026-08-29 to prevent overfit): 10 single-bar
        indicators, 2 higher-timeframe context features (htf_trend_agreement,
        htf_bb_pct_b), 1 volatility compression feature (range_coil_10), and
        4 one-hot trading-session flags.
    FEATURE_COLS -- BASE_FEATURE_COLS plus cost_ratio and/or the HMM columns
        when those experiments are enabled. This exact list and order must
        match what the live strategy feeds the model.

    FoldMetrics -- one walk-forward fold's scorecard: sizes, Brier, expected
        value, win rate, and both trade counts.
    angel_proposed_trades vs devil_approved_trades -- how many bars stage one
        liked versus how many survived stage two. The ratio is the Devil's
        selectivity.
    ValidationReport -- the aggregate verdict across folds, plus gate_passed
        and the human-readable rejection_reasons list that gets posted to
        Discord.
    chop_veto_rate -- fraction of training rows the chop veto discarded.
    effective_trade_floor -- the drop-rate-scaled trade count actually required.

    fetch_training_data -- pulls raw bars per symbol from the configured
        provider and stacks them into one frame.
    _compute_devil_targets_atr -- the MACRO label: replay each bar forward up
        to max_hold and record whether target or stop came first. Stop is
        checked first each bar, so a bar touching both is scored a loss.
    _compute_devil_survival_target -- the label actually trained on: did price
        avoid the stop for the next 5 bars. Introduced because the Devil's
        inputs describe a 1-5 minute horizon, so asking it about a 45-bar
        outcome was an unlearnable mismatch.
    _compute_chop_veto_mask -- vectorised copy of the live pre-trade veto, so
        the model only ever learns from bars the live bot would actually
        trade. Gate A rejects bars where the stop is too tight relative to
        trading cost; Gate B rejects bars whose volatility sits too low in its
        own recent range (dead, choppy conditions).
    engineer_features_and_labels -- runs the feature pipeline, builds both
        labels, applies the veto, and returns the clean training frame.
    generate_time_decay_weights -- weights recent rows more heavily
        (decay_factor 0.95) so the model leans toward current market behaviour.
    refit_models -- trains the Angel then the Devil on one window.
    _find_optimal_threshold -- sweeps candidate Devil cut-offs and picks the
        one maximising expected value, subject to a minimum trade count.
    validate_candidate -- the fold gate: 3 expanding walk-forward folds,
        each trained on the past and scored on the future it never saw. The
        fold schedule scales with the actual span of the input frame, so it
        runs unchanged on the remainder after a holdout is carved off. The
        final model is trained on the same input frame only after the fold gate
        passes; main() passes the remainder so the served model never sees the
        holdout.
    _GATE_THRESHOLDS -- the promotion bars bundled for the Discord embed, so
        the notification cannot drift from the constants it quotes.
    _resolved_model_dir -- where this run's artifacts land; mirrors
        get_asset_config's resolution.
    _is_production_model_dir -- False when RETRAIN_MODEL_DIR redirected the run
        to a side directory, which suppresses every "now live" claim in the
        Discord report. Passing the gate is not the same as deploying.
    promote_or_reject -- the single decision point. On pass it writes the model
        files; on fail it returns False and the previous production weights are
        left untouched.
    save_models / save_threshold / save_spread_table -- the atomic writers for
        the artifacts listed above.
    Exit codes -- 0 promoted, 1 execution error, 2 trained but rejected. Note
        that 2 is not a failure of this script; it means the gate did its job.
    MODEL_FAMILY -- which estimator class refit_models/validate_candidate build:
        "lightgbm" (default, incumbent) or "catboost" (Stage-2 A/B candidate).
        Override with the env var MODEL_FAMILY directly (no RETRAIN_ prefix,
        unlike the rest of this module's env knobs).
    make_classifier -- returns an unfitted Angel/Devil classifier for the
        active MODEL_FAMILY; every LGBMClassifier(**params) construction site
        routes through it so the family is a single seam, not four.
    _catboost_params -- translates a LightGBM param dict into CatBoost's
        vocabulary (iterations/depth/min_data_in_leaf/rsm/random_seed),
        dropping LightGBM-only keys by name and intersecting monotone
        constraints with the features actually present.
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
_project_root = str(Path(__file__).resolve().parent.parent.parent)
_src_dir = str(Path(__file__).resolve().parent.parent)
_core_dir = str(Path(__file__).resolve().parent)
while _core_dir in sys.path:
    sys.path.remove(_core_dir)
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
# ATR BRACKET PARAMETERS (must match evaluate_performance.py and LiveOrchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

SL_ATR_MULTIPLIER = 0.5
TP_ATR_MULTIPLIER = 3.0
MAX_HOLD_BARS = 45
SURVIVAL_BARS = 5  # Phase 5.5: Devil survival window (bars)

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
    # Macro (45-bar bracket) wins among this fold's scored approvals — the
    # raw (wins, trades) evidence the pooled Clopper-Pearson PF lower bound
    # is computed from. 0 on the degenerate worst-case paths.
    macro_wins: int = 0


@dataclass
class HoldoutMetrics:
    """Metrics for the chronologically last holdout slice on the final artifact."""

    used: bool
    fraction: float
    start_date: Optional[str]
    end_date: Optional[str]
    brier_score: float
    expected_value: float
    win_rate: float
    profit_factor: float
    trades: int
    angel_proposed_trades: int
    bypass_reason: Optional[str] = None
    # Macro wins among the scored trades -- the PF gate's raw evidence, carried
    # so the confidence bound can be recomputed from metadata alone.
    wins: int = 0
    # Lower confidence bound on the holdout PF (see HOLDOUT_PF_CONFIDENCE).
    # 0.0 for a zero-trade scored holdout; None when the holdout was never
    # scored (bypassed, disabled, or no fold models to score).
    pf_lower_bound: Optional[float] = None
    pf_confidence: Optional[float] = None
    # Rows dropped at the tail of this slice because their macro walk ran off
    # the end of the fetched window (unresolvable outcome).
    purged_tail_rows: int = 0
    # True when the fold gate already failed and the holdout was scored for
    # diagnostics only -- the metrics inform, the fold verdict stands.
    diagnostic_only: bool = False


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
    pooled_oos_trades: int = 0
    chop_veto_rate: float = 0.0
    effective_trade_floor: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)
    holdout: Optional[HoldoutMetrics] = None
    # Pooled macro wins across folds — with pooled_oos_trades, the evidence
    # behind pooled_pf_lower_bound.
    pooled_oos_wins: int = 0
    # Clopper-Pearson PF lower bounds (see _holdout_pf_lower_bound) on the
    # pooled fold trades and on Fold 3 alone — the fold gate's evidential
    # bars, replacing the old flat 300-trade floor and Fold-3 point PF.
    pooled_pf_lower_bound: float = 0.0
    fold3_pf_lower_bound: float = 0.0
    # The Angel proposal bar the returned models were trained with:
    # calibrated from OOF probabilities unless ANGEL_THRESHOLD pinned a fixed
    # value. Written into threshold.json on promotion; MLStrategy prefers the
    # pinned value over the constant, keeping train/serve symmetry.
    production_angel_threshold: float = 0.0


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
    logger.info(f"Timeframe: {timeframe_minutes}-minute bars")

    all_frames: List[pl.DataFrame] = []
    failed_symbols: List[str] = []

    # OANDA intermittently rejects burst requests with a 401 ("Insufficient
    # authorization") that succeeds seconds later; the provider returns an
    # empty frame on any error, so retry-on-empty covers both cases.
    fetch_retries = int(os.getenv("RETRAIN_FETCH_RETRIES", "3"))

    for ticker in symbols:
        df = None
        for attempt in range(1, fetch_retries + 1):
            try:
                df = provider.get_historical_bars(
                    symbol=ticker,
                    timeframe_minutes=timeframe_minutes,
                    start=start_date,
                    end=end_date,
                )
            except Exception as e:
                df = None
                logger.error(f"Error fetching {ticker} (attempt {attempt}/{fetch_retries}): {e}")
            if df is not None and not df.is_empty():
                break
            logger.warning(f"No data returned for {ticker} (attempt {attempt}/{fetch_retries})")
            if attempt < fetch_retries:
                time.sleep(5 * attempt)

        if df is None or df.is_empty():
            failed_symbols.append(ticker)
            continue

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # Add symbol column if not present
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(ticker).alias("symbol"))

        all_frames.append(df)
        logger.info(f"Fetched {len(df):,} bars for {ticker}")

    if failed_symbols:
        # Training on a silently shrunk basket corrupts the experiment AND the
        # metadata sidecar (trained_on_symbols would claim the full list).
        raise ValueError(
            f"Fetch failed for {failed_symbols} after {fetch_retries} attempts — "
            "refusing to train on a partial basket"
        )

    if not all_frames:
        raise ValueError("No data fetched for any symbol")

    # Combine all symbols
    combined = pl.concat(all_frames, how="vertical_relaxed")
    combined = combined.sort(["symbol", "timestamp"])

    logger.info(f"Combined dataset: {len(combined):,} total rows")
    return combined


def _split_holdout(
    df: pl.DataFrame, frac: float
) -> Tuple[pl.DataFrame, pl.DataFrame, Optional[Tuple[datetime, datetime]]]:
    """
    Carve a chronologically last holdout slice from raw fetched data.

    The split is by timestamp BEFORE feature engineering, so no indicator,
    label, veto, or model parameter can see across the boundary. Returns
    ``(remainder, holdout, (holdout_start, holdout_end))``. When ``frac`` is 0
    or the holdout would be empty, the holdout frame is empty and the date
    range is None.

    Args:
        df: Raw stacked bars with a ``timestamp`` column.
        frac: Fraction of the chronological span to reserve (0.0–1.0).

    Returns:
        Tuple of (remainder DataFrame, holdout DataFrame, optional date range).
    """
    if frac <= 0.0 or df.is_empty():
        return df, pl.DataFrame(), None

    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    span_seconds = (max_ts - min_ts).total_seconds()
    if span_seconds <= 0:
        return df, pl.DataFrame(), None

    holdout_seconds = span_seconds * frac
    holdout_start = max_ts - timedelta(seconds=holdout_seconds)
    # Keep the boundary clean: remainder is strictly before holdout_start,
    # holdout is from holdout_start onward. A bar exactly on the boundary
    # belongs to the holdout.
    remainder = df.filter(pl.col("timestamp") < holdout_start)
    holdout = df.filter(pl.col("timestamp") >= holdout_start)

    if holdout.is_empty():
        return remainder, holdout, None

    return remainder, holdout, (holdout_start, max_ts)


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
    df: pl.DataFrame,
    profile: RiskProfile,
    sl_mult: float,
    alpha_table: Optional[dict] = None,
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

    ``alpha_table`` (2026-07-07): optional per-instrument spread alphas
    ({symbol: alpha_emp} from a bake_spread_alphas.py table). When provided,
    each symbol's measured alpha replaces the flat ``profile.spread_atr_alpha``
    in Gate A — instruments the table doesn't list fall back to the profile
    value. This fixes the asymmetry where training priced GBP_NZD (measured
    ~0.90) at the flat 0.15 and kept setups live always vetoes.

    Returns a boolean array (True = veto/drop) aligned to ``df`` rows. Rows are
    dropped only as trade *entries*; the bracket walk in the target functions
    still sees the full contiguous price path (so this must run AFTER target
    generation, not before).

    Gate C (time-of-day blackout) is mirrored here since 2026-09-09: the same
    America/New_York window as ``RiskManager._in_blackout`` (DST-correct,
    start-inclusive / end-exclusive, midnight wrap), applied to the rows' UTC
    timestamps (naive timestamps are assumed UTC, matching the live gate). The
    live bot drops NY-rollover entries (≈16:55–17:30 ET); training now vetoes
    the same bars instead of labelling un-executable rollover entries.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n_total = df.height
    veto = np.zeros(n_total, dtype=bool)
    if not _chop_filter_enabled() or n_total == 0:
        return veto

    # Gate C (time-of-day blackout) — vectorized mirror of
    # RiskManager._in_blackout, DST-correct via America/New_York conversion.
    # Naive timestamps are assumed UTC (the live gate does the same).
    if (
        profile.time_gate_enabled
        and profile.blackout_start is not None
        and profile.blackout_end is not None
        and "timestamp" in df.columns
    ):
        ts = df["timestamp"]
        if getattr(ts.dtype, "time_zone", None) is None:
            ts = ts.dt.replace_time_zone("UTC")
        ny = ts.dt.convert_time_zone("America/New_York")
        # .dt.hour() is Int8 — cast before multiplying or 22*3600 overflows.
        secs = (
            ny.dt.hour().cast(pl.Int64) * 3600
            + ny.dt.minute().cast(pl.Int64) * 60
            + ny.dt.second().cast(pl.Int64)
        )
        start_s = profile.blackout_start.hour * 3600 + profile.blackout_start.minute * 60
        end_s = profile.blackout_end.hour * 3600 + profile.blackout_end.minute * 60
        if start_s <= end_s:
            gate_c = (secs >= start_s) & (secs < end_s)
        else:
            gate_c = (secs >= start_s) | (secs < end_s)  # window wraps midnight
        veto |= gate_c.to_numpy()

    w = int(profile.regime_window)
    mins = int(profile.regime_min_samples)
    p_thresh = profile.regime_pctile / 100.0

    symbols = df["symbol"].to_numpy() if "symbol" in df.columns else np.zeros(n_total)
    natr_all = df["natr_14"].to_numpy().astype(float)

    for sym in np.unique(symbols):
        # Per-instrument measured alpha when a table is provided; flat profile
        # value otherwise (and for symbols the table doesn't list).
        alpha = (alpha_table or {}).get(str(sym), profile.spread_atr_alpha)
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

        # OR into the existing veto (Gate C may already have marked rows):
        # assigning here would silently clobber the blackout mask.
        veto[idx] |= gate_a | gate_b
        # Per-symbol diagnostics — critical when alpha_table is active: an
        # expensive instrument (GBP_NZD ~0.90) should thin dramatically.
        logger.info(
            "  Chop veto [%s]: alpha=%.4f | gate_a=%d gate_b=%d gate_c=%d | vetoed %d/%d (%.1f%%)",
            sym, alpha, int(gate_a.sum()), int(gate_b.sum()),
            int(veto[idx].sum() - (gate_a | gate_b).sum()),
            int(veto[idx].sum()), m,
            100.0 * veto[idx].mean() if m else 0.0,
        )

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
    angel_mult: Optional[float] = None,
    risk_profile: Optional[RiskProfile] = None,
    alpha_table: Optional[dict] = None,
) -> Tuple[pl.DataFrame, List[str], float]:
    """
    Engineer technical features and generate ATR-dynamic target labels.

    Delegates feature computation to FeaturePipeline to
    guarantee zero training/inference skew with the production MLStrategy.

    Features (produced by FeaturePipeline, matches MLStrategy.feature_names):
        rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct,
        price_sma50_ratio, log_return, hour_of_day, dist_sma50, vol_rel

    Targets:
        angel_target: 1 if close 3 bars ahead > close + angel_mult × ATR
            (ATR-relative). angel_mult falls back to sl_mult when not given,
            but production passes it explicitly -- the Angel's momentum bar is
            not the execution stop. See _ANGEL_ATR_MULT_BY_CLASS.
        devil_target: 1 if TP (tp_mult × ATR) hit before SL (sl_mult × ATR) in ≤max_hold bars

    Args:
        df: Raw OHLCV DataFrame with columns:
            open, high, low, close, volume, symbol, timestamp
        sl_mult: Stop-loss ATR multiplier (Devil labels + chop veto)
        angel_mult: ATR multiple for the Angel's 3-bar momentum bar. None
            falls back to sl_mult (the legacy coupling).
        tp_mult: Take-profit ATR multiplier
        max_hold: Maximum hold bars
        survival_bars: Number of survival bars for devil training
        htf_timeframe: Higher timeframe representation for HTFFeatures

    Returns:
        Tuple of (features DataFrame with targets, feature column names,
        chop_veto_rate) — the row-drop fraction feeds the pooled dynamic
        trade-count floor in validate_candidate().
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
            # No-op when alpha_table is None; needs natr_14 → after V3Base.
            V3CostFeatures(
                alpha_table=alpha_table,
                default_alpha=(
                    risk_profile.spread_atr_alpha if risk_profile else 0.15
                ),
                regime_window=(
                    risk_profile.regime_window if risk_profile else 260
                ),
            ),
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
    _angel_mult = sl_mult if angel_mult is None else angel_mult
    df = df.with_columns(
        (
            pl.col("close").shift(-3).over("symbol")
            > pl.col("close") + _angel_mult * (pl.col("close") * pl.col("natr_14") / 100.0)
        )
        .cast(pl.Int8)
        .alias("angel_target")
    )
    logger.info(
        f"Generated angel_target (ATR-relative 3-bar momentum with "
        f"angel_mult={_angel_mult}; execution sl_mult={sl_mult})"
    )

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
    chop_veto_rate = 0.0
    if risk_profile is not None:
        pre_veto = df.height
        veto_mask = _compute_chop_veto_mask(
            df, risk_profile, sl_mult, alpha_table=alpha_table
        )
        n_veto = int(veto_mask.sum())
        chop_veto_rate = n_veto / pre_veto if pre_veto else 0.0
        if n_veto > 0:
            df = df.filter(~pl.Series(veto_mask))
        logger.info(
            f"Hybrid chop veto dropped {n_veto:,} untradeable rows "
            f"({chop_veto_rate:.1%}; mode={risk_profile.spread_k_coupling_mode}, "
            f"k_base={risk_profile.spread_k_base}, coupling={risk_profile.spread_k_coupling}, "
            f"P{risk_profile.regime_pctile:.0f}, alpha={risk_profile.spread_atr_alpha})"
        )

    # ═══════════════════════════════════════════════════════════════════
    # BEHAVIOR VETO (experimental, OFF unless RETRAIN_BEHAVIOR_VETO is set)
    # ═══════════════════════════════════════════════════════════════════
    # Drops bars whose market behavior is a measured money-loser — 2026-08-23
    # put `trend_high` at 19-26% wins against a 33.3% break-even, the only
    # behavior cell significantly negative in every run.
    #
    # ⚠️ TRAIN/SERVE SYMMETRY: a model trained with this veto MUST be served
    # behind a matching live gate, or it is skew of exactly the kind the chop
    # veto exists to avoid. The label is written into metadata.json
    # ("behavior_veto") so a candidate declares the gate it requires; nothing
    # serves it today. Applied AFTER target generation for the same reason as
    # the chop veto — the bracket walk needs the contiguous price path.
    if BEHAVIOR_VETO_LABELS:
        from ml.regimes.behavior_tagger import tag_series, trend_strength_from_ppo

        if "symbol" in df.columns and "ppo" in df.columns:
            tagged = []
            for sym in df["symbol"].unique(maintain_order=True).to_list():
                d = df.filter(pl.col("symbol") == sym).sort("timestamp")
                tags = tag_series(
                    d["natr_14"].to_numpy().astype(float),
                    trend_strength_from_ppo(d["ppo"].to_numpy().astype(float)),
                )
                tagged.append(
                    d.with_columns(pl.Series("behavior_label", [t.label for t in tags]))
                )
            df = pl.concat(tagged)
            before = df.height
            df = df.filter(~pl.col("behavior_label").is_in(list(BEHAVIOR_VETO_LABELS)))
            n_behavior = before - df.height
            # Fold into chop_veto_rate so the gate's dynamic trade floor scales
            # with the combined drop; otherwise the veto is penalised purely
            # for trading less.
            chop_veto_rate = 1.0 - (df.height / pre_veto) if pre_veto else 0.0
            logger.info(
                "Behavior veto dropped %s rows tagged %s (%.1f%% of %s); "
                "combined drop rate now %.1f%%",
                f"{n_behavior:,}", sorted(BEHAVIOR_VETO_LABELS),
                100.0 * n_behavior / before if before else 0.0, f"{before:,}",
                100.0 * chop_veto_rate,
            )
        else:
            logger.warning(
                "RETRAIN_BEHAVIOR_VETO=%s set but the frame lacks symbol/ppo — "
                "behavior veto SKIPPED", sorted(BEHAVIOR_VETO_LABELS),
            )

    # ═══════════════════════════════════════════════════════════════════
    # CLEANUP: Drop NaN/null rows (uses FeaturePipeline.clean_data)
    # ═══════════════════════════════════════════════════════════════════
    initial_count = len(df)
    # cost_ratio exists iff an alpha_table was provided (V3CostFeatures is a
    # no-op otherwise) — include it in cleaning and the returned schema so it
    # reaches the models. Keyed off the function param, not the module global,
    # so behavior follows what was actually computed.
    base_cols = BASE_FEATURE_COLS + (["cost_ratio"] if alpha_table else [])
    # Clean on BASE features only — HMM regime probs (when enabled) are
    # appended later inside validate_candidate so each fold fits its own HMM.
    df = FeaturePipeline.clean_data(
        df, feature_cols=base_cols + ["angel_target", "devil_target"]
    )
    dropped_count = initial_count - len(df)

    logger.info(
        f"Dropped {dropped_count:,} rows with nulls ({dropped_count / initial_count:.1%})"
    )
    logger.info(f"Final dataset: {len(df):,} rows")
    logger.info(f"Base feature columns ({len(base_cols)}): {base_cols}")
    if USE_HMM_FEATURES:
        logger.info(f"HMM regime features ENABLED — will be appended per-fold: {HMM_OUTPUT_COLS}")

    return df, base_cols, chop_veto_rate


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-DECAY WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════


def generate_time_decay_weights(
    n_samples: int,
    decay_factor: float = 0.95,
    timestamps: Optional["pl.Series"] = None,
) -> np.ndarray:
    """
    Generate time-decay sample weights.

    More recent samples get higher weights to prevent catastrophic forgetting.
    Weights decay exponentially from 1.0 (most recent) to 0.1 (oldest).

    ``timestamps`` (2026-09-09): the training frame is symbol-blocked
    (sorted ["symbol","timestamp"]), so row INDEX encodes basket position,
    not recency — the old row-index weighting silently upweighted the
    last-listed instruments (verified: symbol 0's newest bar got weight 0.1,
    symbol N's newest got 1.0) and reordering RETRAIN_SYMBOLS changed the
    model. When timestamps are passed, weight by dense rank of the timestamp
    instead.

    Args:
        n_samples: Number of samples in dataset
        decay_factor: Decay rate per time step (default: 0.95)
        timestamps: Optional polars Series of per-row timestamps

    Returns:
        NumPy array of sample weights
    """
    if timestamps is not None and n_samples > 0:
        # Dense rank: 1..n in chronological order, ties share a rank.
        ranks = timestamps.rank("dense").to_numpy().astype(float)
    else:
        ranks = np.arange(1, n_samples + 1, dtype=float)

    # Exponential decay from the newest rank down to the oldest.
    weights = np.power(decay_factor, ranks.max() - ranks)

    if weights.max() - weights.min() < 1e-12:
        # Degenerate (all rows same rank / single sample): uniform weights.
        return np.full(n_samples, 1.0)

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
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
) -> Tuple["lgb.LGBMClassifier", "lgb.LGBMClassifier", List[str], List[str], float]:
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
        sl_mult: Stop-loss ATR multiplier (Angel threshold EV objective)
        tp_mult: Take-profit ATR multiplier (Angel threshold EV objective)

    Returns:
        Tuple of (Angel model, Devil model, angel_features, devil_features,
        angel_threshold) — the proposal bar the Devil's training population
        was filtered at: calibrated from OOF probabilities unless the
        ANGEL_THRESHOLD env var pinned a fixed value. Callers must use THIS
        value (not the global constant) for validation masking and artifact
        pinning, or the served model runs at a different bar than its Devil
        and brackets were fitted for.
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

    # Generate time-decay weights — ranked by TIMESTAMP, not row index
    # (the frame is symbol-blocked; see generate_time_decay_weights).
    ts_col = df["timestamp"] if "timestamp" in df.columns else None
    sample_weights = generate_time_decay_weights(
        len(X_base), timestamps=ts_col
    )
    logger.info(
        f"Time-decay weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}"
    )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Train the Angel (Primary Model - Direction)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 1/4] Training Angel model (Direction)...")
    angel_model = make_classifier(a_params, feature_cols)
    df_base = pl.DataFrame(X_base, schema=feature_cols).to_pandas()
    angel_model.fit(df_base, y_angel, sample_weight=sample_weights)
    logger.info(
        f"✓ Angel model trained on {len(feature_cols)} features "
        f"(family={MODEL_FAMILY})"
    )

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
    # 2026-09-09 CRITICAL FIX: the frame is symbol-blocked (sorted
    # ["symbol","timestamp"]), so row index is NOT a chronological axis —
    # TimeSeriesSplit over raw indices made each val fold one or two WHOLE
    # symbol blocks, and the "OOF" Angel probabilities for late-listed
    # symbols came from models trained on OTHER symbols' full history
    # INCLUDING dates after the scored row. On correlated FX pairs that is
    # genuine future leakage into the Devil's key meta-feature AND into the
    # threshold calibration. The split now runs on a chronological
    # permutation and probabilities are written back to original indices.
    #
    # n_splits=5: 5 expanding folds. Early folds → noisier Angel probs,
    # which is realistic (production Angel also starts uncertain).

    tss = TimeSeriesSplit(n_splits=5)
    angel_probs_oof = np.full(len(X_base), np.nan)

    ts_vals = df["timestamp"].to_numpy()
    perm = np.argsort(ts_vals, kind="stable")
    X_perm = X_base[perm]
    y_perm = y_angel[perm]
    w_perm = sample_weights[perm]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        for fold_train_idx, fold_val_idx in tss.split(X_perm):
            fold_weights = w_perm[fold_train_idx]
            fold_angel = make_classifier(a_params, feature_cols)
            fold_angel.fit(
                X_perm[fold_train_idx],
                y_perm[fold_train_idx],
                sample_weight=fold_weights,
            )
            probs = fold_angel.predict_proba(X_perm[fold_val_idx])[:, 1]
            angel_probs_oof[perm[fold_val_idx]] = probs

        # Fill train-only head (the EARLIEST rows never appear in val — in
        # chronological space now, not basket space) using a model trained
        # solely on that window. In-sample for the head itself, but no
        # future leakage: the head is the oldest slice and its model has
        # seen nothing newer.
        head_missing = np.isnan(angel_probs_oof)
        if head_missing.sum() > 0:
            first_train_idx, _ = next(iter(tss.split(X_perm)))
            head_angel = make_classifier(a_params, feature_cols)
            head_angel.fit(
                X_perm[first_train_idx],
                y_perm[first_train_idx],
                sample_weight=w_perm[first_train_idx],
            )
            angel_probs_oof[head_missing] = head_angel.predict_proba(
                X_base[head_missing]
            )[:, 1]
            logger.info(
                f"  Head fill: {head_missing.sum()} earliest rows scored by "
                f"Fold-1 Angel (in-sample for the head, no future leakage)"
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
    # STEP 2.5: Calibrate the Angel proposal bar (unless env-pinned fixed)
    # ═══════════════════════════════════════════════════════════════════
    # The bar must be chosen HERE, on train-frame OOF probabilities, because
    # the next step filters the Devil's training population with it — and the
    # caller needs the same value for validation masking and artifact pinning.
    # Chosen on OOF (not in-sample) scores so the bar reflects honest Angel
    # confidence, the same reason the Devil trains on OOF meta-features.
    if _FIXED_ANGEL_THRESHOLD:
        angel_threshold = ANGEL_THRESHOLD
        logger.info(
            f"Angel threshold FIXED by ANGEL_THRESHOLD env var: {angel_threshold:.4f}"
        )
    else:
        angel_threshold, angel_thr_ev, angel_thr_n = _find_optimal_angel_threshold(
            angel_probs_oof,
            df["devil_target_macro"].to_numpy(),
            sl_mult=sl_mult,
            tp_mult=tp_mult,
        )
        logger.info(
            f"Dynamic Angel threshold: {angel_threshold:.4f} "
            f"(OOF-calibrated: EV {angel_thr_ev:+.4f}, {angel_thr_n:,} proposals "
            f"= {angel_thr_n / len(angel_probs_oof):.2%} of train rows; "
            f"set ANGEL_THRESHOLD to pin a fixed bar)"
        )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Train the Devil (Meta Model - Conviction)
    # Phase 5.5: Train ONLY on Angel-approved subpopulation.
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 3/4] Training Devil model (Conviction with meta-features)...")

    devil_features = feature_cols + ["angel_prob"]
    X_devil_full = df[devil_features].to_numpy()

    # Phase 5.5 — Population Fix:
    # The Devil is deployed exclusively on rows where angel_prob >= the
    # proposal bar. Training on the full global population (all ~117k rows)
    # violates meta-labeling semantics: the Devil learns to discriminate
    # across all market conditions, not within the Angel-approved subset
    # where it actually operates.
    # Solution: filter training data to only Angel-approved rows using the OOF
    # angel_probs (already computed in Step 2 — zero leakage).
    angel_approved_mask = angel_probs_oof >= angel_threshold
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

    # Devil-scale min_child to the APPROVED population, not the Angel's —
    # see _devil_min_child for why; RETRAIN_DEVIL_MIN_CHILD pins a fixed value.
    if DEVIL_MIN_CHILD_FIXED:
        devil_min_child = int(DEVIL_MIN_CHILD_FIXED)
    else:
        devil_min_child = _devil_min_child(
            int(d_params["min_child_samples"]), n_approved
        )
    if devil_min_child != d_params["min_child_samples"]:
        logger.info(
            f"Devil min_child_samples auto-scaled: "
            f"{d_params['min_child_samples']} -> {devil_min_child} "
            f"(approved population n={n_approved:,}; a split needs "
            f">= {2 * devil_min_child} rows)"
        )
    d_params_fit = {**d_params, "min_child_samples": devil_min_child}

    devil_model = make_classifier(d_params_fit, devil_features)
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
        # CatBoost's score() takes no sample_weight; its training-accuracy line
        # is summary telemetry only, so skip the weighting for that family
        # rather than fake the metric.
        if MODEL_FAMILY == "catboost":
            angel_acc = angel_model.score(df_base, y_angel)
            devil_acc = devil_model.score(df_devil, y_devil_train)
        else:
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

    return angel_model, devil_model, feature_cols, devil_features, angel_threshold


def _devil_min_child(configured: int, n_approved: int) -> int:
    """
    Scale the Devil's min_child_samples to its actual training population.

    min_child_samples is a per-LEAF minimum: splitting a node needs at least
    2x that many rows. The Angel-side value (80) applied to an Angel-approved
    subpopulation of dozens-to-hundreds makes every split impossible, and the
    Devil degenerates to a constant (2026-08-29 gate matrix: separation gap
    0.0000, 100% approval — a two-stage architecture on one stage). A tenth
    of the population keeps leaves meaningful at any scale; the cap preserves
    the configured value when the population is large (old behaviour
    unchanged); the floor keeps LightGBM off degenerate 1-row leaves.
    """
    return max(5, min(int(configured), n_approved // 10))


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


def _find_optimal_angel_threshold(
    angel_probs: np.ndarray,
    macro_targets: np.ndarray,
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    min_proposals: int = MIN_ANGEL_PROPOSALS,
) -> Tuple[float, float, int]:
    """
    Calibrate the Angel's proposal bar from out-of-fold probabilities.

    Why this exists: the Angel threshold was a global constant (0.40) while
    model capacity changes the score distribution underneath it. The trim
    100x15/min_child=80 architecture compresses predicted probabilities so
    severely that a fixed 0.40 bar passed 11/3/11 proposals per ~103k-row
    fold (2026-08-29 matrix) — starving both the Devil's training population
    and the fold gate's trade count. Expressing the bar in the model's OWN
    score units (quantiles of its OOF distribution) makes the proposal rate
    a property of the evidence, not of an arbitrary constant.

    Discipline mirrors _find_optimal_threshold (Devil), with one role
    difference: the Angel is the RECALL stage, so the sweep maximizes EV
    subject to a minimum proposal count that keeps the Devil's training
    population learnable — precision is the Devil's job downstream.

    Candidates are quantiles of the OOF scores (median through max), so the
    grid adapts to any distribution shape, including compressed ones. The
    sweep runs on TRAIN-frame OOF probabilities only; the chosen value is
    then applied frozen to validation/holdout/live scoring, so no validation
    information leaks into the parameter (same discipline as the Devil's
    frozen calibration_threshold).

    Args:
        angel_probs:   OOF Angel probabilities on the training frame.
        macro_targets: 45-bar bracket outcome ground truth (0/1), aligned
                       with angel_probs. Used for the EV objective.
        sl_mult:       Stop-loss ATR multiplier.
        tp_mult:       Take-profit ATR multiplier.
        min_proposals: Minimum proposal count for a candidate to be valid
                       (guarantees the Devil a training population).

    Returns:
        Tuple of (threshold, ev_at_threshold, n_proposals). When no candidate
        yields min_proposals (frame smaller than min_proposals), returns the
        observed minimum score — propose everything and let the gate judge.
    """
    n = len(angel_probs)
    if n == 0:
        return 0.5, -float("inf"), 0

    # Quantile grid from the median up to the observed max; dedupe guards a
    # distribution compressed to (near-)constant scores.
    grid_q = np.linspace(0.50, 1.0 - 1.0 / n, 400)
    candidates = np.unique(np.quantile(angel_probs, grid_q))

    rr_ratio = tp_mult / sl_mult
    best_threshold = float(candidates[0])
    best_ev = -float("inf")
    best_n = 0

    for t in candidates:
        mask = angel_probs >= t
        n_approved = int(mask.sum())
        if n_approved < min_proposals:
            continue
        win_rate = float(macro_targets[mask].mean())
        ev = win_rate * rr_ratio - (1.0 - win_rate)
        if ev > best_ev:
            best_ev = ev
            best_threshold = float(t)
            best_n = n_approved

    if best_ev == -float("inf"):
        # No candidate met the proposal floor (frame smaller than
        # min_proposals): propose EVERYTHING — the frame is degenerate
        # anyway, and starving the Devil of the little data that exists
        # only makes it worse. The fold gate judges the result.
        best_threshold = float(angel_probs.min())
        mask = angel_probs >= best_threshold
        best_n = int(mask.sum())
        win_rate = float(macro_targets[mask].mean()) if best_n else 0.0
        best_ev = win_rate * rr_ratio - (1.0 - win_rate)
        logger.warning(
            "Angel threshold sweep: no candidate reached min_proposals=%d "
            "(frame n=%d) — falling back to propose-everything %.4f "
            "(%d proposals)",
            min_proposals, n, best_threshold, best_n,
        )

    return best_threshold, float(best_ev), best_n


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION GATE
# ═══════════════════════════════════════════════════════════════════════════════


def _tradeable_scoring_mask(
    val_df: pl.DataFrame,
    signal_mask: np.ndarray,
    approved_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Narrow an approval mask to instruments the account can actually trade.

    Returns ``(scored_mask, n_excluded)``, where ``scored_mask`` is aligned to
    the Angel-proposed subset exactly like ``approved_mask`` is. A gate metric
    is a prediction about live results, so a trade that could never be placed
    must not move it — see :data:`UNTRADEABLE_SYMBOLS` for why this inverts the
    verdict rather than merely adding noise.

    Training is unaffected: this runs on validation approvals only, well after
    the models have been fit on the full basket.

    Falls back to the unmodified mask when nothing is marked untradeable or the
    frame carries no ``symbol`` column, so single-symbol callers and older
    frames behave exactly as before.
    """
    if not UNTRADEABLE_SYMBOLS or "symbol" not in val_df.columns:
        return approved_mask, 0

    proposed_symbols = (
        val_df.filter(pl.Series(signal_mask))["symbol"]
        .cast(pl.Utf8)
        .str.to_uppercase()
        .to_numpy()
    )
    tradeable = ~np.isin(proposed_symbols, list(UNTRADEABLE_SYMBOLS))
    scored_mask = approved_mask & tradeable
    return scored_mask, int(approved_mask.sum() - scored_mask.sum())


def _capture_oos_ledger(
    ledger: List[pl.DataFrame],
    val_df: pl.DataFrame,
    signal_mask: np.ndarray,
    approved_mask: np.ndarray,
    macro_targets: np.ndarray,
    devil_probs: np.ndarray,
    angel_probs: np.ndarray,
    fold_number: int,
    carry_cols: Sequence[str],
) -> None:
    """
    Append this fold's Devil-approved OOS trades to ``ledger``.

    Read-only with respect to the gate: it observes masks the fold has already
    computed and appends a frame. Nothing here can change a promotion decision.

    The rows are strictly out-of-sample by construction — ``val_df`` is the
    fold's validation window and the models were fit on the train window only,
    so the ledger inherits the walk-forward's honesty rather than re-deriving
    it. Columns absent from the frame are skipped rather than raising, so a
    caller may ask for optional feature columns without knowing the schema.
    """
    proposed = val_df.filter(pl.Series(signal_mask))
    approved = proposed.filter(pl.Series(approved_mask))
    if approved.height == 0:
        return

    present = [c for c in carry_cols if c in approved.columns]
    ledger.append(
        approved.select(present).with_columns(
            [
                pl.Series("fold", np.full(approved.height, fold_number, dtype=np.int32)),
                pl.Series("macro_win", macro_targets[approved_mask].astype(np.int8)),
                pl.Series("devil_prob", devil_probs[approved_mask].astype(np.float64)),
                pl.Series("angel_prob", angel_probs[approved_mask].astype(np.float64)),
            ]
        )
    )


def _evaluate_holdout(
    holdout_df: pl.DataFrame,
    angel_model: "lgb.LGBMClassifier",
    devil_model: "lgb.LGBMClassifier",
    angel_features: List[str],
    devil_features: List[str],
    threshold: float,
    sl_mult: float,
    tp_mult: float,
    angel_threshold: Optional[float] = None,
) -> dict:
    """
    Score the final served artifact on the held-out chronologically last slice.

    Uses the FROZEN production thresholds returned by validate_candidate — no
    parameter is chosen or tuned on the holdout. ``threshold`` is the Devil
    bar; ``angel_threshold`` is the proposal bar the Angel/Devil pair was
    trained with (OOF-calibrated unless env-pinned). When None it falls back
    to the global ANGEL_THRESHOLD constant — the pre-2026-08-29 behaviour,
    kept for older tooling that predates calibrated Angel bars. Scoring is
    restricted to tradeable instruments for the same reason the fold gate
    does: the metric is a prediction about live results.

    Returns a dict with the holdout metrics and an ``approved_mask`` aligned to
    the Angel-proposed subset, mirroring the fold computation.
    """
    n_total = holdout_df.height
    if n_total == 0:
        return {
            "brier_score": float("nan"),
            "expected_value": float("nan"),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trades": 0,
            "angel_proposed_trades": 0,
            "devil_approved_raw": 0,
            "approved_mask": np.array([], dtype=bool),
            "wins": 0,
            "losses": 0,
        }

    X_base = holdout_df[angel_features].to_numpy()
    y_angel = holdout_df["angel_target"].to_numpy()
    y_devil = holdout_df["devil_target"].to_numpy()
    y_devil_macro = holdout_df["devil_target_macro"].to_numpy()

    if angel_threshold is None:
        angel_threshold = ANGEL_THRESHOLD

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        angel_probs = angel_model.predict_proba(X_base)[:, 1]

    signal_mask = angel_probs >= angel_threshold
    n_angel_proposed = int(signal_mask.sum())

    if n_angel_proposed == 0:
        return {
            "brier_score": float("nan"),
            "expected_value": float("nan"),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trades": 0,
            "angel_proposed_trades": 0,
            "devil_approved_raw": 0,
            "approved_mask": np.array([], dtype=bool),
            "wins": 0,
            "losses": 0,
        }

    proposed_base = X_base[signal_mask]
    proposed_angel_probs = angel_probs[signal_mask]
    proposed_devil_targets = y_devil[signal_mask]
    proposed_devil_targets_macro = y_devil_macro[signal_mask]

    meta_df = pl.DataFrame(proposed_base, schema=angel_features).with_columns(
        pl.Series("angel_prob", proposed_angel_probs)
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        devil_proba_full = devil_model.predict_proba(meta_df)

    if devil_proba_full.shape[1] == 1:
        only_class = int(devil_model.classes_[0])
        const_prob = 1.0 if only_class == 1 else 0.0
        devil_probs = np.full(len(meta_df), const_prob, dtype=np.float64)
    else:
        devil_probs = devil_proba_full[:, 1]

    approved_mask = devil_probs >= threshold
    n_devil_approved_raw = int(approved_mask.sum())

    scored_mask, n_excluded = _tradeable_scoring_mask(
        holdout_df, signal_mask, approved_mask
    )
    approved_mask = scored_mask
    n_approved = int(approved_mask.sum())

    if n_approved == 0:
        return {
            "brier_score": float("nan"),
            "expected_value": float("nan"),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trades": 0,
            "angel_proposed_trades": n_angel_proposed,
            "devil_approved_raw": n_devil_approved_raw,
            "approved_mask": approved_mask,
            "wins": 0,
            "losses": 0,
        }

    approved_devil_probs = devil_probs[approved_mask]
    approved_targets = proposed_devil_targets[approved_mask]
    approved_macro = proposed_devil_targets_macro[approved_mask]

    brier = float(brier_score_loss(approved_targets, approved_devil_probs))

    macro_wins = int(approved_macro.sum())
    macro_losses = n_approved - macro_wins
    gross_profit = macro_wins * tp_mult
    gross_loss = macro_losses * sl_mult
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )
    macro_wr = float(approved_macro.mean()) if n_approved > 0 else 0.0

    # EV from the MACRO approval rate (2026-09-09): the old code multiplied
    # the 5-bar SURVIVAL win rate by the 45-bar MACRO R:R — survival is not
    # the complement of macro loss (a trade can survive 5 bars and still
    # time out), so the value systematically overstated per-trade expectancy
    # (the served artifact recorded 1.58 against a 0.0005 bar — the gate
    # could not fail).
    ev = float(macro_wr * (tp_mult / sl_mult) - (1.0 - macro_wr))

    return {
        "brier_score": brier,
        "expected_value": ev,
        "win_rate": macro_wr,
        "profit_factor": profit_factor,
        "trades": n_approved,
        "angel_proposed_trades": n_angel_proposed,
        "devil_approved_raw": n_devil_approved_raw,
        "approved_mask": approved_mask,
        "n_excluded_untradeable": n_excluded,
        "wins": macro_wins,
        "losses": macro_losses,
    }


def _holdout_pf_lower_bound(
    wins: int,
    trades: int,
    sl_mult: float,
    tp_mult: float,
    confidence: float = HOLDOUT_PF_CONFIDENCE,
) -> float:
    """
    Exact lower confidence bound on the holdout profit factor.

    The old point-estimate gate flipped with the clock because a PF on 55-82
    trades cannot separate 0.98 from 1.44 (audit 2026-08-24, three window
    endpoints). Rather than pretend the sample is larger, the gate demands
    what the sample can actually prove: the Clopper-Pearson one-sided lower
    bound on the macro win rate p, mapped through the PF formula
    PF = p*tp / ((1-p)*sl). A promotion now requires the holdout to exclude
    break-even at ``confidence``. The bound is exact for independent trades;
    the 45-bar macro walks overlap in price, so in practice it is
    conservative rather than a literal coverage guarantee -- which is the
    safe direction for a promotion gate.

    Clopper-Pearson, not the Wilson approximation: Wilson under-covers at
    n < ~40, the exact regime that used to flip. Example: a perfect 3-for-3
    holdout clears Wilson's bound, but CP's is 0.3684 -- below the 0.375
    break-even at 2:1 -- so it is (correctly) rejected; a perfect 4-for-4
    (CP 0.4729) passes.

    Args:
        wins: Macro wins among the scored holdout trades.
        trades: Scored holdout trades (tradeable approvals).
        sl_mult: Stop-loss ATR multiple of the bracket.
        tp_mult: Take-profit ATR multiple of the bracket.
        confidence: One-sided confidence level (default HOLDOUT_PF_CONFIDENCE).

    Returns:
        Lower confidence bound on PF; 0.0 when trades == 0.
    """
    if trades <= 0:
        return 0.0
    wins = min(max(int(wins), 0), int(trades))
    if wins == 0:
        # scipy's beta.ppf is undefined at a=0; the bound on a win rate with
        # zero observed wins is exactly 0.
        return 0.0
    # Beta(1-confidence; wins, losses+1) quantile; exactly 0 at wins == 0.
    p_lb = float(_scipy_beta.ppf(1.0 - confidence, wins, trades - wins + 1))
    return (p_lb * tp_mult) / ((1.0 - p_lb) * sl_mult)


def _holdout_verdict(
    scores: dict, sl_mult: float, tp_mult: float
) -> Tuple[bool, List[str]]:
    """
    Apply the artifact holdout bars to an ``_evaluate_holdout`` score dict.

    Returns ``(passed, reasons)``. The PF bar gates on the exact confidence
    lower bound rather than the point estimate; Brier and EV keep their point
    bars (both were stable across the audit's windows). The ``not (x <= bar)``
    forms are deliberate: a zero-trade holdout returns NaN metrics, and NaN
    comparisons are False -- that shape makes NaN fail loudly instead of
    passing vacuously.
    """
    reasons: List[str] = []
    pf_lb = _holdout_pf_lower_bound(
        scores["wins"], scores["trades"], sl_mult, tp_mult
    )
    if not (pf_lb >= PROFIT_FACTOR_THRESHOLD):
        reasons.append(
            f"Holdout PF point estimate {scores['profit_factor']:.4f} on "
            f"{scores['trades']} trades, but {HOLDOUT_PF_CONFIDENCE:.0%} lower "
            f"bound {pf_lb:.4f} < {PROFIT_FACTOR_THRESHOLD} -- the sample cannot "
            f"support a pass"
        )
    if not (scores["brier_score"] <= BRIER_THRESHOLD):
        reasons.append(
            f"Holdout Brier {scores['brier_score']:.4f} > {BRIER_THRESHOLD}"
        )
    if not (scores["expected_value"] >= EV_THRESHOLD):
        reasons.append(
            f"Holdout EV {scores['expected_value']:.6f} < {EV_THRESHOLD}"
        )
    return len(reasons) == 0, reasons


def _tail_cutoff_by_symbol(raw_df: pl.DataFrame, max_hold: int) -> dict:
    """
    Per-symbol timestamp where the unresolvable tail begins.

    A row whose ``max_hold``-bar bracket walk needs bars beyond the frame's
    end resolves as "timeout -> loss" even though its true outcome is
    unknowable -- the last ``max_hold`` bars per symbol of a raw slice are
    systematically wrong labels, not evidence. The cutoff MUST be derived
    from the RAW series (the walk's true domain, before the chop veto removes
    rows) and applied AFTER engineering, because the walk itself needs the
    contiguous price path.

    Returns ``{symbol: first_unresolvable_timestamp}``. Symbols with at most
    ``max_hold`` bars are absent: every one of their rows is unresolvable,
    and dropping them wholesale could empty a degenerate frame entirely, so
    they are kept untouched as the defensive call.
    """
    cutoffs: dict = {}
    for sym in raw_df["symbol"].unique(maintain_order=True).to_list():
        ts = raw_df.filter(pl.col("symbol") == sym)["timestamp"].sort()
        if len(ts) > max_hold:
            cutoffs[sym] = ts[max(0, len(ts) - max_hold)]
    return cutoffs


def _purge_boundary_tail(
    df: pl.DataFrame, cutoffs: dict
) -> Tuple[pl.DataFrame, int]:
    """
    Drop engineered rows at or after their symbol's tail cutoff.

    Returns ``(frame, n_dropped)``. Symbols without a cutoff are untouched;
    rows are removed after engineering so the veto, labels, and walks all saw
    the full contiguous path.
    """
    if not cutoffs or df.is_empty():
        return df, 0
    kept_parts: List[pl.DataFrame] = []
    n_dropped = 0
    for sym in df["symbol"].unique(maintain_order=True).to_list():
        sub = df.filter(pl.col("symbol") == sym)
        if sym in cutoffs:
            before = sub.height
            sub = sub.filter(pl.col("timestamp") < cutoffs[sym])
            n_dropped += before - sub.height
        if not sub.is_empty():
            kept_parts.append(sub)
    if not kept_parts:
        # Every surviving row sat in a purged tail (possible in degenerate
        # tiny frames): return an empty frame with the same schema rather
        # than handing back rows the count says were dropped.
        return df.clear(), n_dropped
    return pl.concat(kept_parts), n_dropped


def _score_artifact_holdout(
    holdout_raw: pl.DataFrame,
    angel_model: "lgb.LGBMClassifier",
    devil_model: "lgb.LGBMClassifier",
    angel_features: List[str],
    devil_features: List[str],
    threshold: float,
    asset_config: dict,
    alpha_table: Optional[dict] = None,
    final_hmm_models: Optional[dict] = None,
    angel_threshold: Optional[float] = None,
) -> Tuple[dict, int]:
    """
    Engineer the holdout slice and score the artifact on it with the frozen
    production threshold.

    The holdout is engineered separately, with the same parameters as the
    remainder but no access to it -- no indicator, label, veto, threshold, or
    model parameter may see across the boundary. The slice's own last
    ``max_hold`` bars per symbol are purged before scoring: their macro walk
    runs off the end of the fetched window and would read as guaranteed
    losses, a pessimistic bias on the gate's own evidence (the mirror image
    of the remainder purge). The indicator warm-up at the slice head is
    already consumed by NaN cleaning during engineering.

    Returns ``(scores, n_purged)`` where ``scores`` is the
    ``_evaluate_holdout`` dict.
    """
    sl_mult = asset_config["sl_mult"]
    tp_mult = asset_config["tp_mult"]
    holdout_features, _, _ = engineer_features_and_labels(
        holdout_raw,
        sl_mult=sl_mult,
        angel_mult=asset_config["angel_mult"],
        tp_mult=tp_mult,
        max_hold=asset_config["max_hold"],
        survival_bars=asset_config["survival_bars"],
        htf_timeframe=asset_config.get("htf_timeframe", "5m"),
        risk_profile=RiskProfile.for_asset_class(asset_config["asset_class"]),
        alpha_table=alpha_table,
    )
    cutoffs = _tail_cutoff_by_symbol(holdout_raw, asset_config["max_hold"])
    holdout_features, n_purged = _purge_boundary_tail(holdout_features, cutoffs)
    if n_purged:
        logger.info(
            "HOLDOUT TAIL PURGE: dropped %d rows whose %d-bar macro walk ran "
            "off the end of the fetched window (unresolvable outcome)",
            n_purged, asset_config["max_hold"],
        )
    if USE_HMM_FEATURES and final_hmm_models is not None and not holdout_features.is_empty():
        holdout_features = predict_regime_probs(holdout_features, final_hmm_models)
    scores = _evaluate_holdout(
        holdout_features,
        angel_model,
        devil_model,
        angel_features,
        devil_features,
        threshold,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        angel_threshold=angel_threshold,
    )
    return scores, n_purged


def validate_candidate(
    df: pl.DataFrame,
    feature_cols: List[str],
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    n_folds: int = 3,
    angel_params: Optional[dict] = None,
    devil_params: Optional[dict] = None,
    chop_veto_rate: float = 0.0,
    oos_ledger: Optional[List[pl.DataFrame]] = None,
    oos_ledger_cols: Sequence[str] = (
        "timestamp",
        "symbol",
        "natr_14",
        "ppo",
        "close",
        "behavior_label",
    ),
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
    Run expanding-window walk-forward cross-validation and apply the fold gate.

    Splits the supplied frame into 3 expanding folds by calendar date (not row
    index) so that all symbols' data for a given date range stays in the same
    fold. For each fold, trains Angel + Devil on the training window and
    evaluates strictly out-of-sample on the validation window.

    Fold schedule (calendar days from the earliest timestamp in df):
        Fold 1: Train 0–½,   Validate ½–⅔
        Fold 2: Train 0–⅔,   Validate ⅔–⅚
        Fold 3: Train 0–⅚,   Validate ⅚–1   ← Profit Factor gate
    The fractions reproduce the legacy 60-day schedule exactly (30/40, 40/50,
    50/60) and scale automatically to the actual span of the input frame.

    TEMPORAL BOUNDARY:
        The Profit Factor gate uses the Fold 3 model evaluated on the Fold 3
        val set. This is strictly OOS. The final model is trained on the input
        frame only AFTER the fold gate passes. In main(), the input frame is
        the remainder after the holdout is carved off, so the served model
        never sees the holdout.

    Promotion thresholds:
        Mean Brier Score   ≤ 0.30   (across all folds)
        Mean EV            ≥ 0.0005 (across all folds)
        Fold 3 PF lb       ≥ 1.20   (Clopper-Pearson bound, most recent fold)
        Pooled PF lb       ≥ 1.20   (Clopper-Pearson bound, all folds pooled)
        Pooled trades      ≥ backstop (BASELINE_POOLED_OOS_TRADES, veto-scaled)
    The two PF bars replaced the old Fold-3 point PF and the flat 300-trade
    floor on 2026-08-29 — one exact instrument (_holdout_pf_lower_bound),
    applied at two scales, so small samples widen the interval and fail on
    their own instead of flipping with the clock.

    Dynamic thresholds:
        With class_weight=None, Devil probabilities reflect the true ~20% base
        rate. Per-fold, _find_optimal_threshold() sweeps 0.10–0.64 and selects
        the threshold maximizing EV. The Fold 3 threshold is returned as the
        production threshold.
        The ANGEL proposal bar is likewise calibrated per refit from that
        frame's OOF probabilities (_find_optimal_angel_threshold) unless the
        ANGEL_THRESHOLD env var pins a fixed value; the fold's own calibrated
        bar masks its validation window (strictly OOS), and the final
        artifact's bar is pinned into threshold.json via the report's
        production_angel_threshold.

    Args:
        df: Feature-engineered DataFrame (output of
            engineer_features_and_labels). main() passes the remainder after
            the holdout carve-out.
        feature_cols: List of base feature column names.
        sl_mult: Stop-loss ATR multiplier
        tp_mult: Take-profit ATR multiplier
        n_folds: Number of expanding folds (default: 3).

    Returns:
        Tuple of:
            - ValidationReport (fold gate decision + per-fold metrics)
            - angel_model (final if gate passed, Fold 3 if rejected)
            - devil_model (final if gate passed, Fold 3 if rejected)
            - angel_feature_names
            - devil_feature_names
            - production_threshold (optimal Devil threshold from Fold 3)
    """
    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION (3 EXPANDING FOLDS)")
    logger.info("=" * 70)

    # Pin the process RNG for fully deterministic sweeps across the cached
    # datasets. (LightGBM is already seeded via random_state=42 + deterministic;
    # this guards any incidental numpy randomness in the validation path.)
    np.random.seed(42)

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
    # Build date-based fold boundaries from the actual frame span.
    # ───────────────────────────────────────────────────────────────────
    min_date = df["timestamp"].min()

    # The fold schedule is derived from the actual span of the supplied frame,
    # not the module constant DAYS_BACK. This lets the same 3-fold expanding
    # shape run on the remainder after a holdout is carved off, while keeping
    # the legacy proportions intact (60 days → 30/40, 40/50, 50/60 exactly).
    # main() still fetches with days_back=DAYS_BACK and warns if the fetched
    # window does not match, but the folds themselves scale with the data they
    # are given.
    span_days = (df["timestamp"].max() - min_date).total_seconds() / 86400.0
    span = int(span_days)
    if span_days > DAYS_BACK * 1.2:
        logger.warning(
            "Data spans %.0f days but DAYS_BACK=%d — the fold schedule now "
            "scales to the full frame, so this run's gate is not comparable to "
            "a DAYS_BACK=%d run. Set RETRAIN_DAYS_BACK=%d if that was intended.",
            span_days, DAYS_BACK, DAYS_BACK, int(span_days),
        )

    # fold_configs: (train_end_days, val_end_days) — exclusive upper bounds.
    # Fractions chosen to reproduce the legacy 60-day schedule exactly:
    #   (30,40),(40,50),(50,60) at span=60.
    # For smaller/larger windows the same expanding-train / fixed-fraction-val
    # shape scales (e.g. span=50 → (25,33),(33,41),(41,50)).
    fold_configs = [
        (span // 2,     span * 2 // 3),   # train 0–½,  val ½–⅔
        (span * 2 // 3, span * 5 // 6),   # train 0–⅔,  val ⅔–⅚
        (span * 5 // 6, span),            # train 0–⅚,  val ⅚–1
    ]

    fold_metrics: List[FoldMetrics] = []

    # Placeholders for Fold 3 outputs (used for PF gate and fallback)
    fold3_angel: Optional["lgb.LGBMClassifier"] = None
    fold3_devil: Optional["lgb.LGBMClassifier"] = None
    fold3_angel_feats: Optional[List[str]] = None
    fold3_devil_feats: Optional[List[str]] = None
    fold3_angel_threshold: float = ANGEL_THRESHOLD  # fallback; overwritten per fold
    # Production HMM dict (fit on full data after the gate passes; persisted
    # alongside Angel/Devil and consumed at inference by MLStrategy).
    final_hmm_models: Optional[dict] = None
    profit_factor: float = 0.0
    final_win_rate: float = 0.0
    final_total_trades: int = 0
    fold3_macro_wins: int = 0
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
        # Train on this fold's training window. The fold's Angel proposal
        # bar comes back calibrated from ITS train-frame OOF probabilities
        # (unless env-pinned fixed) — applying it to the val window is
        # strictly OOS, the same discipline as the Devil's frozen
        # calibration_threshold.
        # ─────────────────────────────────────────────────────────────
        (
            angel_model,
            devil_model,
            angel_feats,
            devil_feats,
            fold_angel_threshold,
        ) = refit_models(
            train_df,
            feature_cols,
            angel_params=angel_params,
            devil_params=devil_params,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
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

        signal_mask = angel_probs_val >= fold_angel_threshold
        n_angel_proposed = int(signal_mask.sum())
        logger.info(
            f"[Fold {fold_number}] Angel proposed {n_angel_proposed} trades "
            f"({n_angel_proposed / len(val_df):.1%} of val rows) "
            f"@ threshold {fold_angel_threshold:.4f}"
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
                fold3_angel_threshold = fold_angel_threshold
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
                fold3_angel_threshold = fold_angel_threshold
            continue

        # Opt-in per-trade OOS capture for offline analysis (behavior matrix).
        # Purely observational; `oos_ledger is None` in every production path.
        if oos_ledger is not None:
            _capture_oos_ledger(
                oos_ledger,
                val_df,
                signal_mask,
                approved_mask,
                proposed_devil_targets_macro,
                devil_probs_val,
                proposed_angel_probs,
                fold_number,
                oos_ledger_cols,
            )

        # ─────────────────────────────────────────────────────────────
        # Restrict SCORING (not training) to instruments the account can
        # actually trade. A gate metric is a prediction about live results, so
        # a trade that could never be placed must not move it. Training keeps
        # every instrument — see UNTRADEABLE_SYMBOLS.
        # ─────────────────────────────────────────────────────────────
        scored_mask, n_excluded = _tradeable_scoring_mask(
            val_df, signal_mask, approved_mask
        )
        if n_excluded:
            logger.info(
                "[Fold %d] Gate scoring excludes %d untradeable approvals "
                "(%s); %d scored. Training basket is unchanged.",
                fold_number, n_excluded, ",".join(sorted(UNTRADEABLE_SYMBOLS)),
                int(scored_mask.sum()),
            )

        n_scored = int(scored_mask.sum())
        if n_scored == 0:
            logger.warning(
                "[Fold %d] Every approval was in an untradeable instrument — "
                "no scoreable trades; setting worst-case metrics",
                fold_number,
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
                fold3_angel_threshold = fold_angel_threshold
            continue

        # From here on the gate sees only tradeable approvals.
        approved_mask = scored_mask
        n_devil_approved = n_scored

        # Macro (45-bar bracket) outcomes of the scored approvals — the
        # (wins, trades) evidence the pooled Clopper-Pearson PF lower bound
        # pools across EVERY fold, not just Fold 3.
        approved_macro_targets = proposed_devil_targets_macro[approved_mask]
        fold_macro_wins = int(approved_macro_targets.sum())

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
        # EV from the MACRO win rate (2026-09-09) — the fold's own bracket
        # outcome, not the 5-bar survival rate mapped through macro R:R.
        macro_win_rate = (
            fold_macro_wins / n_devil_approved if n_devil_approved > 0 else 0.0
        )
        ev = float(
            macro_win_rate * (tp_mult / sl_mult) - (1.0 - macro_win_rate)
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
            macro_wins=fold_macro_wins,
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
            fold3_angel_threshold = fold_angel_threshold

            # Phase 5.5: Profit Factor and win rate computed from MACRO
            # outcomes (45-bar bracket) on Devil-approved trades.
            # approved_targets (survival) is used for Brier only — the PF
            # gate must reflect actual bracket R:R, not survival rate.
            fold3_macro_wins = fold_macro_wins
            macro_losses = n_devil_approved - fold3_macro_wins
            gross_profit = fold3_macro_wins * tp_mult
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

    # Fold-gate evidence, judged with the same exact instrument as the
    # artifact holdout gate: Clopper-Pearson lower bound on the macro win
    # rate mapped through PF (_holdout_pf_lower_bound). This replaces two
    # predecessors the 2026-08-29 gate matrix showed were broken: the flat
    # 300-trade floor (a cliff the shipped 200x63 config itself cleared only
    # once in three pins) and the Fold-3 POINT PF (a coin-flip on 11 trades).
    # One instrument, applied at two scales:
    #   Fold 3  — the recency check: does the LATEST regime beat break-even?
    #   Pooled  — the evidence check: does the WHOLE walk-forward prove it?
    # plus an absolute backstop count so a handful of lucky wins can never
    # reach the instrument at all.
    pooled_oos_trades = int(sum(fm.devil_approved_trades for fm in fold_metrics))
    pooled_oos_wins = int(sum(fm.macro_wins for fm in fold_metrics))
    effective_trade_floor = BASELINE_POOLED_OOS_TRADES * (1.0 - chop_veto_rate)
    pooled_pf_lb = _holdout_pf_lower_bound(
        pooled_oos_wins, pooled_oos_trades, sl_mult, tp_mult
    )
    fold3_pf_lb = _holdout_pf_lower_bound(
        fold3_macro_wins, final_total_trades, sl_mult, tp_mult
    )

    rejection_reasons: List[str] = []
    if mean_brier > BRIER_THRESHOLD:
        rejection_reasons.append(
            f"Brier {mean_brier:.4f} > {BRIER_THRESHOLD} threshold"
        )
    if mean_ev < EV_THRESHOLD:
        rejection_reasons.append(f"EV {mean_ev:.6f} < {EV_THRESHOLD} threshold")
    if not (fold3_pf_lb >= PROFIT_FACTOR_THRESHOLD):
        rejection_reasons.append(
            f"Fold {n_folds} PF point estimate {profit_factor:.4f} on "
            f"{final_total_trades} trades, but {HOLDOUT_PF_CONFIDENCE:.0%} lower "
            f"bound {fold3_pf_lb:.4f} < {PROFIT_FACTOR_THRESHOLD} — the most "
            f"recent regime cannot prove it beats break-even"
        )
    if not (pooled_pf_lb >= PROFIT_FACTOR_THRESHOLD):
        rejection_reasons.append(
            f"Pooled fold PF {HOLDOUT_PF_CONFIDENCE:.0%} lower bound "
            f"{pooled_pf_lb:.4f} < {PROFIT_FACTOR_THRESHOLD} "
            f"(evidence: {pooled_oos_wins} wins / {pooled_oos_trades} trades "
            f"across {n_folds} folds)"
        )
    if pooled_oos_trades < effective_trade_floor:
        rejection_reasons.append(
            f"Pooled OOS trades {pooled_oos_trades} < backstop floor "
            f"{effective_trade_floor:.0f} "
            f"(= {BASELINE_POOLED_OOS_TRADES} × (1 − chop_veto_rate {chop_veto_rate:.1%})) "
            f"— too few trades for any statistical claim"
        )

    gate_passed = len(rejection_reasons) == 0

    logger.info("=" * 70)
    logger.info("VALIDATION GATE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Mean Brier Score : {mean_brier:.4f} (threshold ≤ {BRIER_THRESHOLD})")
    logger.info(f"Mean EV          : {mean_ev:.6f} (threshold ≥ {EV_THRESHOLD})")
    logger.info(
        f"Fold {n_folds} PF     : {profit_factor:.4f} point | "
        f"{HOLDOUT_PF_CONFIDENCE:.0%} lower bound {fold3_pf_lb:.4f} "
        f"(bar ≥ {PROFIT_FACTOR_THRESHOLD}; {fold3_macro_wins}/{final_total_trades} wins)"
    )
    logger.info(
        f"Pooled PF (folds): {HOLDOUT_PF_CONFIDENCE:.0%} lower bound "
        f"{pooled_pf_lb:.4f} (bar ≥ {PROFIT_FACTOR_THRESHOLD}; "
        f"{pooled_oos_wins} wins / {pooled_oos_trades} trades)"
    )
    logger.info(
        f"Pooled OOS Trades: {pooled_oos_trades} "
        f"(backstop ≥ {effective_trade_floor:.0f} = "
        f"{BASELINE_POOLED_OOS_TRADES}×(1−{chop_veto_rate:.1%}))"
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
        (
            final_angel,
            final_devil,
            final_angel_feats,
            final_devil_feats,
            production_angel_threshold,
        ) = refit_models(
            df,
            feature_cols,
            angel_params=angel_params,
            devil_params=devil_params,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
        )
    else:
        logger.info(
            "Gate failed — skipping full-data training. Production weights retained."
        )
        final_angel = fold3_angel
        final_devil = fold3_devil
        final_angel_feats = fold3_angel_feats
        final_devil_feats = fold3_devil_feats
        production_angel_threshold = fold3_angel_threshold

    report = ValidationReport(
        fold_metrics=fold_metrics,
        mean_brier=mean_brier,
        mean_ev=mean_ev,
        final_profit_factor=profit_factor,
        final_win_rate=final_win_rate,
        final_total_trades=final_total_trades,
        gate_passed=gate_passed,
        pooled_oos_trades=pooled_oos_trades,
        chop_veto_rate=chop_veto_rate,
        effective_trade_floor=effective_trade_floor,
        rejection_reasons=rejection_reasons,
        pooled_oos_wins=pooled_oos_wins,
        pooled_pf_lower_bound=pooled_pf_lb,
        fold3_pf_lower_bound=fold3_pf_lb,
        production_angel_threshold=production_angel_threshold,
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


# Single source of truth for the bars the Discord embed quotes. Kept next to
# promote_or_reject so it cannot drift from the constants above the way the
# hardcoded copies in notification_manager.py did.
_GATE_THRESHOLDS = {
    "brier": BRIER_THRESHOLD,
    "ev": EV_THRESHOLD,
    "profit_factor": PROFIT_FACTOR_THRESHOLD,
}


def _resolved_model_dir(asset_config: Optional[dict]) -> str:
    """Where this run's artifacts land — the same resolution get_asset_config does."""
    cfg = asset_config or {}
    return cfg.get("model_dir") or f"models/{cfg.get('asset_class', 'equities')}"


def _is_production_model_dir() -> bool:
    """
    False when RETRAIN_MODEL_DIR redirected this run to a side directory.

    An explicit override means the operator deliberately aimed somewhere other
    than the default path, so the run must not be announced as going live.
    """
    return not os.getenv("RETRAIN_MODEL_DIR", "").strip()


def promote_or_reject(
    report: ValidationReport,
    angel_model: "lgb.LGBMClassifier",
    devil_model: "lgb.LGBMClassifier",
    threshold: float = 0.20,
    asset_config: dict = None,
    hmm_models: Optional[dict] = None,
    angel_threshold: Optional[float] = None,
) -> bool:
    """
    Promote or reject candidate models based on the validation report.

    If gate passed: the models passed in were trained on the frame supplied to
    validate_candidate() (the remainder after the holdout carve-out when the
    holdout is enabled). Saves them atomically via save_models(), then saves
    the optimal threshold via save_threshold().

    If gate failed: the models passed in are Fold 3 models (not saved).
    Production weights are retained, rejection alert is sent to Discord.

    Args:
        report: ValidationReport from validate_candidate()
        angel_model: Final model (gate-passed if gate passed, Fold 3 if failed)
        devil_model: Final model (gate-passed if gate passed, Fold 3 if failed)
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

        save_models(angel_model, devil_model, asset_config, report=report)
        save_threshold(threshold, asset_config, angel_threshold=angel_threshold)
        if SPREAD_TABLE is not None:
            save_spread_table(_SPREAD_TABLE_PATH, asset_config)
        if hmm_models is not None:
            asset_class = asset_config.get("asset_class", "equities")
            model_dir = Path(asset_config.get("model_dir") or f"models/{asset_class}")
            hmm_path = model_dir / "hmm_latest.pkl"
            save_hmm_models(hmm_models, hmm_path)

        notifier.send_retraining_report(
            report,
            promoted=True,
            model_dir=_resolved_model_dir(asset_config),
            is_production_path=_is_production_model_dir(),
            gate_thresholds=_GATE_THRESHOLDS,
        )
        return True
    else:
        logger.warning("=" * 70)
        logger.warning("🚫 VALIDATION GATE FAILED — MODELS REJECTED")
        logger.warning("=" * 70)
        for reason in report.rejection_reasons:
            logger.warning(f"  Rejection: {reason}")

        notifier.send_retraining_report(
            report,
            promoted=False,
            model_dir=_resolved_model_dir(asset_config),
            is_production_path=_is_production_model_dir(),
            gate_thresholds=_GATE_THRESHOLDS,
        )
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SERIALIZATION (ATOMIC)
# ═══════════════════════════════════════════════════════════════════════════════


def save_models(
    angel_model: "lgb.LGBMClassifier",
    devil_model: "lgb.LGBMClassifier",
    asset_config: dict,
    report: Optional[ValidationReport] = None,
) -> None:
    """
    Serialize models to disk using joblib with POSIX atomic writes.

    Uses temporary files and os.replace() to ensure zero-downtime atomic swaps,
    preventing the live hot-reloader from reading partially written files.

    Args:
        angel_model: Trained Angel model
        devil_model: Trained Devil model
        asset_config: Asset configuration dictionary.
        report: Optional ValidationReport; when present, holdout metadata is
            written to metadata.json so a served model can be checked against
            what it actually earned.
    """
    logger.info("=" * 70)
    logger.info("SERIALIZING MODELS (ATOMIC)")
    logger.info("=" * 70)

    asset_class = asset_config.get("asset_class", "equities")
    model_dir = Path(asset_config.get("model_dir") or f"models/{asset_class}")
    model_dir.mkdir(parents=True, exist_ok=True)

    angel_path = model_dir / "angel_latest.pkl"
    devil_path = model_dir / "devil_latest.pkl"
    angel_temp = model_dir / "angel_temp.pkl"
    devil_temp = model_dir / "devil_temp.pkl"

    # ═══════════════════════════════════════════════════════════════════
    # TWO-PHASE ATOMIC WRITE (2026-09-09): dump BOTH pickles to temp files
    # first, then os.replace back-to-back. The old order (replace Angel,
    # THEN serialize Devil) left a window where a bar's hot-reload ingested
    # a NEW Angel beside an OLD Devil — and if serializing the Devil failed
    # (disk full, SIGKILL), the served pair stayed mixed until the next
    # retrain. The live strategy's pair-seam stand-down tolerates the much
    # smaller between-replaces gap.
    # ═══════════════════════════════════════════════════════════════════
    try:
        joblib.dump(angel_model, angel_temp)
        angel_size = angel_temp.stat().st_size / (1024 * 1024)
    except Exception as e:
        logger.error(f"[ATOMIC] Failed to serialize Angel model: {e}")
        if angel_temp.exists():
            angel_temp.unlink()
        raise

    try:
        joblib.dump(devil_model, devil_temp)
        devil_size = devil_temp.stat().st_size / (1024 * 1024)
    except Exception as e:
        logger.error(f"[ATOMIC] Failed to serialize Devil model: {e}")
        if angel_temp.exists():
            angel_temp.unlink()
        if devil_temp.exists():
            devil_temp.unlink()
        raise

    # Both serialized successfully — now the two replaces, back to back.
    os.replace(angel_temp, angel_path)
    logger.info(f"[ATOMIC] Angel model saved: {angel_path} ({angel_size:.1f} MB)")
    try:
        os.replace(devil_temp, devil_path)
        logger.info(f"[ATOMIC] Devil model saved: {devil_path} ({devil_size:.1f} MB)")
    except Exception as e:
        # Angel is already live but Devil is not: the live strategy's
        # pair-seam stand-down refuses to score the mixed pair, so the
        # window is guarded — log loudly rather than pretending all is well.
        logger.error(
            f"[ATOMIC] Angel replaced but Devil replace FAILED ({e}) — "
            "live pair is mixed; the strategy will stand down until the "
            "next retrain lands both."
        )
        if devil_temp.exists():
            devil_temp.unlink()
        raise

    # ═══════════════════════════════════════════════════════════════════
    # ATOMIC WRITE: Metadata sidecar
    # ═══════════════════════════════════════════════════════════════════
    metadata_path = model_dir / "metadata.json"
    metadata_temp = model_dir / "metadata_temp.json"
    # The Angel bar the pair was actually trained at: the OOF-calibrated
    # value when the report carries one (dynamic mode), else the global
    # constant (fixed mode or reports that predate calibration).
    meta_angel_threshold = ANGEL_THRESHOLD
    if report is not None and getattr(report, "production_angel_threshold", 0.0):
        meta_angel_threshold = report.production_angel_threshold
    metadata = {
        "asset_class": asset_class,
        "timeframe_minutes": asset_config.get("timeframe_minutes", 1),
        "htf_timeframe": asset_config.get("htf_timeframe", "5m"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on_symbols": asset_config.get("tickers", []),
        "data_source": os.getenv("DATA_SOURCE", "alpaca").strip().lower(),
        "angel_threshold": meta_angel_threshold,
        # The brackets this model's labels were built from. Serving it under
        # different multiples is train/serve skew, and until now the artifact
        # carried no record of them — so a candidate could not be checked
        # against the tree it was about to be served from.
        "sl_atr_multiplier": asset_config.get("sl_mult"),
        "tp_atr_multiplier": asset_config.get("tp_mult"),
        "lookback_days": DAYS_BACK,
        # Declares the live gate this artifact requires. A non-empty
        # list served without a matching veto is train/serve skew.
        "behavior_veto": sorted(BEHAVIOR_VETO_LABELS),
        # Holdout record: what the served artifact earned on data it never saw.
        # If the holdout was disabled or empty, "used" is false and the reason
        # is recorded so the artifact cannot be mistaken for one that passed a
        # real holdout gate.
        "holdout": {
            "used": False,
            "fraction": HOLDOUT_FRAC,
            "bypass_reason": "disabled" if HOLDOUT_FRAC <= 0.0 else None,
        },
    }
    if report is not None and report.holdout is not None:
        ho = report.holdout
        metadata["holdout"] = {
            "used": ho.used,
            "fraction": ho.fraction,
            "start_date": ho.start_date,
            "end_date": ho.end_date,
            "brier_score": (
                round(ho.brier_score, 4) if not np.isnan(ho.brier_score) else None
            ),
            "expected_value": (
                round(ho.expected_value, 6) if not np.isnan(ho.expected_value) else None
            ),
            "win_rate": round(ho.win_rate, 4) if ho.trades > 0 else None,
            "profit_factor": (
                round(ho.profit_factor, 4) if ho.trades > 0 else None
            ),
            "trades": ho.trades,
            "angel_proposed_trades": ho.angel_proposed_trades,
            # The PF verdict rests on the exact confidence bound, not the
            # point estimate; both are recorded so a served model can be
            # re-checked against the evidence that actually decided it.
            "wins": ho.wins,
            "pf_lower_bound": (
                round(ho.pf_lower_bound, 4) if ho.pf_lower_bound is not None else None
            ),
            "pf_confidence": ho.pf_confidence,
            "purged_tail_rows": ho.purged_tail_rows,
            "bypass_reason": ho.bypass_reason,
        }
    with open(metadata_temp, "w") as f:
        json.dump(metadata, f, indent=2)
    os.replace(metadata_temp, metadata_path)
    logger.info(f"[ATOMIC] Metadata saved: {metadata_path}")

    logger.info(
        "[ATOMIC] Model serialization complete — live bot can hot-reload safely"
    )


def save_threshold(
    threshold: float,
    asset_config: dict,
    angel_threshold: Optional[float] = None,
) -> None:
    """
    Save the optimal Devil threshold to disk as a JSON sidecar file.

    Written atomically alongside the model .pkl files.  The LiveOrchestrator
    and MLStrategy read this on startup and via hot-reload so the live bot
    always uses the threshold that maximises EV on the most recent data.

    Args:
        threshold: The optimal Devil probability threshold (e.g., 0.28)
        asset_config: Asset configuration dictionary.
        angel_threshold: The Angel proposal bar the pair was trained with
            (OOF-calibrated unless env-pinned). None falls back to the
            global ANGEL_THRESHOLD constant — the pre-2026-08-29 behaviour.
    """
    if angel_threshold is None:
        angel_threshold = ANGEL_THRESHOLD
    asset_class = asset_config.get("asset_class", "equities")
    model_dir = Path(asset_config.get("model_dir") or f"models/{asset_class}")
    model_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = model_dir / "threshold.json"

    data = {
        # Full precision on both bars: the Devil's training population and the
        # bracket fit were conditioned on the exact tuned floats, and
        # MLStrategy compares against this file. round(x, 4) here shifted the
        # live Angel bar ~5e-5 looser than the population the pair was
        # fitted for.
        "devil_threshold": threshold,
        # Pin the Angel bar the pair was trained at: the Devil's training
        # population and the bracket fit are conditioned on it, so the live
        # strategy must run the model at this value (MLStrategy overrides
        # its default with this key when present).
        "angel_threshold": angel_threshold,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Atomic write — same pattern as model serialisation
    temp_path = model_dir / "threshold_temp.json"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, threshold_path)

    logger.info(
        f"[ATOMIC] Threshold saved: {threshold_path} (devil_threshold={threshold:.4f})"
    )


def save_spread_table(source_path: str, asset_config: dict) -> None:
    """
    Copy the per-instrument spread-alpha table into the model directory as
    ``spread_alphas.json`` so the model and its cost assumptions travel
    together.  MLStrategy / run_oanda load it from the model dir — a model
    trained against one cost table must never run against another.

    Written atomically, same pattern as save_threshold().
    """
    asset_class = asset_config.get("asset_class", "equities")
    model_dir = Path(asset_config.get("model_dir") or f"models/{asset_class}")
    model_dir.mkdir(parents=True, exist_ok=True)
    table_path = model_dir / "spread_alphas.json"

    with open(source_path, "r") as f:
        table = json.load(f)

    temp_path = model_dir / "spread_alphas_temp.json"
    with open(temp_path, "w") as f:
        json.dump(table, f, indent=2)
    os.replace(temp_path, table_path)

    logger.info(
        f"[ATOMIC] Spread table saved: {table_path} "
        f"({len(table.get('alphas', {}))} instruments, "
        f"denomination={table.get('denomination_minutes')}m)"
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

        # ─── Phase 2a: Carve holdout FIRST, before any feature engineering ───
        # The holdout is the chronologically last slice. No indicator, label,
        # veto, threshold, or model parameter may see across this boundary.
        # RETRAIN_HOLDOUT_FRAC=0 disables the holdout and preserves legacy
        # full-data behaviour.
        remainder_raw, holdout_raw, holdout_range = _split_holdout(
            raw_data, HOLDOUT_FRAC
        )
        if HOLDOUT_FRAC <= 0.0:
            logger.warning(
                "⚠️  HOLDOUT DISABLED (RETRAIN_HOLDOUT_FRAC=0). "
                "The gate will judge fold models and the served artifact will be "
                "trained on all data — the exact leakage this gate was built to stop."
            )
        elif holdout_raw.is_empty():
            logger.warning(
                "⚠️  HOLDOUT REQUESTED BUT EMPTY — bypassing artifact holdout gate. "
                "Check the fetched window and RETRAIN_HOLDOUT_FRAC value."
            )
        else:
            logger.info(
                "HOLDOUT: reserved %.1f%% of chronological window (%.0f rows); "
                "training/validation will use %.0f rows",
                100.0 * HOLDOUT_FRAC,
                holdout_raw.height,
                remainder_raw.height,
            )
            if holdout_range:
                logger.info(
                    "HOLDOUT RANGE: %s → %s",
                    holdout_range[0].isoformat(),
                    holdout_range[1].isoformat(),
                )

        # ─── Phase 2.5: Per-instrument spread-cost table (optional) ────────
        spread_alphas: Optional[dict] = None
        if SPREAD_TABLE is not None:
            spread_alphas = SPREAD_TABLE["alphas"]
            denom = SPREAD_TABLE.get("denomination_minutes")
            if denom != asset_config["timeframe_minutes"]:
                logger.warning(
                    "⚠️  SPREAD TABLE DENOMINATION MISMATCH: table measured on "
                    "%s-minute bars but retraining on %s-minute bars. Baseline "
                    "NATR is timeframe-dependent — these alphas are NOT valid "
                    "here. Re-bake from a soak at this granularity.",
                    denom, asset_config["timeframe_minutes"],
                )
            logger.info(
                "Per-instrument spread table ACTIVE (%s): %s",
                _SPREAD_TABLE_PATH,
                {k: round(v, 4) for k, v in sorted(spread_alphas.items())},
            )
            logger.info(
                "cost_ratio feature enabled → %d features", len(FEATURE_COLS)
            )

        # ─── Phase 3: Engineer features with ATR-dynamic labels ────────────
        # Feature engineering runs ONLY on the remainder. The holdout is
        # engineered separately AFTER the fold gate, using the same parameters
        # but no information from the future.
        features_df, feature_cols, chop_veto_rate = engineer_features_and_labels(
            remainder_raw,
            sl_mult=asset_config["sl_mult"],
            angel_mult=asset_config["angel_mult"],
            tp_mult=asset_config["tp_mult"],
            max_hold=asset_config["max_hold"],
            survival_bars=asset_config["survival_bars"],
            htf_timeframe=asset_config.get("htf_timeframe", "5m"),
            # Same RiskProfile path that sources sl_mult/tp_mult → the chop
            # veto simulated here is identical to the live execution gate.
            risk_profile=RiskProfile.for_asset_class(asset_config["asset_class"]),
            alpha_table=spread_alphas,
        )

        # ─── Phase 3a: Purge the unresolvable tail ─────────────────────────
        # The remainder's last max_hold bars per symbol can only resolve their
        # macro labels against bars that now live in the holdout: their walks
        # ran off the frame and every one of them reads "timeout -> loss".
        # Those are systematically wrong labels, not evidence (audit finding
        # 3, ~360 rows). Cutoffs come from the RAW remainder because the walk
        # runs over the pre-veto series; the drop happens after engineering.
        tail_cutoffs = _tail_cutoff_by_symbol(remainder_raw, asset_config["max_hold"])
        features_df, n_purged_tail = _purge_boundary_tail(features_df, tail_cutoffs)
        if n_purged_tail:
            logger.info(
                "BOUNDARY PURGE: dropped %d training rows within %d bars of the "
                "holdout boundary (unresolvable macro walk)",
                n_purged_tail, asset_config["max_hold"],
            )

        # ─── Phase 4: Walk-forward validation (3-fold expanding window) ────
        # TEMPORAL BOUNDARY: CV folds and the Profit Factor gate are evaluated
        # strictly OOS on the remainder. The final artifact is trained on the
        # remainder too, never on the holdout.
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
            chop_veto_rate=chop_veto_rate,
        )
        logger.info(f"Optimal Devil threshold (from Fold 3): {optimal_threshold:.4f}")

        # validate_candidate fits the production HMM only on a fold-gate pass.
        # The fold-fail diagnostic still needs the same feature space the Fold
        # 3 models were trained with, so fit it here on the remainder (never
        # the holdout) when the gate didn't. Diagnostic only: a remainder-fit
        # HMM behind a fold-failed candidate produces indicative numbers, not
        # a certifiable score -- the fold verdict stands regardless.
        if USE_HMM_FEATURES and final_hmm_models is None:
            logger.info(
                "Fitting production HMM on remainder for fold-fail holdout diagnostic..."
            )
            final_hmm_models = fit_regime_models(features_df)

        # ─── Phase 4.5: Artifact-level holdout gate ────────────────────────
        # Passing the fold gate is necessary but not sufficient. The served
        # model has never seen the holdout, so its score here is the honest
        # estimate of live performance. The PF bar gates on the exact
        # confidence lower bound (HOLDOUT_PF_CONFIDENCE), not the point
        # estimate -- see _holdout_pf_lower_bound. When the FOLD gate failed,
        # the holdout is still scored (Fold 3 models, diagnostic only): the
        # comparison answers whether the fold gate was too strict or the
        # model genuinely bad, and the fold verdict stands either way.
        holdout_metrics = HoldoutMetrics(
            used=False,
            fraction=HOLDOUT_FRAC,
            start_date=None,
            end_date=None,
            brier_score=float("nan"),
            expected_value=float("nan"),
            win_rate=0.0,
            profit_factor=0.0,
            trades=0,
            angel_proposed_trades=0,
            bypass_reason=None,
        )
        if HOLDOUT_FRAC > 0.0 and not holdout_raw.is_empty():
            logger.info("=" * 70)
            logger.info("ARTIFACT HOLDOUT EVALUATION")
            logger.info("=" * 70)

            if angel_model is None or devil_model is None:
                # Fold 3 never produced models (degenerate empty split) — there
                # is nothing to score on either side of the fold verdict.
                holdout_metrics.bypass_reason = "fold models unavailable"
                report.holdout = holdout_metrics
                logger.warning(
                    "⚠️  HOLDOUT NOT SCORED: no fold models exist to score."
                )
            else:
                holdout_scores, holdout_purged = _score_artifact_holdout(
                    holdout_raw,
                    angel_model,
                    devil_model,
                    angel_feats,
                    devil_feats,
                    optimal_threshold,
                    asset_config,
                    alpha_table=spread_alphas,
                    final_hmm_models=final_hmm_models,
                    angel_threshold=report.production_angel_threshold or None,
                )
                holdout_passed, holdout_reasons = _holdout_verdict(
                    holdout_scores,
                    sl_mult=asset_config["sl_mult"],
                    tp_mult=asset_config["tp_mult"],
                )
                pf_lb = _holdout_pf_lower_bound(
                    holdout_scores["wins"],
                    holdout_scores["trades"],
                    asset_config["sl_mult"],
                    asset_config["tp_mult"],
                )

                holdout_metrics = HoldoutMetrics(
                    used=True,
                    fraction=HOLDOUT_FRAC,
                    start_date=holdout_range[0].date().isoformat() if holdout_range else None,
                    end_date=holdout_range[1].date().isoformat() if holdout_range else None,
                    brier_score=holdout_scores["brier_score"],
                    expected_value=holdout_scores["expected_value"],
                    win_rate=holdout_scores["win_rate"],
                    profit_factor=holdout_scores["profit_factor"],
                    trades=holdout_scores["trades"],
                    angel_proposed_trades=holdout_scores["angel_proposed_trades"],
                    wins=holdout_scores["wins"],
                    pf_lower_bound=pf_lb,
                    pf_confidence=HOLDOUT_PF_CONFIDENCE,
                    purged_tail_rows=holdout_purged,
                    diagnostic_only=not report.gate_passed,
                )
                report.holdout = holdout_metrics

                logger.info(
                    "HOLDOUT METRICS: Brier=%.4f | EV=%.6f | WR=%.1f%% | "
                    "PF=%.4f [%.0f%% lower bound %.4f] | Trades=%d "
                    "(wins=%d, tail purged=%d)",
                    holdout_scores["brier_score"],
                    holdout_scores["expected_value"],
                    100.0 * holdout_scores["win_rate"],
                    holdout_scores["profit_factor"],
                    100.0 * HOLDOUT_PF_CONFIDENCE,
                    pf_lb,
                    holdout_scores["trades"],
                    holdout_scores["wins"],
                    holdout_purged,
                )

                if not report.gate_passed:
                    # The models scored are the Fold 3 placeholders, and the
                    # comparison is diagnostic only: is the fold gate too
                    # strict, or the model genuinely bad? The fold verdict
                    # stands and the holdout cannot rescue it.
                    logger.warning(
                        "⚠️  FOLD GATE FAILED — holdout scored for diagnostics "
                        "only (Fold 3 models); the fold verdict stands."
                    )
                elif not holdout_passed:
                    report.gate_passed = False
                    report.rejection_reasons.append(
                        "Holdout gate failed: " + "; ".join(holdout_reasons)
                    )
                    logger.warning(
                        "🚫 HOLDOUT GATE FAILED — artifact rejected by data it never saw"
                    )
                else:
                    logger.info("✅ HOLDOUT GATE PASSED")
        elif HOLDOUT_FRAC > 0.0 and holdout_raw.is_empty():
            holdout_metrics.bypass_reason = "empty holdout"
            report.holdout = holdout_metrics
            logger.warning(
                "⚠️  HOLDOUT BYPASSED: requested fraction %.2f produced an empty holdout. "
                "This run is gated by folds only.",
                HOLDOUT_FRAC,
            )
        elif HOLDOUT_FRAC <= 0.0:
            holdout_metrics.bypass_reason = "disabled"
            report.holdout = holdout_metrics
            logger.warning(
                "⚠️  HOLDOUT BYPASSED: RETRAIN_HOLDOUT_FRAC=0. "
                "Served model trains on the full window."
            )

        # ─── Phase 5: Gate decision ─────────────────────────────────────────
        # If gate passed: angel_model/devil_model are trained on the remainder.
        # If gate failed: they are Fold 3 models (will NOT be saved)
        promoted = promote_or_reject(
            report,
            angel_model,
            devil_model,
            optimal_threshold,
            asset_config,
            hmm_models=final_hmm_models,
            angel_threshold=report.production_angel_threshold or None,
        )

        if promoted:
            asset_class = asset_config.get("asset_class", "equities")
            saved_dir = asset_config.get("model_dir") or f"models/{asset_class}"
            # Feature-distribution sidecar for the drift probe
            # (scripts/probe_model.py). Computed from the exact post-veto,
            # post-clean population the promoted models trained on.
            stats = compute_feature_stats(features_df, feature_cols)
            save_feature_stats(stats, saved_dir)
            logger.info("=" * 70)
            logger.info(f"✅ MODELS PROMOTED ({asset_class}) — Ready for next market open")
            logger.info(f"  Models saved in: {saved_dir}/")
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
