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
    RETRAIN_LEARN_BARRIERS -- 1 (enabled by default). When set, fits and
        persists BarrierEstimator (CatBoost quantile regression for MAE tail
        and MFE median) into the model directory alongside Angel/Devil models.
        Writes barriers_mae.pkl, barriers_mfe.pkl, and barriers_meta.json.
        NOTE this hook fires off the ANGEL/DEVIL validation gate, not off the
        barrier promotion gate in scripts/evaluate_barriers.py, so artifacts
        can exist whose barrier gate failed — which is what the verdict below
        is for.
    RETRAIN_BARRIER_VERDICT -- path to a promotion verdict written by
        ``scripts/evaluate_barriers.py BARRIER_VERDICT_OUT=<path>``. When set
        and readable it is recorded in barriers_meta.json, and the live loader
        REFUSES an artifact that records FAIL. Unset/unreadable means the
        artifact carries no verdict: the loader then serves it with a warning,
        because absence is "unknown" rather than evidence.
    MAX_HOLD_BARS -- 45. A trade that reaches neither level within 45 bars is
        labelled a loss (timeout), because capital was tied up for nothing.
    SURVIVAL_BARS -- 5. The horizon for the Devil's survival label, below.
    RETRAIN_DEVIL_LABEL -- "survival" (default, what ships) or "macro": which
        label the Devil is trained and Brier-scored on. "macro" is the validated
        fix — the shipping survival-trained Devil is anti-informative about the
        served bracket (AUC 0.4722 vs 0.5839 measured 2026-09-14). See
        devil_label_col() for the numbers and the reasoning.
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
    _macro_base_rate -- what a RANDOM long entry would have won on the same
        bars under the same bracket: the macro outcome's mean over the
        population it is given, dropping non-finite rows and returning nan
        (never 0.0) for an unlabelled frame. The benchmark a fold's win rate
        has to beat before it means anything — a PF lower bound can be cleared
        by a zero-skill model when the base rate is high, and a skilled one
        rejected when it is low.
    FoldMetrics.base_rate / ValidationReport.pooled_base_rate -- that
        benchmark per fold (on the fold's own tradeable bars) and pooled,
        weighted by each fold's bar count. Reported telemetry; the verdict does
        not read it.
    ValidationReport.edge_over_random -- pooled fold win rate minus
        pooled_base_rate: the one number that separates skill from a
        favourable regime. Negative means the model did worse than taking
        every bar.
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
    apply_labels_and_veto -- (2026-09-21) the shared second half extracted from
        that function: builds both labels, the excursion targets, and both
        vetoes on an already-featured frame, then cleans. The feature lab
        (src/lab) reuses it with candidate generator lists so its labels and
        vetoes are byte-for-byte the production path.
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

Layout note (2026-09-16): this file is now a PACKAGE facade. Everything
re-exported below lives in the neighboring _common / _types / _data /
_labels / _features / _train / _thresholds / _gate / _persist / _pipeline
modules; the identifier meanings documented in this glossary still apply.
"""

from ._common import (  # noqa: F401
    ANGEL_PARAMS,  # tuple-target assign, not caught by auto-map
    DEVIL_PARAMS,  # same
    ANGEL_THRESHOLD,
    BASELINE_POOLED_OOS_TRADES,
    BASE_FEATURE_COLS,
    BEHAVIOR_VETO_LABELS,
    BRIER_THRESHOLD,
    BarrierEstimator,
    DAYS_BACK,
    DEFAULT_HORIZON,
    DEFAULT_RR_FLOOR,
    DEFAULT_TAU_MAE,
    DEFAULT_TAU_MFE,
    DEVIL_LABEL_DEFAULT,
    DEVIL_MIN_CHILD_FIXED,
    DEVIL_THRESHOLD,
    ENV_DEVIL_LABEL,
    EV_THRESHOLD,
    FEATURE_COLS,
    FeaturePipeline,
    HMM_OUTPUT_COLS,
    HOLDOUT_FRAC,
    HOLDOUT_PF_CONFIDENCE,
    MAX_HOLD_BARS,
    MIN_ANGEL_PROPOSALS,
    MIN_OOS_TRADES_FOR_PF,
    MODEL_FAMILY,
    MarketDataProvider,
    NotificationManager,
    PROFIT_FACTOR_THRESHOLD,
    Path,
    RETRAIN_LEARN_BARRIERS,
    RiskProfile,
    SL_ATR_MULTIPLIER,
    SPREAD_TABLE,
    SURVIVAL_BARS,
    TP_ATR_MULTIPLIER,
    UNTRADEABLE_SYMBOLS,
    USE_HMM_FEATURES,
    V3BaseFeatures,
    V3CostFeatures,
    V3HTFFeatures,
    V3SessionFeatures,
    _ANGEL_ATR_MULT_BY_CLASS,
    _CATBOOST_DROPPED_KEYS,
    _DEFAULT_TICKERS_BY_CLASS,
    _FIXED_ANGEL_THRESHOLD,
    _HTF_FOR_TIMEFRAME,
    _SPREAD_TABLE_PATH,
    _asset_class_for_source,
    _catboost_params,
    _core_dir,
    _load_spread_table,
    _project_root,
    _src_dir,
    compute_excursions,
    compute_feature_stats,
    datetime,
    devil_label_col,
    fit_regime_models,
    get_asset_config,
    get_hyperparameters,
    get_market_provider,
    joblib,
    lgb,
    logger,
    make_classifier,
    np,
    pl,
    predict_regime_probs,
    save_feature_stats,
    save_hmm_models,
    timedelta,
    timezone,
)

from ._types import (  # noqa: F401
    FoldMetrics,
    HoldoutMetrics,
    ValidationReport,
)

from ._data import (  # noqa: F401
    _split_holdout,
    fetch_training_data,
)

from ._labels import (  # noqa: F401
    _compute_chop_veto_mask,
    _compute_devil_survival_target,
    _compute_devil_targets_atr,
)

from ._features import (  # noqa: F401
    apply_labels_and_veto,
    engineer_features_and_labels,
    generate_time_decay_weights,
)

from ._train import (  # noqa: F401
    _devil_min_child,
    refit_models,
)

from ._thresholds import (  # noqa: F401
    _find_optimal_angel_threshold,
    _find_optimal_threshold,
)

from ._gate import (  # noqa: F401
    _capture_oos_ledger,
    _evaluate_holdout,
    _holdout_pf_lower_bound,
    _holdout_verdict,
    _macro_base_rate,
    _purge_boundary_tail,
    _score_artifact_holdout,
    _tail_cutoff_by_symbol,
    _tradeable_scoring_mask,
    validate_candidate,
)

from ._persist import (  # noqa: F401
    ENV_BARRIER_VERDICT,
    _GATE_THRESHOLDS,
    _is_production_model_dir,
    _load_barrier_verdict,
    _resolved_model_dir,
    fit_and_save_barriers,
    promote_or_reject,
    save_models,
    save_spread_table,
    save_threshold,
)

from ._pipeline import (  # noqa: F401
    main,
)

