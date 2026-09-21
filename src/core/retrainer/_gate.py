"""The walk-forward validation gate: per-fold scoring, edge-over-random telemetry,
tradeable-universe masking, holdout evaluation and verdicts, and validate_candidate.
Split out of core/retrainer.py on 2026-09-16.
"""
from __future__ import annotations

from . import _common as common
from . import _train as train
from ._common import (
    ANGEL_THRESHOLD,
    BASELINE_POOLED_OOS_TRADES,
    BRIER_THRESHOLD,
    DAYS_BACK,
    EV_THRESHOLD,
    HMM_OUTPUT_COLS,
    HOLDOUT_PF_CONFIDENCE,
    List,
    Optional,
    PROFIT_FACTOR_THRESHOLD,
    RiskProfile,
    SL_ATR_MULTIPLIER,
    Sequence,
    TP_ATR_MULTIPLIER,
    Tuple,
    USE_HMM_FEATURES,
    _scipy_beta,
    brier_score_loss,
    devil_label_col,
    fit_regime_models,
    logger,
    np,
    pl,
    predict_regime_probs,
    timedelta,
    warnings,
)
from ._types import (FoldMetrics, ValidationReport,)
from ._thresholds import (_find_optimal_threshold,)
from ._features import (engineer_features_and_labels,)


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION GATE
# ═══════════════════════════════════════════════════════════════════════════════


def _macro_base_rate(df: pl.DataFrame) -> float:
    """
    The macro bracket's BASE RATE on a population: what a random long entry at the
    same bars would have won, under the served bracket convention.

    This is the benchmark a fold's win rate has to beat before it means anything, and
    until 2026-09-14 the gate did not compute it. Measured on the H4 CatBoost
    candidate: its metals approvals won 57.1% against a period base rate of 56.6%
    (+0.005, p=0.50 — no selectivity at all) inside a regime where a 2:1 long bracket
    won 56.6% against a 33.3% break-even, i.e. the regime alone cleared a 1.2 PF lower
    bound. A gate scoring ABSOLUTE win rate and PF will therefore pass a zero-skill
    model in a high-base-rate period and reject a skilled one in a low-base-rate
    period. Reported, not yet gated: this is telemetry so the promotion decision can
    see the difference between skill and weather.

    Non-finite outcomes are dropped; an empty or label-less frame returns nan rather
    than a misleading 0.0.
    """
    if "devil_target_macro" not in df.columns or df.height == 0:
        return float("nan")
    y = df["devil_target_macro"].to_numpy().astype(float)
    y = y[np.isfinite(y)]
    return float(y.mean()) if y.size else float("nan")


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
    if not common.UNTRADEABLE_SYMBOLS or "symbol" not in val_df.columns:
        return approved_mask, 0

    proposed_symbols = (
        val_df.filter(pl.Series(signal_mask))["symbol"]
        .cast(pl.Utf8)
        .str.to_uppercase()
        .to_numpy()
    )
    tradeable = ~np.isin(proposed_symbols, list(common.UNTRADEABLE_SYMBOLS))
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
    y_devil = holdout_df[devil_label_col()].to_numpy()
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
    # (base rate, tradeable rows) per fold — pooled at the end into the edge-over-
    # random summary. See _macro_base_rate.
    base_rate_samples: List[tuple] = []

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
        ) = train.refit_models(
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
        # The Devil's own label — survival by default, macro under
        # RETRAIN_DEVIL_LABEL=macro (see devil_label_col).
        y_val_devil = val_df[devil_label_col()].to_numpy()
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
                fold_number, n_excluded, ",".join(sorted(common.UNTRADEABLE_SYMBOLS)),
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

        # ── the benchmark the win rate has to beat: random longs, same bars ──
        # Scored on the same tradeable population the fold's metrics use, so the
        # comparison is like-for-like. Reported, not gated (see _macro_base_rate for
        # why a PF lower bound alone cannot tell skill from a favourable regime).
        base_rows = (
            val_df.filter(~pl.col("symbol").is_in(list(common.UNTRADEABLE_SYMBOLS)))
            if common.UNTRADEABLE_SYMBOLS
            else val_df
        )
        fold_base_rate = _macro_base_rate(base_rows)
        fold_edge = (
            macro_win_rate - fold_base_rate
            if np.isfinite(fold_base_rate)
            else float("nan")
        )
        base_rate_samples.append((fold_base_rate, base_rows.height))

        logger.info(
            f"[Fold {fold_number}] "
            f"Brier={brier:.4f} | EV={ev:.6f} | WR={win_rate:.1%} | "
            f"Trades={n_devil_approved}"
        )
        logger.info(
            f"[Fold {fold_number}] EDGE OVER RANDOM: macro win {macro_win_rate:.4f} vs "
            f"base rate {fold_base_rate:.4f} -> {fold_edge:+.4f} "
            f"({base_rows.height:,} tradeable bars). A positive PF is worth nothing "
            f"unless this is positive."
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
            base_rate=fold_base_rate,
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

    # ── pooled edge over the bracket's own base rate ──
    # Weighted by each fold's tradeable bar count, because a fold with more bars
    # describes more of the market. The pooled win rate is the scored approvals'.
    _br = [(r, n) for r, n in base_rate_samples if np.isfinite(r)]
    if _br:
        _w = sum(n for _, n in _br)
        pooled_base_rate = (
            float(sum(r * n for r, n in _br) / _w) if _w > 0
            else float(np.mean([r for r, _ in _br]))
        )
    else:
        pooled_base_rate = float("nan")
    pooled_macro_win_rate = (
        pooled_oos_wins / pooled_oos_trades if pooled_oos_trades > 0 else float("nan")
    )
    edge_over_random = (
        pooled_macro_win_rate - pooled_base_rate
        if np.isfinite(pooled_base_rate) and np.isfinite(pooled_macro_win_rate)
        else float("nan")
    )
    logger.info(
        "EDGE OVER RANDOM (pooled): macro win %.4f vs base rate %.4f -> %+.4f "
        "on %d trades. This is the metric that separates skill from a favourable "
        "regime; a PF lower bound above %.2f can be cleared by a zero-skill model "
        "when the base rate is high.",
        pooled_macro_win_rate, pooled_base_rate, edge_over_random, pooled_oos_trades,
        PROFIT_FACTOR_THRESHOLD,
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
        ) = train.refit_models(
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
        pooled_base_rate=pooled_base_rate,
        edge_over_random=edge_over_random,
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
