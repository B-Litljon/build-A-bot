"""main() — end-to-end wiring: config -> fetch -> holdout carve -> features ->
validate -> promote. Split out of core/retrainer.py on 2026-09-16.

Glossary:
    _rollover_exclusion_mask -- boolean mask (True = drop) of the bars whose
        America/New_York local time falls inside the Gate C rollover window
        (execution.risk_manager.get_blackout_window_et: RISK_BLACKOUT_ET, then
        _DEFAULT_BLACKOUT_ET = 16:55-17:30 ET). Runs on the pre-featurization
        RAW frame. RETRAIN_EXCLUDE_ROLLOVER_BARS=1 (default) enables it.
    _exclude_rollover_bars -- applies that mask to the raw remainder/holdout
        and reports the dropped count. Complements, does not replace, the
        post-labeling chop veto: the veto decides ENTRIES after targets are
        built (so the bracket walk keeps the contiguous path), this exclusion
        keeps the transition bars OUT of the indicators entirely so they cannot
        poison neighboring rows' rolling features. Both default on; toggles are
        independent (RETRAIN_EXCLUDE_ROLLOVER_BARS vs RISK_TIME_GATE_ENABLED).
    ROLLOVER_WINDOW_DESC -- "16:55-17:30" (America/New_York), shown in the log.
"""
from __future__ import annotations

import polars as pl

from . import _common as common
from . import _data as data
from . import _persist as persist
from ._common import (
    FEATURE_COLS,
    HOLDOUT_PF_CONFIDENCE,
    Optional,
    RETRAIN_EXCLUDE_ROLLOVER_BARS,
    RETRAIN_LEARN_BARRIERS,
    RiskProfile,
    SPREAD_TABLE,
    Tuple,
    _SPREAD_TABLE_PATH,
    compute_feature_stats,
    fit_regime_models,
    get_asset_config,
    get_hyperparameters,
    logger,
    os,
    save_feature_stats,
)
from ._types import (HoldoutMetrics,)
from ._gate import (_holdout_pf_lower_bound, _holdout_verdict, _purge_boundary_tail, _score_artifact_holdout, _tail_cutoff_by_symbol, validate_candidate,)
from ._data import (_split_holdout, fetch_training_data,)
from ._features import (apply_labels_and_veto, engineer_features_and_labels,)
from ._persist import (fit_and_save_barriers,)

ROLLOVER_WINDOW_DESC = "16:55-17:30"


# ═══════════════════════════════════════════════════════════════════════════════
# NY-ROLLOVER BAR EXCLUSION (training-side, pre-featurization)
# ═══════════════════════════════════════════════════════════════════════════════


def _rollover_exclusion_mask(raw_df: pl.DataFrame) -> pl.Series:
    """
    Boolean mask (True = drop) for the NY rollover window, pre-featurization.

    WHY a second exclusion exists alongside Gate C's entry veto: the chop veto
    (_compute_chop_veto_mask, post-labeling) drops rollover rows only as trade
    ENTRIES, deliberately after target generation so the bracket walk keeps the
    contiguous price path. Their bars still feed the rolling features of
    NEIGHBORING rows (group_by_dynamic / rolling windows do not know the veto
    exists), and the 16:55–17:30 ET transition bars' extreme hour_of_day /
    bb_pct_b / vol_rel excursions are exactly what the trees keyed on — every
    Angel proposal the live soak has ever seen was on an illiquid rollover bar
    (soak evidence, 2026-09). Dropping the bars BEFORE feature generation keeps
    that poison out of the indicators; the post-labeling veto keeps the entry
    discipline. Two mechanisms, two layers, both needed.

    The window is THE Gate C definition, not a copy: times come from
    ``execution.risk_manager.get_blackout_window_et`` (RISK_BLACKOUT_ET /
    _DEFAULT_BLACKOUT_ET), imported lazily and both-spelled (``src.execution`` /
    ``execution``) exactly as _common.py imports the retrainer's execution-side
    dependencies — no copy of the window lives here (the 2026-09-15 lesson).
    Converted per-row to America/New_York so it tracks DST like the live gate.
    Naive timestamps are assumed UTC (the same assumption everywhere in the
    retrainer).

    Gated by RETRAIN_EXCLUDE_ROLLOVER_BARS (default on); returns an all-False
    mask when disabled.
    """
    n_total = raw_df.height
    if not RETRAIN_EXCLUDE_ROLLOVER_BARS or n_total == 0:
        return pl.Series(values=[False] * n_total, dtype=pl.Boolean)

    # Execution-side import, lazily and both-spelled like _common.py does:
    # this pulls the Gate C window from risk_manager WITHOUT re-deriving it
    # here (the 2026-09-15 lesson: one DST-anchored definition, not three).
    try:
        from src.execution.risk_manager import (
            ENV_BLACKOUT_ET,
            get_blackout_window_et,
        )
    except ImportError:
        from execution.risk_manager import (
            ENV_BLACKOUT_ET,
            get_blackout_window_et,
        )

    window = get_blackout_window_et()
    if window is None:
        logger.warning(
            "Rollover exclusion ON but RISK_BLACKOUT_ET=%r does not parse — "
            "no bars excluded",
            os.getenv(ENV_BLACKOUT_ET, ""),
        )
        return pl.Series(values=[False] * n_total, dtype=pl.Boolean)
    start, end = window

    if "timestamp" not in raw_df.columns:
        logger.warning(
            "Rollover exclusion ON but frame lacks a 'timestamp' column — "
            "no bars excluded"
        )
        return pl.Series(values=[False] * n_total, dtype=pl.Boolean)

    ts = raw_df["timestamp"]
    if getattr(ts.dtype, "time_zone", None) is None:
        ts = ts.dt.replace_time_zone("UTC")
    ny = ts.dt.convert_time_zone("America/New_York")
    # .dt.hour() is Int8 — cast before multiplying or 22*3600 overflows.
    secs = (
        ny.dt.hour().cast(pl.Int64) * 3600
        + ny.dt.minute().cast(pl.Int64) * 60
        + ny.dt.second().cast(pl.Int64)
    )
    start_s = start.hour * 3600 + start.minute * 60
    end_s = end.hour * 3600 + end.minute * 60
    if start_s <= end_s:
        inside = (secs >= start_s) & (secs < end_s)
    else:
        inside = (secs >= start_s) | (secs < end_s)  # window wraps midnight
    return inside


def _exclude_rollover_bars(raw_df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
    """
    Drop pre-featurization rows inside the Gate C NY-rollover window.

    Applies the mask from _rollover_exclusion_mask and logs how many bars went
    (0 with an explanation when the exclusion is disabled / unparseable / the
    frame carries no timestamp column).
    """
    mask = _rollover_exclusion_mask(raw_df)
    n_drop = int(mask.sum()) if raw_df.height else 0
    if n_drop == 0:
        return raw_df, 0
    filtered = raw_df.filter(~mask)
    logger.info(
        "ROLLOVER BAR EXCLUSION: dropped %d of %d pre-featurization bars inside "
        "the %s ET Gate C window (poisons rolling features of neighboring rows; "
        "the post-labeling chop veto still applies separately)",
        n_drop,
        raw_df.height,
        ROLLOVER_WINDOW_DESC,
    )
    return filtered, n_drop


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
        provider = common.get_market_provider()
        
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
        raw_data = data.fetch_training_data(
            provider=provider,
            symbols=asset_config["tickers"],
            days_back=common.DAYS_BACK,
            timeframe_minutes=asset_config["timeframe_minutes"]
        )

        # ─── Phase 2a: Carve holdout FIRST, before any feature engineering ───
        # The holdout is the chronologically last slice. No indicator, label,
        # veto, threshold, or model parameter may see across this boundary.
        # RETRAIN_HOLDOUT_FRAC=0 disables the holdout and preserves legacy
        # full-data behaviour.
        remainder_raw, holdout_raw, holdout_range = _split_holdout(
            raw_data, common.HOLDOUT_FRAC
        )
        if common.HOLDOUT_FRAC <= 0.0:
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
                100.0 * common.HOLDOUT_FRAC,
                holdout_raw.height,
                remainder_raw.height,
            )
            if holdout_range:
                logger.info(
                    "HOLDOUT RANGE: %s → %s",
                    holdout_range[0].isoformat(),
                    holdout_range[1].isoformat(),
                )

        # ─── Phase 2b: Rollover bar exclusion (pre-featurization) ───────────
        # Applied to each slice AFTER the carve so the boundary timestamp still
        # splits the same bars. Feature/label engineering never looks forward
        # across rows by calendar, so an identical fraction of removed bars in
        # both slices preserves every property the carve guarantees — and each
        # slice's indicators must not see the rollover bars, whichever slice
        # they landed in.
        remainder_raw, n_rollover_remainder = _exclude_rollover_bars(remainder_raw)
        holdout_raw, n_rollover_holdout = _exclude_rollover_bars(holdout_raw)
        if RETRAIN_EXCLUDE_ROLLOVER_BARS and remainder_raw.is_empty():
            raise ValueError(
                "Rollover bar exclusion emptied the training remainder — "
                "check RISK_BLACKOUT_ET and the data window"
            )
        if RETRAIN_EXCLUDE_ROLLOVER_BARS and holdout_raw.is_empty() and n_rollover_holdout:
            raise ValueError(
                "Rollover bar exclusion emptied the holdout slice — "
                "check RISK_BLACKOUT_ET and the data window"
            )
        if n_rollover_remainder or n_rollover_holdout:
            logger.info(
                "Rollover exclusion totals: remainder %d bars, holdout %d bars",
                n_rollover_remainder,
                n_rollover_holdout,
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
        if common.USE_HMM_FEATURES and final_hmm_models is None:
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
            fraction=common.HOLDOUT_FRAC,
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
        if common.HOLDOUT_FRAC > 0.0 and not holdout_raw.is_empty():
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
                    fraction=common.HOLDOUT_FRAC,
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
        elif common.HOLDOUT_FRAC > 0.0 and holdout_raw.is_empty():
            holdout_metrics.bypass_reason = "empty holdout"
            report.holdout = holdout_metrics
            logger.warning(
                "⚠️  HOLDOUT BYPASSED: requested fraction %.2f produced an empty holdout. "
                "This run is gated by folds only.",
                common.HOLDOUT_FRAC,
            )
        elif common.HOLDOUT_FRAC <= 0.0:
            holdout_metrics.bypass_reason = "disabled"
            report.holdout = holdout_metrics
            logger.warning(
                "⚠️  HOLDOUT BYPASSED: RETRAIN_HOLDOUT_FRAC=0. "
                "Served model trains on the full window."
            )

        # ─── Phase 5: Gate decision ─────────────────────────────────────────
        # If gate passed: angel_model/devil_model are trained on the remainder.
        # If gate failed: they are Fold 3 models (will NOT be saved)
        promoted = persist.promote_or_reject(
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

            if RETRAIN_LEARN_BARRIERS:
                try:
                    fit_and_save_barriers(features_df, feature_cols, asset_config)
                except Exception as e:
                    logger.error(
                        "[BARRIERS] Failed to fit and save learned barriers: %s",
                        e,
                        exc_info=True,
                    )

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
