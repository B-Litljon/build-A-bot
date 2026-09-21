"""refit_models: trains Angel and Devil on the approved bars, including the
conformal/calibration pre-split, and _devil_min_child sample-floor logic.
Split out of core/retrainer.py on 2026-09-16.
"""
from __future__ import annotations

from ._common import (
    ANGEL_PARAMS,
    DEVIL_PARAMS,
    ANGEL_THRESHOLD,
    DEVIL_MIN_CHILD_FIXED,
    List,
    MODEL_FAMILY,
    Optional,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
    TimeSeriesSplit,
    Tuple,
    _FIXED_ANGEL_THRESHOLD,
    devil_label_col,
    logger,
    make_classifier,
    np,
    pl,
    warnings,
)
from ._thresholds import (_find_optimal_angel_threshold,)
from ._features import (generate_time_decay_weights,)


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
    devil_col = devil_label_col()
    y_devil = df[devil_col].to_numpy()

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
