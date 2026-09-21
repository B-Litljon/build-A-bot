"""engineer_features_and_labels (bars -> full feature/label frame) and
generate_time_decay_weights. Split out of core/retrainer.py on 2026-09-16.
"""
from __future__ import annotations

from ._common import (
    BASE_FEATURE_COLS,
    BEHAVIOR_VETO_LABELS,
    FeaturePipeline,
    HMM_OUTPUT_COLS,
    List,
    MAX_HOLD_BARS,
    Optional,
    RiskProfile,
    SL_ATR_MULTIPLIER,
    SURVIVAL_BARS,
    TP_ATR_MULTIPLIER,
    Tuple,
    USE_HMM_FEATURES,
    V3BaseFeatures,
    V3CostFeatures,
    V3HTFFeatures,
    V3SessionFeatures,
    compute_excursions,
    logger,
    np,
    pl,
)
from ._labels import (_compute_chop_veto_mask, _compute_devil_survival_target, _compute_devil_targets_atr,)


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
    # EXCURSION TARGETS: Realized forward MAE / MFE (NATR-normalized)
    # Computed per symbol before chop veto to maintain continuous price
    # series for the forward sliding window.
    # ═══════════════════════════════════════════════════════════════════
    logger.info(
        f"Computing forward MAE/MFE excursion targets (horizon={max_hold} bars)..."
    )
    if "symbol" in df.columns:
        excursion_parts = []
        for sym in df["symbol"].unique(maintain_order=True).to_list():
            sym_df = df.filter(pl.col("symbol") == sym)
            excursion_parts.append(compute_excursions(sym_df, horizon=max_hold))
        df = pl.concat(excursion_parts)
    else:
        df = compute_excursions(df, horizon=max_hold)
    n_resolvable = int(df["resolvable"].sum()) if "resolvable" in df.columns else 0
    logger.info(
        f"Generated excursion labels (mae_natr, mfe_natr): "
        f"{n_resolvable:,} resolvable / {len(df):,} total"
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
