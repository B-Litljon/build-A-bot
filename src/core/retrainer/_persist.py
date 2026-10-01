"""promote_or_reject decision, barrier sidecars, Discord verdict post, and the
atomic artifact writers (save_models / save_threshold / save_spread_table).
Split out of core/retrainer.py on 2026-09-16.
"""
from __future__ import annotations

from . import _common as common
from ._common import (
    ANGEL_THRESHOLD,
    BEHAVIOR_VETO_LABELS,
    BRIER_THRESHOLD,
    BarrierEstimator,
    DAYS_BACK,
    DEFAULT_HORIZON,
    DEFAULT_RR_FLOOR,
    DEFAULT_TAU_MAE,
    DEFAULT_TAU_MFE,
    EV_THRESHOLD,
    List,
    NotificationManager,
    Optional,
    PROFIT_FACTOR_THRESHOLD,
    Path,
    RETRAIN_EXCLUDE_ROLLOVER_BARS,
    SPREAD_TABLE,
    _SPREAD_TABLE_PATH,
    datetime,
    joblib,
    json,
    logger,
    np,
    os,
    pl,
    save_hmm_models,
    timezone,
)
from ._types import (ValidationReport,)
from ._features import (generate_time_decay_weights,)


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


ENV_BARRIER_VERDICT = "RETRAIN_BARRIER_VERDICT"


def _load_barrier_verdict() -> Optional[dict]:
    """
    Read the barrier PROMOTION VERDICT to record beside the weights.

    The barrier fit hook runs off the Angel/Devil validation gate, so nothing
    about the artifact's own fitness has been established by the time it is
    written — ``scripts/evaluate_barriers.py`` is the gate that decides whether
    learned geometry may replace the static bracket, and it is a separate run
    (it evaluates the M15 basket, not this model's frame). Its output is
    therefore an INPUT here, not something this function can re-derive.
    ``RETRAIN_BARRIER_VERDICT`` points at it.

    Failure policy, deliberately soft in two directions: an unset, unreadable
    or malformed verdict records NOTHING and warns, because killing a completed
    retrain over a missing sidecar is worse than a serve-time warning; but the
    artifact never CLAIMS evidence it does not have, and a verdict that parses
    and says ``passed: false`` is recorded as-is so the live loader refuses it.
    """
    raw = os.getenv(ENV_BARRIER_VERDICT, "").strip()
    if not raw:
        logger.warning(
            "[BARRIERS] %s is unset — the artifact will carry NO promotion "
            "verdict. That is servable (with a warning) but it is not "
            "evidence: run scripts/evaluate_barriers.py with "
            "BARRIER_VERDICT_OUT=<path> and point %s at the result.",
            ENV_BARRIER_VERDICT,
            ENV_BARRIER_VERDICT,
        )
        return None
    path = Path(raw)
    try:
        verdict = json.loads(path.read_text())
    except Exception as exc:
        logger.warning(
            "[BARRIERS] promotion verdict %s is unreadable (%s) — recording "
            "none rather than a claim",
            path,
            exc,
        )
        return None
    if not isinstance(verdict, dict) or not isinstance(verdict.get("passed"), bool):
        logger.warning(
            "[BARRIERS] promotion verdict %s has no boolean 'passed' — "
            "recording none; a verdict-shaped blob nothing can act on is "
            "worse than no verdict",
            path,
        )
        return None
    if verdict["passed"]:
        logger.info(
            "[BARRIERS] recording PASSED promotion verdict from %s "
            "(%d folds, coverage floor %s)",
            path,
            len(verdict.get("folds") or []),
            verdict.get("coverage_floor"),
        )
    else:
        logger.warning(
            "[BARRIERS] the recorded promotion gate FAILED (%s) — writing the "
            "artifact anyway for research, and recording the failure so the "
            "live loader REFUSES to serve it",
            path,
        )
    return verdict


def fit_and_save_barriers(
    df: pl.DataFrame,
    feature_cols: List[str],
    asset_config: dict,
) -> Optional[dict]:
    """
    Fit and atomically persist learned quantile barrier models.

    Uses BarrierEstimator (CatBoost backend with monotonic constraints on
    volatility) to predict instance-specific stop loss (Q_MAE) and take profit
    (Q_MFE) distances.

    Saves barriers_mae.pkl, barriers_mfe.pkl, and barriers_meta.json in the
    model directory. Write order ensures atomic hot-reload safety: weights are
    replaced before the meta JSON, which is replaced last. Updates metadata.json
    with barrier provenance.

    Args:
        df: Engineered DataFrame with features and excursion labels (mae_natr, mfe_natr).
        feature_cols: Feature columns for training.
        asset_config: Asset configuration dictionary.

    Returns:
        The metadata dict written to barriers_meta.json, or None if skipped/failed.
    """
    asset_class = asset_config.get("asset_class", "equities")
    model_dir = Path(asset_config.get("model_dir") or f"models/{asset_class}")
    horizon = int(asset_config.get("max_hold", DEFAULT_HORIZON))

    if "mae_natr" not in df.columns or "mfe_natr" not in df.columns:
        logger.warning(
            "[BARRIERS] Frame lacks mae_natr / mfe_natr columns — barrier fit skipped."
        )
        return None

    y_mae = df["mae_natr"].to_numpy().astype(float)
    y_mfe = df["mfe_natr"].to_numpy().astype(float)
    valid_mask = np.isfinite(y_mae) & np.isfinite(y_mfe)
    if "resolvable" in df.columns:
        valid_mask = valid_mask & df["resolvable"].to_numpy().astype(bool)

    n_valid = int(valid_mask.sum())
    if n_valid < 100:
        logger.warning(
            "[BARRIERS] Only %d valid labeled rows (< 100 floor) — barrier fit skipped.",
            n_valid,
        )
        return None

    logger.info("=" * 70)
    logger.info("FITTING LEARNED QUANTILE BARRIERS (MAE / MFE)")
    logger.info("=" * 70)

    family = os.getenv("BARRIER_FAMILY", "catboost").strip().lower()
    tau_mae = float(os.getenv("BARRIER_TAU_MAE", str(DEFAULT_TAU_MAE)))
    tau_mfe = float(os.getenv("BARRIER_TAU_MFE", str(DEFAULT_TAU_MFE)))
    rr_floor = float(os.getenv("BARRIER_RR_FLOOR", str(DEFAULT_RR_FLOOR)))

    estimator = BarrierEstimator(
        feature_cols=feature_cols,
        tau_mae=tau_mae,
        tau_mfe=tau_mfe,
        rr_floor=rr_floor,
        family=family,
    )

    weights = generate_time_decay_weights(
        len(df),
        timestamps=df["timestamp"] if "timestamp" in df.columns else None,
    )

    estimator.fit(df, df, sample_weight=weights)
    meta = estimator.save(model_dir, horizon=horizon, verdict=_load_barrier_verdict())

    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                cur_meta = json.load(f)
            cur_meta["learned_barriers"] = True
            cur_meta["barriers"] = meta
            metadata_temp = model_dir / "metadata_temp.json"
            with open(metadata_temp, "w") as f:
                json.dump(cur_meta, f, indent=2)
            os.replace(metadata_temp, metadata_path)
            logger.info(f"[ATOMIC] Updated {metadata_path} with learned_barriers=True")
        except Exception as exc:
            logger.warning(f"[BARRIERS] Failed to update {metadata_path}: {exc}")

    return meta


def save_models(
    angel_model: "lgb.LGBMClassifier",
    devil_model: "lgb.LGBMClassifier",
    asset_config: dict,
    report: Optional[ValidationReport] = None,
    barrier_model: Optional["BarrierEstimator"] = None,
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

    barrier_meta = None
    if barrier_model is not None:
        try:
            horizon = int(asset_config.get("max_hold", DEFAULT_HORIZON))
            barrier_meta = barrier_model.save(model_dir, horizon=horizon)
        except Exception as e:
            logger.error(f"[ATOMIC] Failed to serialize barrier model: {e}")
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
        # (2026-10-01) Whether the training window excluded NY-rollover bars
        # BEFORE feature generation. Served models that trained with this must
        # eventually be fed matching (rounded) live prices — recorded here so
        # the serving-side OANDA_ROUND_MID_TO_PIPETTE rollout can be checked
        # against the artifact, not guessed.
        "rollover_bar_exclusion": RETRAIN_EXCLUDE_ROLLOVER_BARS,
        # Holdout record: what the served artifact earned on data it never saw.
        # If the holdout was disabled or empty, "used" is false and the reason
        # is recorded so the artifact cannot be mistaken for one that passed a
        # real holdout gate.
        "holdout": {
            "used": False,
            "fraction": common.HOLDOUT_FRAC,
            "bypass_reason": "disabled" if common.HOLDOUT_FRAC <= 0.0 else None,
        },
        "learned_barriers": barrier_model is not None,
    }
    if barrier_meta is not None:
        metadata["barriers"] = barrier_meta
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

    Written atomically alongside the model .pkl files.  The live strategy
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
