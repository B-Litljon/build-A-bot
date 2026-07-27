"""
Meta-Labeling Machine Learning Trading Strategy (Angel & Devil Architecture).

Implements a two-stage inference system with hot-reloading:
1. The Angel (Primary Model): Learns Direction (high recall, threshold 0.40)
2. The Devil (Meta Model): Learns Conviction (high precision, threshold 0.50)

Usage:
    from strategies.concrete_strategies.ml_strategy import MLStrategy

    strategy = MLStrategy(
        angel_path="models/angel_latest.pkl",
        devil_path="models/devil_latest.pkl",
        angel_threshold=0.40,
        devil_threshold=0.50,
        warmup_period=60
    )

This is the decision-maker: bars in, Signal-or-None out. It owns no broker
connection and places no orders -- an orchestrator calls it and acts on the
result. Everything about it is arranged so that live inference matches training
exactly; the feature pipeline is IMPORTED from src/ml rather than reimplemented,
and its generator order is identical to the retrainer's.

Glossary:
    asset_class -- "equities" or "forex". Picks the default model directory
        (models/<asset_class>/) when explicit paths are not given.
    angel_path / devil_path -- the two model pickles. Resolved relative to the
        project root if not found as given.
    angel_threshold -- 0.40; how confident stage one must be to propose.
    devil_threshold -- the approval bar for stage two. Set from the
        constructor, then OVERWRITTEN by _load_threshold() from the model's
        threshold.json, because the retrainer tunes this value per model.
    warmup_period -- 260 bars minimum before trading. Sized for a 50-period
        average on 5-minute bars (50 x 5 = 250 plus headroom); acting sooner
        means acting on half-computed indicators.
    timeframe / htf_timeframe -- fast bar size in minutes, and the slower
        context timeframe ("5m").
    regime_window -- 260; must match RiskProfile.regime_window or the cost
        feature and the live cost gate would disagree about baseline volatility.

    self.pipeline -- the imported FeaturePipeline, built as
        [V3BaseFeatures, V3HTFFeatures, V3SessionFeatures, V3CostFeatures].
        This order is IDENTICAL to retrainer.py's; keeping them in lockstep is
        what prevents training/inference skew.
    _cost_gen -- the V3CostFeatures instance kept as a handle so hot-reload can
        swap its alpha table in place without rebuilding the pipeline.
    feature_names -- read from the trained model's own feature_names_in_, NOT
        hardcoded. That way a retrain which changes the feature set propagates
        without a code edit. Raises at construction if the model exposes no
        names, since there would be no safe way to order the columns.

    Fail-loud guards -- if the model was trained with cost_ratio but
        spread_alphas.json is missing, or with regime features but the HMM
        artifact is missing, the constructor raises. Deliberate: a boot-time
        error is far better than silently scoring against a wrong-width
        feature vector on the first live bar.
    _validate_metadata -- checks metadata.json next to the model to confirm the
        model's asset class matches. Warns (not raises) if absent.
    _load_threshold -- reads threshold.json from the model dir; falls back to
        the constructor value.
    _load_spread_table -- reads spread_alphas.json from the model dir, so a
        model always uses the cost assumptions it was trained against.

    Hot reload -- _check_model_updates() runs at the start of every bar and
        compares file modification times; when the retrainer atomically swaps in
        new pickles the strategy picks them up without a restart.
    _reload_lock -- guards that swap so a reload cannot interleave with
        inference.
    angel_mtime / devil_mtime / _spread_table_mtime -- the last-seen
        modification times driving that comparison.
    n_jobs = 1 -- forced on both models: single-row inference is so small that
        multi-process parallelism costs more in overhead than it saves.

    generate_signals -- the entry point. Returns a Signal only when BOTH stages
        approve, otherwise None.
    Stale-bar guard -- feature cleaning drops rows with missing values. If the
        NEWEST bar was the one dropped, the tail of the frame is an older bar,
        and scoring it against the current price would trade on the wrong bar.
        The strategy returns None instead.
    _heartbeat_window / _heartbeat_counter -- per-symbol ring buffer of recent
        Angel probabilities, summarised to the log every
        _heartbeat_every_n_bars (default 15, MLSTRATEGY_HEARTBEAT_EVERY_N).
        Exists so an operator can see the model is alive and evaluating during
        long stretches with no trades -- rejections themselves log at debug
        level and are normally invisible.
"""

import json
import logging
import os
import threading
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", message=".*join_asof.*")

from strategies.base import BaseStrategy, Signal
from core.notification_manager import NotificationManager

# CRITICAL: Import FeaturePipeline to prevent training/inference skew
from ml.feature_pipeline import FeaturePipeline
from ml.features.v3_features import (
    V3BaseFeatures,
    V3CostFeatures,
    V3HTFFeatures,
    V3SessionFeatures,
)
from ml.regimes.hmm_regime import (
    HMM_OUTPUT_COLS,
    load_hmm_models,
    predict_regime_probs,
)
from ml.trainers.v3_rf_trainer import V3RandomForestTrainer

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    Meta-Labeling ML strategy using two-stage Angel/Devil architecture.

    The Angel (primary model) proposes trades with high recall.
    The Devil (meta model) filters false positives with high precision.

    Parameters
    ----------
    angel_path : str | Path
        Path to the Angel (primary) model joblib file.
    devil_path : str | Path
        Path to the Devil (meta) model joblib file.
    angel_threshold : float
        Probability threshold for Angel to propose a trade (default: 0.40).
    devil_threshold : float
        Probability threshold for Devil to approve a trade (default: 0.50).
    warmup_period : int
        Minimum candles required before trading (default: 260).
    """

    def __init__(
        self,
        asset_class: str = "equities",
        angel_path: str | Path = None,
        devil_path: str | Path = None,
        angel_threshold: float = 0.40,
        devil_threshold: float = 0.50,
        warmup_period: int = 260,  # default for 1m base / 5m HTF
        timeframe: int = 1,
        htf_timeframe: str = "5m",
        angel_trainer=None,
        devil_trainer=None,
        regime_window: int = 260,  # must match RiskProfile.regime_window
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.asset_class = asset_class
        if angel_path is None:
            angel_path = f"models/{asset_class}/angel_latest.pkl"
        if devil_path is None:
            devil_path = f"models/{asset_class}/devil_latest.pkl"

        self._reload_lock = threading.Lock()
        self.timeframe = timeframe
        self.warmup = warmup_period
        self.angel_threshold = angel_threshold
        self.devil_threshold = devil_threshold

        # Load both models
        angel_file = Path(angel_path)
        devil_file = Path(devil_path)

        if not angel_file.exists():
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            angel_file = project_root / angel_path

        if not devil_file.exists():
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            devil_file = project_root / devil_path

        # Store model paths for hot-reloading
        self.angel_path = angel_file
        self.devil_path = devil_file

        # Load models and track modification times
        logger.info(f"Loading Angel model from {angel_file}")
        self.angel_trainer = (
            angel_trainer if angel_trainer is not None else V3RandomForestTrainer()
        )
        self.angel_trainer.load(str(angel_file))
        if hasattr(self.angel_trainer, "model") and hasattr(
            self.angel_trainer.model, "n_jobs"
        ):
            self.angel_trainer.model.n_jobs = (
                1  # Prevent joblib IPC overhead on single-row inference
            )

        self.angel_mtime = os.path.getmtime(angel_file)
        logger.info(f"Angel model loaded via trainer (mtime: {self.angel_mtime})")

        logger.info(f"Loading Devil model from {devil_file}")
        self.devil_trainer = (
            devil_trainer if devil_trainer is not None else V3RandomForestTrainer()
        )
        self.devil_trainer.load(str(devil_file))
        if hasattr(self.devil_trainer, "model") and hasattr(
            self.devil_trainer.model, "n_jobs"
        ):
            self.devil_trainer.model.n_jobs = (
                1  # Prevent joblib IPC overhead on single-row inference
            )

        self.devil_mtime = os.path.getmtime(devil_file)
        logger.info(f"Devil model loaded via trainer (mtime: {self.devil_mtime})")

        # Initialize notification manager for hot-reload alerts
        self.notification_manager = NotificationManager()

        # Per-instrument spread-cost table (spread_alphas.json in the model
        # dir, written by the retrainer on gate pass). None → V3CostFeatures
        # is a no-op and the pipeline is bit-identical to the pre-cost era.
        spread_table, self._spread_table_mtime = self._load_spread_table()

        # Initialize feature pipeline (imported, not duplicated!)
        # V3CostFeatures must follow V3BaseFeatures (needs natr_14). Keep a
        # handle so hot-reload can swap the alpha table in place.
        self._cost_gen = V3CostFeatures(
            alpha_table=spread_table["alphas"] if spread_table else None,
            default_alpha=(
                spread_table.get("default_alpha", 0.15) if spread_table else 0.15
            ),
            regime_window=regime_window,
        )
        self.pipeline = FeaturePipeline(
            feature_generators=[
                V3BaseFeatures(),
                V3HTFFeatures(timeframe=htf_timeframe),
                V3SessionFeatures(),
                self._cost_gen,
            ]
        )

        # Feature columns (excluding absolute price columns to prevent leakage).
        # Source the schema from the trained model itself so a retrain that
        # changes the feature space (e.g. enabling the HMM regime experiment
        # via RETRAIN_USE_HMM=1) propagates here without a code edit.
        model_features = self.angel_trainer.feature_names_in_
        if model_features is None:
            raise RuntimeError(
                "Loaded Angel model exposes no feature_names_in_ — cannot "
                "establish inference schema. Re-run retrainer with a model "
                "type that records feature names (sklearn / LightGBM)."
            )
        self.feature_names = list(model_features)
        logger.info(
            "MLStrategy feature schema sourced from model: %d features",
            len(self.feature_names),
        )

        # If the model trained on cost_ratio, the spread table is REQUIRED —
        # fail loudly at boot rather than obscurely at column selection on
        # the first bar. (Same philosophy as the HMM guard below; note we
        # load from angel_path.parent so side model dirs carry their own.)
        if "cost_ratio" in self.feature_names and spread_table is None:
            raise RuntimeError(
                f"Angel model expects the 'cost_ratio' feature but "
                f"{self.angel_path.parent / 'spread_alphas.json'} is missing "
                "or unreadable. Retrain with RETRAIN_SPREAD_TABLE so the "
                "table is persisted alongside the model, or restore the file."
            )

        # If the model trained on HMM regime probs, load the per-symbol HMM
        # artifact persisted alongside Angel/Devil and apply it at inference.
        self.hmm_models: Optional[dict] = None
        if any(c in self.feature_names for c in HMM_OUTPUT_COLS):
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            hmm_path = project_root / "models" / self.asset_class / "hmm_latest.pkl"
            try:
                self.hmm_models = load_hmm_models(hmm_path)
                fitted = sum(1 for m in self.hmm_models.values() if m is not None)
                logger.info(
                    "MLStrategy loaded HMM regime artifact: %s (%d/%d symbols fitted)",
                    hmm_path, fitted, len(self.hmm_models),
                )
            except FileNotFoundError:
                raise RuntimeError(
                    f"Angel model expects HMM features but {hmm_path} is missing. "
                    "Re-run the retrainer with RETRAIN_USE_HMM=1 so the HMM "
                    "artifact is persisted alongside the model."
                )

        # Validate metadata sidecar
        self._validate_metadata()

        # Override devil_threshold with the value persisted by the retrainer
        # (models/threshold.json).  Must be called AFTER self.devil_threshold is
        # set above so _load_threshold() can use it as a fallback.
        self.devil_threshold = self._load_threshold()

        # Heartbeat state: every N bars per symbol, log a summary of the
        # angel_prob distribution. Lets the operator see the model is
        # actively evaluating even when no signals fire (most rejections
        # happen at logger.debug, which the project's logging setup
        # silently suppresses at INFO root level).
        self._heartbeat_window: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=30)
        )
        self._heartbeat_counter: Dict[str, int] = defaultdict(int)
        self._heartbeat_every_n_bars = int(
            os.getenv("MLSTRATEGY_HEARTBEAT_EVERY_N", "15")
        )

    def _validate_metadata(self) -> None:
        """
        Validate that the loaded model matches the expected asset class using the metadata sidecar.

        The sidecar lives next to the model artifacts (angel_path's directory) so
        side models (e.g. models/forex_m15/) carry their own metadata.
        """
        metadata_path = self.angel_path.parent / "metadata.json"
        
        if not metadata_path.exists():
            logger.warning(
                "_validate_metadata: %s not found. Skipping distribution drift check.", 
                metadata_path
            )
            return
            
        try:
            with open(metadata_path, "r") as fh:
                data = json.load(fh)
                
            trained_class = data.get("asset_class")
            if trained_class != self.asset_class:
                raise RuntimeError(
                    f"Distribution drift detected: Strategy instantiated for asset class '{self.asset_class}', "
                    f"but model was trained on '{trained_class}'."
                )
            logger.info("_validate_metadata: passed (asset_class=%s)", trained_class)
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            logger.warning("_validate_metadata: failed to read %s (%s)", metadata_path, exc)

    def _load_threshold(self) -> float:
        """
        Load the Devil model's optimal threshold from the model directory's
        threshold.json (next to the pkl artifacts, so side models carry their own).

        Written by retrainer.save_threshold() after a successful validation gate.
        Falls back to self.devil_threshold (the value passed to __init__) if the
        file is absent or corrupt.

        Returns:
            float: The production threshold for Devil approval decisions.
        """
        threshold_path = self.angel_path.parent / "threshold.json"
        if not threshold_path.exists():
            logger.warning(
                "_load_threshold: %s not found — "
                "using constructor default devil_threshold=%.2f",
                threshold_path,
                self.devil_threshold,
            )
            return self.devil_threshold
        try:
            with open(threshold_path, "r") as fh:
                data = json.load(fh)
            threshold = float(data["devil_threshold"])
            logger.info(
                "_load_threshold: loaded production threshold=%.4f from %s",
                threshold,
                threshold_path,
            )
            return threshold
        except Exception as exc:
            logger.warning(
                "_load_threshold: failed to read %s (%s) — "
                "using constructor default devil_threshold=%.2f",
                threshold_path,
                exc,
                self.devil_threshold,
            )
            return self.devil_threshold

    def _load_spread_table(self) -> tuple:
        """
        Load the per-instrument spread-alpha table from the model directory's
        spread_alphas.json (next to the pkl artifacts, so side models carry
        their own — model and cost assumptions must travel together).

        Written by retrainer.save_spread_table() on gate pass; baked from live
        SPREAD_CALIB measurements by scripts/bake_spread_alphas.py.

        Returns:
            (table dict or None, file mtime or 0.0). None means no table —
            V3CostFeatures no-ops and the feature space has no cost_ratio.
        """
        table_path = self.angel_path.parent / "spread_alphas.json"
        if not table_path.exists():
            logger.info(
                "_load_spread_table: %s not found — cost_ratio feature "
                "disabled (pre-cost model dir)",
                table_path,
            )
            return None, 0.0
        try:
            with open(table_path, "r") as fh:
                table = json.load(fh)
            if not isinstance(table.get("alphas"), dict) or not table["alphas"]:
                raise ValueError("no 'alphas' mapping")
            logger.info(
                "_load_spread_table: loaded %d instrument alphas from %s "
                "(denomination=%sm): %s",
                len(table["alphas"]),
                table_path,
                table.get("denomination_minutes"),
                {k: round(v, 4) for k, v in sorted(table["alphas"].items())},
            )
            return table, os.path.getmtime(table_path)
        except Exception as exc:
            logger.warning(
                "_load_spread_table: failed to read %s (%s) — cost_ratio "
                "feature disabled",
                table_path,
                exc,
            )
            return None, 0.0

    @property
    def warmup_period(self) -> int:
        """Returns minimum candles required for indicators to warm up."""
        return self.warmup

    def _check_model_updates(self) -> bool:
        """
        Check for model file updates and hot-reload if necessary.

        Monitors the modification times of model files and reloads
        models in memory if they have been updated on disk.

        Returns:
            bool: True if any model was reloaded, False otherwise.
        """
        reloaded = False

        try:
            # Check Angel model
            if self.angel_path.exists():
                current_angel_mtime = os.path.getmtime(self.angel_path)
                if current_angel_mtime > self.angel_mtime:
                    logger.info(
                        f"[HOT-RELOAD] Detected new Angel model: {self.angel_path}"
                    )
                    try:
                        with self._reload_lock:
                            self.angel_trainer.load(str(self.angel_path))
                            if hasattr(self.angel_trainer, "model") and hasattr(
                                self.angel_trainer.model, "n_jobs"
                            ):
                                self.angel_trainer.model.n_jobs = 1
                            self.angel_mtime = current_angel_mtime
                        logger.info(f"[HOT-RELOAD] Angel model updated successfully")
                        reloaded = True
                    except Exception as e:
                        logger.error(f"[HOT-RELOAD] Failed to reload Angel model: {e}")

            # Check Devil model
            if self.devil_path.exists():
                current_devil_mtime = os.path.getmtime(self.devil_path)
                if current_devil_mtime > self.devil_mtime:
                    logger.info(
                        f"[HOT-RELOAD] Detected new Devil model: {self.devil_path}"
                    )
                    try:
                        with self._reload_lock:
                            self.devil_trainer.load(str(self.devil_path))
                            if hasattr(self.devil_trainer, "model") and hasattr(
                                self.devil_trainer.model, "n_jobs"
                            ):
                                self.devil_trainer.model.n_jobs = 1
                            self.devil_mtime = current_devil_mtime
                        logger.info(f"[HOT-RELOAD] Devil model updated successfully")
                        reloaded = True
                    except Exception as e:
                        logger.error(f"[HOT-RELOAD] Failed to reload Devil model: {e}")

            # Send notification if any model was reloaded
            if reloaded:
                # Refresh the inference schema from the reloaded Angel — a
                # retrain may change the feature space; predicting through a
                # stale column list would silently misalign features.
                new_features = self.angel_trainer.feature_names_in_
                if new_features is not None and list(new_features) != self.feature_names:
                    logger.warning(
                        "[HOT-RELOAD] Feature schema changed: %d -> %d features",
                        len(self.feature_names),
                        len(new_features),
                    )
                    self.feature_names = list(new_features)

                # Consistency check: Devil must be Angel's features + angel_prob.
                devil_features = self.devil_trainer.feature_names_in_
                expected = self.feature_names + ["angel_prob"]
                if devil_features is not None and list(devil_features) != expected:
                    msg = (
                        "[HOT-RELOAD] SCHEMA MISMATCH: Devil features do not "
                        "equal Angel features + angel_prob. Angel/Devil pair "
                        "on disk is inconsistent — predictions are suspect "
                        "until the next retrain completes."
                    )
                    logger.critical(msg)
                    self.notification_manager.send_system_message(msg)

                # Also reload the threshold — a retrain always produces a new
                # threshold.json alongside the new model weights.
                old_threshold = self.devil_threshold
                self.devil_threshold = self._load_threshold()
                if self.devil_threshold != old_threshold:
                    logger.info(
                        "[HOT-RELOAD] Devil threshold updated: %.4f -> %.4f",
                        old_threshold,
                        self.devil_threshold,
                    )

                # Reload the spread-alpha table when it changed on disk or the
                # refreshed schema newly requires cost_ratio (a cost-aware
                # retrain landed over a pre-cost model dir).
                table_path = self.angel_path.parent / "spread_alphas.json"
                table_mtime = (
                    os.path.getmtime(table_path) if table_path.exists() else 0.0
                )
                needs_cost = "cost_ratio" in self.feature_names
                if table_mtime != self._spread_table_mtime or (
                    needs_cost and self._cost_gen.alpha_table is None
                ):
                    new_table, self._spread_table_mtime = self._load_spread_table()
                    with self._reload_lock:
                        self._cost_gen.alpha_table = (
                            new_table["alphas"] if new_table else None
                        )
                        if new_table:
                            self._cost_gen.default_alpha = new_table.get(
                                "default_alpha", 0.15
                            )
                    logger.info(
                        "[HOT-RELOAD] Spread table %s",
                        "updated" if new_table else "removed/unreadable",
                    )
                if needs_cost and self._cost_gen.alpha_table is None:
                    msg = (
                        "[HOT-RELOAD] SCHEMA MISMATCH: model expects "
                        "cost_ratio but spread_alphas.json is missing from "
                        f"{self.angel_path.parent} — predictions will fail "
                        "until the table is restored."
                    )
                    logger.critical(msg)
                    self.notification_manager.send_system_message(msg)

                alert_message = (
                    "🔄 [HOT-RELOAD] New model weights ingested from disk. "
                    f"Angel: {self.angel_path.name}, Devil: {self.devil_path.name} "
                    f"| devil_threshold={self.devil_threshold:.4f}"
                )
                logger.critical(alert_message)
                self.notification_manager.send_system_message(alert_message)

        except Exception as e:
            logger.error(f"[HOT-RELOAD] Error checking for model updates: {e}")

        return reloaded

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        """
        Analyze single-symbol market data using two-stage Meta-Labeling.

        Stage 1: Angel proposes trades (high recall, low threshold).
        Stage 2: Devil filters false positives (high precision).

        Args:
            df: Polars DataFrame with OHLCV data for a single symbol.
                Must contain a 'symbol' column (added by callers) so the
                strategy can tag emitted signals with their instrument.

        Returns:
            base.Signal on joint Angel & Devil approval, or None.
        """
        # Check for model updates at the start of each bar processing cycle
        self._check_model_updates()

        self.validate_input(df)

        if len(df) < self.warmup_period:
            logger.debug(f"Insufficient data ({len(df)} < {self.warmup_period})")
            return None

        symbol: Optional[str] = None  # resolved inside try; used in except
        try:
            # Generate features using imported FeatureEngineer
            features_df = self._generate_features(df)

            if features_df is None or len(features_df) == 0:
                return None

            # Guard: clean_data drops rows with null/NaN/Inf features. If
            # the *newest* bar was dropped, the tail of features_df is a
            # stale bar — scoring it against the current price would trade
            # on the wrong bar. Skip the signal instead.
            if "timestamp" in features_df.columns and "timestamp" in df.columns:
                feat_ts = features_df["timestamp"].tail(1)[0]
                raw_ts = df["timestamp"].tail(1)[0]
                if feat_ts != raw_ts:
                    logger.warning(
                        "Latest bar (%s) dropped by feature cleaning — "
                        "newest valid features are from %s; skipping signal",
                        raw_ts,
                        feat_ts,
                    )
                    return None

            # Get latest bar's features for prediction. Pass a pandas
            # DataFrame (with column names) so LightGBM does not emit a
            # per-call UserWarning about missing feature names. Predictions
            # are identical either way (positional matching), but the live
            # log fills with warnings without this.
            latest_features_df = features_df[self.feature_names].tail(1)
            X_angel = latest_features_df.to_pandas()

            # Get current price for signal
            current_price = float(df["close"].tail(1)[0])

            # Resolve symbol — callers add this as a literal column before
            # invoking generate_signals (Option A design).
            symbol = str(df["symbol"].tail(1)[0]) if "symbol" in df.columns else None

            # ═══════════════════════════════════════════════════════════
            # STAGE 1: THE ANGEL (DIRECTION)
            # ═══════════════════════════════════════════════════════════
            angel_prob = self.angel_trainer.predict_proba(X_angel)[0, 1]

            # Heartbeat: track this bar's prob and periodically emit a
            # per-symbol distribution summary so silence in the logs is
            # distinguishable from a hung evaluation loop.
            heartbeat_key = symbol or "_anon"
            self._heartbeat_window[heartbeat_key].append(float(angel_prob))
            self._heartbeat_counter[heartbeat_key] += 1
            if self._heartbeat_counter[heartbeat_key] >= self._heartbeat_every_n_bars:
                probs = list(self._heartbeat_window[heartbeat_key])
                proposed = sum(1 for p in probs if p >= self.angel_threshold)
                # Surface the model's view of trading cost when the cost
                # feature is active — lets the operator read per-instrument
                # affordability straight off the heartbeat.
                cost_note = ""
                if "cost_ratio" in features_df.columns:
                    cost_note = " | cost_ratio=%.3f" % float(
                        features_df["cost_ratio"].tail(1)[0]
                    )
                logger.info(
                    "[%s] Heartbeat: last %d bars angel_prob "
                    "median=%.3f p75=%.3f max=%.3f | proposed=%d/%d (%.1f%%) "
                    "vs threshold=%.2f%s",
                    heartbeat_key,
                    len(probs),
                    float(np.median(probs)),
                    float(np.percentile(probs, 75)),
                    float(np.max(probs)),
                    proposed,
                    len(probs),
                    100.0 * proposed / len(probs),
                    self.angel_threshold,
                    cost_note,
                )
                self._heartbeat_counter[heartbeat_key] = 0

            if angel_prob < self.angel_threshold:
                logger.debug(
                    f"[{symbol}] Angel rejected | Prob: {angel_prob:.4f} < {self.angel_threshold}"
                )
                return None

            logger.debug(f"[{symbol}] Angel proposed trade | Prob: {angel_prob:.4f}")

            # ═══════════════════════════════════════════════════════════
            # STAGE 2: THE DEVIL (CONVICTION)
            # ═══════════════════════════════════════════════════════════
            # Build meta-feature frame: base features + Angel's probability,
            # preserving column names so LightGBM does not warn here either.
            X_devil = X_angel.copy()
            X_devil["angel_prob"] = angel_prob

            devil_prob = self.devil_trainer.predict_proba(X_devil)[0, 1]

            if devil_prob < self.devil_threshold:
                logger.debug(
                    f"[{symbol}] Devil veto | Angel: {angel_prob:.2f}, Devil: {devil_prob:.2f} < {self.devil_threshold}"
                )
                return None

            # Both Angel and Devil agree — emit raw ATR volatility.
            # RiskManager applies multipliers and floor checks (Path Alpha).
            natr_value = float(latest_features_df["natr_14"].to_numpy()[0])
            # TA-Lib NATR is a percentage; convert to absolute ATR
            atr_abs = (natr_value / 100.0) * current_price

            logger.info(
                f"[{symbol}] ANGEL & DEVIL AGREEMENT | "
                f"Price={current_price:.2f} | "
                f"Angel Prob: {angel_prob:.2f} | "
                f"Devil Prob: {devil_prob:.2f} | "
                f"raw_ATR={atr_abs:.4f}"
            )

            return Signal(
                direction="long",
                entry_price=current_price,
                raw_sl_distance=atr_abs,
                raw_tp_distance=atr_abs,
                metadata={
                    "symbol": symbol,
                    "angel_prob": float(angel_prob),
                    "devil_prob": float(devil_prob),
                    "atr_abs": atr_abs,
                    "timestamp": df["timestamp"].tail(1)[0],
                },
            )

        except Exception as e:
            logger.error(f"[{symbol}] Error in ML analysis: {e}", exc_info=True)
            return None

    def _generate_features(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Generate ML features using imported FeaturePipeline.

        This method ensures zero training/inference skew by using the exact
        same feature computation logic as the training pipeline.

        Args:
            df: Raw OHLCV DataFrame.

        Returns:
            DataFrame with computed features, or None if insufficient data.
        """
        try:
            # Use imported FeaturePipeline.run(). clean_data filters its
            # null-drop subset to columns that exist in the frame, so HMM
            # cols added after this call don't interfere with base cleaning.
            features_df = self.pipeline.run(df, feature_cols=self.feature_names)

            # Append HMM regime posteriors if the loaded model expects them.
            if self.hmm_models is not None:
                features_df = predict_regime_probs(features_df, self.hmm_models)

            return features_df

        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            return None
