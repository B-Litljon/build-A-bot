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
    angel_threshold -- how confident stage one must be to propose. Default
        from core.thresholds (0.40, env-overridable), then OVERRIDDEN by the
        angel_threshold pinned in the model's threshold.json when present —
        a pair must run at the bar its Devil and brackets were fitted for.
    devil_threshold -- the approval bar for stage two. Set from the
        constructor, then OVERWRITTEN by _load_thresholds() from the model's
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
    _close_enough -- float-tolerant equality used by _validate_metadata's
        bracket check (added 2026-09-09): metadata sl/tp multipliers must
        match RiskProfile.for_asset_class(asset_class) or boot refuses —
        train/serve skew on the Devil's labels is not survivable live.
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
    _load_thresholds -- reads threshold.json from the model dir (the tuned
        Devil bar and, since 2026-07, the pinned Angel bar); falls back to
        the constructor values for missing keys.
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

    ── Learned barrier geometry (optional sidecar) ──
    use_barriers -- whether this strategy serves LEARNED stop/target geometry.
        Constructor argument, defaulting from BARRIER_GEOMETRY_ENABLED; OFF
        unless asked for, so a soak keeps running the static profile
        multipliers until someone deliberately enables the learned ones on
        evidence.
    _barrier_estimator -- the loaded BarrierEstimator (ml.barriers), or None
        when the sidecar is off. Two quantile models: SL <- Q_MAE(tau_mae),
        TP <- Q_MFE(tau_mfe).
    _barrier_mtime -- the mtime of barriers_meta.json, the reload trigger. The
        META alone is watched because save() writes both pickles before it, so
        seeing a new meta means a complete matching pair (see estimator.save).
    _load_barriers -- reads the sidecar and REFUSES to boot on a missing,
        unparsable or horizon-mismatched set. Fail-loud is deliberate: an
        operator who enabled learned stops and silently got static ones could
        not tell, which is the same trap the cost_ratio/HMM guards close.
    _barrier_geometry -- one bar's learned geometry as the payload the
        orchestrator hands to RiskManager, in NATR MULTIPLES (the same units as
        RiskProfile.sl_atr_multiplier) so the learned quantiles substitute for
        the static constants and every downstream step -- rounding, the three
        gates, sizing -- stays on one code path.
    BARRIER_GEOMETRY_KEY -- the Signal.metadata key that payload travels
        under (defined in strategies.base, so execution and the offline
        backtester can read it without importing this module).
    BARRIER_GEOMETRY_ENABLED -- "1" serves learned geometry; unset/0 does not.
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

from strategies.base import BARRIER_GEOMETRY_KEY, BaseStrategy, Signal
from core import events
from core.notification_manager import NotificationManager
from core.thresholds import ANGEL_THRESHOLD as DEFAULT_ANGEL_THRESHOLD
# NOTE: execution.risk_manager is imported lazily inside _validate_metadata —
# a module-level import here cycles through execution/__init__.py.

# CRITICAL: Import FeaturePipeline to prevent training/inference skew
from ml.barriers.estimator import (
    BARRIER_META_FILENAME,
    BarrierEstimator,
)
from ml.barriers.labels import DEFAULT_HORIZON
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

ENV_BARRIER_GEOMETRY_ENABLED = "BARRIER_GEOMETRY_ENABLED"

# Features a barrier model may read from the served frame beyond the Angel's own
# schema: `close` is needed for the NATR%->price conversion and is present in
# the pipeline output, but is deliberately not a model feature (absolute prices
# are leakage), so it is not in feature_names_in_.
_BARRIER_FRAME_ONLY_COLS = frozenset({"close"})


def _verdict_summary(verdict: dict) -> str:
    """
    One line of a barrier promotion verdict, for a log or an exception.

    Reads only the fields worth naming and tolerates the rest, because the
    verdict is written by the producer (``scripts/evaluate_barriers.py
    --verdict-out``) and its schema is not this module's to enforce beyond the
    boolean ``passed`` that the refusal turns on.
    """
    parts = []
    folds = verdict.get("folds")
    if isinstance(folds, list) and folds:
        coverages = [
            f.get("coverage") for f in folds if isinstance(f, dict)
        ]
        coverages = [c for c in coverages if isinstance(c, (int, float))]
        if coverages:
            parts.append(
                "coverage " + "/".join(f"{c:.3f}" for c in coverages)
            )
    for key in ("eval_date", "granularity", "rows", "coverage_floor"):
        if verdict.get(key) is not None:
            parts.append(f"{key}={verdict[key]}")
    return ", ".join(parts) if parts else "no detail recorded"


def _barriers_requested(explicit: Optional[bool] = None) -> bool:
    """
    Resolve the barrier sidecar switch: explicit argument first, then
    BARRIER_GEOMETRY_ENABLED. Anything unset/0/false/no/off means OFF.

    Default OFF is the point, not an accident: the learned geometry is only
    promotable once it has beaten the static bracket on every evaluation fold,
    and the served Devil's labels still encode the static multiples. An off
    sidecar makes this class behave exactly as it did before the barriers
    existed, which is what keeps a live soak safe across a restart onto this
    branch.
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.getenv(ENV_BARRIER_GEOMETRY_ENABLED, "0").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def _close_enough(a: float, b: float, rel_tol: float = 1e-6) -> bool:
    """Float-tolerant equality for metadata bracket comparison."""
    return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1.0)


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
    use_barriers : bool | None
        Serve learned quantile barrier geometry instead of the profile's static
        stop/target multipliers. None (the default) defers to
        BARRIER_GEOMETRY_ENABLED, which is OFF unless set — see the module
        glossary.
    """

    def __init__(
        self,
        asset_class: str = "equities",
        angel_path: str | Path = None,
        devil_path: str | Path = None,
        angel_threshold: float = DEFAULT_ANGEL_THRESHOLD,
        devil_threshold: float = 0.50,
        warmup_period: int = 260,  # default for 1m base / 5m HTF
        timeframe: int = 1,
        htf_timeframe: str = "5m",
        angel_trainer=None,
        devil_trainer=None,
        regime_window: int = 260,  # must match RiskProfile.regime_window
        use_barriers: Optional[bool] = None,
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

        # Barrier geometry request flag (must precede metadata validation)
        self.use_barriers = _barriers_requested(use_barriers)

        # Validate metadata sidecar
        self._validate_metadata()

        # Override both thresholds with the values persisted by the retrainer
        # (threshold.json in the model dir). Must be called AFTER the
        # instance values are set above so _load_thresholds() can use them as
        # fallbacks. The Angel value pins the population the pair was
        # trained/bracketed at; artifacts predating 2026-07 pin only the
        # Devil and fall back to the constructor default for the Angel.
        self.angel_threshold, self.devil_threshold = self._load_thresholds()

        # Sidecar mtimes for the independent reload path (2026-09-09): the
        # threshold/spread sidecars reload on their OWN mtimes now, not only
        # when a pickle changed — a retrain writes pkls first, and a bar
        # landing in between used to pin the new pair to the OLD bars forever.
        thr_path = self.angel_path.parent / "threshold.json"
        self._threshold_mtime = (
            os.path.getmtime(thr_path) if thr_path.exists() else 0.0
        )
        # _pair_mixed: set when one half of the Angel/Devil pair advances
        # without the other (the retrainer replaces them separately); cleared
        # once the other half has also been reloaded. While set,
        # generate_signals stands down rather than score a mixed generation.
        # _pair_pending: which side we are waiting for ("angel"/"devil"/None).
        self._pair_mixed = False
        self._pair_pending = None

        # ── learned barrier geometry (optional sidecar) ───────────────────
        # Loaded AFTER the sidecar mtimes above so a promotion that swapped
        # the whole directory is read once, at boot, from the settled files.
        self._barrier_estimator: Optional[BarrierEstimator] = None
        self._barrier_meta_path = self.angel_path.parent / BARRIER_META_FILENAME
        self._barrier_mtime = 0.0
        if self.use_barriers:
            self._barrier_estimator = self._load_barriers()
            self._barrier_mtime = self._barrier_meta_mtime()
        else:
            logger.info(
                "Barrier geometry OFF (%s unset) — stop/target distances come "
                "from the static profile multipliers",
                ENV_BARRIER_GEOMETRY_ENABLED,
            )

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
        Validate that the loaded model matches the expected asset class and
        bracket multiples using the metadata sidecar.

        The sidecar lives next to the model artifacts (angel_path's directory) so
        side models (e.g. models/forex_m15/) carry their own metadata.

        Bracket check (added 2026-09-09): the retrainer records
        sl_atr_multiplier/tp_atr_multiplier into metadata; the Devil's labels
        encode those multiples, so serving a model trained on one bracket set
        under another is silent train/serve skew (the exact failure
        soak.service warns about). Raise when present-and-mismatched.
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

            # Bracket enforcement — skip silently when the keys are absent
            # (pre-2026-09 artifacts predate them); raise on a real mismatch.
            # When learned barrier geometry is active (use_barriers=True),
            # instance-specific MAE/MFE barriers govern live execution brackets
            # rather than the static profile multipliers, so a profile-vs-
            # metadata mismatch is EXPECTED and the check stands down.
            #
            # Standing down is not the same as "nothing to see": the Devil is
            # trained on a label built from the STATIC multiples
            # (retrainer.py:1469, `_compute_devil_survival_target(sl_mult=2.0)`),
            # so serving learned brackets runs the selection model against a
            # geometry its conviction was never fitted on — the exact class of
            # train/serve skew the branch below exists to catch. Report the pair
            # rather than skipping in silence; the direction of that error is
            # NOT known (a wider stop survives more often, a closer target hits
            # more often), which is why it must be visible in the log.
            # See llm_reports/m2m-prompts/2026-09-14_barrier-live-seam.md.
            if getattr(self, "use_barriers", False):
                from execution.risk_manager import RiskProfile  # lazy: avoids
                # an import cycle through execution/__init__.py at module load.

                profile = RiskProfile.for_asset_class(self.asset_class)
                trained_sl = data.get("sl_atr_multiplier")
                trained_tp = data.get("tp_atr_multiplier")
                logger.warning(
                    "_validate_metadata: learned barrier geometry active — "
                    "bracket check STOOD DOWN. Profile runs %sx/%sx; this "
                    "artifact's Devil was labelled at %sx/%sx. A learned "
                    "bracket is a different walk than the one the Devil's "
                    "conviction was fitted on (see Phase 3 option A/B in the "
                    "handoff), so do not read the Devil's approval bar as "
                    "calibrated for the geometry actually placed.",
                    profile.sl_atr_multiplier,
                    profile.tp_atr_multiplier,
                    trained_sl,
                    trained_tp,
                )
            elif "sl_atr_multiplier" in data or "tp_atr_multiplier" in data:
                from execution.risk_manager import RiskProfile  # lazy: avoids
                # an import cycle through execution/__init__.py at module load.

                profile = RiskProfile.for_asset_class(self.asset_class)
                trained_sl = data.get("sl_atr_multiplier")
                trained_tp = data.get("tp_atr_multiplier")
                if (
                    trained_sl is not None
                    and not _close_enough(trained_sl, profile.sl_atr_multiplier)
                ) or (
                    trained_tp is not None
                    and not _close_enough(trained_tp, profile.tp_atr_multiplier)
                ):
                    raise RuntimeError(
                        "Bracket mismatch: model trained with "
                        f"sl={trained_sl}x/tp={trained_tp}x but this tree's "
                        f"forex profile runs sl={profile.sl_atr_multiplier}x/"
                        f"tp={profile.tp_atr_multiplier}x. Refusing to serve "
                        "train/serve-skewed labels — retrain or point the "
                        "model dir at the matching artifact."
                    )

            logger.info("_validate_metadata: passed (asset_class=%s)", trained_class)
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            logger.warning("_validate_metadata: failed to read %s (%s)", metadata_path, exc)

    def _load_thresholds(self) -> "tuple[float, float]":
        """
        Load pinned decision thresholds from the model directory's
        threshold.json (next to the pkl artifacts, so side models carry
        their own).

        Written by retrainer.save_threshold() after a successful validation
        gate. ``devil_threshold`` has been persisted since the beginning;
        ``angel_threshold`` since 2026-07 — it pins the population the pair
        was trained and bracketed at, so a deployed model cannot be run at a
        different Angel bar than it was fitted for. Either key falls back to
        the current instance value (the constructor default) when absent, so
        older artifacts keep working.

        Returns:
            (angel_threshold, devil_threshold) for production decisions.
        """
        threshold_path = self.angel_path.parent / "threshold.json"
        if not threshold_path.exists():
            logger.warning(
                "_load_thresholds: %s not found — using defaults "
                "angel=%.2f devil=%.2f",
                threshold_path,
                self.angel_threshold,
                self.devil_threshold,
            )
            return self.angel_threshold, self.devil_threshold
        try:
            with open(threshold_path, "r") as fh:
                data = json.load(fh)
            devil = float(data["devil_threshold"])
            angel = float(data.get("angel_threshold", self.angel_threshold))
            logger.info(
                "_load_thresholds: loaded production thresholds "
                "angel=%.4f%s devil=%.4f from %s",
                angel,
                "" if "angel_threshold" in data else " (default; not pinned)",
                devil,
                threshold_path,
            )
            return angel, devil
        except Exception as exc:
            logger.warning(
                "_load_thresholds: failed to read %s (%s) — using defaults "
                "angel=%.2f devil=%.2f",
                threshold_path,
                exc,
                self.angel_threshold,
                self.devil_threshold,
            )
            return self.angel_threshold, self.devil_threshold

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

    def _barrier_meta_mtime(self) -> float:
        """mtime of the barrier meta sidecar, or 0.0 when it does not exist."""
        return (
            os.path.getmtime(self._barrier_meta_path)
            if self._barrier_meta_path.exists()
            else 0.0
        )

    def _load_barriers(self) -> BarrierEstimator:
        """
        Load the learned barrier sidecar from the model directory.

        Three files travel together — barriers_mae.pkl, barriers_mfe.pkl and
        barriers_meta.json — beside the Angel/Devil pickles, so a model dir
        stays self-describing and a promotion moves the weights and the
        geometry contract as one unit.

        Fail-loud, in the same spirit as the cost_ratio and HMM guards: dynamic
        geometry was explicitly requested (BARRIER_GEOMETRY_ENABLED=1 or
        use_barriers=True), so an absent or inconsistent artifact set is a boot
        error, never a silent fall back to the static bracket. An operator who
        asked for learned stops and got static ones has no way to tell.

        Two contracts are checked beyond "the files parse":

        * the horizon. The learned stop is the quantile of a walk N bars long;
          serving it under a different execution lifetime sizes the bracket off
          a distribution the trade never experiences.
        * the feature vocabulary. Every feature the barrier model reads must be
          one the served feature schema produces, or the geometry is being fed
          a column some other build computed.

        And one piece of EVIDENCE, which is the difference between "this
        artifact exists" and "this artifact earned promotion":

        * a recorded verdict. Artifacts carry the Phase 1 gate's result in
          ``barriers_meta.json`` (see ``BarrierEstimator.save(verdict=...)``).
          A recorded FAIL is a REFUSAL here — the gate that decides whether
          learned geometry may replace the static bracket is not advisory, and
          a producer that keeps writing artifacts after a failed gate (a
          retrain hook fires off the Angel/Devil gate, not off the barrier
          gate) must not thereby get them served. An artifact with NO recorded
          verdict is served with a warning: absence is "unknown", and every
          artifact written before this field existed is in that state.
        """
        model_dir = self.angel_path.parent
        est = BarrierEstimator.load(model_dir)

        verdict = est.verdict_
        if verdict is not None and not verdict.get("passed", False):
            raise RuntimeError(
                f"Barrier artifact in {model_dir} records a FAILED promotion "
                f"verdict ({_verdict_summary(verdict)}). The learned geometry "
                "must beat the static bracket on every evaluation fold with "
                "adequate coverage before it may replace the profile "
                "multipliers — refit and re-run scripts/evaluate_barriers.py, "
                "or leave the static bracket in place."
            )
        if verdict is None:
            logger.warning(
                "Barrier artifact in %s carries NO promotion verdict — serving "
                "it on trust. Re-run the producer so the gate result is "
                "recorded beside the weights (scripts/evaluate_barriers.py "
                "--verdict-out).",
                model_dir,
            )
        else:
            logger.info(
                "Barrier artifact in %s records a PASSED promotion verdict (%s)",
                model_dir,
                _verdict_summary(verdict),
            )

        if est.horizon_ != DEFAULT_HORIZON:
            raise RuntimeError(
                f"Barrier artifact in {model_dir} was labelled at a "
                f"{est.horizon_}-bar horizon but this tree's execution "
                f"lifetime is {DEFAULT_HORIZON} bars "
                "(ml.barriers.labels.DEFAULT_HORIZON). Refit at the served "
                "horizon — do not scale a bracket off the wrong walk length."
            )

        unknown = [
            c
            for c in est.feature_cols
            if c not in self.feature_names and c not in _BARRIER_FRAME_ONLY_COLS
        ]
        if unknown:
            raise RuntimeError(
                f"Barrier artifact in {model_dir} reads features the served "
                f"schema does not carry: {unknown}. The Angel's schema is what "
                "the live pipeline produces; a barrier model fitted on "
                "anything else would predict on a column this build does not "
                "compute."
            )
        if "natr_14" not in self.feature_names:
            raise RuntimeError(
                "Barrier geometry needs 'natr_14' — it is the conversion from "
                "the model's NATR-space quantiles to a price distance, and the "
                f"served schema ({model_dir}) does not carry it."
            )
        return est

    def _barrier_geometry(self, features_df: pl.DataFrame) -> Optional[dict]:
        """
        Learned barrier geometry for the newest bar, as the payload execution
        consumes — or None to leave the static bracket in place.

        The multipliers returned are NATR multiples, exactly the units of
        RiskProfile.sl_atr_multiplier / tp_atr_multiplier. That is the whole
        integration: the learned quantiles substitute for the two static
        constants and nothing else about bracket sizing changes, so rounding,
        the three gates and position sizing stay on one code path whether the
        geometry is learned or static.

        Returns None (static bracket) when the sidecar is off, when prediction
        raises, or when the estimator has not been fitted. It does NOT withhold
        geometry over `admissible`: that flag compares Q_MFE(0.50) against
        Q_MAE(0.95), a ratio that is structurally below 1, while its floor was
        written for the static 4.0/2.0 payoff — measured 2026-09-14 on the
        cached basket, rr came out 0.28–0.30 on every fold, i.e. no bar is ever
        admissible, so vetoing on it would silently disable the feature
        outright. The flag and its rr travel in the payload as telemetry until
        the floor is calibrated against the tau pair (see llm_reports).
        """
        est = self._barrier_estimator
        if est is None:
            return None
        try:
            # tail(1): the models are row-independent, so one bar's features
            # are enough — predicting the whole buffer every bar would burn
            # CPU proportional to the warm-up window for identical numbers.
            out = est.predict(features_df.tail(1))[0]
        except Exception as exc:
            logger.error(
                "[barrier] prediction failed (%s) — static bracket this bar", exc
            )
            return None
        return {
            "source": "barrier",
            "sl_atr_mult": float(out.q_mae),
            "tp_atr_mult": float(out.q_mfe),
            "rr": float(out.rr),
            "admissible": bool(out.admissible),
            "tau_mae": float(est.tau_mae),
            "tau_mfe": float(est.tau_mfe),
            "backend": est.backend_,
        }

    def _reload_barriers_if_changed(self) -> None:
        """
        Swap in a newly promoted barrier artifact set, hot.

        Triggered by the META's mtime alone. save() writes both pickles before
        the meta and the meta is replaced last, so a bar that sees a new meta
        reads a complete, matching pair; a pickle replaced on its own is
        invisible here deliberately, because there is no way to tell which of
        the two quantile models a half-written set actually holds.

        A failed reload KEEPS the previously loaded estimator and alerts —
        never a silent disable. Stale geometry that passed its audit is better
        than none, and much better than a half-read pair, which is the same
        policy the Angel/Devil reload follows.
        """
        mtime = self._barrier_meta_mtime()
        if mtime <= self._barrier_mtime:
            return
        try:
            est = self._load_barriers()
        except Exception as exc:
            msg = (
                "[HOT-RELOAD] barrier artifact set in "
                f"{self.angel_path.parent} is unreadable ({exc}) — keeping the "
                "previously loaded geometry until a complete set lands."
            )
            logger.critical(msg)
            self.notification_manager.send_system_message(msg)
            return
        with self._reload_lock:
            self._barrier_estimator = est
        self._barrier_mtime = mtime
        msg = (
            "🔄 [HOT-RELOAD] Learned barrier geometry ingested: "
            f"backend={est.backend_} tau_mae={est.tau_mae:.2f} "
            f"tau_mfe={est.tau_mfe:.2f} horizon={est.horizon_} bars"
        )
        logger.critical(msg)
        self.notification_manager.send_system_message(msg)

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
        angel_new = False
        devil_new = False

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
                        angel_new = True
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
                        devil_new = True
                    except Exception as e:
                        logger.error(f"[HOT-RELOAD] Failed to reload Devil model: {e}")

            # ── pair-seam detection (2026-09-09) ──────────────────────────
            # The retrainer replaces angel and devil separately (two os.replace
            # calls). A bar landing between them used to score a MIXED
            # generation (new Angel + old Devil) — same-schema pairs pass the
            # name-list check below silently. Stand down instead, and only
            # clear when the OTHER half has been reloaded too (or both landed
            # together in one pass).
            if angel_new and devil_new:
                if self._pair_mixed:
                    logger.info("[HOT-RELOAD] Angel/Devil pair consistent again")
                self._pair_mixed = False
                self._pair_pending = None
            elif angel_new:
                self._pair_mixed = True
                self._pair_pending = "devil"
                logger.warning(
                    "[HOT-RELOAD] Angel advanced without Devil — standing "
                    "down signals until the Devil lands"
                )
            elif devil_new:
                if self._pair_pending == "devil":
                    # The half we were waiting for just landed.
                    self._pair_mixed = False
                    self._pair_pending = None
                    logger.info("[HOT-RELOAD] Angel/Devil pair consistent again")
                else:
                    self._pair_mixed = True
                    self._pair_pending = "angel"
                    logger.warning(
                        "[HOT-RELOAD] Devil advanced without Angel — standing "
                        "down signals until the Angel lands"
                    )

            # ── sidecar reloads, independent of the pickle branch ─────────
            # threshold.json and spread_alphas.json land AFTER the pickles in
            # a promotion. Gating them on `if reloaded:` meant a bar that saw
            # only the pickle change re-read the OLD sidecars and then never
            # looked again (pickle mtimes were then equal) — the new pair ran
            # at the old bars forever. Each sidecar now reloads on its own
            # mtime, every bar.
            thr_path = self.angel_path.parent / "threshold.json"
            thr_mtime = os.path.getmtime(thr_path) if thr_path.exists() else 0.0
            if thr_mtime > self._threshold_mtime:
                old_angel = self.angel_threshold
                old_devil = self.devil_threshold
                self.angel_threshold, self.devil_threshold = self._load_thresholds()
                self._threshold_mtime = thr_mtime
                if self.angel_threshold != old_angel:
                    logger.info(
                        "[HOT-RELOAD] Angel threshold updated: %.4f -> %.4f",
                        old_angel,
                        self.angel_threshold,
                    )
                if self.devil_threshold != old_devil:
                    logger.info(
                        "[HOT-RELOAD] Devil threshold updated: %.4f -> %.4f",
                        old_devil,
                        self.devil_threshold,
                    )

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

            # ── barrier sidecar reload (own mtime, same reasoning) ────────
            if self.use_barriers:
                self._reload_barriers_if_changed()

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

        # Pair-seam stand-down (2026-09-09): one half of the Angel/Devil pair
        # advanced without the other (the retrainer replaces them separately).
        # Scoring a mixed generation would judge the new Angel with the old
        # Devil — a silent train/serve skew. Skip the bar instead; the flag
        # clears itself once both loaded weights match disk.
        if self._pair_mixed:
            logger.warning(
                "[MLStrategy] Angel/Devil pair mid-promotion — "
                "skipping signal generation this bar"
            )
            return None

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
                events.emit(
                    "heartbeat",
                    sym=heartbeat_key,
                    median=round(float(np.median(probs)), 4),
                    p75=round(float(np.percentile(probs, 75)), 4),
                    max=round(float(np.max(probs)), 4),
                    proposed=proposed,
                    n_bars=len(probs),
                    threshold=self.angel_threshold,
                )
                self._heartbeat_counter[heartbeat_key] = 0

            # Per-bar record. The heartbeat above summarises 30 bars; this is
            # the bar itself, which is what makes "why isn't it trading?"
            # answerable as a series rather than a rolling median.
            bar_ts = (
                str(df["timestamp"].tail(1)[0]) if "timestamp" in df.columns else None
            )

            if angel_prob < self.angel_threshold:
                logger.debug(
                    f"[{symbol}] Angel rejected | Prob: {angel_prob:.4f} < {self.angel_threshold}"
                )
                events.emit(
                    "bar",
                    sym=heartbeat_key,
                    bar_ts=bar_ts,
                    close=current_price,
                    angel=round(float(angel_prob), 4),
                    devil=None,
                    outcome="angel_reject",
                    proposed=False,
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
                events.emit(
                    "bar",
                    sym=heartbeat_key,
                    bar_ts=bar_ts,
                    close=current_price,
                    angel=round(float(angel_prob), 4),
                    devil=round(float(devil_prob), 4),
                    outcome="devil_veto",
                    proposed=False,
                )
                return None

            # Both Angel and Devil agree — emit raw ATR volatility.
            # RiskManager applies multipliers and floor checks (Path Alpha).
            natr_value = float(latest_features_df["natr_14"].to_numpy()[0])
            # TA-Lib NATR is a percentage; convert to absolute ATR
            atr_abs = (natr_value / 100.0) * current_price

            # Learned geometry, when the sidecar is on. These are NATR
            # multiples: RiskManager multiplies them by the same raw ATR the
            # static profile multipliers would have used, so the learned
            # quantiles REPLACE 2.0x/4.0x rather than compounding with them.
            geometry = self._barrier_geometry(features_df)
            if geometry:
                geometry_note = (
                    f"barrier sl={geometry['sl_atr_mult']:.3f}x "
                    f"tp={geometry['tp_atr_mult']:.3f}x "
                    f"(rr={geometry['rr']:.2f}, {geometry['backend']})"
                )
            else:
                geometry_note = "static profile multipliers"

            # Telemetry for the emitted bar, computed into locals first: the
            # events sink sits on the bar path, and a source-level test forbids
            # arithmetic or indexing inside an emit() call (test_events.py).
            # The keys are always present and null when the sidecar is off, so
            # a consumer's schema does not change with the switch.
            geometry_source = "barrier" if geometry else "static"
            sl_atr_mult = geometry["sl_atr_mult"] if geometry else None
            tp_atr_mult = geometry["tp_atr_mult"] if geometry else None
            barrier_rr = geometry["rr"] if geometry else None
            barrier_admissible = geometry["admissible"] if geometry else None

            logger.info(
                f"[{symbol}] ANGEL & DEVIL AGREEMENT | "
                f"Price={current_price:.2f} | "
                f"Angel Prob: {angel_prob:.2f} | "
                f"Devil Prob: {devil_prob:.2f} | "
                f"raw_ATR={atr_abs:.4f} | "
                f"geometry: {geometry_note}"
            )

            events.emit(
                "bar",
                sym=heartbeat_key,
                bar_ts=bar_ts,
                close=current_price,
                angel=round(float(angel_prob), 4),
                devil=round(float(devil_prob), 4),
                outcome="agreement",
                proposed=True,
                atr=round(atr_abs, 6),
                geometry=geometry_source,
                sl_atr_mult=sl_atr_mult,
                tp_atr_mult=tp_atr_mult,
                barrier_rr=barrier_rr,
                barrier_admissible=barrier_admissible,
            )

            metadata = {
                "symbol": symbol,
                "angel_prob": float(angel_prob),
                "devil_prob": float(devil_prob),
                "atr_abs": atr_abs,
                "timestamp": df["timestamp"].tail(1)[0],
            }
            if geometry:
                metadata[BARRIER_GEOMETRY_KEY] = geometry

            return Signal(
                direction="long",
                entry_price=current_price,
                raw_sl_distance=atr_abs,
                raw_tp_distance=atr_abs,
                metadata=metadata,
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
