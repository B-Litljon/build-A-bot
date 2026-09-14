"""
Quantile barrier estimator — predicts instance-specific MAE/MFE quantiles.

Replaces the static ATR-multiple bracket with two quantile regressions fitted
on excursion labels (see labels.py):

    stop distance   <- Q_MAE(0.95 | X_t)   upper tail of adverse excursion
    target distance <- Q_MFE(0.50 | X_t)   median favourable excursion

Fit on the same feature vocabulary the Angel/Devil models see (BASE_FEATURE_COLS
from the retrainer), so no new live feature plumbing is needed — the barrier
models read the frame FeaturePipeline already produces.

Why quantiles and not the static 2.0x/4.0x multiples: the calibration-inversion
finding (2026-08-24, reproduced 2026-09-08) shows the fixed multiple stops out
73% of high-conviction entries — those bars are a volatility subpopulation where
a constant multiple is proportionally wrong. A conditional quantile widens the
stop exactly where adverse excursion fat-tails and keeps the target honest.

Monotonicity on volatility features is enforced by AUDIT, not construction:
LightGBM's quantile objective rejects monotone_constraints outright ("Cannot
use ``monotone_constraints`` in quantile objective"), so the fitted model is
checked on the natr_14/vol_rel decile ladder at fit time and the estimator
raises rather than return a geometry where higher volatility yields a tighter
excursion quantile. The no-lightgbm binned fallback is monotone by
construction instead (bin ladder is isotonised at fit). CatBoost DOES accept
monotone_constraints under its quantile loss, so on that backend (the default)
the constraint holds by construction too and the audit is a redundant check
rather than the only guard.

``save``/``load`` are the live sidecar contract: the artifact set is two
pickles (one per quantile) plus a JSON meta, written pickles-first and
os.replace'd into place, because the live strategy hot-reloads the model
directory and triggers on the META's mtime — so a half-promoted pair is
invisible rather than silently mixed.

Glossary:
    DEFAULT_TAU_MAE / DEFAULT_TAU_MFE -- the stop-side and target-side
        quantiles. 0.95 admits ~5% of walks past the stop (the tail the trade
        tolerates); 0.50 aims the target at the median favourable path.
    BarrierOutput -- one inference: stop/target distances in PRICE units (the
        Signal contract carries distances, not levels) plus the raw NATR-
        space quantiles and the implied reward:risk, for telemetry and the
        pre-trade veto.
    BarrierEstimator -- the fit/predict contract; two quantile regressions
        under the hood, kept behind the interface so the family is swappable
        (CatBoost by default via BARRIER_FAMILY, LightGBM and a binned
        fallback behind it).
    feature_names_in_ -- the fitted feature list; parity with the served
        feature frame is checked at fit time so a silent schema drift cannot
        produce garbage barriers. Also the contract the live sidecar checks
        against the strategy's own feature schema before it will predict.
    horizon_ -- the forward-window length (BARS) the served weights were
        LABELLED at, restored by load() from the meta. It must equal the
        execution lifetime the geometry is served into; the live side
        refuses an artifact that disagrees rather than sizing a bracket off a
        horizon the trade never experiences. None until load() — fit() cannot
        know it, because the labels frame does not carry it.
    rr_floor -- minimum implied reward:risk for a BarrierOutput to be
        tradeable. Reported at predict time (outputs below it are flagged,
        not hidden) so a pre-trade veto can see honest numbers.
    _AUDIT_TOL_ATR -- materiality floor of the fit-time monotonicity audit,
        in ATR units. A response curve may wiggle below its running peak by
        up to this much before the fit is refused; brackets quantize far
        coarser, so smaller dips are untradeable noise.
    used_lightgbm_ / used_catboost_ / backend_ -- set by fit() (and restored
        by load()): which backend produced the served geometry. Telemetry,
        and the thing that tells an operator whether a fitted model was
        constrained by construction (catboost) or merely audited (lightgbm).
    BARRIER_MAE_FILENAME / BARRIER_MFE_FILENAME / BARRIER_META_FILENAME --
        the three files in a persisted artifact set, laid out beside
        angel_latest.pkl so a model dir stays one self-describing directory.
"""

from dataclasses import dataclass
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import polars as pl

from ml.barriers.labels import DEFAULT_HORIZON

logger = logging.getLogger(__name__)

DEFAULT_TAU_MAE = 0.95
DEFAULT_TAU_MFE = 0.50
DEFAULT_RR_FLOOR = 2.0

# The persisted artifact set, written beside the Angel/Devil pickles so a model
# directory stays self-describing. Fixed names (not configurable) because the
# live sidecar looks for exactly these.
BARRIER_MAE_FILENAME = "barriers_mae.pkl"
BARRIER_MFE_FILENAME = "barriers_mfe.pkl"
BARRIER_META_FILENAME = "barriers_meta.json"

# Materiality floor for the monotonicity audit: a fitted model is refused only
# if its stop/target response to rising volatility dips by more than this many
# ATRs below the running peak. Set well above bracket price quantization noise
# but far below the 73%-stop-out kind of inversion the audit exists to catch.
_AUDIT_TOL_ATR = 0.05

# Volatility features whose conditional effect on excursion size is monotone
# increasing by construction — the model cannot learn "more volatile bars get
# tighter stops", which is precisely the failure mode the inversion finding
# points at.
MONOTONE_INCREASING = ("natr_14", "vol_rel")


@dataclass(frozen=True)
class BarrierOutput:
    """One bar's barrier geometry, in price units and NATR units."""

    raw_sl_distance: float
    raw_tp_distance: float
    q_mae: float
    q_mfe: float
    rr: float
    admissible: bool


class _BinnedLadder:
    """
    The no-lightgbm backend: an isotonised empirical tau-quantile per natr_14
    quartile bin (edges from the training frame only, so it stays causal).

    A module-level class rather than the closure the first version used,
    because a locally-defined function cannot be pickled — the fallback was
    therefore unpersistable, which would have made the degraded backend
    unusable by the live sidecar exactly when the libraries were missing.
    """

    def __init__(self, edges, ladder, natr_idx: int):
        self.edges = np.asarray(edges, dtype=float)
        self.ladder = np.asarray(ladder, dtype=float)
        self.natr_idx = int(natr_idx)

    def predict(self, X) -> np.ndarray:
        b = np.digitize(np.asarray(X, dtype=float)[:, self.natr_idx], self.edges)
        return self.ladder[b]

    # The dict this is wrapped in has always held a plain CALLABLE under
    # "predict" (the closure it replaced), and both _predict_quantile and the
    # fallback tests address it that way. Keeping the handle callable means the
    # backend swap is invisible to every caller.
    __call__ = predict


class BarrierEstimator:
    """
    Two quantile regressions (MAE tail, MFE median) on the shared feature set.

    LightGBM with quantile objective; if lightgbm is unavailable the estimator
    falls back to unconditional empirical quantiles per feature-quantile bin of
    natr_14 — a deliberate degradation that stays causal rather than crashing.
    """

    def __init__(
        self,
        feature_cols: List[str],
        tau_mae: float = DEFAULT_TAU_MAE,
        tau_mfe: float = DEFAULT_TAU_MFE,
        rr_floor: float = DEFAULT_RR_FLOOR,
        family: Optional[str] = None,
    ):
        if tau_mfe >= tau_mae:
            raise ValueError("tau_mfe must be below tau_mae (median target, tail stop)")
        self.feature_cols = list(feature_cols)
        self.tau_mae = tau_mae
        self.tau_mfe = tau_mfe
        self.rr_floor = rr_floor
        self.feature_names_in_: List[str] = list(feature_cols)
        self.family = (family or os.getenv("BARRIER_FAMILY", "catboost")).strip().lower()
        self._model_mae = None
        self._model_mfe = None
        # Set by fit(): True when the LightGBM path was used, False when the
        # estimator degraded to the binned fallback (no lightgbm installed).
        self.used_lightgbm_: Optional[bool] = None
        self.used_catboost_: Optional[bool] = None
        self.backend_: Optional[str] = None
        # The forward-window length (BARS) these weights were labelled at.
        # fit() cannot know it — the labels frame does not carry it — so it is
        # restored by load() from the artifact meta and checked against the
        # execution lifetime by whoever serves the geometry.
        self.horizon_: Optional[int] = None

    def fit(
        self,
        df: pl.DataFrame,
        labels: pl.DataFrame,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BarrierEstimator":
        """
        Fit both quantile models.

        df: the feature frame (X). labels: the frame produced by
        compute_excursions() (joined row-for-row with df). Rows that are null
        in either label are dropped before fitting, never imputed.
        """
        y_mae = labels["mae_natr"].to_numpy().astype(float)
        y_mfe = labels["mfe_natr"].to_numpy().astype(float)
        X = self._X(df)
        ok = np.isfinite(y_mae) & np.isfinite(y_mfe) & np.isfinite(X).all(axis=1)
        X, y_mae, y_mfe = X[ok], y_mae[ok], y_mfe[ok]
        if len(X) < 100:
            raise ValueError(
                f"barrier fit needs >=100 fully-labelled rows, got {len(X)}"
            )
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)[ok]

        self._model_mae = self._fit_quantile(X, y_mae, self.tau_mae, sample_weight)
        self._model_mfe = self._fit_quantile(X, y_mfe, self.tau_mfe, sample_weight)
        return self

    def predict(self, df: pl.DataFrame) -> List[BarrierOutput]:
        """Barrier geometry for each row of df, in the same order."""
        X = self._X(df)
        q_mae = np.maximum(self._predict_quantile(self._model_mae, X, self.tau_mae), 1e-6)
        q_mfe = np.maximum(self._predict_quantile(self._model_mfe, X, self.tau_mfe), 0.0)
        natr = df["natr_14"].to_numpy().astype(float)
        close = df["close"].to_numpy().astype(float)
        atr_abs = close * natr / 100.0

        out: List[BarrierOutput] = []
        for i in range(len(X)):
            rr = float(q_mfe[i] / q_mae[i]) if q_mae[i] > 0 else 0.0
            sl = float(q_mae[i] * atr_abs[i])
            tp = float(q_mfe[i] * atr_abs[i])
            admissible = (
                np.isfinite(sl)
                and np.isfinite(tp)
                and sl > 0
                and rr >= self.rr_floor
            )
            out.append(
                BarrierOutput(
                    raw_sl_distance=sl,
                    raw_tp_distance=tp,
                    q_mae=float(q_mae[i]),
                    q_mfe=float(q_mfe[i]),
                    rr=rr,
                    admissible=admissible,
                )
            )
        return out

    # ── persistence (the live sidecar contract) ────────────────────────────

    def save(self, model_dir, horizon: int = DEFAULT_HORIZON) -> dict:
        """
        Persist both quantile models + the meta contract into ``model_dir``.

        Three files, mirroring Angel/Devil + their metadata sidecar: the two
        pickles are opaque to the loader, the JSON is what a human (and
        ``load``) reads to know what geometry these weights encode. Every file
        is written under a temp name and replaced, because the live strategy
        watches this directory.

        Write ORDER is load-bearing: both pickles land BEFORE the meta, and
        the live reload triggers on the meta's mtime alone. A bar that sees a
        new meta therefore reads a complete matching pair, and a half-promoted
        set (one pickle replaced, meta unchanged) is invisible rather than
        silently mixed.

        ``horizon`` is recorded, never inferred: it is the forward-window
        length the labels were built at, and it must match the execution
        lifetime the geometry is served into. Returns the meta it wrote.
        """
        if self._model_mae is None or self._model_mfe is None:
            raise RuntimeError(
                "BarrierEstimator.save() called before fit() — refusing to "
                "persist an unfitted estimator"
            )
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        for name, model in (
            (BARRIER_MAE_FILENAME, self._model_mae),
            (BARRIER_MFE_FILENAME, self._model_mfe),
        ):
            target_path = model_dir / name
            temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
            joblib.dump(model, temp_path)
            temp_path.replace(target_path)

        meta = {
            "artifact": "barriers",
            "backend": self.backend_,
            "family": self.family,
            "feature_cols": list(self.feature_cols),
            "tau_mae": float(self.tau_mae),
            "tau_mfe": float(self.tau_mfe),
            "rr_floor": float(self.rr_floor),
            "horizon": int(horizon),
            "monotone_increasing": [
                c for c in MONOTONE_INCREASING if c in self.feature_cols
            ],
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = model_dir / BARRIER_META_FILENAME
        temp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        temp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True))
        temp_meta.replace(meta_path)
        logger.info(
            "[ATOMIC] barrier artifact set saved: %s (%s, %d features, "
            "tau_mae=%.2f tau_mfe=%.2f, horizon=%d bars)",
            model_dir,
            self.backend_,
            len(self.feature_cols),
            self.tau_mae,
            self.tau_mfe,
            meta["horizon"],
        )
        return meta

    @classmethod
    def load(cls, model_dir) -> "BarrierEstimator":
        """
        Restore an artifact set written by :meth:`save`.

        Raises rather than returning a half-loaded estimator: a missing file,
        an unparsable meta, or a meta without a horizon or a feature list is a
        broken promotion, and predicting through it would size brackets off
        weights nobody can identify. The live strategy decides what a failure
        means (it keeps the previously loaded pair and alerts); the estimator's
        job is to refuse. Feature parity against the served frame is still
        enforced at predict time by ``_X``, the same check the fit path uses.
        """
        model_dir = Path(model_dir)
        meta_path = model_dir / BARRIER_META_FILENAME
        mae_path = model_dir / BARRIER_MAE_FILENAME
        mfe_path = model_dir / BARRIER_MFE_FILENAME
        missing = [p.name for p in (meta_path, mae_path, mfe_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"barrier artifact set incomplete in {model_dir}: missing "
                f"{missing} — expected {BARRIER_MAE_FILENAME}, "
                f"{BARRIER_MFE_FILENAME} and {BARRIER_META_FILENAME} together"
            )

        meta = json.loads(meta_path.read_text())
        feature_cols = meta.get("feature_cols")
        if not feature_cols or not all(isinstance(c, str) for c in feature_cols):
            raise ValueError(
                f"{meta_path} carries no usable 'feature_cols' — cannot "
                "establish the inference schema, so the weights are "
                "unidentifiable"
            )
        if "horizon" not in meta:
            raise ValueError(
                f"{meta_path} carries no 'horizon' — the label window the "
                "geometry was fitted at is unknown, and serving it would size "
                "brackets off a trade lifetime nobody declared"
            )

        est = cls(
            feature_cols=feature_cols,
            tau_mae=float(meta.get("tau_mae", DEFAULT_TAU_MAE)),
            tau_mfe=float(meta.get("tau_mfe", DEFAULT_TAU_MFE)),
            rr_floor=float(meta.get("rr_floor", DEFAULT_RR_FLOOR)),
            family=str(meta.get("family") or "catboost"),
        )
        est._model_mae = joblib.load(mae_path)
        est._model_mfe = joblib.load(mfe_path)
        est.backend_ = meta.get("backend")
        est.used_catboost_ = est.backend_ == "catboost"
        est.used_lightgbm_ = est.backend_ == "lightgbm"
        est.horizon_ = int(meta["horizon"])
        logger.info(
            "BarrierEstimator restored from %s: backend=%s tau_mae=%.2f "
            "tau_mfe=%.2f horizon=%d features=%d",
            model_dir,
            est.backend_,
            est.tau_mae,
            est.tau_mfe,
            est.horizon_,
            len(est.feature_cols),
        )
        return est

    # ── internals ──────────────────────────────────────────────────────────

    def _X(self, df: pl.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"barrier feature frame missing columns: {missing}")
        X = df.select(self.feature_cols).to_numpy().astype(float)
        if np.isnan(X).any():
            raise ValueError("barrier feature frame carries NaNs; impute upstream")
        return X

    def _fit_quantile(self, X, y, tau, w):
        if self.family == "catboost":
            try:
                return self._fit_catboost(X, y, tau, w)
            except ImportError:
                pass
        return self._fit_lightgbm(X, y, tau, w)

    def _fit_catboost(self, X, y, tau, w):
        from catboost import CatBoostRegressor

        mc = [1 if c in MONOTONE_INCREASING else 0 for c in self.feature_cols]
        params = {
            "loss_function": f"Quantile:alpha={tau}",
            "iterations": 250,
            "learning_rate": 0.05,
            "depth": 5,
            "random_seed": 42,
            "verbose": 0,
            # CatBoost otherwise writes a catboost_info/ training-log directory
            # into the CURRENT WORKING DIRECTORY, so simply running the Phase 1
            # evaluation from the repo root dirties a tracked tree (it was
            # committed once already, 2026-09-14). The estimator's fits are
            # research/eval runs that must leave no trace on disk.
            "allow_writing_files": False,
        }
        if any(mc):
            params["monotone_constraints"] = mc
        model = CatBoostRegressor(**params)
        try:
            model.fit(X, y, sample_weight=w)
        except Exception as e:
            if "constant" in str(e).lower() or "ignored" in str(e).lower():
                # All features are constant (e.g. synthetic test frames); fallback to LightGBM
                return self._fit_lightgbm(X, y, tau, w)
            raise
        self._audit_monotone(model, X)
        self.used_catboost_ = True
        self.used_lightgbm_ = False
        self.backend_ = "catboost"
        return model

    def _fit_lightgbm(self, X, y, tau, w):
        try:
            import lightgbm as lgb

            params = {
                "objective": "quantile",
                "alpha": tau,
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 40,
                "subsample": 0.9,
                "subsample_freq": 1,
                "colsample_bytree": 0.9,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y, sample_weight=w)
            # LightGBM explicitly rejects monotone_constraints under the
            # quantile objective ("Cannot use ``monotone_constraints`` in
            # quantile objective"). Enforcement-by-construction is unavailable
            # there, so the constraint is enforced by AUDIT instead: the
            # fitted model's response on the natr_14 ladder must not drop
            # materially below its running peak, and this estimator refuses
            # to return it otherwise. Binned fallback is monotone by
            # construction.
            self._audit_monotone(model, X)
            self.used_lightgbm_ = True
            self.used_catboost_ = False
            self.backend_ = "lightgbm"
            return model
        except ImportError:
            self.used_lightgbm_ = False
            self.used_catboost_ = False
            self.backend_ = "binned"
            return self._fit_binned_fallback(X, y, tau)

    def _monotone_constraints(self):
        """
        Retained for introspection/tests: the +1/0/… vector the LightGBM path
        WOULD pass if the quantile objective accepted monotone_constraints.
        The quantile objective does not, so this vector is informational only —
        enforcement happens in ``_audit_monotone``.
        """
        vec = [1 if c in MONOTONE_INCREASING else 0 for c in self.feature_cols]
        return vec if any(vec) else None

    def _audit_monotone(self, model, X):
        """
        Refuse a fitted model whose response to rising volatility is a tighter
        excursion quantile. Sweeps each MONOTONE_INCREASING feature over its
        observed decile ladder with all other features held at their medians,
        and raises when predictions drop materially below the running peak.

        "Materially" = more than ``_AUDIT_TOL_ATR`` of one ATR across the
        whole ladder. Smaller wiggles are noise on plateaus (LightGBM quantile
        fits can drift a few hundredths of an ATR on a flat relationship) and
        carry no tradeable meaning — bracket prices quantize far coarser than
        that. A genuine inversion (higher natr -> tighter stop, the failure
        mode from the 2026-08 calibration finding) is far larger than the
        tolerance and is refused.
        """
        med = np.nanmedian(X, axis=0)
        for name in MONOTONE_INCREASING:
            if name not in self.feature_cols:
                continue
            idx = self.feature_cols.index(name)
            col = X[:, idx]
            ladder = np.nanquantile(col, np.linspace(0.05, 0.95, 19))
            probe = np.repeat(med[None, :], len(ladder), axis=0)
            probe[:, idx] = ladder
            preds = np.asarray(model.predict(probe), dtype=float)
            peak = np.maximum.accumulate(preds)
            if (preds < peak - _AUDIT_TOL_ATR).any():
                raise ValueError(
                    f"barrier fit violated monotonicity on '{name}': rising "
                    f"volatility produced a tighter quantile by more than "
                    f"{_AUDIT_TOL_ATR} ATR — refit or drop the feature before "
                    f"this geometry reaches a bracket."
                )

    def _predict_quantile(self, model, X, tau):
        if model is None:
            return np.full(len(X), np.nan)
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X), dtype=float)
        return np.asarray(model["predict"](X), dtype=float)

    def _fit_binned_fallback(self, X, y, tau):
        """
        No-lightgbm fallback: empirical tau-quantile of y within natr_14
        quartile bins. Causal (bin edges come from the training frame only)
        and honest about being coarse.

        Raw per-bin quantiles are NOT monotone — adjacent bins can invert on
        sampling noise, which is exactly the "more volatile bars get tighter
        stops" failure the monotone constraint exists to prevent. The bin
        ladder is therefore forced isotone with maximum.accumulate; empty
        leading bins seed from the global quantile so the ladder always has
        four rungs.
        """
        natr_idx = self.feature_cols.index("natr_14")
        natr = X[:, natr_idx]
        edges = np.nanquantile(natr, [0.25, 0.5, 0.75])
        bins = np.digitize(natr, edges)
        qs = {}
        for b in np.unique(bins):
            qs[b] = float(np.nanquantile(y[bins == b], tau))
        global_q = float(np.nanquantile(y, tau))

        # Dense ladder over all 4 bins; forward-fill empty bins from the
        # previous rung (or the global quantile before the first real rung).
        rung = global_q
        ladder = []
        for b in range(len(edges) + 1):
            rung = qs.get(b, rung)
            ladder.append(rung)
        ladder = np.maximum.accumulate(np.asarray(ladder, dtype=float))

        # Wrapped in a callable handle (not a closure) so it pickles — see
        # _BinnedLadder. The dict shape is kept: _predict_quantile and the
        # tests address the model as model["predict"].
        return {"predict": _BinnedLadder(edges, ladder, natr_idx)}


def static_baseline_loss(
    y: np.ndarray, sl_mult: float, tau: float
) -> float:
    """
    Pinball loss of the incumbent static multiple (2.0x SL etc.) treated as a
    constant quantile predictor in NATR space — the number the learned
    estimator must beat on every fold and the holdout for Phase 1 promotion.
    The constant is already in NATR units, so no ATR vector is needed.
    """
    from ml.barriers.labels import pinball_loss

    return pinball_loss(y, np.full_like(y, sl_mult), tau)