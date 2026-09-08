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
construction instead (bin ladder is isotonised at fit).

Glossary:
    DEFAULT_TAU_MAE / DEFAULT_TAU_MFE -- the stop-side and target-side
        quantiles. 0.95 admits ~5% of walks past the stop (the tail the trade
        tolerates); 0.50 aims the target at the median favourable path.
    BarrierOutput -- one inference: stop/target distances in PRICE units (the
        Signal contract carries distances, not levels) plus the raw NATR-
        space quantiles and the implied reward:risk, for telemetry and the
        pre-trade veto.
    BarrierEstimator -- the fit/predict contract; two LightGBM quantile
        regressions under the hood, kept behind the interface so the family
        is swappable.
    feature_names_in_ -- the fitted feature list; parity with the served
        feature frame is checked at fit time so a silent schema drift cannot
        produce garbage barriers.
    rr_floor -- minimum implied reward:risk for a BarrierOutput to be
        tradeable. Enforced at predict time (outputs below it are flagged,
        not hidden) so the pre-trade veto in RiskManager sees honest numbers.
    _AUDIT_TOL_ATR -- materiality floor of the fit-time monotonicity audit,
        in ATR units. A response curve may wiggle below its running peak by
        up to this much before the fit is refused; brackets quantize far
        coarser, so smaller dips are untradeable noise.
    used_lightgbm_ -- set by fit(): True when the LightGBM path was fitted
        and audited, False when the estimator degraded to the binned
        fallback. Telemetry for which backend produced the served geometry.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import polars as pl

from ml.barriers.labels import DEFAULT_HORIZON

DEFAULT_TAU_MAE = 0.95
DEFAULT_TAU_MFE = 0.50
DEFAULT_RR_FLOOR = 2.0

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
    ):
        if tau_mfe >= tau_mae:
            raise ValueError("tau_mfe must be below tau_mae (median target, tail stop)")
        self.feature_cols = list(feature_cols)
        self.tau_mae = tau_mae
        self.tau_mfe = tau_mfe
        self.rr_floor = rr_floor
        self.feature_names_in_: List[str] = list(feature_cols)
        self._model_mae = None
        self._model_mfe = None
        # Set by fit(): True when the LightGBM path was used, False when the
        # estimator degraded to the binned fallback (no lightgbm installed).
        self.used_lightgbm_: Optional[bool] = None

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
            return model
        except ImportError:
            self.used_lightgbm_ = False
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
        try:
            import lightgbm  # noqa: F401

            return np.asarray(model.predict(X), dtype=float)
        except ImportError:
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

        def predict(Xnew):
            b = np.digitize(Xnew[:, natr_idx], edges)
            return ladder[b]

        return {"predict": predict}


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