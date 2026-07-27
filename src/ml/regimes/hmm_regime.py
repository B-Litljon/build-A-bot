"""
Per-instrument Gaussian HMM regime detector — soft-feature mode.

Each symbol gets its own GaussianHMM fit on (log_return, natr_14) so that
session/volatility regimes are modelled in the space of returns rather than
price levels. The fitted model's smoothed posterior P(state | observations)
becomes 3 new feature columns the downstream classifier can condition on.

Walk-forward leakage is the caller's responsibility:
  - Pass training-fold data ONLY to fit_regime_models().
  - Score both train and val frames with the resulting dict via predict_regime_probs().
  - For the final production model, fit on all retraining data and persist
    via save_hmm_models(); MLStrategy loads it for live inference.

EXPERIMENTAL and OFF by default -- only active when RETRAIN_USE_HMM=1. The
idea: markets move between hidden modes (quiet drift, trending, violent), which
nobody labels directly. The model infers those modes statistically and hands
the classifier its confidence in each one as three extra inputs.

Glossary:
    N_STATES -- 3 regimes. Not named or interpreted anywhere; they are whatever
        the fit finds, so "state 0" has no fixed meaning across symbols or runs.
    HMM_INPUT_COLS -- what the regime model observes: log_return and natr_14,
        i.e. direction-of-move and size-of-move. Deliberately NOT price level,
        so the regimes are comparable across instruments and eras.
    HMM_OUTPUT_COLS -- the three added columns hmm_state_0_prob ... _2_prob.
        These are what get appended to the classifier's feature list.
    MIN_FIT_ROWS -- 200. Fewer rows than this and the symbol is skipped
        entirely rather than fitted on noise.
    _clean_for_hmm -- replaces non-finite values with that column's mean; the
        HMM fitter rejects NaN and infinity outright.
    fit_regime_models -- one model per symbol. Returns None for any symbol that
        was skipped or whose fit failed, so callers must handle None.
    predict_regime_probs -- adds the three columns. Symbols with no model get a
        uniform 1/3 across all states -- a deliberately uninformative value the
        classifier can ignore, rather than a gap that would drop the row.
    Uniform padding -- if a fit collapses to fewer states than requested, the
        missing columns are padded uniform so the feature vector always has the
        expected width.
    save_hmm_models / load_hmm_models -- joblib round-trip for the per-symbol
        dict, written atomically next to the Angel and Devil models so the
        three artifacts always travel together.
    Leakage rule -- fit on TRAINING rows only, then score both training and
        validation with that fitted model. Fitting on all the data first would
        let the validation fold's own future inform its regime labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

N_STATES: int = 3
HMM_INPUT_COLS: List[str] = ["log_return", "natr_14"]
HMM_OUTPUT_COLS: List[str] = [f"hmm_state_{i}_prob" for i in range(N_STATES)]
MIN_FIT_ROWS: int = 200


def _clean_for_hmm(X: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with per-column finite means. HMM training rejects both."""
    finite_mask = np.isfinite(X)
    has_any_finite = finite_mask.any(axis=0)
    safe = np.where(finite_mask, X, np.nan)
    means = np.where(has_any_finite, np.nanmean(safe, axis=0), 0.0)
    means = np.where(np.isfinite(means), means, 0.0)
    return np.where(finite_mask, X, means)


def fit_regime_models(
    train_df: pl.DataFrame,
    n_states: int = N_STATES,
    random_state: int = 42,
) -> Dict[str, Optional[GaussianHMM]]:
    """Fit one GaussianHMM per symbol on train_df's HMM_INPUT_COLS."""
    symbols = train_df["symbol"].unique().to_list()
    models: Dict[str, Optional[GaussianHMM]] = {}

    for sym in symbols:
        sym_df = train_df.filter(pl.col("symbol") == sym).select(HMM_INPUT_COLS)
        X = _clean_for_hmm(sym_df.to_numpy())

        if len(X) < MIN_FIT_ROWS:
            logger.warning(
                "HMM[%s]: skipped (%d rows < %d minimum)",
                sym, len(X), MIN_FIT_ROWS,
            )
            models[sym] = None
            continue

        try:
            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=50,
                tol=1e-3,
                random_state=random_state,
            )
            hmm.fit(X)
            models[sym] = hmm
            logger.info(
                "HMM[%s]: fit on %d rows | converged=%s | log_likelihood=%.2f",
                sym, len(X), hmm.monitor_.converged, float(hmm.score(X)),
            )
        except Exception as exc:
            logger.warning("HMM[%s]: fit failed (%s)", sym, exc)
            models[sym] = None

    return models


def predict_regime_probs(
    df: pl.DataFrame,
    models: Dict[str, Optional[GaussianHMM]],
    n_states: int = N_STATES,
) -> pl.DataFrame:
    """
    Augment df with HMM_OUTPUT_COLS posterior probabilities.

    Symbols with no fitted model (insufficient data, fit failure, missing key)
    get uniform 1/n_states across all states — a no-op feature value the
    classifier can ignore.
    """
    out_frames: List[pl.DataFrame] = []
    uniform = 1.0 / n_states

    for sym in df["symbol"].unique().to_list():
        sym_df = df.filter(pl.col("symbol") == sym)
        hmm = models.get(sym)

        if hmm is None:
            sym_df = sym_df.with_columns(
                [pl.lit(uniform).alias(c) for c in HMM_OUTPUT_COLS[:n_states]]
            )
            out_frames.append(sym_df)
            continue

        X = _clean_for_hmm(sym_df.select(HMM_INPUT_COLS).to_numpy())
        try:
            posteriors = hmm.predict_proba(X)
        except Exception as exc:
            logger.warning(
                "HMM[%s]: predict_proba failed (%s) — filling uniform",
                sym, exc,
            )
            posteriors = np.full((len(sym_df), n_states), uniform)

        # Guard: if the fit produced fewer states than requested (rare, but
        # convergence can collapse states), pad with uniform.
        if posteriors.shape[1] < n_states:
            pad = np.full(
                (posteriors.shape[0], n_states - posteriors.shape[1]),
                uniform,
            )
            posteriors = np.hstack([posteriors, pad])

        sym_df = sym_df.with_columns(
            [
                pl.Series(HMM_OUTPUT_COLS[i], posteriors[:, i])
                for i in range(n_states)
            ]
        )
        out_frames.append(sym_df)

    return pl.concat(out_frames, how="vertical_relaxed").sort(["symbol", "timestamp"])


def save_hmm_models(
    models: Dict[str, Optional[GaussianHMM]],
    target_path: Path,
) -> None:
    """Atomic joblib write of the per-symbol HMM dict alongside Angel/Devil."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    joblib.dump(models, temp_path)
    temp_path.replace(target_path)
    fitted = sum(1 for m in models.values() if m is not None)
    logger.info(
        "[ATOMIC] HMM regime models saved: %s (%d/%d symbols fitted)",
        target_path, fitted, len(models),
    )


def load_hmm_models(source_path: Path) -> Dict[str, Optional[GaussianHMM]]:
    """Load the per-symbol HMM dict written by save_hmm_models."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"HMM artifact not found: {source_path}")
    return joblib.load(source_path)
