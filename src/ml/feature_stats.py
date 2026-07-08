"""
Feature-distribution statistics + Population Stability Index (PSI).

One shared implementation used by three consumers (symmetry, as always):
  * the retrainer — computes stats from the FINAL training frame (post-veto,
    post-clean, i.e. exactly the population the models saw) and saves them as
    a ``feature_stats.json`` sidecar in the model dir on gate pass;
  * scripts/generate_feature_stats.py — backfills the sidecar for model dirs
    trained before this existed (rebuilds the training frame for the model's
    recorded window and computes the same stats);
  * scripts/probe_model.py — compares LIVE feature distributions against the
    sidecar to answer "is the model quiet because its inputs drifted out of
    the training distribution, or because the setups genuinely aren't there?"

PSI convention (standard 10-bin):
    Bin edges are the training deciles, so expected mass is ~10% per bin by
    construction. PSI = Σ (actual% − expected%) · ln(actual% / expected%).
        < 0.10  stable
        0.10–0.25  moderate drift
        > 0.25  severe drift (the model is looking at a foreign regime)
    Low-cardinality features (session flags, hour_of_day) use per-value
    categorical bins instead of deciles.

Stats are stored pooled AND per-symbol: the model trains pooled, but drift
diagnosis is per-instrument (instruments have different baseline levels for
several features, so live-vs-pooled alone would false-positive).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Features with ≤ this many unique training values get categorical bins.
_CATEGORICAL_MAX_UNIQUE = 24  # covers hour_of_day (24) and the 0/1 sessions

# Epsilon smoothing so empty bins don't produce infinite PSI terms.
_PSI_EPS = 1e-4

STATS_FILENAME = "feature_stats.json"


# ═══════════════════════════════════════════════════════════════════════════
# COMPUTE (training side)
# ═══════════════════════════════════════════════════════════════════════════


def _column_stats(values: np.ndarray) -> Optional[dict]:
    """Stats for one feature over one population (pooled or one symbol)."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    uniq = np.unique(finite)
    base = {
        "n": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }
    if len(uniq) <= _CATEGORICAL_MAX_UNIQUE:
        counts = {
            str(float(v)): int((finite == v).sum()) for v in uniq
        }
        return {**base, "kind": "categorical", "freqs": counts}
    # Continuous: decile edges (interior 9 edges; outer bins are open-ended).
    edges = np.quantile(finite, np.linspace(0.1, 0.9, 9))
    # Expected mass per bin — nominally 0.1 each, but duplicate edges (heavy
    # ties) can collapse bins, so store the realized training mass.
    bins = np.concatenate([[-np.inf], edges, [np.inf]])
    hist, _ = np.histogram(finite, bins=bins)
    expected = (hist / hist.sum()).tolist()
    return {
        **base,
        "kind": "continuous",
        "decile_edges": [float(e) for e in edges],
        "expected": expected,
    }


def _null_psi_quantiles(
    values: np.ndarray, ref: dict, window: int, n_samples: int, rng
) -> Optional[dict]:
    """
    Null distribution of window-PSI: what PSI do ORDINARY contiguous
    training windows of ``window`` bars produce against the full training
    reference?  Market features are heavily autocorrelated, so short live
    windows always concentrate in a narrow band of the full distribution —
    raw PSI screams "drift" on perfectly normal data (the textbook
    0.10/0.25 cutoffs assume iid samples).  Calibrating against same-length
    training windows fixes the false alarm: only a live PSI beating the
    null's upper tail is evidence of a regime the training data never
    produced.
    """
    if len(values) < window * 3:
        return None
    starts = rng.integers(0, len(values) - window, size=n_samples)
    psis = []
    for s in starts:
        p = psi(values[s : s + window], ref)
        if p is not None:
            psis.append(p)
    if len(psis) < n_samples // 2:
        return None
    psis = np.array(psis)
    return {
        "window_bars": int(window),
        "p50": float(np.quantile(psis, 0.50)),
        "p90": float(np.quantile(psis, 0.90)),
        "p95": float(np.quantile(psis, 0.95)),
        "p99": float(np.quantile(psis, 0.99)),
    }


def compute_feature_stats(
    df: pl.DataFrame,
    feature_cols: List[str],
    null_window: int = 100,
    null_samples: int = 200,
    seed: int = 7,
) -> dict:
    """
    Full stats artifact from a (post-veto, post-clean) training frame.

    Returns a JSON-serializable dict:
        {schema_version, generated_at, n_rows, feature_cols,
         pooled: {feat: stats}, per_symbol: {sym: {feat: stats+null_psi}}}

    Per-symbol stats include ``null_psi`` — quantiles of the PSI that
    ordinary contiguous ``null_window``-bar training windows produce, the
    calibration reference the probe compares live PSI against. Frames must
    be time-ordered within each symbol (retrainer frames are).
    """
    rng = np.random.default_rng(seed)
    out = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": df.height,
        "feature_cols": list(feature_cols),
        "null_window": null_window,
        "pooled": {},
        "per_symbol": {},
    }
    for col in feature_cols:
        s = _column_stats(
            df[col].fill_nan(None).drop_nulls().to_numpy().astype(float)
        )
        if s is not None:
            out["pooled"][col] = s
    if "symbol" in df.columns:
        for sym in sorted(df["symbol"].unique().to_list()):
            sym_df = df.filter(pl.col("symbol") == sym)
            sym_stats = {}
            for col in feature_cols:
                vals = (
                    sym_df[col].fill_nan(None).drop_nulls().to_numpy().astype(float)
                )
                s = _column_stats(vals)
                if s is None:
                    continue
                null = _null_psi_quantiles(
                    vals, s, null_window, null_samples, rng
                )
                if null is not None:
                    s["null_psi"] = null
                sym_stats[col] = s
            out["per_symbol"][str(sym)] = sym_stats
    return out


def save_feature_stats(stats: dict, model_dir: Path | str) -> Path:
    """Atomic write of the sidecar (same pattern as threshold.json)."""
    import os

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / STATS_FILENAME
    tmp = model_dir / f"{STATS_FILENAME}.tmp"
    with open(tmp, "w") as fh:
        json.dump(stats, fh, indent=2)
    os.replace(tmp, path)
    logger.info(
        "[ATOMIC] Feature stats saved: %s (%d features, %d symbols, n=%d)",
        path, len(stats["pooled"]), len(stats["per_symbol"]), stats["n_rows"],
    )
    return path


def load_feature_stats(model_dir: Path | str) -> Optional[dict]:
    """Load the sidecar; None when absent/corrupt (probe degrades gracefully)."""
    path = Path(model_dir) / STATS_FILENAME
    if not path.exists():
        return None
    try:
        with open(path, "r") as fh:
            stats = json.load(fh)
        if "pooled" not in stats:
            raise ValueError("no 'pooled' block")
        return stats
    except Exception as exc:
        logger.warning("load_feature_stats: failed to read %s (%s)", path, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# PSI (probe side)
# ═══════════════════════════════════════════════════════════════════════════


def _psi_from_masses(actual: np.ndarray, expected: np.ndarray) -> float:
    a = np.clip(actual, _PSI_EPS, None)
    e = np.clip(expected, _PSI_EPS, None)
    a, e = a / a.sum(), e / e.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi(live_values: np.ndarray, train_stats: dict) -> Optional[float]:
    """
    PSI of a live sample against one feature's stored training stats.
    Returns None when the live sample is empty/degenerate.
    """
    live = live_values[np.isfinite(live_values)]
    if len(live) < 10:
        return None

    if train_stats["kind"] == "categorical":
        freqs = train_stats["freqs"]
        cats = sorted(float(k) for k in freqs)
        total = sum(freqs.values())
        expected = np.array([freqs[str(c)] / total for c in cats])
        # Live values outside the training categories get a synthetic
        # "unseen" bin with ~zero expected mass (maximal drift signal).
        actual = np.array(
            [np.mean(np.isclose(live, c)) for c in cats]
            + [np.mean([not any(np.isclose(v, cats)) for v in live])]
        )
        expected = np.append(expected, 0.0)
        return _psi_from_masses(actual, expected)

    edges = np.array(train_stats["decile_edges"])
    bins = np.concatenate([[-np.inf], edges, [np.inf]])
    hist, _ = np.histogram(live, bins=bins)
    actual = hist / hist.sum()
    expected = np.array(train_stats["expected"])
    return _psi_from_masses(actual, expected)


def psi_report(
    live_df: pl.DataFrame,
    stats: dict,
    symbol: Optional[str] = None,
) -> Dict[str, float]:
    """
    PSI per feature for a live frame. Uses per-symbol training stats when
    ``symbol`` is provided and present in the sidecar; pooled otherwise.
    """
    ref: dict = stats["pooled"]
    if symbol is not None and symbol in stats.get("per_symbol", {}):
        ref = stats["per_symbol"][symbol]
    out: Dict[str, float] = {}
    for col, tr in ref.items():
        if col not in live_df.columns:
            continue
        v = live_df[col].fill_nan(None).drop_nulls().to_numpy().astype(float)
        p = psi(v, tr)
        if p is not None:
            out[col] = p
    return out


def classify_psi(value: float) -> str:
    """Raw textbook thresholds — only meaningful for iid samples. Prefer
    classify_psi_calibrated when a null distribution is available."""
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "SEVERE"


def classify_psi_calibrated(value: float, train_stats: dict) -> str:
    """
    Drift call calibrated against the null distribution of same-length
    training windows (see _null_psi_quantiles). A live window is only
    'drifted' if it beats what ordinary training windows produce:
        > p99 of null → SEVERE (training never produced a window like this)
        > p95 of null → moderate
        otherwise     → normal-for-a-window
    Falls back to raw thresholds when the sidecar predates calibration.
    """
    null = train_stats.get("null_psi")
    if not null:
        return classify_psi(value)
    if value > null["p99"]:
        return "SEVERE"
    if value > null["p95"]:
        return "moderate"
    return "stable"


def drift_flags(
    live_psi: Dict[str, float], stats: dict, symbol: Optional[str] = None
) -> Dict[str, str]:
    """Calibrated label per feature for a live PSI report."""
    ref: dict = stats["pooled"]
    if symbol is not None and symbol in stats.get("per_symbol", {}):
        ref = stats["per_symbol"][symbol]
    return {
        f: classify_psi_calibrated(v, ref.get(f, {}))
        for f, v in live_psi.items()
    }
