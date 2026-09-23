"""build_frame -- one spec + raw bars -> the labelled, vetoed training frame.

The pipeline order is the retrainer's, verbatim and for the same reasons:

    1. per-symbol feature generation (generator list from the registry),
    2. labels + excursion targets + the chop/behavior vetoes, applied to the
       still-contiguous price path (apply_labels_and_veto),
    3. null/inf cleanup on the model-facing columns.

Feature generation and label generation must not be separated by a cleaning
step: the bracket walk in step 2 uses index-based lookahead, so dropping rows
first would change the labels of the survivors. That is why this module calls
apply_labels_and_veto (the 2026-09-21 extraction) rather than
FeaturePipeline.run().

Glossary:
    FrameResult -- the whole artefact: the frame, the model-facing feature
        column names, the chop/behavior veto drop rate, the purge count, the
        resolved alpha table, and the spec's content hash. Everything a gate or
        backtest run needs, so no consumer re-derives it.
    build_frame -- also purges the unresolvable tail: the last ``max_hold``
        bars per symbol whose bracket walk runs off the end of the fetched
        window and reads "timeout -> loss" regardless of the true outcome.
        Production does this in main() Phase 3a against the holdout boundary;
        the lab does it against the end of the data, for the same reason
        (systematically wrong labels must not train the fold models).
    stack_bars -- {symbol: bars} -> one frame with a symbol column, sorted
        ["symbol", "timestamp"], matching fetch_training_data's layout.
    build_frame -- the entry point; also verifies that every declared feature
        column is actually present (a typo'd registry column must fail loudly,
        not silently shrink the feature set).
    angel_mult resolution -- None in the spec means the environment's
        RETRAIN_ANGEL_ATR_MULT if set, else the asset class's production
        multiple (forex: 1.0). This mirrors get_asset_config so a control spec
        reproduces a production run's labels exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import polars as pl

from lab.data import resolve_alpha_table
from lab.registry import feature_columns, get_generators


@dataclass
class FrameResult:
    """A built feature/label frame plus everything needed to score it."""

    spec_name: str
    content_hash: str
    df: pl.DataFrame
    feature_cols: Tuple[str, ...]
    chop_veto_rate: float
    alpha_table: Optional[dict]
    purged_tail_rows: int = 0

    @property
    def n_rows(self) -> int:
        return self.df.height

    def features(self) -> pl.DataFrame:
        """The model-facing columns only, in declared order."""
        return self.df.select(list(self.feature_cols))


def stack_bars(bars: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Concat per-symbol frames into the retrainer's stacked layout."""
    parts: List[pl.DataFrame] = []
    for sym, frame in bars.items():
        if frame.is_empty():
            continue
        if "symbol" in frame.columns:
            frame = frame.with_columns(pl.lit(sym).alias("symbol"))
        else:
            frame = frame.with_columns(pl.lit(sym).alias("symbol"))
        parts.append(frame)
    if not parts:
        raise ValueError("stack_bars received no non-empty frames")
    return pl.concat(parts, how="vertical_relaxed").sort(["symbol", "timestamp"])


def _resolve_angel_mult(spec) -> Optional[float]:
    """Explicit spec value, else the env/class default get_asset_config uses."""
    import os

    from core.retrainer._common import _ANGEL_ATR_MULT_BY_CLASS

    if spec.label.angel_mult is not None:
        return float(spec.label.angel_mult)
    env = os.getenv("RETRAIN_ANGEL_ATR_MULT", "").strip()
    if env:
        return float(env)
    return float(_ANGEL_ATR_MULT_BY_CLASS.get(spec.asset_class, 0.5))


def build_frame(spec, bars: Dict[str, pl.DataFrame]) -> FrameResult:
    """
    Build the labelled frame for a spec over already-loaded bars.

    The cost table comes from the spec alone — never a caller override — because
    the spec's content hash is the frame-cache key and must describe every input
    that changed the frame.
    """
    from core.retrainer._features import apply_labels_and_veto
    from core.retrainer._gate import _purge_boundary_tail, _tail_cutoff_by_symbol
    from execution.risk_manager import RiskProfile

    table = resolve_alpha_table(spec)
    stacked = stack_bars(bars)

    df = stacked
    for gen in get_generators(spec, table):
        df = gen.generate(df)

    cols = feature_columns(spec, table)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"spec {spec.name!r} declares feature columns that no generator "
            f"produced: {missing} (check registry column declarations and the "
            "family order in feature_sets)"
        )

    profile = RiskProfile.for_asset_class(spec.asset_class)
    df, chop_veto_rate = apply_labels_and_veto(
        df,
        cols,
        sl_mult=spec.geometry.sl_mult,
        tp_mult=spec.geometry.tp_mult,
        max_hold=spec.geometry.max_hold,
        survival_bars=spec.label.survival_bars,
        angel_mult=_resolve_angel_mult(spec),
        risk_profile=profile,
        alpha_table=table,
    )

    # Phase-3a mirror: cutoffs from the RAW stacked series (the walk's true
    # domain, before the veto removed rows), applied to the engineered frame.
    cutoffs = _tail_cutoff_by_symbol(stacked, spec.geometry.max_hold)
    df, purged_tail_rows = _purge_boundary_tail(df, cutoffs)

    return FrameResult(
        spec_name=spec.name,
        content_hash=spec.content_hash(),
        df=df,
        feature_cols=tuple(cols),
        chop_veto_rate=chop_veto_rate,
        alpha_table=table,
        purged_tail_rows=purged_tail_rows,
    )
