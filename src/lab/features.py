"""Seed candidate feature family -- the template for a new lab feature.

``LabMicrostructureFeatures`` adds four short-horizon, per-symbol features on
top of the V3 stack. It needs the columns V3BaseFeatures produces (log_return,
natr_14), so a spec must order it after ``v3_base``:

    FeatureSpec(feature_sets=("v3_base", "microstructure"), ...)

Every candidate family follows this shape: a BaseFeatureGenerator whose
``generate`` is causal, computed per symbol with polars ``.over("symbol")``, and
a ``feature_cols`` attribute naming exactly the model-facing columns. Nothing
else in the repo changes to score it.

Glossary:
    LabMicrostructureFeatures -- the seed generator (see class docstring).
    ms_close_pos_20 -- where close sits in its trailing 20-bar high/low range
        (0 = at the low, 1 = at the high). Short-horizon position pressure.
    ms_ret_z_10 -- log_return's z-score over the trailing 10 bars. How stretched
        this bar's move is versus its own recent behaviour.
    ms_updown_vol_10 -- mean positive return divided by mean absolute return
        over 10 bars, in [0, 1]. Return asymmetry: > 0.5 means up-moves dominate.
    ms_autocorr_20 -- rolling lag-1 autocorrelation of log_return over 20 bars.
        Positive = trending persistence, negative = mean reversion.
"""

from __future__ import annotations

import polars as pl

from ml.core.interfaces import BaseFeatureGenerator
from lab.registry import register_feature

_EPS = 1e-12
_WINDOW = 20


class LabMicrostructureFeatures(BaseFeatureGenerator):
    """
    Short-horizon bar-shape / return-autocorrelation / range-position features.

    Must run after a generator that provides ``log_return`` (V3BaseFeatures).
    All rolling windows are per symbol and end at the current bar -- no shift
    beyond the lag-1 term, so there is no lookahead.
    """

    feature_cols = (
        "ms_close_pos_20",
        "ms_ret_z_10",
        "ms_updown_vol_10",
        "ms_autocorr_20",
    )

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [c for c in ("log_return", "close", "high", "low") if c not in df.columns]
        if missing:
            raise ValueError(
                "LabMicrostructureFeatures needs %s — order it after V3BaseFeatures "
                "(feature_sets=('v3_base', 'microstructure'))." % missing
            )

        close = pl.col("close")
        ret = pl.col("log_return")

        df = df.with_columns(
            (
                (close - pl.col("low").rolling_min(_WINDOW).over("symbol"))
                / (
                    pl.col("high").rolling_max(_WINDOW).over("symbol")
                    - pl.col("low").rolling_min(_WINDOW).over("symbol")
                    + _EPS
                )
            ).alias("ms_close_pos_20"),
            (
                (ret - ret.rolling_mean(10).over("symbol"))
                / (ret.rolling_std(10).over("symbol") + _EPS)
            ).alias("ms_ret_z_10"),
            (
                pl.when(ret > 0)
                .then(ret)
                .otherwise(0.0)
                .rolling_mean(10)
                .over("symbol")
                / (ret.abs().rolling_mean(10).over("symbol") + _EPS)
            ).alias("ms_updown_vol_10"),
        )
        return df.with_columns(
            pl.rolling_corr(
                ret, ret.shift(1), window_size=_WINDOW, min_samples=_WINDOW
            )
            .over("symbol")
            .alias("ms_autocorr_20")
        )


register_feature(
    "microstructure",
    description=(
        "Seed candidate family: short-horizon range position, return stretch, "
        "up/down asymmetry, and lag-1 autocorrelation (needs v3_base first)."
    ),
)(LabMicrostructureFeatures)
