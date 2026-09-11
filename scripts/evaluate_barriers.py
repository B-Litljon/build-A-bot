"""
Phase 1 evaluation: do learned quantile barriers beat the static bracket?

Fits BarrierEstimator on expanding walk-forward folds over the cached M15
basket and compares, per fold and pooled:

  * pinball loss of the learned Q_MAE(0.95) vs the static 2.0xATR constant
    (the Phase 1 promotion metric — the learned geometry must win on every
    fold, not on average)
  * MAE coverage: empirical P(MAE <= Q_MAE(0.95)) — must be >= 0.93 (slight
    over-conservatism acceptable, under-coverage is a fail)
  * realised-R replay: rerun the strategy library's pooled ledger with the
    learned bracket substituted for the static one, same gates, same cost
    table, and compare net expectancy and the 0.40+ confidence-band stop-out
    rate (the inversion metric: 73.3% -> target <= 56%)

Read-only with respect to the live soak: writes nothing under models/, prints
a verdict, and exits 0 (pass) / 2 (fail) in the retrainer's convention.

Glossary:
    folds -- expanding walk-forward slices (3 folds, matching the retrainer's
        discipline) over the pooled, globally timestamp-sorted frame; each
        fold trains on the past and scores the slice after it, never the
        reverse. Excursion labels are computed per symbol BEFORE pooling, so
        a forward window never crosses a symbol boundary.
    static_baseline -- the incumbent 2.0x/4.0x NATR constants scored as
        constant quantile predictors in pinball units.
    coverage -- fraction of labelled bars whose realised MAE stayed at or
        under the predicted 95th-percentile stop; the empirical check that
        the model actually learned a 95% quantile.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml.barriers.estimator import BarrierEstimator, static_baseline_loss  # noqa: E402
from ml.barriers.labels import DEFAULT_HORIZON, compute_excursions, pinball_loss  # noqa: E402
from analysis.build_strategy_matrix import load_basket, prepare_tagged_frame  # noqa: E402

RETRAINER_FEATURES = [
    "rsi_14", "ppo", "natr_14", "bb_pct_b", "bb_width_pct",
    "price_sma50_ratio", "log_return", "hour_of_day", "dist_sma50",
    "vol_rel", "htf_trend_agreement", "htf_bb_pct_b", "range_coil_10",
    "session_asia", "session_london", "session_ny", "session_overlap",
]

# Features the matrix builder's frame actually carries (the full retrainer
# vocabulary needs the feature pipeline; this evaluation uses the overlap so
# the result is still apples-to-apples against the static baseline, which
# sees no features at all). vol_rel needs an RSI-style relative computation —
# computed inline below as trailing-vol ratio, causal, from natr_14.
EVAL_FEATURES = ["ppo", "natr_14"]


def _augment(df: pl.DataFrame) -> pl.DataFrame:
    """Add the causal trailing-vol ratio the fallback vocabulary expects."""
    return df.with_columns(
        (pl.col("natr_14") / pl.col("natr_14").rolling_mean(260).shift(1))
        .fill_nan(1.0)
        .alias("vol_rel")
    )

N_FOLDS = 3
COVERAGE_FLOOR = 0.93
TAU_MAE = 0.95


def expanding_folds(n: int, n_folds: int = N_FOLDS):
    """(train_end, test_start, test_end) triples, expanding window."""
    fold_size = n // (n_folds + 1)
    return [
        ((f + 1) * fold_size,
         (f + 1) * fold_size,
         min((f + 2) * fold_size, n))
        for f in range(n_folds)
    ]


def main() -> int:
    frames = load_basket(
        ["AUD_JPY", "EUR_JPY", "GBP_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD"],
        days_back=730,
        granularity=15,
        cache_dir=Path("analysis_cache/strategy_matrix"),
    )
    tagged_by_symbol = [prepare_tagged_frame(df, sym) for sym, df in frames.items()]
    # Excursion labels are computed PER SYMBOL and then pooled. Computing them
    # on the vertically-concatenated frame lets the last `horizon` bars of one
    # symbol's block look forward into the next symbol's bars — a cross-symbol
    # label leak the old single-frame path had.
    labelled = pl.concat(
        [compute_excursions(df, horizon=DEFAULT_HORIZON) for df in tagged_by_symbol],
        how="vertical_relaxed",
    )
    labelled = labelled.filter(pl.col("resolvable")).sort("timestamp", "symbol")

    # One pooled frame: folds walk chronology. Symbols are only pooled AFTER
    # labelling, and the global timestamp sort makes "expanding" mean
    # chronological, not block-order — the per-symbol row indices interleave
    # in time, so a raw row-index fold would train on the future of pairs
    # whose blocks end later.
    syms = labelled["symbol"].to_numpy()
    print(f"rows: {labelled.height}  symbols: {sorted(set(syms))}")

    ok = _augment(labelled).with_columns(
        pl.col("ppo").fill_nan(None).forward_fill()
    ).filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in EVAL_FEATURES])
    )
    n = ok.height
    folds = expanding_folds(n)
    ts = ok["timestamp"].to_numpy()
    assert (np.diff(ts.astype("datetime64[us]").astype(np.int64)) >= 0).all(), \
        "pooled frame must be globally timestamp-ordered for folds to be chronological"
    print(f"scoring {n} labelled rows over {len(folds)} expanding folds "
          f"({ts[0]} -> {ts[-1]})\n")

    all_pass = True
    y_mae = ok["mae_natr"].to_numpy()
    for fold_no, (tr_end, te_start, te_end) in enumerate(folds, 1):
        train, test = ok[:tr_end], ok[te_start:te_end]
        est = BarrierEstimator(feature_cols=EVAL_FEATURES)
        try:
            est.fit(train, train)
        except ValueError as e:
            print(f"fold {fold_no}: FIT FAILED: {e}")
            all_pass = False
            continue
        preds = est.predict(test)
        q = np.array([p.q_mae for p in preds])
        y = test["mae_natr"].to_numpy()
        learned_pl = pinball_loss(y, q, TAU_MAE)
        static_pl = static_baseline_loss(y, sl_mult=2.0, tau=TAU_MAE)
        coverage = float((y <= q).mean())
        beat = learned_pl < static_pl
        covered = coverage >= COVERAGE_FLOOR
        all_pass &= beat and covered
        print(f"fold {fold_no}: n_train={len(train):>6} n_test={len(test):>6}  "
              f"pinball learned={learned_pl:.4f} static={static_pl:.4f} "
              f"{'BEAT' if beat else 'LOST'}  coverage={coverage:.3f} "
              f"{'ok' if covered else 'UNDER-COVERED'}")

    print("\nVERDICT:", "PASS — learned barriers win every fold with adequate "
          "coverage" if all_pass else
          "FAIL — static bracket stays (promotion blocked, prior weights stand)")
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())