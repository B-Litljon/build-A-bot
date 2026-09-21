#!/usr/bin/env python
"""
Is the promotion gate REACHABLE for this model? Sweep the Angel bar for
population instead of for EV and look for a point that satisfies both criteria.

Why this exists: on 2026-09-14 the retrainer's gate rejected every candidate
configuration tried — bracket geometry, τ pair, calibration method, Devil label,
Angel bar, basket composition. `validate_candidate` reports *why* a given run
failed, but not whether ANY setting could succeed. Those are different questions,
and the second is the one that decides between "keep tuning" and "the model has no
certifiable edge".

The gate wants two things at once:

    pooled OOS trades  >= backstop floor  (23 on the reference basket, scaled by
                                           the chop-veto rate)
    pooled PF 95% lb   >= PROFIT_FACTOR_THRESHOLD (1.2)

`_find_optimal_angel_threshold` picks the bar that MAXIMISES EV over a quantile grid
of the OOF scores (median → max) subject to `MIN_ANGEL_PROPOSALS`, which pushes it
to the thin top of the distribution — hence ~30 approved trades and a trade-count
rejection. This script rewrites that one choice: the bar becomes a fixed quantile of
the training OOF scores ("approve the top X%"), which dissolves the trade-count
problem — and then measures what happens to the evidence.

Reference result (cached M15 basket, 6 fiat pairs, 226,992 engineered rows):

    keep     bar  pooled  wins    win   pf_lb  fold3_lb  gate
    0.50  0.1654   55957 15092  0.270  0.7271    0.7157  False   EV < 0.0005 binds
    0.15  0.2194   16251  4349  0.268  0.7097    0.6458  False   EV < 0.0005 binds
    0.05  0.2431    3314   877  0.265  0.6739    0.5716  False   EV < 0.0005 binds
    0.02  0.2656     912   214  0.235  0.5370    0.4505  False   EV < 0.0005 binds

    0 of 6 points satisfy both criteria; best PF lower bound 0.7271 against 1.2.

So for that model the gate is unreachable at any bar, and the EV-maximising
calibration is what confines it to the only population that is not obviously
negative. Read a failure here as information about the model, not as a reason to
loosen a threshold.

Usage:
    PYTHONPATH=src:. python scripts/angel_bar_frontier.py
    PYTHONPATH=src:. python scripts/angel_bar_frontier.py --keep 0.5,0.2,0.1,0.02
    PYTHONPATH=src:. python scripts/angel_bar_frontier.py --granularity 240

Read-only: nothing under models/ is touched, nothing is promoted, and the frame
comes from the cached parquet basket so the result is deterministic and needs no
network. Exit 0 when some bar satisfies both criteria, 2 when none does (the
retrainer's "trained but rejected" convention).

Glossary:
    keep -- the fraction of training-frame bars the Angel is allowed to propose,
        i.e. 1 minus the quantile used as the bar. 0.50 = approve the top half.
    frontier -- the (trades, win rate, PF lower bound) curve traced by sweeping
        keep; the gate is reachable only if one point satisfies both criteria.
    htf pairing -- the higher-timeframe feature generator's bar size. MUST match
        run_oanda.py's _GRANULARITY_PROFILES for the bar size being evaluated, or
        the features differ from production. `get_asset_config`'s default assumes
        M1 ("5m"), so this script maps granularity explicitly and prints it.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import core.retrainer as R  # noqa: E402
from execution.risk_manager import RiskProfile  # noqa: E402

# Mirror run_oanda.py's _GRANULARITY_PROFILES (bar size -> higher timeframe).
HTF_FOR_GRANULARITY = {1: "5m", 5: "30m", 15: "1h"}
DEFAULT_SYMBOLS = ["AUD_JPY", "EUR_JPY", "GBP_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD"]
DEFAULT_KEEP = [0.50, 0.30, 0.15, 0.10, 0.05, 0.02]
CACHE_DIR = Path("analysis_cache/strategy_matrix")
MAX_HOLD = 45


def load_basket(symbols, granularity):
    """The cached parquet basket, one frame per symbol, concatenated."""
    parts = []
    for sym in symbols:
        path = CACHE_DIR / f"{sym}_M{granularity}.parquet"
        if not path.exists():
            raise SystemExit(
                f"missing cached bars: {path} — run src/analysis/build_strategy_matrix.py "
                f"first (this tool is cache-only on purpose: deterministic, no network)"
            )
        df = pl.read_parquet(path)
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(sym).alias("symbol"))
        parts.append(df)
    return pl.concat(parts, how="vertical_relaxed")


def quantile_calibration(keep):
    """
    Replace EV-maximisation with a fixed population quantile.

    The patched function has the same contract as
    ``retrainer._find_optimal_angel_threshold`` — (threshold, ev, n_proposals) —
    so it drops straight into ``validate_candidate``'s per-fold calibration.
    """
    def patched(angel_probs, macro_targets, sl_mult=R.SL_ATR_MULTIPLIER,
                tp_mult=R.TP_ATR_MULTIPLIER, min_proposals=None):
        bar = float(np.quantile(angel_probs, 1.0 - keep))
        mask = angel_probs >= bar
        win = float(macro_targets[mask].mean()) if mask.any() else 0.0
        return bar, win * (tp_mult / sl_mult) - (1.0 - win), int(mask.sum())
    return patched


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--granularity", type=int, default=15, choices=sorted(HTF_FOR_GRANULARITY),
                   help="bar size in minutes (default 15, the M15 soak)")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                   help="comma-separated basket (must exist in the cache)")
    p.add_argument("--keep", default=",".join(str(k) for k in DEFAULT_KEEP),
                   help="proposal fractions to sweep, e.g. 0.5,0.1,0.02")
    p.add_argument("--chop-floor", type=float, default=None,
                   help="override the trade-count backstop (default: the gate's own)")
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    keeps = [float(k) for k in args.keep.split(",")]
    htf = HTF_FOR_GRANULARITY[args.granularity]

    raw = load_basket(symbols, args.granularity)
    feats, feature_cols, chop = R.engineer_features_and_labels(
        raw, sl_mult=2.0, angel_mult=R._ANGEL_ATR_MULT_BY_CLASS["forex"],
        tp_mult=4.0, max_hold=MAX_HOLD, survival_bars=5, htf_timeframe=htf,
        risk_profile=RiskProfile.for_asset_class("forex"), alpha_table=None,
    )
    ap, dp = R.get_hyperparameters("forex")

    print(f"basket        : {len(symbols)} symbols, {raw.height:,} raw rows")
    print(f"engineered    : {feats.height:,} rows, {len(feature_cols)} features, "
          f"chop veto {chop:.3f}")
    print(f"bar size      : M{args.granularity} with htf={htf} "
          f"(mirrors run_oanda.py's _GRANULARITY_PROFILES)")
    print(f"gate criteria : pooled trades >= backstop "
          f"(chop-scaled) AND pooled PF 95% lb >= {R.PROFIT_FACTOR_THRESHOLD}\n")

    original = R._find_optimal_angel_threshold
    rows = []
    try:
        header = (f"{'keep':>6} {'bar':>8} {'pooled':>8} {'wins':>7} {'win':>7} "
                  f"{'base':>7} {'edge':>8} {'pf_lb':>8} {'fold3_lb':>9} "
                  f"{'folds':>22} {'gate':>6} {'binding rejection':>34}")
        print(header)
        for keep in keeps:
            R._find_optimal_angel_threshold = quantile_calibration(keep)
            rep, *_ = R.validate_candidate(
                feats, feature_cols, sl_mult=2.0, tp_mult=4.0, n_folds=3,
                angel_params=ap, devil_params=dp, chop_veto_rate=chop,
            )
            n, w = rep.pooled_oos_trades, rep.pooled_oos_wins
            folds = [f.devil_approved_trades for f in rep.fold_metrics]
            rejection = rep.rejection_reasons[0][:34] if rep.rejection_reasons else "-"
            # Edge over the bracket's own base rate (retrainer._macro_base_rate):
            # the metric that separates skill from a favourable regime. A PF lower
            # bound can be cleared by a zero-skill model when the base rate is high,
            # and the M15 arm's own base rate is ~0.25 against a 0.33 break-even.
            print(f"{keep:>6.2f} {rep.production_angel_threshold:>8.4f} {n:>8} {w:>7} "
                  f"{w / max(n, 1):>7.3f} {rep.pooled_base_rate:>7.3f} "
                  f"{rep.edge_over_random:>+8.4f} {rep.pooled_pf_lower_bound:>8.4f} "
                  f"{rep.fold3_pf_lower_bound:>9.4f} {str(folds):>22} "
                  f"{str(rep.gate_passed):>6} {rejection:>34}")
            rows.append((keep, n, w / max(n, 1), rep.pooled_pf_lower_bound,
                         rep.gate_passed, float(rep.effective_trade_floor),
                         float(rep.edge_over_random)))
    finally:
        R._find_optimal_angel_threshold = original

    # The gate's own backstop comes back on the report (it scales the baseline
    # trade floor by the surviving population), so use it rather than recomputing.
    floor = args.chop_floor
    if floor is None:
        floor = rows[-1][5] if rows else 0.0
    passing = [r for r in rows if r[1] >= floor and r[3] >= R.PROFIT_FACTOR_THRESHOLD]
    best = max(rows, key=lambda r: r[3]) if rows else None

    print(f"\ntrade-count floor used (from the gate's own report): {floor:.1f}")
    print(f"points satisfying BOTH criteria: {len(passing)} of {len(rows)}")
    if best:
        print(f"best PF lower bound on the frontier: {best[3]:.4f} "
              f"at keep={best[0]:.0%} ({best[1]} trades, win {best[2]:.3f})")
    finite_edge = [r for r in rows if np.isfinite(r[6])]
    if finite_edge:
        eb = max(finite_edge, key=lambda r: r[6])
        selective = [r for r in rows if np.isfinite(r[6]) and r[6] > 0]
        print(f"best edge over random: {eb[6]:+.4f} at keep={eb[0]:.0%} "
              f"({eb[1]} trades, win {eb[2]:.3f}); "
              f"{len(selective)} of {len(finite_edge)} points have positive edge")
    if passing:
        print("\nVERDICT: REACHABLE — the gate can be satisfied at some bar. Promote "
              "through the normal gate, not through these numbers.")
        return 0
    print("\nVERDICT: UNREACHABLE at any bar on this model and basket — the two "
          "criteria have no intersection. That is a statement about the model's "
          "evidence, not about its configuration; do NOT loosen a threshold to "
          "convert this into a pass.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
