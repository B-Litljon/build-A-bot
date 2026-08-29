"""
Capacity sweep: does a smaller model generalize better?

The served model has 200 trees x 63 leaves -- up to 12,600 decision regions to
describe a few hundred real opportunities. The 2026-08-24 diagnosis showed that
shrinking it collapses the in-sample/out-of-sample gap from 0.108 to 0.014
while out-of-sample skill slightly IMPROVES. This sweeps that properly.

Every variant is trained on the SAME remainder and scored on the SAME untouched
holdout, at each of several PINNED windows -- because this project has been
misled four times by comparing across windows that moved underneath it.

Read-only: trains in memory, writes no model artifact, never touches the live
bot. Results land in logs/capacity_sweep.json for plotting.

Glossary:
    VARIANTS -- (label, hyperparameter overrides, features to drop). The
        baseline reproduces the shipped configuration exactly.
    DEAD_FEATURES -- the five the diagnosis found are used heavily in-sample
        but contribute nothing out-of-sample (two score NEGATIVE, i.e.
        shuffling them helps). NOT the time features -- hour_of_day measured
        as the single most valuable feature on unseen data.
    auc_in / auc_oos -- Angel AUC on the data it trained on vs the holdout.
        The GAP between them is the overfitting measure; auc_oos alone is the
        only one that predicts live behaviour.
    pf_lower_bound -- the honest gate's criterion: Clopper-Pearson lower bound
        on holdout profit factor. A thin sample widens the interval and fails
        on its own, which is why a PF of 40 on 21 trades does not pass.
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
logging.basicConfig(level=logging.ERROR, format="%(message)s")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from core import retrainer as R  # noqa: E402
from data.factory import get_market_provider  # noqa: E402
from execution.risk_manager import RiskProfile  # noqa: E402

DEAD_FEATURES = [
    "htf_vol_rel",
    "htf_rsi_14",
    "bar_body_pct",
    "bar_upper_wick_pct",
    "bar_lower_wick_pct",
]

VARIANTS = [
    ("shipped (200x63)",        {},                                                      []),
    ("150x31",                  {"n_estimators": 150, "num_leaves": 31,
                                 "min_child_samples": 40},                               []),
    ("100x15",                  {"n_estimators": 100, "num_leaves": 15,
                                 "min_child_samples": 80},                               []),
    ("100x15 - dead feats",     {"n_estimators": 100, "num_leaves": 15,
                                 "min_child_samples": 80},                    DEAD_FEATURES),
    ("shipped - dead feats",    {},                                           DEAD_FEATURES),
]

PINS = os.getenv("SWEEP_PINS", "2026-08-09,2026-08-16,2026-08-23").split(",")


def sweep_one_window(pin: str) -> list:
    """Fetch once, split once, then train every variant on identical data."""
    os.environ["RETRAIN_END_DATE"] = pin
    cfg = R.get_asset_config("oanda")
    sl, tp = cfg["sl_mult"], cfg["tp_mult"]

    raw = R.fetch_training_data(
        get_market_provider(), symbols=cfg["tickers"],
        days_back=R.DAYS_BACK, timeframe_minutes=cfg["timeframe_minutes"],
    )
    rem_raw, hold_raw, rng = R._split_holdout(raw, R.HOLDOUT_FRAC)

    def engineer(df):
        f, cols, _ = R.engineer_features_and_labels(
            df, sl_mult=sl, angel_mult=cfg["angel_mult"], tp_mult=tp,
            max_hold=cfg["max_hold"], survival_bars=cfg["survival_bars"],
            htf_timeframe=cfg["htf_timeframe"],
            risk_profile=RiskProfile.for_asset_class(cfg["asset_class"]),
            alpha_table=None,
        )
        return f, cols

    rem, cols = engineer(rem_raw)
    hold, _ = engineer(hold_raw)
    rem, _purged = R._purge_boundary_tail(
        rem, R._tail_cutoff_by_symbol(rem_raw, cfg["max_hold"])
    )
    ap0, dp0 = R.get_hyperparameters(cfg["asset_class"])

    out = []
    for label, overrides, drop in VARIANTS:
        feats = [c for c in cols if c not in drop]
        a_params, d_params = {**ap0, **overrides}, {**dp0, **overrides}
        angel, devil, a_feats, d_feats = R.refit_models(
            rem, feats, angel_params=a_params, devil_params=d_params
        )
        auc_in = roc_auc_score(
            rem["angel_target"].to_numpy(),
            angel.predict_proba(rem.select(a_feats).to_pandas())[:, 1],
        )
        auc_oos = roc_auc_score(
            hold["angel_target"].to_numpy(),
            angel.predict_proba(hold.select(a_feats).to_pandas())[:, 1],
        )
        sc = R._evaluate_holdout(
            hold, angel, devil, a_feats, d_feats, 0.66, sl_mult=sl, tp_mult=tp
        )
        lb = (
            R._holdout_pf_lower_bound(sc["wins"], sc["trades"], sl, tp)
            if sc["trades"] else float("nan")
        )
        passed, _reasons = R._holdout_verdict(sc, sl, tp)
        rec = {
            "pin": pin, "variant": label, "n_features": len(feats),
            "auc_in": round(float(auc_in), 4),
            "auc_oos": round(float(auc_oos), 4),
            "gap": round(float(auc_in - auc_oos), 4),
            "trades": int(sc["trades"]), "wins": int(sc["wins"]),
            "win_rate": round(float(sc["win_rate"]), 4),
            "profit_factor": round(float(sc["profit_factor"]), 4),
            "pf_lower_bound": round(float(lb), 4),
            "brier": round(float(sc["brier_score"]), 4),
            "holdout_passed": bool(passed),
        }
        out.append(rec)
        print(
            f"  {label:<24}{rec['auc_in']:>8.3f}{rec['auc_oos']:>9.3f}"
            f"{rec['gap']:>8.3f}{rec['trades']:>8}{rec['win_rate']*100:>8.1f}"
            f"{rec['pf_lower_bound']:>9.3f}   {'PASS' if passed else 'fail'}"
        )
    return out


def main() -> int:
    rows = []
    for pin in PINS:
        print(f"\n=== window ending {pin} ===")
        print(f"  {'variant':<24}{'AUC in':>8}{'AUC oos':>9}{'gap':>8}"
              f"{'trades':>8}{'win%':>8}{'PF lb':>9}   gate")
        print("  " + "-" * 82)
        rows.extend(sweep_one_window(pin))

    Path("logs").mkdir(exist_ok=True)
    Path("logs/capacity_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote logs/capacity_sweep.json ({len(rows)} rows)")

    print("\n=== AVERAGED ACROSS WINDOWS (what actually matters) ===")
    print(f"{'variant':<24}{'AUC oos':>9}{'gap':>8}{'trades':>8}{'PF lb':>9}{'passes':>8}")
    print("-" * 66)
    for label, _o, _d in VARIANTS:
        sel = [r for r in rows if r["variant"] == label]
        if not sel:
            continue
        print(
            f"{label:<24}{np.mean([r['auc_oos'] for r in sel]):>9.3f}"
            f"{np.mean([r['gap'] for r in sel]):>8.3f}"
            f"{np.mean([r['trades'] for r in sel]):>8.0f}"
            f"{np.mean([r['pf_lower_bound'] for r in sel]):>9.3f}"
            f"{sum(r['holdout_passed'] for r in sel):>5}/{len(sel)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
