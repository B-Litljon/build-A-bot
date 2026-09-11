"""
Capture the OOS behavior ledger for a capacity variant and score it.

Drives the retrainer's walk-forward the same way main() does — same fetch,
same holdout carve, same engineering, same boundary purge — but tags the
engineered frame with per-symbol ``behavior_label`` first and passes an
``oos_ledger`` into validate_candidate. The ledger is therefore the exact
Devil-approved out-of-sample trade population the fold gate judged (BEFORE
the untradeable-metals scoring exclusion, matching the 2026-08-23 recon's
population), scored per behavior tag by analysis/behavior_matrix.py.

Answers: does the trim architecture still bleed on ``trend_high``, or did the
2026-08-29 gate rebuild (calibrated Angel bar + living Devil) change which
behaviors the system loses money in?

Env: standard retrain vars (DATA_SOURCE, RETRAIN_TIMEFRAME_MINUTES,
RETRAIN_DAYS_BACK, RETRAIN_END_DATE, ...) plus the GATE_* capacity overrides
from run_gate_capacity_variant.py, plus:
    LEDGER_CANDIDATE -- name attached to the scored cells.
    LEDGER_TAG       -- output suffix: writes logs/behavior_ledger_<tag>.parquet
                        and logs/behavior_matrix_<tag>.csv. Default "unnamed".

Read-only with respect to models/: nothing here trains a final artifact or
writes a model directory.

Glossary:
    oos_ledger -- validate_candidate's opt-in per-trade capture: one row per
        Devil-approved validation trade with macro_win, both stage probs, and
        the carried behavior_label. Never collected in production paths.
    tag_frame -- behavior_matrix.py's causal per-symbol behavior tagger; must
        run BEFORE the walk-forward so the label rides into the ledger.
    GATE_* -- capacity env overrides, same contract as
        run_gate_capacity_variant.py (shared value, GATE_ANGEL_/GATE_DEVIL_
        per-stage precedence).
"""

import logging
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("core.retrainer").setLevel(logging.WARNING)

logger = logging.getLogger("capture_behavior_ledger")

import polars as pl  # noqa: E402

from analysis.behavior_matrix import Candidate, score_ledger, tag_frame, to_frame  # noqa: E402
from core import retrainer as R  # noqa: E402
from data.factory import get_market_provider  # noqa: E402
from execution.risk_manager import RiskProfile  # noqa: E402

# Same GATE_* contract as scripts/run_gate_capacity_variant.py.
_ENV_KEYS = {
    "GATE_N_ESTIMATORS": "n_estimators",
    "GATE_NUM_LEAVES": "num_leaves",
    "GATE_MIN_CHILD": "min_child_samples",
    "GATE_MAX_DEPTH": "max_depth",
}


def _overrides(stage: str) -> dict:
    out = {}
    for env_name, param in _ENV_KEYS.items():
        shared = os.environ.get(env_name, "").strip()
        specific = os.environ.get(f"GATE_{stage}_{env_name[5:]}", "").strip()
        value = specific or shared
        if value:
            out[param] = int(value)
    return out


_orig = R.get_hyperparameters


def _patched(asset_class):
    angel, devil = _orig(asset_class)
    angel.update(_overrides("ANGEL"))
    devil.update(_overrides("DEVIL"))
    return angel, devil


R.get_hyperparameters = _patched


def main() -> int:
    tag = os.environ.get("LEDGER_TAG", "unnamed").strip() or "unnamed"
    name = os.environ.get("LEDGER_CANDIDATE", tag).strip() or tag
    cfg = R.get_asset_config(os.getenv("DATA_SOURCE", "alpaca").strip().lower())

    logger.info("=== fetch (%s, %sd, %sm tf) ===", name, R.DAYS_BACK, cfg["timeframe_minutes"])
    raw = R.fetch_training_data(
        get_market_provider(),
        symbols=cfg["tickers"],
        days_back=R.DAYS_BACK,
        timeframe_minutes=cfg["timeframe_minutes"],
    )

    # Carve the holdout exactly like main(): the fold walk-forward the ledger
    # captures must run on the remainder only — the holdout stays untouched.
    rem_raw, _hold_raw, _rng = R._split_holdout(raw, R.HOLDOUT_FRAC)
    feats, cols, chop_veto = R.engineer_features_and_labels(
        rem_raw,
        sl_mult=cfg["sl_mult"],
        angel_mult=cfg["angel_mult"],
        tp_mult=cfg["tp_mult"],
        max_hold=cfg["max_hold"],
        survival_bars=cfg["survival_bars"],
        htf_timeframe=cfg["htf_timeframe"],
        risk_profile=RiskProfile.for_asset_class(cfg["asset_class"]),
        alpha_table=None,
    )
    feats, _ = R._purge_boundary_tail(
        feats, R._tail_cutoff_by_symbol(rem_raw, cfg["max_hold"])
    )
    # Tag BEFORE the walk-forward so the label rides into the ledger.
    feats = tag_frame(feats)

    ap, dp = R.get_hyperparameters(cfg["asset_class"])
    ledger = []
    logger.info("=== walk-forward with OOS ledger capture ===")
    _report, *_rest = R.validate_candidate(
        feats,
        cols,
        sl_mult=cfg["sl_mult"],
        tp_mult=cfg["tp_mult"],
        n_folds=3,
        angel_params=ap,
        devil_params=dp,
        chop_veto_rate=chop_veto,
        oos_ledger=ledger,
    )

    if not ledger:
        logger.warning("ledger is empty (no Devil-approved trades in any fold)")
        return 1
    lf = pl.concat(ledger)
    candidate = Candidate(
        name=name,
        sl_mult=cfg["sl_mult"],
        tp_mult=cfg["tp_mult"],
        lookback_days=R.DAYS_BACK,
    )
    cells = score_ledger(lf, candidate)
    frame = to_frame(cells)

    lf.write_parquet(f"logs/behavior_ledger_{tag}.parquet")
    frame.write_csv(f"logs/behavior_matrix_{tag}.csv")
    logger.info("wrote logs/behavior_ledger_%s.parquet (%d trades)", tag, lf.height)
    logger.info("wrote logs/behavior_matrix_%s.csv", tag)
    print(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
