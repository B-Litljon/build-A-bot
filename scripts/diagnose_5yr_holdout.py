"""
Diagnostic ONLY — not a gate run. Train the 5-year-config final artifact on
the post-holdout remainder and score it on the untouched holdout, so the
report can quantify the overfitting gap for a config the fold gate rejected
(pooled-trade floor: 221 < 232). No artifact is saved; nothing is promoted.

Replicates main()'s pipeline exactly, minus the fold gate:
  fetch (pinned window) -> _split_holdout -> engineer remainder ->
  refit_models (final artifact) -> _score_artifact_holdout (tail purge + score)
with the Fold-3 swept threshold from logs/holdout_5yr.log (0.10).

Env (set by the wrapper): DATA_SOURCE=oanda, RETRAIN_TIMEFRAME_MINUTES=15,
RETRAIN_DAYS_BACK=1825, RETRAIN_END_DATE=2026-08-09.

Glossary:
    FOLD3_THRESHOLD -- the Devil threshold swept on Fold 2/3 of the rejected
        5-year gate run (logs/holdout_5yr.log: "Optimal Devil threshold (from
        Fold 3): 0.1000"). Diagnostic scoring uses it frozen, as the gate
        would have.
"""

import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from core import retrainer as R  # noqa: E402
from data.factory import get_market_provider  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnose_5yr")

FOLD3_THRESHOLD = 0.10


def main() -> int:
    asset_config = R.get_asset_config("oanda")
    provider = get_market_provider()
    angel_params, devil_params = R.get_hyperparameters(asset_config["asset_class"])

    raw = R.fetch_training_data(
        provider=provider,
        symbols=asset_config["tickers"],
        days_back=R.DAYS_BACK,
        timeframe_minutes=asset_config["timeframe_minutes"],
    )

    remainder_raw, holdout_raw, holdout_range = R._split_holdout(raw, R.HOLDOUT_FRAC)
    logger.info(
        "Holdout: %s rows, %s -> %s",
        holdout_raw.height, holdout_range[0], holdout_range[1],
    )

    features_df, feature_cols, _ = R.engineer_features_and_labels(
        remainder_raw,
        sl_mult=asset_config["sl_mult"],
        angel_mult=asset_config["angel_mult"],
        tp_mult=asset_config["tp_mult"],
        max_hold=asset_config["max_hold"],
        survival_bars=asset_config["survival_bars"],
        htf_timeframe=asset_config["htf_timeframe"],
        risk_profile=R.RiskProfile.for_asset_class(asset_config["asset_class"]),
    )

    # Mirror main()'s Phase 3a: the last max_hold bars per symbol carry
    # labels the walk could not resolve, and the real pipeline drops them
    # before training. A diagnostic that keeps them trains on rows the
    # artifact never saw.
    tail_cutoffs = R._tail_cutoff_by_symbol(remainder_raw, asset_config["max_hold"])
    features_df, n_purged = R._purge_boundary_tail(features_df, tail_cutoffs)
    logger.info("BOUNDARY PURGE: dropped %d training rows (unresolvable tail)", n_purged)

    angel, devil, angel_feats, devil_feats = R.refit_models(
        features_df, feature_cols,
        angel_params=angel_params, devil_params=devil_params,
    )

    scores, n_purged = R._score_artifact_holdout(
        holdout_raw,
        angel,
        devil,
        angel_feats,
        devil_feats,
        FOLD3_THRESHOLD,
        asset_config,
    )
    pf_lb = R._holdout_pf_lower_bound(
        scores["wins"], scores["trades"],
        asset_config["sl_mult"], asset_config["tp_mult"],
    )
    passed, reasons = R._holdout_verdict(
        scores, sl_mult=asset_config["sl_mult"], tp_mult=asset_config["tp_mult"]
    )

    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 5YR ARTIFACT ON HOLDOUT (would-be artifact; gate rejected it)")
    logger.info(
        "Brier=%.4f | EV=%.6f | WR=%.4f | PF=%.4f [%.0f%% lower bound %.4f] | "
        "Trades=%d (wins=%d, tail purged=%d) | Angel proposed=%d",
        scores["brier_score"], scores["expected_value"], scores["win_rate"],
        scores["profit_factor"], 100.0 * R.HOLDOUT_PF_CONFIDENCE, pf_lb,
        scores["trades"], scores["wins"], n_purged,
        scores["angel_proposed_trades"],
    )
    logger.info(
        "Stable-gate verdict: %s%s",
        "PASS" if passed else "FAIL",
        "" if passed else " — " + "; ".join(reasons),
    )
    logger.info("Bars: Brier<=%.2f EV>=%.4f PF lower bound>=%.2f",
                R.BRIER_THRESHOLD, R.EV_THRESHOLD, R.PROFIT_FACTOR_THRESHOLD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
