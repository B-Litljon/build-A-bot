"""
Diagnostic ONLY — not a gate run. Train the 5-year-config final artifact on
the post-holdout remainder and score it on the untouched holdout, so the
report can quantify the overfitting gap for a config the fold gate rejected
(pooled-trade floor: 221 < 232). No artifact is saved; nothing is promoted.

Replicates main()'s pipeline exactly, minus the fold gate:
  fetch (pinned window) -> _split_holdout -> engineer remainder ->
  refit_models (final artifact) -> engineer holdout -> _evaluate_holdout
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

    features_df, feature_cols, chop_veto_rate = R.engineer_features_and_labels(
        remainder_raw,
        sl_mult=asset_config["sl_mult"],
        angel_mult=asset_config["angel_mult"],
        tp_mult=asset_config["tp_mult"],
        max_hold=asset_config["max_hold"],
        survival_bars=asset_config["survival_bars"],
        htf_timeframe=asset_config["htf_timeframe"],
        risk_profile=R.RiskProfile.for_asset_class(asset_config["asset_class"]),
    )

    angel, devil, angel_feats, devil_feats = R.refit_models(
        features_df, feature_cols,
        angel_params=angel_params, devil_params=devil_params,
    )

    holdout_features, _, _ = R.engineer_features_and_labels(
        holdout_raw,
        sl_mult=asset_config["sl_mult"],
        angel_mult=asset_config["angel_mult"],
        tp_mult=asset_config["tp_mult"],
        max_hold=asset_config["max_hold"],
        survival_bars=asset_config["survival_bars"],
        htf_timeframe=asset_config["htf_timeframe"],
        risk_profile=R.RiskProfile.for_asset_class(asset_config["asset_class"]),
    )

    scores = R._evaluate_holdout(
        holdout_features,
        angel,
        devil,
        angel_feats,
        devil_feats,
        FOLD3_THRESHOLD,
        sl_mult=asset_config["sl_mult"],
        tp_mult=asset_config["tp_mult"],
    )

    floor = R.BASELINE_POOLED_OOS_TRADES * R.HOLDOUT_FRAC * (1.0 - chop_veto_rate)
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 5YR ARTIFACT ON HOLDOUT (would-be artifact; gate rejected it)")
    logger.info(
        "Brier=%.4f | EV=%.6f | WR=%.4f | PF=%.4f | Trades=%d (floor=%.0f) | "
        "Angel proposed=%d",
        scores["brier_score"], scores["expected_value"], scores["win_rate"],
        scores["profit_factor"], scores["trades"], floor,
        scores["angel_proposed_trades"],
    )
    logger.info("Bars: Brier<=%.2f EV>=%.4f PF>=%.2f",
                R.BRIER_THRESHOLD, R.EV_THRESHOLD, R.PROFIT_FACTOR_THRESHOLD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
