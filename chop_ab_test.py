#!/usr/bin/env python
"""
Chop-filter A/B harness.

Fetches the forex basket ONCE, caches it to parquet, then runs the walk-forward
validation gate twice on the byte-identical dataset — toggling only
RISK_CHOP_FILTER_ENABLED — so any Profit-Factor difference is attributable solely
to the dynamic hybrid chop filter (not to a different data window or fetch).

  Arm A (control):   filter OFF — trains on the full population (the old measure)
  Arm B (treatment): filter ON  — trains on the live-tradeable population

Calls only engineer_features_and_labels + validate_candidate, so it does NOT
promote or overwrite production model weights.

Run (from repo root, env loaded):
    PYTHONPATH=src:. python chop_ab_test.py
Honors RETRAIN_DAYS_BACK / RETRAIN_SYMBOLS like the retrainer.
"""
import hashlib
import logging
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.core.retrainer import (  # noqa: E402
    DAYS_BACK,
    engineer_features_and_labels,
    fetch_training_data,
    get_asset_config,
    get_hyperparameters,
    validate_candidate,
)
from src.data.factory import get_market_provider  # noqa: E402
from src.execution.risk_manager import RiskProfile  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s"
)
log = logging.getLogger("chop_ab")


def get_data(asset_config, days_back):
    """Load the dataset from parquet cache, fetching + caching on first run."""
    syms = asset_config["tickers"]
    tf = asset_config["timeframe_minutes"]
    key = hashlib.md5(
        f"{','.join(syms)}_{days_back}d_{tf}m".encode()
    ).hexdigest()[:10]
    cache = ROOT / "data" / "cache" / f"forex_{days_back}d_{tf}m_{key}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        log.info("Loading cached dataset: %s", cache)
        return pl.read_parquet(cache), cache
    log.info("No cache present — fetching %dd of %d instruments (slow)...", days_back, len(syms))
    provider = get_market_provider()
    raw = fetch_training_data(
        provider=provider, symbols=syms, days_back=days_back, timeframe_minutes=tf
    )
    raw.write_parquet(cache)
    log.info("Cached %s rows -> %s", f"{raw.height:,}", cache)
    return raw, cache


def run_arm(name, chop_on, raw, cfg, angel_params, devil_params):
    os.environ["RISK_CHOP_FILTER_ENABLED"] = "1" if chop_on else "0"
    profile = RiskProfile.for_asset_class("forex")
    log.info("=" * 70)
    log.info("ARM %s | RISK_CHOP_FILTER_ENABLED=%s", name, os.environ["RISK_CHOP_FILTER_ENABLED"])
    log.info("=" * 70)
    feats, cols, veto_rate = engineer_features_and_labels(
        raw.clone(),
        sl_mult=cfg["sl_mult"],
        tp_mult=cfg["tp_mult"],
        max_hold=cfg["max_hold"],
        survival_bars=cfg["survival_bars"],
        htf_timeframe=cfg.get("htf_timeframe", "5m"),
        risk_profile=profile,
    )
    report, *_ = validate_candidate(
        feats,
        cols,
        sl_mult=cfg["sl_mult"],
        tp_mult=cfg["tp_mult"],
        n_folds=3,
        angel_params=angel_params,
        devil_params=devil_params,
        chop_veto_rate=veto_rate,
    )
    return {
        "arm": name,
        "rows": feats.height,
        "pf": report.final_profit_factor,
        "brier": report.mean_brier,
        "ev": report.mean_ev,
        "wr": report.final_win_rate,
        "pooled": report.pooled_oos_trades,
        "floor": report.effective_trade_floor,
        "gate": report.gate_passed,
    }


def main():
    days = int(os.getenv("RETRAIN_DAYS_BACK", str(DAYS_BACK)))
    cfg = get_asset_config("oanda")
    angel_params, devil_params = get_hyperparameters("forex")
    raw, cache = get_data(cfg, days)
    log.info(
        "Dataset: %s rows | %d instruments | %dd | cache=%s",
        f"{raw.height:,}", len(cfg["tickers"]), days, cache.name,
    )

    results = [
        run_arm("A_control_OFF", False, raw, cfg, angel_params, devil_params),
        run_arm("B_treatment_ON", True, raw, cfg, angel_params, devil_params),
    ]

    log.info("=" * 70)
    log.info("CHOP FILTER A/B RESULTS — identical data (%s)", cache.name)
    log.info(
        "%-16s %12s %8s %8s %9s %8s %7s %7s %6s",
        "arm", "rows", "PF", "Brier", "EV", "WR", "pooled", "floor", "gate",
    )
    for r in results:
        log.info(
            "%-16s %12s %8.4f %8.4f %9.5f %8.4f %7d %7.0f %6s",
            r["arm"], f"{r['rows']:,}", r["pf"], r["brier"], r["ev"], r["wr"],
            r["pooled"], r["floor"], str(r["gate"]),
        )
    a, b = results
    vetoed = a["rows"] - b["rows"]
    log.info("-" * 70)
    log.info("Rows vetoed by filter : %s (%.1f%% of control)", f"{vetoed:,}", 100.0 * vetoed / max(a["rows"], 1))
    log.info("delta PF (B - A)      : %+.4f  (treatment %s control)", b["pf"] - a["pf"], "OUTPERFORMS" if b["pf"] > a["pf"] else "UNDERPERFORMS")
    log.info("Gate (pooled floor)   : control=%s  treatment=%s", a["gate"], b["gate"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
