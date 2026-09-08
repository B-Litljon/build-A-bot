"""
Stage 2 — CatBoost A/B against the LightGBM incumbent on IDENTICAL folds.

Per the stage-02 brief (`llm_reports/m2m-prompts/2026-09-08_stage-02-catboost-swap.md`):
one variable changed (the estimator family), same folds / features / labels /
threshold logic, compared on the fold gate's own metrics:

  * mean Brier (gate bar <= 0.30)
  * mean EV   (gate bar >= 0.0005)
  * pooled / fold-3 Clopper-Pearson PF lower bounds (gate bar >= 1.20)
  * 0.40+ confidence-band realised win rate on the pooled OOS ledger — the
    calibration-inversion metric; baseline 11.8% (n=34, 2026-09-08 report)

The driver follows capture_behavior_ledger.py's recipe: same fetch via the
provider (cache-backed), holdout carved first, engineering + boundary purge on
the remainder, then validate_candidate per family with an oos_ledger capture
for the band analysis. Nothing is written under models/; artifacts are two
CSVs + two ledger parquets under logs/. Exit 0 = promotion bar 1 met (CB
strictly better on ALL THREE of Brier/EV/CP-PF bounds); exit 2 = incumbent
stands.

Env (in addition to the standard retrain vars):
    CB_DAYS_BACK  -- training window for BOTH arms (default 60, matching
                     DAYS_BACK; set to 730 to reproduce the decision-report
                     population span).
    CB_TAG        -- output suffix (default stamps MODEL_FAMILY).

Glossary:
    arm -- one run of validate_candidate under a fixed MODEL_FAMILY with its
        own ledger capture; both arms see the same engineered frame, so fold
        boundaries (calendar-derived) are identical by construction.
    top band -- ledger rows with angel_prob >= the arm's OWN calibrated
        proposal bar (report.production_angel_threshold); 0.40-fixed when the
        arm calibrates below it, so inversion is measured on a population the
        model would actually propose at the live bar.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import polars as pl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("core.retrainer").setLevel(logging.WARNING)
logger = logging.getLogger("ab_catboost")

OUT_DIR = Path("logs")
CACHE_DIR = Path("data/cache/ab_catboost")
_FETCH_PAUSE_S = 6.0  # the live soak shares this API key; stay well under it


def _fetch_cached(provider, symbols, days_back, timeframe_minutes) -> pl.DataFrame:
    """
    Per-symbol parquet cache ahead of the provider. retrainer's own fetch is
    uncached (main() always wants live bars); the experiment wants to re-read
    the same window twice (or more) without re-hammering the API the live
    soak is drawing from.
    """
    import core.retrainer as R

    end_override = os.getenv("RETRAIN_END_DATE", "").strip()
    end = (
        datetime.strptime(end_override, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_override
        else datetime.now(timezone.utc)
    )
    start = end - timedelta(days=days_back)

    out = []
    for sym in symbols:
        path = CACHE_DIR / f"{sym}_M{timeframe_minutes}_{days_back}d_{end:%Y%m%d}.parquet"
        if path.exists():
            out.append(pl.read_parquet(path))
            logger.info("%s: cache hit (%s)", sym, path.name)
            continue
        # Mirror retrainer.fetch_training_data: OANDA intermittently 401s on
        # bursts and the provider returns empty, so retry with backoff. The
        # extra sleep after a success keeps us well clear of the soak's key.
        df = None
        for attempt in range(1, 4):
            try:
                df = provider.get_historical_bars(sym, timeframe_minutes, start, end)
            except Exception as e:  # noqa: BLE001 — same all-errors contract as retrainer
                logger.error("%s fetch attempt %d/3: %s", sym, attempt, e)
                df = None
            if df is not None and not df.is_empty():
                break
            time.sleep(5 * attempt)
        if df is None or df.is_empty():
            logger.warning("%s: provider returned nothing; skipping", sym)
            continue
        df.columns = [c.lower() for c in df.columns]
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(sym).alias("symbol"))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)  # write-then-read on next run; nothing
        # hot-reloads this dir, so no atomic-rename ceremony needed
        out.append(df)
        logger.info("%s: fetched %d bars", sym, df.height)
        time.sleep(_FETCH_PAUSE_S)
    if not out:
        raise RuntimeError("no symbols fetched or cached")
    combined = pl.concat(out, how="vertical_relaxed")
    return combined.sort(["symbol", "timestamp"])



def _run_arm(family: str, feats, cols, cfg, ap, dp):
    """One arm: walk-forward under MODEL_FAMILY=family; report + pooled ledger."""
    import core.retrainer as R

    R.MODEL_FAMILY = family
    ledger = []
    report, *_rest = R.validate_candidate(
        feats,
        cols,
        sl_mult=cfg["sl_mult"],
        tp_mult=cfg["tp_mult"],
        n_folds=3,
        angel_params=ap,
        devil_params=dp,
        chop_veto_rate=0.0,
        oos_ledger=ledger,
        oos_ledger_cols=("timestamp", "symbol", "natr_14", "ppo", "close"),
    )
    led = pl.concat(ledger) if ledger else pl.DataFrame()
    return report, led


def _band_stats(led: pl.DataFrame, bar: float) -> dict:
    """Realised win rate in angel_prob bands, on the arm's own proposals."""
    bands = [(0.20, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.50), (0.50, 1.01)]
    out = {}
    eff_bar = max(bar, 0.40)  # the live bar: see module docstring
    for lo, hi in bands:
        rows = led.filter(
            (pl.col("angel_prob") >= lo) & (pl.col("angel_prob") < hi)
        )
        n = rows.height
        out[f"{lo:.2f}-{hi:.2f}"] = (
            (float(rows["macro_win"].mean()), n) if n else (float("nan"), 0)
        )
    rows = led.filter(pl.col("angel_prob") >= eff_bar)
    n = rows.height
    out[f">={eff_bar:.2f}"] = (
        (float(rows["macro_win"].mean()), n) if n else (float("nan"), 0)
    )
    return out


def main() -> int:
    import core.retrainer as R
    from data.factory import get_market_provider
    from execution.risk_manager import RiskProfile

    days = int(os.environ.get("CB_DAYS_BACK", R.DAYS_BACK))
    tag = os.environ.get("CB_TAG", "").strip()
    cfg = R.get_asset_config(os.getenv("DATA_SOURCE", "alpaca").strip().lower())

    logger.info("=== Stage 2 CatBoost A/B: %sd, %sm tf ===", days, cfg["timeframe_minutes"])
    provider = get_market_provider()
    raw = _fetch_cached(
        provider, cfg["tickers"], days, cfg["timeframe_minutes"]
    )
    # Holdout carved exactly like main(); the fold run uses the remainder only.
    rem_raw, hold_raw, rng = R._split_holdout(raw, R.HOLDOUT_FRAC)
    logger.info(
        "holdout %s -> %s (%d rows); remainder %d rows",
        rng[0].date() if rng else "-", rng[1].date() if rng else "-",
        hold_raw.height, rem_raw.height,
    )
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
    feats, n_purged = R._purge_boundary_tail(
        feats, R._tail_cutoff_by_symbol(rem_raw, cfg["max_hold"])
    )
    logger.info("engineered %d rows (boundary purge dropped %d)", feats.height, n_purged)

    ap, dp = R.get_hyperparameters(cfg["asset_class"])

    results = {}
    for family in ("lightgbm", "catboost"):
        logger.info("=== arm: %s ===", family)
        report, led = _run_arm(family, feats, cols, cfg, ap, dp)
        suffix = f"_{tag}" if tag else ""
        led_path = OUT_DIR / f"ab_ledger_{family}{suffix}.parquet"
        if led.height:
            led.write_parquet(led_path)
        results[family] = (report, led)
        fm = report.fold_metrics
        lines = [
            f"family={family}",
            f"  gate_passed={report.gate_passed}  reasons={report.rejection_reasons}",
            f"  mean_brier={report.mean_brier:.4f}  mean_ev={report.mean_ev:+.6f}",
            f"  pooled PF lb={report.pooled_pf_lower_bound:.4f} "
            f"(wins={report.pooled_oos_wins}/trades={report.pooled_oos_trades})",
            f"  fold3  PF lb={report.fold3_pf_lower_bound:.4f}",
            f"  angel bar={report.production_angel_threshold:.4f} "
            f"devil bar(F3 calibration froze into report)",
        ]
        for f in fm:
            lines.append(
                f"  fold{f.fold_number}: brier={f.brier_score:.4f} "
                f"ev={f.expected_value:+.6f} win={f.win_rate:.3f} "
                f"trades={f.devil_approved_trades} macro_wins={f.macro_wins}"
            )
        # inversion bands
        for band, (wr, n) in _band_stats(led, report.production_angel_threshold).items():
            lines.append(f"  band {band}: win={wr:.3f} n={n}")
        text = "\n".join(lines)
        print(text)
        (OUT_DIR / f"ab_result_{family}{suffix}.txt").write_text(text + "\n")

    (rl, _), (rc, _) = results["lightgbm"], results["catboost"]
    strictly_better = (
        rc.mean_brier < rl.mean_brier
        and rc.mean_ev > rl.mean_ev
        and rc.pooled_pf_lower_bound > rl.pooled_pf_lower_bound
        and rc.fold3_pf_lower_bound > rl.fold3_pf_lower_bound
    )
    verdict = (
        "CATBOOST WINS ALL FOUR (promotion bar 1 met; human decision)"
        if strictly_better
        else "INCUMBENT STANDS (tie or loss on at least one gate metric)"
    )
    print("\nVERDICT:", verdict)
    return 0 if strictly_better else 2


if __name__ == "__main__":
    sys.exit(main())
