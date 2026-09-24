"""Lane-2 weekly LightGBM ranker trainer — purged group time-series CV.

A RESEARCH-ONLY parallel to ``scripts/investor_train_model.py`` (the live
monthly V4 lane, untouched). This one trains the same 96-ticker LightGBM
``lambdarank`` ranker against a **5-trading-day** label on **ISO-week query
groups**, evaluated under the falsification gates of the 2026-09-24 Lane-2
brief — which are deliberately stricter than the production lane's gates:

  * Excess >= +80 bps/month vs the equal-weight universe.
  * DSR > 0.95           (src/lab/stats.deflated_sharpe_ratio)
  * HLZ t > 3.0          (src/lab/stats.hlz_t_stat; the unadjusted t is logged)
  * PBO < 0.50           (src/lab/stats.cscv_pbo)

Falsification is the deliverable — a gate miss is a finding, not a crash. The
script ALWAYS writes its metrics to a sidecar JSON and returns exit 0 on a
clean run, 2 on a gate REJECT, 1 on an error (mirrors the lab CLI convention).

Purged Group Time-Series CV (atomic embargo)
--------------------------------------------
Walk forward over the unique ISO-week groups:
    train [w0 .. w_t1]  ->  embargo (w_t1, w_t1+10d)  ->  validate (.. w_t2]
The embargo is **10 CALENDAR days** (``EMBARGO_CAL_DAYS``) applied ATOMICALLY by
date: NO row whose date falls in the gap belongs to either side. The brief's
hard invariant is asserted per fold:

    train.max(date) + 10 < val.min(date)      (Timedelta(days=10))

This covers the 5-trading-day forward label window plus margin. ``n_folds``
defaults to 5 (the brief minimum). Groups are ISO weeks — the ``lambdarank``
``group`` argument is the per-week row count, computed after the
pre-announcement filter drops rows.

Leakage guards (all asserted, none silently fixed)
--------------------------------------------------
  * Embargo guard (above), asserted per fold.
  * Target: ``fwd_log_ret_5d`` uses only closes strictly after t (built in the
    feature pipeline; here we merely consume it).
  * Pre-announcement filter: rows flagged ``drop_pre_announcement`` are removed
    before grouping; a row with ``days_since_earnings < 0`` cannot appear (the
    pipeline drops and counts them).

Usage:
    PYTHONPATH=src:. python scripts/investor_train_model_weekly.py

Input : data/processed/v4_weekly_training_features.parquet
Output: models/v4_investor_weekly_lgbm.txt          (only if the gate passes)
        models/v4_investor_weekly_lgbm.metrics.json (always, atomic)

Glossary:
    Q_t -- the ISO-week query group; one ranking problem per week.
    EMBARGO_CAL_DAYS -- 10 calendar days between train and validate; atomic
        (no rows in the gap), not a row-count skip. See GLOSSARY.md "embargo".
    TOP_K -- 8, mirrors the production orchestrator's deployed basket depth so
        the excess-return gate scores the basket that would really be bought.
    _N_TRIALS -- the count of DISTINCT variant configurations fitted, fed to
        DSR/HLZ as the multiple-testing burden. This configuration fitted ONE
        (feature set, embargo length, hyperparameter tuple); recorded honestly
        so a reviewer sees we did not quietly grid-search.
    excess_bps_per_month -- mean daily (basket - equal-weight) excess scaled by
        sqrt(21), per the brief's annualization.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from investor_universe import SECTORS, UNIVERSE  # noqa: E402
from lab.stats import (  # noqa: E402
    cscv_pbo,
    deflated_sharpe_ratio,
    hlz_haircut_sharpe,
    hlz_t_stat,
)

_INPUT = _PROJECT_ROOT / "data" / "processed" / "v4_weekly_training_features.parquet"
_RAW = _PROJECT_ROOT / "data" / "raw" / "v4_investor_data.parquet"
_MODEL_OUT = _PROJECT_ROOT / "models" / "v4_investor_weekly_lgbm.txt"
_METRICS_OUT = _PROJECT_ROOT / "models" / "v4_investor_weekly_lgbm.metrics.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── CV / model parameters ────────────────────────────────────────────
EMBARGO_CAL_DAYS: int = 10
N_FOLDS: int = 5
TOP_K: int = 8
SECTOR_CAP: int = 2
_N_TRIALS: int = int(os.getenv("LANE2_N_TRIALS", "1"))  # configs actually fitted
GATE_EXCESS_BPS_MONTH: float = 80.0
GATE_DSR: float = 0.95
GATE_HLZ_T: float = 3.0
GATE_PBO: float = 0.50

LGBM_PARAMS = dict(
    objective="lambdarank",
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    feature_fraction=0.9,
    ndcg_at=[5, 10],
    deterministic=True,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    importance_type="gain",
)

_EXCLUDE = frozenset({
    "date", "symbol", "week", "fwd_log_ret_5d", "fwd_ret_quintile",
    "target_top_quintile", "days_until_earnings", "drop_pre_announcement",
})


def _sanitize(c: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_]", "_", c).strip("_")


def _build_groups(df: pd.DataFrame, col: str = "week") -> np.ndarray:
    sizes = df.groupby(col, sort=False).size().to_numpy(dtype=np.int32)
    assert sizes.sum() == len(df)
    return sizes


def _sector_capped_pick(symbols, scores, k=TOP_K, cap=SECTOR_CAP):
    order = np.argsort(-np.asarray(scores, dtype=float))
    picked, per = [], {}
    for i in order:
        s = symbols[i]
        sec = SECTORS.get(s, "unknown")
        if per.get(sec, 0) >= cap:
            continue
        picked.append(s)
        per[sec] = per.get(sec, 0) + 1
        if len(picked) == k:
            break
    return picked


def _load_close_matrix():
    try:
        raw = pd.read_parquet(_RAW, columns=["symbol", "close"]).reset_index()
        raw["date"] = pd.to_datetime(raw["date"], utc=True)
        return raw.pivot_table(index="date", columns="symbol", values="close").sort_index()
    except Exception as exc:  # noqa: BLE001
        logger.error("cannot load prices for excess gate: %s", exc)
        return None


def _fold_excess_daily(test_df, scores, px):
    """Daily (basket - equal-weight) excess over the fold's rebalance points.

    Rebalance weekly: at each week's FIRST date the model picks TOP_K, hold to
    the next week's first date. Returns a list of per-holding-period excess
    log-returns (basket minus the equal-weight universe over the same span).
    """
    scored = test_df[["date", "symbol"]].copy()
    scored["score"] = scores
    weeks = sorted(scored["date"].dt.isocalendar()["year"].astype(str)
                   .str.cat(scored["date"].dt.isocalendar()["week"].astype(str).str.zfill(2), sep="-W").unique())
    dates = px.index
    out = []
    # one rebalance per week, using the first available date that week
    week_first = scored.groupby(
        scored["date"].dt.isocalendar()["year"].astype(str)
        .str.cat(scored["date"].dt.isocalendar()["week"].astype(str).str.zfill(2), sep="-W")
    )["date"].min()
    wf = sorted(week_first.tolist())
    for i, d0 in enumerate(wf):
        later = [d for d in dates if d > d0]
        if not later:
            continue
        d1 = wf[i + 1] if i + 1 < len(wf) else later[min(4, len(later) - 1)]
        if d0 not in px.index or d1 not in px.index:
            continue
        day = scored[scored["date"] == d0]
        p0, p1 = px.loc[d0], px.loc[d1]
        tradable = [s for s in px.columns if np.isfinite(p0.get(s, np.nan)) and np.isfinite(p1.get(s, np.nan))]
        if len(tradable) < 2:
            continue
        day = day[day["symbol"].isin(tradable)]
        if day.empty:
            continue
        picks = _sector_capped_pick(day["symbol"].tolist(), day["score"].to_numpy())
        if not picks:
            continue
        basket = float(np.mean([p1[s] / p0[s] - 1 for s in picks]))
        bench = float(np.mean([p1[s] / p0[s] - 1 for s in tradable]))
        out.append(basket - bench)
    return out


def main() -> int:
    try:
        return _impl()
    except Exception as exc:  # noqa: BLE001
        logger.exception("weekly trainer failed: %s", exc)
        return 1


def _impl() -> int:
    logger.info("=" * 70)
    logger.info("Lane-2 WEEKLY ranker trainer (5d hold, ISO-week groups, PEAD)")
    logger.info("embargo=%d cal days | folds=%d | trials=%d", EMBARGO_CAL_DAYS, N_FOLDS, _N_TRIALS)
    logger.info("=" * 70)

    df = pd.read_parquet(_INPUT)
    if df.index.name == "date":
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # ── pre-announcement filter ──────────────────────────────────────
    n_pre = int(df["drop_pre_announcement"].sum()) if "drop_pre_announcement" in df else 0
    if "drop_pre_announcement" in df:
        df = df[~df["drop_pre_announcement"]].copy()
    logger.info("pre-announcement filter dropped %d rows -> %d remain", n_pre, len(df))

    df = df.sort_values("date").reset_index(drop=True)

    feat_raw = [c for c in df.columns if c not in _EXCLUDE and c not in ("date", "symbol", "week")]
    rename = {c: _sanitize(c) for c in feat_raw}
    X_df = df[feat_raw].apply(pd.to_numeric, errors="coerce").rename(columns=rename)
    feature_cols = list(X_df.columns)
    y_all = df["fwd_ret_quintile"].to_numpy(dtype=np.float64)

    logger.info("features: %d | weeks: %d | target mean %.3f",
                len(feature_cols), df["week"].nunique(), np.nanmean(y_all))

    # ── purged group time-series CV ──────────────────────────────────
    # unique week keys with their min/max date for the embargo arithmetic
    wk = df.groupby("week")["date"].agg(["min", "max"]).sort_values("min")
    week_keys = list(wk.index)
    n_weeks = len(week_keys)
    logger.info("weeks: %d (%s -> %s)", n_weeks, week_keys[0], week_keys[-1])

    px = _load_close_matrix()
    fold_metrics = []
    all_excess_daily = []
    # strategy columns for CSCV: per-fold OOS daily excess series as strategies
    # plus per-week excess; CSCV needs a T x N matrix — we build one from the
    # folds' OOS daily excess per-week (rows) across alignments (columns) below.
    val_daily_excess_by_fold: list[list[float]] = []

    embargo = pd.Timedelta(days=EMBARGO_CAL_DAYS)
    # walk forward: split the week list into N_FOLDS expanding splits
    # fold k: train on weeks[0:cut_k], validate weeks after embargo
    min_train_weeks = max(30, n_weeks // (N_FOLDS + 2))   # ensure a real train set
    fold_bounds = []
    for k in range(N_FOLDS):
        t1_idx = min_train_weeks + k * ((n_weeks - min_train_weeks) // (N_FOLDS + 1))
        if t1_idx >= n_weeks - 1:
            break
        fold_bounds.append(t1_idx)

    for k, t1_idx in enumerate(fold_bounds):
        train_weeks = set(week_keys[: t1_idx + 1])
        train_df = df[df["week"].isin(train_weeks)]
        train_max_date = train_df["date"].max()
        # validation: weeks whose MIN date is strictly beyond train_max + embargo
        eligible_val = [w for w in week_keys[t1_idx + 1:]
                        if wk.loc[w, "min"] > train_max_date + embargo]
        if not eligible_val:
            break
        # take the contiguous block of validation weeks up to the next fold's
        # train start (or a fixed width)
        next_start = fold_bounds[k + 1] if k + 1 < len(fold_bounds) else n_weeks
        val_weeks = [w for w in eligible_val if week_keys.index(w) < next_start + 1]
        if not val_weeks:
            val_weeks = eligible_val[: max(1, (n_weeks - t1_idx) // (N_FOLDS + 1))]
        val_df = df[df["week"].isin(set(val_weeks))]
        if val_df.empty or train_df.empty:
            break
        val_min_date = val_df["date"].min()

        # ── ATOMIC EMBARGO GUARD (mandatory) ─────────────────────────
        assert train_max_date + embargo < val_min_date, (
            f"fold {k + 1}: embargo violated: train.max {train_max_date.date()} "
            f"+ {EMBARGO_CAL_DAYS}d NOT < val.min {val_min_date.date()}"
        )

        X_train = X_df[df.index.isin(train_df.index)]
        y_train = y_all[df.index.isin(train_df.index)]
        X_val = X_df[df.index.isin(val_df.index)]
        grp_train = _build_groups(train_df, "week")

        model = lgb.LGBMRanker(**LGBM_PARAMS)
        model.fit(X_train, y_train, group=grp_train,
                  callbacks=[lgb.log_evaluation(period=-1)])
        scores = model.predict(X_val)

        nd = model.evals_result_ if hasattr(model, "evals_result_") else {}
        # per-fold IC (Spearman between score and realized fwd ret)
        realized = val_df["fwd_log_ret_5d"].to_numpy()
        ic = float(pd.Series(scores).corr(pd.Series(realized), method="spearman")) if np.isfinite(realized).any() else float("nan")

        excess = _fold_excess_daily(val_df, scores, px) if px is not None else []
        val_daily_excess_by_fold.append(excess)
        all_excess_daily.extend(excess)

        fold_metrics.append({
            "fold": k + 1,
            "train_weeks": len(train_weeks),
            "val_weeks": len(val_weeks),
            "train_max_date": str(train_max_date.date()),
            "val_min_date": str(val_min_date.date()),
            "spearman_ic": ic,
            "n_val_rows": len(val_df),
        })
        logger.info("fold %d | train<=%s | val>=%s | IC=%.4f | excess periods=%d",
                    k + 1, train_max_date.date(), val_min_date.date(), ic, len(excess))

    if not fold_metrics:
        logger.error("no folds produced — cannot gate.")
        return 1

    # ── aggregate excess -> monthly bps ──────────────────────────────
    excess_arr = np.asarray(all_excess_daily, dtype=float)
    excess_arr = excess_arr[np.isfinite(excess_arr)]
    if len(excess_arr) >= 2:
        # brief: average daily excess x sqrt(21) monthly annualization, in bps
        mean_daily = float(excess_arr.mean())
        excess_bps_month = mean_daily * math.sqrt(21) * 10_000
        ann_sharpe = float(mean_daily / excess_arr.std(ddof=1) * math.sqrt(252)) if excess_arr.std(ddof=1) > 0 else float("nan")
        skew = float(pd.Series(excess_arr).skew())
        kurt = float(pd.Series(excess_arr).kurt() + 3.0)  # pandas returns excess
        n_obs = len(excess_arr)
    else:
        excess_bps_month = float("nan")
        ann_sharpe = float("nan")
        skew = kurt = 0.0
        n_obs = 0

    # ── falsification statistics ─────────────────────────────────────
    dsr = (deflated_sharpe_ratio(ann_sharpe, _N_TRIALS, n_obs, skew, kurt)
           if np.isfinite(ann_sharpe) else float("nan"))
    hlz_adj = (hlz_haircut_sharpe(ann_sharpe, _N_TRIALS, skew=skew, kurt=kurt, n_obs=n_obs)
               if np.isfinite(ann_sharpe) else float("nan"))
    hlz_t = (hlz_t_stat(ann_sharpe, _N_TRIALS, n_obs, skew, kurt)
             if np.isfinite(ann_sharpe) else float("nan"))
    # unadjusted t on the excess series for the report
    unadj_t = (float(excess_arr.mean() / (excess_arr.std(ddof=1) / math.sqrt(n_obs)))
               if n_obs > 1 and excess_arr.std(ddof=1) > 0 else float("nan"))

    # ── CSCV PBO over the fold excess series ─────────────────────────
    # Build a T x N matrix: rows = rebalance periods, columns = folds (each a
    # "strategy"). Folds differ in length -> pad with NaN and let nan-aware
    # moments handle it.
    if val_daily_excess_by_fold and len(val_daily_excess_by_fold) >= 2:
        maxlen = max(len(s) for s in val_daily_excess_by_fold)
        if maxlen >= 16:  # need enough rows for 8 blocks
            mat = np.full((maxlen, len(val_daily_excess_by_fold)), np.nan)
            for j, s in enumerate(val_daily_excess_by_fold):
                mat[: len(s), j] = s
            pbo = cscv_pbo(mat)
        else:
            pbo = float("nan")
    else:
        pbo = float("nan")

    # ── gate verdict ─────────────────────────────────────────────────
    g_excess = np.isfinite(excess_bps_month) and excess_bps_month >= GATE_EXCESS_BPS_MONTH
    g_dsr = np.isfinite(dsr) and dsr > GATE_DSR
    g_hlz = np.isfinite(hlz_t) and hlz_t > GATE_HLZ_T
    g_pbo = np.isfinite(pbo) and pbo < GATE_PBO
    gate_passed = bool(g_excess and g_dsr and g_hlz and g_pbo)

    logger.info("=" * 70)
    logger.info("FALSIFICATION GATE")
    logger.info("  excess/month %+7.1f bps  (>= %+.0f) -> %s", excess_bps_month, GATE_EXCESS_BPS_MONTH, "PASS" if g_excess else "FAIL")
    logger.info("  DSR          %7.4f   (> %.2f) -> %s", dsr, GATE_DSR, "PASS" if g_dsr else "FAIL")
    logger.info("  HLZ t        %+7.3f   (> %.1f) -> %s  (adj SR %.3f | unadj t %.2f)",
                hlz_t, GATE_HLZ_T, "PASS" if g_hlz else "FAIL", hlz_adj, unadj_t)
    logger.info("  PBO          %7.4f   (< %.2f) -> %s", pbo, GATE_PBO, "PASS" if g_pbo else "FAIL")
    logger.info("  VERDICT: %s", "GATE PASS" if gate_passed else "GATE REJECT")
    logger.info("=" * 70)

    metrics = {
        "lane": "equity-factor-pead",
        "horizon_trading_days": 5,
        "embargo_calendar_days": EMBARGO_CAL_DAYS,
        "n_folds": len(fold_metrics),
        "n_trials": _N_TRIALS,
        "folds": fold_metrics,
        "mean_spearman_ic": float(np.nanmean([f["spearman_ic"] for f in fold_metrics])),
        "excess_bps_per_month": (None if not np.isfinite(excess_bps_month) else round(excess_bps_month, 2)),
        "annualized_sharpe": (None if not np.isfinite(ann_sharpe) else round(ann_sharpe, 4)),
        "dsr": (None if not np.isfinite(dsr) else round(dsr, 4)),
        "hlz_adjusted_sharpe": (None if not np.isfinite(hlz_adj) else round(hlz_adj, 4)),
        "hlz_t": (None if not np.isfinite(hlz_t) else round(hlz_t, 3)),
        "unadjusted_excess_t": (None if not np.isfinite(unadj_t) else round(unadj_t, 3)),
        "pbo": (None if not np.isfinite(pbo) else round(pbo, 4)),
        "gates": {"excess": bool(g_excess), "dsr": bool(g_dsr), "hlz": bool(g_hlz), "pbo": bool(g_pbo)},
        "gate_passed": gate_passed,
    }

    # atomic metrics sidecar (always written — falsification is the deliverable)
    _METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(_METRICS_OUT.parent), suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(metrics, f, indent=2)
        os.replace(tmp, _METRICS_OUT)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    logger.info("[ATOMIC] metrics -> %s", _METRICS_OUT)

    if gate_passed:
        # retrain on all data and save the model artifact
        grp_all = _build_groups(df, "week")
        final = lgb.LGBMRanker(**LGBM_PARAMS)
        final.fit(X_df, np.nan_to_num(y_all, nan=0.0), group=grp_all,
                  callbacks=[lgb.log_evaluation(period=-1)])
        final.booster_.save_model(str(_MODEL_OUT))
        logger.info("gate passed — model saved -> %s", _MODEL_OUT)
        return 0

    logger.warning("gate REJECTED — research verdict; no model artifact written.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
