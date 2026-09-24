"""Lane-2 weekly investor feature pipeline — 5-day rebalance + PEAD/SUE.

A RESEARCH-ONLY parallel to ``scripts/investor_feature_pipeline.py`` (the live
monthly V4 lane). That production script is untouched; this one re-targets the
same 96-ticker universe to a **5-trading-day** hold with two additions the
monthly lane never had:

  1. **PEAD / SUE** — Post-Earnings Announcement Drift, driven by a
     point-in-time-safe Standardized Unexpected Earnings signal built from
     SimFin quarterly actuals.
  2. **Announcement hygiene** — a pre-announcement avoidance filter and an
     "impossible negative days_since_earnings" counter that surface bad
     publish-date bookkeeping instead of silently absorbing it.

It MATERIALIZES the earnings calendar to ``data/processed/earnings_calendar.parquet``
(atomic write) with columns ``[symbol, earnings_date, eps_reported,
publish_date, source]``; the trainer and the leakage-guard tests read it back.

Point-in-time invariant (HARD, tested)
--------------------------------------
For every feature row dated ``t``, every fundamental/SUE input comes from a
SimFin row whose ``Publish Date <= t``. The publish date is the date the filing
hit the wire — SimFin bulk files ship it directly, fully populated (0 nulls in
45,936 income rows, verified 2026-09-24), so **no announcement-date
reconstruction is required** (brief §3 source #2 was available; §6.5 abort is
NOT triggered). SimFin's own ``Restated Date`` and any TTM/derived columns are
deliberately NOT used for SUE — they are not point-in-time safe.

Forward label & embargo
-----------------------
Target ``fwd_log_ret_5d = ln(close_{t+5}/close_t)`` uses ONLY closes strictly
after ``t``. The trainer applies a 10-CALENDAR-day atomic embargo between the
train and validation windows — wide enough to cover the 5-day label horizon
plus margin — and asserts ``train.max(date) + 10d < val.min(date)`` per fold.

Usage:
    PYTHONPATH=src:. python scripts/investor_feature_pipeline_weekly.py
    PYTHONPATH=src:. python scripts/investor_feature_pipeline_weekly.py --inference

Inputs (read-only, git-ignored data/ symlinked from the live checkout):
    data/raw/v4_investor_data.parquet          OHLCV + VIX + 10Y + fundamentals
    data/raw/simfin_cache/us-income-quarterly.csv   SimFin income (Publish Date)
    data/raw/simfin_cache/us-balance-quarterly.csv  SimFin balance (Publish Date)

Outputs (atomic):
    data/processed/earnings_calendar.parquet
    data/processed/v4_weekly_training_features.parquet   (default)
    data/processed/v4_weekly_inference_features.parquet  (--inference)

Glossary:
    SUE -- Standardized Unexpected Earnings, see GLOSSARY.md. Here
        ``sue_q = (EPS_q - E[EPS_q]) / std(EPS_q - E[EPS_q])`` with the
        expectation a seasonal random walk: EPS of the SAME fiscal quarter one
        year prior (8-quarter trailing window, min 4 usable). Point-in-time safe
        because every quarter feeding it has Publish Date <= the signal date.
    PEAD -- Post-Earnings Announcement Drift, see GLOSSARY.md. The tendency for
        prices to keep drifting in the direction of an earnings surprise for
        days after the announcement; the reason ``days_since_earnings`` and the
        negative-SUE carry a signal at all.
    earnings_date -- the SimFin ``Report Date`` (fiscal quarter end). The
        *announcement* is the ``Publish Date``; the calendar carries both.
    publish_date -- SimFin ``Publish Date``; the only timestamp a feature row is
        allowed to look at, and the field the PIT guard enforces on.
    days_since_earnings -- t minus the most recent publish_date, in trading
        days. Negative => the calendar claims an announcement the date says has
        not happened yet; counted and logged, the rows dropped.
    days_until_earnings -- next publish_date minus t, in trading days. Used
        ONLY for the pre-announcement filter (drop 0 <= d <= 2), never fed to
        the model as a feature or a target input.
    Q_t -- the ISO week a trading date belongs to; the LightGBM ``lambdarank``
        query-group unit for the weekly ranker.
    fwd_log_ret_5d -- ln(close_{t+5}/close_t); the 5-trading-day forward
        close-to-close log return. Label quintiles floor(4*rank_pct) per week.
    FORWARD_DAYS -- 5 trading days, the hold horizon this lane tests.
    EMBARGO_CAL_DAYS -- 10 CALENDAR days (the trainer's atomic embargo); covers
        the 5-day label window plus margin. Kept here as a constant so the
        pipeline and trainer agree on the number.
    _SUE_LOOKBACK_QTRS -- 8, the trailing-quarter window for the SUE expectation
        and standard deviation. _SUE_MIN_QTRS -- 4, the fewest trailing quarters
        accepted before a SUE is reported (fewer -> NaN, LightGBM-native).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from investor_universe import UNIVERSE  # noqa: E402

_INPUT_PATH = _PROJECT_ROOT / "data" / "raw" / "v4_investor_data.parquet"
_SIMFIN_INCOME = _PROJECT_ROOT / "data" / "raw" / "simfin_cache" / "us-income-quarterly.csv"
_CALENDAR_PATH = _PROJECT_ROOT / "data" / "processed" / "earnings_calendar.parquet"
_OUTPUT_TRAIN = _PROJECT_ROOT / "data" / "processed" / "v4_weekly_training_features.parquet"
_OUTPUT_INFER = _PROJECT_ROOT / "data" / "processed" / "v4_weekly_inference_features.parquet"

# ── logging ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────
MOM_WINDOWS: dict[str, int] = {"mom_3m": 63, "mom_6m": 126, "mom_12m": 252}
MACRO_WINDOW: int = 20
FORWARD_DAYS: int = 5            # 5-trading-day hold horizon (weekly rebase)
EMBARGO_CAL_DAYS: int = 10       # atomic embargo, CALENDAR days (covers 5d label)
_SUE_LOOKBACK_QTRS: int = 8
_SUE_MIN_QTRS: int = 4
# Pre-announcement avoidance: drop a ranking row whose next earnings is this
# many trading days away (inclusive band). See §4.3 of the brief.
PREANN_FILTER_LO: int = 0
PREANN_FILTER_HI: int = 2

_NUMERATOR_COLS: dict[str, str] = {
    "gross_margin": "Gross Profit",
    "operating_margin": "Operating Income",
    "net_margin": "Net Income",
    "ebitda_margin": "EBITDA",
}
_REVENUE_COL = "Total Revenue"

# The 14 production cross-sectional factors PLUS the three PEAD additions.
FACTOR_COLS = [
    "mom_3m", "mom_6m", "mom_12m", "mom_12_1", "reversal_1m",
    "vol_60d", "vol_120d",
    "roa", "debt_to_equity", "gross_profitability",
    "gross_margin", "operating_margin", "net_margin", "ebitda_margin",
    "sue", "days_since_earnings",
]
# days_until_earnings is intentionally EXCLUDED from FACTOR_COLS: it is a
# prefilter field only, never a model input.


# ─────────────────────────────────────────────────────────────────────
# Atomic write helper
# ─────────────────────────────────────────────────────────────────────

def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet atomically (temp file + os.replace) in the target dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    logger.info("[ATOMIC] wrote %s (%d rows x %d cols)", path, *df.shape)


# ─────────────────────────────────────────────────────────────────────
# Earnings calendar (SimFin actuals -> earnings_calendar.parquet)
# ─────────────────────────────────────────────────────────────────────

def build_earnings_calendar(
    income_csv: Path = _SIMFIN_INCOME,
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """Materialize [symbol, earnings_date, eps_reported, publish_date, source].

    Source preference per the brief §3: SimFin bulk files ship a fully
    populated ``Publish Date`` column, so it is used directly (source #2 was
    available; no quarter-end-lag reconstruction is attempted). ``eps_reported``
    is diluted EPS = Net Income (Common) / Shares (Diluted) — point-in-time
    safe because both numerator and denominator belong to the same filing whose
    Publish Date we carry.
    """
    universe = universe if universe is not None else UNIVERSE
    inc = pd.read_csv(income_csv, sep=";")
    inc = inc[inc["Ticker"].isin(universe)].copy()
    if inc.empty:
        raise RuntimeError(f"No SimFin income rows for universe in {income_csv}")

    inc["publish_date"] = pd.to_datetime(inc["Publish Date"], utc=True)
    inc["earnings_date"] = pd.to_datetime(inc["Report Date"], utc=True)

    shares = pd.to_numeric(inc["Shares (Diluted)"], errors="coerce").replace(0, np.nan)
    ni = pd.to_numeric(inc["Net Income (Common)"], errors="coerce")
    inc["eps_reported"] = ni / shares

    cal = (
        inc[["Ticker", "earnings_date", "eps_reported", "publish_date"]]
        .rename(columns={"Ticker": "symbol"})
        .assign(source="simfin_us_income_quarterly")
    )
    # Order + drop exact duplicate filings (same symbol, same publish date).
    cal = cal.sort_values(["symbol", "publish_date"]).drop_duplicates(
        subset=["symbol", "publish_date"], keep="first"
    )
    # Fiscal-period ordering for the SUE trailing window.
    cal = cal.reset_index(drop=True)

    # Guard: publish_date must be >= earnings_date (you cannot publish a filing
    # before the fiscal period it reports on ends). Count, don't fix.
    bad = (cal["publish_date"] < cal["earnings_date"]).sum()
    if bad:
        logger.warning("earnings calendar: %d rows with publish_date < earnings_date", int(bad))

    logger.info(
        "earnings calendar: %d filings | %d symbols | publish %s -> %s",
        len(cal), cal["symbol"].nunique(),
        cal["publish_date"].min().date(), cal["publish_date"].max().date(),
    )
    return cal


# ─────────────────────────────────────────────────────────────────────
# SUE (point-in-time safe)
# ─────────────────────────────────────────────────────────────────────

def _compute_sue(cal: pd.DataFrame) -> pd.Series:
    """Seasonal-random-walk SUE per filing (vectorized, per symbol).

    For filing q: expectation = EPS of the SAME fiscal quarter one year prior
    (``eps.shift(4)`` for quarterly reporters), and the surprise is
    ``eps - eps.shift(4)``. The standardization divisor is the trailing rolling
    std of that surprise over ``_SUE_LOOKBACK_QTRS`` filings, requiring
    ``_SUE_MIN_QTRS`` observations. This is a pure function of the publish-sorted
    calendar — it has no notion of a signal date, so it cannot leak by itself;
    the PIT guard fires at JOIN time (only filings with publish_date <= t are
    attached to a row dated t).
    """
    eps = pd.to_numeric(cal["eps_reported"], errors="coerce")
    # Preserve the caller's index through the sort so the result aligns on the
    # original row labels (no positional reindex confusion). Use transform-based
    # ops only (groupby.apply returns a MultiIndex here and misaligns).
    cal_sorted = cal.sort_values(["symbol", "publish_date"])
    eps_s = pd.to_numeric(cal_sorted["eps_reported"], errors="coerce")
    sym = cal_sorted["symbol"]
    same_q_prior = eps_s.groupby(sym, sort=False).shift(4)
    surprise = eps_s - same_q_prior
    std = (
        surprise.groupby(sym, sort=False)
        .transform(lambda s: s.rolling(_SUE_LOOKBACK_QTRS, min_periods=_SUE_MIN_QTRS).std())
    )
    sue_sorted = surprise / std.replace(0, np.nan)
    return sue_sorted.reindex(cal.index)


# ─────────────────────────────────────────────────────────────────────
# Trading-day offset helpers (publish_date <-> trading-day deltas)
# ─────────────────────────────────────────────────────────────────────

def _trading_day_delta(dates_index: pd.DatetimeIndex, event: pd.Timestamp, t: pd.Timestamp) -> int:
    """Signed count of trading days between two dates on the shared calendar."""
    dates = dates_index.values
    ei = int(np.searchsorted(dates, np.datetime64(event.tz_convert(None).to_datetime64())))
    ti = int(np.searchsorted(dates, np.datetime64(t.tz_convert(None).to_datetime64())))
    return ti - ei


# ─────────────────────────────────────────────────────────────────────
# Label helper (top quintile per week -> floor(4*rank_pct) buckets later)
# ─────────────────────────────────────────────────────────────────────

def _top_k_label(series: pd.Series) -> pd.Series:
    """1 if in the top-quintile of the (NaN-aware) group, else 0 — informational."""
    result = pd.Series(float("nan"), index=series.index)
    valid = series.dropna()
    if len(valid) < 2:
        return result
    try:
        q = pd.qcut(valid, q=5, labels=False, duplicates="drop")
        top = int(q.max())
        result.loc[valid.index] = (q == top).astype(float)
    except Exception:
        ranks = valid.rank(pct=True)
        result.loc[valid.index] = (ranks >= 0.8).astype(float)
    return result


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lane-2 weekly feature pipeline + PEAD.")
    parser.add_argument("--inference", action="store_true",
                        help="retain the embargo window; write inference parquet")
    parser.add_argument("--skip-calendar", action="store_true",
                        help="reuse an existing earnings_calendar.parquet")
    args = parser.parse_args(argv)
    inference = bool(args.inference)
    out_path = _OUTPUT_INFER if inference else _OUTPUT_TRAIN

    logger.info("=" * 70)
    logger.info("Lane-2 WEEKLY investor feature pipeline (5d hold + PEAD)")
    logger.info("Mode: %s", "INFERENCE" if inference else "TRAINING")
    logger.info("=" * 70)

    # ── earnings calendar (atomic) ───────────────────────────────────
    if args.skip_calendar and _CALENDAR_PATH.exists():
        cal = pd.read_parquet(_CALENDAR_PATH)
        logger.info("reusing existing calendar: %d filings", len(cal))
    else:
        cal = build_earnings_calendar()
        _atomic_to_parquet(cal, _CALENDAR_PATH)

    # SUE per filing (function of the calendar alone; PIT enforced at join).
    cal["sue"] = _compute_sue(cal)

    # ── load prices ──────────────────────────────────────────────────
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(f"{_INPUT_PATH} missing — symlink/copy it from the live data dir.")
    df = pd.read_parquet(_INPUT_PATH)
    if df.index.name == "date":
        df = df.reset_index()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    logger.info("prices: %d rows | %d symbols | %s -> %s",
                len(df), df["symbol"].nunique(), df["date"].min().date(), df["date"].max().date())

    trading_dates = pd.DatetimeIndex(sorted(df["date"].unique()))

    # ── base cross-sectional factors (mirror production conventions) ──
    logger.info("[Stage 1/4] momentum / volatility ...")
    for name, w in MOM_WINDOWS.items():
        df[name] = df.groupby("symbol")["close"].pct_change(periods=w)
    daily_ret = df.groupby("symbol")["close"].pct_change()
    df["vol_60d"] = daily_ret.groupby(df["symbol"]).transform(lambda x: x.rolling(60, min_periods=60).std())
    df["vol_120d"] = daily_ret.groupby(df["symbol"]).transform(lambda x: x.rolling(120, min_periods=120).std())
    df["mom_12_1"] = df.groupby("symbol")["close"].transform(lambda x: x.shift(21) / x.shift(252) - 1)
    df["reversal_1m"] = df.groupby("symbol")["close"].transform(lambda x: x / x.shift(21) - 1)

    logger.info("[Stage 2/4] fundamental margin / quality ratios ...")
    if _REVENUE_COL in df.columns:
        for ratio, num in _NUMERATOR_COLS.items():
            if num in df.columns:
                df[ratio] = df[num] / df[_REVENUE_COL].replace(0, np.nan)
    df["roa"] = df["Net Income"] / df["Total Assets"].replace(0, np.nan) if {"Net Income", "Total Assets"} <= set(df.columns) else np.nan
    df["debt_to_equity"] = df["Total Liabilities"] / df["Total Equity"].replace(0, np.nan) if {"Total Liabilities", "Total Equity"} <= set(df.columns) else np.nan
    df["gross_profitability"] = df["Gross Profit"] / df["Total Assets"].replace(0, np.nan) if {"Gross Profit", "Total Assets"} <= set(df.columns) else np.nan

    # ── PEAD join (point-in-time) ────────────────────────────────────
    # For each (symbol, date) row, the most recent filing with publish_date<=t
    # supplies sue + days_since_earnings. We do an asof-merge per symbol.
    logger.info("[Stage 3/4] PEAD point-in-time join ...")
    cal_small = cal[["symbol", "publish_date", "sue"]].dropna(subset=["publish_date"]).copy()
    cal_small = cal_small.sort_values("publish_date")
    # asof merges require identical datetime resolution on both keys; normalise
    # to ns/UTC (pandas reads parquet as us/UTC but the price frame is ns/UTC).
    cal_small["publish_date"] = pd.to_datetime(cal_small["publish_date"], utc=True).astype("datetime64[ns, UTC]")

    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"], utc=True).astype("datetime64[ns, UTC]")
    parts = []
    pit_violations = 0
    for sym, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("date")
        csub = cal_small[cal_small["symbol"] == sym]
        if csub.empty:
            grp["sue"] = np.nan
            grp["_last_pub"] = pd.NaT
        else:
            merged = pd.merge_asof(
                grp[["date"]].reset_index(drop=True),
                csub[["publish_date", "sue"]].reset_index(drop=True),
                left_on="date", right_on="publish_date", direction="backward",
            )
            grp["sue"] = merged["sue"].to_numpy()
            grp["_last_pub"] = merged["publish_date"].to_numpy()
            # PIT guard: any row whose matched publish_date is AFTER t is a leak.
            v = int((merged["publish_date"] > merged["date"]).sum())
            pit_violations += v
        parts.append(grp)
    df = pd.concat(parts).sort_values(["symbol", "date"]).reset_index(drop=True)
    if pit_violations:
        raise AssertionError(
            f"PIT guard: {pit_violations} rows matched a publish_date in the future — "
            "point-in-time invariant violated."
        )

    # trading-day deltas for PEAD timing features
    last_pub = pd.to_datetime(df["_last_pub"], utc=True)
    ds = []
    for pub, t in zip(last_pub, df["date"]):
        if pd.isna(pub):
            ds.append(np.nan)
        else:
            ds.append(float(_trading_day_delta(trading_dates, pub, t)))
    df["days_since_earnings"] = ds

    # next earnings (for the FILTER only) — forward asof
    df = df.sort_values("date")
    parts = []
    for sym, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("date")
        csub = cal_small[cal_small["symbol"] == sym]
        if csub.empty:
            grp["days_until_earnings"] = np.nan
        else:
            merged = pd.merge_asof(
                grp[["date"]].reset_index(drop=True),
                csub[["publish_date"]].reset_index(drop=True).rename(columns={"publish_date": "_next_pub"}),
                left_on="date", right_on="_next_pub", direction="forward",
            )
            nxt = pd.to_datetime(merged["_next_pub"], utc=True)
            vals = []
            for npub, t in zip(nxt, grp["date"]):
                if pd.isna(npub):
                    vals.append(np.nan)
                else:
                    vals.append(float(_trading_day_delta(trading_dates, t, npub)))
            grp["days_until_earnings"] = vals
        parts.append(grp)
    df = pd.concat(parts).sort_values(["symbol", "date"]).reset_index(drop=True)

    # Negative days_since_earnings => impossible (would mean publish in future).
    n_neg = int((df["days_since_earnings"] < 0).sum())
    if n_neg:
        logger.warning("PEAD: %d rows with days_since_earnings < 0 (bad publish bookkeeping) — dropping", n_neg)
        df = df[~(df["days_since_earnings"] < 0)]

    # ── forward target + ISO week grouping ──────────────────────────
    logger.info("[Stage 4/4] 5-day forward target + weekly groups ...")
    df["fwd_log_ret_5d"] = df.groupby("symbol")["close"].transform(
        lambda x: np.log(x.shift(-FORWARD_DAYS) / x)
    )
    iso = df["date"].dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    # rank-normalize the 14 cross-sectional factors per week (NOT sue/days_since:
    # those keep raw scale so the ranker learns the PEAD magnitude, but we DO
    # rank the 14 cross-sectionals, mirroring production)
    cs_cols = [c for c in FACTOR_COLS if c not in ("sue", "days_since_earnings")]
    for f in cs_cols:
        if f in df.columns:
            df[f] = df.groupby("week")[f].rank(pct=True)

    # quintile label per week for the lambdarank relevance target. rank(pct) is
    # in (0,1]; floor(4*pct) strands the top quintile at bucket 3 because pct
    # hits exactly 1.0 only for a single max row. Use ceil(5*pct)-1 instead,
    # which places the top ~20% in bucket 4, clipped to [0,4].
    rankpct = df.groupby("week")["fwd_log_ret_5d"].rank(pct=True)
    df["fwd_ret_quintile"] = (np.ceil(5 * rankpct) - 1).clip(0, 4)
    df.loc[df["fwd_log_ret_5d"].isna(), "fwd_ret_quintile"] = np.nan
    df["target_top_quintile"] = df.groupby("week")["fwd_log_ret_5d"].transform(_top_k_label)

    # ── pre-announcement filter (prefilter only) ─────────────────────
    n_ann = int(((df["days_until_earnings"] >= PREANN_FILTER_LO)
                 & (df["days_until_earnings"] <= PREANN_FILTER_HI)).sum())
    df["drop_pre_announcement"] = (
        (df["days_until_earnings"] >= PREANN_FILTER_LO)
        & (df["days_until_earnings"] <= PREANN_FILTER_HI)
    )
    logger.info("pre-announcement filter flags %d rows (0<=days_until<=2)", n_ann)

    # ── embargo handling ─────────────────────────────────────────────
    if inference:
        keep = df.copy()
        logger.info("inference mode: retaining %d embargo rows", int(df["fwd_ret_quintile"].isna().sum()))
    else:
        keep = df.dropna(subset=["fwd_ret_quintile"]).copy()
        logger.info("training mode: dropped %d embargo rows", len(df) - len(keep))

    keep_cols = (
        ["date", "symbol", "week", "fwd_log_ret_5d", "fwd_ret_quintile",
         "target_top_quintile", "days_until_earnings", "drop_pre_announcement"]
        + [c for c in FACTOR_COLS if c in keep.columns]
    )
    keep = keep[keep_cols].set_index("date")
    _atomic_to_parquet(keep, out_path)
    logger.info("saved %s (%d rows x %d cols)", out_path, len(keep), len(keep.columns))


if __name__ == "__main__":
    main()
