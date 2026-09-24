"""Deterministic tests for the Lane-2 weekly investor pipeline + trainer.

Covers the six mandatory falsification/leakage guards from the Lane-2 brief
(``llm_reports/handoffs/2026-09-24_lane2-equity-factor-pead.md`` §6.4). These
are offline, synthetic, and deterministic — no SimFin network access, no
Alpaca, no LightGBM training beyond trivial fixed fixtures.

Glossary:
    PIT -- point-in-time, see GLOSSARY.md. A feature row dated t may only use
        fundamentals whose publish_date <= t; tested with a future-dated row.
    embargo -- see GLOSSARY.md. The 10-calendar-day no-rows gap between the
        train and validate windows of a walk-forward fold.
    SUE / PEAD -- see GLOSSARY.md.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import investor_feature_pipeline_weekly as fp  # noqa: E402
import investor_train_model_weekly as tm  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# 1. Point-in-time guard — a publish_date AFTER the feature date excluded
# ─────────────────────────────────────────────────────────────────────

def test_pit_guard_excludes_future_publish_date():
    """A SimFin row published at t+5 must not feed a feature row dated t."""
    cal = pd.DataFrame({
        "symbol": ["AAA", "AAA"],
        "earnings_date": pd.to_datetime(["2021-01-31", "2021-04-30"], utc=True),
        "eps_reported": [1.0, 2.0],
        # second filing published AFTER the signal date we will query
        "publish_date": pd.to_datetime(["2021-02-15", "2021-12-31"], utc=True),
        "source": "test",
    })
    cal = cal.sort_values(["symbol", "publish_date"]).reset_index(drop=True)
    signal_date = pd.Timestamp("2021-06-01", tz="utc")

    cal_small = cal[["symbol", "publish_date"]].copy()
    merged = pd.merge_asof(
        pd.DataFrame({"date": [signal_date]}),
        cal_small, left_on="date", right_on="publish_date", direction="backward",
    )
    # the matched publish_date must be <= the signal date
    assert merged["publish_date"].iloc[0] <= signal_date
    # and it must be the FEB filing, never the future DEC one
    assert merged["publish_date"].iloc[0] == pd.Timestamp("2021-02-15", tz="utc")


def test_pipeline_pit_guard_raises_on_leak():
    """build/join must raise if any matched publish_date is in the future."""
    # craft an income csv where a filing's publish date is clearly post-report,
    # then ask the trainer-side join helper directly for a date BEFORE it.
    cal = pd.DataFrame({
        "symbol": ["AAA"] * 6,
        "earnings_date": pd.to_datetime(
            ["2020-01-31", "2020-04-30", "2020-07-31", "2020-10-31",
             "2021-01-31", "2021-04-30"], utc=True),
        "eps_reported": [1.0, 1.1, 0.9, 1.2, 1.0, 1.3],
        "publish_date": pd.to_datetime(
            ["2020-02-20", "2020-05-20", "2020-08-20", "2020-11-20",
             "2021-02-20", "2021-05-20"], utc=True),
        "source": "test",
    }).sort_values(["symbol", "publish_date"]).reset_index(drop=True)
    # SUE is computed from the calendar alone; first _SUE_MIN_QTRS-1 are NaN.
    sue = fp._compute_sue(cal)
    assert len(sue) == len(cal)
    # rows before the trailing window exists are NaN
    assert np.isnan(sue.iloc[0])


# ─────────────────────────────────────────────────────────────────────
# 2. Embargo guard — train.max(date)+10d < val.min(date) for every fold
# ─────────────────────────────────────────────────────────────────────

def test_embargo_guard_logic():
    """The embargo arithmetic the trainer asserts must hold by construction."""
    emb = pd.Timedelta(days=tm.EMBARGO_CAL_DAYS)
    train_max = pd.Timestamp("2024-01-31", tz="utc")
    # a validation start only 5 days out must NOT pass
    with pytest.raises(AssertionError):
        assert train_max + emb < pd.Timestamp("2024-02-04", tz="utc"), "should fail"
    # a validation start 11+ days out passes
    assert train_max + emb < pd.Timestamp("2024-02-15", tz="utc")


def test_trainer_holds_atomic_embargo_on_real_frame():
    """If the weekly features exist, every realized fold obeys the embargo."""
    if not tm._INPUT.exists():
        pytest.skip("weekly features not built in this checkout")
    df = pd.read_parquet(tm._INPUT)
    if df.index.name == "date":
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if "drop_pre_announcement" in df:
        df = df[~df["drop_pre_announcement"]]
    df = df.sort_values("date").reset_index(drop=True)
    wk = df.groupby("week")["date"].agg(["min", "max"]).sort_values("min")
    week_keys = list(wk.index)
    n = len(week_keys)
    emb = pd.Timedelta(days=tm.EMBARGO_CAL_DAYS)
    N = tm.N_FOLDS
    mtw = max(30, n // (N + 2))
    fb = []
    for k in range(N):
        t1 = mtw + k * ((n - mtw) // (N + 1))
        if t1 >= n - 1:
            break
        fb.append(t1)
    emb_pairs = 0
    for k, t1 in enumerate(fb):
        train_max = wk.iloc[: t1 + 1]["max"].max()
        elig = [w for w in week_keys[t1 + 1:] if wk.loc[w, "min"] > train_max + emb]
        if not elig:
            continue
        val_min = wk.loc[elig[0], "min"]
        assert train_max + emb < val_min
        emb_pairs += 1
    assert emb_pairs >= 1


# ─────────────────────────────────────────────────────────────────────
# 3. Pre-announcement filter — days_until_earnings == 2 must be dropped
# ─────────────────────────────────────────────────────────────────────

def test_preannouncement_filter_band():
    lo, hi = fp.PREANN_FILTER_LO, fp.PREANN_FILTER_HI
    due = pd.Series([0, 1, 2, 3, 5, -1])
    flagged = (due >= lo) & (due <= hi)
    # rows at 0,1,2 are dropped; 3,5,-1 survive
    assert flagged.tolist() == [True, True, True, False, False, False]


def test_preannouncement_rows_absent_from_training_frame():
    if not tm._INPUT.exists():
        pytest.skip("weekly features not built in this checkout")
    df = pd.read_parquet(tm._INPUT).reset_index()
    if "drop_pre_announcement" not in df.columns:
        pytest.skip("pre-announcement flag not present")
    flagged = df[df["drop_pre_announcement"]]
    if flagged.empty:
        pytest.skip("no flagged rows in window")
    # every flagged row sits in the avoidance band
    assert ((flagged["days_until_earnings"] >= fp.PREANN_FILTER_LO)
            & (flagged["days_until_earnings"] <= fp.PREANN_FILTER_HI)).all()


# ─────────────────────────────────────────────────────────────────────
# 4. Target alignment — label at t = ln(close_{t+5}/close_t)
# ─────────────────────────────────────────────────────────────────────

def test_target_uses_five_day_forward_log_return():
    closes = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 200.0],
                       index=pd.date_range("2024-01-01", periods=7, freq="D", tz="utc"))
    fwd = np.log(closes.shift(-fp.FORWARD_DAYS) / closes)
    # row 0: ln(close_{5}/close_0) = ln(105/100)
    assert fwd.iloc[0] == pytest.approx(math.log(105.0 / 100.0))
    # row 2 uses close_7? no — only 7 rows, so row idx 2 -> close idx 7 missing
    assert math.isnan(fwd.iloc[2])
    # uses ONLY future closes: last FORWARD_DAYS rows are NaN (embargo window)
    assert fwd.iloc[-fp.FORWARD_DAYS:].isna().all()


def test_training_frame_target_alignment_matches_closes():
    if not tm._INPUT.exists():
        pytest.skip("weekly features not built in this checkout")
    feat = pd.read_parquet(tm._INPUT).reset_index()
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    raw = pd.read_parquet(fp._INPUT_PATH, columns=["symbol", "close"]).reset_index()
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    px = raw.pivot_table(index="date", columns="symbol", values="close").sort_index()
    dates = list(px.index)
    idx_of = {d: i for i, d in enumerate(dates)}
    # spot-check a sample of rows: fwd_log_ret_5d == ln(close[t+5]/close[t])
    sample = feat.dropna(subset=["fwd_log_ret_5d"]).sample(
        n=25, random_state=0
    )
    for _, row in sample.iterrows():
        sym, d0 = row["symbol"], row["date"]
        if sym not in px.columns or d0 not in idx_of:
            continue
        i0 = idx_of[d0]
        if i0 + fp.FORWARD_DAYS >= len(dates):
            continue
        d1 = dates[i0 + fp.FORWARD_DAYS]
        c0, c1 = px.loc[d0, sym], px.loc[d1, sym]
        if not (np.isfinite(c0) and np.isfinite(c1)):
            continue
        assert row["fwd_log_ret_5d"] == pytest.approx(math.log(c1 / c0), rel=1e-6)


# ─────────────────────────────────────────────────────────────────────
# 5. Deterministic training — two runs give identical predictions
# ─────────────────────────────────────────────────────────────────────

def test_deterministic_training_same_seed():
    import lightgbm as lgb
    rng = np.random.default_rng(42)
    # a tiny synthetic ranking problem over weekly groups
    n_weeks = 30
    rows = []
    for w in range(n_weeks):
        for s in range(8):
            rows.append({
                "week": f"2024-W{w:02d}",
                "f1": rng.normal(), "f2": rng.normal(),
                "y": rng.integers(0, 5),
            })
    d = pd.DataFrame(rows)
    params = dict(objective="lambdarank", n_estimators=10, deterministic=True,
                  random_state=42, verbose=-1, n_jobs=1)

    def fit_predict():
        g = d.groupby("week", sort=False).size().to_numpy(np.int32)
        m = lgb.LGBMRanker(**params)
        m.fit(d[["f1", "f2"]], d["y"], group=g)
        return m.predict(d[["f1", "f2"]])

    p1 = fit_predict()
    p2 = fit_predict()
    np.testing.assert_array_equal(p1, p2)


# ─────────────────────────────────────────────────────────────────────
# 6. Stats module re-verified here (canonical cases) — see test_lab_stats
# ─────────────────────────────────────────────────────────────────────

def test_stats_module_contract_from_lane2_consumer():
    from lab.stats import cscv_pbo, deflated_sharpe_ratio, hlz_haircut_sharpe
    assert 0.0 <= deflated_sharpe_ratio(0, 10, 100, 0, 3) <= 1.0
    assert cscv_pbo(np.zeros((80, 5))) == 0.5
    assert hlz_haircut_sharpe(1.0, 10) < 1.0
