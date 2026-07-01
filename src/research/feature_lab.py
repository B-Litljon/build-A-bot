"""
Feature Research Lab — score a feature's predictive signal BEFORE building a model.

The idea: a feature is never judged against OHLCV alone, but against a *forward-
looking target*. We measure two things, the quant workhorses:

  • Information Coefficient (IC) — rank-correlation between the feature now and the
    return later. The headline number, plus its consistency over time (IC mean / std
    = the "information ratio"). A small-but-steady IC beats a big-but-flaky one.
  • Decile analysis — sort every bar by the feature, chop into 10 buckets, and check
    whether the top buckets actually move more than the bottom (a clean staircase).

Discipline baked in so we don't fool ourselves:
  • the target is strictly forward (shift(-h)); features come from the production
    pipeline, which already prevents lookahead (HTF `available_at` asof-join);
  • everything is grouped per symbol (no cross-pair bleed);
  • IC is also computed per-period then averaged (mitigates overlapping-return
    autocorrelation inflation);
  • out-of-sample is always reported separately;
  • a built-in `noise_random` control MUST score ~0 — if it doesn't, the harness lies;
  • decile spreads are shown in basis points next to a cost threshold, so a
    sub-cost "edge" is flagged dead.

Reuses the EXACT production feature code (`engineer_features_and_labels` /
`V3*Features`) so what we score == what a model would see (zero train/inference skew).

Run (from repo root, with creds loaded):
    set -a; . ./.env; set +a
    PYTHONPATH=src:. <venv-python> -m research.feature_lab --profile swing --export

Outputs:
    data/research/feature_panel_<profile>.parquet   (portable: loads in cloud Colab)
    data/research/feature_report_<profile>.csv      (ranked metrics table)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List

import numpy as np
import polars as pl
from scipy import stats

# Repo import paths (mirror backtest_london.py + retrainer expectations).
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from data.oanda_provider import OandaMarketProvider  # noqa: E402

PANEL_DIR = "data/research"


# ── profiles ──────────────────────────────────────────────────────────────────
@dataclass
class Profile:
    """A model's data recipe: which bars, which horizon, which instruments."""

    name: str
    symbols: List[str]
    timeframe_minutes: int
    htf_timeframe: str
    horizon: int          # bars ahead the forward-return target looks
    years: int


PROFILES: Dict[str, Profile] = {
    # Swing — the friendly-cost sweet spot (H1, ~6h ahead, cross-pair + metals).
    "swing": Profile(
        name="swing",
        symbols=["GBP_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY",
                 "GBP_AUD", "GBP_NZD", "XAU_USD", "XAG_USD"],
        timeframe_minutes=60,
        htf_timeframe="4h",
        horizon=6,
        years=3,
    ),
    # Scalper — fast metals (M1, ~3 bars). Cost gate is brutal here.
    "scalper": Profile(
        name="scalper",
        symbols=["XAU_USD", "XAG_USD"],
        timeframe_minutes=1,
        htf_timeframe="5m",
        horizon=3,
        years=1,
    ),
}


# ── candidate-feature registry ─────────────────────────────────────────────────
# Register a new idea as `name -> (df) -> df` adding one named column. These are
# scored exactly like production features. This is where Brandon/Gemini plug ideas.
CandidateFn = Callable[[pl.DataFrame], pl.DataFrame]
CANDIDATE_FEATURES: Dict[str, CandidateFn] = {}


def candidate(name: str) -> Callable[[CandidateFn], CandidateFn]:
    def _decorator(fn: CandidateFn) -> CandidateFn:
        CANDIDATE_FEATURES[name] = fn
        return fn
    return _decorator


@candidate("noise_random")
def _noise_random(df: pl.DataFrame) -> pl.DataFrame:
    """Null control: seeded white noise. MUST score ~0 IC / flat deciles."""
    rng = np.random.default_rng(42)
    return df.with_columns(pl.Series("noise_random", rng.standard_normal(df.height)))


@candidate("close_to_high_5")
def _close_to_high_5(df: pl.DataFrame) -> pl.DataFrame:
    """Example real candidate: where does close sit in the last-5-bar range (0..1)."""
    return df.with_columns(
        (
            (pl.col("close") - pl.col("low").rolling_min(5).over("symbol"))
            / (
                pl.col("high").rolling_max(5).over("symbol")
                - pl.col("low").rolling_min(5).over("symbol")
                + 1e-9
            )
        ).alias("close_to_high_5")
    )


# ── data + features ────────────────────────────────────────────────────────────
def load_ohlcv(profile: Profile, refresh: bool) -> pl.DataFrame:
    """Fetch the profile's instrument basket as one long OHLCV frame (+symbol)."""
    cache = f"{PANEL_DIR}/ohlcv_{profile.name}.parquet"
    if not refresh and os.path.exists(cache):
        df = pl.read_parquet(cache)
        print(f"Loaded cached OHLCV: {df.height:,} rows from {cache}")
        return df

    provider = OandaMarketProvider(environment="practice")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * profile.years + 5)
    parts = []
    for sym in profile.symbols:
        print(f"  fetching {sym} ({profile.timeframe_minutes}m) ...")
        d = provider.get_historical_bars(sym, profile.timeframe_minutes, start, end)
        if d.is_empty():
            print(f"    WARNING: no bars for {sym}, skipping")
            continue
        parts.append(d.with_columns(pl.lit(sym).alias("symbol")))
    if not parts:
        sys.exit("ERROR: OANDA returned no bars for any symbol.")
    df = pl.concat(parts, how="vertical_relaxed")
    os.makedirs(PANEL_DIR, exist_ok=True)
    df.write_parquet(cache)
    print(f"Fetched {df.height:,} rows, cached to {cache}")
    return df


def build_panel(
    ohlcv: pl.DataFrame, profile: Profile, with_labels: bool
) -> tuple[pl.DataFrame, List[str]]:
    """
    Run the production feature pipeline + targets, returning (panel, feature_names).

    with_labels=True also computes the Angel/Devil trade labels by reusing the
    retrainer's exact `engineer_features_and_labels` (heavier — pulls LightGBM).
    """
    if with_labels:
        from core.retrainer import engineer_features_and_labels, BASE_FEATURE_COLS
        df, feat_cols = engineer_features_and_labels(
            ohlcv, htf_timeframe=profile.htf_timeframe
        )
    else:
        from ml.features.v3_features import (
            V3BaseFeatures, V3HTFFeatures, V3SessionFeatures,
        )
        from core.retrainer import BASE_FEATURE_COLS
        df = ohlcv
        for gen in (V3BaseFeatures(),
                    V3HTFFeatures(timeframe=profile.htf_timeframe),
                    V3SessionFeatures()):
            df = gen.generate(df)
        feat_cols = list(BASE_FEATURE_COLS)
        df = df.drop_nulls(subset=BASE_FEATURE_COLS)

    # Forward-return target — strictly forward, per symbol (lookahead-safe).
    h = profile.horizon
    df = df.sort(["symbol", "timestamp"]).with_columns(
        (pl.col("close").shift(-h).over("symbol") / pl.col("close") - 1.0)
        .alias("fwd_return")
    )

    # Candidate features (incl. the noise control) — scored just like the rest.
    for name, fn in CANDIDATE_FEATURES.items():
        df = fn(df)
        if name not in feat_cols:
            feat_cols.append(name)

    return df, feat_cols


# ── metrics ─────────────────────────────────────────────────────────────────────
def _spearman(x: np.ndarray, y: np.ndarray, min_n: int = 30) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < min_n:
        return float("nan")
    r, _ = stats.spearmanr(x[mask], y[mask])
    return float(r)


def _period_ic(df: pl.DataFrame, feat: str, tgt: str, every: str = "1d") -> np.ndarray:
    """IC computed within each period, returned as a series (for IR / stability)."""
    d = (
        df.select(["timestamp", feat, tgt])
        .drop_nulls()
        .with_columns(pl.col("timestamp").dt.truncate(every).alias("_p"))
    )
    ics: List[float] = []
    for _, g in d.group_by("_p", maintain_order=True):
        if g.height >= 20:
            r = _spearman(g[feat].to_numpy(), g[tgt].to_numpy(), min_n=20)
            if not np.isnan(r):
                ics.append(r)
    return np.array(ics)


def _decile_spread_bps(df: pl.DataFrame, feat: str, tgt: str, q: int = 10) -> float:
    """Top-minus-bottom decile mean forward return, in basis points."""
    d = df.select([feat, tgt]).drop_nulls()
    if d.height < q * 20:
        return float("nan")
    try:
        d = d.with_columns(
            pl.col(feat).qcut(q, labels=[str(i) for i in range(q)],
                              allow_duplicates=True).alias("_b")
        )
    except Exception:
        return float("nan")
    g = d.group_by("_b").agg(pl.col(tgt).mean().alias("m"))
    g = g.with_columns(pl.col("_b").cast(pl.Int32)).sort("_b")
    if g.height < 2:
        return float("nan")
    return float((g["m"][-1] - g["m"][0]) * 1e4)


def feature_report(
    df: pl.DataFrame,
    feat_cols: List[str],
    split: datetime,
    cost_bps: float,
    score_labels: bool,
) -> pl.DataFrame:
    """Rank every feature by its out-of-sample information ratio vs forward return."""
    in_mask = df.filter(pl.col("timestamp") < split)
    oos_mask = df.filter(pl.col("timestamp") >= split)

    # Redundancy: |corr| with the most-correlated OTHER feature (native polars,
    # no pandas round-trip). polars .corr() is positional: column i and row i
    # both map to present[i], so masking the diagonal drops self-correlation.
    present = [c for c in feat_cols if c in df.columns]
    max_abs_corr: Dict[str, float] = {}
    if len(present) > 1:
        cm = df.select(present).drop_nulls().corr().to_numpy()
        np.fill_diagonal(cm, np.nan)
        cm = np.abs(cm)
        for i, f in enumerate(present):
            col = cm[:, i]
            max_abs_corr[f] = (
                float(np.nanmax(col)) if np.isfinite(col).any() else float("nan")
            )

    rows = []
    for f in feat_cols:
        if f not in df.columns:
            continue
        ic_full = _spearman(df[f].to_numpy(), df["fwd_return"].to_numpy())
        ic_oos = _spearman(oos_mask[f].to_numpy(), oos_mask["fwd_return"].to_numpy())
        ics = _period_ic(in_mask, f, "fwd_return")
        ir = float(ics.mean() / ics.std()) if ics.size > 5 and ics.std() > 0 else float("nan")
        hit = float((np.sign(ics) == np.sign(ics.mean())).mean()) if ics.size else float("nan")
        spread = _decile_spread_bps(df, f, "fwd_return")
        max_corr = max_abs_corr.get(f, float("nan"))
        row = {
            "feature": f,
            "ic_fwd": round(ic_full, 4),
            "ic_oos": round(ic_oos, 4),
            "ic_ir_is": round(ir, 3),
            "hit_rate": round(hit, 3),
            "decile_bps": round(spread, 2),
            "beats_cost": (abs(spread) > cost_bps) if not np.isnan(spread) else False,
            "max_abs_corr": round(max_corr, 3),
        }
        if score_labels and "angel_target" in df.columns:
            row["ic_angel"] = round(_spearman(df[f].to_numpy(),
                                               df["angel_target"].to_numpy()), 4)
            row["ic_devil"] = round(_spearman(df[f].to_numpy(),
                                               df["devil_target"].to_numpy()), 4)
        rows.append(row)

    rep = pl.DataFrame(rows)
    return rep.sort(pl.col("ic_ir_is").abs(), descending=True, nulls_last=True)


def yearly_stability(df: pl.DataFrame, feat: str) -> pl.DataFrame:
    """Headline IC of one feature, year by year (sign flips = fragile)."""
    d = df.select(["timestamp", feat, "fwd_return"]).drop_nulls()
    d = d.with_columns(pl.col("timestamp").dt.year().alias("year"))
    out = []
    for _, g in d.group_by("year", maintain_order=True):
        out.append({"year": int(g["year"][0]),
                    "ic": round(_spearman(g[feat].to_numpy(),
                                          g["fwd_return"].to_numpy()), 4),
                    "n": g.height})
    return pl.DataFrame(out).sort("year")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Feature Research Lab")
    ap.add_argument("--profile", default="swing", choices=list(PROFILES))
    ap.add_argument("--oos-months", type=int, default=12)
    ap.add_argument("--cost-bps", type=float, default=2.0,
                    help="Cost threshold for decile spreads (basis points)")
    ap.add_argument("--targets", default="both", choices=["both", "return"],
                    help="'both' also scores Angel/Devil labels (heavier)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--export", action="store_true",
                    help="Write the portable feature panel + report to data/research/")
    args = ap.parse_args()

    profile = PROFILES[args.profile]
    score_labels = args.targets == "both"
    print(f"\n=== Feature Lab — profile '{profile.name}' "
          f"({profile.timeframe_minutes}m bars, {profile.horizon}-bar horizon) ===")
    print(f"basket: {', '.join(profile.symbols)}\n")

    ohlcv = load_ohlcv(profile, args.refresh)
    panel, feat_cols = build_panel(ohlcv, profile, with_labels=score_labels)

    span_end = panel["timestamp"].max()
    split = span_end - timedelta(days=30 * args.oos_months)
    print(f"\nPanel: {panel.height:,} rows, {panel['timestamp'].min().date()} → "
          f"{span_end.date()}  |  OOS split {split.date()}  |  {len(feat_cols)} features\n")

    rep = feature_report(panel, feat_cols, split, args.cost_bps, score_labels)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=30):
        print(rep)

    print("\nReminder: 'noise_random' is the control — it should sit near IC 0 with a "
          "flat decile spread. Real signal must clearly beat it AND survive OOS + cost.")

    if args.export:
        os.makedirs(PANEL_DIR, exist_ok=True)
        panel_path = f"{PANEL_DIR}/feature_panel_{profile.name}.parquet"
        rep_path = f"{PANEL_DIR}/feature_report_{profile.name}.csv"
        keep = ["timestamp", "symbol", "open", "high", "low", "close", "volume",
                "fwd_return"] + [c for c in feat_cols if c in panel.columns]
        for lbl in ("angel_target", "devil_target"):
            if lbl in panel.columns:
                keep.append(lbl)
        panel.select([c for c in keep if c in panel.columns]).write_parquet(panel_path)
        rep.write_csv(rep_path)
        print(f"\nExported portable panel → {panel_path}")
        print(f"Exported report        → {rep_path}")
        print("(Load the panel in cloud Colab with plain pandas — no repo/talib needed.)")


if __name__ == "__main__":
    main()
