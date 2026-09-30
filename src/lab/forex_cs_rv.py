"""Lane 3 — cross-sectional relative-value (RV) forex basket [falsification].

Single-instrument directional bracket trading on M15 spot forex is measurably
closed (recons/2026-09-14, 2026-09-20, 2026-09-21, 2026-09-23): the market is
a driftless martingale at the served 2.0x/4.0x geometry, and the binary
survival label's ~25% base rate compresses the output distribution until the
0.3833 bar approves 0.028% of bars. This module replaces that bet with a
cross-sectional relative-value bet — "will currency X's index outperform the
basket median over horizon tau" — which by construction is a 50/50 label and
therefore cannot compress.

What ships here:

    * index math: geometric-mean currency index I_{c,t} = (prod S)^(1/K) over
      a CLOSED pair universe (every currency appears in >= 2 pairs), quote-side
      appearances inverted; the 6-pair fiat basket closes with 5 currencies
      (AUD/EUR/GBP/JPY/NZD).
    * the RV label: y_{c,t} = I(dL_{c,t->t+tau} > cross-sectional median at t),
      exactly ceil(K/2) positives per timestamp (ties broken deterministically
      by currency code). tau defaults to 4 bars (1h) and counts in the trials
      budget.
    * the "cs_rv" feature family (version 1): per-currency index features
      attached to that currency's CARRIER PAIR row (the pair whose quote side
      most directly prices the currency against the funding leg; see
      CARRIER_CURRENCY).
    * build_csrv_frame / run_csrv_gate / run_csrv_experiment: an honest
      one-model-per-currency walk-forward gate on the rv_target label, using
      the retrainer's chronological-permutation OOF calibration discipline
      (the 2026-09-09 leakage fix) and a Clopper-Pearson PF lower bound in
      the gate's own one-sided-90% spirit.

Why the retrainer's two-stage gate is NOT reused verbatim: its
validate_candidate hardcodes angel_target / devil_target[_macro] — the
single-instrument labels this lane exists to replace
(core/retrainer/_gate.py:742). The lane therefore drives the SAME fold
schedule, estimator family, hyperparameters and threshold-calibration
discipline on the RV label in this module, and the recon report says so.
build_frame IS used for the veto/tail-purge path (the production chop veto
must see this frame), but run_gate/run_model_backtest are not, for the same
reason. The bracket EV conversion of the RV edge uses the brief's stated
1 pp = 0.03R at 2:1 and subtracts the measured 0.25R toll.

Glossary:
    currency index -- I_{c,t}, geometric mean of the pair prices carrying
        currency c (quote side inverted); see GLOSSARY.md "currency index".
    cross-sectional relative value (CS-RV) -- betting a currency outperforms
        the basket median over tau, not that a pair rises in absolute terms.
    funding leg -- the currency a basket's RV is priced against (USD once it
        is isolable; implicitly JPY/GBP on the closed 6-pair universe).
    50%-balanced label -- exactly ceil(K/2) positives per timestamp by
        construction, immune to the base-rate compression that zeroed the
        production Angel.
    closed pair universe -- every currency in the pair list appears in >= 2
        pairs, so every index has >= 2 constituents and currency strengths
        net out internally.
    carrier pair -- the pair row a currency's index features and label attach
        to (frame rows are pairs; the RV signal is per currency). Mapping at
        CARRIER_CURRENCY.
    tau -- RV label horizon in bars; TAU_BARS=4 (1 hour on M15), a priori,
        deliberately not swept (a sweep would multiply the trial count).
    rv_target -- the Float64 label column (1/0/null) written into carrier rows
        by CsRVLabelGenerator before build_frame's veto path. Never a feature.
    CsRVIndexFeatures -- causal per-currency index momentum/vol/RSI/z-score
        features; CsRVRelativeFeatures -- cross-sectional within-timestamp
        rank/z of that momentum. Both attach via _attach_carrier coalescing.
    CsRVLabelGenerator -- writes rv_target; build_csrv_frame calls it on the
        stacked raw bars BEFORE build_frame applies labels, chop veto, purge.
    run_csrv_gate -- the lane's gate: date-based expanding folds in the
        retrainer's proportions, one LightGBM per currency with the
        retrainer's forex angel hyperparameters, per-fold train-only EV-max
        threshold from chronological OOF probabilities.
    RVGateResult -- pooled signals/wins, edge_pp vs the 0.50 base, the pp->R
        conversion (R_PER_PP_2_TO_1) and the toll subtraction in one place.
    TOLL_R -- 0.25R, the measured per-trade toll at 2.0x/4.0x M15
        (recons/2026-09-14); edge_R must clear EDGE_R_GATE = +0.10R net of it.
    N_TRIALS -- 4: this gate, the a-priori tau=4 choice, the 2026-09-23
        CatBoost A/B, and the USD-cross availability probe. Reported so the
        reconstructed DSR/HLZ (and Lane 5's audit) do not under-deflate.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from ml.core.interfaces import BaseFeatureGenerator

logger = logging.getLogger(__name__)

TAU_BARS: int = 4  # RV horizon: 1 hour of M15 bars. A priori, not swept.

# Measured toll per trade at the served 2.0x/4.0x M15 geometry
# (recons/2026-09-14_session-evidence-and-options.md); the falsification bar
# is edge_R net of it.
TOLL_R: float = 0.25
EDGE_R_GATE: float = 0.10
# Brief section-6.1 conversion, verbatim: at the served 2:1 geometry
# 1 pp of win rate ~= 0.03R.
R_PER_PP_2_TO_1: float = 0.03
# Measured random-entry win rate at that geometry (same recon). Informational
# only — the RV label's own base is 0.50 by construction.
BASE_RATE_2_TO_1: float = 0.2540

EPS = 1e-12

# The seven crosses that would isolate the USD funding leg (OANDA instrument
# names). USD crosses are absent from the local cache; fetch is opportunistic.
USD_CROSSES: Tuple[str, ...] = (
    "EUR_USD",
    "GBP_USD",
    "AUD_USD",
    "NZD_USD",
    "USD_JPY",
    "USD_CHF",
    "USD_CAD",
)

# pair -> the currency whose RV signal that row carries. On the 6-pair
# baseline EUR is un-indexable (1 leg, see _check_closed's NOTE), so EUR_JPY's
# carrier slot serves JPY (its quote side) and GBP_AUD/GBP_NZD carry no RV
# label (their currencies are already carried by the JPY crosses). With the
# USD crosses present, USD_JPY serves USD and EUR_USD serves EUR — carriers
# are claimed in dict order, first-come-first-served, so EUR takes EUR_USD
# before GBP_USD-style conflicts arise. Changing the mapping is a documented
# new trial, not a rerun.
CARRIER_CURRENCY: Dict[str, str] = {
    "AUD_JPY": "AUD",
    "EUR_JPY": "JPY",
    "GBP_JPY": "GBP",
    "NZD_JPY": "NZD",
    "USD_JPY": "USD",
    "EUR_USD": "EUR",
    # The remaining crosses (GBP_USD / AUD_USD / NZD_USD / USD_CHF / USD_CAD)
    # stay uncarried: their indexable currencies are already served above, and
    # CHF/CAD are single-leg on the combined universe.
}

DEFAULT_CACHE_DIR = Path("analysis_cache/strategy_matrix")
USD_CROSS_CACHE = DEFAULT_CACHE_DIR / "USD_CROSS_M15.parquet"

# Model-facing cs_rv columns (registry-declared; both generators' union).
RV_FEATURES: Tuple[str, ...] = (
    "rv_mom_4",
    "rv_mom_16",
    "rv_vol_16",
    "rv_rsi_14",
    "rv_z_mom_16",
    "rv_rel_mom4_rank",
    "rv_rel_mom4_z",
)

# The RV label column name. Defined here (not at line ~N after the generators)
# because run_csrv_gate is below and Python resolves the name at CALL time —
# but the module must define it before any caller executes.
CS_RV_LABEL_COL = "rv_target"

N_TRIALS: int = 4  # see module docstring "trials budget"


# ═══════════════════════════════════════════════════════════════════════════
# Pair-universe closure and index construction (pure math, no I/O)
# ═══════════════════════════════════════════════════════════════════════════


def _split_pair(symbol: str) -> Tuple[str, str]:
    """'GBP_JPY' -> ('GBP', 'JPY'). Raises on anything else."""
    parts = symbol.strip().upper().split("_")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"not a BASE_QUOTE pair symbol: {symbol!r}")
    return parts[0], parts[1]


def _check_closed(pairs: Sequence[str]) -> set:
    """Return the currencies of a pair list that have >= 2 legs (the only ones
    a geometric index can be built from), raising ONLY when fewer than 3
    currencies qualify (a 2-currency cross-sectional median is degenerate: it
    is the midpoint of two numbers, so the label is coin-flip noise).

    NOTE — the 6-pair fiat basket is NOT fully closed: EUR appears in exactly
    one pair (EUR_JPY). The brief's claim "closes with 5 currencies
    (AUD/EUR/GBP/JPY/NZD)" mis-counts EUR. VERIFIED 2026-09-24 by direct count:
    {JPY:4, GBP:3, AUD:2, NZD:2, EUR:1}. On the baseline the currency universe
    is therefore {AUD, GBP, JPY, NZD}; EUR_JPY rows stay in the FRAME (they
    carry v3_base features and the JPY carrier's rv signal) but EUR itself has
    no index until/unless a EUR_USD cross is fetched. Documented in the recon.
    """
    counts: Dict[str, int] = {}
    for pair in pairs:
        for ccy in _split_pair(pair):
            counts[ccy] = counts.get(ccy, 0) + 1
    ok = {c for c, n in counts.items() if n >= 2}
    if len(ok) < 3:
        under = sorted(c for c, n in counts.items() if n < 2)
        raise ValueError(
            f"pair universe {sorted(pairs)} leaves {len(ok)} indexable "
            f"currencies ({sorted(ok)}); single-leg currencies {under}; a "
            "cross-sectional RV basket needs >= 3 indexable currencies"
        )
    return ok


def currency_universe(pairs: Sequence[str]) -> Tuple[str, ...]:
    """Sorted currency codes of a CLOSED pair list (closure enforced)."""
    return tuple(sorted(_check_closed(pairs)))


def close_price_matrix(
    bars_by_symbol: Dict[str, pl.DataFrame],
    pairs: Sequence[str],
) -> pl.DataFrame:
    """Pivot closes to one row per timestamp x pair on the union calendar.

    Each pair's close is forward-filled by at most ONE bar (a single missing
    M15 bar is a feed gap, not a regime); anything deeper stays null and voids
    the affected cross-sections at the label step. Never backward-filled.
    """
    frames = []
    for pair in pairs:
        if pair not in bars_by_symbol:
            raise ValueError(f"no bars for pair {pair!r} in the loaded basket")
        df = bars_by_symbol[pair]
        if "timestamp" not in df.columns or "close" not in df.columns:
            raise ValueError(f"bars for {pair} lack timestamp/close columns")
        frames.append(
            df.select("timestamp", pl.col("close").alias(pair)).sort("timestamp")
        )
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="timestamp", how="full", coalesce=True)
    return out.sort("timestamp").with_columns(
        [pl.col(p).forward_fill(limit=1) for p in pairs]
    )


def currency_indexes(closes: pl.DataFrame, pairs: Sequence[str]) -> pl.DataFrame:
    """timestamp + one I_c column per currency of the closed pair universe.

    I_c = (prod over pairs carrying c of S or 1/S)^(1/K_c); quote-side legs
    are inverted so the index is always "strength of c".
    """
    _check_closed(pairs)
    exprs = []
    for ccy in currency_universe(pairs):
        terms = []
        for pair in pairs:
            base, quote = _split_pair(pair)
            if ccy == base:
                terms.append(pl.col(pair))
            elif ccy == quote:
                terms.append(1.0 / pl.col(pair))
        prod = terms[0]
        for t in terms[1:]:
            prod = prod * t
        exprs.append((prod ** (1.0 / len(terms))).alias(ccy))
    return closes.select("timestamp", *exprs)


def index_log_changes(
    indexes: pl.DataFrame,
    currencies: Sequence[str],
    *,
    tau: int = TAU_BARS,
) -> pl.DataFrame:
    """Forward tau-bar log change of every index: dL_c = ln I_c(t+tau) - ln I_c(t).

    The forward shift is the ONLY lookahead in the module and it lives in the
    LABEL path only; feature generators never call this. Rows without a full
    tau-ahead bar (natural tail, interior gaps) are null.
    """
    exprs = []
    for ccy in currencies:
        log_idx = pl.col(ccy).log()
        exprs.append((log_idx.shift(-tau) - log_idx).alias(f"dl_{ccy}"))
    return indexes.select("timestamp", *exprs)


def rv_labels(dl: pl.DataFrame, currencies: Sequence[str]) -> pl.DataFrame:
    """y_c = I(dL_c > median cross-section) per timestamp.

    EXACTLY ceil(K/2) positives per timestamp by construction. Ties at the
    median fill remaining positive slots in ascending currency-code order, so
    the labeller is deterministic; a void cross-section (any null leg) has all
    labels null, never guessed.
    """
    cols = [f"dl_{c}" for c in currencies]
    ordered = list(currencies)
    n = len(ordered)
    n_pos = (n + 1) // 2  # ceil(K/2)

    # keep the original timestamp dtype by carrying the polars column, not the
    # numpy view (which would round-trip datetimes to float on some versions)
    ts_col = dl["timestamp"]
    vals = dl.select(cols).to_numpy().astype(float)

    out = np.full((len(dl), n), np.nan)
    for i in range(vals.shape[0]):
        row = vals[i]
        if not np.isfinite(row).all():
            continue
        med = float(np.median(row))
        chosen = [j for j in range(n) if row[j] > med + EPS]
        if len(chosen) < n_pos:
            ties = sorted(
                (j for j in range(n) if abs(row[j] - med) <= EPS and j not in chosen),
                key=lambda j: ordered[j],
            )
            chosen.extend(ties[: max(0, n_pos - len(chosen))])
        if len(chosen) < n_pos:
            # degenerate case: fewer strict-than-median + ties than n_pos
            # (can only happen with duplicated values just BELOW the median on
            # an even K). Fill by descending value, code-ascending on ties.
            rest = sorted(
                (j for j in range(n) if j not in chosen),
                key=lambda j: (-row[j], ordered[j]),
            )
            chosen.extend(rest[: max(0, n_pos - len(chosen))])
        keep = set(chosen[:n_pos])
        for j in range(n):
            out[i, j] = 1.0 if j in keep else 0.0

    data: Dict[str, object] = {}
    out_df = pl.DataFrame({"timestamp": ts_col})
    for j, ccy in enumerate(ordered):
        out_df = out_df.with_columns(pl.Series(f"y_{ccy}", out[:, j]))
    return out_df


# ═══════════════════════════════════════════════════════════════════════════
# Carrier plumbing (pair rows <-> currency signals)
# ═══════════════════════════════════════════════════════════════════════════


def _carrier_map(df: pl.DataFrame) -> Dict[str, str]:
    """currency -> carrier pair symbol, restricted to pairs present in df.

    Production pairs use CARRIER_CURRENCY verbatim. For anything else
    (synthetic universes in tests), each indexable currency deterministically
    rides its lexicographically smallest pair, first-come-first-served so two
    currencies never share one carrier. The override table is the PRODUCTION
    contract; the fallback exists so the math is testable off-basket.
    """
    pairs = sorted(df["symbol"].unique().to_list())
    indexable = _check_closed(pairs)
    out: Dict[str, str] = {}
    for pair, ccy in CARRIER_CURRENCY.items():
        if pair in pairs and ccy in indexable and ccy not in out:
            out[ccy] = pair
    # fallback assignment for currencies the override table does not cover
    used = set(out.values())
    for ccy in sorted(indexable):
        if ccy in out:
            continue
        for pair in pairs:
            if pair in used:
                continue
            base, quote = _split_pair(pair)
            if ccy == base or ccy == quote:
                out[ccy] = pair
                used.add(pair)
                break
    return out


def _attach_carrier(df: pl.DataFrame) -> pl.DataFrame:
    """Adds ``carrier_ccy`` = the currency each pair row carries (null for
    uncarried pairs; those rows are excluded from the per-currency splits)."""
    cmap = _carrier_map(df)
    pair_to_ccy = {pair: ccy for ccy, pair in cmap.items()}
    return df.with_columns(
        pl.col("symbol").replace_strict(pair_to_ccy, default=None).alias("carrier_ccy")
    )


def _basket_index_frame(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """Stacked pair frame -> (currency index matrix, currency list)."""
    pairs = sorted(df["symbol"].unique().to_list())
    _check_closed(pairs)
    closes = close_price_matrix(
        {p: df.filter(pl.col("symbol") == p) for p in pairs}, pairs
    )
    indexes = currency_indexes(closes, pairs)
    return indexes, list(currency_universe(pairs))


def _coalesce_carriers(
    df: pl.DataFrame,
    per_ccy_frame: pl.DataFrame,
    *,
    suffix_to_name: Dict[str, str],
    prefix_fmt: str,
) -> pl.DataFrame:
    """Join ``per_ccy_frame`` (columns f"{prefix_fmt}{ccy}") onto df by
    timestamp, then collapse the suffixed columns into the model-facing names:
    a row carries its own currency's value, other currencies' columns stay
    null and are dropped."""
    cmap = _carrier_map(df)
    out = df.join(per_ccy_frame, on="timestamp", how="left")
    for suffix, name in suffix_to_name.items():
        exprs = []
        for ccy, carrier in cmap.items():
            col = f"{prefix_fmt}{ccy}"
            if col not in out.columns:
                continue
            exprs.append(
                pl.when(pl.col("symbol") == carrier)
                .then(pl.col(col))
                .otherwise(None)
            )
        if not exprs:
            continue
        combined = exprs[0]
        for e in exprs[1:]:
            combined = pl.coalesce([combined, e])
        out = out.with_columns(combined.alias(name))
    drop = [
        c
        for c in out.columns
        if c.startswith(prefix_fmt) and c not in suffix_to_name.values()
    ]
    return out.drop(drop)


# ═══════════════════════════════════════════════════════════════════════════
# Feature generators + label generator (the "cs_rv" family)
# ═══════════════════════════════════════════════════════════════════════════


class CsRVIndexFeatures(BaseFeatureGenerator):
    """Per-currency index momentum/vol/RSI/z-score, on carrier-pair rows.

    rv_mom_4 / rv_mom_16 -- L_c(t) - L_c(t-4 / t-16); index log momentum.
    rv_vol_16 -- rolling 16-bar std of the index's 1-bar log change.
    rv_rsi_14 -- RSI-14 on the log index level.
    rv_z_mom_16 -- cross-sectional z-score of the 16-bar momentum within the
        same timestamp (never across time).

    All windows are trailing and end at the current bar — causal by
    construction (no shift(-k) anywhere in this class).
    """

    feature_cols = (
        "rv_mom_4",
        "rv_mom_16",
        "rv_vol_16",
        "rv_rsi_14",
        "rv_z_mom_16",
    )

    # NOTE: feature generation re-derives L per call, so in a stacked pipeline
    # (v3_base first) this generator sees feature columns too — it only reads
    # symbol/timestamp/close, which are positionally stable.

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        if "carrier_ccy" not in df.columns:
            df = _attach_carrier(df)
        indexes, currencies = _basket_index_frame(df)
        L = indexes.select(
            "timestamp", *[pl.col(c).log().alias(c) for c in currencies]
        ).sort("timestamp")

        d_exprs = []
        f_exprs = []
        for c in currencies:
            lc = pl.col(c)
            d1 = lc - lc.shift(1)
            d_exprs.append(d1.alias(f"__d1_{c}"))
            f_exprs.extend(
                [
                    (lc - lc.shift(4)).alias(f"rvf_mom4_{c}"),
                    (lc - lc.shift(16)).alias(f"rvf_mom16_{c}"),
                ]
            )
        feat = L.with_columns(d_exprs).with_columns(f_exprs)
        for c in currencies:
            d1 = pl.col(f"__d1_{c}")
            feat = feat.with_columns(
                d1.rolling_std(16, min_samples=8).alias(f"rvf_vol16_{c}"),
                (
                    100.0
                    - 100.0
                    / (
                        1.0
                        + pl.when(d1 > 0)
                        .then(d1)
                        .otherwise(0.0)
                        .rolling_mean(14, min_samples=7)
                        / (
                            pl.when(d1 < 0).then(-d1).otherwise(0.0).rolling_mean(
                                14, min_samples=7
                            )
                            + EPS
                        )
                    )
                ).alias(f"rvf_rsi14_{c}"),
            )
        mom_cols = [pl.col(f"rvf_mom16_{c}") for c in currencies]
        feat = feat.with_columns(
            pl.mean_horizontal(mom_cols).alias("__mom16_mean"),
            pl.concat_list(mom_cols).list.std().alias("__mom16_std"),
        )
        for c in currencies:
            feat = feat.with_columns(
                (
                    (pl.col(f"rvf_mom16_{c}") - pl.col("__mom16_mean"))
                    / (pl.col("__mom16_std") + EPS)
                ).alias(f"rvf_z16_{c}")
            )

        keep = ["timestamp"] + [
            c for c in feat.columns if c.startswith("rvf_")
        ]
        per_ccy = feat.select(keep)

        out = df
        for suffix, name in (
            ("mom4", "rv_mom_4"),
            ("mom16", "rv_mom_16"),
            ("vol16", "rv_vol_16"),
            ("rsi14", "rv_rsi_14"),
            ("z16", "rv_z_mom_16"),
        ):
            pass  # coalesced below per generator prefix
        cmap = _carrier_map(df)
        out = out.join(per_ccy, on="timestamp", how="left")
        for suffix, name in (
            ("mom4", "rv_mom_4"),
            ("mom16", "rv_mom_16"),
            ("vol16", "rv_vol_16"),
            ("rsi14", "rv_rsi_14"),
            ("z16", "rv_z_mom_16"),
        ):
            exprs = []
            for ccy in currencies:
                col = f"rvf_{suffix}_{ccy}"
                if ccy not in cmap:
                    continue
                exprs.append(
                    pl.when(pl.col("symbol") == cmap[ccy])
                    .then(pl.col(col))
                    .otherwise(None)
                )
            if not exprs:
                continue
            combined = exprs[0]
            for e in exprs[1:]:
                combined = pl.coalesce([combined, e])
            out = out.with_columns(combined.alias(name))
        return out.drop([c for c in out.columns if c.startswith("rvf_")])


class CsRVRelativeFeatures(BaseFeatureGenerator):
    """The carrier currency's 4-bar momentum versus the rest of the basket,
    cross-sectionally WITHIN one timestamp.

    rv_rel_mom4_rank -- descending rank (1..K) of the carrier's 4-bar index
        momentum among all currencies at t (K = strongest).
    rv_rel_mom4_z -- z-score of the same momentum across currencies at t.
    """

    feature_cols = ("rv_rel_mom4_rank", "rv_rel_mom4_z")

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        if "carrier_ccy" not in df.columns:
            df = _attach_carrier(df)
        indexes, currencies = _basket_index_frame(df)
        L = indexes.select(
            "timestamp", *[pl.col(c).log().alias(c) for c in currencies]
        ).sort("timestamp")
        mom = L.select(
            "timestamp",
            *[(pl.col(c) - pl.col(c).shift(4)).alias(c) for c in currencies],
        )
        ts_vals = mom["timestamp"]
        vals = mom.select([c for c in currencies]).to_numpy().astype(float)
        n = len(currencies)
        ranks = np.full(vals.shape, np.nan)
        zs = np.full(vals.shape, np.nan)
        for i in range(vals.shape[0]):
            row = vals[i]
            if not np.isfinite(row).all():
                continue
            order = np.argsort(-row, kind="stable")
            rank = np.empty(n)
            rank[order] = np.arange(1, n + 1)
            ranks[i] = rank
            zs[i] = (row - row.mean()) / (row.std() + EPS)
        rel: Dict[str, object] = {"timestamp": ts_vals}
        for j, c in enumerate(currencies):
            rel[f"rvr_rank_{c}"] = ranks[:, j]
            rel[f"rvr_z_{c}"] = zs[:, j]
        rel_df = pl.DataFrame(rel)

        out = df.join(rel_df, on="timestamp", how="left")
        cmap = _carrier_map(df)
        for prefix, name in (("rvr_rank", "rv_rel_mom4_rank"), ("rvr_z", "rv_rel_mom4_z")):
            exprs = []
            for ccy in currencies:
                if ccy not in cmap:
                    continue
                exprs.append(
                    pl.when(pl.col("symbol") == cmap[ccy])
                    .then(pl.col(f"{prefix}_{ccy}"))
                    .otherwise(None)
                )
            combined = exprs[0]
            for e in exprs[1:]:
                combined = pl.coalesce([combined, e])
            out = out.with_columns(combined.alias(name))
        return out.drop([c for c in out.columns if c.startswith("rvr_")])


class CsRVLabelGenerator(BaseFeatureGenerator):
    """Writes rv_target (Float64 1/0/null) onto carrier rows.

    Called by build_csrv_frame on the stacked RAW bars BEFORE build_frame, so
    the production chop veto / tail purge / cleanup path sees exactly the
    columns production expects, and removal of vetoed rows can never look
    forward through the label. rv_target is a TARGET — it is never declared a
    feature column (that would be instant lookahead).
    """

    feature_cols: Tuple[str, ...] = ()

    def __init__(self, tau: int = TAU_BARS):
        self.tau = int(tau)

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        if "carrier_ccy" not in df.columns:
            df = _attach_carrier(df)
        indexes, currencies = _basket_index_frame(df)
        dl = index_log_changes(indexes, currencies, tau=self.tau)
        labels = rv_labels(dl, currencies)

        out = df.join(labels, on="timestamp", how="left")
        cmap = _carrier_map(df)
        exprs = []
        for ccy in currencies:
            if ccy not in cmap:
                continue
            exprs.append(
                pl.when(pl.col("symbol") == cmap[ccy])
                .then(pl.col(f"y_{ccy}"))
                .otherwise(None)
            )
        combined = exprs[0]
        for e in exprs[1:]:
            combined = pl.coalesce([combined, e])
        out = out.with_columns(combined.cast(pl.Float64).alias(CS_RV_LABEL_COL))
        return out.drop([f"y_{c}" for c in currencies])


def _build_csrv(spec, alpha_table) -> List[BaseFeatureGenerator]:
    """Registry build fn. Index features first, then the cross-sectional
    spreads (the ordering the column declaration assumes)."""
    return [CsRVIndexFeatures(), CsRVRelativeFeatures()]


def _cols_csrv(spec, alpha_table) -> Tuple[str, ...]:
    return RV_FEATURES


from lab.registry import register_family
from lab.spec import (  # noqa: E402
    DEFAULT_TRADEABLE_6,
    FeatureSpec,
    GateConfig,
    GeometrySpec,
    LabelSpec,
)

register_family(
    "cs_rv",
    build=_build_csrv,
    columns=_cols_csrv,
    version=1,
    description=(
        "Lane-3 cross-sectional relative value: geometric currency indexes "
        "(momentum/vol/RSI/z + within-t cross-sectional spreads) carried on "
        "carrier-pair rows. The rv_target label is written separately by "
        "CsRVLabelGenerator via build_csrv_frame."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# Spec and frame
# ═══════════════════════════════════════════════════════════════════════════


def forex_cs_rv_spec(symbols: Tuple[str, ...] = DEFAULT_TRADEABLE_6) -> FeatureSpec:
    """The lane's FeatureSpec, exactly as the dispatch pins it (served
    geometry, survival kind, lightgbm/3-fold)."""
    return FeatureSpec(
        name="forex_cs_rv",
        symbols=symbols,
        granularity=15,
        days_back=730,
        feature_sets=("v3_base", "cs_rv"),
        geometry=GeometrySpec(sl_mult=2.0, tp_mult=4.0, max_hold=45),
        label=LabelSpec(kind="survival"),
        gate=GateConfig(model_family="lightgbm", n_folds=3),
    )


def build_csrv_frame(spec, bars: Dict[str, pl.DataFrame], *, tau: int = TAU_BARS):
    """build_frame with the RV label injected at the one safe point.

    CsRVLabelGenerator writes rv_target onto the stacked raw bars FIRST; then
    lab.frames.build_frame runs the unchanged production order (v3_base +
    cs_rv features, production labels, chop veto on the contiguous path,
    tail purge, cleanup on feature cols + production targets). The veto
    therefore operates on the same rows it would in production, and rv_target
    — never a feature col — simply follows its rows through the filter.
    """
    from lab.frames import build_frame, stack_bars

    stacked = stack_bars(bars)
    stacked = CsRVLabelGenerator(tau=tau).generate(stacked)
    labelled = {
        sym: stacked.filter(pl.col("symbol") == sym).drop("symbol")
        for sym in spec.symbols
    }
    empty = [s for s, d in labelled.items() if d.is_empty()]
    if empty:
        raise ValueError(f"no rows for {empty} after RV labelling")
    return build_frame(spec, labelled)


# ═══════════════════════════════════════════════════════════════════════════
# The lane's gate (RV label, retrainer's fold schedule and estimator)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CsRVFoldMetrics:
    """One fold's OOS scoreboard on the RV label."""

    fold_number: int
    train_size: int
    val_size: int
    signals: int = 0
    wins: int = 0
    threshold: float = 0.5

    @property
    def win_rate(self) -> float:
        return self.wins / self.signals if self.signals else float("nan")


@dataclass
class RVGateResult:
    """The lane's verdict, in both currencies (pp and R), toll subtracted.

    edge_pp               -- pooled win rate minus 0.50 (THIS label's base).
    edge_r_gross          -- that edge priced at the 2:1 bracket: the briefing's
                             conversion, pool_ev = p*4 - (1-p)*2 implied by the
                             measured win rate, expressed relative to break-even.
    edge_r_net            -- edge_r_gross minus TOLL_R; the decisive number.
    """

    folds: List[CsRVFoldMetrics]
    pooled_signals: int
    pooled_wins: int
    pooled_win_rate: float
    pooled_pf_lower_bound: float
    mean_threshold: float
    edge_pp: float
    edge_r_gross: float
    edge_r_net: float
    gate_passed: bool
    model_family: str
    elapsed_s: float
    models: Dict[str, object] = field(default_factory=dict)
    feature_cols: Tuple[str, ...] = ()
    trials: int = N_TRIALS


def _pf_lower_bound_2to1(wins: int, trades: int) -> float:
    """One-sided 90% Clopper-Pearson PF lower bound at the 2:1 bracket:
    PF_lb = 2*p_lb/(1-p_lb), same instrument family the production gate uses
    (core/retrainer/_gate._holdout_pf_lower_bound), restated for this label."""
    if trades <= 0:
        return 0.0
    if wins <= 0:
        return 0.0
    from scipy.stats import beta as _beta

    p_lb = float(_beta.ppf(0.10, wins, trades - wins + 1))
    if p_lb >= 1.0:
        return float("inf")
    return 2.0 * p_lb / (1.0 - p_lb)


def _ev_max_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    """Proposal bar from TRAIN-ONLY OOF probabilities.

    NOTE (measured 2026-09-24 on iid noise): a 2:1 EV-max sweep picks
    whichever bar best flatters the train residual, and with a 50/50 label
    every sweep lands at wr >= 0.5 by construction — val then inherits that
    bias and a driftless stream can print a FALSE pass (the pure-noise pin
    test drove this change: EV-max printed pooled wr 0.635 / edge_R +0.154 on
    zero-skill data). The calibration the gate needs is therefore the bar at
    which the train OOF PRECISION has statistically separated from base rate
    with the smallest population: the lowest bar whose Clopper-Pearson lower
    bound clears 0.50, subject to a minimum-signal pool so the fold still
    trades. When NO bar separates (the honest null), the fold's bar falls
    back to 0.5 and the val win rate prints its raw ~0.50 — the falsification
    the lane exists to measure.
    """
    if len(probs) < 60:
        return 0.5
    from scipy.stats import beta as _beta

    candidates: List[Tuple[float, float, int]] = []  # (t, wr, n)
    for t in np.arange(0.30, 0.8001, 0.01):
        mask = probs >= t
        n = int(mask.sum())
        if n < 20:
            continue
        wins = int(y[mask].sum())
        wr = wins / n
        lb = float(_beta.ppf(0.10, wins, n - wins + 1)) if wins else 0.0
        candidates.append((float(t), wr, n, lb))

    separated = [c for c in candidates if c[3] > 0.50]
    if separated:
        # among statistically-separated bars, take the one with the highest
        # precision, tie-broken toward the larger pool (more evidence)
        separated.sort(key=lambda c: (-c[1], -c[2]))
        return separated[0][0]
    # no significant separation anywhere: maximum-sensitivity bar keeps the
    # fold measuring (its wr will print ~0.5 on honest nulls)
    candidates.sort(key=lambda c: (-c[2], abs(c[1] - 0.5)))
    return float(max(0.40, candidates[0][0])) if candidates else 0.5


def _oof_probs(X: np.ndarray, y: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """3-block OOF probabilities on the train window, split on the
    CHRONOLOGICAL permutation of the window (the retrainer's 2026-09-09
    OOF-leakage fix: never raw row order, core/retrainer/_train.py:137-171)."""
    import lightgbm as lgb

    n = len(y)
    if n < 60 or y.sum() == 0 or y.sum() == n:
        return np.array([]), np.array([])
    perm = np.arange(n)  # train window is timestamp-ascending within currency
    k = 3
    oof = np.full(n, np.nan)
    bounds = [int(round(i * n / k)) for i in range(k + 1)]
    for i in range(k):
        val_idx = perm[bounds[i] : bounds[i + 1]]
        tr_idx = np.concatenate([perm[: bounds[i]], perm[bounds[i + 1] :]])
        y_tr = y[tr_idx]
        if len(val_idx) == 0 or len(tr_idx) < 40:
            continue
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr_idx], y_tr)
        oof[val_idx] = m.predict_proba(X[val_idx])[:, 1]
    ok = np.isfinite(oof)
    return oof[ok], y[ok]


def run_csrv_gate(frame, spec, *, n_folds: int = 3, keep_models: bool = True) -> RVGateResult:
    """Walk-forward gate on rv_target: one LightGBM per currency per fold.

    Fold schedule is the retrainer's calendar-fraction scheme verbatim
    (train 0-1/2 | val 1/2-2/3; 0-2/3 | 2/3-5/6; 0-5/6 | 5/6-1 of the frame's
    actual span), so these folds are comparable to every prior lab report.
    Per fold, the Angel's forex hyperparameters (get_hyperparameters) train one
    model per currency on that currency's carrier rows; the proposal bar comes
    from train-only OOF EV maximisation — validation rows never calibrate.
    """
    import lightgbm as lgb

    from core.retrainer._common import get_hyperparameters

    started = time.monotonic()
    angel_params, _ = get_hyperparameters(spec.asset_class)

    df = frame.df
    if "carrier_ccy" not in df.columns:
        df = _attach_carrier(df)
    df = df.filter(
        pl.col(CS_RV_LABEL_COL).is_not_null()
        & pl.col(CS_RV_LABEL_COL).is_finite()  # NaN = abstained cross-section
        & pl.col("carrier_ccy").is_not_null()
    ).sort(["carrier_ccy", "timestamp"])

    feature_cols = [c for c in frame.feature_cols if c in df.columns]
    currencies = sorted(df["carrier_ccy"].unique().to_list())

    min_date = df["timestamp"].min()
    span = int((df["timestamp"].max() - min_date).total_seconds() / 86400.0)
    fold_configs = [
        (span // 2, span * 2 // 3),
        (span * 2 // 3, span * 5 // 6),
        (span * 5 // 6, span),
    ][: max(1, int(n_folds))]

    ts = df["timestamp"].to_numpy()
    carrier = df["carrier_ccy"].to_numpy()
    y_all = df[CS_RV_LABEL_COL].to_numpy().astype(float)
    X_all = df.select(feature_cols).to_numpy()

    # numpy datetime64 for cheap folds; polars datetimes are tz-aware while the
    # cutoff math below is tz-naive, so convert once through to_numpy() on the
    # UTC-normalised column.
    ts = df["timestamp"].dt.replace_time_zone(None).to_numpy()

    folds: List[CsRVFoldMetrics] = []
    models: Dict[str, object] = {}
    pooled_signals = 0
    pooled_wins = 0
    thresholds: List[float] = []

    for fold_idx, (train_end_day, val_end_day) in enumerate(fold_configs):
        train_cutoff = np.datetime64(
            (min_date + timedelta(days=train_end_day)).replace(tzinfo=None), "us"
        )
        val_cutoff = np.datetime64(
            (min_date + timedelta(days=val_end_day)).replace(tzinfo=None), "us"
        )
        train_mask = ts < train_cutoff
        val_mask = (ts >= train_cutoff) & (ts < val_cutoff)

        fm = CsRVFoldMetrics(
            fold_number=fold_idx + 1,
            train_size=int(train_mask.sum()),
            val_size=int(val_mask.sum()),
        )
        fold_models: Dict[str, object] = {}
        fold_thresholds: List[float] = []
        for ccy in currencies:
            ctrain = train_mask & (carrier == ccy)
            cval = val_mask & (carrier == ccy)
            if ctrain.sum() < 60 or cval.sum() < 10:
                continue
            y_tr = y_all[ctrain]
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue
            model = lgb.LGBMClassifier(**angel_params)
            model.fit(X_all[ctrain], y_tr)

            p_oof, y_oof = _oof_probs(X_all[ctrain], y_tr, angel_params)
            if len(p_oof):
                thr = _ev_max_threshold(p_oof, y_oof)
            else:
                thr = 0.5
            fold_thresholds.append(thr)

            probs = model.predict_proba(X_all[cval])[:, 1]
            sig = probs >= thr
            fm.signals += int(sig.sum())
            fm.wins += int(y_all[cval][sig].sum())
            fold_models[ccy] = model

        if fold_thresholds:
            fm.threshold = float(np.mean(fold_thresholds))
            thresholds.extend(fold_thresholds)
        folds.append(fm)
        pooled_signals += fm.signals
        pooled_wins += fm.wins
        if fold_idx == len(fold_configs) - 1 and keep_models:
            models = fold_models

    pooled_win_rate = pooled_wins / pooled_signals if pooled_signals else float("nan")
    pf_lb = _pf_lower_bound_2to1(pooled_wins, pooled_signals)
    mean_thr = float(np.mean(thresholds)) if thresholds else 0.5

    edge_pp = (pooled_win_rate - 0.50) if pooled_signals else float("nan")
    # Brief 6.1: 1 pp ~ 0.03R at the served 2:1 geometry. Gross EV per trade
    # relative to the break-even win rate is the edge in pp converted, i.e.
    # edge_r_gross = edge_pp(pp) * R_PER_PP; stated explicitly in the report.
    edge_r_gross = edge_pp * 100.0 * R_PER_PP_2_TO_1 if pooled_signals else float("nan")
    edge_r_net = edge_r_gross - TOLL_R if pooled_signals else float("nan")

    gate_passed = bool(
        pooled_signals > 0 and np.isfinite(edge_r_net) and edge_r_net > EDGE_R_GATE
    )
    return RVGateResult(
        folds=folds,
        pooled_signals=pooled_signals,
        pooled_wins=pooled_wins,
        pooled_win_rate=pooled_win_rate,
        pooled_pf_lower_bound=pf_lb,
        mean_threshold=mean_thr,
        edge_pp=edge_pp,
        edge_r_gross=edge_r_gross,
        edge_r_net=edge_r_net,
        gate_passed=gate_passed,
        model_family="lightgbm",
        elapsed_s=time.monotonic() - started,
        models=models,
        feature_cols=tuple(feature_cols),
    )


def run_csrv_experiment(
    spec, bars: Dict[str, pl.DataFrame], *, tau: int = TAU_BARS
) -> Tuple[object, RVGateResult]:
    """Frame -> gate; the lane's single entry point."""
    frame = build_csrv_frame(spec, bars, tau=tau)
    gate = run_csrv_gate(frame, spec, n_folds=spec.gate.n_folds)
    return frame, gate


# ═══════════════════════════════════════════════════════════════════════════
# Stats contract (src/lab/stats.py absent on dispatch; Lane 5 may land the
# shared module. This is the identical contract, lane-local so no other
# lane's file is touched. If Lane 1/2/5 land src/lab/stats.py first, these
# wrappers delegate.)
# ═══════════════════════════════════════════════════════════════════════════


def _stats_module():
    try:
        from lab import stats as _s

        return _s
    except Exception:
        return None


def deflated_sharpe_proxy(
    *,
    n_trials: int,
    pooled_win_rate: float,
    n: int,
    base_rate: float = 0.50,
) -> Dict[str, float]:
    """DSR in the simple iid-binomial form usable on a win rate.

    SR_obs = (p - p0) * sqrt(n / (p0(1-p0))); the expected max SR of
    ``n_trials`` null trials is sqrt(2 ln n_trials) (unit-variance null).
    PSR-style probability that the true SR beats the null max. If
    src/lab/stats.py exists with a DSR under a compatible name, it is used.
    """
    s = _stats_module()
    if s is not None and hasattr(s, "deflated_sharpe"):
        try:
            return dict(s.deflated_sharpe(
                pooled_win_rate, n=n, base_rate=base_rate, n_trials=n_trials
            ))
        except Exception:
            pass
    if n <= 0 or not math.isfinite(pooled_win_rate):
        return {"sr_obs": float("nan"), "e_max_sr": float("nan"), "psr": float("nan")}
    var0 = base_rate * (1.0 - base_rate)
    sr_obs = (pooled_win_rate - base_rate) * math.sqrt(n / var0)
    e_max = math.sqrt(2.0 * math.log(max(n_trials, 1)))
    # normal approximation, skew/kurt of a binomial proportion at this n
    z = (sr_obs - e_max)
    psr = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return {"sr_obs": sr_obs, "e_max_sr": e_max, "psr": psr}


def hlz_min_backtest_length(*, n_trials: int, alpha: float = 0.05) -> Dict[str, float]:
    """Haircut/HLZ-style multiple-testing note: the z a single result must beat
    after ``n_trials`` (Bonferroni), and the corresponding minimum edge in pp
    at the observed n, reported by the recon. Same caveats as stats.py's."""
    z_single = 1.6448536269514722  # one-sided 95%
    z_bonf = None
    try:
        from scipy.stats import norm

        z_bonf = float(norm.ppf(1.0 - alpha / max(n_trials, 1)))
    except Exception:
        z_bonf = float("nan")
    return {"n_trials": float(n_trials), "z_single": z_single, "z_bonferroni": z_bonf}


# ═══════════════════════════════════════════════════════════════════════════
# USD-cross availability probe
# ═══════════════════════════════════════════════════════════════════════════


def fetch_usd_crosses(
    *,
    days_back: int = 730,
    granularity: int = 15,
    cache_path: Path = USD_CROSS_CACHE,
    bars_by_symbol: Optional[Dict[str, pl.DataFrame]] = None,
) -> Tuple[List[str], List[str], Optional[Path]]:
    """Try to fetch the 7 USD crosses at M15 and cache the survivors.

    Returns (fetched, failed, cache_path_or_None). The brief's abort rule is
    on the CALLER: > half 404/failed -> document and fall back to the 6-pair
    baseline. Atomic cache write (temp + rename), matching the repo's artifact
    convention. ``bars_by_symbol`` injects synthetic frames in tests.
    """
    fetched: Dict[str, pl.DataFrame] = {}
    failed: List[str] = []

    if bars_by_symbol is None:
        bars_by_symbol = {}
        try:
            from datetime import datetime, timedelta, timezone

            from data.oanda_provider import OandaMarketProvider

            provider = OandaMarketProvider(environment="practice")
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days_back)
            for sym in USD_CROSSES:
                try:
                    bars_by_symbol[sym] = provider.get_historical_bars(
                        sym, granularity, start, end
                    )
                except Exception as exc:  # 404/permission surface inside provider
                    logger.warning("USD cross %s fetch raised: %s", sym, exc)
                    bars_by_symbol[sym] = None
        except Exception as exc:
            logger.warning("USD-cross provider unavailable: %s", exc)
            return [], list(USD_CROSSES), None

    for sym in USD_CROSSES:
        df = bars_by_symbol.get(sym)
        if df is None or df.is_empty():
            failed.append(sym)
        else:
            fetched[sym] = df

    if not fetched:
        return [], failed, None

    stacked = pl.concat(
        [d.with_columns(pl.lit(sym).alias("symbol")) for sym, d in fetched.items()],
        how="vertical_relaxed",
    ).sort(["symbol", "timestamp"])
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    stacked.write_parquet(tmp)
    os.replace(tmp, cache_path)
    return sorted(fetched), failed, cache_path


def load_usd_crosses(cache_path: Path = USD_CROSS_CACHE) -> Dict[str, pl.DataFrame]:
    """{symbol: bars} from the cross cache, or {} when absent."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return {}
    df = pl.read_parquet(cache_path)
    return {
        sym: df.filter(pl.col("symbol") == sym).drop("symbol")
        for sym in df["symbol"].unique().to_list()
    }
