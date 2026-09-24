"""
Audit A — altcoin cross-sectional momentum (Lane 5 falsification audit).

The 2026-09-14 crypto study closed D1/H4 forex-style *bracket* geometries
(0 of 27 positive at random entries) but never tested **cross-sectional
momentum** across a basket of liquid altcoins — a documented crypto anomaly
this audit measures honestly. The question: does a weekly LightGBM
``lambdarank`` top-quintile harvester beat BOTH an equal-weight basket AND
BTC buy-and-hold, after a 33+25 bps/side round-trip toll, with the
multiple-testing adjustments (DSR / CSCV PBO / HLZ) from ``lab.stats``?

Research-only — no orders, no live path. Driven by ``run_a()``, which the
report-generating script calls; every public seam is also directly testable
on synthetic frames.

Glossary:
    UNIVERSE -- the 20 Alpaca crypto pairs the brief nominates. Each is
        admitted to a given week's tradable universe only if it clears the
        liquidity floor using data up to and including that week (no
        look-ahead into future volume).
    LIQ_MIN_DAYS / LIQ_MIN_MEDIAN_DOLLAR_VOL -- the floor: >= 30 days of
        non-zero volume in the sampled window AND median daily dollar volume
        (volume × close) >= $250k over the lookback trailing up to the
        signal week. Alpaca crypto volume is unreliable for some alts; a
        symbol failing the floor is dropped from that week's universe and
        the drop is documented in the run's ``liquidity_drops``.
    LOOKBACKS -- the 4 momentum lookbacks {21, 63, 126, 252} traded as log
        returns of daily closes. A week's feature for asset i is the 4-vector
        of its trailing log returns; assets without enough history that week
        are excluded from the ranking (documented, not silently filled).
    TARGET_HORIZON_DAYS -- 7: the ranking target is the NEXT week's simple
        return, quantiled into 0..4 across the week's universe (qcut with
        duplicates dropped; with < 5 assets in a week the week is skipped).
    K_TOP -- 5: with ndcg_at=[5] the "top-quintile" harvester is the top-K
        of the weekly ranking; K=5 on a ~15–20 asset universe is one
        quintile. With a shrunken universe the top floor(len/5, 1) names are
        taken, never more than K_TOP.
    FRICTION_BPS_SPREAD / FRICTION_BPS_TAKER -- 6.6 bps spread + 25 bps
        taker per side per the lane brief. Cost is charged on ONE-WAY
        turnover Σ max(Δw, 0) (the weight actually bought/sold), so a full
        book swap costs (6.6+25)/10000 × 1.0 (sell 1 + buy 1 = one round
        trip per name) and the brief's forced-0.5-turnover fixture costs
        exactly 0.5 × 31.6/10000 of NAV.
    DSR/PBO/HLZ gates -- the run must satisfy all of: DSR > 0.95 vs the
        trial count (one distinct config was evaluated), CSCV PBO < 0.50
        on the strategy-vs-benchmarks daily-excess matrix, and the
        Clopper–Pearson 95% one-sided lower bound on the weekly
        beat-both-benchmarks success rate > 0.
    run_a -- the public entry. Loads (or uses a caller-supplied) daily bar
        history, builds the weekly ranking walk-forward, applies the
        harvester with friction, computes the dual-benchmark excess series,
        scores the gates through ``lab.stats`` and returns the audit record.

Leakage guards (per the brief §4.4):
    * weekly query groups use CLOSED daily bars only — the signal at the
      week's Friday close uses only data ≤ that close;
    * the target window is strictly AFTER the signal bar (next 7 days);
    * universe membership at week t uses only volume data ≤ t.

Data source: Alpaca crypto daily bars via
``alpaca.data.historical.CryptoHistoricalDataClient`` (NOT the Binance
geo-blocked feed, NOT OANDA — this audit is on the Alpaca-venue crypto
universe). Volume is Alpaca's report and carries the documented unreliability;
the floor drops thin names rather than trusting their numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from lab import stats as _stats

# ── constants ─────────────────────────────────────────────────────────────

UNIVERSE: Tuple[str, ...] = (
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
    "DOT/USD", "LTC/USD", "BCH/USD", "UNI/USD", "AAVE/USD",
    "MKR/USD", "CRV/USD", "SNX/USD", "COMP/USD", "YFI/USD",
    "SUSHI/USD", "UMA/USD", "ZRX/USD", "BAT/USD", "GRT/USD",
)

LOOKBACKS: Tuple[int, ...] = (21, 63, 126, 252)
TARGET_HORIZON_DAYS: int = 7
K_TOP: int = 5
LIQ_MIN_DAYS: int = 30
LIQ_MIN_MEDIAN_DOLLAR_VOL: float = 250_000.0
LIQ_LOOKBACK_DAYS: int = 60
FRICTION_BPS_SPREAD: float = 6.6
FRICTION_BPS_TAKER: float = 25.0
FRICTION_BPS_PER_SIDE: float = FRICTION_BPS_SPREAD + FRICTION_BPS_TAKER  # 31.6
MIN_UNIVERSE_PER_WEEK: int = 5
WEEKS_PER_YEAR: float = 52.0

# One configuration is evaluated: (LOOKBACKS tuple, K_TOP=5, weekly cadence).
N_TRIALS: int = 1


# ── data containers ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class WeeklyBasketRecord:
    """One rebalancing week's decision and outcome."""

    week: np.datetime64  # the signal Friday (week key), ns
    selected: Tuple[str, ...]  # the top-floor quintile names, sorted
    gross_return: float  # equal-weight next-week return before toll
    turnover: float  # sum of |Δw| entering this week (0..2)
    friction: float  # NAV cost = turnover × FRICTION_BPS_PER_SIDE/1e4
    net_return: float  # gross_return − friction
    universe_size: int  # admitted names that week


@dataclass
class AuditAResult:
    """Everything the report needs, all deterministic from the inputs."""

    weekly: List[WeeklyBasketRecord] = field(default_factory=list)
    liquidity_drops: Dict[str, List[str]] = field(default_factory=dict)
    bench_weekly_equal_weight: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    bench_weekly_btc_hold: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # gates
    dsr: float = 0.0
    pbo: float = 0.5
    hlz_adj_t: float = 0.0
    cp_lower_vs_equal: float = 0.0
    cp_lower_vs_btc: float = 0.0
    beats_equal_weight: bool = False
    beats_btc_hold: bool = False
    gate_pass: bool = False
    verdict: str = ""
    # descriptive
    n_weeks: int = 0
    median_universe_size: float = 0.0

    @property
    def weekly_net(self) -> np.ndarray:
        return np.array([w.net_return for w in self.weekly], dtype=float)

    @property
    def weekly_gross(self) -> np.ndarray:
        return np.array([w.gross_return for w in self.weekly], dtype=float)


# ── leakage-guarded helpers ───────────────────────────────────────────────


def _week_key(ts: np.datetime64) -> np.datetime64:
    """
    The ISO week key for a daily-bar timestamp: the MONDAY of that week.
    Query groups are formed on this key; the signal is computed on the
    week's LAST available daily bar and the target is the next 7 days.
    Vectorised: accepts a scalar or an array of datetime64 and returns the
    same shape of datetime64[D] Mondays.
    """
    d = np.asarray(ts).astype("datetime64[D]").astype("int64")
    monday = d - ((d - 3) % 7)  # 1970-01-05 (int 4) is the anchor Monday
    return monday.astype("datetime64[D]")


def admit_universe(
    dates: np.ndarray,
    symbols: Sequence[str],
    close: np.ndarray,
    volume: np.ndarray,
    as_of: np.datetime64,
    *,
    min_days: int = LIQ_MIN_DAYS,
    min_median_dollar_vol: float = LIQ_MIN_MEDIAN_DOLLAR_VOL,
    lookback_days: int = LIQ_LOOKBACK_DAYS,
) -> Tuple[List[str], List[str]]:
    """
    The week's tradable universe: symbols that clear the liquidity floor using
    ONLY data with timestamp <= as_of. Returns (admitted, dropped_reasons).

    ``close`` and ``volume`` are (T × N) daily matrices; ``dates`` is the
    length-T datetime64[D] axis; ``symbols`` the length-N column names.
    The lookback is the trailing ``lookback_days`` calendar days ending at
    as_of (inclusive).
    """
    admitted: List[str] = []
    dropped: List[str] = []
    cutoff = as_of.astype("datetime64[D]") - np.timedelta64(lookback_days, "D")
    mask = (dates <= as_of.astype("datetime64[D]")) & (dates > cutoff)
    if not mask.any():
        return [], [f"{s}: no data in lookback" for s in symbols]
    for j, sym in enumerate(symbols):
        v = volume[mask, j]
        c = close[mask, j]
        finite = np.isfinite(v) & np.isfinite(c)
        n_days = int(np.count_nonzero(finite & (v > 0)))
        if n_days < min_days:
            dropped.append(f"{sym}: {n_days} non-zero-vol days < {min_days}")
            continue
        dollar_vol = v[finite] * c[finite]
        med = float(np.median(dollar_vol)) if dollar_vol.size else 0.0
        if not np.isfinite(med) or med < min_median_dollar_vol:
            dropped.append(
                f"{sym}: median $vol {med:,.0f} < {min_median_dollar_vol:,.0f}"
            )
            continue
        admitted.append(sym)
    return admitted, dropped


def momentum_features(
    dates: np.ndarray,
    close: np.ndarray,
    as_of_idx: int,
    lookbacks: Sequence[int] = LOOKBACKS,
) -> np.ndarray:
    """
    The 4-vector of trailing log returns per asset ending at row as_of_idx
    (INCLUSIVE — the signal uses the close at the signal bar). Shape (N, L).
    Assets lacking history for a lookback get NaN (excluded downstream).

    Pure arithmetic on closed bars: features at as_of_idx never use a row
    > as_of_idx — the leakage-guard test truncates and re-computes.
    """
    N = close.shape[1]
    out = np.full((N, len(lookbacks)), np.nan)
    for li, lb in enumerate(lookbacks):
        j0 = as_of_idx - lb
        if j0 < 0:
            continue
        c0 = close[j0]
        c1 = close[as_of_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where((c0 > 0) & (c1 > 0), np.log(c1 / c0), np.nan)
        out[:, li] = lr
    return out


def next_week_returns(
    dates: np.ndarray,
    close: np.ndarray,
    signal_idx: int,
    horizon_days: int = TARGET_HORIZON_DAYS,
) -> np.ndarray:
    """
    The per-asset simple return over the next ``horizon_days`` calendar days
    strictly AFTER the signal bar: from close at signal_idx to the close at
    the first daily bar with date ≥ signal_date + horizon_days (the
    bar-on-or-after convention — the target is the horizon-day close, never
    a bar beyond). Shape (N,). NaN where the entry or exit close is NaN.
    """
    N = close.shape[1]
    out = np.full(N, np.nan)
    T = close.shape[0]
    if signal_idx >= T - 1:
        return out
    entry = close[signal_idx]
    sig_date = dates[signal_idx]
    target_date = sig_date + np.timedelta64(horizon_days, "D")
    exit_idx = int(np.searchsorted(dates, target_date, side="left"))
    if exit_idx >= T:
        return out
    exitp = close[exit_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(
            (entry > 0) & np.isfinite(entry) & np.isfinite(exitp),
            exitp / entry - 1.0,
            np.nan,
        )
    return r


def apply_friction(prev_weights: Dict[str, float], new_weights: Dict[str, float]) -> Tuple[float, float]:
    """
    The NAV cost of moving from prev_weights to new_weights.

    Turnover is the ONE-WAY traded weight Σ max(new − prev, 0) over the union
    of names (equivalently Σ max(prev − new, 0) for a fully-invested book);
    cost = turnover × FRICTION_BPS_PER_SIDE/1e4. A no-change week costs 0;
    a full swap (prev and new disjoint, each summing to 1) has turnover 1
    and costs one side-rate.
    """
    names = set(prev_weights) | set(new_weights)
    buys = sum(max(new_weights.get(s, 0.0) - prev_weights.get(s, 0.0), 0.0) for s in names)
    sells = sum(max(prev_weights.get(s, 0.0) - new_weights.get(s, 0.0), 0.0) for s in names)
    turnover = max(buys, sells)  # one-way: the buys fund the sells
    return turnover, turnover * FRICTION_BPS_PER_SIDE / 10_000.0


def _rank_scores_lambdarank(
    feats_train: np.ndarray,
    tgt_train_quint: np.ndarray,
    groups_train: np.ndarray,
    feats_eval: np.ndarray,
) -> np.ndarray:
    """
    Score an evaluation week's assets with a LightGBM lambdarank model fitted
    on all PRIOR weeks. Returns a (N_eval,) score array (higher = stronger).
    ``groups_train`` is the query-group size vector (one entry per training
    week). ``tgt_train_quint`` is the integer relevance label 0..4, computed
    per training week by qcut of that week's realized next-week returns.
    """
    import lightgbm as lgb

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[5, 10],
        n_estimators=80,
        num_leaves=15,
        learning_rate=0.08,
        min_data_in_leaf=10,
        random_state=7,
        verbose=-1,
    )
    ranker.fit(
        feats_train,
        tgt_train_quint,
        group=[int(g) for g in groups_train],
    )
    return np.asarray(ranker.predict(feats_eval), dtype=float)


def _nan_average(x: np.ndarray) -> np.ndarray:
    """Row-wise mean ignoring NaNs; NaN where a row is all-NaN."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(x, axis=1)


# ── walk-forward engine ───────────────────────────────────────────────────


def _weekly_signal_indices(dates: np.ndarray) -> List[Tuple[np.datetime64, int]]:
    """The (week_key, last_bar_index) list for every ISO week with >= 1 bar."""
    out: List[Tuple[np.datetime64, int]] = []
    if dates.size == 0:
        return out
    prev_key = _week_key(dates[0])
    for i in range(1, dates.size):
        k = _week_key(dates[i])
        if k != prev_key:
            out.append((prev_key, i - 1))
            prev_key = k
    out.append((prev_key, dates.size - 1))
    return out


def run_a(
    dates: np.ndarray,
    symbols: Sequence[str],
    close: np.ndarray,
    volume: np.ndarray,
    *,
    lookbacks: Sequence[int] = LOOKBACKS,
    k_top: int = K_TOP,
    min_days: int = LIQ_MIN_DAYS,
    min_median_dollar_vol: float = LIQ_MIN_MEDIAN_DOLLAR_VOL,
    warmup_weeks: int = 26,
    ranker: str = "lambdarank",
) -> AuditAResult:
    """
    Walk the weekly top-quintile harvester across the daily history.

    ``dates`` is the sorted datetime64[D] axis, ``close``/``volume`` the
    (T × N) matrices aligned to ``symbols``. ``warmup_weeks`` is the minimum
    count of completed training weeks before the first lambdarank model is
    fitted; earlier weeks are warmup (no trade). ``ranker`` selects the
    scorer: ``"lambdarank"`` (the production path) or ``"momentum"`` (the
    hand-rolled mean-of-lookback-returns used by the synthetic fixture
    tests, where monotonic-in-momentum data makes the two equivalent —
    the fixture never exercises lambdarank on real-data behaviour).
    Returns the AuditAResult.
    """
    res = AuditAResult()
    T, N = close.shape
    sig_weeks = _weekly_signal_indices(dates)
    weekly_net: List[float] = []
    weekly_ew: List[float] = []
    weekly_btc: List[float] = []
    prev_weights: Dict[str, float] = {}
    btc_j = symbols.index("BTC/USD") if "BTC/USD" in symbols else None

    # Rolling training memory for the lambdarank path: features, quintile
    # labels and the query-group sizes, all from weeks STRICTLY before the
    # current signal week (train on the past, score the present).
    train_feats: List[np.ndarray] = []
    train_q: List[np.ndarray] = []
    train_g: List[int] = []
    n_trained_weeks = 0

    for week_key, sig_idx in sig_weeks:
        if sig_idx + 1 >= T:
            break
        as_of = dates[sig_idx]
        admitted, dropped = admit_universe(
            dates, symbols, close, volume, as_of,
            min_days=min_days, min_median_dollar_vol=min_median_dollar_vol,
        )
        for d in dropped:
            sym = d.split(":")[0]
            res.liquidity_drops.setdefault(sym, []).append(str(week_key))
        if len(admitted) < MIN_UNIVERSE_PER_WEEK:
            continue
        adm_idx = np.array([symbols.index(s) for s in admitted])

        feats = momentum_features(dates, close, sig_idx, lookbacks)
        tgt = next_week_returns(dates, close, sig_idx, TARGET_HORIZON_DAYS)

        sub_feats = feats[adm_idx]
        sub_tgt = tgt[adm_idx]
        valid = np.all(np.isfinite(sub_feats), axis=1) & np.isfinite(sub_tgt)
        if valid.sum() < MIN_UNIVERSE_PER_WEEK:
            continue

        v_feats = sub_feats[valid]
        v_tgt = sub_tgt[valid]
        v_syms = [admitted[i] for i, ok in enumerate(valid) if ok]
        n_v = len(v_syms)

        # ── per-week quintile labels 0..4 of the realized next-week return
        # (the rank target). Duplicate edges with small universes are
        # broken by rank so every label is populated.
        order = np.argsort(v_tgt, kind="stable")
        quint = np.empty(n_v, dtype=int)
        ranks = np.empty(n_v, dtype=int)
        ranks[order] = np.arange(n_v)
        quint = np.minimum(ranks * 5 // n_v, 4)

        # ── score this week
        if ranker == "lambdarank" and n_trained_weeks >= warmup_weeks:
            tr_feats = np.concatenate(train_feats)
            tr_q = np.concatenate(train_q)
            scores = _rank_scores_lambdarank(tr_feats, tr_q, np.array(train_g), v_feats)
        elif ranker == "lambdarank":
            # warmup: no tradable signal yet — record the week as training-
            # only and DO NOT enter a position.
            train_feats.append(v_feats)
            train_q.append(quint)
            train_g.append(n_v)
            n_trained_weeks += 1
            continue
        else:  # "momentum" fixture path
            scores = _nan_average(v_feats)
            train_feats.append(v_feats)
            train_q.append(quint)
            train_g.append(n_v)
            n_trained_weeks += 1

        k = max(1, min(k_top, n_v // 5))
        top_idx = np.argsort(-scores, kind="stable")[:k]
        picks = sorted(v_syms[i] for i in top_idx)
        gross = float(np.mean(v_tgt[top_idx]))
        new_w = {s: 1.0 / k for s in picks}
        turnover, fric = apply_friction(prev_weights, new_w)
        prev_weights = new_w

        res.weekly.append(
            WeeklyBasketRecord(
                week=np.datetime64(week_key, "ns"),
                selected=tuple(picks),
                gross_return=gross,
                turnover=turnover,
                friction=fric,
                net_return=gross - fric,
                universe_size=len(admitted),
            )
        )
        weekly_net.append(gross - fric)
        weekly_ew.append(float(np.nanmean(v_tgt)))
        weekly_btc.append(float(tgt[btc_j]) if btc_j is not None else float("nan"))

        # fold the just-measured week into training memory AFTER scoring it
        train_feats.append(v_feats)
        train_q.append(quint)
        train_g.append(n_v)
        n_trained_weeks += 1

    res.bench_weekly_equal_weight = np.array(weekly_ew, dtype=float)
    res.bench_weekly_btc_hold = np.array(weekly_btc, dtype=float)
    res.n_weeks = len(res.weekly)
    res.median_universe_size = (
        float(np.median([w.universe_size for w in res.weekly])) if res.weekly else 0.0
    )

    net = res.weekly_net
    if net.size >= 4:
        # gates on the weekly series: per-week simple returns, 52 weeks/yr
        sr = _stats.sharpe_ratio(net, periods_per_year=WEEKS_PER_YEAR)
        n_obs = net.size
        skew = float(_series_skew(net))
        kurt = float(_series_kurt(net))
        res.dsr = _stats.deflated_sharpe_ratio(sr, N_TRIALS, n_obs, skew, kurt)
        # HLZ in t-stat units: mean/std × sqrt(n)
        se = float(np.std(net, ddof=1))
        if se > 0:
            t_eq = float(net.mean() / se * math.sqrt(n_obs))
            res.hlz_adj_t = _stats.hlz_haircut_sharpe(t_eq, N_TRIALS)
        excess_eq = net - res.bench_weekly_equal_weight
        excess_btc = net - res.bench_weekly_btc_hold
        res.cp_lower_vs_equal = _stats.clopper_pearson_lower(
            int(np.count_nonzero(excess_eq > 0)), int(np.count_nonzero(np.isfinite(excess_eq)))
        )
        res.cp_lower_vs_btc = _stats.clopper_pearson_lower(
            int(np.count_nonzero(excess_btc > 0)), int(np.count_nonzero(np.isfinite(excess_btc)))
        )
        # PBO over weekly matrix: columns = strategy, EW bench, BTC bench
        m = np.column_stack([net, res.bench_weekly_equal_weight, res.bench_weekly_btc_hold])
        m = np.nan_to_num(m, nan=0.0)
        res.pbo = _stats.cscv_pbo(m) if m.shape[0] >= 8 else 0.5
        res.beats_equal_weight = (excess_eq.mean() > 0) and (res.cp_lower_vs_equal > 0.0)
        res.beats_btc_hold = (excess_btc.mean() > 0) and (res.cp_lower_vs_btc > 0.0)
        res.gate_pass = bool(
            res.beats_equal_weight
            and res.beats_btc_hold
            and res.dsr > 0.95
            and res.pbo < 0.50
        )
        res.verdict = (
            "PASS — the top quintile beats both benchmarks after toll with DSR > 0.95 and PBO < 0.50"
            if res.gate_pass
            else _fail_reason(res)
        )
    else:
        res.verdict = "INCONCLUSIVE — fewer than 4 rebalancing weeks measured"
    return res


def _fail_reason(res: AuditAResult) -> str:
    if not res.beats_equal_weight:
        return "FAIL — top quintile does not beat the equal-weight basket"
    if not res.beats_btc_hold:
        return "FAIL — top quintile cannot beat BTC buy-and-hold"
    if res.dsr <= 0.95:
        return f"FAIL — DSR {res.dsr:.3f} ≤ 0.95 (multiple-testing deflated)"
    if res.pbo >= 0.50:
        return f"FAIL — PBO {res.pbo:.3f} ≥ 0.50 (backtest-overfit probability)"
    return "FAIL"


def _series_skew(x: np.ndarray) -> float:
    """Fisher-Pearson sample skewness (bias-corrected), 0 for tiny inputs."""
    n = x.size
    if n < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    m3 = ((x - m) / s) ** 3
    return float(n / ((n - 1) * (n - 2)) * m3.sum())


def _series_kurt(x: np.ndarray) -> float:
    """Pearson (non-excess) sample kurtosis, bias-corrected; 3 = Gaussian."""
    n = x.size
    if n < 4:
        return 3.0
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 3.0
    z4 = (((x - m) / s) ** 4).mean()
    # bias-corrected excess kurtosis then back to Pearson
    g2 = z4 - 3.0
    g2_adj = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
    return float(g2_adj + 3.0)
