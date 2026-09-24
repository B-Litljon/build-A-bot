"""Multiple-testing falsification statistics for the research lanes.

Lane 1 (crypto trend momentum + vol targeting) is the first lane to need
deflated-Sharpe / CSCV-PBO / Harvey-Liu-Zhu adjustments, and this module is the
single implementation every later lane imports. The three entry-point
signatures are a cross-lane contract — they are pinned by the dispatch brief
(`llm_reports/handoffs/2026-09-24_lane1-crypto-trend-mom-tv.md` §6.2) and by
`tests/test_lab_stats.py`; do not change them without updating both.

Glossary:
    sr_hat -- the measured Sharpe ratio of the selected (best-looking) strategy,
        in any consistent annualization; every adjustment here works on the
        ratio directly plus the return series' skew/kurtosis.
    n_trials -- the number of DISTINCT strategy parametrizations actually
        evaluated by the caller, not the size of the grid that was merely
        considered. The adjustments grow with the log/quantile of this, so
        honest counting is the whole point.
    n_obs -- number of return observations behind sr_hat.
    skew / kurt -- moment shape of the strategy's daily returns; kurt is the
        FULL (Pearson) kurtosis, matching the (kurt - 1) / 4 term of both papers.
    var_sr -- sampling variance of Sharpe under the no-skill null; default
        1 / n_obs, the iid-Normal benchmark from Bailey & Lopez de Prado (2014).
    DSR -- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the probability
        that a Sharpe of sr_hat exceeds E[max(SR)] under a null of n_trials
        independent trials, after correcting the sampling distribution for
        skew/kurtosis. Returned as a probability in [0, 1].
    PBO -- Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado,
        Zhu 2014) via combinatorially symmetric cross-validation: fraction of
        the 70 eight-block train/test splits in which the in-sample-best
        strategy ranks at-or-below the test-half median. Scalar in [0, 1].
    HLZ -- Harvey, Liu, Zhu (2016) haircut Sharpe: sr_hat minus the
        expected-max null draw at quantile z_{1 - 1/(2*n_trials)} on the
        unit-information benchmark. An adjusted Sharpe, NOT a probability;
        reported alongside its implied t-stat (hlz_t_stat).
    BLOCKS -- CSCV block count S = 8, giving C(8, 4) = 70 train/test splits, the
        exact design of the 2014 paper.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import norm

BLOCKS = 8
"""CSCV block count: S=8 contiguous blocks -> C(8,4)=70 train/test splits."""

_COMBOS = tuple(itertools.combinations(range(BLOCKS), BLOCKS // 2))
"""The 70 train/train block combinations, fixed at import so the statistic
cannot drift from the paper's design."""

_EULER_GAMMA = 0.5772156649015329
"""Euler-Mascheroni constant; appears in E[max] of null Sharpe draws."""


def deflated_sharpe_ratio(
    sr_hat: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurt: float,
    *,
    var_sr: float | None = None,
) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio, as a probability.

    DSR = Phi( (sr_hat - SR0) * sqrt(n_obs - 1)
               / sqrt(1 - skew*sr_hat + ((kurt - 1)/4) * sr_hat^2) )

    where SR0 = E[max(SR)] under the null: the expected maximum of n_trials
    iid Normal draws with variance var_sr (default 1/n_obs), via scipy's
    Normal quantile (per the brief contract):

        SR0 = sqrt(var_sr)
              * ( (1 - gamma) * Phi^-1(1 - 1/n_trials)
                  + gamma * Phi^-1(1 - 1/(n_trials * e)) )

    Returns 0.0 outright when n_trials <= 1 (no selection => nothing to
    deflate) and raises ValueError when the variance-adjusted denominator is
    non-positive (a wildly non-Normal series where the formula is undefined).
    """
    if n_trials <= 1:
        return 0.0
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2 for a Sharpe ratio, got {n_obs}")
    if var_sr is None:
        var_sr = 1.0 / n_obs
    if var_sr <= 0:
        raise ValueError(f"var_sr must be positive, got {var_sr}")

    gamma = _EULER_GAMMA
    n = float(n_trials)
    sr0 = math.sqrt(var_sr) * (
        (1.0 - gamma) * float(norm.ppf(1.0 - 1.0 / n))
        + gamma * float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    )
    denom_sq = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat * sr_hat
    if denom_sq <= 0:
        raise ValueError(
            "DSR undefined: 1 - skew*SR + ((kurt-1)/4)*SR^2 <= 0 "
            f"(skew={skew}, kurt={kurt}, sr_hat={sr_hat})"
        )
    z = (sr_hat - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def cscv_pbo(logret_matrix: "np.ndarray") -> float:
    """CSCV Probability of Backtest Overfitting (scalar in [0, 1]).

    Input: T x N matrix of daily log returns, T observations x N strategies.
    Split rows into BLOCKS=8 contiguous blocks; for each of the 70
    combinations C of 4 blocks, train on the rows in C and test on the
    complement; rank the N strategies by train-period Sharpe and let n* be
    the argmax; omega = average-rank_test(n*) / (N + 1); lambda =
    logit(omega) clipped to +/-10; PBO = #{lambda <= 0} / 70. A train-best
    strategy must rank strictly ABOVE the test-half median to avoid counting
    as overfit. Requires T >= 8 (one row per block) and N >= 2.
    """
    x = np.asarray(logret_matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"logret_matrix must be 2-D (T, N), got shape {x.shape}")
    t, n = x.shape
    if t < BLOCKS:
        raise ValueError(f"need at least {BLOCKS} rows for {BLOCKS} blocks, got {t}")
    if n < 2:
        raise ValueError(f"need at least 2 strategy columns for a ranking, got {n}")

    blocks = np.array_split(np.arange(t), BLOCKS)
    all_idx = np.arange(t)
    overfit = 0

    for combo in _COMBOS:
        train_idx = np.concatenate([blocks[i] for i in combo])
        test_idx = np.setdiff1d(all_idx, train_idx, assume_unique=False)
        train_sr = _sharpes(x[train_idx])
        test_sr = _sharpes(x[test_idx])
        n_star = int(np.argmax(train_sr))
        omega = _average_rank(test_sr, n_star) / (n + 1.0)
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        lam = max(-10.0, min(10.0, math.log(omega / (1.0 - omega))))
        if lam <= 0.0:
            overfit += 1
    return overfit / len(_COMBOS)


def _sharpes(x: np.ndarray) -> np.ndarray:
    """Per-column Sharpe (mean / std, ddof=1) of a (T, N) matrix."""
    sd = np.std(x, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = np.mean(x, axis=0) / sd
    return sr


def _average_rank(values: np.ndarray, idx: int) -> float:
    """1-based average rank of values[idx], ascending (worst=1, best=N)."""
    v = values[idx]
    less = float(np.sum(values < v))
    ties = float(np.sum(values == v))
    return less + (ties + 1.0) / 2.0


def hlz_haircut_sharpe(sr_hat: float, n_trials: int) -> float:
    """Harvey-Liu-Zhu (2016) haircut Sharpe, in Sharpe units (a number, not a
    probability). The lane contract pins this two-argument form with the
    unit-information benchmark:

        HLZ = SR_hat - z_{1 - 1/(2*n_trials)} * sqrt(Var(SR)),  Var = 1

    i.e. the expected-max-null adjustment at the Bonferroni-style two-sided
    quantile, delivered as a level shift of the point estimate. For
    n_trials <= 1 the quantile sits at level 1/2 so z = 0 and the haircut is
    exactly zero — an un-selected strategy carries no multiple-testing
    penalty. The adjustment is always non-negative in magnitude; a positive
    sr_hat at very high trial counts CAN be deflated below zero, which is the
    statistic working as designed, not a bug.
    """
    if n_trials <= 1:
        return float(sr_hat)
    z = float(norm.ppf(1.0 - 1.0 / (2.0 * n_trials)))
    return float(sr_hat - z)


def hlz_t_stat(sr_hat: float, n_obs: int, n_trials: int) -> float:
    """HLZ expressed as a t-statistic: haircut SR / iid SE 1/sqrt(n_obs - 1).

    The Lane-1 gate's "HLZ t > 3.0" bar evaluates this helper. It is NOT part
    of the shared two-signature contract; Lanes 2+ compose it from
    `hlz_haircut_sharpe` and their own n_obs.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    return hlz_haircut_sharpe(sr_hat, n_trials) / (1.0 / math.sqrt(n_obs - 1))
