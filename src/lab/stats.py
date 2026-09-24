"""
Shared multiple-comparison statistics for the quant research lanes.

Falsification audit lanes (Lane 5: ``altcoin_topquint``, ``fix_audit``,
``vwap_reversion``) all need the same three multiple-testing adjustments —
there was previously NO deflated-Sharpe / CSCV / HLZ code anywhere in the
repo (verified 2026-09-24), so this module is the single shared
implementation, against the signatures pinned in the lane handoffs
(``llm_reports/handoffs/2026-09-24_lane1-crypto-trend-mom-tv.md`` §6.2,
verbatim formulas).

Glossary:
    sharpe_ratio -- mean/std of an excess-return series, annualised by
        ``periods_per_year``; the naive point estimate every adjustment below
        starts from. Annualisation is linear-scaling (√periods), which is the
        convention across the repo even though crypto trades daily.
    deflated_sharpe_ratio -- Bailey & López de Prado (2014). Probability in
        [0, 1] that an observed Sharpe ``sr_hat`` exceeds the expected max
        Sharpe of ``n_trials`` independent trials of a zero-skill strategy.
        ``skew``/``kurt`` are the sample skewness and (non-excess) kurtosis of
        the return series; ``var_sr`` defaults to 1/n_obs.
    cscv_pbo -- Bailey, Borwein, López de Prado, Zhu (2014) combinatorially
        symmetric cross-validation. Input is a T×N matrix of per-period log
        returns (T observations, N strategies). Splits rows into S=8
        contiguous blocks, enumerates all C(8,4)=70 train/test combos, asks
        how often the train-best strategy's test performance ranks below
        median. Returns the PBO scalar in [0, 1]: the estimated probability
        that a backtest-selected configuration underperforms out of sample.
    hlz_haircut_sharpe -- Harvey, Liu, Zhu (2016). Returns the haircut
        (negative) Sharpe adjustment: the expected maximum of ``n_trials``
        zero-skill Sharpe estimates, to be SUBTRACTED from sr_hat (or
        compared against the hurdle). Reported as a number, not a probability.
    clopper_pearson_lower -- one-sided CP lower bound on a Binomial success
        rate (scipy.stats.beta). Used by Audit A for the weekly
        outperformance-rate gate.
    n_trials conventions -- each lane counts the triples/configs it actually
        EVALUATED (not the grid thought about): Audit A counts each
        (lookback set, K, cadence) as one; Audit B uses the honest
        conservative n=2 (two fixes); Audit C counts the 4-point k-sweep.

All functions are pure and deterministic; the only I/O this module performs
is none.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.stats import beta, norm

__all__ = [
    "sharpe_ratio",
    "deflated_sharpe_ratio",
    "cscv_pbo",
    "hlz_haircut_sharpe",
    "clopper_pearson_lower",
]


def sharpe_ratio(returns: np.ndarray, *, periods_per_year: float = 252.0) -> float:
    """
    Annualised naive Sharpe of a return series.

    ``sr = mean(returns) / std(returns) * sqrt(periods_per_year)``, with
    std ddof=1. Returns 0.0 for a degenerate (zero-variance or empty)
    series — callers gate on it, and an uninformative 0 is the honest answer
    for "no dispersion measured".
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(r.mean() / std * math.sqrt(periods_per_year))


def _sr_variance_term(sr_hat: float, skew: float, kurt: float, n_obs: int) -> float:
    """The DSR/HLZ denominator: `1 - skew*sr + ((kurt-1)/4)*sr^2`, floored > 0."""
    term = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat * sr_hat
    # Numerically the term can dip at/below zero for a very fat-tailed
    # negative-skew series; the formulas are undefined there, so clamp to a
    # tiny positive epsilon. Better a conservative huge denominator than NaN.
    return max(term, 1e-12)


def deflated_sharpe_ratio(
    sr_hat: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurt: float,
    *,
    var_sr: float | None = None,
) -> float:
    """
    Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    DSR = Φ( (sr_hat − E[max(SR_0)]) · √(n_obs−1) /
             √(1 − skew·sr_hat + ((kurt−1)/4)·sr_hat²) )

    ``E[max(SR_0)]`` is the expectation of the maximum of ``n_trials`` draws
    from a Normal centred at 0 with variance ``var_sr``
    (default 1/n_obs). Returns the probability in [0, 1]. ``n_trials`` is the
    number of distinct configurations actually evaluated; ``n_obs`` is the
    sample length the Sharpe was estimated on.
    """
    if n_obs < 2:
        return 0.0
    if var_sr is None:
        var_sr = 1.0 / n_obs
    var_sr = float(var_sr)
    # Expected max of n iid N(0, var_sr): use the approximation
    #   E[max] ≈ sqrt(var_sr) * ( (1−γ)·Φ⁻¹(1 − 1/n) + γ·Φ⁻¹(1 − 1/(n·e)) )
    # from Bailey & López de Prado (2014), with γ the Euler–Mascheroni
    # constant. Exact for n=1 (→ 0 by the norm.ppf(1−1/1) edge handled below).
    euler_gamma = 0.5772156649015328606
    n = max(int(n_trials), 1)
    if n == 1:
        e_max = 0.0
    else:
        e_max = math.sqrt(var_sr) * (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / n)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (n * math.e))
        )
    denom = _sr_variance_term(sr_hat, skew, kurt, n_obs)
    num = (sr_hat - e_max) * math.sqrt(n_obs - 1)
    return float(norm.cdf(num / math.sqrt(denom)))


def cscv_pbo(logret_matrix: np.ndarray) -> float:
    """
    Bailey, Borwein, López de Prado, Zhu (2014) CSCV Probability of Backtest
    Overfitting.

    Input: T×N daily log-return matrix, T = observations, N = strategies.
    Returns the PBO scalar in [0, 1].

    Split rows into S=8 contiguous blocks; enumerate all C(8,4)=70 combos.
    For each combo C, stack the four train blocks and the four test blocks,
    compute per-strategy train SR and test SR (mean/std, non-annualised so
    T need not equal years), rank strategies by train SR, take the argmax
    n*, measure its test rank ω = rank_test(n*)/(N+1) (average-rank on ties),
    λ = logit(ω) clipped to ±10; PBO = (#{λ < 0}) / 70.

    With N < 2 the statistic is degenerate; returns 0.5 (no information).
    T must be >= 8 so each of the 8 blocks is at least one row.
    """
    X = np.asarray(logret_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("logret_matrix must be T×N two-dimensional")
    T, N = X.shape
    if N < 2 or T < 8:
        return 0.5
    S = 8
    # Contiguous block split — rows are assumed time-ordered.
    block_bounds = [int(round(i * T / S)) for i in range(S + 1)]
    blocks = [X[block_bounds[i] : block_bounds[i + 1]] for i in range(S)]

    n_lambdas_negative = 0
    n_combos = 0
    for combo in combinations(range(S), S // 2):
        train_idx = sorted(combo)
        test_idx = sorted(set(range(S)) - set(combo))
        train = np.concatenate([blocks[i] for i in train_idx], axis=0)
        test = np.concatenate([blocks[i] for i in test_idx], axis=0)

        def _per_strategy_sr(M: np.ndarray) -> np.ndarray:
            mu = M.mean(axis=0)
            sd = M.std(axis=0, ddof=1)
            # Zero-variance strategy has undefined SR; rank it last by
            # assigning -inf (never wins a train argmax on skill).
            with np.errstate(divide="ignore", invalid="ignore"):
                sr = np.where(sd > 0, mu / sd, -np.inf)
            return sr

        train_sr = _per_strategy_sr(train)
        test_sr = _per_strategy_sr(test)
        if not np.isfinite(train_sr).any():
            continue
        # Argmax of train SR, ties broken by lowest strategy index (stable).
        n_star = int(np.argmax(train_sr))
        # Average-rank of n_star in the TEST distribution, 1..N.
        order = np.argsort(test_sr, kind="stable")
        ranks = np.empty(N, dtype=float)
        ranks[order] = np.arange(1, N + 1)
        # Average the ranks of ties with n_star.
        ties = test_sr == test_sr[n_star]
        omega = float(ranks[ties].mean()) / (N + 1.0)
        omega = min(max(omega, 1e-10), 1.0 - 1e-10)
        lam = math.log(omega / (1.0 - omega))
        lam = max(-10.0, min(10.0, lam))
        if lam < 0.0:
            n_lambdas_negative += 1
        n_combos += 1
    if n_combos == 0:
        return 0.5
    return n_lambdas_negative / n_combos


def hlz_haircut_sharpe(sr_hat: float, n_trials: int) -> float:
    """
    Harvey, Liu, Zhu (2016) haircut Sharpe — returns the multiple-testing
    ADJUSTED Sharpe (a number in SR units, not a probability).

    The pinned formula (Lane 1 brief §6.2) is

        HLZ = SR_hat − z_{1−1/(2·n_trials)}
                  · sqrt( (1 − skew·SR_hat + ((kurt−1)/4)·SR_hat²) / (n_obs−1) )

    but the pinned SIGNATURE carries only ``(sr_hat, n_trials)`` — no skew,
    kurt or n_obs. With the correction term unavailable, the
    variance-of-SR factor must be absorbed into the caller's units: pass
    ``sr_hat`` as a **t-statistic-equivalent Sharpe**

        sr_hat = mean(r) / std(r) · sqrt(n_obs)

    for which that correction is exactly 1 under normality, and then

        HLZ = sr_hat − z_{1−1/(2·n_trials)}      (adjusted SR, t-stat units)

    ``z`` is ``scipy.stats.norm.ppf(1 − 1/(2·n_trials))``. Audits reading
    this against a "HLZ t > 3.0" gate therefore compare the returned number
    directly — when the input is in t units the output is an HLZ t-stat.
    For the full finite-moment correction use ``deflated_sharpe_ratio``.
    """
    n = max(int(n_trials), 1)
    z = norm.ppf(1.0 - 1.0 / (2.0 * n))
    return float(sr_hat - z)


def clopper_pearson_lower(successes: int, n: int, alpha: float = 0.05) -> float:
    """
    One-sided Clopper–Pearson lower bound on a Binomial success rate.

    ``scipy.stats.beta.ppf(alpha, successes, n − successes + 1)``; returns 0.0
    when successes == 0. Used by Audit A's "beat both benchmarks" gate: the
    strategy must beat the equal-weight basket AND BTC buy-and-hold with the
    one-sided 95% CP lower bound on the weekly outperformance rate > 0.
    """
    if n <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    successes = min(int(successes), int(n))
    return float(beta.ppf(alpha, successes, n - successes + 1))
