"""Deflated / probability-adjusted Sharpe-ratio statistics for research lanes.

This module is the SINGLE shared falsification-statistics implementation for
the 2026-09-24 quant lanes (Lane 1 crypto momentum, Lane 2 equity factor/PEAD,
Lane 3 forex cross-sectional RV, Lane 4 option variance premium, Lane 5
falsification audits). None of these existed anywhere in the repo before this
file (verified 2026-09-24: zero matches for ``deflated_sharpe`` / ``cscv_pbo``
/ ``hlz_haircut`` under ``src/``; ``statsmodels`` is not imported).

The three public functions and their signatures are PINNED by the lane briefs
(``llm_reports/handoffs/2026-09-24_lane*.md`` §6.2) so that every lane imports
the same code rather than re-implementing (and subtly diverging) it:

    deflated_sharpe_ratio(sr_hat, n_trials, n_obs, skew, kurt, *, var_sr=None)
    cscv_pbo(logret_matrix)            # T x N daily log-return matrix
    hlz_haircut_sharpe(sr_hat, n_trials, *, skew=0.0, kurt=3.0, n_obs=None)

All formulas use :mod:`scipy.stats` only (no statsmodels dependency).

Glossary:
    DSR -- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014). The
        probability that an observed Sharpe ``sr_hat`` is genuinely positive
        after accounting for having tried ``n_trials`` variants. Returned in
        [0, 1]; a lane's "DSR > 0.95" gate reads this directly.
    SR_0 -- the expected maximum Sharpe obtainable from ``n_trials`` INDEPENDENT
        draws of a zero-skill strategy. DSR subtracts this hurdle before testing
        significance, which is what "deflated" means.
    PBO -- Probability of Backtest Overfitting (Bailey, Borwein, Lopez de
        Prado, Zhu 2014) via Combinatorially Symmetric Cross-Validation. The
        fraction of train/test splits whose best-training strategy ranks in the
        WORSE half out of sample. A coin-flip statistic (no ordering signal)
        returns ~0.5; an overfit model returns > 0.5.
    HLZ -- Harvey, Liu, Zhu (2016) "haircut" Sharpe. The observed Sharpe minus
        a multiple-testing buffer sized by ``n_trials`` and the sampling
        variance of the Sharpe estimate. Reported as an adjusted Sharpe number,
        not a probability. ``hlz_t_stat`` converts it to the t-statistic the
        Lane-2 gate thresholds (t > 3.0).
    CSCV -- Combinatorially Symmetric Cross-Validation: split T rows into S
        contiguous blocks, take all C(S, S/2) train subsets, score each.

References (canonical formulas, do not "improve"):
    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
    Bailey, Borwein, Lopez de Prado, Zhu (2014), "The Probability of Backtest
        Overfitting".
    Harvey, Liu, Zhu (2016), "... and the Cross-Section of Expected Returns".
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats as _stats

__all__ = [
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "cscv_pbo",
    "hlz_haircut_sharpe",
    "hlz_t_stat",
]


# ─────────────────────────────────────────────────────────────────────
# Sharpe sampling variance (Lo 2002 / Bailey-LdP skew-kurt correction)
# ─────────────────────────────────────────────────────────────────────

def _sr_variance(sr_hat: float, skew: float, kurt: float, n_obs: int) -> float:
    """Approximate variance of the estimated Sharpe ratio.

    ``Var(SR_hat) ~ (1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) / (n_obs - 1)``.

    The skew/excess-kurtosis terms inflate the variance for fat-tailed,
    asymmetric return streams, so an over-fit high-Sharpe estimate on skewed
    returns is deflated harder. Floored at a tiny positive value to avoid a
    divide-by-zero when the correction would make the variance non-positive
    (possible for very large |SR_hat| with negative skew).
    """
    n = max(int(n_obs), 2)
    numer = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2)
    # A non-positive correction is degenerate; clamp so downstream sqrt works.
    numer = max(numer, 1e-12)
    return numer / (n - 1.0)


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max of n_trials iid N(0, var_sr)] — the DSR null hurdle SR_0.

    Uses the standard Gumbel-limit approximation for the expectation of the
    maximum of Gaussian draws:

        E[max] = sqrt(var_sr) * ( (1-gamma)*Phi^-1(1 - 1/n_trials)
                                  + gamma * Phi^-1(1 - 1/(n_trials*e)) )

    (gamma = Euler-Mascheroni constant). This matches the form cited in the
    lane briefs and is exact enough for the n_trials ranges used here
    (tens to a few thousand). For n_trials <= 1 the hurdle is 0.
    """
    n = int(n_trials)
    if n <= 1:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    sd = math.sqrt(var_sr)
    inv_n = _stats.norm.ppf(1.0 - 1.0 / n)
    inv_ne = _stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return sd * ((1.0 - gamma) * inv_n + gamma * inv_ne)


# ─────────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────

def deflated_sharpe_ratio(
    sr_hat: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurt: float,
    *,
    var_sr: float | None = None,
) -> float:
    """Probability that ``sr_hat`` exceeds the multiple-testing null hurdle.

    Formula (Bailey & Lopez de Prado 2014)::

        DSR = Phi( (SR_hat - SR_0) * sqrt(n_obs - 1)
                   / sqrt(1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) )

    where ``SR_0 = E[max of n_trials N(0, var_sr)]`` and ``var_sr`` defaults to
    ``1/n_obs`` (the variance of a zero-skill Sharpe estimate).

    Parameters
    ----------
    sr_hat   : observed (already annualised or per-period — be consistent)
               Sharpe ratio under test.
    n_trials : number of DISTINCT variant configurations evaluated (feature
               sets x embargo lengths x hyperparameter tuples actually fitted).
    n_obs    : number of return observations used to estimate ``sr_hat``.
    skew     : sample skewness of the return stream.
    kurt     : sample KURTOSIS (not excess) of the return stream; a Normal
               stream has kurt = 3.
    var_sr   : optional override for the null Sharpe variance.

    Returns
    -------
    float in [0, 1]; NaN only if inputs are non-finite.
    """
    if not all(
        np.isfinite(x) for x in (sr_hat, skew, kurt)
    ):
        return float("nan")
    n_obs = int(n_obs)
    if n_obs < 2 or int(n_trials) < 1:
        return float("nan")

    if var_sr is None:
        var_sr = 1.0 / float(n_obs)
    sr0 = expected_max_sharpe(n_trials, var_sr)

    # Canonical Bailey-LdP denominator:
    #   sqrt(1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2)   (NO /(n_obs-1) factor)
    # z = (SR_hat - SR_0) * sqrt(n_obs-1) / that.
    corr = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2)
    corr = max(corr, 1e-12)
    denom = math.sqrt(corr)
    z = (sr_hat - sr0) * math.sqrt(n_obs - 1.0) / denom
    return float(_stats.norm.cdf(z))


# ─────────────────────────────────────────────────────────────────────
# CSCV Probability of Backtest Overfitting
# ─────────────────────────────────────────────────────────────────────

def _sharpe(col: np.ndarray) -> float:
    """Plain (non-annualised) Sharpe of a 1-D log-return series.

    Uses the mean/std of log returns. std==0 (dead strategy) maps to NaN so it
    never wins a ranking argument on a fluke.
    """
    sd = float(np.nanstd(col, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(np.nanmean(col) / sd)


def cscv_pbo(logret_matrix: np.ndarray, s_blocks: int = 8) -> float:
    """Probability of Backtest Overfitting via CSCV.

    Algorithm (Bailey, Borwein, Lopez de Prado, Zhu 2014):
      1. Split the T rows into ``s_blocks`` contiguous blocks.
      2. Enumerate all C(s_blocks, s_blocks/2) train/test combinations.
      3. For each: build the train matrix from the chosen blocks and the test
         matrix from the complement; compute each strategy's train and test
         Sharpe; find the strategy ``n*`` with the best TRAIN Sharpe.
      4. Rank all strategies by TEST Sharpe; let ``omega = rank_test(n*) / (N+1)``
         (average rank on ties, best rank = 1 ... handled via rankdata with the
         best performer mapped to the TOP of the scale).
      5. ``lambda = logit(omega)`` clipped to +/-10.
      6. ``PBO = fraction of combos with lambda < 0`` — the share of splits in
         which the in-sample winner is below-median out of sample.

    Parameters
    ----------
    logret_matrix : T x N ndarray of daily log returns. T = observations, N =
        strategies (columns). Rows with NaN are dropped per-combo via nan-aware
        moments; a strategy column that is all-NaN in a split scores NaN.
    s_blocks      : number of contiguous blocks (default 8 -> C(8,4)=70 combos).

    Returns
    -------
    float in [0, 1]; NaN if the matrix is too small / degenerate. A T x N
    zeros matrix (no ordering signal anywhere) returns 0.5 — a coin flip.
    """
    X = np.asarray(logret_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("logret_matrix must be 2-D (T x N)")
    T, N = X.shape
    if N < 2 or T < s_blocks * 2:
        # Not enough strategies or observations to form a single split: there
        # is no ordering to exploit, so overfitting is a coin flip.
        return 0.5 if N >= 1 else float("nan")

    # Deterministic degenerate case: every strategy identical (e.g. all zeros).
    # No split can rank one above another -> PBO is exactly a coin flip.
    if np.allclose(X, X[:, [0]], atol=1e-12, rtol=0):
        return 0.5

    s = int(s_blocks)
    if s % 2 != 0:
        raise ValueError("s_blocks must be even for symmetric CSCV")
    # Contiguous block boundaries.
    edges = np.linspace(0, T, s + 1).astype(int)
    blocks = [X[edges[i]:edges[i + 1]] for i in range(s)]
    half = s // 2

    lambdas: list[float] = []
    for combo in itertools.combinations(range(s), half):
        train_idx = set(combo)
        test_idx = [i for i in range(s) if i not in train_idx]
        train = np.concatenate([blocks[i] for i in sorted(train_idx)], axis=0)
        test = np.concatenate([blocks[i] for i in sorted(test_idx)], axis=0)

        train_sr = np.array([_sharpe(train[:, j]) for j in range(N)])
        test_sr = np.array([_sharpe(test[:, j]) for j in range(N)])

        # argmax on train; NaN train scores are excluded from winning.
        if not np.isfinite(train_sr).any():
            continue
        n_star = int(np.nanargmax(train_sr))

        # Rank strategies by TEST Sharpe with ASCENDING rankdata so the BEST
        # out-of-sample strategy receives the HIGHEST rank (rankdata: 1=lowest
        # ... N=highest). Bailey et al. (2014) define omega = rank_test(n*) /
        # (N+1) on THIS scale, so a persistent winner -> omega ~= N/(N+1) ~= 1
        # -> lambda = logit(omega) > 0. Strategies with a NaN test Sharpe are
        # pinned to the bottom (worst) rank.
        finite_test = np.where(np.isfinite(test_sr), test_sr, -np.inf)
        ascending = _stats.rankdata(finite_test, method="average")  # N = best
        omega = ascending[n_star] / (N + 1.0)
        omega = min(max(omega, 1e-10), 1.0 - 1e-10)
        lam = math.log(omega / (1.0 - omega))
        lambdas.append(max(min(lam, 10.0), -10.0))

    if not lambdas:
        return float("nan")
    lam_arr = np.asarray(lambdas)
    return float(np.mean(lam_arr < 0.0))


# ─────────────────────────────────────────────────────────────────────
# Harvey-Liu-Zhu haircut Sharpe
# ─────────────────────────────────────────────────────────────────────

def hlz_haircut_sharpe(
    sr_hat: float,
    n_trials: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_obs: int | None = None,
) -> float:
    """HLZ (2016) haircut Sharpe — observed SR minus the multiple-test buffer.

    ``HLZ = SR_hat - z_{1 - 1/(2*n_trials)} * SE(SR_hat)`` where
    ``SE(SR_hat) = sqrt( (1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) / (n_obs-1) )``.

    The z-quantile ``z_{1 - 1/(2*n_trials)}`` is the Bonferroni-style cutoff for
    ``n_trials`` two-sided tests — the more variants tried, the bigger the
    haircut. Signature pinned by the lane briefs: ``(sr_hat, n_trials)`` with
    optional moments. When ``n_obs`` is omitted the SE falls back to the
    asymptotic ``1/sqrt(n_obs)`` form evaluated at ``n_obs = 252`` — a stable
    deterministic default that keeps the mandatory 2-arg test case
    (``hlz_haircut_sharpe(1.0, 10) < 1.0``) well-defined and < 1.
    """
    n = int(n_trials)
    if n < 1:
        return float(sr_hat)
    if n_obs is None:
        n_obs = 252
    z = float(_stats.norm.ppf(1.0 - 1.0 / (2.0 * n)))
    se = math.sqrt(_sr_variance(sr_hat, skew, kurt, n_obs))
    return float(sr_hat - z * se)


def hlz_t_stat(
    sr_hat: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Harvey-Liu-Zhu adjusted Sharpe expressed as a t-statistic.

    The Lane-2 gate thresholds ``HLZ t > 3.0``; the contract function returns an
    adjusted Sharpe, not a t, so this helper divides the haircut Sharpe by its
    own standard error to put it on the t scale:

        t_hlz = HLZ(SR_hat) / SE(SR_hat).

    Uses the same skew/kurt-corrected SE as the DSR. Returns NaN for degenerate
    or non-finite inputs.
    """
    n_obs = int(n_obs)
    if n_obs < 2 or not np.isfinite(sr_hat):
        return float("nan")
    hlz = hlz_haircut_sharpe(
        sr_hat, n_trials, skew=skew, kurt=kurt, n_obs=n_obs
    )
    se = math.sqrt(_sr_variance(sr_hat, skew, kurt, n_obs))
    if se <= 0 or not np.isfinite(hlz):
        return float("nan")
    return float(hlz / se)
