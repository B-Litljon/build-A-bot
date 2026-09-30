"""
Cross-lane falsification statistics, shared by the 2026-09-24 quant-lane
dispatches (Lanes 1-5).

The lead architect pinned these three signatures in Lane 1's brief (section
6.2) so that every lane reports the SAME multiple-testing statistics on
identical definitions. Do not change the signatures or the formulas here;
if a correction is needed, coordinate on
``llm_reports/m2m-prompts/2026-09-24_quant-lanes.md`` first.

Glossary:
    DSR (deflated_sharpe_ratio) -- probability in [0,1] that the observed
        Sharpe is above the expected max of n_trials null Sharpe draws
        (Bailey & Lopez de Prado 2014). Carries the skew/kurt variance
        correction, so a short-vol strategy's negative skew tightens it.
    SR_0 / E[max(SR)] -- the null hurdle: expected maximum Sharpe of
        ``n_trials`` independent Normal(0, var_sr) draws, approximated as
        sqrt(var_sr) * ((1-gamma)*Phi^-1(1-1/T) + gamma*Phi^-1(1-1/(T*e))).
    cscv_pbo -- Probability of Backtest Overfitting from the
        Combinatorially Symmetric Cross-Validation scheme (Bailey, Borwein,
        Lopez de Prado, Zhu 2014). Fraction of the 70 train/test splits in
        which the in-sample best strategy ranks below median out-of-sample.
    hlz_haircut_sharpe -- the multiple-testing-adjusted Sharpe bound
        (Harvey, Liu, Zhu 2016): SR_hat minus a haircut of
        z_{1-1/(2*n_trials)} standard errors. Reported as an adjusted SR
        number, not a probability.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649015329
_CSCV_BLOCKS = 8  # S in Bailey et al. (2014); C(8,4) = 70 partitions


def deflated_sharpe_ratio(
    sr_hat: float, n_trials: int, n_obs: int,
    skew: float, kurt: float, *, var_sr: float | None = None,
) -> float:
    """
    Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    ``sr_hat`` is the observed (annualised or per-bar) Sharpe; ``n_trials`` is
    every distinct configuration *evaluated*, not the grid thought about;
    ``kurt`` is non-excess kurtosis of returns (Normal = 3). Returns a
    probability in [0, 1]; the gate convention is DSR > 0.95.
    """
    if n_obs < 2:
        raise ValueError("n_obs < 2: no variance of a Sharpe is defined")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if var_sr is None:
        var_sr = 1.0 / n_obs
    if n_trials == 1:
        sr_0 = 0.0
    else:
        t = float(n_trials)
        # E[max of n_trials Normal(0, var_sr)] approximations.
        e_max = (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / t) + _EULER_GAMMA * norm.ppf(
            1.0 - 1.0 / (t * math.e)
        )
        sr_0 = math.sqrt(var_sr) * e_max
    var_term = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat * sr_hat
    if var_term <= 0.0:
        return 0.0
    z = (sr_hat - sr_0) * math.sqrt(n_obs - 1.0) / math.sqrt(var_term)
    return float(norm.cdf(z))


def cscv_pbo(logret_matrix: "np.ndarray") -> float:
    """
    Input: T x N daily log-return matrix, T = observations, N = strategies.
    Returns the PBO scalar.

    Splits rows into S=8 contiguous blocks, enumerates all C(8,4)=70
    train/test partitions, and counts the share where the in-sample argmax
    strategy lands on the below-median half of the out-of-sample ranking
    (lambda = logit(rank/(N+1)) < 0, clipped at +/-10). N = 1 is degenerate:
    the sole strategy always ranks 1 of 1 on both sides, returning 0.5.
    """
    m = np.asarray(logret_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("logret_matrix must be 2-D (T x N)")
    t_rows, n_strat = m.shape
    if t_rows < _CSCV_BLOCKS:
        raise ValueError(f"need >= {_CSCV_BLOCKS} observations, got {t_rows}")
    if n_strat < 1:
        raise ValueError("need >= 1 strategy")
    m = np.nan_to_num(m, nan=0.0)
    bounds = np.linspace(0, t_rows, _CSCV_BLOCKS + 1, dtype=int)
    blocks = [m[bounds[i]:bounds[i + 1]] for i in range(_CSCV_BLOCKS)]

    def _sharpe(sub: "np.ndarray") -> "np.ndarray":
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0, ddof=1)
        return np.divide(mu, sd, out=np.zeros_like(mu), where=(sd > 0))

    lambdas = []
    rank_axis = np.arange(1, n_strat + 1, dtype=float)

    def _pick_n_star(sr: "np.ndarray") -> int:
        """Train-side argmax with a stable first-index tie-break.

        ``+0.0 == -0.0`` in IEEE-754, so a simple ``float`` comparison cannot
        split tie groups on a zero-flat matrix — but numpy's argsort CAN order
        signed zeros inconsistently, which made an all-zero tie look
        non-degenerate. Canonicalize exact zeros first, then a tie counts as
        within ``1e-12`` of the max and the first such index wins.
        """
        sr = np.where(sr == 0.0, 0.0, sr)
        best = float(np.max(sr))
        tied = np.flatnonzero(np.isclose(sr, best, atol=1e-12))
        return int(tied[0])

    def _avg_ranks(vals_in: "np.ndarray") -> "np.ndarray":
        """1 = lowest of N (ties share the midpoint rank)."""
        vals_in = np.where(vals_in == 0.0, 0.0, vals_in)
        order = np.argsort(vals_in, kind="mergesort")
        out = np.empty(n_strat, dtype=float)
        lo = 0
        while lo < n_strat:
            hi = lo
            while hi + 1 < n_strat and math.isclose(
                vals_in[order[hi + 1]], vals_in[order[lo]], abs_tol=1e-12
            ):
                hi += 1
            out[order[lo:hi + 1]] = rank_axis[lo:hi + 1].mean()
            lo = hi + 1
        return out

    for combo in itertools.combinations(range(_CSCV_BLOCKS), _CSCV_BLOCKS // 2):
        train_idx = np.concatenate([np.r_[bounds[i]:bounds[i + 1]] for i in combo])
        mask = np.ones(t_rows, dtype=bool)
        mask[train_idx] = False
        sr_train = _sharpe(m[train_idx])
        sr_test = _sharpe(m[mask])
        n_star = _pick_n_star(sr_train)
        omega = _avg_ranks(sr_test)[n_star] / (n_strat + 1.0)
        omega = min(max(omega, 1e-12), 1.0 - 1e-12)
        lam = math.log(omega / (1.0 - omega))
        lambdas.append(max(-10.0, min(10.0, lam)))
    # Degenerate tie: with zero ordering signal every omega is the exact
    # midpoint (lambda = 0 on all 70 partitions). A strict `lam < 0` would
    # report 0.0 ("never overfit") for total informationlessness; a `<=` would
    # report 1.0. Lane 2 pins this case to 0.5 ("no ordering signal = coin
    # flip"), which is also the only reading in which PBO means what it says:
    # the in-sample winner's OOS result is indistinguishable from a coin toss.
    if all(lam == 0.0 for lam in lambdas):
        return 0.5
    return float(np.mean([lam < 0.0 for lam in lambdas]))


def hlz_se(skew: float, kurt: float, sr_hat: float, n_obs: int) -> float:
    """
    Standard error of the Sharpe estimate with the skew/kurt correction
    (Lo 2002 / HLZ 2016 form):
    ``SE = sqrt( (1 - skew*SR + ((kurt-1)/4)*SR^2) / (n_obs-1) )``.

    Lane briefs phrase the gate as "HLZ t-stat > 3.0": that t-statistic is
    ``sr_hat / hlz_se(skew, kurt, sr_hat, n_obs)``. This helper exists for
    those callers; it does not change the pinned Lane 1 signatures.
    """
    if n_obs < 2:
        raise ValueError("n_obs < 2: no standard error of a Sharpe is defined")
    var_term = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat * sr_hat
    return math.sqrt(max(var_term, 0.0) / (n_obs - 1.0))


def hlz_haircut_sharpe(sr_hat: float, n_trials: int) -> float:
    """
    Harvey, Liu, Zhu (2016) haircut Sharpe bound.

    ``HLZ = sr_hat - z_{1-1/(2*n_trials)} * SE`` with the Gaussian
    ``SE = 1/sqrt(n_obs)`` under the Lane 1 contract, which pins this at two
    arguments with n_obs unavailable: the haircut z multiple is applied against
    a unit-information assumption, so callers that need the skew/kurt-adjusted
    standard error should use :func:`hlz_se` and form
    ``sr_hat - z * hlz_se(...)`` themselves. Returns the adjusted Sharpe
    number (a bound, not a probability). With ``n_trials`` = 1 the z is the
    median and the result is just ``sr_hat``.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    z = norm.ppf(1.0 - 1.0 / (2.0 * n_trials))
    return float(sr_hat - z)
