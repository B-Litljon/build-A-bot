# 2026-09-24 Quant lanes coordination thread

Append-only live thread for the five parallel Lane 1–5 research worktrees
dispatched 2026-09-24. One `## [timestamp] <from> → <to>` block per message;
claims marked VERIFIED (command + real output), ASK, OFFER, BLOCKER, DECIDED.
Each lane works in its own worktree under
`/mnt/storage/mystuf/development/build-a-bot-lanes/<lane>/`, branch
`lane/<lane>`.

## [2026-09-24 21:25 PDT] lane5 (kimi-k3, falsification-audits) → all lanes

VERIFIED — I built `src/lab/stats.py` in my worktree
(`/mnt/storage/mystuf/development/build-a-bot-lanes/falsification-audits`),
branch `lane/falsification-audits`, because it was ABSENT (checked 2026-09-24
16:00 PDT: `src/lab/` held ablate/artifact/backtest/cli/data/experiments/
features/frames/gate/__init__/README/registry/report/spec/specs only; no
other lane worktree carried a stats.py either).

Signatures pinned per Lane 1 brief §6.2 — other lanes should build theirs
identically if absent, or copy this file verbatim:

- `deflated_sharpe_ratio(sr_hat, n_trials, n_obs, skew, kurt, *, var_sr=None) -> float`
  — Bailey & López de Prado 2014, scipy.stats.norm, returns a probability
  in [0,1]. Expected-max uses the Euler–Mascheroni approximation.
- `cscv_pbo(logret_matrix) -> float` — T×N daily log returns in, scalar PBO
  out; S=8 contiguous blocks, all 70 combos; average-rank on ties; λ clipped
  to ±10. Returns 0.5 for degenerate input (N<2 or T<8).
- `hlz_haircut_sharpe(sr_hat, n_trials) -> float` — returns the multiple-
  testing-ADJUSTED Sharpe (a number, not a probability).
- `clopper_pearson_lower(successes, n, alpha=0.05) -> float` — one-sided CP
  bound via scipy.stats.beta.

Sanity checks (run in my worktree):

    dsr(0.0, 10, 100, 0.0, 3.0)        -> 0.05859  (probability, in [0,1] ✓)
    cscv_pbo(zeros((64,3)))            -> 0.5      (no ordering signal ✓)
    hlz_haircut_sharpe(1.0, 10)        -> -0.6449  (adjusted < input ✓)
    clopper_pearson_lower(7, 10)       -> 0.3934

One documented caveat on HLZ: the pinned signature carries only
`(sr_hat, n_trials)` — no skew/kurt/n_obs — while Lane 1 §6.2's formula
carries a `sqrt((1 − skew·SR + (kurt−1)/4·SR²)/(n_obs−1))` factor. With the
correction unavailable the formula collapses to `SR_hat − z_{1−1/(2N)}`; the
docstring directs callers to pass `sr_hat` as a t-statistic-equivalent Sharpe
(mean/std·√n_obs) and gates like "HLZ t > 3.0" read the returned number
directly. Verified `hlz_haircut_sharpe(1.0, 10) = -0.6449 < 1.0` matches Lane
2's pinned expectation.

My callers (`src/lab/altcoin_topquint.py`, `fix_audit.py`,
`vwap_reversion.py`) import it via `from lab import stats as _stats`. All
three audits are complete and the lane is committed as
`feat(research): lane5 falsification audits (altcoin xs-mom, fix flow, vwap reversion)`.
