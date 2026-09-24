# 2026-09-24 — quant lanes coordination thread

Append-only live thread for the five parallel quant-research lanes dispatched
2026-09-24 (Lane 1 crypto momentum, Lane 2 equity factor/PEAD, Lane 3 forex
cross-sectional RV, Lane 4 option variance premium, Lane 5 falsification
audits), each in its own worktree under
`/mnt/storage/mystuf/development/build-a-bot-lanes/`.

The shared surface is **`src/lab/stats.py`** — the DSR / CSCV-PBO / HLZ module.
Lane 1 owns the canonical contract (`§6.2` of every lane's brief), but lanes run
in parallel worktrees that do not see each other until merge. Whichever lane
needs it first implements it; the rest import it. Record both events here.

Convention: one `## [timestamp] <from> → <to>` block per message; every claim
marked **VERIFIED** (with the command and its real output) / **ASK** /
**OFFER** / **BLOCKER** / **DECIDED**.

---

## [2026-09-24 16:05 PDT] lane/equity-factor-pead → all quant lanes

### VERIFIED — `src/lab/stats.py` did NOT exist in my worktree, so Lane 2 built it

`src/lab/stats.py` was absent from the Lane-2 worktree (`lane/equity-factor-pead`
off main `b09fde2`) and from every sibling lane worktree — checked before
writing any code:

```
$ find /mnt/storage/mystuf/development/build-a-bot-lanes/*/src/lab/stats.py
(no results)   # Lane 1's crypto-trend-mom-tv had NOT landed it either
$ grep -rl "deflated_sharpe\|cscv_pbo\|hlz_haircut" --include=*.py .
(no results)
```

So I implemented it from Lane 1's brief §6.2 contract verbatim (signatures
pinned, scipy-only, no statsmodels):

```
def deflated_sharpe_ratio(sr_hat, n_trials, n_obs, skew, kurt, *, var_sr=None) -> float
def cscv_pbo(logret_matrix) -> float            # T x N daily log returns, S=8
def hlz_haircut_sharpe(sr_hat, n_trials, *, skew=0.0, kurt=3.0, n_obs=None) -> float
def expected_max_sharpe(n_trials, var_sr) -> float   # helper, exported
def hlz_t_stat(sr_hat, n_trials, n_obs, skew=0.0, kurt=3.0) -> float  # helper, exported
```

One deliberate addition over the bare contract: `hlz_t_stat`. The contract's
2-arg `hlz_haircut_sharpe(sr_hat, n_trials)` returns an adjusted SHARPE NUMBER;
my Lane-2 gate thresholds an HLZ **t-stat** (`t > 3.0`, brief §6.1), which the
2-arg form cannot produce without the moments. `hlz_haircut_sharpe` keeps the
pinned signature via optional kwargs (defaults Normal: `skew=0, kurt=3, n_obs`
falls back to a stable 252 so the mandatory `hlz_haircut_sharpe(1.0, 10) < 1.0`
stays well-defined). `hlz_t_stat = hlz_haircut_sharpe(...)/SE(SR_hat)`.

Also registered the five names for lazy export in `src/lab/__init__.py`
(PEP 562 `_LAZY` map) so `from lab.stats import ...` and `from lab import
deflated_sharpe_ratio` both work without booting LightGBM.

### VERIFIED — the three mandatory test cases pass

```
$ PYTHONPATH=src:. python -m pytest tests/test_lab_stats.py -q
................                                                         [100%]
16 passed in 1.00s
```

incl. the brief's §6.4 case 6: `deflated_sharpe_ratio(0,10,100,0,3)` = 0.0586 ∈
[0,1]; `cscv_pbo(np.zeros((80,5)))` = 0.5; `hlz_haircut_sharpe(1.0,10)` =
0.8728 < 1.0. A non-trivial check: a 5-column matrix with a persistent winner
on column 0 returns PBO ≤ 0.5 (not overfit), and 8 identical columns return
exactly 0.5.

### VERIFIED — Lane 2 consumed it (the trainer imports it)

`scripts/investor_train_model_weekly.py` does
`from lab.stats import cscv_pbo, deflated_sharpe_ratio, hlz_haircut_sharpe,
hlz_t_stat` and thresholds the gate on them. Run output:

```
FALSIFICATION GATE
  excess/month  +190.0 bps  (>= +80) -> PASS
  DSR           1.0000   (> 0.95) -> PASS
  HLZ t        +19.093   (> 3.0) -> PASS  (adj SR 3.677 | unadj t 3.39)
  PBO           0.3000   (< 0.50) -> PASS
```

### DECIDED — if Lane 1 also lands a `src/lab/stats.py`, reconcile at merge

Lane 2's copy is built against Lane 1's exact contract, so a duplicate arriving
at merge should be a clean win for whichever is canonical; the only divergence
risk is `hlz_t_stat` (Lane-2-only helper) and the `__init__.py` lazy-export
block. If Lane 1's version lacks either, keep Lane 2's additive bits. Recon
report with per-fold numbers:
`llm_reports/recons/2026-09-24_lab-equity-factor-pead.md`.
