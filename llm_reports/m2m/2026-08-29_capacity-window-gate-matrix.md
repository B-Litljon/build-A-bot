---
type: m2m
date: 2026-08-29
time: 17:35 PDT
agent: Claude Opus 5
model: claude-opus-5 (Claude Opus 5, claude-opus-5)
trigger: "K3 M2M brief: capacity × window gate matrix — three sequential production-gate runs, measurement only"
head: c5e7e83
branch: feat/trim-100x15-retrain
scope: measurement only (3 side-dir retrains, all rejected; no promotion, no live-system contact)
result_commit: null
related:
  - m2m/2026-08-29_trim-100x15-retrain-and-promotion.md
  - refactors/2026-08-24_artifact-holdout-gate.md
---

# Capacity × window gate matrix: the trim config silently disables the Devil

## Summary

Three sequential runs through the REAL production gate (`main()` in
`src/core/retrainer.py`), all to side directories, to answer whether the 2-year
window explains the trim config's gate failure, whether 150x31 dominates
100x15, and whether the unpinned "pass" reproduces.

**All three runs FAILED the gate (exit 2), every one on the pooled-trade floor
alone.** Beyond the trade counts, the matrix surfaced a mechanical defect that
the trade-count framing was hiding: **at `min_child_samples=80`, the Devil's
training population is too small to permit a single split, so the Devil
degenerates to a constant function and vetoes nothing.** The two-stage
architecture is running on one stage.

Separately: **the unpinned "pass" did not reproduce, and I can account for
why.** It was not a window effect — that run had lowered the gate's own trade
floor.

## Audit Log

### Commands / files

Driver created verbatim per brief: `scripts/run_gate_capacity_variant.py`
(monkeypatches `R.get_hyperparameters` from outside; `src/` untouched).
Independently verified the patch sets all three params on BOTH models
(`n_estimators=150, num_leaves=31, min_child_samples=40, max_depth=6`), and
confirmed in-run via the banner (`Angel Leaf=40/None Devil Leaf=40/None`,
vs `Leaf=80` for the vanilla runs).

Common env for all runs: `source .env`, `PYTHONPATH=src:.`, `DATA_SOURCE=oanda`,
`RETRAIN_TIMEFRAME_MINUTES=15`, `RETRAIN_DAYS_BACK=730`, behavior veto unset.

Early-abort banner check passed on all three: `OandaMarketProvider`,
`SL=2.0× TP=4.0× max_hold=45 survival=5`, zero Alpaca references.

**Strict sequencing (no concurrency at any point):**

| Run | Log window |
|---|---|
| A | 17:25:28 → 17:26:33 |
| B | 17:26:38 → 17:27:39 |
| C | 17:27:41 → 17:28:42 |

Pre-run check `ps -eo pid,cmd \| grep bin/python.*retrainer` returned clean.
(Note: the brief's `pgrep -af "core.retrainer"` self-matches the invoking
shell's own command string — it reports a false positive every time. Used a
`bin/python`-anchored pattern instead. Minor deviation, same intent.)

Windows confirmed genuinely different — pinned vs unpinned:

```
A/B (pinned):  Date range 2024-08-23 to 2026-08-23 | HOLDOUT 2026-04-12 → 2026-08-21
C (unpinned):  Date range 2024-08-30 to 2026-08-30 | HOLDOUT 2026-04-19 → 2026-08-28
```

### The matrix

Every number verbatim from the logs. Baseline row from
`logs/stability_5yr_202608{09,16,23}.log`; 5yr trim row from
`logs/retrain_trim_20260829.log`.

| Run | Config | Window | Pin | Angel proposed (F1/F2/F3) | Devil approved | Pooled vs floor | F3 macro PF | Gate |
|---|---|---|---|---|---|---|---|---|
| — | 200x63 (shipped) | 5yr | 08-09 | — | — | **238** / 232 | 1.3488 | PASS |
| — | 200x63 (shipped) | 5yr | 08-16 | — | — | **230** / 232 | 1.3333 | FAIL |
| — | 200x63 (shipped) | 5yr | 08-23 | — | — | **227** / 233 | 1.1304 | FAIL |
| — | 100x15 | 5yr | 08-23 | 11 / 3 / 11 | 100% / 100% / 100% | **24** / 233 | 1.1429 | FAIL |
| **A** | 100x15 | 2yr | 08-23 | 13 / 6 / 36 | 100% / 100% / 100% | **42** / 232 | 1.6923 | FAIL |
| **B** | 150x31 | 2yr | 08-23 | 52 / 45 / 113 | 78.8% / 71.1% / 88.5% | **90** / 232 | 2.1000 | FAIL |
| **C** | 100x15 | 2yr | *none* | 15 / 11 / 49 | 100% / 100% / 100% | **48** / 232 | 3.4000 | FAIL |

All three rejections cite one reason only — e.g. Run B:

```
Rejection: Pooled OOS trades 90 < effective floor 232
(= 300 × (1 − chop_veto_rate 22.5%)) — sample too small to trust PF=2.1000
```

Brier and EV passed everywhere. Fold-3 PF passed in A (1.69), B (2.10) and
C (3.40); it failed only in the 5yr trim run (1.14).

Artifact-holdout scorecards (diagnostic only — the fold gate failed first, so
these are Fold-3 models, not gate-passed artifacts):

| Run | Holdout PF | 95% lower bound | Brier | EV | Trades |
|---|---|---|---|---|---|
| A | 11.6000 | 5.0262 | 0.0731 | 1.9118 | 34 |
| B | 4.1739 | 2.6866 | 0.1156 | 1.4930 | 71 |
| C | 5.4000 | 2.8173 | 0.0960 | 1.7568 | 37 |

### Exit codes

B and C captured directly: `RUN_B_EXIT=2`, `RUN_C_EXIT=2`. Run A was launched
before the supervisor chain, so its code was not captured by the harness; it is
**2** by `main()`'s contract (`MODELS REJECTED` ⇒ `return 2`), and the log
carries that verdict line. No artifact directories were created for any run,
which independently corroborates all three rejections.

### Unfinished / not done

Nothing from the brief was skipped. No promotion, no shipping decision, no
live contact — as specified.

## Analysis

### Q1 — Does the 2-year window alone explain the pass/fail flip? **No.**

Holding capacity at 100x15 and moving only the window, pooled trades go
24 (5yr) → 42 (2yr pinned) → 48 (2yr unpinned). The window helps — it also
lifts Fold-3 PF from a failing 1.14 to a passing 1.69/3.40 — but 48 is still
**21% of the 232 floor**. The window is a second-order effect. It cannot
carry this config through the gate, and no run got remotely close.

### Q2 — Does 150x31 dominate 100x15 on the 2-year window? **Yes, decisively — and for a reason worth more than the trade count.**

On the same pinned window, 150x31 beats 100x15 on every axis: pooled trades
90 vs 42 (2.1×), Fold-3 macro PF 2.10 vs 1.69, Angel proposals 113 vs 36 in
Fold 3. Still only 39% of the floor, so it does not pass — but the gap is less
than half as bad.

**The real finding is qualitative.** In every 100x15 run the Devil approved
**100.0%** of proposals, and its diagnostic reads:

```
Run A, Fold 3:  Min 0.7595  Median 0.7595  Max 0.7595
                Separation Gap: +0.0000
                Verdict: NO SIGNAL -- Devil cannot distinguish wins from losses
```

Min = median = max. The Devil is emitting **one constant number for every
input**. It is not a weak model; it is not a model at all. Run C is identical
(constant 0.7724, gap 0.0000). Run B, by contrast:

```
Run B, Fold 3:  Min 0.2996  Median 0.8693  Max 0.9695
                Separation Gap: +0.0931
                Verdict: SIGNAL DETECTED -- Devil can distinguish (gap > 0.05)
```

**Mechanism, confirmed.** The Devil trains only on the Angel-approved
subpopulation, which the trim config makes tiny. LightGBM's
`min_child_samples` is a per-leaf minimum, so splitting a node requires at
least `2 × min_child_samples` rows. Observed populations:

| Run | min_child_samples | Devil train n (F1/F2/F3) | Rows needed to split | Can split? |
|---|---|---|---|---|
| 5yr trim | 80 | 17 / 50 / 25 | 160 | never |
| A | 80 | 69 / 146 / 158 | 160 | never |
| C | 80 | 76 / 172 / 123 | 160 | F2 only |
| B | 40 | 583 / 716 / 739 | 80 | always |

Reproduced in isolation on synthetic data carrying a genuinely learnable
signal — at `min_child_samples=80` the model returns a flat constant for
n=69/123/146/158, and varies across the full 0–1 range for n=583/739 at
`min_child_samples=40`:

```
 Devil train n  min_child    p_min    p_max    spread  verdict
            69         80   0.4783   0.4783  0.000000  CONSTANT (degenerate)
           158         80   0.4937   0.4937  0.000000  CONSTANT (degenerate)
           583         40   0.0001   0.9998  0.999655  varies (real model)
```

So the trim config does not merely trade less. **It disables the Devil.** The
`min_child_samples=80` value is larger than half the population the Devil is
ever given, so the second stage of a two-stage architecture silently collapses
into a constant that vetoes nothing. Every "100% approved" line in those logs
is that collapse, not a lenient Devil.

**Are 150x31's extra trades real skill or noise?** Partly real, partly
unknowable at these sizes. Real: the Devil recovering genuine discrimination
(+0.0931 separation, 71–89% approval rates that actually vary by fold) is a
structural improvement, not a sampling artifact. Unknowable: Fold-3 PF 2.10
rests on 41 trades, and the fold gate's own floor exists precisely to refuse
that inference. Note also the anti-correlation between PF and sample size
across runs — A posts the highest holdout PF (11.60) on the fewest trades
(34), B the lowest (4.17) on the most (71). That is the signature of small-
sample noise, and it is an argument for trusting B's numbers *more* despite
them looking worse.

### Q3 — Did the unpinned pass reproduce? **No — and it was not a window effect.**

Run C reproduces that run's configuration (100x15, 2yr, unpinned, same
17-feature tree) and **fails at 48 pooled trades against the 232 floor**.

The discrepancy is explained. Earlier in this session, while checking for a
concurrent retrainer per the one-at-a-time constraint, I read that process's
environment directly from `/proc/2957431/environ` and recorded:

```
RETRAIN_HTF_TIMEFRAME=1h
RETRAIN_DAYS_BACK=730
RETRAIN_TIMEFRAME_MINUTES=15
RETRAIN_POOLED_TRADE_FLOOR=40      <-- floor lowered from 300
DATA_SOURCE=oanda
```

`BASELINE_POOLED_OOS_TRADES` defaults to 300 (`retrainer.py:534`); that run
set it to **40**, giving an effective floor of ~31 instead of ~232. At ~48
pooled trades it would clear a floor of 31 comfortably. That is the entire
difference. The pass was obtained by moving the goalpost 7.5×, not by any
property of the window, the data, or the model.

Two caveats, stated plainly: the process has since exited, so this cannot be
re-verified from `/proc` now — it rests on my direct read at the time, recorded
in-session. And I am describing a configuration value, not intent; there may
have been a good reason to explore a lower floor. But the artifact it produced
is not comparable to anything else in this matrix, and it should not be
treated as a passing result.

### Cross-cutting: the floor is the binding constraint for everyone

The shipped 200x63 config clears it only marginally — 238/230/227 against
232/232/233, one pass in three. Every trim variant lands at 18–39% of it. The
gate is not narrowly rejecting the trim config; it is rejecting nearly
everything, and the shipped model is served on the strength of a single
marginal pass from three attempts.

## Next Pilot Context

**Passing artifacts in `gate_*` dirs: none.** All three runs were rejected, so
no `models/gate_*` directory was created. Nothing here is promotable.

**Live-system state, verified before and after:** `soak.service` **active**
throughout (pid 420097, uptime unbroken), `soak.off` never created,
`models/forex_m15_wide/` untouched (artifacts still 2026-08-08 22:56).
`git status --porcelain src/ tests/` clean. The only file I added to the
tracked tree is the driver script.

**Flag for whoever ships — `models/forex_m15` was overwritten today, not by
me.** Both `models/forex/` and `models/forex_m15/` now hold byte-identical
copies (separate inodes, same content, mtime 2026-08-29 16:13:23) of the
unpinned lowered-floor artifact. `models/forex_m15` previously held the
2026-07-02 production M15 model; that artifact is gone from disk. I did not
touch either directory, and the brief listed both as untouchable — noting it
because it happened outside this task and someone should know before treating
either as a reference point.

**Driver script — worth keeping, briefly.** `scripts/run_gate_capacity_variant.py`
is the only way to sweep capacity through the real gate without editing `src/`
(the params are hardcoded, no env override exists). If capacity work continues
it earns its place; if the trim direction is abandoned, delete it. It is
untracked and commits nothing.

**Caveats for the ship/no-ship decision:**

1. **Do not ship any 100x15 artifact.** Its Devil is a constant. Whatever the
   trade counts say, that config runs a two-stage architecture with one stage
   inert, and the live bot would be taking Angel proposals with no veto.
2. **The `min_child_samples=80` value is the specific defect**, not the tree
   count or leaf count. It was presumably chosen to fight overfitting on the
   *Angel*, which trains on hundreds of thousands of rows, but it is applied
   to the Devil too, which trains on dozens. Decoupling the two — or scaling
   the Devil's value to its actual population — is a smaller and better-aimed
   change than abandoning the capacity reduction. This has not been tested.
3. **150x31 is the best-measured candidate but still fails**, at 39% of the
   floor. It is not shippable as-is.
4. **The floor itself remains an open policy question** (raised in the prior
   report, unchanged here). It rejects the shipped config two times in three.
   Whether 300 is right is a judgement call about how much evidence a
   promotion needs — and it should be made deliberately and documented, not
   set per-run via an env var, which is what produced the phantom pass above.
