---
type: recon
date: 2026-09-23
time: 00:20 PDT
agent: opencode
model: glm-5.3-flash
trigger: "W4 estimator A/B (build-tasks 2026-09-22): same frame, LightGBM vs CatBoost"
head: a1a60a1 (lab v1) + uncommitted lab/w1-w4 working tree
scope: lab run only — no production model, config, or live path touched
related:
  - handoffs/2026-09-22_feature-lab-v2.md
  - handoffs/2026-09-22_feature-lab-v2-build-tasks.md
  - recons/2026-09-23_lab-v3-base-control.md (the LightGBM arm's own report)
---

# Feature lab — estimator A/B: LightGBM vs CatBoost (W4)

## Question

Is the incumbent estimator the bottleneck? The 2026-09-14 edge-budget work
concluded the lab's edge (~+0.045R) is a third of what any bracket geometry
needs (~0.09R), with **a genuinely different feature/target design the only
untested lever** — but that conclusion assumed the estimator was not the
problem. The 2026-09-13 H4/CatBoost A/B was confounded (window shift +
metals), so this is the controlled re-measurement: the identical seed spec
under both estimators, everything else held fixed.

## Method (what "same" means here)

- Spec: the seed `v3_base_control` (production 17 features, served geometry
  2.0x/4.0x/45, cost table off), unchanged.
- Frame: **one cached frame shared by both arms** — content hash
  `aa4acb5d6958e3e4`, `frame_from_cache: true` on the second arm. The W1 fix
  (this branch) excludes `gate.model_family` from the frame hash, so the
  estimator A/B cannot fork the frame.
- The gate's `make_classifier` seam (`core/retrainer/_common.py:410-416`) was
  the only thing that needed touching, and only in the LAB's launcher, not the
  retrainer: the CLI's `_apply_model_family` used to overwrite
  `MODEL_FAMILY=catboost` with the spec's pinned `lightgbm` (the seed specs are
  lightgbm contracts), and `run_gate`'s guard refused the mismatch. Both now
  treat an **env-selected family as the run's own arm** (precedence: explicit
  `--model-family` flag > env `MODEL_FAMILY` > the spec's declared family),
  loudly, with the bare spec-contract refusal still intact for spec files
  (`tests/test_lab_gate.py`).
- Cost accounting identical: flat toll both arms (the served configuration).

## Results

| metric | LightGBM (default) | CatBoost (`MODEL_FAMILY=catboost`) |
|---|---:|---:|
| content hash | `aa4acb5d6958e3e4` | **same** |
| gate verdict | FAIL | FAIL |
| pooled OOS trades | 30 | 94 |
| pooled OOS wins | 13 (43.3%) | 17 (18.1%) |
| pooled base rate | 0.2540 | 0.2540 |
| **edge over random** | **+0.1793** | **−0.0731** |
| pooled PF lower bound | 0.7727 | 0.2692 |
| fold-3 PF lower bound | 0.5965 | 0.1210 |
| mean Brier | 0.0857 | 0.2502 |
| mean EV | +0.3655 | −0.3324 |
| Angel bar (calibrated) | 0.3564 | 0.2948 |
| Devil bar | 0.6600 | 0.2800 |
| gate wall time | 14.5s | 33.7s |

Per-fold evidence (proposed → approved, win rate):

| fold | LightGBM | CatBoost |
|---|---|---|
| 1 | 2 → 2 (100%) | 15 → 13 (76.9%) |
| 2 | 14 → 9 (100%) | 23 → 22 (63.6%) |
| 3 | 24 → 19 (79.0%) | 62 → 59 (54.2%) |

Live-gated backtest (Fold-3 placeholder models — indicative only, both arms
failed the gate): LightGBM 74 trades / net EV +0.2548R / net PF 1.46; CatBoost
189 trades / net EV −0.2102R / net PF 0.73.

## Is the delta distinguishable from sampling noise on this trade count?

**No.** 30 vs 94 pooled trades: the LightGBM arm's +0.1793 edge over random
carries a Clopper-Pearson 90% one-sided win-rate interval wide enough to cover
zero-to-double its point (this is the same +0.179-on-30-trades reading the
2026-09-21 build report already withdraws as the known small-sample artifact —
`edge_over_random` is in win-rate units and the EV-max Angel bar confines the
gate to a thin top). CatBoost's −0.0731 on 94 trades is firmer (more evidence)
but its own interval also spans zero at this count. What IS settled, and is
not noise-dependent: **neither arm produces a promotable `edge_over_random`**
— CatBoost's is NEGATIVE with 3× the trade count of the incumbent, its pooled
PF lower bound (0.2692) is farther below the 1.2 bar than LightGBM's (0.7727),
its mean Brier is ~3× worse (0.2502 vs 0.0857), and every fold's EV is
negative (−0.31 / −0.05 / −0.64) where LightGBM's are positive. More trades at
a worse-than-random win rate is the opposite of "the estimator was the
bottleneck".

## Verdict (the handoff's decision rule, applied)

> **Decision rule for the record:** if CatBoost does not produce a promotable
> `edge_over_random` on the same frame + geometry + cost accounting, then a
> bespoke architecture harness is *not* the next lever.

CatBoost did not produce one — it measured **worse than random** (−0.0731 on
94 pooled OOS trades against a 0.2540 base rate) on the identical frame,
geometry and cost accounting. **The verdict is therefore: a bespoke
architecture harness is NOT the next lever. The answer is features (or
targets), consistent with the 2026-09-14 edge-budget conclusion.** The
`MODEL_FAMILY` A/B is now measured and closed for this frame; the ablation
verb (W3, built this session) is the tool the "features" answer starts with.

## Caveats

- Both arms FAILED the gate; the models replayed in the backtest section are
  Fold-3 placeholders, not promoted artifacts. The gate-table numbers (trades,
  wins, edge, PF bounds) are the evidence; the backtest row is context.
- `edge_over_random` is in win-rate units, not R. LightGBM's +0.1793 on 30
  trades is the withdrawn small-sample artifact (the report's own Caveats
  section says so); the honest headline is "both arms fail the gate, CatBoost
  with strictly worse evidence on more trades".
- CatBoost's lower Angel bar (0.2948 vs 0.3564) is the calibration responding
  to a worse-scored population, not a selectivity gain — the Devil bar
  collapsed with it (0.2800 vs 0.6600).
- One run per arm, 3 folds each, identical seeds in the retrainer. Two runs is
  the A/B; per-fold windows are identical by construction (same frame, same
  date-based fold boundaries), so the arms are paired — but single runs of
  anything stochastic are noise at the fold level and the deltas above are
  read accordingly.

## Command

```bash
PYTHONPATH=src:. python -m lab.cli run --name v3_base_control
MODEL_FAMILY=catboost PYTHONPATH=src:. python -m lab.cli run --name v3_base_control
```

## Interpretation

_To be written after reading the numbers above._ (Draft: the incumbent
estimator is not the bottleneck — the incumbent is the better of the two by
every measured column, and neither is promotable. The architecture axis is
closed at the estimator-swap level; remaining levers are features/targets, and
the lab's ablation verb is the tool for the next question.)