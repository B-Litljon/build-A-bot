---
type: recon
date: 2026-09-21
time: 16:38 PDT
agent: opencode
model: deepseek-flash
trigger: "Feature-lab run spread_table_control (spec hash a5be9548e73d392a)"
head: 14fe328
scope: lab run only — no production model, config, or live path touched
related:
  - handoffs/2026-09-21_feature-lab-plan.md
---

# Feature lab — spread_table_control

## Spec

- content hash: `a5be9548e73d392a` (frame cache key)
- feature families: `v3_base`
- symbols: AUD_JPY, EUR_JPY, GBP_JPY, NZD_JPY, GBP_AUD, GBP_NZD
- window: 730 days @ M15, htf 1h
- geometry: 2.0x/4.0x/45 bars
- labels: kind=survival, survival_bars=5
- spread table: on (config/spread_alphas_m15.json)
- estimator: lightgbm

## Frame

- rows: 174,730 | features: 18 | chop/behavior veto drop: 41.21% | unresolvable tail purged: 74
- loaded from frame cache

## Gate (retrainer `validate_candidate`)

- verdict: **FAIL**
- mean Brier 0.2441 | mean EV 0.071053 | pooled trades 40 | pooled wins 14
- pooled PF lower bound 0.5824 | fold-3 PF lower bound 0.5965
- pooled base rate 0.2495 | **edge over random 0.1005**
- production thresholds: Angel 0.3723, Devil 0.1000
- gate wall time: 7.4s

| fold | train | val | Brier | EV | proposed | approved | win rate | base rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 86,888 | 29,497 | 0.1957 | 0.200000 | 5 | 5 | 80.00% | 0.2459 |
| 2 | 116,385 | 29,842 | 0.2740 | -0.250000 | 16 | 16 | 50.00% | 0.2604 |
| 3 | 146,227 | 28,423 | 0.2628 | 0.263158 | 19 | 19 | 68.42% | 0.2418 |

Rejection reasons:

- Fold 3 PF point estimate 1.4545 on 19 trades, but 95% lower bound 0.5965 < 1.2 — the most recent regime cannot prove it beats break-even
- Pooled fold PF 95% lower bound 0.5824 < 1.2 (evidence: 14 wins / 40 trades across 3 folds)

## Backtest (live-gated replay)

- toll pricing: spread_table | trades 72 | wins 34 | win rate 47.22%
- gross EV 0.5769R | net EV 0.4251R | net PF 1.8140 | max drawdown 8.9015R
- gate veto funnel: regime=18, spread=6

| symbol | trades | win rate | net EV (R) | net PF |
|---|---:|---:|---:|---:|
| AUD_JPY | 22 | 59.09% | 0.7611 | 2.8609 |
| EUR_JPY | 14 | 57.14% | 0.8501 | 3.6804 |
| GBP_AUD | 3 | 0.00% | -1.2779 | 0.0000 |
| GBP_JPY | 17 | 52.94% | 0.5677 | 2.3083 |
| GBP_NZD | 1 | 0.00% | -2.2848 | 0.0000 |
| NZD_JPY | 15 | 26.67% | -0.1046 | 0.8529 |

## Caveats

- Gate FAILED — the replayed models are the Fold-3 placeholders, not a promoted artifact. Backtest numbers are indicative only.
- Only 40 pooled OOS trades. At this count the win rate (and therefore edge-over-random) is dominated by sampling noise — a positive number here is not evidence of skill.

## Not run by v1

The artifact-level holdout gate is not reproduced here: it engineers the
holdout slice with the PRODUCTION feature list, so a candidate feature set
cannot be scored by it without generalizing `_score_artifact_holdout`.
The fold gate's `edge_over_random` is the lab's verdict; fold 3's
validation window is the recent-regime check.

## Interpretation

**This closes the 2026-07-07 experiment that a window shift had confounded —
and the cost table does not help.** Turning the measured per-instrument alphas
on thins the basket hard (chop veto 23.68% → 41.21%; frame 226,909 → 174,730
rows) and the gate's evidence gets *worse*, not better: pooled PF lower bound
0.7727 → 0.5824 on 40 trades, with fold-3 at 0.5965. No promotion, correctly.

**The backtest's +0.4251R looks better than the control's +0.2548R but is not
comparable.** It is a different row population under a different toll model (72
trades with measured alphas vs 74 with the flat default), and the replayed
models are Fold-3 placeholders either way. The honest reading is the gate's:
pricing measured costs removes trades without adding evidence of edge. The
long-standing "configured but unused" asymmetry is now measured and closed.

## Command

```bash
PYTHONPATH=src:. python -m lab.cli run --name spread_table_control
```
