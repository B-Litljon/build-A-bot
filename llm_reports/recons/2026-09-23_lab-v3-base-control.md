---
type: recon
date: 2026-09-23
time: 00:04 PDT
agent: opencode
model: deepseek-flash
trigger: "Feature-lab run v3_base_control (spec hash aa4acb5d6958e3e4)"
head: a1a60a1
scope: lab run only — no production model, config, or live path touched
related:
  - handoffs/2026-09-21_feature-lab-plan.md
---

# Feature lab — v3_base_control

## Spec

- content hash: `aa4acb5d6958e3e4` (frame cache key)
- feature families: `v3_base`
- symbols: AUD_JPY, EUR_JPY, GBP_JPY, NZD_JPY, GBP_AUD, GBP_NZD
- window: 730 days @ M15, htf 1h
- geometry: 2.0x/4.0x/45 bars
- labels: kind=survival, survival_bars=5
- spread table: off
- estimator: lightgbm

## Frame

- rows: 226,909 | features: 17 | chop/behavior veto drop: 23.68% | unresolvable tail purged: 83
- loaded from frame cache

## Gate (retrainer `validate_candidate`)

- verdict: **FAIL**
- mean Brier 0.0857 | mean EV 0.365497 | pooled trades 30 | pooled wins 13
- pooled PF lower bound 0.7727 | fold-3 PF lower bound 0.5965
- pooled base rate 0.2540 | **edge over random 0.1793**
- production thresholds: Angel 0.3564, Devil 0.6600
- gate wall time: 14.5s

| fold | train | val | Brier | EV | proposed | approved | win rate | base rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 113,068 | 37,974 | 0.0456 | 0.500000 | 2 | 2 | 100.00% | 0.2562 |
| 2 | 151,042 | 38,549 | 0.0330 | 0.333333 | 14 | 9 | 100.00% | 0.2561 |
| 3 | 189,591 | 37,226 | 0.1785 | 0.263158 | 24 | 19 | 78.95% | 0.2495 |

Rejection reasons:

- Fold 3 PF point estimate 1.4545 on 19 trades, but 95% lower bound 0.5965 < 1.2 — the most recent regime cannot prove it beats break-even
- Pooled fold PF 95% lower bound 0.7727 < 1.2 (evidence: 13 wins / 30 trades across 3 folds)

## Backtest (live-gated replay)

- toll pricing: flat | trades 74 | wins 33 | win rate 44.59%
- gross EV 0.5848R | net EV 0.2548R | net PF 1.4558 | max drawdown 7.8764R
- gate veto funnel: regime=12

| symbol | trades | win rate | net EV (R) | net PF |
|---|---:|---:|---:|---:|
| AUD_JPY | 20 | 60.00% | 0.6325 | 2.5853 |
| EUR_JPY | 13 | 46.15% | 0.5408 | 2.3216 |
| GBP_AUD | 10 | 50.00% | 0.2879 | 1.5265 |
| GBP_JPY | 10 | 30.00% | 0.1393 | 1.2618 |
| GBP_NZD | 5 | 60.00% | 0.4709 | 1.8851 |
| NZD_JPY | 16 | 25.00% | -0.4656 | 0.4908 |

## Caveats

- Gate FAILED — the replayed models are the Fold-3 placeholders, not a promoted artifact. Backtest numbers are indicative only.
- Only 30 pooled OOS trades. At this count the win rate (and therefore edge-over-random) is dominated by sampling noise — a positive number here is not evidence of skill.
- Cost table OFF — the backtest priced trades at the flat default toll, not per-instrument measured alphas. Cross-instrument comparisons in the backtest table are especially weak.

## Not run by v1

The artifact-level holdout gate is not reproduced here: it engineers the
holdout slice with the PRODUCTION feature list, so a candidate feature set
cannot be scored by it without generalizing `_score_artifact_holdout`.
The fold gate's `edge_over_random` is the lab's verdict; fold 3's
validation window is the recent-regime check.

## Interpretation

_To be written after reading the numbers above._

## Command

```bash
PYTHONPATH=src:. python -m lab.cli run --name v3_base_control
```
