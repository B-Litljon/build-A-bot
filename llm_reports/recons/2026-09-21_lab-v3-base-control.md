---
type: recon
date: 2026-09-21
time: 16:38 PDT
agent: opencode
model: deepseek-flash
trigger: "Feature-lab run v3_base_control (spec hash d952dd3deff5b755)"
head: 14fe328
scope: lab run only — no production model, config, or live path touched
related:
  - handoffs/2026-09-21_feature-lab-plan.md
---

# Feature lab — v3_base_control

## Spec

- content hash: `d952dd3deff5b755` (frame cache key)
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
- gate wall time: 8.4s

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

**The lab is calibrated: its frame is production's frame, and the gate's FAIL is
the known, correct answer.** 226,909 engineered rows + 83 tail-purged = the
known 226,992, and the pooled base rate 0.25399 against the known 0.2540. The
fold evidence is 13 wins / 30 OOS trades, pooled PF lower bound 0.7727 against
the 1.2 bar — no promotion, correctly.

**The headline `edge over random +0.1793` is the withdrawn small-sample artifact,
not signal.** It is a win-rate difference on 30 rows at the EV-maximising Angel
bar; that exact measurement was retracted on 2026-09-14. Read it with
`pooled_oos_trades` beside it, as the Caveats say.

**The backtest is indicative only.** The Fold-3 placeholder models replay to
+0.2548R net on 74 trades with the flat toll and live Gates A/B/C, but these are
not the served artifact and the population is thin. The served artifact's own
replay is recorded separately
(`recons/2026-09-21_lab-served-artifact-baseline.md`): it proposes 63 times in
730 days and nothing after its training data ends.

So the control does its job — it proves the lab can be trusted with a candidate
— and the edge question is unchanged.

## Command

```bash
PYTHONPATH=src:. python -m lab.cli run --name v3_base_control
```
