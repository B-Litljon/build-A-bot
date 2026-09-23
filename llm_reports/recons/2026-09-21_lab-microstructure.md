---
type: recon
date: 2026-09-21
time: 16:38 PDT
agent: opencode
model: deepseek-flash
trigger: "Feature-lab run microstructure (spec hash beedea038b2b1020)"
head: 14fe328
scope: lab run only — no production model, config, or live path touched
related:
  - handoffs/2026-09-21_feature-lab-plan.md
---

# Feature lab — microstructure

## Spec

- content hash: `beedea038b2b1020` (frame cache key)
- feature families: `v3_base, microstructure`
- symbols: AUD_JPY, EUR_JPY, GBP_JPY, NZD_JPY, GBP_AUD, GBP_NZD
- window: 730 days @ M15, htf 1h
- geometry: 2.0x/4.0x/45 bars
- labels: kind=survival, survival_bars=5
- spread table: off
- estimator: lightgbm

## Frame

- rows: 226,909 | features: 21 | chop/behavior veto drop: 23.68% | unresolvable tail purged: 83
- loaded from frame cache

## Gate (retrainer `validate_candidate`)

- verdict: **FAIL**
- mean Brier 0.1580 | mean EV -0.077984 | pooled trades 101 | pooled wins 32
- pooled PF lower bound 0.6337 | fold-3 PF lower bound 0.3401
- pooled base rate 0.2540 | **edge over random 0.0628**
- production thresholds: Angel 0.3436, Devil 0.4800
- gate wall time: 11.2s

| fold | train | val | Brier | EV | proposed | approved | win rate | base rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 113,068 | 37,974 | 0.0592 | 0.038462 | 62 | 52 | 94.23% | 0.2562 |
| 2 | 151,042 | 38,549 | 0.2240 | -0.100000 | 21 | 20 | 75.00% | 0.2561 |
| 3 | 189,591 | 37,226 | 0.1907 | -0.172414 | 30 | 29 | 72.41% | 0.2495 |

Rejection reasons:

- EV -0.077984 < 0.0005 threshold
- Fold 3 PF point estimate 0.7619 on 29 trades, but 95% lower bound 0.3401 < 1.2 — the most recent regime cannot prove it beats break-even
- Pooled fold PF 95% lower bound 0.6337 < 1.2 (evidence: 32 wins / 101 trades across 3 folds)

## Backtest (live-gated replay)

- toll pricing: flat | trades 149 | wins 65 | win rate 43.62%
- gross EV 0.4952R | net EV 0.1652R | net PF 1.2689 | max drawdown 12.6565R
- gate veto funnel: regime=24

| symbol | trades | win rate | net EV (R) | net PF |
|---|---:|---:|---:|---:|
| AUD_JPY | 38 | 44.74% | 0.1654 | 1.2626 |
| EUR_JPY | 22 | 45.45% | 0.2925 | 1.5377 |
| GBP_AUD | 16 | 56.25% | 0.4315 | 1.8495 |
| GBP_JPY | 19 | 52.63% | 0.5587 | 2.3302 |
| GBP_NZD | 10 | 40.00% | 0.1573 | 1.2748 |
| NZD_JPY | 44 | 34.09% | -0.1635 | 0.7873 |

## Caveats

- Gate FAILED — the replayed models are the Fold-3 placeholders, not a promoted artifact. Backtest numbers are indicative only.
- Cost table OFF — the backtest priced trades at the flat default toll, not per-instrument measured alphas. Cross-instrument comparisons in the backtest table are especially weak.

## Not run by v1

The artifact-level holdout gate is not reproduced here: it engineers the
holdout slice with the PRODUCTION feature list, so a candidate feature set
cannot be scored by it without generalizing `_score_artifact_holdout`.
The fold gate's `edge_over_random` is the lab's verdict; fold 3's
validation window is the recent-regime check.

## Interpretation

**The candidate family raises the trade population and lowers the edge, in the
direction of the base rate.** 101 pooled OOS approvals against the control's 30,
but the edge over random falls +0.1793 → +0.0628 (both thin, so neither number
is evidence) and the mean EV turns negative (-0.0780 against the +0.0005 bar) —
the gate's FAIL here is on EV, not only on the PF bounds. The backtest agrees:
149 trades, +0.1652R net, positive but below the control's +0.2548R and with a
larger drawdown (12.66R vs 7.88R).

**As a template it works; as evidence of signal it does not.** The run proves a
new family can be added, registered, and scored end-to-end without touching the
pipeline or the live path. But the 2026-09-14 edge budget says a feature set has
to find roughly 3pp of win rate to matter, and four short-horizon
bar-shape/autocorrelation features did not: this is close to "more trades, same
coin". Any follow-up family should be judged on the recorded-holdout slice the
served-artifact baseline now defines
(`recons/2026-09-21_lab-served-artifact-baseline.md`), not on pooled net EV
alone.

## Command

```bash
PYTHONPATH=src:. python -m lab.cli run --name microstructure
```
