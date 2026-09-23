---
type: recon
date: 2026-09-21
time: 20:01 PDT
agent: opencode
model: deepseek-flash
trigger: "Served-artifact replay models/forex_m15_wide on spec v3_base_control (hash d952dd3deff5b755)"
head: 14fe328
scope: lab replay only — no production model, config, or live path touched
related:
  - handoffs/2026-09-21_feature-lab-plan.md
---

# Feature lab — served-artifact replay (`v3_base_control`)

## Served artifact

- model dir: `models/forex_m15_wide` | trained: 2026-08-30T02:20:16.562224+00:00 | bars: Angel 0.3833 / Devil 0.4400 (from threshold.json)
- schema: 17 Angel features (`feature_names_in_`); Devil adds `angel_prob`
- trained on: XAU_USD, XAG_USD, GBP_JPY, AUD_JPY, EUR_JPY, NZD_JPY, GBP_AUD, GBP_NZD

## Frame

- rows: 226,909 | features: 17 | chop/behavior veto drop: 23.68% | unresolvable tail purged: 83
- loaded from frame cache

## Raw population (pre-live-gates)

- rows 226,909 | base rate 0.2569
- Angel proposals 63 | Devil approvals 56 | approval win rate 0.4464 | edge over random 0.1895 | bracket PF 1.6129

## Window split (artifact's recorded holdout)

| window | rows | base | proposed | approved | approval WR | edge | trades | trade WR | net EV (R) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before 2026-04-19 | 183,167 | 0.2600 | 48 | 44 | 0.4773 | 0.2173 | 33 | 0.4545 | 0.2785 |
| 2026-04-19 to 2026-08-28 (recorded holdout) | 41,051 | 0.2500 | 15 | 12 | 0.3333 | 0.0833 | 9 | 0.4444 | 0.1570 |
| after 2026-08-28 | 2,691 | 0.1550 | 0 | 0 | nan | nan | 0 | nan | nan |

## Replay (live-gated)

- toll pricing: flat | trades 42 | wins 19 | win rate 45.24%
- gross EV 0.5825R | net EV 0.2525R | net PF 1.4356 | max drawdown 6.6500R
- gate veto funnel: regime=6

| symbol | trades | win rate | net EV (R) | net PF |
|---|---:|---:|---:|---:|
| AUD_JPY | 16 | 50.00% | 0.3278 | 1.5633 |
| EUR_JPY | 4 | 25.00% | -0.1356 | 0.7961 |
| GBP_AUD | 5 | 60.00% | 0.4691 | 1.8818 |
| GBP_JPY | 2 | 0.00% | -1.3300 | 0.0000 |
| GBP_NZD | 5 | 40.00% | 0.0551 | 1.0898 |
| NZD_JPY | 10 | 50.00% | 0.5941 | 2.4890 |

## Caveats

- Flat toll — the served artifact's schema has no `cost_ratio`, so trades are priced at the live default alpha (what the soak currently runs). The measured per-instrument alphas are a separate arm, not applied here.
- Only 42 replayed trades; win rate and net EV at this count are dominated by sampling noise.
- Replay, not a promotion gate: ONE artifact, ONE cached frame, no PASS/FAIL. The frame is largely in-sample for the artifact; the recorded-holdout rows in the window table are the only slice it never saw.

## Interpretation

**The served model is a lottery-ticket sampler, and on data it never saw it
proposes almost nothing.** At its own pinned bars (Angel 0.3833, Devil 0.44)
the artifact proposes on 63 of 226,909 bars (0.028%) and the Devil then approves
56 of them — the Devil bar is inert at 0.44, so the Angel is the only real
selection. The bar sits at the 99.97th percentile of the model's own score
distribution on this frame (q99.99 = 0.416, max 0.526). That is the mechanism
behind the soak's 0 fills: the live bot is not failing to execute, it is being
told "no" almost every bar. The last 2,691 rows of the frame (2026-08-29 →
09-07, after the artifact's training data ends) produce **zero proposals**.

**Where it does fire, almost all of the evidence is in-sample.** 44 of the 56
raw approvals are before 2026-04-19 — rows the artifact trained on. The gated
replay pools to 42 trades / +0.2525R net / PF 1.44, but 33 of those trades are
in the same in-sample region (+0.2785R), and only 9 fall in the recorded holdout
(+0.1570R on nine trades — no evidence either way).

**The recorded promotion holdout is not reproduced.** `metadata.json` records
the artifact's holdout as 36 trades, 69.4% win rate, PF 4.55, EV 1.583 — an EV
above the bracket's +2R maximum, i.e. the pre-fix survival/macro mix-up the
2026-09-14 provenance recon identified. Scoring the served pkls themselves on
the same window at the same bars gives 15 proposals / 12 tradeable approvals /
33.3% macro WR / PF 1.00 — break-even on twelve rows, nowhere near 69.4%. Part
of the gap is expected (the recorded slice was engineered separately on the full
8-symbol basket, and its metrics predate the 2026-09-09 scoring fix), but the
gap is large enough that the recorded holdout should not be treated as evidence
in either direction. **Reproducing the retrainer's own slice exactly** —
8-symbol basket, separately engineered, the same call `_score_artifact_holdout`
makes — is the follow-up that would settle whether the difference is slice
engineering or something worse. The lab has the frame machinery for it; it needs
the metals bars cached for the full window.

**What this baseline is for.** Candidate feature runs are now compared against:
(a) the raw approval population — currently 63 proposals / 56 approvals over 730
days; (b) the recorded-holdout slice — currently 12 approvals, 33.3% WR, PF 1.0.
An improvement that vanishes on (b) has not improved anything. The pooled net EV
(+0.2525R) is deliberately **not** a bar to beat: it is in-sample-dominated and
rests on 42 trades.

One thing this replay does not touch: the Devil is decorative at its pinned bar
(56/63 pass), so the 2026-09-14 addendum's "train the Devil on the macro label"
proposal remains open and unchanged by these numbers.

## Command

```bash
PYTHONPATH=src:. python -m lab.cli replay --name v3_base_control
```
