---
type: refactor
date: 2026-08-03
time: 22:20 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "Fix the retrain gate so it tests lift over equal-weighting, not just lift over random"
head: 6523254
scope: modifies-source
files_touched:
  - scripts/investor_train_model.py
  - scripts/README.md
  - tests/test_investor_benchmark_gate.py
  - tests/README.md
  - GLOSSARY.md
related:
  - 2026-08-02_investor-ebitda-gap-and-rebalance-policy.md
  - 2026-08-03_investor-target-horizon-study.md
---

# Investor benchmark gate — and the measurement noise it exposed

## Context

The investor's promotion gate thresholds on Precision@K lift over a *random*
picker. That cannot distinguish "better than guessing" from "better than doing
nothing", and on 2026-07-03 a model passed every threshold while failing to
beat an equal-weighted basket of the same 96 names out of sample (2026-08-02
recon, finding 3 — the top follow-up).

This adds the missing question: does the deployed basket beat equal-weighting
the universe?

## Investigation

Building it surfaced something that changed its design.

The first version measured one number — mean monthly excess over equal-weight,
across all walk-forward folds — with a floor of 0.0 bps. Run against the
current configuration it reported **+47.3 bps and PASSED**, contradicting both
the 2026-08-02 recon (no edge) and the 2026-08-03 horizon study (−6.3 bps for
this same 60-day target).

Three of my own measurements of one quantity disagreed, so I isolated the
cause by varying one input at a time (`diagnose_disagreement.py`):

| Input frame | Features | Lab harness | Gate harness |
|---|---|---|---|
| `v4_training_features` | 14 | +35.5 bps | +38.2 bps |
| `v4_training_features` | 13 | +38.5 bps | +41.8 bps |
| `v4_inference_features` | 13 | +17.0 bps | +15.2 bps |

**The two harnesses agree within ~3 bps**, so the gate's simulation is sound.
The disagreement was the *label*.

The horizon study built its own top-quintile label from close prices; the
pipeline's shipped label comes from the parquet. The underlying forward
returns are **bitwise identical** (max difference 0.0, correlation 1.0). The
labels agree on 98.96% of rows — they differ on 1,195 of 114,720, which is
exactly **one row per trading day**. With 96 symbols a top quintile is 19.2
names: `pd.qcut` keeps 19, a `quantile(0.80)` cut with `>=` keeps 20.

**One extra stock per day in the training label moves the measured monthly
excess by 23 bps** (−6.3 → +17.0). Counting the 2026-08-02 recon's own
implementation, four reasonable versions of this same measurement span roughly
50 bps — as large as any effect any of them claimed.

## Findings / Changes

### 1. The gate exists and fails closed

`scripts/investor_train_model.py` now simulates the deployed basket
(`TOP_K=8`, `SECTOR_CAP=2`, mirroring `portfolio_orchestrator`) at each
month-end of every out-of-sample fold, holds to the next month-end, and
compares against equal-weighting every symbol with prices at both ends.
Prices come from `data/raw/v4_investor_data.parquet` because the training
frame deliberately carries none. If they cannot be loaded, or no held month
is produced, the gate **fails** — an unmeasurable model must not be promoted.

The holding period may close *outside* the test window. That is correct, not
leakage: the pick used only in-window information, and the outcome is what
live trading would realise. A test pins this behaviour.

### 2. The floor is not zero, and one measurement is not enough

Given the ~50 bps implementation spread, a 0.0 bps floor gates on noise —
which is precisely why the first version waved through the model this gate
was built to catch. Two changes:

- `GATE_BENCH_MIN_EXCESS_BPS` defaults to **25.0**, above the noise band.
- `GATE_BENCH_ALIGNMENTS` (default `0,7,14,21,28`) re-runs the entire
  walk-forward with the start slid by N trading days, and
  `GATE_BENCH_MIN_PASS_SHARE` (default **1.0**) requires *every* alignment to
  clear the floor.

The multi-alignment requirement is the part doing the real work: it asks
whether a result survives re-slicing, which a single number cannot.

### 3. Verified against the model it was built to catch

Current configuration, new gate:

```
offset   0 trading days:    +47.3 bps  (28 months)
offset   7 trading days:    +26.7 bps  (29 months)
offset  14 trading days:    +29.2 bps  (29 months)
offset  21 trading days:    +34.4 bps  (29 months)
offset  28 trading days:     -7.4 bps  (29 months)
Cleared +25 bps at : 80% of alignments vs required 100% -> FAIL
🚫 GATE FAILED — existing model retained, nothing written
```

Exit code 2, `models/v4_investor_lgbm.txt` unchanged (md5 verified before and
after). Note the shipped model looks *mildly positive* on this frame — the
gate's verdict is "not stably better", not "worse", which is the honest
reading of the evidence.

### 4. Consequence for the horizon study

The 2026-08-03 study's effect sizes (+80 to +110 bps) were measured with a
single label construction across all five alignments, so its stability check
could not see this source of variation. The horizon *ordering* was reproduced
under several conditions and still looks real; the *magnitudes* should be
treated as having a ~50 bps implementation band around them. That band is
wide enough to swallow the 21-day result entirely.

## Verification

- 174 tests pass (`PYTHONPATH=src:. python -m pytest -q`); `compileall` clean.
- 12 new tests in `tests/test_investor_benchmark_gate.py`: sector cap
  behaviour (including cap exhaustion and unknown-sector symbols), a perfect
  picker beating equal-weight, a flat universe producing exactly zero excess,
  exclusion of symbols missing prices, the outside-window holding period, and
  fail-closed on unreadable prices.
- One test asserts the gate's `TOP_K`/`SECTOR_CAP` still equal the
  orchestrator's, so a change to the traded basket cannot leave the gate
  silently measuring a basket nobody trades.
- Live artifacts untouched: `models/v4_investor_lgbm.txt` md5 identical
  before and after (LightGBM retraining on identical data is deterministic),
  and the metadata sidecar was restored from backup after an exploratory run
  rewrote its `trained_at`.

**Not verified / limits:** the 25 bps floor is calibrated against a spread of
four implementations, not a formal null distribution — it is a defensible
guess, not a derived threshold. Transaction costs are still unmodelled.
Everything inherits the 2.4-year, single-regime, survivorship-clean window.

## Risk & follow-ups

1. **The gate will now block retrains that previously passed.** That is the
   point, but it means a standing answer is needed for "nothing passes" —
   most likely falling back to equal-weighting, which the evidence currently
   supports as well as anything the model produces.
2. **The label rounding (19 vs 20 names) is worth settling deliberately.**
   Neither choice is wrong; the finding is that the result is sensitive to it,
   which argues for a label less brittle than a hard quintile cut.
3. **Do not deploy a short-horizon retrain yet.** The candidate should be run
   through this gate first — that was the point of building it.
4. The alignment loop retrains 5x, adding ~25 s to a run that happens rarely.
   If the universe or history grows materially, consider parallelising.

## Files touched

- `scripts/investor_train_model.py` — `_load_close_matrix`, `_month_ends`,
  `_sector_capped_pick`, `_fold_basket_months`, `_benchmark_at_alignment`;
  benchmark gate wired into the pass/fail decision, the summary log, and the
  metadata sidecar; module Glossary extended.
- `tests/test_investor_benchmark_gate.py` — new, 12 tests.
- `scripts/README.md`, `tests/README.md`, `GLOSSARY.md` — documentation for
  the gate and for the lift-over-random vs lift-over-benchmark distinction.
