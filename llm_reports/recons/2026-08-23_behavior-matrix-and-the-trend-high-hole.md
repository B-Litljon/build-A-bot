---
type: recon
date: 2026-08-23
time: 13:50 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "Brandon chose the retrainer path for Phase 2; built the candidate scorer and matrix and ran the first real comparisons."
head: b76f375eafe9d69e920c617fdcd559a41958b293
scope: modifies-source
related:
  - recons/2026-08-22_behavior-tagger-and-the-low-vol-band.md
  - recons/2026-08-22_market-behavior-classifier-and-algorithm-recommender.md
  - recons/2026-08-17_5yr-candidate-and-the-untradeable-basket.md
files_touched:
  - src/core/retrainer.py
  - src/analysis/behavior_matrix.py
  - tests/test_behavior_matrix.py
  - tests/test_oos_ledger_capture.py
  - src/analysis/README.md
---

# The behavior matrix, and the hole in trend_high

## Context

Phases 2 and 3 of the behavior tool, built on the retrainer's walk-forward
rather than the Alpaca-era `replay_test.py` / `evaluate_performance.py`
(Brandon's call, 2026-08-23). Phase 1 (the tagger) is the prior report.

**This report supersedes the preliminary matrix in the 2026-08-22 report.**
That preview used the July threshold-sweep parquet — a different trade
population, scored gross, at thresholds below the live 0.40 bar. Its cell
ranking does not survive proper walk-forward measurement and should not be
used. Details under Finding 4.

## Investigation

**A candidate is a configuration, not an artifact.** Scoring saved model
directories was considered and rejected: their `trained_at` values span
2026-07-02 to 2026-08-11, so a common honest OOS window is ~11 days — far too
thin — and scoring each on its own post-cutoff window reproduces exactly the
different-periods error the 2026-08-17 five-year review discarded. The M30/M45
siblings are also a different bar size and cannot share a matrix with M15.

**Capture mechanism.** `validate_candidate` gained an opt-in `oos_ledger`
parameter that appends each fold's Devil-approved OOS trades. It reads masks the
fold already computed and mutates nothing; `oos_ledger is None` in every
production path.

**A measurement bug I introduced and then caught.** My first three runs passed
`days_back` to the fetch while leaving `DAYS_BACK` at its default 60. The fold
schedule is derived from the module constant:

```python
fold_configs = [
    (DAYS_BACK // 2,     DAYS_BACK * 2 // 3),
    ...
]
```

so a 365-day fetch trained on days 0–30 of year-old data and silently discarded
~97% of the frame. Those three runs produced gate PFs of 0.49–0.68 and are
**void**. The correct knob is `RETRAIN_DAYS_BACK`, which drives fetch and folds
together. `validate_candidate` now warns when the frame's span exceeds
`DAYS_BACK` by more than 20%; the schedule itself was left alone because
changing it would move the promotion gate's goalposts.

Sanity check after the fix: the 730-day, 8-instrument config scores gate
PF **1.245** and passes — the right neighbourhood for the recorded 1.373 on the
shipped model, with two weeks of newer data and a shifted end date.

## Findings / Changes

**Finding 1 (high) — `trend_high` is a reliable money-loser, and it is the
biggest cell.** The only cell whose bootstrap CI excludes zero in *every* run,
always negative:

| run | n | win% | net PF | 95% CI on net EV |
|---|---:|---:|---:|---|
| all 8 instruments | 135 | 25.9 | 0.44 | [-0.77, -0.33] |
| all 8, crosses only | 51 | 23.5 | 0.39 | significant |
| crosses-only training | 78 | 19.2 | 0.30 | [-0.98, -0.48] |

Break-even at 2R needs 33.3% wins; `trend_high` delivers 19–26%. It is also one
of the largest cells by bar count (~16% of warm bars). The mechanism is
plausible and worth confirming: in violent trends a 2×ATR stop is inside the
noise, so price takes out the stop before reaching the 4×ATR target.

This is a **hypothesis for the gate to test**, not a config edit. Candidate
experiment: veto Angel signals tagged `trend_high`, retrain, and see whether the
gate improves — cheap, since a full walk-forward run is ~60s.

**Finding 2 (high) — pooled gate PF says metals hurt; the tradeable-only
comparison says the opposite.** Scoring both configs on the six broker-tradeable
crosses:

| | trades | win% | gross PF | PF @0.10R toll | PF @0.33R toll |
|---|---:|---:|---:|---:|---:|
| trained **with** metals | 247 | 46.6 | 1.742 | 1.505 | 1.094 |
| trained **crosses only** | 334 | 42.2 | 1.461 | 1.262 | 0.917 |

Yet the *pooled* gate scores read 1.245 (with metals) vs 1.500 (crosses only) —
the reverse ordering, because the pooled figure is diluted by the metals' own
untradeable trades.

This sharpens the 2026-08-17 recommendation rather than contradicting it: score
the gate on broker-tradeable instruments only, but **do not shrink the training
basket** — consistent with the basket-shrink attempt already rejected on
2026-07-02. Single window; per the same 2026-08-17 work, the metals' sign flips
by window, so this is direction, not law.

**Finding 3 (medium) — nothing is proven profitable at cell level.** Apart from
`trend_high` (negative), every cell's CI straddles zero, and many are below the
30-trade floor. `recommend()` correctly returns "no candidate clears the bar"
for four of nine behaviors. That is the tool working: with ~250–350 OOS trades
spread over nine cells, per-cell resolution is genuinely low.

**Finding 4 (medium) — the 2026-08-22 preliminary matrix is superseded and was
misleading.** It ranked `trend_high` best (gross PF 1.40) and `range_low` worst
(0.63). The walk-forward reverses both: `trend_high` is the worst cell in all
three runs, and `range_low` is mildly positive (net PF 1.32 crosses-only). The
preview drew on the July sweep's population — lower thresholds, a different
simulation, one window, gross. Lesson: tag-conditioned statistics are only as
good as the trade population underneath them.

**Finding 5 (low) — the toll never changes the ranking.** It is a constant
subtracted from every trade, so it moves only the zero line. Break-even for the
8-instrument config lands at a toll of ~0.10R (gross PF 1.182 → 1.021), matching
the shipped model's recorded gross 1.373 → net 1.004.

**Changes.** `src/analysis/behavior_matrix.py` (Candidate, CellStats,
`tag_frame`, `net_r`, `score_ledger`, `to_frame`, `recommend`); ledger capture
plus span warning in `src/core/retrainer.py`; READMEs and GLOSSARY terms.

## Verification

`PYTHONPATH=src:. python -m pytest -q` → **267 passed** (200 at session start;
+10 log filters, +25 tagger, +23 matrix, +9 ledger capture).
`python -m compileall -q src/` clean.

The ledger capture is pinned on the property that actually matters:
`approved_mask` indexes the Angel-proposed subset, not `val_df`, so
`test_approved_mask_is_relative_to_proposed_not_val_df` builds a frame where
mis-composition yields visibly wrong symbols rather than a wrong count. Also
pinned: inputs unmutated, zero approvals append nothing, missing carry columns
skipped, and a source-level check that the call sits under
`if oos_ledger is not None:` with the parameter defaulting to None.

Matrix tests pin the arithmetic by hand (EV, PF), that the toll is charged on
winners too, that a gross-positive cell can go net-negative, that cold trades
are dropped rather than pooled, that thin cells are flagged, and that
`recommend()` refuses on thin or losing evidence.

Two test-fixture bugs of my own were found and fixed during the run (a
"coinflip" cell that was in fact significantly negative; tagger windows built
with the current bar at the wrong end). The modules were correct both times.

End-to-end: four walk-forward runs against live OANDA history, ~60s each. No
`save_models()` call exists in this path, so no model artifact can be written.

## Follow-up experiment (same day): does vetoing trend_high help?

Three arms — control, `trend_high` vetoed, and `range_low` vetoed as a placebo
of similar size (16.1% vs 15.1% of rows) — run across four pinned end dates at
730-day lookback. The veto mirrors the existing chop veto: label first, then
drop rows as trade entries, with `chop_veto_rate` rescaled to the combined drop
so the gate's dynamic trade floor scales fairly.

Net profit factor at a 0.10R toll (the measured break-even toll):

| window | control | trend_high vetoed | placebo (range_low) |
|---|---:|---:|---:|
| 2026-08-22 | 1.043 | **1.143** (+0.065 R) | 0.944 (-0.070 R) |
| 2026-05-22 | 1.115 | **1.312** (+0.119 R) | 1.103 (-0.008 R) |
| 2026-02-22 | 0.923 | **1.133** (+0.142 R) | 1.140 (+0.148 R) |
| 2025-11-22 | 0.997 | **1.012** (+0.012 R) | 0.732 (-0.201 R) |
| beats control | — | **4/4** | 1/4 |

Win rate rose in 4/4 (+0.3 to +4.8 points); gate PF rose in 4/4; the vetoed
config passed the gate in 4/4 where control passed 3/4 and the placebo 2/4.

**Verdict: promising, NOT proven.** No individual window's bootstrap interval
excludes zero — not one. And the windows use a 730-day lookback stepped three
months apart, so they share ~87% of their data: 4/4 is not four independent
trials. What can be said is that the veto's deltas are same-signed and modest
while the placebo's scatter (+0.148 to -0.201, 1/4) is what noise looks like.

**Method note.** An unpinned first pass — each arm re-fetching to "now", minutes
apart — showed the veto at +5.4 points of win rate and a jump from losing
(0.980) to profitable (1.232). Pinning the window cut that to +2.2 points and
inside the noise. Same code, same hypothesis; the difference was entirely
measurement discipline. Two supposedly identical runs an hour apart differed by
498 vs 453 trades and PF 1.245 vs 1.364. **Pin `RETRAIN_END_DATE` and run a
placebo arm; it costs ~2 minutes and this project has now been bitten three
times by mismatched windows** (the 5-year candidate, the pooled metals score,
and this).

## Implemented (2026-08-23, later): tradeable-only gate scoring

`UNTRADEABLE_SYMBOLS` (XAU_USD, XAG_USD; env `RETRAIN_UNTRADEABLE_SYMBOLS`)
now narrows the promotion gate's metrics to instruments the account can trade.
**Training keeps the full basket** — Brandon's call, backed by the crosses
measuring 1.742 vs 1.461 with metals in training. Live already refused metals
at boot since 2026-08-06 (`_drop_untradeable_symbols`); this closes the offline
half. Logic sits in `_tradeable_scoring_mask` so it is unit-testable.

Effect across four pinned 730-day windows:

| window | old PF | old | new PF | new | scored trades |
|---|---:|:--:|---:|:--:|---|
| 2026-08-22 | 1.291 | pass | **2.118** | pass | 547 → 276 |
| 2026-05-22 | 1.256 | pass | **1.951** | FAIL (sample) | 390 → 221 |
| 2026-02-22 | 1.292 | pass | **2.000** | FAIL (sample) | 267 → 161 |
| 2025-11-22 | 1.200 | fail | 1.037 | fail | 194 → 135 |

On tradeable instruments the model measures far better than the pooled number
claimed (3 of 4 windows); the fourth flips, matching 2026-08-17's finding that
the metals' sign is window-dependent.

**Side effect — the sample floor is now the binding constraint.** Metals were
~36-45% of approvals, so scoring halves the evidence while the floor
(`300 × (1 − veto rate)` ≈ 232) was calibrated for the pooled era. Two windows
now fail on sample size while showing PF ~2.0. The floor was NOT touched:
moving a promotion goalpost to flatter one's own change is backwards, and it is
Brandon's call. **A 5-year window resolves it without moving anything** — 302
scored trades vs the 232 floor, gate passed. Recommendation: lengthen the
lookback rather than lower the floor.

## Verdict on the trend_high veto: DO NOT SHIP

Implemented as opt-in `RETRAIN_BEHAVIOR_VETO` (empty by default), mirroring the
chop veto, with the labels recorded in `metadata.json` as `behavior_veto` so an
artifact declares the live gate it requires. **No live gate was built** — the
soak runs from this tree, and execution-path changes buy nothing until a model
is actually promoted.

Final evidence, five windows:

| window | control | vetoed | delta |
|---|---:|---:|---|
| 730d ×4 | — | — | beat control 4/4, +0.012 to +0.142 R |
| 5yr (tradeable only) | 1.378 | 1.481 | +0.054 R, CI [-0.187, +0.297] |

The 5-year window is the largest sample obtainable (302 vs 273 tradeable
trades) and **still cannot distinguish the effect from zero** (P(not better) =
0.33). Direction has been consistently positive across every window tried, and
win rate rose in all five — but the magnitude is small enough that no available
sample resolves it. More windows will not fix this; the data ceiling is the
binding constraint, not the experiment design.

**Recommendation: keep the code inert, do not train a production candidate on
it.** Revisit if live trade volume grows materially. The `trend_high` cell being
a *loser* remains solid (significant in every behavior-matrix run); what is
unproven is that removing it *helps enough to matter*.

## Risk & follow-ups

1. **Nothing here is promotable.** Single window, one timeframe, in-sample
   relative to the config search. The `trend_high` veto is the next experiment,
   not the next commit.
2. **`RETRAIN_DAYS_BACK` is a footgun** for any caller of `validate_candidate`.
   Now warned about; consider deriving the fold schedule from the frame's actual
   span, accepting that it perturbs gate outputs at the edges.
3. **Uncommitted, on `feat/wider-brackets-and-rename`.** `retrainer.py` already
   carried unrelated uncommitted work (`_resolved_model_dir`,
   `_is_production_model_dir`, `promote_or_reject` edits) before this session —
   not mine, and not reviewed here.
4. **Soak untouched.** It relaunches from this tree; the only live-path file
   changed this session is `run_oanda.py` (the log filter), verified booting.
