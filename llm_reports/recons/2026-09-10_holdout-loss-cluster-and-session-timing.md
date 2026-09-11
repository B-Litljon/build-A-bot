---
type: recon
date: 2026-09-10
time: 16:32 PDT
agent: Claude Sonnet 5
model: claude-sonnet-5
trigger: "Brandon asked why the retrain's artifact-holdout evaluation scored so badly (WR 14.6%, PF 0.34) after the 2026-09-09 safety/training-integrity pass; a second retrain (fresh data, debug instrumentation) was run to dig into the per-trade detail before he decides whether to build a fix."
head: 535d95054c5097e414cbe5ff4d1f64de1b57258b
scope: read-only
related:
  - refactors/2026-09-09_safety-and-training-integrity-pass.md
  - audits/2026-09-08_high-benefit-fixes-ranked.md
files_touched: []
---

## Context

Following the 2026-09-09 safety-and-training-integrity pass (committed as
`535d950` on `fix/safety-and-training-integrity-pass`), a retrain was launched
into a side directory (`RETRAIN_MODEL_DIR=models/forex_m15_candidate_20260909`,
`DATA_SOURCE=oanda`, `RETRAIN_TIMEFRAME_MINUTES=15`, `RETRAIN_DAYS_BACK=730`,
matching the recipe that produced the currently-served
`models/forex_m15_wide`). **The gate correctly rejected the candidate** —
nothing was promoted, production is untouched. That part is not in question.

What needed digging into: the diagnostic "artifact holdout" evaluation (the
most recent ~18% chronological slice, scored on Fold 3's models even though
the fold gate had already failed) came back very bad — WR 14.6%, PF 0.34,
EV -0.56 on 41 trades — far worse than the walk-forward folds themselves
(Fold 3's own validation split scored PF 1.08, WR 83.8% on 37 trades). The
question: is this a real, structural weakness in the strategy, or an
artifact of a small/noisy sample?

## Investigation

`_evaluate_holdout` (`src/core/retrainer.py:2153`) computes the holdout
metrics from a boolean `approved_mask` over the holdout frame but only ever
logs aggregate stats — no per-trade detail. To see what the 41 (later 25,
see below) approved trades actually were, I added a temporary debug dump
right before the `return` in `_evaluate_holdout`, gated behind an
`HOLDOUT_DEBUG_DUMP` env var, writing `{symbol, timestamp, devil_prob,
macro_win}` for the approved rows to CSV. This is **not part of the
committed pass** — it was added, used, and reverted (`git checkout --
src/core/retrainer.py`) within this investigation; `git status` on that file
is clean as of this report.

Re-running the identical retrain recipe one day later (2026-09-10 instead of
2026-09-09, same env/config) to capture the dump produced a **different**
holdout sample: 25 trades (3 wins, WR 12.0%, PF 0.27, 271 tail-purged rows)
vs. the original run's 41 trades (6 wins, WR 14.6%, PF 0.34, 180 tail-purged).
This is expected, not a bug: `RETRAIN_DAYS_BACK` and the holdout fraction are
relative to "now," so a one-day-later fetch shifts every boundary (fold
splits, holdout start, the tail-purge cutoff) by one day's worth of bars, and
LightGBM refits on a slightly different window. **Worth flagging to whichever
model reviews this**: retrain results are not exactly reproducible day-to-day
even with an identical recipe, which matters for how much weight to put on
any single run's point estimates (this is why the gate already uses
Clopper-Pearson lower bounds rather than point PF — see
`_holdout_pf_lower_bound`, `src/core/retrainer.py:2304`).

The per-trade dump of the 25-trade rerun (full CSV reproduced below) showed
two patterns:

```
symbol,timestamp,devil_prob,macro_win
AUD_JPY,2026-07-29T00:45:00+0000,0.705,false
AUD_JPY,2026-08-03T00:15:00+0000,0.898,false
AUD_JPY,2026-08-03T00:30:00+0000,0.785,false
AUD_JPY,2026-08-03T00:45:00+0000,0.867,false
AUD_JPY,2026-09-07T23:30:00+0000,0.731,false
AUD_JPY,2026-09-07T23:45:00+0000,0.690,false
EUR_JPY,2026-08-03T00:15:00+0000,0.869,false
EUR_JPY,2026-08-03T00:30:00+0000,0.785,false
EUR_JPY,2026-08-03T00:45:00+0000,0.871,false
EUR_JPY,2026-09-07T23:45:00+0000,0.773,false
GBP_AUD,2026-05-14T23:00:00+0000,0.940,true
GBP_AUD,2026-06-14T21:30:00+0000,0.907,true
GBP_AUD,2026-08-26T23:00:00+0000,0.967,false
GBP_JPY,2026-07-10T00:45:00+0000,0.824,false
GBP_JPY,2026-08-03T00:15:00+0000,0.869,false
GBP_JPY,2026-08-03T00:30:00+0000,0.785,false
GBP_JPY,2026-08-03T00:45:00+0000,0.871,false
GBP_JPY,2026-09-07T23:45:00+0000,0.798,false
GBP_NZD,2026-05-28T23:00:00+0000,0.947,false
GBP_NZD,2026-06-14T21:30:00+0000,0.917,true
GBP_NZD,2026-07-20T23:00:00+0000,0.953,false
NZD_JPY,2026-07-30T13:30:00+0000,0.671,false
NZD_JPY,2026-08-03T00:30:00+0000,0.822,false
NZD_JPY,2026-08-03T00:45:00+0000,0.888,false
NZD_JPY,2026-09-07T08:15:00+0000,0.697,false
```

1. **Correlated clustering, not independent trials.** 8 of 25 rows (32%) are
   two calendar moments — `2026-08-03T00:15–00:45` and `2026-09-07T23:30–23:45`
   — where 4 JPY-cross symbols (AUD/EUR/GBP/NZD_JPY) all fire on the *same
   bar*. That is one directional yen move being counted as 4 trades, both
   times a loss. The PF/Brier/Clopper-Pearson math (`_evaluate_holdout`,
   `_holdout_pf_lower_bound`) treats every approved row as an independent
   Bernoulli draw; these aren't independent — they're the same bet 4x. There
   is no correlation cap anywhere in the pipeline (confirmed by grep — no
   per-timestamp or per-direction dedup in `_evaluate_holdout` or the fold
   scoring path). This has been flagged before but not fixed: memory note
   `project_m15_soak_2026-07-13` records the first live multi-fill day as "4
   fills in 20 min (3 simultaneous JPY-cross longs + ...)" with "no
   correlation cap across instruments" as an open gap.

2. **Session-timing skew.** Excluding the two clusters, essentially all
   remaining approvals sit in the 21:00–01:00 UTC band (NY close rolling into
   the Asia session open): `21:30, 23:00×2, 23:45×3, 00:15×3, 00:30×3,
   00:45×4`. Only two rows fall outside that band (`07:30:00` was actually
   `13:30` NZD_JPY and `08:15` NZD_JPY — both outliers). Gate C's blackout
   (`_DEFAULT_BLACKOUT_ET = "16:55-17:30"`,
   `src/execution/risk_manager.py:170`) is 16:55–17:30 ET — roughly
   20:55–21:30 UTC in EDT, 21:55–22:30 UTC in EST. The chop-veto log lines
   from this same holdout engineering pass confirm Gate C *did* fire inside
   the holdout (`gate_c=83–128` per JPY-cross symbol out of ~9,000 rows,
   `src/core/retrainer.py` chop-veto logging) — but its window is a narrow,
   mechanical "avoid the literal rollover spread-widening" filter, not a
   broad "avoid thin post-rollover liquidity" filter. The approvals cluster
   in the 1–4 hours *after* that window closes, which Gate C was never
   designed to cover.

3. Both observations point at the same underlying mechanism: this is a
   45-bar-hold, 2:1 R:R (SL=2.0×ATR / TP=4.0×ATR) momentum/trend bracket. It
   needs a sustained directional move to clear TP. The NY-close→Asia-open
   window is characteristically low-trend, low-liquidity chop — a bad match
   for what the bracket needs — and this reproduces the standing
   `trend_high` finding from the behavior-matrix work (memory:
   `project_behavior_matrix_tool` — "trend_high reliably LOSES... only cell
   significant in all 3 runs").

No claim is made here about *why* Aug-3 and Sep-7 specifically moved the way
they did (no macro/news correlation was checked) — only that the mechanism
(correlated same-bar cross-pair firing + session-timing skew) is visible
directly in the trade list and is structural, not one-off noise.

## Findings / Changes

1. **[Informational, not a bug]** Retrain point estimates are not exactly
   reproducible run-to-run because the fetch window is relative to "now."
   Any reviewer should weight the Clopper-Pearson lower bound, not the point
   PF/WR, and expect ±one day of data to move both a fair amount on a sample
   this small (25–41 trades).
2. **[Real gap, pre-existing]** No correlation cap across simultaneously-firing
   correlated symbols (same JPY-cross bar, same direction) exists anywhere in
   the training-side scoring OR the live orchestrator. A single directional
   event is over-counted as multiple independent trades in both the holdout
   metric and (per the 2026-07-30 live multi-fill day) live P&L exposure.
3. **[Real gap, pre-existing]** Gate C's rollover blackout (16:55–17:30 ET)
   does not cover the broader NY-close→Asia-open low-liquidity window where
   this holdout's losing trades actually cluster (21:00–01:00 UTC observed
   vs. ~20:55–21:30/21:55–22:30 UTC blackout).
4. Neither (2) nor (3) has been implemented. This report is diagnostic only —
   no fix was attempted or prototyped.

## Verification

- Reran the exact retrain recipe with `HOLDOUT_DEBUG_DUMP` set; confirmed the
  dump's own aggregate (25 trades, 3 wins) matches that run's own logged
  `HOLDOUT METRICS` line exactly (`logs/retrain_20260909_debug.log:369`).
- Confirmed the chop-veto log lines showing non-zero `gate_c` counts belong
  to the *holdout* frame's own engineering pass (they appear under this run's
  "ARTIFACT HOLDOUT EVALUATION" header, not the training/remainder pass),
  so Gate C is confirmed active on the holdout, not merely on training.
- Confirmed the debug instrumentation was fully reverted:
  `git status --short src/core/retrainer.py` is clean; `git diff` empty.
- Raw evidence retained for a follow-up agent:
  `logs/holdout_debug_dump.csv` (25-row per-trade dump),
  `logs/retrain_20260909_debug.log` (full run log with the dump enabled),
  `logs/retrain_20260909.log` (the original, non-instrumented run: 41 trades).

## Risk & follow-ups

- **Decision needed from Brandon** (this report exists to support that
  decision, not preempt it): whether to prototype (a) a correlation
  cap/dedup so simultaneous same-direction cross-pair signals score as one
  trade, (b) a broader session-timing stand-down than Gate C's rollover
  window, both, or neither. Any prototype should land as a training-side
  experiment in a side model dir first — no live effect — consistent with
  how every other candidate change in this repo has been staged.
- If a fix is attempted, re-validate against **both** this run's 41-trade
  sample and the rerun's 25-trade sample (or refetch fresh) rather than
  tuning to one day's snapshot, given finding #1 above.
- `logs/holdout_debug_dump.csv` and `logs/retrain_20260909_debug.log` are
  untracked scratch artifacts from this investigation — not cleaned up,
  left for the next agent to inspect; safe to delete once reviewed.
- No source, config, or model artifact changed as a result of this
  investigation. `models/forex_m15_candidate_20260909/` was never created
  (gate failed both times, before full-data training).

## Files touched

_Read-only investigation. Files read/grepped:_
- `src/core/retrainer.py` (`_evaluate_holdout`, `_score_artifact_holdout`,
  `_holdout_pf_lower_bound`, `_split_holdout`, chop-veto logging) —
  temporarily instrumented and reverted, see Investigation.
- `src/execution/risk_manager.py` (`_in_blackout`, `_DEFAULT_BLACKOUT_ET`,
  `RiskProfile` blackout fields)
- `logs/retrain_20260909.log`, `logs/retrain_20260909_debug.log`,
  `logs/holdout_debug_dump.csv` (this investigation's own output)
- `llm_reports/refactors/2026-09-09_safety-and-training-integrity-pass.md`
  (prior context)
