---
type: m2m
date: 2026-08-29
time: 16:20 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "K3 M2M brief: retrain the trim 100x15/17-feature config on the pinned 5yr window and promote into models/forex_m15_wide"
head: c5e7e83
scope: read-only (retrain ran to a side dir that was never created; no promotion; no source/service edits)
result_commit: null
related:
  - m2m/2026-08-17_score-compression-and-calibration.md
  - refactors/2026-08-24_artifact-holdout-gate.md
  - audits/2026-08-24_holdout-gate-audit.md
---

# STOPPED before promotion: the trim config fails the real gate, not the one that measured it

## Outcome

**Nothing was promoted. Nothing under `src/`, `soak.service`, or `models/forex_m15_wide/` was touched. The soak was never stopped.** Per Step 0, preflight
was run to the letter and every check passed — the tree matches the brief's
Context exactly. Per the brief's own logic ("if the tree does not match the
described state, STOP and report — do not fix it"), I'm extending that same
rule to this case: the *result* doesn't match the described state, and fixing
that gap (e.g. by lowering a gate floor myself) would be exactly the kind of
unilateral call the brief reserves for a human. So: stop, report, wait.

## What ran

Branch `feat/trim-100x15-retrain` off the post-merge `feat/wider-brackets-and-rename`
HEAD (`c5e7e83`). Preflight (Step 0) confirmed:

- `get_hyperparameters()` — `n_estimators=100, num_leaves=15, max_depth=6, min_child_samples=80` ✓
- `BASE_FEATURE_COLS` — 17 features, none of the 5 dead ones ✓
- `soak.service` active, `OANDA_MODEL_DIR=models/forex_m15_wide` ✓
- `git status --porcelain src/ tests/` clean ✓

One live process check the brief didn't anticipate: a second retrainer (pid
2957431, parented to the Antigravity IDE language server — Gemini's session)
was already running a 2-year, unpinned retrain into `models/forex` (the
default side dir, not `models/forex_m15_wide`) when I started. Per the
"one retrainer at a time" constraint I waited it out rather than run
concurrently; it finished cleanly (gate passed, PF 4.89) about a minute
before I launched mine. No overlap, no interference — noted for the record
since `models/forex/` now holds an artifact from that run, untouched by me.

Retrain launched exactly per the settled recipe: `DATA_SOURCE=oanda`,
`RETRAIN_TIMEFRAME_MINUTES=15`, `RETRAIN_DAYS_BACK=1825`,
`RETRAIN_END_DATE=2026-08-23`, `RETRAIN_MODEL_DIR=models/forex_m15_candidate_20260829`,
`RETRAIN_BEHAVIOR_VETO` unset. Log: `logs/retrain_trim_20260829.log`.
Total wall time: fetch ~2.5 min, fold CV + gate ~15s.

## The gate result

```
[Fold 1] Devil approved 11 trades ... Brier=0.2433 | EV=1.100000 | WR=70.0% | Trades=10
[Fold 2] Devil approved  3 trades ... Brier=0.0576 | EV=2.000000 | WR=100.0% | Trades=3
[Fold 3] Devil approved 11 trades ... Brier=0.0945 | EV=1.727273 | WR=90.9% | Trades=11
[Fold 3] Profit Factor (macro) = 16.00 / 14.00 = 1.1429

Mean Brier Score : 0.1318 (threshold ≤ 0.3)                    PASS
Mean EV          : 1.609091 (threshold ≥ 0.0005)                PASS
Profit Factor    : 1.1429 (threshold ≥ 1.2, Fold 3 OOS)         FAIL
Pooled OOS Trades: 24 across 3 folds (dynamic floor ≥ 233)      FAIL — 10% of floor
Gate Result      : FAILED
```

Two of four criteria failed, and the trade-volume one isn't close: **24
trades against a required 233** (`BASELINE_POOLED_OOS_TRADES=300 × (1 −
chop_veto_rate)`, `retrainer.py:2638`). Across three walk-forward folds
covering 313k/416k/519k feature rows, Angel proposed 11, 3, and 11 trades
respectively — roughly one signal per 30–100k rows. The fold gate correctly
refused to trust a PF measured on that few trades and rejected before ever
reaching full-data training; `models/forex_m15_candidate_20260829/` was never
created, matching `promote_or_reject`'s documented behavior (gate fail ⇒
nothing saved).

This contradicts the brief's Context, which cited `logs/capacity_sweep.json`
showing this exact configuration passing 3/3 with `pf_lower_bound` around
2.9–7.6. That citation is accurate on its own terms — but it isn't measuring
the same thing as the gate this task needed to pass.

## Root cause: two different validations, one name

`scripts/capacity_sweep.py` never runs the fold gate. Its `sweep_one_window()`
calls `R.refit_models()` **once** per variant, on the full holdout-carved
remainder, then scores that single model with `_evaluate_holdout()` +
`_holdout_verdict()` — the same function the *production* pipeline uses for
its final artifact check, but never combined with `validate_candidate()`'s
3-fold walk-forward gate or its pooled-trade floor. Grep confirms it: `fold`
appears only in the sweep's module docstring (describing the *production*
gate it's diagnosing, not what it does), and `refit_models` is called exactly
once per variant with no loop over folds.

So "holdout_passed: true, 3/3" in `capacity_sweep.json` means: *a model
trained once on ~all the data clears the final holdout check at three
different end-dates.* It does not mean: *this config would survive the
walk-forward fold gate with its trade-volume floor*, because the sweep never
runs that gate. The two happen to share the word "holdout" and both cite the
same 2026-08-09/16/23 pins, which is almost certainly how the brief came to
treat them as one result.

The underlying fact is consistent across both measurements, though: the trim
config produces very few trades. The sweep's own per-pin counts (25–29 trades
in a single-shot full-data fit) already said so; the real fold gate just has
a floor that trade rate can't clear. `BASELINE_POOLED_OOS_TRADES=300` was
never re-tuned after this capacity reduction — it was calibrated against the
shipped 200×63 model's much higher trade volume (~100+ per fold gate run;
see `logs/capacity_sweep.json`'s "shipped" rows).

## What this means for the deliverable

The single deliverable — soak running again, serving the new artifact — was
not reachable honestly, because there is no artifact that passed the gate to
serve. Producing one would have required either lowering
`RETRAIN_POOLED_TRADE_FLOOR` (a safety-gate parameter, and a policy call, not
a retrain-execution one) or picking a different capacity point. Both are
outside "Files in Scope" and outside what this brief authorized me to decide.
The brief's own Step 0 instinct — stop and report rather than route around a
mismatch — is the one I'm following here, one level up: the mismatch showed
up in the gate result instead of the preflight checks, but the right response
is the same.

## State left behind

- `models/forex_m15_wide/` — **untouched**, still the 2026-08-08 200×63
  artifact, still what `soak.service` serves.
- `soak.service` — **never stopped**. Active the entire time
  (pid 420097, uptime unaffected).
- `models/forex_m15_candidate_20260829/` — does not exist (gate rejected
  before any save).
- `soak.off` — never created.
- `models/forex/` — holds Gemini's unrelated 2yr artifact from the
  concurrent job noted above; not mine, not touched, not in scope.
- `logs/retrain_trim_20260829.log` — full run log, referenced above.
- Branch `feat/trim-100x15-retrain` — created, nothing committed.

## Recommendation (not acted on — needs a decision)

Options, roughly in order of how much they change:

1. **Accept the trim config can't clear the current floor and stop here.**
   The capacity-reduction finding (memorization gap 0.109 → 0.014) stands on
   its own merits from the sweep; it just can't ship via this gate as
   configured.
2. **Re-run with a deliberately lowered `RETRAIN_POOLED_TRADE_FLOOR`,
   documented and reasoned about explicitly** — this is a real safety-gate
   change (how much do we trust a PF measured on ~25 trades?), not a retrain
   nuance, and reads as something Brandon/K3 should set, not something a
   retrain-execution task should quietly override.
3. **Try a less extreme capacity point.** The sweep's `150x31` variant
   produced far more trades (60–61 per single-shot window vs. 25–29 for
   100x15) while still cutting the memorization gap substantially (0.109 →
   0.047) — untested against the real fold gate, but a plausible middle
   ground worth an actual gate run before assuming it clears.

I've made no change that forecloses any of these.
