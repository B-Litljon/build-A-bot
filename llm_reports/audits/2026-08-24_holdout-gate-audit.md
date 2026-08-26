---
type: audit
date: 2026-08-24
time: 15:10 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "Audit Kimi K3's holdout implementation (c7d5a93) before anything relies on it."
head: ced5600
scope: read-only
related:
  - m2m/2026-08-24_overfit-gate-and-the-holdout-fix.md
  - refactors/2026-08-24_artifact-holdout-gate.md
  - recons/2026-08-23_behavior-matrix-and-the-trend-high-hole.md
---

# Audit: the artifact holdout gate

## Context

Commit `c7d5a93` (Kimi K3) implements the holdout brief: carve a chronologically
last slice before feature engineering, train folds *and* the final artifact on
the remainder only, then gate the artifact on the untouched slice. The change
touches `src/core/retrainer.py`, the most consequential offline file in the
repo, and its whole value rests on one property being literally true — so it was
verified by execution, not by reading.

## Investigation

**The no-leak property, tested empirically.** Rather than trust the data flow, I
monkeypatched `refit_models` and `_evaluate_holdout` to record the timestamp
range of every frame they received, then ran a real 365-day retrain into a side
directory:

```
refit_models       2025-08-25 12:00 → 2026-01-21 11:45   62,283 rows  (fold 1)
refit_models       2025-08-25 12:00 → 2026-03-11 11:45   83,061 rows  (fold 2)
refit_models       2025-08-25 12:00 → 2026-04-30 11:45  103,610 rows  (fold 3)
refit_models       2025-08-25 12:00 → 2026-06-19 20:00  124,843 rows  (ARTIFACT)
holdout starts at  2026-06-20 04:44
_evaluate_holdout  2026-06-22 12:00 → 2026-08-24 20:00   26,158 rows
```

Every training call ends strictly before the holdout begins. **CLEAN.**

**Reproducibility.** Two runs at a pinned `RETRAIN_END_DATE=2026-08-23` produced
byte-identical metrics (Brier 0.1372, EV 1.345455, WR 40.0%, PF 1.3333,
55 trades). The pipeline is deterministic given a fixed window.

## Findings

**Finding 1 (pass) — the core property holds, and the gate bites.** Beyond the
data-flow proof above: the Devil threshold is frozen from fold *n-1* and passed
into the holdout evaluation rather than re-tuned on it; `feature_stats` is
computed from the remainder; `metadata.json` records the holdout fraction, date
range, metrics and a `bypass_reason`; disabling or emptying the holdout logs a
loud warning and records why. A failing holdout sets `report.gate_passed=False`,
which `promote_or_reject` honours.

It demonstrably rejects: one run passed the fold gate at PF 1.6111 and was then
refused at holdout PF 0.9818 — "rejected by data it never saw". That is exactly
the failure the previous gate was blind to.

**Finding 2 (high) — the gate's verdict is not stable enough to decide
promotion.** The same configuration, at three window endpoints:

| window end | holdout PF | win rate | trades | verdict |
|---|---:|---:|---:|---|
| 2026-08-24 ~22:00 | 1.444 | 41.9% | 62 | **PASS** |
| 2026-08-24 ~23:00 | 0.982 | 32.9% | 82 | **FAIL** |
| 2026-08-23 (pinned) | 1.333 | 40.0% | 55 | **PASS** |

The first two are the *same day, an hour apart*. Since pinned runs are
bit-identical, this is not nondeterminism — it is sampling noise on a small
holdout. Whether a model promotes currently depends on what time the retrain
runs.

Root cause is the holdout trade floor:

```python
holdout_trade_floor = BASELINE_POOLED_OOS_TRADES * HOLDOUT_FRAC * (1.0 - chop_veto_rate)
# 300 * 0.18 * 0.776 ≈ 42
```

The fold gate requires **233** pooled trades; the holdout — the more decisive
test — requires **42**. Evidentially backwards. A profit factor computed on
55–82 trades cannot separate 0.98 from 1.44.

**Finding 3 (low) — no purge gap between remainder and holdout.** The
remainder's final `max_hold` (45) bars need bars that now live in the holdout to
resolve their labels. Split first, engineered separately, those walks run off the
end of the frame and resolve to "timeout → loss". About 45 bars × 8 symbols ≈
360 rows of the 161,494-row training set (0.2%) carry a systematically wrong
label. Not leakage, and small — but it is standard walk-forward hygiene and
cheap to fix by dropping the last `max_hold` bars of the remainder.

Partly self-mitigating in the other direction: indicator warm-up consumes the
holdout's first ~2 days (holdout starts 06-20 04:44, scoring starts 06-22
12:00), which happens to create an embargo. That is accidental, not designed.

**Finding 4 (low) — a failed fold gate suppresses the holdout score.** The
holdout only runs `if ... and report.gate_passed`. A model that fails folds is
never scored on the holdout, so the two numbers can never be compared for a
rejected candidate — which is precisely when the comparison would be most
informative.

**Finding 5 (informational) — fold boundaries now scale with the frame span.**
Beyond the brief's scope. It is the right fix (the schedule previously derived
from the module constant `DAYS_BACK`), but it changes gate behaviour, so gate
numbers from before `c7d5a93` are **not comparable** to numbers after it. Worth
stating loudly wherever August's scores are cited.

## Verification

- Leak test: instrumented run, `RETRAIN_MODEL_DIR=models/_audit_leak_check`,
  365-day window, 8 instruments. Result above. Side dirs removed afterwards.
- Reproducibility: two pinned runs, identical to the digit.
- Rejection behaviour: observed live on an unpinned run (fold 1.6111 → holdout
  0.9818 → rejected).
- Suite: **310 passing**, `compileall` clean.
- The live model (`models/forex_m15_wide`) was not touched at any point —
  verified by mtime (still 2026-08-08 22:56). All runs used
  `RETRAIN_MODEL_DIR`.

## Risk & follow-ups

1. **Do not promote on a single unpinned run.** Until Finding 2 is addressed,
   require a pinned `RETRAIN_END_DATE` and agreement across several endpoints —
   the same discipline that settled the 5-year question.
2. **Raise the holdout floor or widen the holdout.** Options: lift the floor
   toward the fold gate's evidential standard, raise `HOLDOUT_FRAC`, or require
   the holdout PF to clear the bar on N pinned windows rather than one.
3. **Drop the last `max_hold` bars of the remainder** to purge the truncated
   labels (Finding 3).
4. **Score the holdout even when folds fail** (Finding 4).
5. Kimi's implementation is otherwise sound. The design is right, the property
   is real, and the gate does reject models the old one would have shipped.
