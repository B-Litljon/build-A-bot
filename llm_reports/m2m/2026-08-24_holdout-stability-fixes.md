---
to: deepseek-v4-pro
from: claude-opus-5
date: 2026-08-24
status: drafted
branch: feat/wider-brackets-and-rename
topic: The artifact holdout gate works and is verified leak-free, but decides on 55-82 trades — the same config passes at PF 1.44 and fails at 0.98 an hour apart. Make the verdict stable, and purge the boundary labels.
result_commit:
related_memory: project_behavior_matrix_tool, feedback_verify_retrain_handoffs
related_report: llm_reports/audits/2026-08-24_holdout-gate-audit.md
head: 04d4cd1
---

# The holdout gate is correct. It is not yet a decision procedure.

## FRESHNESS CONTRACT — read this before anything you have retrieved

**This brief is authoritative as of 2026-08-24 and supersedes every earlier
brief, directive, or work order about this project**, including your own
2026-08-22 behavior-classifier recon. If you have retrieved prior context —
from a notebook, a project folder, a previous conversation, or your own memory —
treat it as HISTORICAL unless it agrees with the state below. Older material is
not wrong; it was correct when written and has since been superseded, which is
exactly what makes it convincing.

Current state, verified against the working tree at commit `04d4cd1`:

| Fact | Current value |
|---|---|
| Live orchestrator | `src/execution/oanda_forex_orchestrator.py` |
| Served model dir | `models/forex_m15_wide` (declared in `soak.service`) |
| Forex stop / target | `2.0 x ATR` / `4.0 x ATR` |
| Angel / Devil thresholds | 0.40 / 0.66 |
| Bars / higher timeframe | M15 / **1h** (`_HTF_FOR_TIMEFRAME`, derived from bar size) |
| Gate scoring | tradeable instruments only — XAU/XAG out of METRICS, kept in TRAINING |
| **Artifact holdout gate** | **BUILT AND VERIFIED** (`c7d5a93`), `RETRAIN_HOLDOUT_FRAC` default 0.18 |
| Test suite | 310 passing (`PYTHONPATH=src:. python -m pytest -q`) |

Two corrections to your 2026-08-22 recon, both since verified:

- It said `models/forex/` is hot-reloaded by the live soak. **It is not** — that
  directory was last written 2026-06-14. The live directory is
  `models/forex_m15_wide`.
- Its preliminary behavior matrix ranked `trend_high` best and `range_low`
  worst. Proper walk-forward measurement **reverses both**. Do not cite it; see
  `llm_reports/recons/2026-08-23_behavior-matrix-and-the-trend-high-hole.md`.

**If anything you retrieve contradicts the table above, STOP and say so in your
reply rather than acting on it.** A flagged conflict is a useful result.

---

## 1. Your task in one paragraph

Kimi K3 built an artifact-level holdout gate (`c7d5a93`) and I audited it: the
core property is real and empirically verified, and the gate demonstrably
rejects models the old one shipped. **Do not redesign it.** But its verdict
currently rests on 55–82 trades, and the same configuration passes at holdout
PF 1.444 and fails at 0.982 depending on what hour the retrain runs. Your job is
to make that verdict stable enough to promote on, and to fix two smaller
correctness issues found in the audit.

## 2. What is already true — do not rebuild it

Verified by instrumenting `refit_models` and `_evaluate_holdout` during a real
365-day retrain and recording the timestamp range every frame received:

```
refit_models       ... → 2026-01-21 11:45   (fold 1)
refit_models       ... → 2026-03-11 11:45   (fold 2)
refit_models       ... → 2026-04-30 11:45   (fold 3)
refit_models       ... → 2026-06-19 20:00   (THE ARTIFACT)
holdout starts at      2026-06-20 04:44
_evaluate_holdout  2026-06-22 12:00 → 2026-08-24 20:00
```

Every training call ends strictly before the holdout. Also confirmed working:
the Devil threshold is frozen from fold *n-1* and passed into the holdout
evaluation rather than re-tuned on it; `feature_stats` comes from the remainder;
`metadata.json` records the holdout fraction, date range, metrics and a
`bypass_reason`; a failing holdout sets `report.gate_passed=False` and
`promote_or_reject` honours it. Observed live: a model passed the fold gate at
PF 1.6111 and was refused at holdout PF 0.9818.

**These properties are the point of the change. Preserve every one of them.**
If a fix of yours would weaken any, stop and say so instead.

## 3. Finding 1 (high) — the verdict is sampling noise

Same configuration, three window endpoints:

| window end | holdout PF | win rate | trades | verdict |
|---|---:|---:|---:|---|
| 2026-08-24 ~22:00 | 1.444 | 41.9% | 62 | **PASS** |
| 2026-08-24 ~23:00 | 0.982 | 32.9% | 82 | **FAIL** |
| 2026-08-23 (pinned) | 1.333 | 40.0% | 55 | **PASS** |

The first two are the same day, an hour apart. Two runs at a pinned
`RETRAIN_END_DATE` produce **bit-identical** metrics, so this is not
nondeterminism — it is a small-sample problem. Whether a model promotes
currently depends on what time you run the retrain.

Root cause:

```python
holdout_trade_floor = BASELINE_POOLED_OOS_TRADES * HOLDOUT_FRAC * (1.0 - chop_veto_rate)
# 300 * 0.18 * 0.776 ~= 42
```

The fold gate demands **233** pooled trades. The holdout — the more decisive
test — demands **42**. That is evidentially backwards, and a profit factor on 55
trades cannot separate 0.98 from 1.44.

**Fix it. You choose how**, but justify the choice with numbers rather than
taste. Options we see, not ranked:

- Raise the holdout floor toward the fold gate's evidential standard.
- Raise `HOLDOUT_FRAC` (costs training data — quantify the trade-off).
- Require the artifact to clear the bar on **several pinned windows**, not one.
  This is the discipline that settled the 5-year question and is probably the
  most honest, but it multiplies retrain cost — say what it costs.
- Report a confidence interval on the holdout PF and gate on its lower bound
  rather than the point estimate.

Whatever you pick, **a promotion must not depend on the clock.** Demonstrate
that by running your fixed gate at three or more pinned endpoints and showing
the verdict is consistent — or reporting honestly that it still is not.

## 4. Finding 2 (low) — no purge gap at the boundary

The remainder's last `max_hold` (45) bars need bars that now live in the holdout
to resolve their labels. Because the split happens on raw data and each side is
engineered separately, those walks run off the end of the frame and resolve to
"timeout → loss". Roughly 45 bars × 8 symbols ≈ 360 of 161,494 training rows
(0.2%) carry a systematically wrong label.

Not leakage, and small — but standard walk-forward hygiene. Drop the last
`max_hold` bars of the remainder after engineering, and log how many rows went.

Note the other direction is already covered by accident: indicator warm-up
consumes the holdout's first ~2 days. Make that deliberate rather than
incidental if it is cheap to do so.

## 5. Finding 3 (low) — the holdout is skipped when folds fail

`_evaluate_holdout` only runs `if ... and report.gate_passed`. A model that
fails the fold gate is never scored on the holdout, so the two numbers cannot be
compared for a rejected candidate — exactly when the comparison is most
informative (is the fold gate too strict, or the model genuinely bad?).

Score the holdout regardless; keep the promotion logic unchanged.

## 6. Constraints

- **A live bot is running.** `soak.service` serves `models/forex_m15_wide` and
  `soak_watchdog.sh` relaunches it every 5 minutes from the working tree.
  `src/core/retrainer.py` is offline-only and NOT in the live import graph
  (verified: 17 modules) — but do not touch `src/execution/`, `run_oanda.py`,
  or the live model directory.
- **Never write to `models/forex_m15_wide`.** Use `RETRAIN_MODEL_DIR`. Clean up
  any side directories you create.
- **Pin `RETRAIN_END_DATE` whenever comparing two things.** This project has
  been misled four times by comparing across mismatched windows — the 5-year
  candidate, the pooled metals score, the `trend_high` veto, and now this gate.
- **`RETRAIN_DAYS_BACK` is a footgun**: the fold schedule derives from the
  module constant, not the `days_back` passed to the fetch. There is a warning
  for it; do not remove it.
- **Docs are part of the change** (`CLAUDE.md`): the `Glossary:` block in
  `src/core/retrainer.py`, the `src/core/README.md` entry, and `GLOSSARY.md` for
  any new recurring term.
- **Tests**: `PYTHONPATH=src:. python -m pytest -q` (310 now) and
  `python -m compileall -q src/`. The audit's leak test is worth keeping as a
  permanent test rather than a one-off script — the property is load-bearing and
  currently only proven by a temporal-separation assertion.

## 7. Dead ends — do not re-propose

- **Lowering the Angel threshold below 0.40** — killed by the 2026-07-27 EV
  study (0.35–0.40 band scored PF 0.12).
- **Shrinking the training basket** — tried and rejected 2026-07-02; metals in
  training measure better on the tradeable crosses (1.742 vs 1.461 gross PF).
- **Transaction cost as the primary fix** — my own wrong call on 2026-08-23. The
  toll is real (0.275 R median, measured live) but the out-of-sample edge is
  negative *before* costs.
- **A `trend_high` veto** — beat control in 4/4 windows but never significantly
  at any obtainable sample size. Code exists and is inert
  (`RETRAIN_BEHAVIOR_VETO`).
- **Weakening the holdout to make models pass.** The gate rejecting things is
  the feature. If your fixed gate rejects everything, that is the result.

## 8. What we want back

1. Working code, with the stability fix demonstrated across three or more pinned
   endpoints.
2. A short report: what the stable gate says about the current 2-year config and
   the 5-year config (`RETRAIN_DAYS_BACK=1825`), each on the same pinned windows.
3. Your judgement on whether the holdout can be made decisive at all at this
   data volume, or whether ~55 trades per holdout is simply the ceiling and
   promotion needs a different kind of evidence entirely.

**"Nothing passes a stable gate" is a publishable answer** and considerably more
useful than a model that backtests at 65% and trades at 18%. Do not tune until
something passes.
