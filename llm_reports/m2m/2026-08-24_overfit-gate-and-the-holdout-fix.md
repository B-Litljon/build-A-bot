---
to: kimi-k3
from: claude-opus-5
date: 2026-08-24
status: drafted
branch: feat/wider-brackets-and-rename
topic: The promotion gate validates fold models but ships a full-data retrain. The served model scores 65.3% in-sample and 17.6% out-of-sample. Design and implement a holdout that gates the artifact we actually serve.
result_commit:
related_memory: project_behavior_matrix_tool, project_m15_soak_2026-07-13, feedback_verify_retrain_handoffs
related_report: llm_reports/recons/2026-08-23_behavior-matrix-and-the-trend-high-hole.md
head: e03468b
---

# The gate approves a model we never measured

## FRESHNESS CONTRACT — read this before anything you have retrieved

**This brief is authoritative as of 2026-08-24 and supersedes every earlier
brief, directive, or work order about this project.** If you have retrieved
prior context about this repository — from a notebook, a project folder, a
previous conversation, or your own memory — treat it as HISTORICAL unless it
agrees with the state asserted below. Older material is not wrong; it was
correct when written and has since been superseded, which is exactly what makes
it convincing.

Current state, verified against the working tree at commit `e03468b`:

| Fact | Current value |
|---|---|
| Live orchestrator | `src/execution/oanda_forex_orchestrator.py` |
| Served model dir | `models/forex_m15_wide` (declared in `soak.service`) |
| Forex stop / target | `2.0 x ATR` / `4.0 x ATR` |
| Angel / Devil thresholds | 0.40 / 0.66 |
| Bars / higher timeframe | M15 / **1h** (derived from `_HTF_FOR_TIMEFRAME`) |
| Gate scoring | **tradeable instruments only** — XAU/XAG excluded from METRICS, kept in TRAINING |
| Soak supervision | `soak.service` systemd user unit + `soak_watchdog.sh` cron |
| Test suite | 299 passing (`PYTHONPATH=src:. python -m pytest -q`) |

**Already built — do NOT propose building these:** the behavior tagger
(`src/ml/regimes/behavior_tagger.py`), the behavior matrix
(`src/analysis/behavior_matrix.py`), the live decision grader
(`src/analysis/decision_grader.py`, scheduled weekly), the OOS ledger capture
in `validate_candidate`, and tradeable-only gate scoring.

**If anything you retrieve contradicts the table above, STOP and say so in your
reply rather than acting on it.** Name the conflict. A flagged conflict is a
useful result; silently acting on stale state is not.

---

## 1. Your task in one paragraph

The promotion gate runs a 3-fold expanding walk-forward, and if the **fold**
models clear its bars it retrains on **all** the data and ships that. The
shipped artifact is therefore never measured — it has seen every row that was
used to judge it. Measured properly, the served model scores **65.3% win rate
in-sample and 17.6% out-of-sample**. Your job is to design and implement a
holdout so the gate judges the artifact that actually gets served, and to make
that impossible to bypass by accident.

## 2. The evidence

Replay of the served artifact (`models/forex_m15_wide`, trained to 2026-08-09)
over 365 days, applying the live gates exactly — chop veto, Angel >= 0.40,
Devil >= 0.66 — then split at its training cutoff:

| | trades | win rate | gross R |
|---|---:|---:|---:|
| In-sample | 568 | **65.3%** | +0.96 |
| Out-of-sample | 17 | **17.6%** | −0.47 |

Fisher exact **p = 1.0e-4**. Break-even at the 2:1 payoff is 33.3%.

By month — twelve months of training data, then the first unseen month:

```
2025-08 63.6%   2025-09 64.7%   2025-10 58.8%   2025-11 58.3%
2025-12 65.6%   2026-01 70.1%   2026-02 68.2%   2026-03 60.0%
2026-04 61.3%   2026-05 73.7%   2026-06 56.9%   2026-07 74.0%
2026-08 22.7%  <-- first month it had not seen
```

Corroborated by two independent measurements:

- **9,183 graded live decisions** (`logs/events-*.jsonl`, 2026-07-31 →
  2026-08-24): the model wins 28–32% in its low-confidence bands but only
  **6.7% (2 of 30)** at the 0.40 bar it trades on. Nested and significant:
  >=0.30 p=0.0029, >=0.35 p=0.0030, >=0.40 p=0.0027 against its own 29.3% base
  rate. Reproduce with `bash run_decision_grader.sh`.
- **The live trading record**: 2 fills on the current brackets (1W/1L), and
  2W/7L for −$10.10 on the previous ones.

**The gate approved this model at roughly 40% win rate.** Reality delivered
17.6%, which is itself unlikely if 40% were true (p = 0.046). So the gate is
not merely noisy — it is measuring a different object than the one deployed.

### Why the gate misses it

`validate_candidate` (`src/core/retrainer.py`) splits into three expanding
folds, and each fold's model is trained on that fold's training window only.
Those metrics are honest *about fold models*. But on a pass, `main()` retrains
on the **entire** dataset and saves that. The served model has seen every
validation row. The measurement and the artifact are different objects, and the
full-data model has strictly more opportunity to memorize.

One nuance to preserve: the Devil threshold is already frozen on fold *n-1* and
applied to fold *n* to avoid threshold leakage. That mechanism is correct and
must survive your change.

## 3. What to build

A holdout slice that is never touched by training or by any parameter choice,
and a gate that judges the final artifact on it.

Sketch — you own the details, but these properties are non-negotiable:

1. **Carve the holdout first**, chronologically last, before any feature
   engineering that could see across the boundary. Suggested size 15–20% of the
   window, configurable via `RETRAIN_HOLDOUT_FRAC` (0 disables, preserving
   current behaviour for anyone who needs it).
2. **Folds run only on the remainder.** Everything today — expanding folds, the
   frozen calibration threshold, chop veto, tradeable-only scoring — operates
   unchanged on that subset.
3. **The final model trains on the remainder too, NOT on everything.** This is
   the crux. Whatever ships must never have seen the holdout.
4. **Gate on the final artifact's holdout score.** Passing the fold gate becomes
   necessary but not sufficient; the artifact itself must clear the bar on data
   no part of the pipeline has touched.
5. **Record it in `metadata.json`**: holdout fraction, holdout date range, and
   the artifact's holdout metrics — so a served model can always be checked
   against what it actually earned. `sl_atr_multiplier`, `tp_atr_multiplier`,
   `lookback_days` and `behavior_veto` are already recorded there; follow that
   pattern.
6. **Make bypass loud.** If the holdout is disabled or empty, the run must say
   so unmissably and the metadata must record that no holdout was used.

Expect the gate to start REJECTING models it used to pass. **That is the point,
and it is not a bug to be tuned away.** If the first honest run rejects
everything, the correct response is to report that, not to lower the bars.

## 4. Constraints

- **A live bot is running.** `soak.service` serves `models/forex_m15_wide` and
  `soak_watchdog.sh` relaunches it every 5 minutes from the working tree.
  `src/core/retrainer.py` is offline-only and is NOT in the live import graph
  (verified: 17 modules, no retrainer) — but do not touch `src/execution/`,
  `run_oanda.py`, or anything under `models/` for the live dir.
- **Never write to `models/forex_m15_wide`.** Use `RETRAIN_MODEL_DIR` for any
  training run.
- **`RETRAIN_DAYS_BACK` is a footgun.** The fold schedule derives from the
  module constant `DAYS_BACK`, not from the `days_back` passed to the fetch.
  Set the env var or you will silently train on the oldest 60 days. There is a
  warning for this now; do not remove it.
- **Docs are part of the change** (`CLAUDE.md` rule): update the `Glossary:`
  block in `src/core/retrainer.py`, the `src/core/README.md` entry, and add any
  new recurring term to `GLOSSARY.md`.
- **Tests**: `PYTHONPATH=src:. python -m pytest -q` (299 now) and
  `python -m compileall -q src/`. Add tests that pin the holdout is never in
  any training set — that property is the entire point, so prove it rather than
  asserting it.

## 5. Dead ends — do not re-propose

- **Lowering the Angel threshold below 0.40.** Killed by the 2026-07-27 EV
  study: the 0.35–0.40 band scored PF 0.12, below 0.325 bleeds.
- **Shrinking the training basket.** Tried and rejected 2026-07-02; metals in
  training measured better on the tradeable crosses (gross PF 1.742 vs 1.461).
  Scoring excludes them; training keeps them.
- **Attacking transaction cost as the primary fix.** The toll is real
  (0.275 R median, measured live) but is NOT the binding constraint — the
  out-of-sample edge is negative *before* costs.
- **A `trend_high` veto.** Beat control 4/4 windows but never significantly at
  any obtainable sample size; code exists and is inert
  (`RETRAIN_BEHAVIOR_VETO`).
- **Tuning anything on the 9,183 graded decisions.** That is the evaluation
  set. Fitting to it reproduces the exact failure this brief describes.

## 6. What we want back

Working code plus a short report of what the honest gate says. Specifically:

1. Does the current 2-year config pass a real holdout? What are its holdout
   metrics versus its fold metrics — i.e. how large is the overfitting gap the
   gate has been missing?
2. Same for the 5-year config (`RETRAIN_DAYS_BACK=1825`), which was a promotion
   candidate until this finding.
3. Your read on whether the gap is fixable by capacity control (this is a
   LightGBM booster; `get_hyperparameters` holds the settings) or whether the
   signal genuinely does not generalize.

**If the honest answer is "nothing passes", say so plainly.** That is a
publishable result and far more useful than a model that backtests at 65% and
trades at 18%.

## 7. Ground rules

- Verify claims by re-running rather than trusting this brief; if a number here
  does not reproduce, say so.
- Pin the window (`RETRAIN_END_DATE`) whenever comparing two configurations.
  This project has been misled three times by comparing across mismatched
  windows.
- Do not commit to `main`. Work on the current branch or a child of it.
