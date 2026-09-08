# STAGE 2 — CatBoost swap for Angel/Devil (one variable changed)

You are implementing Stage 2 of the quant-architecture roadmap in this repo
(build-A-bot). Stage 1 (quantile barriers, `src/ml/barriers/`) is complete and
committed. Your job: replace the LightGBM Angel/Devil with CatBoost
(ordered boosting + monotone constraints) on **identical folds**, and measure
whether it clears the promotion gate AND narrows the calibration inversion.

## Hard rails (violating any of these voids your work)

- The M15 soak is LIVE. Check `ps aux | grep '[r]un_oanda'`. Never write under
  `models/`. Never stop/restart the soak. If you need it stopped for any
  reason, do not — your work must not require it.
- Train/live symmetry: any gate/veto/feature change must land in BOTH the live
  path and `retrainer._compute_chop_veto_mask` in the same change.
- Realised-R accounting; per-instrument `config/spread_alphas_m15.json` costs,
  never flat constants. n>=30 per reported cell. Clopper-Pearson bounds, never
  point estimates, for anything gating a decision.
- Tests: `PYTHONPATH=src:. /home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python -m pytest -q`
  must be green (407 tests as of 2026-09-08) before and after your change.
- `catboost` is NOT installed. Install it user-space if you need it:
  `pip install --user catboost` — or use the venv's pip. Do not sudo-install.
- Do not commit. Report and leave the working tree with your changes staged
  for review; promotion to live is a human step.

## What to build

1. In `src/core/retrainer.py`, add a trainer family parallel to
   `V3RandomForestTrainer` (which actually holds a LightGBM model — beware,
   the name lies): CatBoost with `boosting_type="Ordered"`, and monotone
   constraints on:
   - `cost_ratio` -> monotonically DECREASING acceptance probability
   - `natr_14`, `vol_rel` -> monotonically INCREASING excursion width
     (only relevant if barriers are joint; for the classifier use them as
     monotone-increasing for acceptance only if evidence supports it —
     otherwise leave unconstrained and say so in your report)
2. Train on the SAME folds, features, labels, thresholds logic as the
   incumbent. One variable changed. A/B, not a rewrite.
3. Compare on the artifact holdout with the SAME gate
   (`HOLDOUT_PF_CONFIDENCE = 0.95`, Clopper-Pearson PF lower bound):
   - gate metrics: Brier, EV, pooled/fold3 PF lower bounds
   - inversion metric: run `scripts/run_decision_report.py`-style grading on
     the CANDIDATE model's OOS scores — does the 0.40+ band's realised win
     rate rise toward the mid-band 26-32%? (Baseline 0.40+ = 11.8% win, n=34,
     fresh through 2026-09-08; see logs/decision_report_2026-09-08.txt)

## Promotion bar (all three required)

1. CatBoost strictly beats incumbent Brier AND EV AND CP-PF lower bound on
   identical folds — a tie keeps LightGBM, no churn without gain.
2. Top-band realised win rate improves (inversion narrows), or is unchanged
   while nothing else degrades.
3. Full test suite green; `python -m compileall -q src/` clean.

## Deliverable

A recon note in `llm_reports/recons/` (dated, per llm_reports/README
convention): numbers, fold-by-fold, with raw command output quoted — not
prose claims. If the gate fails, that is a healthy outcome: exit with the
note saying so, prior weights stay.