# STAGE 3 — Beta calibration + IVAP veto layer

You are implementing Stage 3 of the quant-architecture roadmap. Stages 1
(barriers, `src/ml/barriers/`) and 2 (CatBoost swap, if it promoted — CHECK
`llm_reports/recons/` for the stage-2 recon note first; if stage 2 failed its
gate, calibrate the INCUMBENT LightGBM instead and say so in your report).

## Hard rails

Same standing rails as every stage in this repo (see
`llm_reports/m2m-prompts/2026-09-08_stage-02-catboost-swap.md` "Hard rails"
section — they apply verbatim to you). Additionally:

- Calibration must be fit on OOF probabilities ONLY (never in-sample). The
  repo has been burned by in-sample calibration flattering itself.
- `u_star` (the IVAP veto bar) is tuned per retrain and saved into
  `threshold.json` — NOT env-tunable at runtime. The promotion gate is not a
  knob.

## What to build

1. `src/ml/calibration/beta_calibrator.py`
   - Beta calibration: p_cal = 1 / (1 + 1/(e^c * s^a * (1-s)^b)), (a,b,c)
     fitted by NLL on OOF scores. Per direction (long/short paths use
     different score populations once stage 4 splits them; for now, per
     Angel/Devil stage).
   - Artifact: `models/<dir>/calibration.json` {a,b,c per stage}, atomic
     write (temp + rename), travels with the model dir like threshold.json.
2. `src/ml/calibration/ivap.py`
   - Inductive Venn-Abers over the CALIBRATED Devil. Fixed k=10 bins fit on
     OOF; per-bar (p0, p1) from the standard IVAP construction.
   - Epistemic metric u = p1 - p0. Veto when u > u_star OR p1 < devil
     threshold (the second condition is load-bearing: an interval wholly
     below the bar means even the optimistic read concedes).
3. Wire into `MLStrategy` between base score and threshold, stateless.
   Telemetry: emit the veto as its own gate code so a bad veto is
   attributable (mirror `RiskManager.last_veto_gate` naming).
4. TRAINING-SIDE TWIN: extend `retrainer._compute_chop_veto_mask` with the
   same IVAP veto in the same change, or the model trains on bars live
   would never take. This is the repo's #1 invariant; do not skip it.

## Promotion bar

- Spiegelhalter Z-statistic on the holdout: p > 0.05 (no evidence of
  miscalibration), and Brier reliability term lower than pre-calibration on
  the same holdout.
- Every trade the IVAP veto removes must have been net-negative in
  EXPECTATION on the holdout (a veto that removes winners fails the stage).
- Train/live symmetry test updated; full suite green.

## Deliverable

Recon note in `llm_reports/recons/` with raw outputs. Gate failure = healthy
failure; report it as such.