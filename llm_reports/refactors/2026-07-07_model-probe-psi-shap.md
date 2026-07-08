---
type: refactor
date: 2026-07-07
time: 20:20 PDT
agent: Claude Fable 5
model: claude-fable-5
trigger: Brandon asked for zero-retraining feature-vetting tooling (prompted by a DeepSeek conversation); built the tree-model version and probed the live soak model
head: a4d5b0736e639de37073a4c81a53a5824a13b970
scope: modifies-source
related:
  - refactors/2026-07-07_cost-awareness-spread-feature.md
files_touched:
  - src/ml/feature_stats.py
  - src/core/retrainer.py
  - scripts/generate_feature_stats.py
  - scripts/probe_model.py
  - tests/test_feature_stats.py
---

## Context

Question: can we vet whether the frozen model's quietness is input drift
(features outside the training distribution → mathematical unfamiliarity) or
honest judgment (regime seen fine, no edge found) WITHOUT retraining?
DeepSeek suggested 4 neural-net probes; two don't apply to LightGBM
(gradient×input — trees have no input gradients; activation-saturation —
no activations). The transferable ideas: PSI distribution-shift testing and
per-prediction attribution — for trees the exact tool is TreeSHAP
(`booster_.predict(pred_contrib=True)`), which subsumes crude
mean-substitution ablation. DeepSeek's "quick fix" (rolling z-score rescaling
of live inputs) was REJECTED: it breaks train/live symmetry and manufactures
confidence in regimes where "no trade" is correct.

## Investigation

- Model pkls are bare sklearn/LightGBM estimators (joblib), `feature_names_in_`
  exposed; `booster_.predict(pred_contrib=True)` returns (n, n_feat+1) exact
  log-odds contributions.
- Training reference distribution: computed from the POST-VETO, post-clean
  frame (exactly the population the models saw). Backfill for pre-existing
  model dirs rebuilds that frame via the retrainer's own fetch/engineer
  functions with RETRAIN_END_DATE.
- **Key methodological finding:** raw PSI with textbook thresholds
  (0.10/0.25) false-alarms catastrophically on market data. First probe run
  flagged 8/8 instruments SEVERE on nearly every feature — yet a spot-check
  showed XAU natr_14 live (~0.207) at the ~70th percentile of training,
  comfortably in-distribution. Cause: 100-bar live windows of heavily
  autocorrelated series concentrate in a narrow band of the 2-year training
  mixture; PSI punishes concentration as drift. Textbook cutoffs assume iid
  samples (credit-scoring populations).
- Fix: **null calibration.** The sidecar stores, per symbol × feature, the
  quantiles (p50/p90/p95/p99) of PSI produced by 200 random contiguous
  SAME-LENGTH training windows against the full training reference. Live PSI
  only counts as drift when it beats the null p99 (SEVERE) / p95 (moderate).
  Regression test reproduces the failure mode: an ordinary window of an
  AR(0.98) series has raw PSI > 0.25 ("SEVERE" by textbook) but calibrated
  → stable; a genuinely off-range regime still flags SEVERE.

## Findings / Changes

- **src/ml/feature_stats.py (new):** shared stats/PSI module —
  `compute_feature_stats` (pooled + per-symbol deciles, categorical bins for
  low-cardinality features, null-PSI quantiles), atomic
  `save/load_feature_stats`, `psi`, `psi_report`, `classify_psi_calibrated`,
  `drift_flags`.
- **retrainer.py:** on promotion, saves `feature_stats.json` sidecar next to
  the pkls (same pattern as threshold.json / spread_alphas.json).
- **scripts/generate_feature_stats.py (new):** backfill for model dirs
  trained before the sidecar existed; guards that the rebuilt feature schema
  matches the model's `feature_names_in_`, records window provenance.
- **scripts/probe_model.py (new):** read-only probe — fetches recent bars,
  builds features via the SAME FeaturePipeline construction as MLStrategy,
  reports per instrument: calibrated PSI drift, top-5 SHAP suppressors /
  top-3 supporters of angel_prob, and a DRIFT vs HONEST verdict
  (drifted ∩ suppressors → DRIFT).

**Probe verdict on the live soak model (models/forex_m15, backfilled stats
for its 730d window ending 2026-07-02, live window 100 bars 2026-07-07):**
- **0 DRIFT / 8 HONEST.** No feature on any instrument beats its null p99;
  only scattered moderates hugging their nulls.
- Suppressors are consistent across the basket: **ppo** (momentum),
  **htf_bb_pct_b**, **vol_rel**, **log_return** — no directional momentum +
  weak relative volume → no 3-bar breakout setup → angel_prob correctly low.
- **natr_14 is the strongest SUPPORTER nearly everywhere** (+0.03..+0.09
  log-odds): current volatility levels are fine; it is the *lack of
  momentum*, not low vol, keeping the model quiet.
- Answer to the month-long question: the model is not blind, stale, or
  drifted — the market since 2026-07-02 genuinely hasn't offered its setup.
  Coheres with the same-day finding that a no-cost control retrain fails
  Brier on the newer window (the recent regime is hard to calibrate on).

## Verification

- tests/test_feature_stats.py: 12 tests — stats shapes/kinds, save/load
  roundtrip, PSI stability on same-distribution draws, severe on collapsed
  vol, categorical unseen-value, pooled-vs-per-symbol reference necessity,
  null-quantile monotonicity, false-alarm reproduction (raw SEVERE →
  calibrated stable), foreign-regime still flags, drift_flags wiring. All
  pass; full suite 121 passed.
- Backfill schema guard exercised against models/forex_m15 (22 features
  matched). Probe run end-to-end against live OANDA practice data.

## Risk & follow-ups

1. Sidecar reference is the POST-VETO population (deliberate: it is what the
   model trained on). A pre-veto variant could separately diagnose "market
   entered the vetoed regime" — future option.
2. Null window is 100 bars; probe warns when --bars differs from the
   sidecar's null_window.
3. Devil model not probed (Angel is the bottleneck at threshold 0.40).
4. Natural autopilot-rails slot later: probe as a pre-retrain gate
   ("features drifted → retrain" vs "honest quiet → wait").
5. Uncommitted at time of writing.

## Files touched

- `src/ml/feature_stats.py` (new, ~300 lines)
- `src/core/retrainer.py` (import + sidecar save in the promoted block)
- `scripts/generate_feature_stats.py` (new, ~120 lines)
- `scripts/probe_model.py` (new, ~260 lines)
- `tests/test_feature_stats.py` (new, 12 tests)
