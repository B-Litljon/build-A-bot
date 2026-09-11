# 2026-09-08 — Stage 2: CatBoost ordered-trees A/B vs LightGBM — incumbent stands

**Question:** Does swapping Angel/Devil from LightGBM to CatBoost
(`boosting_type="Ordered"`, monotone `cost_ratio`), on identical folds and
labels, clear the promotion gate *and* narrow the calibration inversion?

**Answer, up front:** No promotion. CatBoost is better on the calibration
metrics (mean Brier, both CP-PF lower bounds) but the 0.40+ band did not
improve — and on this window CatBoost's calibrated Angel bar collapses to
0.314, which is a change in the comparison, not just the estimator. With the
fold gate failing for **both** arms at 60 days, this window cannot certify
either model; the honest read is "one variable moved the metrics, the window
is too thin to say more."

## What changed (one variable)

`MODEL_FAMILY` env seam in `src/core/retrainer.py`, plus
`scripts/run_catboost_ab.py`. `make_classifier(params, feature_cols)` returns
the family estimator; all four `LGBMClassifier(**params)` construction sites
in `refit_models` now route through it. The CatBoost arm translates the LGBM
param dict (iterations/depth/min_data_in_leaf/rsm/seed) and adds
`boosting_type="Ordered"` and `monotone_constraints={"cost_ratio": -1}`.
`subsample` is dropped in translation (invalid under Ordered / Bayesian
bootstrap); the drop is logged by name. Step-4 training-accuracy telemetry
guards off CatBoost's weightless `score()`. LightGBM arm is bit-identical to
pre-experiment. `catboost==1.2.10` added to the Pipfile venv.

## Evidence (raw)

Command: `RETRAIN_TIMEFRAME_MINUTES=15 pipenv run python
scripts/run_catboost_ab.py` (60 days, M15, 8-symbol basket, OANDA practice;
per-symbol parquet cache under `data/cache/ab_catboost/`). Full transcript:
`logs/ab_catboost_run.log`. Row-level OOS: `logs/ab_ledger_{family}.parquet`.

Holdout carved first, unused by either arm: 2026-08-28 → 2026-09-08, 5,148
rows; remainder 26,751; engineered 19,526 after the boundary purge.

| | LightGBM (incumbent) | CatBoost (candidate) |
|---|---|---|
| gate_passed | **False** | **False** |
| mean Brier | 0.2927 | **0.2837** ✓ |
| mean EV | +0.833333 | +0.824031 ✗ (tie-ish) |
| pooled PF 95% lb | 0.1335 (6/43) | **0.2227 (10/56)** ✓ |
| fold-3 PF 95% lb | 0.0345 | **0.5722** ✓ |
| Angel bar (OOF-calibrated) | 0.4099 | **0.3137** ⚠ |
| fold1 brier / trades | 0.2249 / 36 | 0.1797 / 43 |
| fold2 brier / trades | 0.1380 / 4 | 0.4401 / 3 |
| fold3 brier / trades | 0.5151 / 3 | **0.2312 / 10** |

Inversion bands (pooled OOS, arm's own proposals):

| band | LGBM win (n) | CB win (n) |
|---|---|---|
| 0.20–0.30 | — (0) | 0.133 (15) |
| 0.30–0.35 | — (0) | 0.226 (31) |
| 0.35–0.40 | 0.111 (18) | 0.143 (7) |
| 0.40–0.50 | 0.258 (31) | **0.000 (3)** ⚠ |
| ≥ arm's bar (floored at 0.40) | 0.267 (n=30, bar 0.41) | 0.000 (n=3, bar 0.3137 → floored to 0.40) |

## Reading

Promotion bar 1 (strict win on Brier AND EV AND both CP-PF bounds) **fails**:
CatBoost wins Brier and both PF bounds, but EV does not improve and fold-2
brier collapses (0.4401). The gate rejected both arms anyway (43 and 56
pooled trades vs the evidential floor; fold-3 CP bounds 0.03 and 0.57 vs the
1.20 bar) — the 60-day window is too thin to certify either, which is itself
the finding to carry forward: **at 60 days the fold gate cannot separate
these families; a decision-grade A/B needs the longer window.**

The inversion metric is inconclusive, not improved: CatBoost's 0.40+ bucket
is 0/3 because its own calibrated Angel bar slid to 0.3137 (vs LGBM's 0.41),
so almost nothing survives at 0.40+. That bar-drift is the load-bearing
caveat — comparing "0.40+ realised win rate" across arms is only meaningful
when both arms still calibrate near the live bar. They did not.

## Two design choices worth knowing

1. **Monotone constraints intersected with the present features.** The brief
   assumes `cost_ratio` exists; it does only when `RETRAIN_SPREAD_TABLE` is
   set. The translator now intersects the constraint dict with the actual
   feature columns — CatBoost hard-errors on a constraint naming an absent
   column (`Unknown feature name: cost_ratio`). On this run (no spread table),
   the CatBoost arm is Ordered boosting with **no** active constraint.
2. **Why the audit direction flipped since the stage-1 barrier work.** For the
   classifier, constraining acceptance probability on `natr_14`/
   `vol_rel` was deliberately left off per the brief's "only if evidence
   supports it." The 2026-08 finding says high-vol entries stop out more
   under a static multiple — evidence about *bracket width*, not about
   *whether to accept*. Nothing learned here changes that.

## Recommended next step

Re-run the same driver at 730 days (`CB_DAYS_BACK=730 RETRAIN_TIMEFRAME_MINUTES=15`),
matching the decision-report population where the inversion was measured
(baseline 0.40+ = 11.8%, n=34, `logs/decision_report_2026-09-08.txt`). The
60d fold gate can't certify either family, and the Angel-bar drift means the
inversion comparison was never actually run. 730d fetch is already staged in
`analysis_cache/strategy_matrix/` schema (6 pairs; metals would need one
slow fetch). Do **not** interpret the 0.2837 brier as a win — fold 2 (0.44)
says the CatBoost fold stability across regimes is not yet demonstrated even
where it looks good pooled.

— z
