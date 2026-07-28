# `src/ml/core`

The plug-in contracts the whole ML factory is built from. Everything else in
`src/ml` implements one of these three abstract classes, which is what lets a
pipeline be *configured* as a list rather than hard-coded.

Deliberately tiny and dependency-free (abc + polars only) so it stays
importable from anywhere without pulling in vendor SDKs or model libraries.

No `__init__.py` — implicit namespace package.

## Files

### `interfaces.py`
Three contracts:

- **`BaseFeatureGenerator`** — DataFrame in, DataFrame out with new columns
  *appended*. Must not drop or reorder rows. Order in a pipeline matters,
  because later generators read columns earlier ones added.
- **`BaseTargetGenerator`** — same shape, but adds the answer column used for
  training. Absent at inference time, where there is no answer.
- **`BaseTrainer`** — `train` / `predict_proba` / `save` / `load`.
  `predict_proba` returning **probabilities rather than hard labels** is
  load-bearing: every tunable decision threshold in the system operates on
  those continuous scores.

- **Imports from repo:** none.
- **Imported by:** `src/ml/feature_pipeline.py`,
  `src/ml/features/v3_features.py`, `src/ml/targets/v3_targets.py`,
  `src/ml/trainers/v3_rf_trainer.py`, `src/ml/train_model.py`,
  `src/day_trading/features.py`, `src/day_trading/targets.py`.
- **Data artifacts:** none.
