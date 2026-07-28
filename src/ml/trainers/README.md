# `src/ml/trainers`

`BaseTrainer` implementations — thin wrappers that give models a uniform
train / predict / save / load surface.

No `__init__.py` — implicit namespace package.

## Files

### `v3_rf_trainer.py` — ⚠️ the name no longer matches reality
`V3RandomForestTrainer` wraps scikit-learn's `RandomForestClassifier`, but in
practice it is now used as a **load-and-predict shell**:

- Production models have been **LightGBM since 2026-05-23** (that swap, not new
  features, is what cleared the validation gate).
- `load()` simply unpickles whatever estimator is on disk and assigns it to
  `self.model`, replacing the forest entirely.
- So `ml_strategy.py:124` constructs a `V3RandomForestTrainer()` and
  immediately `.load()`s a LightGBM model into it. A variable named for a
  random forest is holding a gradient-boosted model at runtime.

Only `train_model.py` (itself legacy) still trains an actual forest through it.

`feature_names_in_` exposes the column names the fitted model expects — worth
knowing about, because a mismatch between live and training features is a
silent-wrong-answer bug rather than a crash.

- **Imports from repo:** `ml.core.interfaces`.
- **Imported by:** `src/ml/train_model.py`, `src/replay_test.py`,
  `src/strategies/concrete_strategies/ml_strategy.py`,
  `src/strategies/concrete_strategies/ml_factory_strategy.py`.
- **Data artifacts:** reads/writes model pickles at paths its callers choose;
  no fixed location of its own.
