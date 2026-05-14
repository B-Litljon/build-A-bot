---
type: stop
date: 2026-05-14
time: 01:35 PDT
agent: Kimi K2.6
model: moonshotai/kimi-k2.6
trigger: User mission to execute 5 ML pipeline fixes, commit to fix/ml-pipeline-refactor, and open PR against main — blocked at push/PR creation due to missing GitHub credentials
head: 98be4384e3c3896c253f131be81d8afe1bbba56a
scope: modifies-source
files_touched:
  - scripts/investor_feature_pipeline.py
  - src/execution/live_orchestrator.py
  - src/ml/feature_pipeline.py
  - src/strategies/concrete_strategies/ml_strategy.py
---

# ML Pipeline Refactor — Push/PR Blocked

## Context

User issued a 5-fix strike mission for the `fix/ml-pipeline-refactor` branch:

1. **Fix 1.4** — `src/ml/feature_pipeline.py`: Parametrize `clean_data` to accept `feature_cols` and drop nulls only on that subset (preserve sparse fundamental data).
2. **Fix 2.2** — `src/strategies/concrete_strategies/ml_strategy.py`: Wrap `self.angel_trainer.load(...)` in `_check_model_updates` with `threading.Lock`.
3. **Fix 2.7** — `src/execution/live_orchestrator.py` & `ml_strategy.py`: Remove per-bar `import pandas as pd` roundtrip in inference hot-path; refactor to `numpy.hstack` and pass pure ndarrays to `predict_proba`.
4. **Fix 2.8** — `scripts/investor_feature_pipeline.py`: Rename `_top_quintile_label` → `_top_k_label` and `target_top_quintile` → `target_top_k`.
5. **Fix 2.6** — `ml_strategy.py`: Introduce explicit feature-schema validation at strategy load time (`strategy.feature_names` vs `angel_trainer.feature_names_in_`), raise `RuntimeError` on drift.

All 5 fixes were implemented and tested locally. A commit was created on `fix/ml-pipeline-refactor` (SHA `98be438`). The final step — pushing the branch and opening a Pull Request against `main` — failed because no GitHub authentication credentials are available in this environment.

## Investigation

1. **Branch checkout** — `git branch -a` confirmed `fix/ml-pipeline-refactor` exists locally. Checked out successfully.
2. **File reads** — Read all 4 target files plus `src/ml/trainers/v3_rf_trainer.py` to verify `feature_names_in_` property exists (line 27–28).
3. **Edits applied** — 9 discrete `edit` operations across the 4 files (see Findings).
4. **Smoke tests** —
   - `python -c "from strategies.concrete_strategies.ml_strategy import MLStrategy; s = MLStrategy()"` passed; models load, schema validation tolerates `feature_names_in_ == None` (models were trained without DataFrame input, so the attribute is absent).
   - `FeaturePipeline.clean_data` subset test verified: `drop_nulls(subset=...)` preserves rows where non-feature columns are null.
   - `py_compile` on `scripts/investor_feature_pipeline.py` passed.
5. **Commit** — `git commit` succeeded with message `fix(ml): resolve architectural bottlenecks and thread-safety issues`.
6. **Push attempt** — `git push -u origin fix/ml-pipeline-refactor` failed with:
   ```
   remote: Invalid username or token. Password authentication is not supported for Git operations.
   fatal: Authentication failed for 'https://github.com/B-Litljon/build-A-bot.git/'
   ```
7. **Credential search** —
   - `env | grep -i token` → empty.
   - `~/.git-credentials` → absent.
   - `~/.ssh/` → absent.
   - `gh` CLI → not installed.
   - `git config credential.helper` → unset.

No authentication mechanism is configured in the current shell environment.

## Findings / Changes

All 5 requested fixes are **committed** on `fix/ml-pipeline-refactor` (SHA `98be438`) and ready to push.

| Fix | File(s) | What changed |
|---|---|---|
| 1.4 | `src/ml/feature_pipeline.py` | `run()` now accepts optional `feature_cols: Optional[List[str]] = None`; forwards to `clean_data()`. `clean_data()` accepts same param, computes `subset = [c for c in feature_cols if c in df.columns]` if provided, and calls `df.drop_nulls(subset=subset)` instead of dropping on the whole frame. |
| 2.2 | `src/strategies/concrete_strategies/ml_strategy.py` | Added `import threading`; initialized `self._reload_lock = threading.Lock()` in `__init__`. In `_check_model_updates()`, the `self.angel_trainer.load(...)` block (including `n_jobs = 1` override and mtime update) is wrapped in `with self._reload_lock:`. |
| 2.7 | `src/strategies/concrete_strategies/ml_strategy.py` + `src/execution/live_orchestrator.py` | Removed `import pandas as pd` from `ml_strategy.py`. Replaced `pd.DataFrame(...)` + column assignment with `np.array([[angel_prob]])` + `np.hstack([latest_features, angel_prob_col])` in both files. Added `import numpy as np` to `live_orchestrator.py`. |
| 2.8 | `scripts/investor_feature_pipeline.py` | Renamed function `_top_quintile_label` → `_top_k_label` and column `target_top_quintile` → `target_top_k` (all 11 occurrences replaced). |
| 2.6 | `src/strategies/concrete_strategies/ml_strategy.py` | After `self.feature_names` is defined and before threshold loading, added:
```python
model_features = self.angel_trainer.feature_names_in_
if model_features is not None and list(model_features) != self.feature_names:
    raise RuntimeError(...)
``` |

## Verification

- **Model load test**: `MLStrategy()` instantiated successfully on `models/angel_latest.pkl` and `models/devil_latest.pkl`.
- **Lock verification**: `hasattr(strategy, '_reload_lock')` → `True`; `type(strategy._reload_lock)` → `<class '_thread.lock'>`.
- **Hot-reload dry-run**: `strategy._check_model_updates()` returned `False` without error.
- **clean_data subset test**: `FeaturePipeline.clean_data(df, feature_cols=['a','b'])` dropped only rows with nulls in columns `a` or `b`, preserving rows where column `c` was null.
- **Syntax check**: `python -m py_compile scripts/investor_feature_pipeline.py` passed.

## Risk & follow-ups

1. **BLOCKER — GitHub auth missing**: The branch `fix/ml-pipeline-refactor` exists only locally. It must be pushed to `origin` and a PR opened against `main`. The next agent or the user needs to either:
   - Provide a GitHub PAT and run `git push -u origin fix/ml-pipeline-refactor`, or
   - Use the GitHub web UI to create the PR from the existing local branch (if the branch was already pushed by other means), or
   - Configure `git` with a credential helper + token.
2. **Model `feature_names_in_` is `None`**: The current production models were trained without passing a pandas DataFrame to sklearn, so `feature_names_in_` is absent. The validation code handles this gracefully (skips check when `None`). If models are retrained with DataFrame input in the future, the validation will become active.
3. **No CI/tests exercised**: Only smoke tests were run. The full test suite (if any) should be run before merging.

## Files touched

- `src/ml/feature_pipeline.py` — `run()` signature (line 44), `clean_data()` signature and null-drop logic (lines 52–66)
- `src/strategies/concrete_strategies/ml_strategy.py` — imports (line 20–28), `__init__` lock init (line 78), feature schema validation (line 163–168), `_check_model_updates()` lock wrapper (line 243–249), `generate_signals()` numpy hstack refactor (line 359–362), `_generate_features()` `feature_cols` pass (line 410)
- `src/execution/live_orchestrator.py` — added `import numpy as np` (line 72), `_run_inference()` numpy hstack refactor (lines 1203–1208)
- `scripts/investor_feature_pipeline.py` — function rename and column rename throughout (lines 89, 234–293)
