# `src/strategies/concrete_strategies`

The actual strategy implementations. See the parent
[`../README.md`](../README.md) for the layer's role and the two-`Signal`
warning; this file is the per-file detail.

## Files

### `ml_strategy.py` (694 lines)
`MLStrategy` — the two-stage Angel/Devil decision-maker used by both live bots.
Stage one proposes with high recall, stage two approves with high precision,
and a `Signal` is returned only when both agree.

Its governing constraint is that **live inference must match training exactly**.
Concretely that means: the feature pipeline is imported from `src/ml` rather
than reimplemented and its generator order matches `retrainer.py:878`
byte-for-byte; the feature list is read from the model's own
`feature_names_in_`; the Devil threshold is loaded from the model's
`threshold.json`; and the constructor raises at boot if the model expects
`cost_ratio` or regime features whose sidecar files are missing.

Runtime behaviours: **hot reload** (compares model file mtimes each bar, so a
retrain lands without a restart), the **stale-bar guard** (if feature cleaning
dropped the newest bar, return `None` rather than score an older bar against
the current price), and a **heartbeat** log every 15 bars so an operator can
see the model is alive during no-trade stretches.

- **Imports from repo:** `strategies.base`, `core.notification_manager`,
  `ml.feature_pipeline`, `ml.features.v3_features`, `ml.regimes.hmm_regime`,
  `ml.trainers.v3_rf_trainer`.
- **Imported by:** `__init__.py`, `ml_factory_strategy.py`,
  `src/execution/oanda_scalper_orchestrator.py`, `run_oanda.py`, and tests.
- **Reads:** `models/<asset_class>/` — `angel_latest.pkl`, `devil_latest.pkl`,
  `threshold.json`, `metadata.json`, `spread_alphas.json`, and `hmm_latest.pkl`
  when regime features are on. **Writes:** nothing.

### `ml_factory_strategy.py`
`MLFactoryStrategy` — a 33-line subclass of `MLStrategy` that presets
`warmup_period=260` (what a 50-period average on 5-minute bars needs) and
supplies two `V3RandomForestTrainer` shells. It adds **no inference logic**.

Note the comment in the file: an earlier version re-declared an identical
feature pipeline, and it was deliberately removed because a duplicate
definition invites the two copies to drift apart.

Its module docstring claims an "18-feature" input, which is **stale** — the
live set is 22 columns (23 with `cost_ratio`), and `MLStrategy` no longer
hardcodes a count at all.

- **Imports from repo:** `strategies.concrete_strategies.ml_strategy`,
  `ml.trainers.v3_rf_trainer`.
- **Imported by:** the Factory orchestrator path (imported directly, not via
  the registry). **Data artifacts:** whatever `MLStrategy` reads.

### `__init__.py`
Exports `MLStrategy` and defines `STRATEGIES`, a name→class registry for
selecting a strategy by config string. Only `"ml_strategy"` is registered;
`MLFactoryStrategy` is intentionally absent.
