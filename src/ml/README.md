# `src/ml` — the ML factory

Everything that turns bars into numbers a model can learn from, plus the
plumbing to fit and inspect models. This folder does **not** decide trades and
does not talk to a broker: it produces feature columns, and the strategy and
retrainer layers consume them.

The one rule that governs this whole folder: **training and live inference run
the same generators, in the same order, with the same cleaning.** A feature
computed even slightly differently in the two places produces a model that
tests well and loses money, with no error message. Everything here is arranged
to make that symmetry hard to break.

Assembly looks like this — callers hand `FeaturePipeline` a list, so adding a
feature family means writing one class rather than editing the pipeline:

```
bars → [V3BaseFeatures, V3SessionFeatures, V3CostFeatures?, V3HTFFeatures]
     → (target generator, training only) → clean_data → model-ready frame
```

See the root [GLOSSARY.md](../../GLOSSARY.md) for domain terms (angel/devil,
NATR, HTF, lookahead, feature, drift, PSI).

## Files

### `features/v3_features.py` — **the vocabulary of the system**
The four generators that produce every column the models see. If you only read
one file here, read this one; its glossary defines each feature in plain terms.

- `V3BaseFeatures` — 14 single-bar columns (momentum, volatility, position in
  range, candle shape). Computed **per symbol** so one instrument's history
  never leaks into another's indicators.
- `V3SessionFeatures` — four 0/1 flags for Asia / London / New York / overlap,
  from the UTC hour. Added because identical indicator values mean different
  things at 03:00 than at 14:00.
- `V3CostFeatures` — adds `cost_ratio`, letting the model *see* trading cost
  rather than having expensive setups silently filtered out behind it. A no-op
  unless a cost table is supplied, so older models are unaffected.
- `V3HTFFeatures` — what the slower (5-minute) chart says, joined onto each
  fast bar. **Contains the lookahead guard** (`available_at`): a 5-minute bar
  stamped 12:00 isn't finished until 12:05, so its timestamp is pushed forward
  before joining. Getting this wrong makes backtests look brilliant and live
  trading fail.

- **Imports from repo:** `ml.core.interfaces`.
- **Imported by:** `src/core/retrainer.py`, `src/execution/live_orchestrator.py`,
  `src/strategies/concrete_strategies/ml_strategy.py`, `src/ml/feature_pipeline.py`,
  `src/replay_test.py`, `src/analysis/failure_modes.py`,
  `src/analysis/optimize_brackets.py`, `scripts/probe_model.py`,
  `tests/test_cost_feature.py`.
- **Data artifacts:** none — pure DataFrame in, DataFrame out.

### `feature_pipeline.py`
`FeaturePipeline` — runs an ordered list of generators, then `clean_data`.
Order matters: later generators depend on columns earlier ones added.
`clean_data` converts both NaN **and infinity** to null before dropping
incomplete rows; infinity is the non-obvious half, arising when a perfectly
flat bar makes a normalising denominator zero.

- **Imports from repo:** `ml.core.interfaces`, `ml.features.v3_features`,
  `ml.targets.v3_targets`.
- **Imported by:** the same broad set as `v3_features.py` above.
- **Reads/writes** (only in `main()`, not the production path):
  `data/raw/*_1min.parquet` → `data/processed/training_data.parquet`.

### `feature_stats.py` — the "is it broken or just quiet?" tool
Computes feature distributions at training time and compares live
distributions against them later, **without retraining anything**.

The load-bearing idea is **null calibration**. Textbook PSI cutoffs
(0.10 / 0.25) assume independent samples; market bars are heavily
autocorrelated, so any short window scores high PSI even when nothing is wrong.
This module measures what PSI *ordinary training windows* produce and only
calls drift when live PSI beats that null's upper tail. Stats are stored both
pooled and per-symbol, because instruments sit at genuinely different baseline
levels.

- **Imports from repo:** none.
- **Imported by:** `src/core/retrainer.py` (writes the sidecar on promotion),
  `scripts/probe_model.py` (reads it), `scripts/generate_feature_stats.py`
  (backfills it), `tests/test_feature_stats.py`.
- **Writes:** `feature_stats.json` inside the model directory.

### `data_miner.py`
`DataMiner` — bulk history fetcher. Pulls years of bars a month at a time with
retries and a polite inter-request delay, caching to disk so experiments re-read
local Parquet instead of hammering the vendor. Goes through the provider
factory, so it works with any configured `DATA_SOURCE`.

- **Imports from repo:** `data.factory`, `data.market_provider`.
- **Imported by:** nothing — run as `python -m ml.data_miner`.
- **Writes:** `data/raw/<SYMBOL>_1min.parquet`.

### `train_model.py` — ⚠️ legacy
The original two-stage trainer: RandomForest, no validation gate, saving to
`src/ml/models/`. Superseded by `src/core/retrainer.py` (LightGBM + promotion
gate + `models/<asset_class>/`). Still the clearest plain statement of the
Angel/Devil idea, which is why it's worth keeping — but don't use it to produce
a model you intend to trade.

- **Imports from repo:** `ml.core.interfaces`, `ml.trainers.v3_rf_trainer`.
- **Imported by:** `src/replay_test.py`.
- **Reads/writes:** `data/processed/training_data.parquet` → `src/ml/models/`.

### `__init__.py`
Empty package marker.

## Subpackages

### `core/` — `interfaces.py`
The three plug-in contracts everything else implements: `BaseFeatureGenerator`,
`BaseTargetGenerator`, `BaseTrainer`. Tiny and dependency-free by design.
Note `predict_proba` returning probabilities (not hard labels) is what makes
the tunable decision thresholds possible.
- **Imported by:** `feature_pipeline.py`, `features/v3_features.py`,
  `targets/v3_targets.py`, `trainers/v3_rf_trainer.py`, `train_model.py`,
  `src/day_trading/features.py`, `src/day_trading/targets.py`.

### `features/` — `v3_features.py`
See above. (No `__init__.py`; implicit namespace package.)

### `targets/` — `v3_targets.py` ⚠️ legacy
`V3DirectionalTarget` labels a bar 1 if price rises ≥0.3% within 15 bars. Only
`feature_pipeline.main()` uses it. The production labels in `retrainer.py`
are harsher and more realistic: they replay stops and targets bar by bar, so a
setup that would have been stopped out first counts as a loss — this one only
asks whether price *ever* reached the level.

### `trainers/` — `v3_rf_trainer.py` ⚠️ misleading name
`V3RandomForestTrainer` is now mostly a **load-and-predict shell**. Production
models are LightGBM (switched 2026-05-23), and `.load()` simply unpickles
whatever estimator is on disk — so the live strategy constructs a
`V3RandomForestTrainer` at `ml_strategy.py:124` and immediately loads a
LightGBM model into it. Only `train_model.py` still trains an actual forest
through it.
- **Imported by:** `train_model.py`, `src/replay_test.py`,
  `src/strategies/concrete_strategies/ml_strategy.py`,
  `src/strategies/concrete_strategies/ml_factory_strategy.py`.

### `regimes/` — `hmm_regime.py` (experimental, off by default)
Fits a per-symbol 3-state model on (return, volatility) and hands the
classifier its confidence in each hidden market mode as three extra features.
Enabled only with `RETRAIN_USE_HMM=1`. States are **not named or interpreted** —
"state 0" has no fixed meaning across symbols or runs. Symbols with too little
data get a uniform 1/3, a deliberately uninformative value rather than a gap.
Leakage rule: fit on training rows only, then score both training and
validation with that fitted model.
- **Imported by:** `src/core/retrainer.py`,
  `src/strategies/concrete_strategies/ml_strategy.py`.
- **Writes:** a joblib dict saved next to the Angel/Devil models.

### `barriers/` — `labels.py` + `estimator.py` (learned bracket geometry)
MAE/MFE excursion labels and the two quantile regressions that turn them into
per-bar stop/target distances — the replacement for the static 2.0×/4.0× ATR
bracket. `labels.compute_excursions` computes each label per symbol and returns
null for any row whose forward window is incomplete (including windows that
contain a gap bar). `BarrierEstimator` fits `Q_MAE(0.95)` for the stop and
`Q_MFE(0.50)` for the target, CatBoost-first because its quantile loss accepts
`monotone_constraints` (LightGBM's rejects them, so that path is
audit-enforced instead). `save`/`load` are the live sidecar contract:
`barriers_mae.pkl` + `barriers_mfe.pkl` + `barriers_meta.json`, meta written
last, label horizon declared.

**Serving it live is gated twice.** `BARRIER_GEOMETRY_ENABLED` must be set on
the strategy side, and the artifact must not record a FAILED promotion verdict:
`scripts/evaluate_barriers.py` prints a `VERDICT_JSON` line (and writes it to
`BARRIER_VERDICT_OUT`), `BarrierEstimator.save(verdict=...)` embeds it in
`barriers_meta.json`, and the live loader refuses anything that is not PASS. The
gate currently FAILS — fold 3 MAE coverage 0.905 against a 0.93 floor,
re-measured 2026-09-14 — so a serving artifact does not exist yet. See the
package README for the verdict table.
- **Imported by:** `scripts/evaluate_barriers.py`,
  `src/strategies/concrete_strategies/ml_strategy.py` (behind the switch),
  tests.
- **Writes:** `models/<dir>/barriers_{mae,mfe}.pkl` + `barriers_meta.json` via
  `save()`. Nothing in the retrainer writes them yet.
