# `src/strategies`

The decision layer. A strategy takes a DataFrame of bars and returns a `Signal`
or `None` — nothing more. It holds no broker connection, places no orders, and
knows nothing about position sizing. An orchestrator in
[`src/execution`](../execution/) calls it and decides what to do with the answer.

`None` is the normal return. Over a typical soak week the strategy declines on
the overwhelming majority of bars.

See the root [GLOSSARY.md](../../GLOSSARY.md) for domain terms (angel/devil,
NATR, warm-up, hot reload, threshold).

## Files

### `base.py`
`BaseStrategy` (the `generate_signals` contract plus a shared input check) and
`Signal`.

> ⚠️ **There are two `Signal` classes.** This one is used by the OANDA/forex
> and Factory paths and carries explicit bracket *distances*. The other,
> [`src/core/signal.py`](../core/), belongs to the Alpaca path and keeps
> bracket levels inside a metadata dict.

Two things about this `Signal` worth knowing:

- `raw_sl_distance` is a **price distance**, not a level or a percentage. The
  `RiskManager` converts it into an actual stop price.
- `raw_tp_distance` is **written but never read.** `MLStrategy` sets it to the
  same value as `raw_sl_distance`, and no execution path consumes it — target
  sizing belongs to the `RiskManager`'s own multipliers, deliberately, so live
  brackets always match the ones the model was trained against. Vestigial
  field; verified by tracing every reference. (A learned barrier payload
  carries its own target, but it travels in `metadata`, not through this field.)

`base.py` also defines **`BARRIER_GEOMETRY_KEY`** (`"barrier_geometry"`), the
one `Signal.metadata` key that is a contract rather than a diagnostic. It
carries the learned bracket geometry — `sl_atr_mult` / `tp_atr_mult` as NATR
multiples, plus `rr`, `admissible`, `tau_mae`, `tau_mfe` and `backend` for
telemetry — which `RiskManager` substitutes for its static profile multipliers.
It lives here, beside the `Signal` that carries it, because three layers need
the same string and only one may own it: the producer
(`concrete_strategies/ml_strategy.py`), the consumer
(`execution/risk_manager.py`, which must stay numpy-only) and the offline replay
(`analysis/strategy_backtester.py`, which keeps execution imports out of module
scope).

Its `generate_signals` docstring still says "18-feature", which is **stale** —
the live set is 22 columns (23 with `cost_ratio`).

- **Imports from repo:** none. **Data artifacts:** none.
- **Imported by:** `concrete_strategies/ml_strategy.py`,
  `src/execution/oanda_forex_orchestrator.py`,
  `src/execution/factory_orchestrator.py`, `tests/test_oanda_forex.py`.

### `__init__.py`
Empty package marker.

## `concrete_strategies/`

### `ml_strategy.py` (694 lines) — **the brain**
`MLStrategy`, the two-stage Angel/Devil decision-maker used by both live bots.

**The design constraint that explains most of this file:** live inference must
match training exactly, or the model quietly scores garbage. So —

- The feature pipeline is **imported** from `src/ml`, never reimplemented, and
  its generator order (`V3BaseFeatures → V3HTFFeatures → V3SessionFeatures →
  V3CostFeatures`) is byte-identical to the retrainer's at `retrainer.py:878`.
- The feature list comes from the **model's own `feature_names_in_`**, not a
  hardcoded constant, so a retrain that changes the feature set propagates with
  no code edit.
- The Devil threshold is **overwritten** from the model's `threshold.json`,
  because the retrainer tunes that value per model.
- If the model was trained with `cost_ratio` but `spread_alphas.json` is
  missing (or with regime features but the HMM artifact is missing), the
  constructor **raises at boot** rather than failing obscurely on the first live
  bar.

Two runtime behaviours worth knowing:

- **Hot reload** — every bar it compares model file modification times, so when
  the retrainer atomically swaps in new pickles the live bot picks them up
  without a restart. Guarded by `_reload_lock`.
- **Stale-bar guard** — feature cleaning drops rows with missing values. If the
  *newest* bar is the one dropped, the frame's tail is an older bar, and scoring
  it against the current price would trade on the wrong bar. It returns `None`.
- **Heartbeat** — every 15 bars per symbol it logs a summary of recent Angel
  probabilities, so an operator can tell the model is alive during long
  no-trade stretches. (Individual rejections log at debug level and are
  normally invisible.) Tunable via `MLSTRATEGY_HEARTBEAT_EVERY_N`.
- **Learned barrier geometry (2026-09-14)** — when enabled
  (`BARRIER_GEOMETRY_ENABLED=1` or `use_barriers=True`), it loads the
  `ml.barriers` quantile sidecar from the model dir and attaches per-bar
  stop/target multiples to `Signal.metadata[BARRIER_GEOMETRY_KEY]`. **OFF by
  default**, and fail-loud when enabled without a usable artifact set. See
  `concrete_strategies/README.md` for the switch and the reload seam.

- **Imports from repo:** `strategies.base`, `core.notification_manager`,
  `ml.barriers.estimator`, `ml.barriers.labels`, `ml.feature_pipeline`,
  `ml.features.v3_features`, `ml.regimes.hmm_regime`,
  `ml.trainers.v3_rf_trainer`.
- **Imported by:** `concrete_strategies/__init__.py`,
  `ml_factory_strategy.py`, `src/execution/oanda_forex_orchestrator.py`,
  `run_oanda.py`, `tests/`.
- **Reads:** `models/<asset_class>/angel_latest.pkl`, `devil_latest.pkl`,
  `threshold.json`, `metadata.json`, `spread_alphas.json`,
  `barriers_mae.pkl` + `barriers_mfe.pkl` + `barriers_meta.json` when the
  barrier sidecar is enabled, and `hmm_latest.pkl` when regime features are
  enabled. **Writes:** nothing.

### `ml_factory_strategy.py`
`MLFactoryStrategy` — a thin subclass that presets `warmup_period=260` and
supplies two trainer shells. **No inference logic of its own.** A previous
version re-declared an identical feature pipeline; that was removed precisely
because a duplicate definition invites the two copies to drift apart.

Its docstring also still claims "18-feature" — stale, same as `base.py`.

- **Imports from repo:** `strategies.concrete_strategies.ml_strategy`,
  `ml.trainers.v3_rf_trainer`.
- **Imported by:** the Factory path. **Data artifacts:** inherited from
  `MLStrategy`.

### `__init__.py`
Exports `MLStrategy` and the `STRATEGIES` name→class registry. Note
`MLFactoryStrategy` is deliberately **not** registered; the Factory
orchestrator imports it directly.
