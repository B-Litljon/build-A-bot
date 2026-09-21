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

#### Learned barrier geometry (added 2026-09-14)

An optional sidecar that replaces the profile's static `2.0×`/`4.0×` bracket
with per-bar quantiles from [`ml.barriers`](../../ml/barriers/): the stop is
`Q_MAE(0.95)` and the target `Q_MFE(0.50)`, both in NATR multiples. Three
things about it matter operationally:

- **OFF by default** — `BARRIER_GEOMETRY_ENABLED=1` (or
  `use_barriers=True`). Off, this class behaves exactly as it did before the
  sidecar existed, which is what makes it safe for a running soak to restart
  onto a tree carrying this code.
- **Fail-loud when enabled.** `_load_barriers()` raises at boot if the three
  artifacts (`barriers_mae.pkl`, `barriers_mfe.pkl`, `barriers_meta.json`) are
  absent or inconsistent, if the meta's label horizon disagrees with
  `ml.barriers.labels.DEFAULT_HORIZON`, or if the barrier model reads a feature
  the served schema does not produce. An operator who asked for learned stops
  and silently got static ones has no way to tell — the same trap the
  `cost_ratio`/HMM guards close.
- **The reload trigger is the META's mtime alone.** `BarrierEstimator.save()`
  writes both pickles first and the meta last, so a bar that sees a new meta
  reads a complete matching pair; a pickle replaced on its own stays invisible.
  A failed reload KEEPS the previously loaded geometry and alerts rather than
  silently disabling it.
- **A recorded FAILED promotion verdict is a refusal.** The artifact carries the
  Phase 1 gate's result in `barriers_meta.json` (written by the retrainer from
  `scripts/evaluate_barriers.py --verdict-out`, i.e. `BARRIER_VERDICT_OUT`), and
  the loader raises on it — because the retrainer's barrier hook fires off the
  **Angel/Devil** gate, not off the barrier gate, so artifacts already exist
  whose barrier gate failed. An artifact with **no** recorded verdict is served
  with a warning: absence is "unknown", which is the state of every artifact
  written before the field existed.

The payload travels in `Signal.metadata[BARRIER_GEOMETRY_KEY]` as NATR
multiples, i.e. `RiskManager` multiplies them by the same raw ATR the static
multipliers use — the learned quantiles substitute for the constants instead of
compounding with them, so rounding, the gates and sizing stay on one path. The
estimator's `rr`/`admissible` travel as telemetry only: its `rr_floor` compares
`Q_MFE(0.50)` with `Q_MAE(0.95)`, a ratio structurally below 1, so enforcing it
live would veto every bar (measured rr 0.28–0.30 across all three evaluation
folds on 2026-09-14).

- **Imports from repo:** `strategies.base`, `core.notification_manager`,
  `ml.barriers.estimator`, `ml.barriers.labels`, `ml.feature_pipeline`,
  `ml.features.v3_features`, `ml.regimes.hmm_regime`,
  `ml.trainers.v3_rf_trainer`.
- **Imported by:** `__init__.py`, `ml_factory_strategy.py`,
  `src/execution/oanda_forex_orchestrator.py`, `run_oanda.py`, and tests.
- **Reads:** `models/<asset_class>/` — `angel_latest.pkl`, `devil_latest.pkl`,
  `threshold.json`, `metadata.json`, `spread_alphas.json`, the
  `barriers_*.{pkl,json}` set when the barrier sidecar is on, and
  `hmm_latest.pkl` when regime features are on. **Writes:** nothing.

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

### The non-ML strategy library

Five ordinary rule-based strategies, added 2026-09-02. Each is a small
`BaseStrategy` subclass: bars in, a `Signal` or `None` out, no broker, no
sizing. They exist to be *compared against each other per market regime*, which
is what the regime router routes over.

> ⚠️ **None of them has a measured edge.** Scored on GBP_JPY M15 through
> `analysis.strategy_backtester`, all five returned negative net expectancy
> (profit factor 0.69–0.81) once timeouts were booked honestly. They are
> research inputs, not candidates for live trading. `run_oanda.py` logs a
> warning if you select one.

| File | Class | Entry rule |
|---|---|---|
| `sma_crossover.py` | `SMACrossoverStrategy` | fast SMA crosses slow SMA |
| `rsi_mean_reversion.py` | `RSIMeanReversionStrategy` | RSI leaves an extreme |
| `bollinger_breakout.py` | `BollingerBreakoutStrategy` | close breaks a band |
| `donchian_breakout.py` | `DonchianBreakoutStrategy` | close breaks an N-bar extreme |
| `momentum.py` | `MomentumStrategy` | sign flip of an N-bar return |

Each emits **both directions**, unlike `MLStrategy` which only ever goes long.
All size `raw_sl_distance` as raw ATR, matching `MLStrategy`, because the
`RiskManager` — not the strategy — owns the bracket multipliers.

- **Imports from repo:** `strategies.base`; TA-Lib for indicators.
- **Imported by:** `__init__.py`, `regime_router.py`, `src/analysis/` tooling,
  and tests. **Reads/writes:** nothing.

### `regime_router.py`
`RegimeRouterStrategy` — the "master" strategy. Each bar it tags the current
market behaviour with `ml.regimes.behavior_tagger.tag_bar` (causal, trailing
window only), looks the label up in a **routing table**, and delegates to the
named sub-strategy — or returns `None`.

Three ways it stands down, all deliberate:

1. the trailing window is **cold** (too few samples) — absence of evidence,
2. the label maps to **`null`** — no strategy has an edge in this regime,
3. the named strategy is **unknown** — a typo must not silently trade.

> ⚠️ **The routing tables in `config/` are hand-authored templates, not
> measurements** (`*.example.json`). They were written before any matrix run
> existed and every assignment in them is a textbook prior. `run_oanda.py`
> therefore has **no default routing config** and refuses to start
> `--strategy regime_router` without an explicit one. Generate a real table
> with `src/analysis/build_strategy_matrix.py` first.

Sub-strategies are registered under **one canonical snake_case key each**. A
`ClassName` spelling in a table still resolves, via `_resolve_name`, but is not
a second registry entry — two names for one object invites config drift.

- **Imports from repo:** `strategies.base`, `ml.regimes.behavior_tagger`, and
  the five library strategies (lazily, inside `_default_strategy_library`).
- **Imported by:** `__init__.py`, `run_oanda.py`, tests.
- **Reads:** a routing table JSON. **Writes:** nothing.

### `__init__.py`
Defines `STRATEGIES`, the name→class registry that `run_oanda.py --strategy`
selects from, plus `build_strategy(name, **params)`. `MLFactoryStrategy` is
intentionally absent.

**The registry resolves lazily.** `STRATEGIES` is a `Mapping`, not a dict:
iterating names and calling `keys()` import nothing, while a lookup imports that
one module and caches it. `MLStrategy` is the sole eager import, because it is
what the soak serves and every live module imports it directly anyway.

This matters for a specific failure: `run_oanda.py` reads this registry to build
its `--strategy` choices. When all seven were imported eagerly, a typo in an
unused research strategy would stop the live bot booting, and the watchdog's
crash-loop brake would then hold it down for 15 minutes. Now the live path loads
`ml_strategy` and nothing else — pinned by
`tests/test_strategy_library.py::test_selecting_one_strategy_does_not_import_the_others`.

A PEP 562 `__getattr__` keeps `from strategies.concrete_strategies import
MomentumStrategy` working for tests and analysis tooling, importing only that
module.
