# `src/ml/features`

The feature generators — **every number the models actually look at** is
defined here. This is the single most useful file in the repo for
understanding what the bot perceives.

Training and live inference run these same classes in the same order. That
symmetry is the point: a feature computed even slightly differently in the two
places yields a model that validates well and loses money, silently.

No `__init__.py` — implicit namespace package.

## Files

### `v3_features.py`
Four `BaseFeatureGenerator` implementations. The module docstring defines every
column in plain language; the short version:

| Class | Adds | Why it exists |
|---|---|---|
| `V3BaseFeatures` | 14 single-bar columns — momentum, volatility, position-in-range, candle shape | The core view of one bar. Computed **per symbol** so one instrument's history never leaks into another's indicators. |
| `V3SessionFeatures` | `session_asia/london/ny/overlap` (0/1) | Identical indicator values mean different things at 03:00 than at 14:00. |
| `V3CostFeatures` | `cost_ratio` | Lets the model *see* trading cost instead of having expensive setups silently filtered out behind it. No-op without a cost table. |
| `V3HTFFeatures` | `htf_rsi_14`, `htf_trend_agreement`, `htf_vol_rel`, `htf_bb_pct_b` | What the slower 5-minute chart says, joined onto each fast bar. |

Two things worth knowing before touching this file:

1. **The lookahead guard (`available_at`) in `V3HTFFeatures`.** A 5-minute bar
   stamped 12:00 is not finished until 12:05, so using it at 12:01 reads the
   future. Each slow bar's timestamp is pushed forward a full timeframe before
   the join. Break this and backtests look brilliant while live trading fails.
2. **Column names must match `retrainer.BASE_FEATURE_COLS` exactly**, and the
   indicator lookback constants (`_RSI_PERIOD`, `_NATR_PERIOD`, …) are baked
   into every saved model. Changing one invalidates existing models rather
   than improving them.

- **Imports from repo:** `ml.core.interfaces`.
- **Imported by:** `src/core/retrainer.py`,
  `src/execution/live_orchestrator.py`,
  `src/strategies/concrete_strategies/ml_strategy.py`,
  `src/ml/feature_pipeline.py`, `src/replay_test.py`,
  `src/analysis/failure_modes.py`, `src/analysis/optimize_brackets.py`,
  `scripts/probe_model.py`, `tests/test_cost_feature.py`.
- **Data artifacts:** none — pure DataFrame transformation.
