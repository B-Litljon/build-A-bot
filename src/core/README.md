# `src/core`

Two unrelated things share this folder, which is worth knowing before you go
looking for something here:

1. **Shared domain types and the notifier** — `signal.py` and
   `notification_manager.py` are imported *by* the live trading paths.
2. **The offline training and evaluation pipeline** — `retrainer.py`,
   `feedback_loop.py` and `resolver.py` are standalone scripts run from
   `run_pipeline.sh`. None of them run while the bot is trading.

The offline loop is meant to work like this: harvest recent bars → replay the
models over them → grade each signal win/loss → measure whether the model's
probabilities still match reality → if they have decayed badly, retrain and
only ship the new model if it survives a walk-forward test. `run_pipeline.sh`
drives all of it, branching on `feedback_loop.py`'s exit code.

See the root [GLOSSARY.md](../../GLOSSARY.md) for domain terms (angel/devil,
ATR/NATR, bracket, chop veto, walk-forward, OOS, Brier score, profit factor,
drift).

## Files

### `retrainer.py` (2166 lines — the big one)
The training pipeline and the promotion gate. Fetches history, engineers
features, builds two labels, runs a 3-fold expanding walk-forward validation,
and overwrites the live model files **only** if the candidate clears every
threshold. Exit code 2 means "trained but rejected", which is a healthy
outcome, not a crash.

- **Imports from repo:** `src.data.factory` (`get_market_provider`),
  `src.data.market_provider`, `src.execution.risk_manager` (`RiskProfile`,
  `coupled_keff`, `_chop_filter_enabled` — this is what keeps the training-time
  chop veto identical to the live one), `src.ml.feature_pipeline`,
  `src.ml.feature_stats`, `src.ml.features.v3_features`,
  `src.ml.regimes.hmm_regime`, `src.core.notification_manager`.
- **Imported by:** `tests/test_retrainer_output_dir.py`. Otherwise run as a
  script (`python -m src.core.retrainer`) from `run_pipeline.sh` Phase 5.
- **Reads:** bars from the configured provider (network); optionally a spread
  table JSON named by `RETRAIN_SPREAD_TABLE`.
- **Writes** (into `model_dir`, default `models/<asset_class>/`, all atomically):
  `angel_latest.pkl`, `devil_latest.pkl`, `metadata.json`, `threshold.json`,
  `feature_stats.json`, and `spread_alphas.json` when the cost experiment is on.

### `feedback_loop.py`
`DriftEvaluator` — scores an already-graded ledger and decides whether the
model has decayed. Gates on calibration (Brier ≤ 0.25) and average return per
trade (≥ 0.05%). Its **exit code is the pipeline's control flow**: 0 healthy,
1 error, 2 drift → run the retrainer.

- **Imports from repo:** `src.core.notification_manager`.
- **Imported by:** nothing — run as `python -m src.core.feedback_loop`.
- **Reads:** `data/resolved_ledger.csv`. **Writes:** no files (terminal output
  plus a Discord alert).

### `resolver.py` — ⚠️ apparently orphaned
`TradeResolver` — replays each BUY signal forward against real bars and labels
it win or loss, checking the stop first so an ambiguous bar counts as a loss.
Uses fixed ±% brackets rather than the volatility-scaled ones production uses.

**Nothing imports it, and `run_pipeline.sh` Phase 3 runs
`python -m src.evaluate_performance` instead** — despite this module's own
usage line claiming otherwise. Both produce `data/resolved_ledger.csv`, so in
practice the file `feedback_loop.py` consumes is written by
`src/evaluate_performance.py`. Flagged for review, not removed.

- **Imports from repo:** none. **Imported by:** nothing.
- **Reads:** `data/signal_ledger.csv`, `data/oos_bars.parquet`.
  **Writes:** `data/resolved_ledger.csv`.

### `notification_manager.py`
`NotificationManager` — every Discord alert the system sends. Silently no-ops
when `DISCORD_WEBHOOK_URL` is unset, and swallows network errors, so callers
can invoke it unconditionally without risking a trading outage. Has separate
entry points for the Alpaca path (takes a `Signal`) and the OANDA path (takes
primitives, because that path uses the *other* `Signal` class).

- **Imports from repo:** `core.signal` — lazily, inside the method, so the
  offline pipeline can import this module without pulling in live-trading code.
- **Imported by:** `src/execution/live_orchestrator.py`,
  `src/execution/oanda_forex_orchestrator.py`,
  `src/strategies/concrete_strategies/ml_strategy.py`,
  `src/core/feedback_loop.py`.
- **Data artifacts:** none (HTTP only).

### `signal.py`
`Signal` and `SignalType` for the **Alpaca** path. Bracket levels travel inside
the `metadata` dict rather than as named fields.

> ⚠️ There is a second, different `Signal` class at
> [`src/strategies/base.py`](../strategies/) used by the OANDA/forex path,
> which has explicit `raw_sl_distance` / `raw_tp_distance` fields. They are not
> interchangeable. See GLOSSARY.md.

- **Imports from repo:** none.
- **Imported by:** `src/execution/live_orchestrator.py`,
  `src/core/notification_manager.py`. **Data artifacts:** none.

### `thresholds.py`
`ANGEL_THRESHOLD` — the single source of truth for the Angel proposal bar
(0.40 unless the `ANGEL_THRESHOLD` env var overrides at process start).
Exists because the value was previously hardcoded independently in eight
files, while the Devil's training population and the bracket fit are both
conditioned on it — the stages must move together or they silently drift.
The retrainer pins the value into `threshold.json` at save time, and
`MLStrategy` prefers that pinned value over this constant, so the env
override is a train/analysis-time knob, not a live-tuning knob.

- **Imports from repo:** none.
- **Imported by:** `core/retrainer.py`, `execution/live_orchestrator.py`,
  `strategies/concrete_strategies/ml_strategy.py`, `ml/train_model.py`,
  `day_trading/train_model.py`, `analysis/optimize_brackets.py`,
  `analysis/failure_modes.py`, `replay_test.py`. **Data artifacts:** none
  directly (retrainer writes the value into `threshold.json` /
  `metadata.json`).

### `events.py` — structured telemetry
The machine-readable half of the bot's output: `emit()` appends one JSON object
per line to `logs/events-YYYY-MM-DD.jsonl`, `write_status()` replaces
`logs/status.json` atomically. Consumers (the dashboard API) read facts instead
of regex-scraping the human log, which also puts several things on the record
that the log never carried: per-bar angel/devil probabilities, entry-guard
blocks, bracket levels at entry.

Three properties are load-bearing, because this runs **inside the live trading
process**: it never raises (every entry point swallows its own exceptions), it
never blocks (a bounded queue plus a daemon writer thread; a full queue drops
events rather than stalling a bar), and it is never called from the tick path.
`tests/test_events.py` pins all three, including a source-level check that no
`events.*` call appears in `_on_tick` and that no call site does arithmetic or
indexing in its arguments — those expressions run *before* `emit`'s safety net.

Telemetry is **opt-in**: nothing is written until `configure()` is called, so
importing a strategy in a test or a backtest cannot append to the live bot's
logs. `run_oanda.py` calls it at startup; `EVENTS_ENABLED=0` is the off switch.

- **Imports from repo:** none (stdlib only).
- **Imported by:** `execution/oanda_forex_orchestrator.py`,
  `strategies/concrete_strategies/ml_strategy.py`, `run_oanda.py`,
  `tests/test_events.py`.
- **Data artifacts:** writes `logs/events-*.jsonl` and `logs/status.json`
  (both gitignored, both regenerable). Read by `dashboard/`.

### `order_management.py` — ⚠️ dead
`OrderParams`, a percentage-multiplier risk config its own docstring describes
as backtest-only and explicitly warns against wiring into live execution. It
justifies its existence by pointing at `grid_search_backtest*.py`, **which no
longer exists** (only the stale `grid_search_results.txt` output remains), and
nothing imports it. Flagged, not removed.

### `ws_stream_simulator.py` — ⚠️ dead
`simulate_ws_stream`, a generator that replays a DataFrame with a sleep between
rows to imitate a live feed. Nothing imports it; the real replay path is
`src/replay_test.py`. Flagged, not removed.

### `__init__.py`
Empty package marker. Note that importers are inconsistent about the prefix —
`core.retrainer` in some files, `src.core.retrainer` in others — depending on
whether `src` or the repo root is on `PYTHONPATH`.
