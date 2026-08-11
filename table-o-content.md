# HAYNES MANUAL — Build-A-Bot

## Complete Codebase Reference Guide

**Generated:** 2026-07-04 (supersedes the 2026-03-16 "Universal Scalper V3.4" edition)
**System:** Two live trading bots sharing one ML factory
**Live surfaces:**
- **V5 OANDA Forex Bot** — async intraday Angel/Devil meta-labeling bot on OANDA v20 (LightGBM), software SL/TP watchdog + dynamic chop filter. *Currently soaking on the practice account (M15 candidate).*
- **V4 Equities Investor** — monthly LightGBM cross-sectional stock ranker on Alpaca paper, diversified top-8 sector-capped basket. *Live on paper, monthly cron.*

> **What changed since V3.4:** The project pivoted off the Alpaca equities/crypto scalper (`live_orchestrator.py`) onto **OANDA forex** for the intraday bot, swapped Angel/Devil from RandomForest to **LightGBM** (2026-05-23), refactored feature computation into a **pluggable ML factory** (`src/ml/`), and stood up a second independent product — the **monthly equities investor**. The old V3.4 Alpaca scalper stack still lives in the tree but is no longer the live system — see [§10 The Boneyard](#10-the-boneyard-legacy--dormant-code).

---

## Table of Contents

0. [Orientation — Two Bots, One Garage](#0-orientation--two-bots-one-garage)
1. [The Engine (V5 OANDA Forex Bot)](#1-the-engine-v5-oanda-forex-bot)
2. [The Drivetrain (OANDA Data & Bar Aggregation)](#2-the-drivetrain-oanda-data--bar-aggregation)
3. [The ECU (The ML Factory — Features, Strategy, Models)](#3-the-ecu-the-ml-factory--features-strategy-models)
4. [The Transmission (Order Execution & Risk / Chop Filter)](#4-the-transmission-order-execution--risk--chop-filter)
5. [The Cure (Retraining Pipeline & Validation Gate)](#5-the-cure-retraining-pipeline--validation-gate)
6. [The Long-Haul Truck (V4 Equities Investor)](#6-the-long-haul-truck-v4-equities-investor)
7. [The Fuel System (Data & Fundamentals Providers)](#7-the-fuel-system-data--fundamentals-providers)
8. [The Dashboard & Comms (Logging & Discord)](#8-the-dashboard--comms-logging--discord)
9. [The Pit Crew (Entry Points, Schedules, DevOps)](#9-the-pit-crew-entry-points-schedules-devops)
10. [The Boneyard (Legacy / Dormant Code)](#10-the-boneyard-legacy--dormant-code)
- [File Index](#file-index)
- [Operational State (2026-07-04)](#operational-state-2026-07-04)

---

## 0. Orientation — Two Bots, One Garage

Build-A-Bot is now **two independent trading systems** that share a common ML factory and a few utilities but run in totally separate processes, brokers, and cadences.

| | **V5 Forex Bot** | **V4 Equities Investor** |
|---|---|---|
| **Broker** | OANDA v20 (practice/live) | Alpaca (paper) |
| **Asset** | FX majors/crosses + metals (XAU/XAG) | US large-cap equities (96-name universe) |
| **Cadence** | Continuous intraday (1/5/15-min bars) | Monthly rebalance (1st of month) |
| **Process** | Long-running async daemon | One-shot synchronous script |
| **Model** | LightGBM Angel/Devil meta-labeling | LightGBM LambdaRank (learning-to-rank) |
| **Entry point** | `run_oanda.py` / `run_soak.sh` | `scripts/portfolio_orchestrator.py` / `run_investor_rebalance.sh` |
| **Model dir** | `models/forex/` (or `models/forex_m15/`) | `models/v4_investor_lgbm.txt` |

They are architecturally isolated: different broker SDKs, different model files, no shared event loop, no shared rate limits. **Running both in parallel is safe.** The only shared runtime surface is the Discord webhook (`NotificationManager`) — their alerts interleave in one channel by design.

**Shared spine:** `src/ml/` (feature factory + trainers), `src/core/retrainer.py` (The Cure — the forex/equities retrainer), `src/core/notification_manager.py`, `src/execution/risk_manager.py`, `src/data/factory.py`.

---

## 1. The Engine (V5 OANDA Forex Bot)

The live intraday bot. A lean async loop wiring `OandaMarketProvider → MLStrategy → OandaOrderManager` with an embedded software SL/TP watchdog and a dynamic chop filter.

### `run_oanda.py` — Launcher

Root entry point. Bootstraps `sys.path`, reads the trained basket/timeframe from the model's `metadata.json`, constructs the components, and runs the orchestrator.

| Concern | Behavior |
|---|---|
| **Default basket** | `_trained_basket()` reads `trained_on_symbols` from `models/forex/metadata.json` — launching with no `--symbols` never trades an out-of-distribution pair. Fallback `["EUR/USD"]`. |
| **OOD warnings** | Warns loudly if a `--symbols` instrument isn't in the trained basket, or if `--granularity` differs from the model's trained timeframe. |
| **Model redirect** | `OANDA_MODEL_DIR` points the whole artifact set (pkls + `metadata.json` + `threshold.json`) at a side model (e.g. `models/forex_m15`) without touching promoted `models/forex`. Mirrors `RETRAIN_MODEL_DIR` on the training side. |
| **Granularity profiles** | `_GRANULARITY_PROFILES`: `1 → ("5m", 260)`, `5 → ("30m", 300)`, `15 → ("1h", 260)` — maps stream granularity to (HTF resample, warmup bars). |
| **CLI/env** | `--symbols`, `--units` (`OANDA_UNITS`, default 1000), `--env practice|live`, `--granularity {1,5,15}`, `--daemon`, `--no-flatten`. |

**Usage:**
```
python3 run_oanda.py                       # trained basket, M1, practice
python3 run_oanda.py --granularity 15      # M15 bars
OANDA_MODEL_DIR=models/forex_m15 python3 run_oanda.py --granularity 15
bash run_soak.sh "" 15                      # daemonized M15 soak (see §9)
```

### `src/execution/oanda_forex_orchestrator.py` → `OandaForexOrchestrator`

The async daemon. One event loop owns the bar path (ML inference), the tick dispatch (watchdog), and graceful shutdown.

**Design constraints (enforced throughout):**
- **Software SL/TP only** — never pass native brackets to the broker.
- The tick callback runs synchronously on the provider's blocking stream thread; it must return in **<50 µs** and do **no blocking I/O**.
- All blocking HTTP (order submit/close) is pushed to an executor so the loop never stalls.

| Method | Purpose |
|---|---|
| `run()` | Registers SIGINT/SIGTERM handlers, calls `_reconcile_on_boot()`, subscribes the stream (bar + tick callbacks), primes history, launches `_stream_with_retry()` and `_liveness_watchdog()`, blocks on the shutdown event. |
| `_on_tick(symbol, bid, ask)` | **Hot path.** Captures the live spread (lock-free dict write — atomic under the GIL) for the cost gate, then checks the open position's SL/TP against bid (long) / ask (short). On breach: sets state `PENDING_CLOSE` under lock (idempotent guard) and dispatches `_watchdog_close` onto the loop via `run_coroutine_threadsafe`. |
| `_on_bar(bar)` | **Cold path.** Drops history/stream seam overlap, appends to the rolling buffer, advances the O(1) Wilder-ATR regime series, samples spread calibration, and once warmed up calls `strategy.generate_signals(df)`. On a signal: computes the SL/TP bracket via `RiskManager.calculate_bracket()`, applies re-entry/reversal guards, submits the target position off-loop, and records position state. |
| `_watchdog_close(symbol)` | Closes a breached position with exponential-backoff retry (`OANDA_CLOSE_MAX_ATTEMPTS`, default 5). On total failure, parks the position in `CLOSE_FAILED` state (still visible to exit-flatten) and fires a **manual-intervention** Discord alert. |
| `_reconcile_on_boot()` | Before trading: verifies broker state per symbol. **Any position found at boot is treated as an orphan** (no known SL/TP) and flattened. If broker state can't even be *verified*, startup aborts — never trade blind. (2026-06-09 ruling.) |
| `_prime_history()` | Fills bar buffers from OANDA REST (last ~5 days) to bypass cold warm-up; also seeds the regime NATR deque + Wilder-ATR state. Records the last historical timestamp for seam dedup. |
| `_stream_with_retry()` | Runs the pricing stream; on disconnect, waits 5s, resets seam state, re-primes history, and resumes. |
| `_liveness_watchdog()` / `_check_stream_liveness()` | Backstop (C3): if the stream delivers nothing (not even heartbeats) for `OANDA_STREAM_STALE_SECONDS` (60s), flattens exposure and forces a reconnect — because software SL/TP is blind during a stall. |
| `shutdown()` / `_flatten_all()` | Dumps final spread calibration, stops the stream, and (if `flatten_on_exit`) closes all positions. Exit-flatten failures raise a manual-intervention alert. |

**Position state machine** (`self._positions[symbol]["state"]`):

| State | Meaning |
|---|---|
| `OPEN` | Live position; tick watchdog armed on its SL/TP. |
| `PENDING_CLOSE` | Watchdog breach detected; close in flight. Blocks new entries and re-fires. |
| `REVERSING` | A flip (long↔short) is submitting; the tick watchdog is muted so it can't fire on stale SL/TP mid-submit. Restored to `OPEN` on a failed flip. |
| `CLOSE_FAILED` | Watchdog close exhausted all retries. Position still open at broker; awaiting a human. Never traded on top of. |

**Spread calibration sink:** once per *closed* bar (off the tick budget), samples `(spread_pct, baseline_natr)`. Every `SPREAD_CALIB_INTERVAL_BARS` (60) and at shutdown, logs per-instrument `alpha_emp = median(spread_pct)/median(baseline_natr)` — grep `SPREAD_CALIB` in the soak log. The US-session median is the value to plug into `RISK_SPREAD_ATR_ALPHA`. This is the soak's primary deliverable.

---

## 2. The Drivetrain (OANDA Data & Bar Aggregation)

### `src/data/oanda_provider.py` → `OandaMarketProvider`

OANDA v20 REST + streaming adapter (implements `MarketDataProvider`).

| Method | Purpose |
|---|---|
| `get_historical_bars(symbol, timeframe_minutes, start, end)` | REST candles (up to `_MAX_CANDLES=5000` per call), mapped to a Polars OHLCV frame with UTC-aware timestamps. Used by `_prime_history`. |
| `subscribe(symbols, bar_callback, tick_callback)` | Registers the bar (loop) and tick (stream-thread) callbacks. |
| `run_stream()` | Blocking v20 pricing-stream loop; aggregates ticks into closed bars at `stream_granularity_minutes` and dispatches them; forwards every tick to `tick_callback`. |
| `_handle_tick` / `_flush_bar` | Tick ingestion and bar sealing. |
| `seconds_since_last_message` | Liveness probe consumed by the orchestrator's C3 watchdog. |
| `force_disconnect` / `reset_stop` / `stop_stream` | Reconnect and shutdown controls. |
| `_to_oanda_symbol(sym)` | Module helper: `"EUR/USD" → "EUR_USD"` (OANDA's underscore convention). Used everywhere as the canonical internal key. |

`_GRANULARITY` maps minutes → OANDA granularity codes (e.g. `1 → "M1"`, `15 → "M15"`, `"5m" → "M5"` for HTF).

> **Note on bar aggregation:** the OANDA provider seals bars itself, so the V3.4 `LiveBarAggregator` is **not** on the forex path. It survives only for the legacy Alpaca stack ([§10](#10-the-boneyard-legacy--dormant-code)).

---

## 3. The ECU (The ML Factory — Features, Strategy, Models)

The brain. Since V3.4 this was refactored from a monolithic `FeatureEngineer` into a **pluggable factory** under `src/ml/`, so features/targets/trainers can be composed per asset class without touching inference code.

### The factory (`src/ml/`)

| File | Role |
|---|---|
| `core/interfaces.py` | ABCs: `BaseFeatureGenerator`, `BaseTargetGenerator`, `BaseTrainer`. |
| `feature_pipeline.py` → `FeaturePipeline` | Composes an ordered list of feature generators into one DataFrame pass. **Single source of truth** shared by training and inference (zero skew). |
| `features/v3_features.py` | `V3BaseFeatures` (RSI/PPO/BBands/NATR + derived + microstructure), `V3SessionFeatures` (time-of-day / session), `V3HTFFeatures` (higher-timeframe resample with lookahead-safe `join_asof`). |
| `targets/v3_targets.py` → `V3DirectionalTarget` | Forward-looking directional label generator. |
| `trainers/v3_rf_trainer.py` → `V3RandomForestTrainer` | Reference sklearn RandomForest trainer for the SDK/factory path. *(The production retrainer builds LightGBM directly — see §5.)* |
| `regimes/hmm_regime.py` | Optional HMM regime features (experiment; loaded only if the model's schema references HMM columns). |

### `src/strategies/concrete_strategies/ml_strategy.py` → `MLStrategy`

The production inference strategy — two-stage Angel/Devil meta-labeling, asset-class aware.

| Aspect | Behavior |
|---|---|
| **Asset class** | `asset_class="equities"` (default) or `"forex"`. Default model paths derive from it: `models/{asset_class}/angel_latest.pkl` / `devil_latest.pkl`. |
| **Feature schema from the model** | Feature names are read from the loaded model's `feature_names_in_` — **not** hardcoded. This lets a retrain change the feature space (e.g. the HMM experiment) without editing the strategy. |
| **Metadata guard** | `_validate_metadata()` refuses to run if the model's `asset_class` doesn't match the strategy's — catches distribution drift (e.g. a forex model loaded into an equities run). |
| **Dynamic threshold** | `_load_threshold()` overrides `devil_threshold` from `models/{asset_class}/threshold.json` (written by the retrainer). |
| **Hot reload** | `_check_model_updates()` watches model mtimes; a newer file triggers in-memory reload + threshold refresh + Discord notice. |
| `generate_signals(df)` | Computes features, runs Angel `predict_proba` (Stage 1, recall gate `angel_threshold` 0.40), appends `angel_prob`, runs Devil `predict_proba` (Stage 2, precision gate), returns a `Signal` (`direction`, `entry_price`, `raw_sl_distance`, metadata with probs/timestamp) on joint approval or `None`. |

**Angel/Devil meta-labeling (unchanged concept, new booster):** the *Angel* is a high-recall directional model; the *Devil* is a high-precision "should we actually take it" model trained only on the Angel-approved subpopulation. Production boosters are **LightGBM** (`models/forex/*.pkl`). `ml_factory_strategy.py` is the SDK-style factory variant of the same idea.

---

## 4. The Transmission (Order Execution & Risk / Chop Filter)

### `src/execution/oanda_order_manager.py` → `OandaOrderManager`

Net-position order manager for OANDA v20.

| Method | Purpose |
|---|---|
| `submit_target_position(symbol, target_units)` | Submits a market order to *reach* a signed target net position. On a reversal the fill spans the closing + opening leg; returns the broker's authoritative resulting `position_units` / `position_avg_price` (not the raw fill), which the orchestrator records. |
| `close_position(symbol)` | Flattens the instrument; raises `OrderCloseError` on failure. Used by the watchdog, boot reconcile, and exit flatten. |
| `sync_position` / `get_net_position` / `get_average_entry_price` | Broker-state reconciliation reads. |

### `src/execution/risk_manager.py` → `RiskManager` + `RiskProfile`

Broker-agnostic bracket calculator **and** the dynamic chop filter. `calculate_bracket(entry, raw_sl_distance, symbol, spread, spread_fresh, regime_series, timestamp)` returns `(sl_dist, tp_dist)` or `None` (vetoed).

**`RiskProfile.for_asset_class("forex")`** — the profile the forex bot runs with:

| Field | Forex value | Meaning |
|---|---|---|
| `sl_atr_multiplier` / `tp_atr_multiplier` | 2.0 / 4.0 | Bracket sizing off signal ATR. Doubled 2026-08-08 to dilute the spread toll; the 2:1 payoff is unchanged. |
| `min_sl_pips` | 2.0 | Absolute FX stop floor (`RISK_FOREX_MIN_SL_PIPS`). |
| `min_sl_pct_metals` | 0.0001 | Metals (XAU/XAG…) stop floor as % of price. |
| `spread_k_base` | 3.0 | **Gate A** (cost): `sl_dist ≥ k_eff × spread`. Equivalently a toll cap — the spread may eat at most `1/k` (33%) of the stop. Was 1.5 (67%) until 2026-08-08. |
| `spread_k_coupling` / `_mode` | 0.0 / `tighten` | Optional vol-coupling of `k_eff` (`coupled_keff`). 0.0 = flat k. |
| `regime_pctile` / `regime_window` / `regime_min_samples` | 20.0 / 260 / 60 | **Gate B** (regime): veto the bottom P% of the rolling volatility distribution; bypassed until `min_samples` bars exist. |
| `spread_atr_alpha` | 0.15 | Proxy spread (= α × baseline ATR) for training/stale-fallback; **the target of soak calibration**. |
| `blackout_start` / `blackout_end` | from `RISK_BLACKOUT_ET` (default `16:55–17:30` ET) | **Gate C** (time): NY-rollover blackout window. |

**The three chop gates** (all toggleable via `RISK_*_GATE_ENABLED`; master kill `RISK_CHOP_FILTER_ENABLED`):

| Gate | Const | Vetoes when |
|---|---|---|
| A — cost | `GATE_SPREAD` / `GATE_STATIC` | The bracket's stop is too tight to clear transaction cost (real spread when fresh, else α-proxy). |
| B — regime | `GATE_REGIME` | Current volatility sits in the dead bottom `regime_pctile` of the rolling window (chop). |
| C — time | `GATE_TIME` | Timestamp falls in the ET blackout window (e.g. 5 pm rollover). |

On a veto the orchestrator increments the matching per-gate counter and logs the running "% of Devil-approved signals vetoed" — the soak's chop-filter telemetry. `last_veto_gate` tells the caller which gate bound.

> **Design ruling:** `Signal.raw_tp_distance` is intentionally discarded — the `RiskManager` multipliers own bracket sizing to preserve training/execution symmetry.

---

## 5. The Cure (Retraining Pipeline & Validation Gate)

### `src/core/retrainer.py` — "The Cure V2"

The forex bot's retrainer. Fetches fresh data, engineers ATR-dynamic labels, runs a **3-fold walk-forward validation gate**, and only promotes models that prove profitable across chronological regimes. **Builds LightGBM boosters** (`import lightgbm as lgb`; translated from RandomForest on 2026-05-23).

**Asset-class routing** — `get_asset_config(DATA_SOURCE)`:

| Key | Behavior |
|---|---|
| `asset_class` | Derived from `DATA_SOURCE` (`oanda → forex`, else `equities`). |
| `model_dir` | `models/{asset_class}` by default; **`RETRAIN_MODEL_DIR`** redirects to a side dir (e.g. `models/forex_m15`) to train a candidate without disturbing the promoted model. |
| `timeframe_minutes` / `htf_timeframe` | `RETRAIN_TIMEFRAME_MINUTES` / `RETRAIN_HTF_TIMEFRAME` — how the M15 candidate was produced. |
| hyperparameters | `get_hyperparameters(asset_class)` — Angel/Devil LightGBM dicts (200 estimators, lr 0.05, depth cap 10/8, `deterministic + force_row_wise` for reproducible threshold-grid picks). Forex uses `min_child_samples=20`, `class_weight=None`; equities `50` / `balanced`. |

**Pipeline (key functions):** `fetch_training_data` → `engineer_features_and_labels` (delegates to the `FeaturePipeline` factory: `V3BaseFeatures` + `V3SessionFeatures` + `V3HTFFeatures`) → `refit_models` (Angel on base features, out-of-fold Angel probs, Devil on the Angel-approved subpopulation vs the **survival target**, time-decay sample weights) → `validate_candidate` (3-fold expanding-window gate; PF measured strictly OOS) → `_find_optimal_threshold` (sweeps Devil threshold to maximize profit factor) → `promote_or_reject` (atomic `save_models` + `save_threshold` + `metadata.json`, or retain production weights).

- **Two Devil targets:** `devil_target` (5-bar *survival* — trained on) vs `devil_target_macro` (45-bar bracket — used for EV/PF gate evaluation).
- **Chop-veto mask** (`_compute_chop_veto_mask`): training-time application of the live chop filter, keeping training and execution symmetric.
- **Gate thresholds:** Brier ≤ 0.30, EV ≥ 0.0005 (R), Profit Factor ≥ 1.2.
- **Exit codes:** `0` promoted · `1` execution error · `2` rejected (production weights intact).

> **Note:** M15 forex retrain (2026-07-02) passed the gate — the first promoted model since 2026-05 — and the candidate lives in `models/forex_m15/` (gitignored). Promotion/soak decisions are the operator's call.

---

## 6. The Long-Haul Truck (V4 Equities Investor)

A completely separate product: a **monthly** cross-sectional stock ranker on Alpaca paper. Where the forex bot trades intraday moves, the investor holds a diversified basket for a month.

### `scripts/portfolio_orchestrator.py` — The rebalancer

| Stage | What it does |
|---|---|
| `refresh_data()` | Subprocesses `investor_data_miner.py` (5y daily OHLCV + macro + fundamentals for the 96-name universe). Skippable with `--skip-refresh`. |
| `build_inference_features()` | Subprocesses `investor_feature_pipeline.py` → `data/processed/v4_inference_features.parquet`. |
| `predict_and_rank()` | Loads `models/v4_investor_lgbm.txt` (LightGBM LambdaRank), scores the latest snapshot per symbol, then **greedy sector-capped selection**: walk the ranking, skip any name whose GICS sector already has `SECTOR_CAP` (2) picks, stop at `TOP_K` (8). |
| `execute_rebalance()` | Liquidates non-top-K positions and rebalances the basket to `TARGET_WEIGHT` (1/8 = 12.5%) each on Alpaca. Sells sized from a live quote and submitted **before** buys. `EQUITY_BUFFER=0.99`, `REBALANCE_DEADBAND=0.005`. `--dry-run` prints intents without trading. |

**Config (top of file):** `TOP_K=8`, `SECTOR_CAP=2`, `TARGET_WEIGHT=1/8`. This replaced the old concentrated `TOP_K=2 @ 50%` design (2026-07-03, commit `1b6f72e`) — a depth sweep showed the ranking edge decays gently (P@8 lift ~1.47× vs P@1 ~1.67×) while deep baskets smooth cold folds (worst fold P@1 0.00 vs P@8 0.24).

### `scripts/investor_universe.py`

Single source of truth for the equities universe, imported by miner, feature pipeline, and orchestrator. **96 large-caps across all 11 GICS sectors.** Exports `UNIVERSE` (list) and `SECTORS` (ticker → sector map that drives the per-sector cap).

### `scripts/investor_train_model.py` — The ranker gate

Trains the LightGBM LambdaRank model with a **walk-forward Precision@K gate** (each trading date = one query group).

| Knob | Default | Meaning |
|---|---|---|
| `TRAIN_DAYS` / `EMBARGO_DAYS` / `TEST_DAYS` | 504 / 60 / 60 | Expanding train window · leakage embargo (= forward-return horizon) · fold width & step. |
| `GATE_P1_MIN_LIFT` | 1.3× | P@1 must beat the positive base rate by 1.3× (`INVESTOR_GATE_P1_LIFT`). |
| `GATE_P2_MIN_LIFT` | 1.2× | P@2 floor. |
| `GATE_P8_MIN_LIFT` | 1.1× | **P@8 floor — the depth actually deployed** (`TOP_K=8`), so a retrain can't sneak through by only being good at pick #1. Added 2026-07-03. |
| `FORCE_SAVE` | off | `INVESTOR_GATE_FORCE=1` escape hatch. |

Model artifact `models/v4_investor_lgbm.txt` + `.metadata.json` are **gitignored** (local artifacts).

### `scripts/investor_data_miner.py` / `investor_feature_pipeline.py`

- **Miner:** 5-year daily OHLCV (yfinance), macro (`VIX`, `10Y_YIELD`), and fundamentals via `get_fundamental_provider()`. Fundamentals lagged `_FUNDAMENTAL_LAG_DAYS=45` to avoid look-ahead. → `data/raw/v4_investor_data.parquet`.
- **Feature pipeline:** momentum (`MOM_WINDOWS`), 1-month reversal, trailing vol, balance-sheet quality (ROA, debt/equity, gross profitability), macro, within-date **cross-sectional rank-normalization** (the biggest historical lift), and the `_top_k_label` forward-return quintile target (`FORWARD_DAYS=60`). Writes both training and inference parquets.

---

## 7. The Fuel System (Data & Fundamentals Providers)

### `src/data/factory.py`

| Factory | Env var | Options |
|---|---|---|
| `get_market_provider()` | `DATA_SOURCE` | `alpaca` (default) · `polygon` · `yahoo` (paper only) · **`oanda`** (forex, V5). |
| `get_fundamental_provider()` | `FUNDAMENTAL_SOURCES` | Comma-separated **chain** — first non-empty result wins. `simfin` (needs `SIMFIN_API_KEY`) · `yfinance`/`yahoo` · `none`. Default `simfin`. |

### Market providers (`src/data/`)
`oanda_provider.py` (§2) · `alpaca_provider.py` · `polygon_provider.py` · `yahoo_provider.py`. Broker-agnostic contracts live in `market_provider.py` / `feed.py` (`MarketDataFeed` ABC), `timeframe.py`, `enums.py`.

### Fundamentals & macro (`src/data/providers/`, `src/data/fundamentals.py`, `src/data/macro.py`)
`FundamentalProvider` / `MacroProvider` ABCs with concrete `simfin_fundamentals.py`, `yf_fundamentals.py`, `composite_fundamentals.py` (the chain), `yf_macro.py`. The composite is what lets the 96-name investor universe fill SimFin-orphaned names from Yahoo's free feed.

---

## 8. The Dashboard & Comms (Logging & Discord)

### `src/core/notification_manager.py` → `NotificationManager`

Discord webhook (silent no-op when `DISCORD_WEBHOOK_URL` unset). Synchronous `requests.post`; on the forex bot loop it's always fired via `_notify()` → executor so posts never stall bar processing.

| Method | Use |
|---|---|
| `send_oanda_trade_alert(...)` | Forex entries, watchdog closes, boot-flattens, close-failed manual alerts. |
| `send_trade_alert(signal, action)` | Legacy Alpaca-style embed. |
| `send_retraining_report(report, promoted)` | The Cure's promote/reject summary. |
| `send_drift_alert(metrics)` | Drift monitor ("The Accountant"). |
| `send_system_message(msg)` | Free-form ops alerts (stream-stale, flatten-failure). |

**Logging:** `run_oanda.py --daemon` uses plain `%(asctime)s [%(levelname)s] %(name)s: %(message)s` stdout logging (journald/soak-log friendly); interactive is DEBUG-level. The V3.4 Rich CLI dashboard belongs to the legacy Alpaca orchestrator ([§10](#10-the-boneyard-legacy--dormant-code)).

---

## 9. The Pit Crew (Entry Points, Schedules, DevOps)

### Live entry points

| File | System | Status | Usage |
|---|---|---|---|
| `run_oanda.py` | V5 Forex Bot | **Live** | `python3 run_oanda.py [--granularity N] [--daemon]` |
| `run_soak.sh` | V5 soak wrapper | **Live** | `bash run_soak.sh [SYMBOLS] [GRANULARITY]` — daemonizes on the practice account, logs `logs/soak_<ts>.log`, PID in `/tmp/soak.pid`, flattens on SIGTERM. Collects `SPREAD_CALIB`. |
| `scripts/portfolio_orchestrator.py` | V4 Investor | **Live (paper)** | `[--dry-run] [--skip-refresh]` |
| `run_investor_rebalance.sh` | V4 wrapper | **Live (paper)** | Sources `.env`, sets `PYTHONPATH`, absolute venv python, logs `logs/investor_rebalance_<ts>.log`. |
| `trading_mcp.py` | MCP server | Active | Trading MCP surface (has `tests/test_trading_mcp.py`, in `.claude/settings.local.json`). |

### Schedule

**User crontab (live):**
```
30 16 1 * * /mnt/storage/mystuf/development/build-A-bot/run_investor_rebalance.sh
```
16:30 PT on the 1st of each month → live paper rebalance. Absolute path is mandatory (cron has no working directory). See `INVESTOR_SCHEDULE.md`.

The forex bot is **not** cron'd — it's a long-running daemon launched by hand via `run_soak.sh` (or a promoted live launch).

### Retrain / soak knobs (env)
`OANDA_MODEL_DIR` (run side) ↔ `RETRAIN_MODEL_DIR` (train side) isolate side models. `RISK_*` tune the chop filter. `FUNDAMENTAL_SOURCES` / `DATA_SOURCE` route data. `OANDA_UNITS`, `OANDA_ENV`, `SOAK_GRANULARITY`.

---

## 10. The Boneyard (Legacy / Dormant Code)

Still in the tree, **not** the live system. Kept for reference / potential reuse; recoverable from git history if deleted.

| Path | What it was | Status |
|---|---|---|
| `src/execution/live_orchestrator.py` | V3.4 Alpaca dual-stream equities/crypto scalper (SymbolState, HTFCache, universal watchdog, Rich dashboard). | **Superseded** by the OANDA orchestrator as the live intraday bot. |
| `run_live.py`, `main.py`, `src/main.py` | V3.4 / Gen-1/2 Alpaca launchers. | Deprecated. |
| `src/core/trading_bot.py`, `order_management.py` | Gen-1 `TradingBot` / `OrderManager`. | Legacy. `OrderParams` still referenced only by old comments. |
| `src/day_trading/` (+ `models/dt_*.pkl`) | "Intraday Trend Engine V4.0" 5-minute day-trade experiment (separate Angel/Devil). | Dormant experiment. |
| `run_pipeline.sh` + `src/replay_test.py`, `evaluate_performance.py`, `data/harvester.py`, `core/resolver.py`, `feedback_loop.py`, `analysis/reinforcement_voter.py` | The Alpaca OOS replay → drift → retrain pipeline. | Mixed: `feedback_loop`/`retrainer` concepts live on; the Alpaca replay harness itself is dormant. |
| `src/utils/bar_aggregator.py` → `LiveBarAggregator` | Clock-aware aggregator for the Alpaca stack. | Legacy (OANDA provider seals its own bars). |
| `backtest_60.py`, `chop_ab_test.py`, `run_chop_ab.sh` | Ad-hoc analysis scripts. | Occasional/manual. |

> **Deleted since V3.4** (recoverable from history): the whole grid-search/backtest script family, the RSI-BBands & SMA-crossover strategies + `strategy_factory`, the Feature Research Lab (`src/research/`, `data/research/`), and the London-Breakout experiment.

---

## File Index

### Root
| File | Category |
|---|---|
| `run_oanda.py` | **V5 forex bot launcher** |
| `run_soak.sh` | V5 practice soak wrapper |
| `run_investor_rebalance.sh` | **V4 investor launcher** |
| `trading_mcp.py` | Trading MCP server |
| `run_live.py`, `main.py` | Legacy Alpaca launchers ([§10](#10-the-boneyard-legacy--dormant-code)) |
| `run_pipeline.sh` | Legacy OOS pipeline orchestrator |
| `Dockerfile`, `docker-compose.yml` | Container build/run (targets legacy `run_live.py`) |

### `scripts/` (V4 Investor)
| File | Category |
|---|---|
| `portfolio_orchestrator.py` | Monthly rebalancer (top-8 sector-capped) |
| `investor_train_model.py` | LightGBM ranker + P@K walk-forward gate |
| `investor_data_miner.py` | 5y OHLCV + macro + fundamentals miner |
| `investor_feature_pipeline.py` | Cross-sectional feature/label builder |
| `investor_universe.py` | 96-name universe + GICS `SECTORS` map |
| `run_paper_live.py`, `smoke_test.py` | Helpers |

### `src/execution/`
| File | Category |
|---|---|
| `oanda_forex_orchestrator.py` | **V5 live daemon** |
| `oanda_order_manager.py` | OANDA net-position order manager |
| `risk_manager.py` | Bracket calculator + dynamic chop filter (Gates A/B/C) |
| `factory_orchestrator.py` | Broker-agnostic SDK orchestrator (factory path) |
| `live_orchestrator.py` | Legacy Alpaca scalper ([§10](#10-the-boneyard-legacy--dormant-code)) |
| `enums.py` | Order enums |

### `src/ml/` (The Factory)
| File | Category |
|---|---|
| `feature_pipeline.py` | `FeaturePipeline` composer |
| `features/v3_features.py` | Base / session / HTF feature generators |
| `targets/v3_targets.py` | Directional target generator |
| `trainers/v3_rf_trainer.py` | Reference RF trainer |
| `regimes/hmm_regime.py` | Optional HMM regime features |
| `core/interfaces.py` | Feature/target/trainer ABCs |
| `data_miner.py`, `train_model.py` | Legacy bulk miner / V3.2 trainer |

### `src/core/`
| File | Category |
|---|---|
| `retrainer.py` | **The Cure V2** — LightGBM forex/equities retrainer + gate |
| `notification_manager.py` | Discord notifications |
| `signal.py` | `Signal` / `SignalType` |
| `feedback_loop.py`, `resolver.py`, `ws_stream_simulator.py` | Legacy OOS pipeline pieces |
| `order_management.py`, `trading_bot.py` | Legacy Gen-1 execution |

### `src/data/`
| File | Category |
|---|---|
| `factory.py` | `get_market_provider` / `get_fundamental_provider` |
| `oanda_provider.py` | OANDA v20 REST + stream |
| `alpaca_provider.py`, `polygon_provider.py`, `yahoo_provider.py` | Market providers |
| `market_provider.py`, `feed.py`, `timeframe.py`, `enums.py` | Broker-agnostic contracts |
| `fundamentals.py`, `macro.py` | Provider ABCs |
| `providers/` | `simfin_`, `yf_`, `composite_fundamentals.py`, `yf_macro.py` |
| `discovery.py`, `harvester.py`, `fetch_training_data.py` | Legacy Alpaca data helpers |

### `src/strategies/`
| File | Category |
|---|---|
| `base.py` | Strategy ABC |
| `concrete_strategies/ml_strategy.py` | **Production Angel/Devil strategy** |
| `concrete_strategies/ml_factory_strategy.py` | SDK/factory variant |

### `models/`
| Path | Category |
|---|---|
| `forex/` | **Promoted V5 forex bot** — `angel_latest.pkl`, `devil_latest.pkl`, `metadata.json`, `threshold.json` |
| `forex_m15/` | M15 candidate (gitignored, staged) |
| `v4_investor_lgbm.txt` (+ `.metadata.json`) | **V4 investor ranker** (gitignored) |
| `forex_swing/`, `dt_*.pkl` | Dormant experiments |

> All of `models/` is **gitignored** — model artifacts are local, rebuilt by the retrainers.

---

## Operational State (2026-07-04)

| System | State |
|---|---|
| **V5 Forex Bot** | **Soaking** on the practice account since 2026-07-02 — M15 candidate promoted via `OANDA_MODEL_DIR=models/forex_m15`, full trained basket, `--granularity 15`, daemon (PID in `/tmp/soak.pid`, log `logs/soak_2026-07-02_0215.log`). Heartbeats are sparse (~hours); may hold positions overnight/weekends. **Verify alive** (`ps`, PID, log mtime) before claiming it's running. |
| **V4 Investor** | **Live on paper.** Diversified top-8 sector-capped basket committed `1b6f72e`; monthly cron installed (`30 16 1 * *`). Rebalance orders queue when markets are closed and fill at the next open. |
| **Git** | Local `main` is ahead of `origin/main` (unpushed by operator choice). Stale remote branches pruned 2026-07-04; `main` is the only remote branch. |

---

**End of HAYNES MANUAL — Build-A-Bot**
