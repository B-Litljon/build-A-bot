# HAYNES MANUAL — Universal Scalper V3.1

## Complete Codebase Reference Guide

**Generated:** 2026-02-26
**System:** Universal Scalper V3.1 — Dual-Stream Live Trading Bot
**Architecture:** Angel/Devil Meta-Labeling + ATR-Based Dynamic Brackets + Rich CLI Dashboard

---

## Table of Contents

1. [The Engine (Core Logic & State Management)](#1-the-engine-core-logic--state-management)
2. [The Drivetrain (Data Ingestion & Aggregation)](#2-the-drivetrain-data-ingestion--aggregation)
3. [The ECU (Machine Learning & Strategy)](#3-the-ecu-machine-learning--strategy)
4. [The Transmission (Order Execution & Risk)](#4-the-transmission-order-execution--risk)
5. [The Simulator (Backtesting & Grid Search)](#5-the-simulator-backtesting--grid-search)
6. [The Dashboard (UI, Logging & Systemd)](#6-the-dashboard-ui-logging--systemd)
7. [The Fuel System (Data Providers & Mining)](#7-the-fuel-system-data-providers--mining)
8. [The Pit Crew (Pipeline Orchestration & DevOps)](#8-the-pit-crew-pipeline-orchestration--devops)

---

## 1. The Engine (Core Logic & State Management)

The engine is the state machine that tracks whether each symbol is flat, pending, in a trade, or cooling down. It is the central nervous system of the live bot.

---

### src/execution/live_orchestrator.py -> `SymbolState` (Enum)

Defines the four lifecycle states for a single symbol's trading slot.

| Member | Purpose |
|---|---|
| `FLAT` | No position; eligible for new signals. |
| `PENDING` | Order submitted to Alpaca; awaiting fill confirmation from TradingStream. |
| `IN_TRADE` | Position filled; bracket TP/SL legs are active on the exchange. |
| `COOLING` | Bracket resolved (TP or SL hit); 5-minute cooldown timer running before re-entry is allowed. |

---

### src/execution/live_orchestrator.py -> `SymbolContext`

Holds all mutable runtime state for a single tracked symbol — one instance per symbol in the basket.

| Property | Type | Purpose |
|---|---|---|
| `symbol` | `str` | The ticker (e.g. `"TSLA"`, `"BTC/USD"`). |
| `is_crypto` | `bool` | `True` if the symbol is a crypto pair (contains `/`); governs clock gating and TIF selection. |
| `aggregator` | `LiveBarAggregator` | Per-symbol bar aggregator that builds 1-min candles and maintains a rolling history window. |
| `state` | `SymbolState` | Current lifecycle state — the single source of truth for whether the bot can enter a trade. |
| `lock` | `asyncio.Lock` | Guards state transitions; prevents race conditions between bar handler and trade update handler. |
| `last_client_order_id` | `Optional[str]` | Deduplication — stores the most recent bracket entry's client order ID to prevent double-submission on the same bar. |
| `entry_price` | `Optional[float]` | The filled average price of the current position (set by `_on_trade_update`). |
| `entry_qty` | `Optional[float]` | The filled quantity (set by `_on_trade_update`). |
| `_cooling_task` | `Optional[asyncio.Task]` | Handle to the cooling timer coroutine so it can be cancelled on shutdown. |
| `last_price` | `Optional[float]` | Most recent close price, used for dashboard display. |
| `last_atr` | `Optional[float]` | Most recent NATR-14 value, used for dashboard display and kill switch evaluation. |
| `last_conviction` | `Optional[float]` | Most recent Devil probability, used for dashboard display. |

---

### src/execution/live_orchestrator.py -> `LiveOrchestrator`

The async daemon that drives live paper-trading. Supports dual-stream operation: equities via `StockDataStream` (IEX) and crypto via `CryptoDataStream`, multiplexed into a single asyncio event loop.

| Method | Purpose |
|---|---|
| `__init__(symbols, api_key, secret_key, paper, angel_model_path, devil_model_path, daemon_mode)` | Initialises all sub-components: `MLStrategy`, `FeatureEngineer`, `NotificationManager`, `TradingClient`, per-symbol `SymbolContext` dict, and stream placeholders. |
| `run()` | Main entry-point: registers SIGTERM/SIGINT handlers, refreshes market clock, warms up aggregators, subscribes WebSocket streams, starts the dashboard loop, then blocks on `asyncio.gather()` until shutdown. |
| `_on_bar(bar)` | Shared callback for both stock and crypto streams — ingests a raw 1-min bar, applies the smart clock gate (equity bars blocked outside RTH; crypto always passes), feeds bar to aggregator, and kicks off `_run_inference` off-thread if enough history exists. |
| `_run_inference(symbol, history_df)` | CPU-bound inference executed via `asyncio.to_thread` — validates schema, computes TA-Lib indicators via `FeatureEngineer`, checks ATR kill switch, runs Angel Stage 1 then Devil Stage 2, and returns a `Signal` on joint approval or `None`. |
| `_handle_signal(ctx, sig)` | Gates the signal through the `SymbolState` machine (must be FLAT), generates a unique `client_order_id`, transitions to PENDING, and calls `_submit_bracket_order` off-thread. |
| `_submit_bracket_order(sig, client_order_id)` | Submits a market entry with OTO bracket (TP + SL) to Alpaca REST API — calculates position size from 2% account risk, selects TIF (GTC for crypto, DAY for equities). |
| `_on_trade_update(data)` | Drives `SymbolState` transitions from Alpaca order lifecycle WebSocket events — `fill`/`partial_fill` -> IN_TRADE, `canceled`/`expired`/`rejected` -> FLAT, sell-side `fill` while IN_TRADE -> COOLING. |
| `_enter_cooling(ctx)` | Transitions symbol to COOLING and schedules `_reset_after_cooling` as an async task. |
| `_reset_after_cooling(ctx)` | Sleeps for `COOLING_SECONDS` (300s / 5min), then returns the symbol to FLAT. |
| `_warmup_aggregator()` | Pre-loads each symbol's `LiveBarAggregator` with the last 60 one-minute bars from Alpaca REST so SMA-50 and all indicators are ready on first live bar. |
| `_fetch_crypto_history(end_time, progress)` | Fetches historical 1-min bars for all crypto symbols using `CryptoHistoricalDataClient`. |
| `_fetch_stock_history(end_time, progress)` | Fetches historical 1-min bars for all stock symbols using `StockHistoricalDataClient` with split adjustment. |
| `_refresh_market_clock()` | Queries Alpaca market clock REST endpoint (rate-limited to every 30s) to determine if the equity market is open. |
| `_request_shutdown()` | Signal handler callback — sets the `_shutdown_event`. |
| `_shutdown()` | Graceful teardown: cancels cooling timers, cancels open Alpaca orders, stops all WebSocket streams, sends Discord notification. |
| `_ensure_utc(ts)` | Static utility to normalise any datetime to UTC-aware. |

**Module-level constants (live tuning parameters):**

| Constant | Value | Purpose |
|---|---|---|
| `ATR_KILL_SWITCH_THRESHOLD` | `0.5204` | NATR-14 percentage above which signals are dropped (high-volatility regime safety). From `drift_report.json`. |
| `ANGEL_THRESHOLD` | `0.40` | Minimum Angel probability to propose a trade (high recall). |
| `DEVIL_THRESHOLD` | `0.50` | Minimum Devil probability to approve a trade (high precision). |
| `SL_ATR_MULTIPLIER` | `1.5` | Stop-loss distance = 1.5x ATR below entry. |
| `TP_ATR_MULTIPLIER` | `3.0` | Take-profit distance = 3.0x ATR above entry. |
| `COOLING_SECONDS` | `300` | 5-minute cooldown after any bracket resolves. |
| `MIN_HISTORY_BARS` | `60` | Minimum bars in history before inference is attempted. |
| `HISTORY_SIZE` | `120` | Rolling window retained by each `LiveBarAggregator`. |
| `ACCOUNT_RISK_PER_TRADE` | `0.02` | 2% of account equity risked per trade. |
| `DEFAULT_SYMBOLS` | `["TSLA", "NVDA", "MARA", "COIN", "SMCI", "BTC/USD", "ETH/USD"]` | Default trading basket. |

---

### src/core/signal.py -> `SignalType` (Enum)

Enumeration of possible signal directions: `BUY`, `SELL`, `HOLD`.

### src/core/signal.py -> `Signal` (dataclass)

The universal data object that bridges strategy output to order execution.

| Field | Type | Purpose |
|---|---|---|
| `symbol` | `str` | Ticker symbol. |
| `type` | `SignalType` | Direction of the signal. |
| `price` | `float` | Price at signal generation (close of the latest bar). |
| `confidence` | `float` | Devil model probability — the conviction score. |
| `timestamp` | `datetime` | Bar timestamp that generated the signal. |
| `metadata` | `Dict[str, Any]` | Carries `angel_prob`, `devil_prob`, `natr_14`, `atr_abs`, `sl_price`, `tp_price`. |

---

### src/core/trading_bot.py -> `TradingBot`

The legacy Gen-1 trading bot class (pre-LiveOrchestrator). Wires together strategy, data provider, order manager, and bar aggregators. Still used by the root `main.py` entry point.

| Method | Purpose |
|---|---|
| `__init__(strategy, capital, trading_client, data_provider, symbols, target_intervals, notification_manager)` | Creates per-symbol `LiveBarAggregator` instances, initialises `OrderManager` with strategy's order params. |
| `warmup()` | Fetches historical 1-min bars from the data provider and feeds them through aggregators to pre-fill indicator windows. |
| `handle_bar_update(bar)` | Async callback for incoming bars — runs `OrderManager.monitor_orders()` on every tick for exit checks, aggregates the bar, and calls `strategy.analyze()` on new aggregated candles. |
| `place_orders(signals)` | Iterates signals and calls `OrderManager.place_order()` for each BUY signal. |
| `run()` | Syncs internal state with Alpaca positions and subscribes to the data provider's stream. |
| `log_status()` | Logs current capital and active orders. |

---

## 2. The Drivetrain (Data Ingestion & Aggregation)

The drivetrain handles getting raw market data into the bot and converting it into clean, aggregated candles that the strategy can consume.

---

### src/utils/bar_aggregator.py -> `LiveBarAggregator`

Clock-aware aggregator that converts a live stream of 1-minute bars into higher-timeframe OHLCV candles using logical, wall-clock-aligned windows. One instance per (symbol, timeframe) pair.

| Method | Purpose |
|---|---|
| `__init__(timeframe, history_size=240)` | Sets the aggregation window (minutes) and the maximum number of retained candles in `history_df`. |
| `add_bar(new_bar) -> bool` | Ingests a single 1-min bar dict; returns `True` if one or more aggregated candles were appended to `history_df` (a window closed), `False` if still accumulating. |
| `_aggregate_and_update(window_timestamp)` | Collapses the current buffer into one aggregated candle (first open, max high, min low, last close, sum volume) and appends it to `history_df`. |
| `_forward_fill_gaps(closed_window, new_window)` | Injects synthetic flat candles (OHLC = prior close, volume = 0) for every missing interval between two windows, keeping the time-series continuous for TA-Lib. |
| `_append_to_history(df)` | Appends rows to `history_df` and trims to `history_size`. |
| `_window_floor(ts)` | Computes the logical window start for a timestamp (e.g. 12:34 -> 12:30 for 5-min timeframe). |
| `_ensure_utc(ts)` | Normalises a timestamp to UTC-aware. |

**Key Property:**

| Property | Type | Purpose |
|---|---|---|
| `history_df` | `pl.DataFrame` | Rolling window of aggregated candles with schema `{timestamp, open, high, low, close, volume}`. This is the DataFrame passed to strategy analysis. |

**Schema Contract (`_SCHEMA`):**
```
timestamp: Datetime(us, UTC)
open:      Float64
high:      Float64
low:       Float64
close:     Float64
volume:    Float64
```

---

### WebSocket Connections (inside LiveOrchestrator)

| Stream | SDK Class | Data Feed | Purpose |
|---|---|---|---|
| Crypto bars | `alpaca.data.live.crypto.CryptoDataStream` | Alpaca crypto feed | Receives 1-min bars for crypto pairs (BTC/USD, ETH/USD). |
| Stock bars | `alpaca.data.live.stock.StockDataStream` | `DataFeed.IEX` | Receives 1-min bars for equity symbols (TSLA, NVDA, etc.). |
| Trade updates | `alpaca.trading.stream.TradingStream` | Alpaca order lifecycle | Receives fill, cancel, expire, reject events that drive `SymbolState` transitions. |

Both bar streams route to the same handler: `LiveOrchestrator._on_bar()`.
The trade update stream routes to: `LiveOrchestrator._on_trade_update()`.

---

### src/core/ws_stream_simulator.py -> `simulate_ws_stream()`

A simple generator function that simulates a WebSocket connection by iterating through historical Polars DataFrames row-by-row with configurable delay. Used for offline testing.

---

## 3. The ECU (Machine Learning & Strategy)

The ECU is the brain that decides when to enter trades. It uses a two-stage Angel/Devil meta-labeling architecture built on Random Forest classifiers.

---

### src/ml/feature_pipeline.py -> `FeatureEngineer`

Computes all technical indicators used for ML inference. This is the **single source of truth** for feature computation — imported by both training and inference code to prevent skew.

| Method | Purpose |
|---|---|
| `compute_indicators(df) -> pl.DataFrame` | Takes raw OHLCV DataFrame, computes all TA-Lib indicators and derived features, returns enriched DataFrame. |
| `generate_labels(df, lookahead, min_gain) -> pl.DataFrame` | Generates forward-looking `target` labels for training (1 if close rises > min_gain% within lookahead bars). |
| `clean_data(df) -> pl.DataFrame` | Replaces NaN with null, then drops all null rows. |
| `run(df) -> pl.DataFrame` | Convenience pipeline: `compute_indicators` -> `generate_labels` -> `clean_data`. |

**Indicators computed by `compute_indicators()`:**

| Feature | TA-Lib Function | Description |
|---|---|---|
| `rsi_14` | `talib.RSI(close, 14)` | 14-period Relative Strength Index. |
| `ppo` | `talib.PPO(close, 12, 26)` | Percentage Price Oscillator (normalized MACD). |
| `bb_upper`, `bb_middle`, `bb_lower` | `talib.BBANDS(close, 20, 2, 2)` | 20-period Bollinger Bands (2 std dev). |
| `sma_50` | `talib.SMA(close, 50)` | 50-period Simple Moving Average. |
| `natr_14` | `talib.NATR(high, low, close, 14)` | 14-period Normalized Average True Range (percentage). |

**Derived features (Polars expressions):**

| Feature | Formula | Description |
|---|---|---|
| `bb_pct_b` | `(close - bb_lower) / (bb_upper - bb_lower)` | Bollinger %B — position within the bands. |
| `bb_width_pct` | `(bb_upper - bb_lower) / bb_middle` | Bollinger Band width as percentage of middle band. |
| `price_sma50_ratio` | `close / sma_50` | Price relative to the 50-SMA (trend strength). |
| `log_return` | `log(close / close.shift(1))` | Logarithmic return bar-over-bar. |
| `hour_of_day` | `timestamp.dt.hour()` | Hour of the day (0-23) for time-of-day effects. |
| `dist_sma50` | `(close - sma_50) / sma_50` | Distance from SMA-50 as a fraction. |
| `vol_rel` | `volume / volume.rolling_mean(20)` | Relative volume vs 20-bar average. |

When `main()` is run directly, it processes all `data/raw/*_1min.parquet` files, applies `run()` to each, concatenates, and writes `data/processed/training_data.parquet`.

---

### src/ml/train_model.py -> `main()`

Meta-labeling model training pipeline that produces the Angel and Devil Random Forest models.

**Training Flow:**

1. **Load** `data/processed/training_data.parquet`.
2. **Stage 1 — Train the Angel (Direction, high recall):** Fit `RandomForestClassifier(n_estimators=100, max_depth=10)` on all features against the `target` label.
3. **Generate meta-features:** Use 3-fold `cross_val_predict` to get out-of-fold Angel probabilities (prevents data leakage).
4. **Stage 2 — Train the Devil (Conviction, high precision):** Build meta-feature set = base features + `angel_prob`. Train `RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced")` only on rows where the Angel proposed a trade, against a meta-target of whether the Angel was correct.
5. **Export** both models as `.joblib` files to `src/ml/models/`.

**Feature columns used for training (10 features):**
`rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct, price_sma50_ratio, log_return, hour_of_day, dist_sma50, vol_rel`

**Output files:**
- `src/ml/models/angel_rf_model.joblib`
- `src/ml/models/devil_rf_model.joblib`

---

### src/strategies/strategy.py -> `Strategy` (ABC)

Abstract base class that all trading strategies must implement.

| Abstract Member | Type | Purpose |
|---|---|---|
| `analyze(data)` | Method | Takes market data, returns trading signals. |
| `get_order_params()` | Method | Returns the `OrderParams` for this strategy's risk profile. |
| `warmup_period` | Property | Minimum number of candles required before analysis can run. |

---

### src/strategies/concrete_strategies/ml_strategy.py -> `MLStrategy`

The production strategy: two-stage Angel/Devil meta-labeling inference with hot-reloading.

| Method | Purpose |
|---|---|
| `__init__(angel_path, devil_path, angel_threshold, devil_threshold, warmup_period)` | Loads both joblib models, stores their file modification times for hot-reload detection, initialises `FeatureEngineer`. |
| `analyze(data) -> Tuple[List[Signal], float]` | Runs `_check_model_updates()`, then for each symbol: computes features, runs Angel predict_proba (Stage 1), appends `angel_prob` as meta-feature, runs Devil predict_proba (Stage 2), emits BUY signal only if both pass thresholds. |
| `_check_model_updates() -> bool` | Monitors model file modification times on disk; if a newer file is detected, hot-reloads the model in memory and sends a Discord notification. |
| `_generate_features(df) -> Optional[pl.DataFrame]` | Calls `FeatureEngineer.compute_indicators()` and drops null rows — ensures zero training/inference skew. |
| `get_order_params() -> OrderParams` | Returns the strategy's risk parameters (2% risk, 0.5% TP, 0.2% SL). |

**Properties:**
- `warmup_period` -> returns `self.warmup` (default 60).
- `timeframe` -> `1` (1-minute bars).
- `feature_names` -> the 10 ML features listed above.
- `angel_model` / `devil_model` -> the loaded scikit-learn RandomForestClassifiers.

---

### src/strategies/concrete_strategies/rsi_bbands.py -> `RSIBBands`

Legacy Gen-1 strategy: two-stage RSI + Bollinger Bands mean reversion with bullish engulfing confirmation.

| Method | Purpose |
|---|---|
| `analyze(data)` | Stage 1: Arms trigger when price < lower BB AND RSI <= threshold. Stage 2: Fires BUY when RSI recovers into range AND bandwidth ROC > threshold AND bullish engulfing candle pattern confirmed. |
| `is_bullish_engulfing(df) -> bool` | Checks the last two candles for a bullish engulfing pattern. |
| `get_order_params()` | Returns `OrderParams(risk_percentage=0.02, tp_multiplier=1.5, sl_multiplier=0.9)`. |

---

### src/strategies/concrete_strategies/sma_crossover.py -> `SMACrossover`

Reference implementation: generates BUY on fast SMA (10) crossing above slow SMA (50).

---

### src/strategies/strategy_factory.py -> `create_strategy(name, **kwargs)`

Factory function that instantiates strategies by name from a registry.

| Key | Class |
|---|---|
| `"rsi_bollinger"` | `RSIBBands` |
| `"sma_crossover"` | `SMACrossover` |

Note: `MLStrategy` is not in the factory — it is instantiated directly.

---

## 4. The Transmission (Order Execution & Risk)

The transmission converts signals into real Alpaca orders and manages position lifecycle.

---

### Live Execution (LiveOrchestrator)

**Where bracket orders are submitted:**
`src/execution/live_orchestrator.py` -> `LiveOrchestrator._submit_bracket_order(sig, client_order_id)`

This method:
1. Queries `TradingClient.get_account()` to get current equity.
2. Calculates `risk_dollars = equity * 0.02` (2% risk per trade).
3. Calculates `qty = risk_dollars / (entry_price - sl_price)`.
4. Submits a `MarketOrderRequest` with `order_class="bracket"`, `TakeProfitRequest(limit_price=tp_price)`, and `StopLossRequest(stop_price=sl_price)`.
5. Uses `TimeInForce.GTC` for crypto, `TimeInForce.DAY` for equities.

**Where stop-loss and take-profit are calculated:**
`src/execution/live_orchestrator.py` -> `LiveOrchestrator._run_inference()`, inside the Signal metadata:
```
sl_price = current_price - (SL_ATR_MULTIPLIER * atr_abs)   # 1.5x ATR below entry
tp_price = current_price + (TP_ATR_MULTIPLIER * atr_abs)   # 3.0x ATR above entry
```
Where `atr_abs = (natr_14 / 100) * current_price` (converting percentage NATR to absolute ATR).

**Where trade_updates are intercepted:**
`src/execution/live_orchestrator.py` -> `LiveOrchestrator._on_trade_update(data)`

This method drives the authoritative state machine:
- `fill` / `partial_fill` event -> PENDING to IN_TRADE (records fill price and qty).
- `canceled` / `expired` / `rejected` event -> resets to FLAT.
- Sell-side `fill` while IN_TRADE -> enters COOLING via `_enter_cooling()`.

---

### Legacy Execution (OrderManager)

**File:** `src/core/order_management.py`

#### `OrderParams`
Container for order calculation parameters: `risk_percentage`, `tp_multiplier`, `sl_multiplier`, `use_trailing_stop`.

#### `OrderCalculator`
Translates strategy parameters into concrete order values.

| Method | Purpose |
|---|---|
| `calculate_quantity(entry_price, current_capital)` | `(capital * risk_percentage) / entry_price`. |
| `calculate_stop_loss(entry_price)` | `entry_price * sl_multiplier`. |
| `calculate_take_profit(entry_price)` | `entry_price * tp_multiplier`. |

#### `OrderManager`
Manages order placement and tick-by-tick SL/TP monitoring for the legacy `TradingBot`.

| Method | Purpose |
|---|---|
| `place_order(signal, current_capital)` | Calculates qty/SL/TP, submits `MarketOrderRequest` to Alpaca, stores order in `active_orders` dict, sends Discord notification. |
| `monitor_orders(market_data)` | Called on every incoming bar — checks all `active_orders` against current prices; submits SELL if SL or TP is hit. |
| `sync_positions()` | On startup, reconciles memory with actual Alpaca positions — adopts unmanaged positions with reconstructed SL/TP. |

---

## 5. The Simulator (Backtesting & Grid Search)

The simulator validates the ML strategy on historical data before live deployment.

---

### OOS Replay Pipeline (Multi-Phase)

The full out-of-sample validation pipeline is orchestrated by `run_pipeline.sh` and consists of these phases:

#### Phase 1: Data Harvesting
**File:** `src/data/harvester.py`

| Function | Purpose |
|---|---|
| `harvest_oos_data()` | Fetches the last 7 days of 1-min IEX bars for `["TSLA", "NVDA", "MARA", "COIN", "SMCI"]` and writes `data/oos_bars.parquet`. |

#### Phase 2: Replay Simulation
**File:** `src/replay_test.py`

| Class | Purpose |
|---|---|
| `MockAlpacaProvider` | Loads `data/oos_bars.parquet` and yields timestamp-grouped bars to simulate real-time cross-sectional market data. |
| `ReplayHarness` | Simulates live trading: maintains per-symbol history, runs Angel/Devil inference on each bar, accumulates BUY signals in-memory, and writes `data/signal_ledger.parquet` as a single batch. Uses **Ironclad Alignment** — selects features by name in exact training order before NumPy conversion. |

| Method | Purpose |
|---|---|
| `ReplayHarness._run_inference(symbol, close_price, timestamp)` | Generates features, runs Angel -> Devil two-stage inference, returns signal dict on joint approval. |
| `ReplayHarness.save_ledger(output_path)` | Writes the signal ledger to Parquet with strict schema preservation (`datetime[us, UTC]`). |

#### Phase 3: Performance Evaluation
**File:** `src/evaluate_performance.py`

| Function | Purpose |
|---|---|
| `load_data()` | Loads `data/oos_bars.parquet` and `data/signal_ledger.parquet`, detects ATR column. |
| `vectorized_backtest(bars_df, signals_df, atr_col)` | Performs vectorized backtest: for each BUY signal, calculates dynamic SL/TP using ATR multipliers (1.5x SL, 3.0x TP), looks ahead up to 15 bars for exit, applies volatility kill switch (higher conviction required in high-ATR regime). |
| `calculate_metrics(trades_df)` | Computes win rate, net profit (R-multiples), max drawdown, and profit factor. |

**Output:** `data/evaluation_results.parquet`

**Dynamic Thresholding:** If `data/drift_report.json` exists, the evaluator applies regime-based conviction requirements — `BASE_THRESHOLD=0.50` for normal, `HIGH_VOLATILITY_THRESHOLD=0.75` for high ATR.

#### Phase 4: Trade Resolution (Legacy)
**File:** `src/core/resolver.py`

| Class | Purpose |
|---|---|
| `TradeResolver` | Resolves BUY signals to Win/Loss outcomes using conservative bracket simulation (+0.5% TP, -0.2% SL). Checks SL first (conservative), then TP, with EOD fallback. |
| `TradeOutcome` | Dataclass holding resolution details: TP/SL targets, exit price/time, outcome (1=Win, 0=Loss). |

**Input:** `data/signal_ledger.csv` + `data/oos_bars.parquet`
**Output:** `data/resolved_ledger.csv`

---

### Root-Level Backtest Scripts

All backtest scripts share the same basic pattern: load SPY 1-min parquet data, instantiate `MLStrategy`, create a `LiveBarAggregator`, iterate bars, run `strategy.analyze()` on each new candle, and track trades with a local `BacktestOrderManager` (or `BOM`) class.

| Script | Date Range | Risk Profile | Purpose |
|---|---|---|---|
| `backtest_full.py` | Full 2024 | Sniper (threshold 0.70, TP 0.5%, SL 0.2%) | Full-year backtest of ML strategy. |
| `backtest_ml_strategy.py` | 2024 onwards | Sniper (threshold 0.70) | Full backtest with detailed results and 15-bar timeout. |
| `backtest_ml_strategy_quick.py` | Jan-Feb 2024 | Sniper (threshold 0.70) | 2-month quick validation. |
| `backtest_quick.py` | Jan 1-15, 2024 | Sniper (threshold 0.70) | 2-week smoke test. |
| `backtest_60.py` | 2024 onwards | Threshold 0.60 | Full-year backtest at lower threshold. |

**BacktestOrderManager / BOM pattern:**
- `place_order(signal, capital)` — calculates qty, SL, TP from `OrderParams`.
- `monitor_orders(market_data)` — checks bar's high/low against SL/TP, records PnL.

---

### Grid Search Scripts

Grid searches sweep risk profiles (SL/TP multipliers) to find the optimal configuration.

| Script | Data | Method | Configurations Tested |
|---|---|---|---|
| `grid_search_vectorized.py` | Full 2024 SPY | Pre-computes all features and batch-predicts all probabilities, then simulates trades per config. | A (Scalper: SL 0.2%, TP 0.5%), B (Balanced: SL 0.5%, TP 0.5%), C (Swinger: SL 0.5%, TP 1.0%) at threshold 0.50. |
| `grid_search_backtest.py` | Full 2024 SPY | Bar-by-bar simulation with `MLStrategy.analyze()` per candle. | Same A/B/C configs, threshold 0.50, 15-bar timeout. |
| `grid_search_backtest_q1.py` | Q1 2024 SPY | Same as above, restricted to Jan-Mar 2024 for faster iteration. | Same A/B/C configs. |
| `grid_search_fast.py` | Jan 1-15, 2024 | Quick 2-week sweep for rapid hypothesis testing. | Same A/B/C configs. |

**Success Criteria:** Profit Factor > 1.2 AND Trades >= 50 (scaled for shorter periods).

---

### Threshold Optimization
**File:** `src/analysis/optimize_threshold.py`

Analyzes model probability distribution on out-of-sample data and sweeps thresholds `[0.30, 0.35, 0.40, 0.42, 0.45, 0.48, 0.50]` to find the optimal operational threshold. Uses vectorized backtest with Scalper risk profile (TP 0.5%, SL 0.2%, 15-bar timeout).

---

## 6. The Dashboard (UI, Logging & Systemd)

---

### Rich CLI Dashboard

**File:** `src/execution/live_orchestrator.py` -> `LiveOrchestrator`

The interactive dashboard is driven by the `_dashboard_update_loop()` coroutine using Rich's `Live` context manager, refreshing once per second.

| Method | Purpose |
|---|---|
| `_generate_dashboard()` | Builds the Rich `Layout` with three panels: header, symbol table, activity console. |
| `_create_header()` | Renders bot name, environment (PAPER/LIVE), market status (OPEN/CLOSED), UTC clock, uptime, and symbol counts. |
| `_create_symbol_table()` | Renders a table with columns: Symbol, Asset (EQUITY/CRYPTO), Last Price, ATR, State, Last Conviction. |
| `_create_activity_panel()` | Renders the last 5 activity entries (timestamped, color-coded by level). |
| `_log_activity(symbol, message, level)` | Appends an `ActivityEntry` to the deque (max 5 entries). |
| `_dashboard_update_loop()` | In interactive mode: runs `Rich.Live` with 1-second refresh. In daemon mode: idles with periodic heartbeat logs. |

**ActivityEntry:** Dataclass with `timestamp`, `symbol`, `message`, `level` (info/success/warning/error).

---

### Headless Daemon Mode

**File:** `src/execution/live_orchestrator.py`

When `daemon_mode=True`:
- **No Rich UI** — no ANSI escape sequences, no alternate-screen buffer.
- **Plain stdout logging** — uses `logging.StreamHandler(sys.stdout)` with `%Y-%m-%dT%H:%M:%S` format for clean journald parsing.
- **Heartbeat** — the dashboard loop emits a DEBUG-level heartbeat once per minute.
- **Warm-up** — bypasses Rich Progress bars; reports status via `logger.info()`.

**Entry point:** `python3 run_live.py --daemon`

---

### Logging Configuration

**File:** `src/execution/live_orchestrator.py` -> `setup_logging(daemon_mode)`

| Mode | Handler | Format | Use Case |
|---|---|---|---|
| Interactive (`daemon_mode=False`) | `RichHandler(rich_tracebacks=True, markup=True)` | `%(message)s` | Terminal / tmux sessions. |
| Daemon (`daemon_mode=True`) | `logging.StreamHandler(sys.stdout)` | `%(asctime)s [%(levelname)s] %(name)s: %(message)s` | systemd / journald. |

---

### Discord Notifications

**File:** `src/core/notification_manager.py` -> `NotificationManager`

| Method | Purpose |
|---|---|
| `__init__(webhook_url)` | Reads `DISCORD_WEBHOOK_URL` from env if not provided. |
| `send_trade_alert(signal, action)` | Sends a formatted embed to Discord with Angel/Devil probabilities, price, and color-coded entry/exit indicator. Username: "Build-A-Bot Executive". |
| `send_system_message(message)` | Sends a generic system status update as plain content. |
| `send_drift_alert(metrics)` | Sends a critical drift alert embed with win rate, EV, Brier score, log loss, and severity color. Username: "The Accountant". |

**Thread safety:** `send_*` methods use `requests.post()` directly (synchronous HTTP). In the `LiveOrchestrator`, they are called from within `asyncio.to_thread()` contexts, so they are inherently thread-safe for the event loop.

---

## 7. The Fuel System (Data Providers & Mining)

---

### Provider Architecture

**File:** `src/data/factory.py` -> `get_market_provider()`

Factory function that reads `DATA_SOURCE` env var and returns the appropriate provider. Supports: `alpaca` (default), `polygon`, `yahoo`.

Note: `src/data/market_provider.py` (the ABC) is **missing from the repository** but is imported by `factory.py`, `polygon_provider.py`, and `yahoo_provider.py`. It should define an abstract `MarketDataProvider` with methods: `get_active_symbols()`, `get_historical_bars()`, `subscribe()`, `run_stream()`.

---

### src/data/alpaca_provider.py -> `AlpacaProvider`

Primary data provider for both historical REST and streaming data.

| Method | Purpose |
|---|---|
| `get_historical_bars(symbol, timeframe_minutes, start, end)` | Fetches OHLCV bars; auto-detects crypto vs stock based on `/` in symbol; returns Polars DataFrame. Uses IEX feed for stocks. |

---

### src/data/polygon_provider.py -> `PolygonDataProvider`

Alternative data provider using Polygon.io API.

| Method | Purpose |
|---|---|
| `get_active_symbols(limit)` | Returns most-active tickers by volume from Polygon snapshot. |
| `get_historical_bars(symbol, timeframe_minutes, start, end)` | Fetches OHLCV via `list_aggs`, converts ms-epoch timestamps to UTC. |
| `subscribe(symbols, callback)` | Registers WebSocket subscriptions on `A.<symbol>` channels. |
| `run_stream()` | Starts the blocking Polygon WebSocket event loop. |

---

### src/data/yahoo_provider.py -> `YahooDataProvider`

Fallback provider using yfinance (no API key required, but 15+ min delayed).

| Method | Purpose |
|---|---|
| `get_historical_bars(symbol, timeframe_minutes, start, end)` | Fetches bars via `yf.download`. |
| `run_stream()` | Blocking poll loop that fetches latest 1-min candle every `poll_interval` seconds. |

---

### src/data/discovery.py -> `DiscoveryService`

Scans the Alpaca universe for in-play tickers.

| Method | Purpose |
|---|---|
| `get_in_play_tickers(min_price, max_price, min_gap_pct, top_n)` | Fetches all active US equity snapshots, filters by price/gap/tradeability, returns top N by volume. |

---

### src/ml/data_miner.py -> `DataMiner`

Bulk historical data fetcher for ML training datasets.

| Method | Purpose |
|---|---|
| `mine_history(symbols, start_date, end_date)` | Fetches 1-min bars in calendar-month chunks with exponential-backoff retry, deduplicates, and writes `{symbol}_1min.parquet` files to `data/raw/`. |
| `_fetch_chunk_with_retry(symbol, start, end)` | Calls `provider.get_historical_bars` with retry logic (3 attempts, doubling backoff). |
| `_month_ranges(start, end)` | Yields `(month_start, month_end)` pairs covering a date range. |

**Default config when run as script:** Fetches `["SPY", "QQQ", "IWM", "NVDA", "AMD", "MSFT", "AAPL"]` from 2020-01-01 to now.

---

### src/data/fetch_training_data.py

Simple script that fetches the last 60 days of 1-min bars for `["SPY", "TSLA", "NVDA", "COIN"]` and writes each to `data/raw/{SYMBOL}_1min.parquet`.

---

### src/data/harvester.py

OOS data collector: fetches the last 7 days of 1-min bars for `["TSLA", "NVDA", "MARA", "COIN", "SMCI"]` and writes `data/oos_bars.parquet`.

---

## 8. The Pit Crew (Pipeline Orchestration & DevOps)

---

### Reinforcement Feedback Loop

The feedback loop is a multi-phase pipeline that detects model drift and triggers automated retraining.

#### Phase 2: Drift Evaluation
**File:** `src/core/feedback_loop.py` -> `DriftEvaluator`

| Method | Purpose |
|---|---|
| `evaluate()` | Calculates win rate, expected value, Brier score, and log loss from resolved OOS trades. |
| `check_drift()` | Returns `(is_drifted, reason)` — drift if Brier > 0.25 or EV < 0 or EV < 0.0005. |
| `trigger_alert(reason)` | Sends drift alert to Discord via `NotificationManager.send_drift_alert()`. |
| `run()` | Full pipeline: evaluate -> print summary -> check drift -> alert if needed. Exit codes: 0=healthy, 1=error, 2=critical drift. |

**Input:** `data/resolved_ledger.csv`

#### Regime Drift Analysis
**File:** `src/analysis/reinforcement_voter.py`

| Function | Purpose |
|---|---|
| `load_evaluation_data()` | Loads `data/evaluation_results.parquet` (Phase 3 output). |
| `calculate_atr_regimes(df)` | Segments trades into Low/Normal/High volatility regimes using `pl.cut()` on ATR percentiles (33rd and 67th). |
| `analyze_regime_drift(df)` | Calculates per-regime: actual win rate, mean Devil conviction, calibration gap (conviction minus win rate). Safety switch triggers if gap > 20%. |
| `generate_drift_report(df, regime_metrics)` | Produces a `DriftReport` with overall stats and per-regime `RegimeMetrics`. |
| `save_drift_report(report)` | Writes `data/drift_report.json` for pipeline orchestration. |

**Exit codes:** 0=healthy, 1=error, 2=safety switch triggered.

#### Phase 5: Automated Retraining ("The Cure")
**File:** `src/core/retrainer.py`

| Function | Purpose |
|---|---|
| `fetch_training_data(client, days_back)` | Fetches 60 days of 1-min bars from Alpaca for `["TSLA", "NVDA", "MARA", "COIN", "SMCI"]`. |
| `engineer_features_and_labels(df)` | Computes TA-Lib indicators (RSI, MACD, BBANDS) and generates targets: Angel (3-bar momentum), Devil (15-bar bracket +0.5%/-0.2%). |
| `generate_time_decay_weights(n_samples, decay_factor)` | Creates exponential time-decay weights (newest=1.0, oldest=0.1) to prevent catastrophic forgetting. |
| `refit_models(df, feature_cols)` | Trains Angel on base features, generates Angel probabilities, trains Devil on base features + angel_prob (meta-labeling). Both use time-decay sample weights. |
| `save_models(angel_model, devil_model)` | Atomic serialization: writes to temp file, then `os.replace()` for zero-downtime swap (safe for live hot-reloader). |

**Output:** `models/angel_latest.pkl`, `models/devil_latest.pkl`

---

### Pipeline Orchestrator
**File:** `run_pipeline.sh`

Bash script that automates the full OOS testing pipeline with colored terminal output:

1. `check_environment` — Loads `.env`, verifies Alpaca credentials.
2. `run_harvester` — Phase 1: `python -m src.data.harvester`.
3. `run_replay` — Phase 2: `python -m src.replay_test`.
4. `run_resolver` — Phase 3: `python -m src.evaluate_performance`.
5. `run_feedback` — Phase 4: `python -m src.core.feedback_loop`.
6. If exit code 2 (critical drift): `run_retrainer` — Phase 5: `python -m src.core.retrainer`.

---

### Entry Points

| File | Purpose | Usage |
|---|---|---|
| `run_live.py` | **Production entry point** for the V3.1 LiveOrchestrator daemon. | `python3 run_live.py` (interactive) or `python3 run_live.py --daemon` (headless/systemd). Accepts `SYMBOLS` env var override. |
| `src/main.py` | Legacy Gen-2 entry point with DiscoveryService and polling loop. | `python src/main.py` — uses `MLStrategy` + `AlpacaProvider` in a synchronous poll-every-minute loop. |
| `main.py` (root) | Legacy Gen-1 entry point using RSIBBands strategy. | `python main.py` — uses `TradingBot` + `OrderManager` + data provider factory. Two-phase: async warmup then blocking stream. |

---

### Docker & Deployment

**File:** `Dockerfile`

- Base image: `python:3.12-slim-bookworm`.
- Compiles TA-Lib 0.4.0 from source.
- Installs Python deps via Pipenv (`--system --deploy`).
- Copies `src/` and `main.py`.
- Runs `python main.py` (legacy Gen-1 entry point).

**File:** `docker-compose.yml`

- Service `trading-bot` with `restart: unless-stopped`.
- Loads `.env` for credentials.
- Mounts `./logs:/app/logs` and `./data:/app/data:Z` (SELinux label).

---

### Test Files

| File | Purpose |
|---|---|
| `test_discord.py` | Smoke test: loads `discord_webhook_url` from `.env` and sends a ping message. |
| `test_speed.py` | Performance benchmarking utility. |
| `tests/test_live_simulation.py` | Live simulation integration tests. |
| `tests/test_strategy_logic.py` | Unit tests for strategy analysis logic. |
| `src/replay_test.py` | OOS replay testing harness (covered in Section 5). |

---

## Data Flow Summary

```
                              LIVE MODE (run_live.py)
                              =======================

    Alpaca CryptoDataStream ──┐
                              ├──> _on_bar() ──> LiveBarAggregator.add_bar()
    Alpaca StockDataStream  ──┘                          │
                                                    bar sealed?
                                                         │ yes
                                                         ▼
                                              FeatureEngineer.compute_indicators()
                                                         │
                                                  ATR Kill Switch
                                                   (natr > 0.5204?)
                                                         │ pass
                                                         ▼
                                              Angel.predict_proba() >= 0.40?
                                                         │ yes
                                                         ▼
                                              Devil.predict_proba() >= 0.50?
                                                         │ yes
                                                         ▼
                                                  Signal(BUY, ...)
                                                         │
                                              SymbolState == FLAT?
                                                         │ yes
                                                         ▼
                                              _submit_bracket_order()
                                              (Market + OTO TP/SL)
                                                         │
    Alpaca TradingStream ────────> _on_trade_update()
                                         │
                                   fill ──> IN_TRADE
                                   sell fill ──> COOLING (5min) ──> FLAT
                                   cancel/reject ──> FLAT
```

```
                           BACKTEST MODE (run_pipeline.sh)
                           ==============================

    harvester.py ──> data/oos_bars.parquet
                          │
    replay_test.py ──> data/signal_ledger.parquet
                          │
    evaluate_performance.py ──> data/evaluation_results.parquet
                          │
    reinforcement_voter.py ──> data/drift_report.json
                          │
    feedback_loop.py ──> exit code 0 (healthy) / 2 (drift)
                          │ if 2
    retrainer.py ──> models/angel_latest.pkl + devil_latest.pkl
                          │
              hot-reload detected by MLStrategy._check_model_updates()
```

---

## File Index

### Root Scripts
| File | Category |
|---|---|
| `run_live.py` | Production launcher (V3.1 LiveOrchestrator) |
| `main.py` | Legacy Gen-1 launcher (RSIBBands + TradingBot) |
| `run_pipeline.sh` | OOS pipeline orchestrator (bash) |
| `backtest_full.py` | Full-year SPY backtest |
| `backtest_ml_strategy.py` | Full backtest with detailed results |
| `backtest_ml_strategy_quick.py` | 2-month quick backtest |
| `backtest_quick.py` | 2-week smoke test |
| `backtest_60.py` | Threshold 0.60 backtest |
| `grid_search_vectorized.py` | Vectorized grid search (fast) |
| `grid_search_backtest.py` | Bar-by-bar grid search (full year) |
| `grid_search_backtest_q1.py` | Bar-by-bar grid search (Q1 only) |
| `grid_search_fast.py` | 2-week grid search |
| `test_discord.py` | Discord webhook smoke test |
| `test_speed.py` | Performance benchmarks |
| `Dockerfile` | Container build definition |
| `docker-compose.yml` | Container orchestration |

### src/execution/
| File | Category |
|---|---|
| `live_orchestrator.py` | Production live trading daemon (SymbolState, SymbolContext, LiveOrchestrator) |

### src/core/
| File | Category |
|---|---|
| `trading_bot.py` | Legacy Gen-1 TradingBot class |
| `order_management.py` | Legacy OrderManager, OrderCalculator, OrderParams |
| `signal.py` | Signal and SignalType data classes |
| `notification_manager.py` | Discord webhook notifications |
| `resolver.py` | Trade resolution (bracket simulation for OOS) |
| `feedback_loop.py` | Drift evaluator (Brier, EV, log loss) |
| `retrainer.py` | Automated model retraining ("The Cure") |
| `ws_stream_simulator.py` | Historical data WebSocket simulator |

### src/ml/
| File | Category |
|---|---|
| `feature_pipeline.py` | FeatureEngineer (TA-Lib indicators + derived features) |
| `train_model.py` | Angel/Devil meta-labeling training pipeline |
| `data_miner.py` | Bulk historical data fetcher |
| `models/` | Directory containing serialized `.joblib` model files |

### src/strategies/
| File | Category |
|---|---|
| `strategy.py` | Strategy ABC |
| `strategy_factory.py` | Strategy registry and factory |
| `concrete_strategies/ml_strategy.py` | MLStrategy (Angel/Devil, hot-reload) |
| `concrete_strategies/rsi_bbands.py` | RSIBBands (2-stage mean reversion) |
| `concrete_strategies/sma_crossover.py` | SMACrossover (reference implementation) |

### src/utils/
| File | Category |
|---|---|
| `bar_aggregator.py` | LiveBarAggregator (clock-aware OHLCV aggregation) |
| `risk_management.py` | Empty placeholder |

### src/analysis/
| File | Category |
|---|---|
| `reinforcement_voter.py` | Regime drift analysis (ATR-based segmentation) |
| `optimize_threshold.py` | Probability threshold sweep optimization |

### src/data/
| File | Category |
|---|---|
| `alpaca_provider.py` | Alpaca REST/streaming data provider |
| `polygon_provider.py` | Polygon.io data provider |
| `yahoo_provider.py` | yfinance data provider (fallback) |
| `factory.py` | Provider factory (`get_market_provider()`) |
| `discovery.py` | DiscoveryService (in-play ticker scanner) |
| `harvester.py` | OOS data harvester (7-day lookback) |
| `fetch_training_data.py` | Training data fetcher (60-day lookback) |

### src/evaluate_performance.py
| File | Category |
|---|---|
| `evaluate_performance.py` | Phase 3: Vectorized backtest with dynamic ATR exits |

### src/replay_test.py
| File | Category |
|---|---|
| `replay_test.py` | Phase 2: OOS replay simulation harness |

---

**End of HAYNES_MANUAL.md**
