# System Architecture: Universal Scalper V3.4

This document details the internal architecture of the Gen-3 "Universal Scalper" system, focusing on its asynchronous execution model, Machine Learning inference pipeline, and automated DevOps feedback loops.

---

## 1. Core Execution Engine: LiveOrchestrator

The `LiveOrchestrator` is the central state machine and async daemon that drives the live trading bot. It supports dual-stream operation, multiplexing market data from both equities and crypto into a single `asyncio` event loop.

### Lifecycle State Machine (`SymbolState`)
Each symbol follows a strict state progression to prevent race conditions and double-firing:
*   **FLAT:** No active position; eligible for new signals.
*   **PENDING:** Entry order submitted to Alpaca; awaiting fill confirmation.
*   **IN_TRADE:** Position filled; actively monitored by the Universal Watchdog.
*   **PENDING_EXIT:** Exit condition met; market sell order submitted; awaiting confirmation.
*   **COOLING:** 5-minute mandatory cooldown after a trade resolves to prevent over-trading in volatile regimes.

---

## 2. Data Pipeline & Feature Engineering

The system uses a 18-feature vector designed to capture both momentum and market microstructure.

### Optimized Inference Path
To maintain high performance, `_run_inference` implements a **Cold/Warm Path** logic:
*   **Cold Path:** Triggered every 5 minutes. Performs a full resample of 400 1-minute bars to compute Higher-Timeframe (5m) features (`htf_rsi`, `htf_trend_agreement`, etc.).
*   **Warm Path:** For every 1-minute bar between cold runs, the system injects cached HTF scalars using `pl.lit()`, drastically reducing CPU overhead.

### Feature Set (18 Dimensions)
1.  **Technicals:** RSI-14, PPO, NATR-14, Bollinger %B, BB Width.
2.  **Trend:** Price/SMA-50 ratio, Log Returns, Distance from SMA-50, Relative Volume.
3.  **HTF (5m):** HTF RSI, Trend Agreement (+1/-1), HTF Rel Vol, HTF %B.
4.  **Microstructure:** Range Coil (10-bar), Bar Body %, Upper Wick %, Lower Wick %.

---

## 3. ML Strategy: Angel/Devil Meta-Labeling

The system employs a two-stage meta-labeling architecture using Random Forest classifiers.

### Stage 1: The Angel (Proposal)
The Angel model is trained for **High Recall**. Its job is to identify all potentially profitable setups based on 3-bar momentum and ATR-relative targets.

### Stage 2: The Devil (Conviction)
The Devil model is the primary filter, trained for **High Precision**. 
*   **Input:** Base features + Angel probability.
*   **Target:** Trained on a 5-bar **Survival Target** (Does the price stay above SL for at least 5 bars?).
*   **Execution:** Only fires a BUY signal if `angel_prob >= 0.40` AND `devil_prob >= threshold` (dynamically loaded from `models/threshold.json`).

---

## 4. Transmission & Risk Management

### Universal Software Watchdog
Unlike legacy bracket orders, V3.4 uses a central `_universal_watchdog_loop` that polls all active positions every 1 second. This allows the system to:
1.  Manage fractional equity shares (which Alpaca server-side brackets often reject).
2.  Provide unified exit logic for both crypto and equity streams.
3.  Implement a volatility "Kill Switch" that prevents entry if NATR-14 exceeds the `ATR_KILL_SWITCH_THRESHOLD`.

### Dynamic Brackets
Exits are calculated at the time of signal based on market volatility:
*   **Stop-Loss (SL):** `entry - (0.5 * ATR_abs)`
*   **Take-Profit (TP):** `entry + (3.0 * ATR_abs)`

---

## 5. The Pit Crew: DevOps & "The Cure V2"

The system includes a fully automated retraining pipeline (`run_pipeline.sh`) to combat model drift.

1.  **Drift Detection:** `DriftEvaluator` monitors Brier scores and Expected Value (EV) from recent OOS trades.
2.  **Retraining:** `Retrainer.py` fetches 60 days of fresh data and refits models using exponential time-decay weights.
3.  **Validation Gate:** New models must pass a 3-fold cross-validation gate requiring:
    *   **Profit Factor > 1.2**
    *   **Expected Value (EV) > 0.0005**
    *   **Brier Score < 0.30**
4.  **Atomic Promotion:** Validated models are swapped into the `models/` directory using `os.replace()` for zero-downtime hot-reloading by the live bot.

---
*Architecture Mapping: Universal Scalper V3.4*
