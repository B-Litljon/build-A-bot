---
type: recon
date: 2026-09-29
time: 23:26 PDT
agent: Gemini 3.8 Flash
model: gemini-3-8-flash
trigger: "Investigating train-serving feature skew across vectorized batch frames and live stream execution"
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: read-only
related:
  - handoffs/2026-07-01_feature-desync-rebuttal.md
  - handoffs/2026-09-14_barrier-seam-status.md
  - handoffs/2026-09-15_stream-bar-pipeline-brief.md
files_touched: []
---

## Context
Investigating Item 1 from the silent bug audit: Train-Serving Feature Skew (The Batch-Stream Seam). Specifically, verifying whether vectorized feature transformations computed across historical Polars DataFrames in retraining produce bit-for-bit identical vectors to those generated incrementally during live execution in `oanda_forex_orchestrator.py` / `ml_strategy.py`, examining differences in warmup window lengths, rolling indicator initializations, boundary alignment, and tick aggregation.

## Investigation
Traced the feature extraction and bar processing pipeline end-to-end across batch training (`src/core/retrainer/_features.py`, `src/ml/features/v3_features.py`, `src/ml/feature_pipeline.py`) and live execution (`src/execution/oanda_forex_orchestrator.py`, `src/strategies/concrete_strategies/ml_strategy.py`, `src/data/oanda_provider.py`, `src/utils/bar_aggregator.py`).

1. **Multi-symbol volume cross-contamination:** Inspected `V3HTFFeatures.generate()` in `src/ml/features/v3_features.py:444-450`. Tested rolling volume computation with synthetic multi-symbol frames with distinct volume levels.
2. **Stream vs. REST bar discrepancies:** Extracted 121 consecutive bars from the active soak telemetry (`logs/events-2026-09-29.jsonl` and `logs/events-2026-09-30.jsonl`) and queried OANDA REST candles (`get_historical_bars`) for the identical timestamps across all configured symbols (`GBP_JPY`, `AUD_JPY`, `EUR_JPY`, `NZD_JPY`, `GBP_AUD`, `GBP_NZD`).
3. **Warmup window and indicator lookback sensitivity:** Evaluated feature vectors produced by `FeaturePipeline` across 4,000 real OANDA M15 bars comparing full-dataset batch calculations against rolling buffers of size 260, 279, and 520 bars. Measured numerical convergence on Wilder smoothers (`rsi_14`, `natr_14`, `htf_rsi_14`) and rolling medians (`cost_ratio`). Scored the promoted artifact `models/forex_m15_wide/angel_latest.pkl` over 500 consecutive bars across batch and buffer modes.
4. **Buffer start timestamp alignment:** Tested `group_by_dynamic("timestamp", every="1h")` under shifting buffer head offsets (:00, :15, :30, :45).
5. **Dormant aggregator inspection:** Traced codebase references to `LiveBarAggregator` in `src/utils/bar_aggregator.py`.

## Findings / Changes

### Finding 1: Multi-Symbol Volume Leakage in `V3HTFFeatures` (High Severity)
In `src/ml/features/v3_features.py:444-450`, `htf_vol_rel` is computed as:
```python
htf_bars = htf_bars.with_columns(
    (pl.col("htf_volume") / pl.col("htf_volume").rolling_mean(window_size=20))
    .fill_nan(1.0)
    .fill_null(1.0)
    .alias("htf_vol_rel")
)
```
In batch training, `htf_bars` concatenates all symbols vertically. Because `rolling_mean(window_size=20)` lacks `.over("symbol")`, the first 19 HTF bars (19 to 76 base bars) of every symbol after the first compute their rolling volume mean over the preceding symbol's volume. On a synthetic test where Symbol A had volume 1000 and Symbol B had volume 10:
- Combined batch training mode: `htf_vol_rel` evaluated to `0.010953`.
- Isolated live execution mode: `htf_vol_rel` evaluated to `1.0` (a 100x discrepancy).
In live execution, `MLStrategy` receives a single symbol's buffer, so live streaming never suffers this cross-symbol bleed.

### Finding 2: Live Stream Tick Aggregation vs. REST Candles (Medium Severity)
Empirically compared live stream closes recorded in `logs/events-2026-09-29.jsonl` and `logs/events-2026-09-30.jsonl` against OANDA REST candles:
- GBP_JPY: 96 / 121 bars (79.3%) differed (max diff: 0.0050).
- AUD_JPY: 66 / 121 bars (54.5%) differed (max diff: 0.0035).
- EUR_JPY: 66 / 121 bars (54.5%) differed (max diff: 0.0045).
- NZD_JPY: 60 / 121 bars (49.6%) differed (max diff: 0.0040).
- GBP_AUD: 9 / 121 bars (7.4%) differed (max diff: 0.000060).
- GBP_NZD: 9 / 121 bars (7.4%) differed (max diff: 0.000095).

Root causes:
1. `src/data/oanda_provider.py:238` computes `mid = (bid + ask) / 2.0`, retaining unrounded half-pipettes (e.g. 178.3485), whereas OANDA REST candles round to 3 decimal places (178.348).
2. Local aggregation in `_handle_tick` only flushes a completed candle upon arrival of the first tick of the subsequent epoch. Dropped or throttled pricing stream ticks produce slightly truncated high/low extremes and different tick volume counts relative to server-side candles.

### Finding 3: Warmup Lookback and Rolling Medians (Medium Severity)
- For M15 with 1h HTF, 260 M15 bars yield 65 1-hour bars. Wilder smoothers on 1h bars (`htf_rsi_14`) do not fully decay over 65 bars, leaving an error of up to 0.22 RSI points between batch (46.608) and live buffer (46.800).
- In `V3CostFeatures` (`src/ml/features/v3_features.py:327`), `baseline = natr_14.rolling_median(window_size=260)`. In a buffer of 260-279 bars, leading NaNs in `natr_14` (from TA-Lib lookback) shift the median by ~0.00026, causing a 2% to 4% relative discrepancy in `cost_ratio`.
- Model resilience: The active production model `models/forex_m15_wide/angel_latest.pkl` relies on 17 features that exclude `htf_rsi_14` and `cost_ratio`. Across 500 consecutive test bars, `angel_latest.pkl` predictions matched batch bit-for-bit (`max diff = 0.000000`, 0 decision flips). Activating `cost_ratio` or `htf_rsi_14` would surface this drift.

### Finding 4: Buffer Head Timestamp Alignment in HTF Resampling (Low Severity)
`V3HTFFeatures.generate()` runs `group_by_dynamic("timestamp", every="1h")`. If the oldest bar in the rolling buffer starts at an off-hour minute (e.g. :15, :30, :45), the first 1-hour bar contains only 1 to 3 15-minute bars. Slicing at :45 vs :00 shifted `htf_rsi_14` from 46.5094 to 46.3478.

### Finding 5: `LiveBarAggregator` is Dormant (Low Severity)
`LiveBarAggregator` in `src/utils/bar_aggregator.py` is not wired into `run_oanda.py` or `OandaForexOrchestrator`. Its `_forward_fill_gaps` method injects flat candles (`volume=0, open=high=low=close=last_close`), which would distort NATR rolling medians and cause `Inf` divisions if ever integrated.

## Verification
- Verified multi-symbol volume leakage via script running `V3HTFFeatures.generate()` on concatenated vs isolated dataframes.
- Verified REST candle discrepancies by querying OANDA practice API via `OandaMarketProvider.get_historical_bars()` and matching against `logs/events-2026-09-29.jsonl`.
- Verified batch vs buffer model scoring by running `FeaturePipeline` and `angel_latest.pkl` over 500 consecutive bars from `data/cache/ab_catboost/EUR_JPY_M15_60d_20260908.parquet`.
- Ran unit test suite: `python -m unittest tests/test_lab_frames.py` (5 tests passed).

## Risk & follow-ups
- Fix `V3HTFFeatures`: add `.over("symbol")` to `rolling_mean(window_size=20)` on `htf_volume` in `src/ml/features/v3_features.py:446`.
- Document or normalize mid-price half-pipette rounding in `OandaMarketProvider._handle_tick` to match REST candle precision.

## Files touched
_n/a — read-only recon_

Examined:
- `src/core/retrainer/_features.py`
- `src/ml/features/v3_features.py`
- `src/ml/feature_pipeline.py`
- `src/execution/oanda_forex_orchestrator.py`
- `src/strategies/concrete_strategies/ml_strategy.py`
- `src/data/oanda_provider.py`
- `src/utils/bar_aggregator.py`
- `run_oanda.py`
- `tests/test_lab_frames.py`
