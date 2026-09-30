---
type: recon
date: 2026-09-30
time: 14:55 PDT
agent: Antigravity
model: gemini-2.5-pro
trigger: Audit item 3 — Right-edge censoring in forward barrier labels
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: read-only
related:
  - llm_reports/recons/2026-09-29_train-serving-feature-skew-audit.md
  - llm_reports/recons/2026-09-30_promotion-gate-leakage-and-overfitting.md
---

## Context

Item 3 of the architectural seams audit investigates how forward barrier and bracket labels are assigned to the right-edge (most recent $H$ bars) of datasets across the training pipeline, validation gate, and live telemetry grader.

Forward-looking barrier labels require a future horizon $H$ (e.g. `max_hold = 45` bars, or `survival_bars = 5` bars) to determine whether a trade reached its Take-Profit (TP), Stop-Loss (SL), or survived without being stopped out. For bars near the end of a dataset (or near fold boundaries), the future price path is either incomplete or non-existent.

The specific questions investigated:
1. Does the code drop unresolvable right-edge rows, truncate the forward walk, or default them to "no touch" / loss classes (`0`)?
2. Does `FeaturePipeline.clean_data` purge censored rows, or do they leak into model training as artificial losses?
3. How do cross-validation fold boundaries and holdout splits handle boundary purges and walk-forward embargos?
4. Does the live decision grader (`decision_grader.py`) penalize recent live decisions due to right-edge censoring?

## Investigation

### 1. Code Inspection: Bracket Simulation in `_labels.py`

Inspected `src/core/retrainer/_labels.py`:

- **Macro Target (`_compute_devil_targets_atr`, lines 27–77):**
  ```python
  targets = np.zeros(n, dtype=np.int8)
  for i in range(n - 1):
      ...
      for j in range(i + 1, min(i + max_hold + 1, n)):
          if symbol[j] != symbol[i]:
              break
          if low[j] <= sl_price:
              targets[i] = 0
              break
          if high[j] >= tp_price:
              targets[i] = 1
              break
      # If loop completes without break -> timeout -> 0 (already default)
  return targets
  ```
  If row $i$ is within $H = 45$ bars of the end of the array ($n$) or symbol boundary, the loop over $j$ terminates early when $j = n$. If neither SL nor TP was hit in that truncated sub-window, the loop completes without setting `targets[i] = 1`. It leaves `targets[i] = 0`.
  The code treats this as a "timeout", but it is actually **right-edge censoring**. A trade with 5 bars remaining that was trending strongly towards TP is falsely labeled as `0` (loss/timeout). The returned array is a dense `int8` with zero nulls, completely concealing which rows were censored.

- **Survival Target (`_compute_devil_survival_target`, lines 80–140):**
  ```python
  targets = np.zeros(n, dtype=np.int8)
  for i in range(n - 1):
      if i + survival_bars >= n or symbol[i + survival_bars] != symbol[i]:
          continue  # leaves targets[i] = 0 (default)
      ...
      for j in range(i + 1, min(i + survival_bars + 1, n)):
          if low[j] <= sl_price:
              survived = False
              break
      targets[i] = np.int8(1) if survived else np.int8(0)
  return targets
  ```
  The survival target defines `1` as survived and `0` as stopped out.
  For the last `survival_bars` rows ($i + \text{survival\_bars} \ge n$), the code executes `continue`, leaving `targets[i] = 0`.
  **Unresolvable rows are explicitly assigned `0` (stopped out).** Even if price rose 100 pips without touching SL, the lack of future bars causes it to be labeled as a stop-out.

### 2. Leakage into Model Training via `clean_data`

Inspected `src/core/retrainer/_features.py:350-362`:
```python
cols = list(feature_cols)
df = FeaturePipeline.clean_data(
    df, feature_cols=cols + ["angel_target", "devil_target"]
)
```
- In `_features.py:180-192`, `angel_target` is generated via:
  ```python
  pl.col("close").shift(-3).over("symbol") > pl.col("close") + _angel_mult * ...
  ```
  `shift(-3)` produces `null` for exactly the last 3 rows of each symbol.
- `devil_target` contains no nulls (it is dense `int8`, where unresolvable tails are `0`).
- When `clean_data` runs on `["angel_target", "devil_target"]`, it drops only rows where `angel_target` is null (the last 3 rows).
- Because `survival_bars = 5`, rows $N-5$ and $N-4$ have valid `angel_target` labels, but their `devil_target` was forced to `0` by right-edge censoring.
- **Result:** Rows $N-5$ and $N-4$ are retained in the training dataset and presented to LightGBM as confirmed stop-outs (`devil_target = 0`), poisoning the training set with false negatives.

### 3. Boundary Purge Coverage and CV Fold Leakage

Inspected `_purge_boundary_tail` in `src/core/retrainer/_gate.py:405-438` and `src/core/retrainer/_pipeline.py:161-177`:
- In `_pipeline.py:168-175`, `_tail_cutoff_by_symbol(remainder_raw, max_hold)` successfully purges the last 45 bars of each symbol from the `remainder` training frame before model training.
- In `_gate.py:440-475`, `_score_artifact_holdout` successfully purges the last 45 bars of each symbol from the `holdout` frame before scoring.
- **However, inside `validate_candidate` (Walk-Forward CV, `src/core/retrainer/_gate.py:650-700`):**
  - Expanding folds (Folds 1, 2, 3) are split *after* `_compute_devil_targets_atr` has run across the entire remainder.
  - When splitting `train_df = df.filter(pl.col("timestamp") < train_cutoff)` and `val_df = df.filter((pl.col("timestamp") >= train_cutoff) & (pl.col("timestamp") < val_cutoff))`, **no boundary purge or embargo is applied**.
  - A trade in `train_df` occurring 10 bars before `train_cutoff` uses a 45-bar forward walk that extends 35 bars into `val_df`. Price action from the validation set directly determines training labels.
  - In `val_df` for Fold 1 and Fold 2, trades occurring within 45 bars of the validation cutoff walk forward into subsequent folds.

### 4. Live Telemetry Grader Degradation (`scripts/run_decision_report.py`)

Inspected `scripts/run_decision_report.py:75-95` and `src/analysis/decision_grader.py:145-175`:
- `run_decision_report.py` fetches the latest market candles (`load_basket`) and labels them with `_compute_devil_targets_atr(tagged, SL_MULT, TP_MULT, MAX_HOLD)`.
- The resulting `tagged` frame contains bars up to the very latest market candle (e.g. Friday market close).
- For every instrument, the last 45 bars (11.25 hours on M15) have truncated forward walks and are assigned `won = 0`.
- In `decision_grader.py:158-164`:
  ```python
  joined = d.join(g, on=["symbol", "_key"], how="inner").drop("_key")
  ```
  The author believed unresolvable decisions would be omitted (`"decisions with no matching graded bar are dropped"`). But `g` *does* match because `tagged` retained the recent bars with `won = 0`.
- **Empirical Confirmation:** In `logs/decision_report_2026-09-27.txt` and `logs/graded_decisions.parquet`, out of 22,226 decisions recorded, 22,214 were graded. Decisions recorded between 19:00 and 20:00 on Friday (which had at most 2 bars before market close) were matched and graded as `won = 0`.
- This right-edge censoring severely depressed observed win rates for high-conviction threshold bands in live monitoring (e.g. threshold 0.45 recorded 1 win / 17 decisions = 5.9%, threshold 0.50 recorded 0 wins / 7 decisions = 0.0%).

### 5. Cross-Symbol Leakage in Legacy Target Generator

Inspected `src/ml/targets/v3_targets.py:38-48`:
```python
future_close = pl.col("close").shift(-self.lookahead)
```
`V3DirectionalTarget` shifts `close` globally without `.over("symbol")`. In multi-symbol DataFrames, the final 15 bars of each symbol shift forward into the first 15 bars of the next symbol, calculating returns across completely different currency pairs (e.g. AUD_JPY close compared against EUR_JPY close).

## Findings / Changes

### Finding 1: Telemetry Grader False-Loss Contamination (Severity: High)
`scripts/run_decision_report.py` and `decision_grader.py` fail to purge the trailing `LOOKAHEAD_BARS` (45 bars / 11.25 hours) from the market bar frame. Every live decision evaluated in the final 11.25 hours of fetched market data is assigned `won = 0` unless it hit TP in the truncated window. This artificially degrades reported live performance, reporting 0% win rates on high-confidence signals and corrupting calibration curves.

### Finding 2: Walk-Forward CV Horizon Leakage (Severity: Medium)
In `src/core/retrainer/_gate.py:validate_candidate`, expanding cross-validation folds are sliced without a 45-bar embargo between `train_df` and `val_df`. Training samples within 45 bars of the fold cutoff evaluate forward price action occurring inside the validation window, violating strict out-of-sample isolation.

### Finding 3: Survival Target Poisoning at Symbol/Data Boundaries (Severity: Medium)
In `src/core/retrainer/_labels.py:_compute_devil_survival_target`, the final `survival_bars` rows are defaulted to `0` (stopped out) instead of `null`. Because `angel_target` only purges 3 rows via `shift(-3)`, rows $N-5$ and $N-4$ survive `clean_data` and enter training labeled as confirmed stop-outs.

### Finding 4: Incomplete Walk Truncation in Macro Bracket Walk (Severity: Low)
`_compute_devil_targets_atr` returns a dense `int8` array where right-edge truncated walks and legitimate 45-bar timeouts are indistinguishable (both `0`). While `_purge_boundary_tail` removes these at the outer boundaries of `remainder` and `holdout`, any downstream script or diagnostic calling `_compute_devil_targets_atr` directly inherits right-edge censoring bias unless an explicit tail purge is applied.

### Finding 5: Cross-Symbol Shift in `v3_targets.py` (Severity: Low)
`src/ml/targets/v3_targets.py` omits `.over("symbol")` during forward shift, leaking price across symbol boundaries in legacy pipeline runs.

## Verification

1. **Empirical Bracket Truncation**: Verified on synthetic price data that a steady upward trend hitting TP at bar +4 is assigned `targets = 0` if only 3 bars remain before data end.
2. **Survival Target Boundary Bias**: Verified on synthetic data with zero SL hits that `_compute_devil_survival_target` assigns `[0, 0, 0, 0, 0]` to the final 5 bars.
3. **Telemetry Grader Tail Contamination**: Verified `logs/graded_decisions.parquet` contains decisions up to `2026-09-25 20:00:00` graded as `won = 0` against a price series ending at Friday close, confirming that the grader evaluates incomplete walks as losses.

## Risk & follow-ups

1. **Fix `decision_grader.py` and `run_decision_report.py`**:
   Before joining live decisions to `graded_bars`, drop all bars whose timestamp is within `LOOKAHEAD_BARS` of that symbol's maximum available timestamp in the fetched bar cache.
2. **Add Embargo to `validate_candidate`**:
   In `src/core/retrainer/_gate.py`, insert a purge/embargo of `max_hold` bars before each `train_cutoff` to eliminate forward label leakage from train into validation folds.
3. **Fix `_compute_devil_survival_target`**:
   Return `np.nan` or a nullable integer series for rows where lookahead is insufficient, and ensure `clean_data` drops them rather than training on false stop-out labels.

## Files read

- `src/core/retrainer/_labels.py`
- `src/core/retrainer/_features.py`
- `src/core/retrainer/_gate.py`
- `src/core/retrainer/_pipeline.py`
- `src/ml/barriers/labels.py`
- `src/ml/barriers/estimator.py`
- `src/ml/targets/v3_targets.py`
- `src/analysis/decision_grader.py`
- `scripts/run_decision_report.py`
- `scripts/evaluate_barriers.py`
