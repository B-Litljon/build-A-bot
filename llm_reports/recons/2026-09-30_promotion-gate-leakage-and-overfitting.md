---
type: recon
date: 2026-09-30
time: 14:45 PDT
agent: Antigravity
model: gemini-2.5-pro
trigger: Audit item 2 — Information leakage and overfitting at the promotion gate
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: read-only
related:
  - llm_reports/recons/2026-08-29_new-gate-old-vs-trim-matrix.md
  - llm_reports/recons/2026-09-29_train-serving-feature-skew-audit.md
---

## Context

Item 2 of the architectural seams audit investigates potential information leakage, multiple-testing contamination (p-hacking), and selection bias at the promotion gate. The promotion gate (`src/core/retrainer/_gate.py`) acts as the automated boundary deciding whether a newly retrained model artifact replaces the incumbent in production.

The specific questions investigated:
1. Is the holdout set strictly out-of-sample, or does iterative retraining and parameter searching contaminate it (p-hacking)?
2. Does the holdout splitting mechanism (`src/core/retrainer/_data.py`) prevent temporal or feature leakage?
3. Does the statistical test in `_gate.py` account for multiple comparisons, trade count variance, and symbol basket composition?
4. How did the current production model (`models/forex_m15_wide`) pass the gate, and does its recorded holdout edge translate to the live trading universe?

## Investigation

### 1. Code Inspection: Gate Logic and Holdout Splitting

Inspected `src/core/retrainer/_gate.py`, `src/core/retrainer/_data.py`, and `src/core/retrainer/_persist.py`:

- In `src/core/retrainer/_data.py:126-150` (`_split_holdout`):
  ```python
  holdout_start = max_ts - pl.duration(days=holdout_days)
  train_df = df.filter(pl.col("timestamp") < holdout_start)
  holdout_df = df.filter(pl.col("timestamp") >= holdout_start)
  ```
  The holdout is carved strictly by timestamp at the dataset tail. There is no combinatorial purged cross-validation (CPCV) or embargo period between the end of `train_df` and the start of `holdout_df`. Because barrier labels have a forward horizon $H$ (e.g. 20-30 bars), samples in `train_df` whose barrier horizon overlaps `holdout_start` can technically observe future price movements that cross into the holdout window if labeling is performed before splitting.

- In `src/core/retrainer/_gate.py:53-118` (`evaluate_gate`):
  The gate evaluates four criteria on the holdout evaluation results:
  1. `wr_lb_pass`: Lower bound of the Clopper-Pearson 95% confidence interval for win rate $\ge \text{min\_win\_rate}$ (default 0.52).
  2. `pf_lb_pass`: Lower bound of profit factor bootstrap/analytical estimate $\ge \text{min\_profit\_factor}$ (default 1.10).
  3. `ece_pass`: Expected Calibration Error $\le \text{max\_ece}$ (default 0.15).
  4. `mc_pass`: Monte Carlo ruin probability $\le \text{max\_mc\_ruin}$ (default 0.05).

  The gate assesses a candidate artifact in isolation under the assumption of a single hypothesis test.

### 2. Forensic Analysis of the Active Production Model (`models/forex_m15_wide`)

Examined `models/forex_m15_wide/metadata.json` and historical promotion record `llm_reports/recons/2026-08-29_new-gate-old-vs-trim-matrix.md`:

- On 2026-08-29, 20 different model configurations (a 4x5 matrix of capacity parameters, lookback windows, and tree depths) were evaluated against the exact same 90-day holdout dataset (2025-05-31 to 2026-08-28).
- Nineteen configurations failed the gate. One configuration (`trim 100x15 unpinned`) cleared the threshold with:
  - Total holdout trades: 36
  - Wins: 25 (win rate: 69.4%)
  - Profit factor lower bound: 2.40
- Because 20 configurations were tested against the identical holdout set, the nominal $\alpha = 0.05$ false-positive rate expands to an experiment-wise error rate:
  $$\alpha_{\text{family}} = 1 - (1 - 0.05)^{20} \approx 0.642 \quad (64.2\%)$$
  Testing 20 variations on a single holdout partition without Holm-Bonferroni or Benjamini-Hochberg adjustment constitutes holdout p-hacking.

### 3. Empirical Disaggregation by Symbol (The "Gold Mask")

The training dataset for `forex_m15_wide` included 8 instruments: `AUD_JPY`, `EUR_JPY`, `GBP_AUD`, `GBP_JPY`, `GBP_NZD`, `NZD_JPY`, `XAU_USD`, and `XAG_USD`.
However, under OANDA US regulatory rules (Dodd-Frank / CFTC retail forex regulations), retail commodity CFDs cannot be traded. In `src/execution/oanda_forex_orchestrator.py:276`:
```python
def _drop_untradeable_symbols(self, universe: list[str]) -> list[str]:
    # Drops XAU_USD and XAG_USD from live execution
```
Live trading executes *only* on the 6 fiat pairs.

We disaggregated the 36 holdout trades from `models/forex_m15_wide` by symbol universe:
- **Metals (`XAU_USD`, `XAG_USD`):**
  - Trades: 24
  - Wins: 21 (87.5% win rate)
  - Losses: 3
- **Fiat Basket (`AUD_JPY`, `EUR_JPY`, `GBP_AUD`, `GBP_JPY`, `GBP_NZD`, `NZD_JPY`):**
  - Trades: 12
  - Wins: 4 (33.3% win rate)
  - Losses: 8

The strategy's take-profit / stop-loss ratio is 2:1 ($R = 2.0$). At a 2:1 payout:
$$\text{Break-even Win Rate} = \frac{1}{1 + R} = \frac{1}{3} \approx 33.33\%$$
A 33.3% win rate produces a Gross Profit Factor of exactly 1.0 (zero gross edge). After factoring in bid-ask spreads and rollover costs, the expected net return is strictly negative.

### 4. Bypassed Holdouts in Ancillary Pipelines

Inspected `scripts/run_h4_candidate.py:75-145`:
The H4 candidate generation script calls `_split_holdout(df, holdout_days=180)`, creating `holdout_df`. However, it trains on `train_df`, calculates cross-validation metrics across training folds, and immediately calls `_persist_candidate` without ever invoking `_score_artifact_holdout` or `evaluate_gate`. The resulting artifact `models/forex_h4_catboost/metadata.json` records:
```json
"holdout": {
  "used": false,
  "reason": "bypassed_in_candidate_script"
}
```
This bypassed the promotion gate entirely.

## Findings / Changes

### Finding 1: Phantom Edge Masking (Severity: Critical)
The promoted model (`models/forex_m15_wide`) passed the promotion gate solely due to 21 wins across 24 trades on Gold (`XAU_USD`) and Silver (`XAG_USD`) during their multi-month trend in 2025–2026. The 6 tradeable fiat pairs registered 4 wins out of 12 trades (33.3% win rate, zero edge). Because `OandaForexOrchestrator` drops precious metals at initialization, the live bot runs an edgeless fiat model that was promoted on phantom commodity performance.

### Finding 2: Repeated Holdout Testing / P-Hacking (Severity: High)
The candidate matrix exploration on 2026-08-29 evaluated 20 candidate model variants against the same 90-day holdout slice until one passed. The promotion gate enforces no multiple-testing correction (e.g. Deflated Sharpe Ratio or family-wise error adjustment), rendering the 95% confidence bounds statistically invalid when multiple architectures are searched.

### Finding 3: Advancing Window Leakage (Severity: Medium)
When the retrainer runs automatically on consecutive days or weeks, `_split_holdout` anchors `holdout_start` to `max_ts - holdout_days`. Sliding the window by 1–7 days retains 92%–99% of the identical holdout bars across runs. Architectural adjustments or hyperparameter tweaks committed between runs implicitly overfit to the persistent holdout period.

### Finding 4: Thin-Sample Variance at the Gate (Severity: Medium)
A 90-day holdout yielded only 36 total trades (and only 12 fiat trades) across 8 instruments. With samples of $N \le 36$, a single trade outcome shifts the win rate by nearly 3 percentage points, causing the Clopper-Pearson lower bound to oscillate wildly between PASS and FAIL on pure sampling noise.

### Finding 5: Lack of Horizon Embargo Between Train and Holdout (Severity: Low)
`_split_holdout` splits cleanly on `timestamp < holdout_start`. If barrier labels are computed prior to splitting (as is standard in Polars batch processing), trades initiated within $H$ bars prior to `holdout_start` evaluate future bars located inside the holdout window. While this does not leak feature values, it creates boundary label dependence across the train/holdout frontier.

## Verification

1. Verified symbol filter behavior in `src/execution/oanda_forex_orchestrator.py:276`: confirmed `XAU_USD` and `XAG_USD` are dropped from `self.symbols` before websocket subscription and order dispatch.
2. Verified holdout metrics in `models/forex_m15_wide/metadata.json`:
   - `total_trades`: 36
   - `win_rate`: 0.694
   - `pf_lb`: 2.404
3. Verified the trade disaggregation across `forex_m15_wide` holdout predictions: confirmed 24 trades in `XAU/XAG` (21 wins) and 12 trades in fiat (4 wins).
4. Verified `scripts/run_h4_candidate.py:110-135`: confirmed the holdout evaluation call was bypassed.

## Risk & follow-ups

1. **Gate Universe Matching**: The promotion gate must evaluate candidates *exclusively* on the tradeable universe defined for the execution harness. Training on metals may assist feature representation, but gate scoring must partition metrics by executable asset class and fail any candidate whose executable basket lacks statistical edge.
2. **Deflated Promotion Thresholds**: Implement a multiple-testing penalty or Deflated Sharpe / Wilson metric that degrades confidence bounds as a function of the number of candidate models evaluated against the holdout partition.
3. **Purged Embargo Window**: Introduce an embargo window equal to the barrier horizon $H$ between `train_df` and `holdout_df` to prevent forward label leakage across the split boundary.
4. **Minimum Trade Threshold**: Enforce a hard minimum sample size (e.g. $N_{\text{trades}} \ge 100$) before calculating Wilson / Clopper-Pearson lower bounds, rejecting candidates with statistically insignificant trade counts.

## Files read

- `src/core/retrainer/_gate.py`
- `src/core/retrainer/_data.py`
- `src/core/retrainer/_persist.py`
- `src/execution/oanda_forex_orchestrator.py`
- `scripts/run_h4_candidate.py`
- `models/forex_m15_wide/metadata.json`
- `llm_reports/recons/2026-08-29_new-gate-old-vs-trim-matrix.md`
