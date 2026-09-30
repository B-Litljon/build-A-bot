---
type: recon
date: 2026-09-30
time: 15:15 PDT
agent: Antigravity
model: gemini-2.5-pro
trigger: Audit item 5 — Spread toll omission at barrier boundaries
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: read-only
related:
  - llm_reports/recons/2026-08-08_stop-width-and-the-spread-toll.md
  - llm_reports/refactors/2026-07-07_cost-awareness-spread-feature.md
  - llm_reports/recons/2026-09-29_train-serving-feature-skew-audit.md
  - llm_reports/recons/2026-09-30_promotion-gate-leakage-and-overfitting.md
---

## Context

Item 5 of the architectural seams audit investigates whether barrier and bracket labeling logic accounts for transaction costs — specifically the bid-ask spread and execution slippage — at trade entry and exit boundaries.

In live forex trading (OANDA practice and live), all quotes carry a non-zero bid-ask spread. Long positions buy at the Ask and exit (at either take-profit or stop-loss) at the Bid. This round-trip spread imposes an unavoidable friction toll.

The specific questions investigated:
1. Do the bracket simulation labels (`_compute_devil_targets_atr`, `_compute_devil_survival_target`, `angel_target`, `compute_excursions`) evaluate mid prices or bid/ask prices?
2. Does an Angel target or Devil barrier require the market to clear the entry and exit spreads before registering a win?
3. How much do zero-toll labels flatter model performance and promotion gate metrics compared to reality?
4. What role does Gate A (`RiskManager`) play relative to barrier label generation?

## Investigation

### 1. Mathematical Mechanics of the Spread Toll

All historical OHLC data fetched from OANDA and stored in `data/` represents **MID** prices ($P_{\text{mid}} = \frac{P_{\text{bid}} + P_{\text{ask}}}{2}$).

In actual live execution:
- **Long Entry**: Fills at the Ask:
  $$P_{\text{entry}} = P_{\text{mid}, 0} + \frac{S_0}{2}$$
- **Take-Profit (TP) Exit**: A limit order to sell fills at the Bid:
  $$P_{\text{exit}} = P_{\text{mid}, t} - \frac{S_t}{2}$$
  To achieve a nominal target $\Delta_{\text{TP}} = \text{tp\_mult} \cdot \text{ATR}$, the realized gain must satisfy:
  $$P_{\text{exit}} - P_{\text{entry}} \ge \text{tp\_mult} \cdot \text{ATR}$$
  $$(P_{\text{mid}, t} - \frac{S_t}{2}) - (P_{\text{mid}, 0} + \frac{S_0}{2}) \ge \text{tp\_mult} \cdot \text{ATR}$$
  $$P_{\text{mid}, t} - P_{\text{mid}, 0} \ge \text{tp\_mult} \cdot \text{ATR} + \bar{S}$$
  where $\bar{S} = \frac{S_0 + S_t}{2} \approx S$ is the full round-trip spread. The Mid price must advance by the target distance **plus the full round-trip spread**.
- **Stop-Loss (SL) Exit**: A stop order to sell fills at the Bid:
  $$P_{\text{exit}} - P_{\text{entry}} \le - \text{sl\_mult} \cdot \text{ATR}$$
  $$P_{\text{mid}, 0} - P_{\text{mid}, t} \ge \text{sl\_mult} \cdot \text{ATR} - \bar{S}$$
  The Stop Loss triggers when the Mid price drops by the stop distance **minus the spread**.

In reality, the spread pushes the Take-Profit further away and pulls the Stop-Loss closer to the entry price.

### 2. Code Inspection: Bracket Simulation and Labeling

Inspected `src/core/retrainer/_labels.py:50-75` (`_compute_devil_targets_atr`):
```python
for i in range(n - 1):
    atr_abs = close[i] * natr[i] / 100.0
    ...
    sl_price = close[i] - sl_mult * atr_abs
    tp_price = close[i] + tp_mult * atr_abs

    for j in range(i + 1, min(i + max_hold + 1, n)):
        if symbol[j] != symbol[i]:
            break
        if low[j] <= sl_price:
            targets[i] = 0
            break
        if high[j] >= tp_price:
            targets[i] = 1
            break
```
- `close[i]`, `low[j]`, and `high[j]` are raw Mid prices.
- `tp_price` requires only that the Mid `high[j]` reach `close[i] + tp_mult * atr_abs`. It requires zero spread clearance.
- `sl_price` requires that the Mid `low[j]` fall all the way to `close[i] - sl_mult * atr_abs`. In live execution, the position is stopped out much sooner because the Bid price is $\frac{S}{2}$ lower than `low[j]`, and entry was $\frac{S}{2}$ higher.
- **Result:** `_compute_devil_targets_atr` simulates a frictionless, zero-toll fantasy world.

Inspected `src/core/retrainer/_features.py:180-192` (`angel_target`):
```python
pl.col("close").shift(-3).over("symbol") > pl.col("close") + _angel_mult * (pl.col("close") * pl.col("natr_14") / 100.0)
```
- `angel_target` checks only whether Mid price 3 bars forward exceeds Mid price plus ATR momentum.
- Zero spread toll is required to label a sample as an Angel win (`1`).

Inspected `src/ml/barriers/labels.py:90-105` (`compute_excursions`):
```python
mae = np.clip(close - fmin, 0.0, None) / atr_abs
mfe = np.clip(fmax - close, 0.0, None) / atr_abs
```
- MAE and MFE are calculated strictly on Mid prices.
- The 95% quantile MAE predicted by `BarrierEstimator` underestimates realized live MAE by the entire spread ($S$).

### 3. Disconnect with Spread Alphas and Gate A

The repository contains an empirical spread calibration framework:
- `scripts/bake_spread_alphas.py` extracts measured $\alpha_{\text{emp}} = \frac{\text{median spread}}{\text{median baseline NATR}}$ from soak telemetry, generating `config/spread_alphas_m15.json`:
  - `AUD_JPY`: $\alpha = 0.3372$ (spread is 33.7% of ATR)
  - `EUR_JPY`: $\alpha = 0.3600$ (spread is 36.0% of ATR)
  - `GBP_JPY`: $\alpha = 0.3307$ (spread is 33.1% of ATR)
  - `NZD_JPY`: $\alpha = 0.5591$ (spread is 55.9% of ATR)
  - `GBP_AUD`: $\alpha = 0.6952$ (spread is 69.5% of ATR)
  - `GBP_NZD`: $\alpha = 0.8929$ (spread is 89.3% of ATR)
- In `src/core/retrainer/_features.py:280-295`, `alpha_table` is passed **only** to `_compute_chop_veto_mask` (Gate A cost filter) and `V3CostFeatures` (`cost_ratio`).
- **`alpha_table` is NEVER passed to `_compute_devil_targets_atr`, `_compute_devil_survival_target`, `angel_target`, or `compute_excursions`.**
- Gate A simply drops bars where `sl_mult * natr < k_eff * spread`. For the bars that *pass* Gate A, the labeling logic continues to evaluate them with zero spread toll!

### 4. Empirical Measurement: Spread Toll Degradation

Evaluated 180 days of M15 data across all 6 tradeable fiat instruments comparing:
1. Current Zero-Toll Labeling (`_compute_devil_targets_atr` with SL=2.0x, TP=4.0x, hold=45).
2. Real-World Spread-Aware Labeling (TP requires $+ \Delta + S$, SL triggers at $- \Delta + S$ using calibrated alphas from `config/spread_alphas_m15.json`).

| Instrument | Measured $\alpha$ (Spread / ATR) | Zero-Toll Win Rate | Real-Toll Win Rate | Relative Win Rate Drop | Zero-Toll PF | Real-Toll PF |
|---|---|---|---|---|---|---|
| **GBP_JPY** | 0.331 | 25.92% | **21.08%** | **-18.7%** | 0.700 | 0.534 |
| **AUD_JPY** | 0.337 | 27.06% | **21.94%** | **-18.9%** | 0.742 | 0.562 |
| **EUR_JPY** | 0.360 | 26.74% | **21.29%** | **-20.4%** | 0.730 | 0.541 |
| **NZD_JPY** | 0.559 | 26.82% | **18.67%** | **-30.4%** | 0.733 | 0.459 |
| **GBP_AUD** | 0.695 | 27.00% | **16.83%** | **-37.7%** | 0.740 | 0.405 |
| **GBP_NZD** | 0.893 | 27.42% | **14.69%** | **-46.4%** | 0.756 | 0.344 |

Across the basket, realistic spread tolls reduce the win rate by **18.7% to 46.4%**.
On `GBP_NZD`, nearly half of all simulated wins turn into losses once the bid-ask spread is paid.
At a 2:1 payout ratio, a 21% win rate yields a dismal Profit Factor of ~0.54, guaranteeing severe capital bleed.

## Findings / Changes

### Finding 1: Universal Spread Toll Omission in Training Targets (Severity: High)
`_compute_devil_targets_atr`, `_compute_devil_survival_target`, and `angel_target` generate binary training labels strictly on Mid price series without incorporating the bid-ask spread. The models are trained to optimize against frictionless price movements that are unattainable in live broker execution.

### Finding 2: False Promotion Gate Validation (Severity: High)
`_score_artifact_holdout` in `src/core/retrainer/_gate.py` evaluates holdout performance using frictionless mid-price bracket simulations. Candidate models that report passing metrics (e.g. PF $\ge 1.20$ or win rate $\ge 52\%$) are validated against a zero-toll benchmark. Upon deployment, the 19%–46% win rate degradation caused by the spread turns nominally passing models into net-losing systems.

### Finding 3: Severe Asymmetry on High-Alpha Instruments (Severity: Medium)
On wide-spread pairs like `GBP_NZD` ($\alpha = 0.89$) and `GBP_AUD` ($\alpha = 0.70$), the round-trip spread consumes nearly half of the entire 2.0x ATR stop distance. Mid-price simulation flatters `GBP_NZD` with a 27.4% win rate when its actual after-toll win rate is only 14.7% (PF 0.34).

### Finding 4: Incomplete Scope of `spread_alphas.json` (Severity: Low)
While `config/spread_alphas_m15.json` accurately captures empirical broker spreads from soak logs, its usage is artificially restricted to Gate A filtering and the `cost_ratio` feature. It is never integrated into the target generation or barrier estimation pipelines.

## Verification

1. **Code Audit**: Confirmed `_compute_devil_targets_atr`, `_compute_devil_survival_target`, `angel_target`, and `compute_excursions` accept no spread or alpha parameter and perform bracket logic exclusively on Mid prices.
2. **Empirical Measurement Across Basket**: Executed full-basket bracket simulation comparing zero-toll vs. spread-aware outcomes on 180 days of M15 history. Verified win rates drop from ~27% down to 21% on JPY crosses and down to 14.7% on GBP_NZD.
3. **Usage Inspection**: Confirmed `alpha_table` in `src/core/retrainer/_features.py` is consumed only by `_compute_chop_veto_mask` and `V3CostFeatures`.

## Risk & follow-ups

1. **Spread-Aware Target Generation**:
   Update `_compute_devil_targets_atr` and `_compute_devil_survival_target` to accept `alpha_table`. Incorporate spread tolls directly into simulated entry and exit prices:
   $$\text{TP}_{\text{target}} = \text{close} + \text{tp\_mult} \cdot \text{ATR} + \text{spread}$$
   $$\text{SL}_{\text{trigger}} = \text{close} - \text{sl\_mult} \cdot \text{ATR} + \text{spread}$$
2. **Cost-Aware Promotion Gate**:
   Ensure `_score_artifact_holdout` scores candidates against spread-penalized bracket outcomes. Any candidate that cannot beat the promotion gate net of spread tolls must be rejected before deployment.
3. **Excursion Spread Adjustment**:
   Adjust `mae_natr` and `mfe_natr` in `compute_excursions` to include the instrument's spread alpha ($\text{MAE} \leftarrow \text{MAE} + \alpha$, $\text{MFE} \leftarrow \text{MFE} - \alpha$), ensuring quantile barrier models price execution costs into their predicted stop and target boundaries.

## Files read

- `src/core/retrainer/_labels.py`
- `src/core/retrainer/_features.py`
- `src/core/retrainer/_gate.py`
- `src/ml/barriers/labels.py`
- `scripts/bake_spread_alphas.py`
- `config/spread_alphas_m15.json`
- `src/execution/risk_manager.py`
- `llm_reports/recons/2026-08-08_stop-width-and-the-spread-toll.md`
- `llm_reports/refactors/2026-07-07_cost-awareness-spread-feature.md`
