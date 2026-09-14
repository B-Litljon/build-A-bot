# 2026-09-13 — H1 and H4 CatBoost A/B vs LightGBM (730-Day Benchmark)

**Question:** Does running CatBoost (ordered trees with monotonic constraints) on higher timeframes (1-hour and 4-hour) overcome the spread toll and negative EV observed on M15?

**Answer, up front:**
- **1-Hour (H1): Fails like M15.** Both LightGBM and CatBoost produce heavy negative EV (LGBM -0.88R, CB -0.71R) and poor win rates (12.5% for CatBoost). The spread drag and chop dynamics on H1 remain too high.
- **4-Hour (H4): Dramatic regime shift to positive expectancy.** CatBoost on H4 achieves a **48.4% win rate** on a 2:1 payoff bracket (31 macro wins / 64 trades), clearing the gate's pooled PF 95% lower bound at **1.2057** (vs LightGBM's 0.8678). Folds 1 and 2 produced strong positive EV (**+0.548R** and **+0.406R**). The model was rejected by the gate solely because Fold 3 had an $n=1$ trade count (which lost, dragging the unweighted 3-fold mean EV to -0.015 and giving Fold 3 a zero CP lower bound).

---

## Benchmark Evidence (730 Days, 8-Symbol Forex Basket)

Both tests ran simultaneously on local CPU after sequentially caching 730 days of OANDA mid-candles for `AUD_JPY`, `EUR_JPY`, `GBP_AUD`, `GBP_JPY`, `GBP_NZD`, `NZD_JPY`, `XAU_USD`, and `XAG_USD`.

- **H1 Arm:** 60m bars, `RETRAIN_HTF_TIMEFRAME=4h`
- **H4 Arm:** 240m bars, `RETRAIN_HTF_TIMEFRAME=1d`

### 1-Hour (H1) Comparison

| Metric | LightGBM (H1) | CatBoost (H1) |
|---|---|---|
| **gate_passed** | False | False |
| **mean Brier** | 0.1500 | 0.1797 |
| **mean EV** | -0.875000 | -0.710322 |
| **pooled PF 95% lb** | 0.0073 (1/14) | 0.1431 (9/72) |
| **fold3 PF 95% lb** | 0.0000 | 0.0000 |
| **Angel bar (calibrated)** | 0.3855 | 0.2976 |
| **Fold 1 EV / trades** | -0.625 / 8 | -0.471 / 17 |
| **Fold 2 EV / trades** | -1.000 / 5 | -0.660 / 53 |
| **Fold 3 EV / trades** | -1.000 / 1 | -1.000 / 2 |
| **Win rate $\ge 0.40$ band** | 26.3% (n=19) | 66.7% (n=3) |

---

### 4-Hour (H4) Comparison

| Metric | LightGBM (H4) | CatBoost (H4) |
|---|---|---|
| **gate_passed** | False | False |
| **mean Brier** | 0.4203 | **0.3627** ✓ |
| **mean EV** | -0.230526 | **-0.015121** ✓ |
| **pooled PF 95% lb** | 0.8678 (35/90) | **1.2057 (31/64)** ✓ (Clears $\ge 1.20$ bar) |
| **fold3 PF 95% lb** | 0.0000 | 0.0000 (n=1 artifact) |
| **Angel bar (calibrated)** | 0.3507 | 0.2633 |
| **Fold 1 EV / trades** | -0.132 / 38 | **+0.548 / 31** (16 wins, 51.6% WR) |
| **Fold 2 EV / trades** | +0.440 / 50 | **+0.406 / 32** (15 wins, 46.9% WR) |
| **Fold 3 EV / trades** | -1.000 / 2 | -1.000 / 1 (0 wins, 0.0% WR) |
| **Win rate $\ge 0.40$ band** | 50.0% (n=26) | **57.1% (n=7)** |
| **Win rate 0.30–0.35 band**| 35.4% (n=99) | **52.8% (n=36)** |
| **Win rate 0.20–0.30 band**| 60.0% (n=5) | **46.8% (n=171)** |

---

## Core Findings

1. **Edge Emerges on H4:**
   On a 2.0x ATR SL / 4.0x ATR TP bracket (2:1 reward:risk), break-even is 33.3% win rate.
   CatBoost on H4 achieved **48.4% pooled win rate** (31 wins on 64 trades), producing **1.2057** pooled PF lower bound. Folds 1 and 2 demonstrated sustained positive expectancy (+0.55R and +0.41R).
2. **Why H4 Failed the Gate:**
   The rejection reasons for CatBoost on H4 were:
   - `Brier 0.3627 > 0.3 threshold` (driven entirely by Fold 3 Brier of 0.7576 on a single trade).
   - `EV -0.015121 < 0.0005 threshold` (the unweighted mean of [+0.548, +0.406, -1.000] is -0.015; Fold 3's single trade dragged the entire average negative).
   - `Fold 3 PF lower bound 0.0000` (because $n=1$, Clopper-Pearson 95% confidence on 0/1 is 0.0000).
3. **The Fold 3 Sample Deficit:**
   Fold 3 had only 1 approved trade for CatBoost (and 2 for LightGBM). This is caused by the tail boundary purge (`_purge_boundary_tail`) removing the last 45 bars of the training slice, leaving Fold 3 with a compressed window on H4 bars (45 bars = 180 hours = 7.5 trading days dropped).

---

## Artifacts Generated

- H1 Results: `logs/ab_result_catboost_h1.txt`, `logs/ab_result_lightgbm_h1.txt`, `logs/ab_ledger_catboost_h1.parquet`
- H4 Results: `logs/ab_result_catboost_h4.txt`, `logs/ab_result_lightgbm_h4.txt`, `logs/ab_ledger_catboost_h4.parquet`
