---
type: handoff
date: 2026-09-14
time: 00:20 PDT
agent: Antigravity IDE (Gemini 3.8 Flash)
model: gemini-3.8-flash-high
trigger: "Brandon requested an architectural collaboration plan to upgrade stop loss and barrier geometry from static ATR brackets to Quantile Regression for Maximum Adverse Excursion (MAE), collaborating with another active LLM harness."
head: bcbd39c12ec11b54ccd06c6438726fd3eeffe647
branch: fix/seam-backfill-status-emit
scope: read-only plan
related:
  - m2m-prompts/README.md
  - recons/2026-09-13_h1-h4-catboost-ab.md
  - recons/2026-09-08_stage02-catboost-ab.md
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
files_touched: []
---

# Architecture Plan: Quantile Regression for Maximum Adverse Excursion (MAE)

**Authored 2026-09-14 00:20 PDT · Antigravity IDE (Gemini 3.8 Flash High). Read-only; no production weights changed.**

This document provides a complete technical specification and execution plan for transitioning the trading bot from static ATR-multiple stop losses ($2.0\times$ ATR SL / $4.0\times$ ATR TP) to instance-specific, learned quantile barriers based on **Maximum Adverse Excursion (MAE)** and **Maximum Favorable Excursion (MFE)**.

---

## 1. Context & Primary Evidence

### Current Live Baseline
- Live execution ([src/execution/risk_manager.py:341](src/execution/risk_manager.py#L341)) and retraining ([src/core/retrainer.py](src/core/retrainer.py)) use static brackets:
  $$\text{SL} = 2.0 \times \text{ATR}_{14}, \quad \text{TP} = 4.0 \times \text{ATR}_{14}$$
- The 2026-08-08 study ([2026-08-08_stop-width-and-the-spread-toll.md](llm_reports/recons/2026-08-08_stop-width-and-the-spread-toll.md)) proved that static stops do not adapt to high-conviction or volatility tail regimes; the fixed multiplier stops out 73% of entries in high-volatility setups.

### What Already Exists in the Codebase
- [src/ml/barriers/labels.py](src/ml/barriers/labels.py): Implements `compute_excursions(df, horizon=45)`, which calculates realized forward MAE and MFE in NATR-normalized units per symbol without cross-symbol leakage. Pinball loss is implemented for quantile evaluation.
- [src/ml/barriers/estimator.py](src/ml/barriers/estimator.py): Implements `BarrierEstimator`, designed to fit $Q_{\text{MAE}}(0.95)$ for stop distance and $Q_{\text{MFE}}(0.50)$ for take-profit distance.
- [scripts/evaluate_barriers.py](scripts/evaluate_barriers.py): Implements the Phase 1 offline evaluation protocol across 3 expanding chronological folds.

### The Breakthrough Finding (Why Stage 1 Previously Failed & How to Fix It)
Running `scripts/evaluate_barriers.py` with the existing LightGBM estimator **fails on all 3 folds**:
```
fold 1: FIT FAILED: barrier fit violated monotonicity on 'natr_14': rising volatility produced a tighter quantile by more than 0.05 ATR
VERDICT: FAIL — static bracket stays
```
**Mechanism:** LightGBM's quantile objective explicitly forbids monotone constraints (`Cannot use monotone_constraints in quantile objective`). As a result, unconstrained LightGBM trees learn spurious response curves where higher volatility produces tighter stops, triggering the estimator's fit-time safety audit (`_audit_monotone`).

**The Solution:** **CatBoost natively supports monotonic constraints under quantile regression.**
Verified live on this machine (Python 3.12, `catboost==1.2.10`):
```python
model = CatBoostRegressor(
    loss_function="Quantile:alpha=0.95",
    monotone_constraints={"natr_14": 1, "vol_rel": 1}
)
```
This guarantees monotonic increasing stop distances with respect to volatility **by construction**, eliminating the post-hoc audit failures.

---

## 2. Invariants & Standing Rails

1. **Train/Live Symmetry:** Stop and target geometries must be computed identically between `retrainer.py` (label generation / validation) and `ml_strategy.py` / `risk_manager.py` (inference).
2. **Vol-Monotonicity Guarantee:** A wider ATR or relative volatility must never predict a tighter absolute stop distance.
3. **Horizon Alignment:** The forward excursion horizon (`DEFAULT_HORIZON = 45`) must match the execution lifetime (`max_hold = 45`).
4. **Evidential Rigor:**
   - Evaluated on 3 expanding chronological folds over 730 days.
   - Promotion requirement: Learned quantile barrier must beat the static baseline pinball loss on **every** fold, with empirical coverage $\ge 0.93$.
5. **Soak Safety:** `soak.service` is running live under systemd. Never touch `models/forex_m15_wide` or un-isolated production paths during research.

---

## 3. Implementation Blueprint

```
Phase 1: Estimator Upgrade (CatBoost Quantile Regression)
  └─ Add CatBoost backend to src/ml/barriers/estimator.py with monotone constraints
  └─ Pass scripts/evaluate_barriers.py (Coverage >= 93%, beats static pinball on all folds)

Phase 2: Timeframe Context (H4 vs M15)
  └─ Evaluate learned barriers on H4 bars (where CatBoost already demonstrated +0.55R EV)
  └─ Benchmark pinball loss, coverage, and realized-R expectancy across both timeframes

Phase 3: Retrainer Integration (Training Pipeline)
  └─ Wire BarrierEstimator into src/core/retrainer.py
  └─ Option A: Dynamic barrier labels for Devil model
  └─ Option B: Dual-model decoupled architecture (Direction classifier + Quantile geometry)

Phase 4: Live Strategy & Execution Wiring
  └─ Export barriers_mae.cbm / barriers_mfe.cbm alongside model artifacts
  └─ MLStrategy loads barrier models and attaches dynamic stop/target distances to Signal
  └─ RiskManager and OandaForexOrchestrator enforce dynamic stop distances
```

---

## 4. Key Decision Points for Collaboration

### D1: Single-Family Architecture (CatBoost Everywhere)
- **Recommendation:** Standardize both classification and quantile barrier regression on CatBoost.
- **Rationale:** CatBoost solves LightGBM's monotonicity defect in quantile regression and showed strong out-of-sample edge on H4 (+0.55R EV).

### D2: Quantile Parameters ($\tau$)
- **Stop Loss ($\tau_{\text{MAE}}$):** Default $0.95$ (leaves 5% tail risk). Consider testing $0.90$ to see if a tighter stop improves expectancy net of spread.
- **Take Profit ($\tau_{\text{MFE}}$):** Default $0.50$ (median favorable excursion).

### D3: Timeframe Priority
- **H4 Priority (Recommended):** On H4, the spread toll is diluted and CatBoost achieved 48.4% win rate on a 2:1 payoff. Adding dynamic MAE stops to H4 removes the arbitrary $2.0\times$ constant.
- **M15 Dual-Track:** Evaluate whether dynamic MAE stops can rescue M15 by selectively widening stops on high-volatility bars.

---

## 5. Division of Work Between LLM Harnesses

- **Harness A (This Harness / Antigravity IDE):**
  - Implement CatBoost quantile backend with native monotonic constraints in `src/ml/barriers/estimator.py`.
  - Update and execute `scripts/evaluate_barriers.py` to confirm all 3 folds pass with coverage $\ge 93\%$.
- **Harness B (Partner LLM Harness):**
  - Review the label horizon and feature parity contracts in `src/ml/barriers/labels.py` and `src/core/retrainer.py`.
  - Design the `RiskManager` / `MLStrategy` sidecar schema for dynamic barrier loading during live inference.
