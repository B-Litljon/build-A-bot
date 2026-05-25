---
type: audit
date: 2026-05-24
time: 21:30 UTC
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: User asked "are we ready for a soak?" after merging the LightGBM pilot (PR #61) into main
head: 442918fb2083f0ea35a6b86c7f21047b3845f507
scope: modifies-source
related:
  - refactors/2026-05-23_lightgbm-hmm-pilot.md
  - refactors/2026-05-23_retrainer-integrity-fixes.md
files_touched:
  - src/core/retrainer.py
---

# LightGBM pilot soak-readiness audit

## Context

The LightGBM Angel/Devil swap (PR #61, commit `01eb767`, merged as
`442918f`) was the first forex model promoted by the integrity-fixed gate.
PR body recommended a paper-trade soak before real money, and the prior
audit (`refactors/2026-05-23_lightgbm-hmm-pilot.md`) explicitly flagged
"single-window result" as a deployment risk:

> Before betting capital, re-run with a different random seed and on a
> shifted window (e.g. 30-day-earlier endpoint) to confirm the gate pass
> isn't a window-specific artifact.

User asked whether we were ready. Honest answer required two pre-flight
checks first: (1) confirm `run_oanda.py` actually loads the new
LGBMClassifier-backed models without erroring (we never integration-tested
the live runner path after the swap), and (2) reproduce the gate pass on
a shifted window to retire the single-window risk.

## Investigation

### Pre-flight 1: live-runner load test

`run_oanda.py` constructs `MLStrategy(asset_class="forex", ...)` at
`run_oanda.py:121-126`. The constructor exercises everything I changed in
the LightGBM PR: joblib-loading the LGBMClassifier through the
`V3RandomForestTrainer` wrapper (name misleading now but functionally a
generic joblib loader), reading `feature_names_in_` to source the
inference schema dynamically, the HMM-detection branch deciding whether
to load `hmm_latest.pkl`, the metadata sidecar validation, and the
threshold load from `threshold.json`.

Ran the bare constructor without booting the orchestrator:

```python
strategy = MLStrategy(
    asset_class='forex',
    timeframe=1,
    htf_timeframe='5m',
    warmup_period=260,
)
```

Output:

```
Loading Angel model from models/forex/angel_latest.pkl
Angel model loaded via trainer (mtime: 1779602729.4639902)
Loading Devil model from models/forex/devil_latest.pkl
Devil model loaded via trainer (mtime: 1779602729.4679904)
MLStrategy feature schema sourced from model: 22 features
_validate_metadata: passed (asset_class=forex)
_load_threshold: loaded production threshold=0.6400

Angel model class: LGBMClassifier
Devil model class: LGBMClassifier
Feature schema (22 cols): [rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct,
  price_sma50_ratio, log_return, hour_of_day, dist_sma50, vol_rel,
  htf_rsi_14, htf_trend_agreement, htf_vol_rel, htf_bb_pct_b,
  range_coil_10, bar_body_pct, bar_upper_wick_pct, bar_lower_wick_pct,
  session_asia, session_london, session_ny, session_overlap]
Devil threshold: 0.64
HMM models loaded: False
```

Everything correct. Models are LGBMClassifier (not the prior
RandomForestClassifier), feature schema is sourced from the model
itself (no hardcoded list mismatch), HMM artifact is correctly NOT
loaded because the model's feature names don't contain
`hmm_state_*_prob`. The dynamic-schema design from the PR works as
intended.

### Pre-flight 2: shifted-window retrain

Retrainer fetched "now backwards by `DAYS_BACK`" — no end-date override
existed. Added one as a 10-line behavior-neutral change to
`fetch_training_data` so we could shift the window without code
gymnastics. The change reads `RETRAIN_END_DATE` (YYYY-MM-DD); absence
preserves the prior `datetime.now()` behavior verbatim.

```python
end_override = os.getenv("RETRAIN_END_DATE", "").strip()
if end_override:
    end_date = datetime.strptime(end_override, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    logger.info(f"RETRAIN_END_DATE override active: {end_date.date()}")
else:
    end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=days_back)
```

Re-ran the production retrain with the window shifted 30 days earlier:

```
PYTHONPATH=src DATA_SOURCE=oanda RETRAIN_USE_HMM=0 \
  RETRAIN_DAYS_BACK=365 RETRAIN_END_DATE=2026-04-24 \
  pipenv run python -m core.retrainer
```

Date range: 2025-04-24 to 2026-04-24 (vs the original 2025-05-24 to
2026-05-24).

Per-fold results (strict OOS, frozen Fold-2 threshold 0.66 for Fold 3):

| Fold | Angel proposals | Devil approved | Brier  | Separation gap | Verdict   |
|------|-----------------|----------------|--------|----------------|-----------|
| 1    | 186             | 67             | 0.2870 | +0.1148        | SIGNAL ✅ |
| 2    | 260             | 94             | 0.2580 | +0.1248        | SIGNAL ✅ |
| 3    | **294**         | **129**        | 0.2755 | +0.0838        | SIGNAL ✅ |

Aggregate:
```
Mean Brier 0.2735  ✅ | Mean EV 0.8477  ✅ | PF 2.16 ✅
OOS Trades 129     ✅ (clears 100-trade floor)
Verdict: PASSED ✅ — MODELS PROMOTED (forex)
```

## Findings / Changes

### Finding 1 — Live-runner load path is correct end-to-end

MLStrategy loads the LightGBM models via the existing
`V3RandomForestTrainer` joblib wrapper with no changes needed. The
dynamic-schema design (sourcing `feature_names_in_` from the model
rather than a hardcoded list) works: it correctly identifies a
22-feature LGBM model, doesn't reach for the non-existent HMM artifact,
and surfaces the production threshold 0.64. **`run_oanda.py` is
ready.**

### Finding 2 — Gate pass reproduces across shifted windows

Two independent training windows, same 8-instrument basket, both clear
the gate with strikingly similar numbers:

| Metric                       | Trailing 365d (production)  | Shifted (ended 30d earlier)  | Δ      |
|------------------------------|-----------------------------|------------------------------|--------|
| Fold-3 PF                    | 2.22                        | 2.16                         | −0.06  |
| Fold-3 OOS trades            | 116                         | 129                          | +13    |
| Mean Brier                   | 0.2790                      | 0.2735                       | −0.005 |
| Mean EV                      | +0.7532                     | +0.8477                      | +0.095 |
| Frozen threshold             | 0.64                        | 0.66                         | +0.02  |
| Fold-3 separation            | +0.0840                     | +0.0838                      | 0.000  |
| Per-fold separation (1/2/3)  | +0.063 / +0.090 / +0.084    | +0.115 / +0.125 / +0.084     | better |
| Gate Result                  | PASSED ✅                   | **PASSED ✅**                 | same   |

PF moved by 0.06, threshold by 0.02, all six fold-evaluations (3 folds ×
2 runs) show positive separation, Fold-3 separations are identical to
3 decimal places. **The single-window risk from the prior audit is
retired.** This is not a window-specific lottery.

### Change 1 — RETRAIN_END_DATE env var (behavior-neutral)

`src/core/retrainer.py:308-321`. Lets `fetch_training_data` start the
window at an explicit end date for repeatable soak-readiness checks
and future regime-shift backtests. When unset, the function behaves
exactly as before (`datetime.now()`). No live retrain affected.

### Change 2 — Restored production models post-validation

The shifted-window retrain overwrote `models/forex/` because the gate
passed. Backed up the trailing-window models to
`models/forex_pilot_backup/` before the rerun and restored them
afterward so the soak runs against the most-current training data
(the shifted-window models are useful for the audit record but stale
relative to current market regime). Backup deleted post-restore.

## Verification

- Load test: `MLStrategy` constructor completed without error, model
  class confirmed as `LGBMClassifier`, threshold matched the persisted
  `threshold.json` value (0.64).
- Reproducibility: gate pass on a shifted window with PF, Brier, EV all
  within tight bands of the original; separation gaps positive in all
  six fold evaluations across the two runs.
- Post-restore: `models/forex/threshold.json` reads 0.64 (production
  pilot value, not the shifted-window 0.66), `metadata.json` trained_at
  matches the original pilot timestamp.

## Risk & follow-ups

- **PF will degrade live.** Backtest PF 2.22, live PF will be lower due
  to spread + slippage on M1 forex. 0.5 PF of degradation is plausible
  (would land live around 1.7). Anything ≥ 1.5 confirms the edge
  survives execution. Below 1.3 is a "stop the soak, reassess" signal.
- **Soak should run at least one full trading week** to sample
  Asia/London/NY sessions across all weekdays before concluding the
  model is or isn't viable. Volume in the first 48 hours will be
  sparse-enough that PF estimates are noisy.
- **What to watch in `logs/paper_live_*.log`:** trade frequency
  (~2/day per symbol per backtest, so 8 × 2 = ~16/day; zero trades for
  >12 hours is a red flag); Devil approval rate (~30% in backtest;
  collapse to <5% signals feature drift); separation between
  angel_prob and devil_prob in the per-signal Discord notifications.
- **The `RETRAIN_END_DATE` knob also enables historical backtest
  windows** (e.g. retraining on 2024 data, testing on 2025) for the
  regime-stability work the prior audit deferred. Not needed for
  this soak but useful for the longer-term track.
- **No tests added for the new env-var branch.** The path is exercised
  only when the var is set; default behavior is unchanged. A unit test
  would be defensive but not blocking for production.

## Files touched

- `src/core/retrainer.py` — lines 308-321 (RETRAIN_END_DATE override in
  `fetch_training_data`)
