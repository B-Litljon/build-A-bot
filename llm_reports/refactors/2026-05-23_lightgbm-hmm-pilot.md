---
type: refactor
date: 2026-05-23
time: 17:10 UTC
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: User asked for alternative algorithms after RF Angel/Devil failed every integrity-fixed gate; picked LightGBM + HMM combined experiment
head: b02295de095c9874b64d8d462f0a72f5a04c1117
scope: modifies-source
related:
  - refactors/2026-05-23_retrainer-integrity-fixes.md
files_touched:
  - src/core/retrainer.py
  - src/strategies/concrete_strategies/ml_strategy.py
  - src/ml/regimes/__init__.py
  - src/ml/regimes/hmm_regime.py
  - Pipfile
  - Pipfile.lock
---

# LightGBM + HMM regime pilot — first promoted forex model of the session

## Context

After the integrity-fixes audit (`refactors/2026-05-23_retrainer-integrity-fixes.md`)
closed three leakage holes in the validation gate, every RF retrain rejected
on the new 100-trade OOS floor. The best RF run (commit `b02295d`, 8-instrument
365d) cleared every quality metric (Mean Brier 0.20, Fold-3 PF 3.20, separation
+0.07 in all folds) but was rejected on sample size — only 13 Fold-3 OOS
trades. The Devil at the frozen 0.66 calibration threshold was approving 6.2%
of Angel proposals, producing a "sniper" that was statistically too small to
honestly underwrite.

The user asked for algorithm alternatives. After laying out the landscape
(gradient boosting, sequence models, HMM regimes, RL), the user picked the
combined-pilot option:

  > LightGBM + HMM together — do both in one feature branch.

Hypothesis: gradient boosting would extract more signal from the same 22-feature
vector (RF's per-bar IID treatment is a known limitation on financial tabular
data), and per-instrument HMM state probabilities would give the booster an
explicit regime axis to condition on. Question being answered: does this
combination get us above the 100-trade floor at acceptable quality?

## Investigation

Read the full RF Angel/Devil cascade in `src/core/retrainer.py` end-to-end
to map the touchpoints:

- Line 42-44 imports: `RandomForestClassifier`, `brier_score_loss`, `TimeSeriesSplit`
- Lines 110-137 `get_hyperparameters()`: per-asset-class param dicts (the
  prior session set `min_samples_leaf=20` for forex after a leaf=1 leakage
  incident)
- Lines 629-834 `refit_models()`: three RF instantiation sites — full-data
  Angel (684), per-fold Angel for OOF generation (723 + 738), Devil (807)
- Lines 914-1433 `validate_candidate()`: walk-forward fold loop and the
  defensive Devil single-class handler at 1116-1132 (degenerate fold from
  the prior session)
- Lines 1441-1493 `promote_or_reject()`: atomic save of Angel/Devil + threshold

Then `src/strategies/concrete_strategies/ml_strategy.py`:

- Line 39 imports `V3RandomForestTrainer` — a thin wrapper exposing
  `predict_proba`, `feature_names_in_`, `load()` from joblib. The trainer
  abstracts away the underlying classifier class, so loading a LightGBM
  model through it works without changes (joblib returns the deserialized
  object; `predict_proba` and `feature_names_in_` exist on LightGBM's
  sklearn wrapper too).
- Lines 154-188: hardcoded `feature_names` list with a schema-drift guard
  that compares against `angel_trainer.feature_names_in_` — this would
  reject any retrain that added/removed features, including the planned
  HMM regime cols.
- Line 384 `_generate_features()`: builds the feature frame via
  `FeaturePipeline.run()` then slices `[self.feature_names].tail(1)` for
  inference.

Per-instrument HMM scoping decision (asked the user): **soft features**
(3 state-probability columns added to FEATURE_COLS, model decides how to
use them) and **3 hidden states** (classic trending/ranging/volatile
partition). Lowest risk; preserves sample size; lets the booster learn
regime-conditional behavior on its own.

Dependency check: `lightgbm` was already listed in Pipfile (V4 investor
uses it) and present in the venv at 4.6.0; `hmmlearn` had to be added,
installed cleanly at 0.3.3.

## Findings / Changes

### Change 1 — Swap RF for LightGBM in retrainer.py

`src/core/retrainer.py`:

- Removed `from sklearn.ensemble import RandomForestClassifier`; added
  `import lightgbm as lgb`.
- Rewrote `get_hyperparameters()` (lines 110-156) to LightGBM param dicts.
  Translation rationale documented inline:
  - `n_estimators 100 → 200` paired with `learning_rate=0.05` — boosting
    needs more rounds to reach RF-equivalent capacity.
  - `max_depth 10/8` retained as a cap on tree depth; `num_leaves=63/31`
    keeps effective complexity in RF's vicinity (below the 2^max_depth
    ceiling).
  - `min_samples_leaf 20 → min_child_samples 20` — direct equivalent.
  - `subsample=0.8, subsample_freq=1, colsample_bytree=0.8` — LightGBM-native
    generalization knobs. RF gets the same effect for free via bootstrap.
  - `verbose=-1` silences LightGBM's per-iter chatter.
- Replaced all four `RandomForestClassifier(**params)` instantiation sites
  with `lgb.LGBMClassifier(**params)`. Updated type annotations to
  `"lgb.LGBMClassifier"` (string-quoted to avoid re-importing in narrow
  contexts).
- Updated the hyperparameter log line at the bottom of `main()` to read
  `min_child_samples` instead of the now-defunct `min_samples_leaf`.

### Change 2 — New per-instrument HMM regime module

`src/ml/regimes/hmm_regime.py` (new, 165 lines):

- `fit_regime_models(train_df, n_states=3)` — fits a Gaussian HMM per
  symbol on `(log_return, natr_14)` with `covariance_type="diag"`,
  `n_iter=50`, `tol=1e-3`. Returns `Dict[symbol, Optional[GaussianHMM]]`;
  symbols with `<200` rows or fit failure get `None`.
- `predict_regime_probs(df, models)` — appends 3 columns
  `hmm_state_{0,1,2}_prob` per row using the symbol's fitted HMM. Symbols
  without a fitted model get uniform `1/3` across states (no-op feature
  the classifier can ignore).
- `save_hmm_models(models, target_path)` / `load_hmm_models(source_path)`
  — joblib serialization with atomic `.tmp` rename, mirroring the existing
  Angel/Devil save pattern.

Critical: NaN/Inf scrubbing via `_clean_for_hmm()` replaces non-finite
values with per-column finite means. `hmmlearn` raises on either, and
TA-Lib's NATR/log_return produce both on warmup bars.

### Change 3 — Walk-forward HMM integration

`src/core/retrainer.py`:

- `BASE_FEATURE_COLS` (the 22 features produced by FeaturePipeline) is now
  a separate constant from `FEATURE_COLS` (which appends the 3 HMM cols
  when `RETRAIN_USE_HMM=1`). `engineer_features_and_labels()` cleans nulls
  on the BASE set only — HMM cols are added downstream per-fold.
- Inside `validate_candidate()`'s fold loop (around line 1080), if HMM is
  enabled: fit on the fold's `train_df`, then `predict_regime_probs` for
  both `train_df` and `val_df`. This preserves the temporal boundary —
  nothing val-side ever influences the HMM parameters.
- After the gate passes, a fresh HMM is fit on the full retraining dataset
  for production. This dict is returned through a new 7th tuple element
  (`final_hmm_models`).
- `promote_or_reject()` gained an optional `hmm_models` kwarg; if present,
  calls `save_hmm_models()` alongside the existing Angel/Devil save to
  `models/{asset_class}/hmm_latest.pkl`.

### Change 4 — MLStrategy adapts to model-declared feature schema

`src/strategies/concrete_strategies/ml_strategy.py`:

- Removed the hardcoded `self.feature_names` list. Now sources the schema
  from `self.angel_trainer.feature_names_in_`, so a retrain that changes
  the feature space (e.g. enabling `RETRAIN_USE_HMM=1`) propagates to
  inference without a code edit. The old schema-drift guard becomes
  redundant by construction.
- If any `hmm_state_*_prob` features are present in the model's schema,
  loads `models/{asset_class}/hmm_latest.pkl` via `load_hmm_models()` and
  stores the per-symbol HMM dict on `self.hmm_models`. Missing artifact
  for a model that expects HMM features raises immediately — fail-fast
  rather than silently feeding zero-prob features at inference.
- `_generate_features()` now appends HMM regime posteriors after the
  standard FeaturePipeline runs (and only if `self.hmm_models is not None`),
  ensuring train/inference symmetry.

### Change 5 — Pipfile dependency

`Pipfile`: added `hmmlearn = "*"` (already had `lightgbm = "*"`). Lockfile
updated.

## Verification

### Smoke test — XAU_USD, 60d, HMM enabled

```
PYTHONPATH=src DATA_SOURCE=oanda RETRAIN_USE_HMM=1 \
  RETRAIN_SYMBOLS=XAU_USD RETRAIN_DAYS_BACK=60 \
  pipenv run python -m core.retrainer
```

Plumbing validated:
- LightGBM trained Angel (25 features) and Devil (26 features, with `angel_prob`)
  without errors or warnings.
- Per-fold HMM fit cleanly on 37k → 48k row training windows; all three
  symbols converged.
- Devil `predict_proba` returned proper (n, 2) shape — no degenerate
  single-class fallback triggered.
- Two-phase threshold strategy operated (Fold 2 swept and froze, Fold 3
  used frozen).
- Result: gate FAILED (Brier 0.3164 > 0.30 floor, PF 1.1875 just below the
  1.20 floor) — expected for a single-symbol 60d window. Notable that
  separation was still negative across all folds (−0.0064, −0.0363, −0.0122)
  at this small data slice. **Plumbing-only validation**; not a signal-quality
  result.

LightGBM runtime: the entire 3-fold gate completed in ~5 seconds (vs RF's
typical multi-minute folds at this size). The booster is dramatically faster
than RF for the same problem.

### Headline run — 8-instrument 365d, HMM enabled

```
PYTHONPATH=src DATA_SOURCE=oanda RETRAIN_USE_HMM=1 \
  RETRAIN_DAYS_BACK=365 pipenv run python -m core.retrainer
```

Instruments: `XAU_USD, XAG_USD, GBP_JPY, AUD_JPY, EUR_JPY, NZD_JPY, GBP_AUD, GBP_NZD`
Raw bars fetched: **2,915,178** across 8 symbols × 365d.

Per-fold (strict OOS, frozen Fold-2 threshold 0.36 for Fold 3):

| Fold | Angel proposals | Devil approved | Brier  | Separation gap | Verdict   |
|------|-----------------|----------------|--------|----------------|-----------|
| 1    | 234             | 107            | 0.2724 | +0.1176        | SIGNAL ✅ |
| 2    | 271             | 203            | 0.2780 | +0.0808        | SIGNAL ✅ |
| 3    | **403**         | **231**        | 0.2627 | +0.1094        | SIGNAL ✅ |

Fold-3 strict-OOS:
- Macro WR 48.1%, Survival WR 47.6%, EV +0.4286
- Profit Factor (macro) = 222.00 / 120.00 = **1.85** ✅

Aggregate gate readout:
```
Mean Brier 0.2710  ✅ | Mean EV 0.6590  ✅ | PF 1.85 ✅
OOS Trades 231     ✅ (clears 100-trade floor with 2.3× margin)
Verdict: PASSED ✅ — MODELS PROMOTED (forex)
```

Production threshold persisted: **0.36** (down from the RF run's 0.66 —
the LightGBM Devil's probabilities are more dispersed and a lower
threshold optimizes EV).

Artifacts written to `models/forex/`:
- `angel_latest.pkl` (1.4 MB)
- `devil_latest.pkl` (0.6 MB)
- `hmm_latest.pkl` (8/8 symbols fitted)
- `metadata.json`, `threshold.json`

### Control run — LightGBM only, HMM disabled

Ran the same 8-instrument 365d retrain with `RETRAIN_USE_HMM=0` to isolate
HMM's marginal contribution.

```
PYTHONPATH=src DATA_SOURCE=oanda RETRAIN_USE_HMM=0 \
  RETRAIN_DAYS_BACK=365 pipenv run python -m core.retrainer
```

Per-fold (strict OOS, frozen Fold-2 threshold 0.64 for Fold 3):

| Fold | Angel proposals | Devil approved | Brier  | Separation gap | Verdict   |
|------|-----------------|----------------|--------|----------------|-----------|
| 1    | 230             | 85             | 0.2975 | +0.0627        | SIGNAL ✅ |
| 2    | 298             | 104            | 0.2505 | +0.0904        | SIGNAL ✅ |
| 3    | **370**         | **116**        | 0.2891 | +0.0840        | SIGNAL ✅ |

Aggregate:
```
Mean Brier 0.2790  ✅ | Mean EV 0.7532  ✅ | PF 2.22 ✅
OOS Trades 116     ✅ (clears 100-trade floor)
Verdict: PASSED ✅ — MODELS PROMOTED (forex)
```

### Three-way comparison — RF baseline vs LightGBM+HMM vs LightGBM only

Same 8-instrument 365d window, same V3.5 features, same integrity-fixed gate:

| Metric                       | RF (b02295d)    | LightGBM + HMM | LightGBM only    |
|------------------------------|-----------------|----------------|------------------|
| Mean Brier                   | 0.20            | 0.2710         | 0.2790           |
| Mean EV                      | +1.27           | +0.6590        | +0.7532          |
| Fold-3 PF                    | 3.20            | 1.85           | **2.22**         |
| Fold-3 OOS trades            | 13              | **231**        | 116              |
| Fold-3 Angel proposals       | 210             | 403            | 370              |
| Fold-3 Devil approval rate   | 6.2%            | 57.3%          | 31.4%            |
| Fold-3 separation gap        | +0.0751         | **+0.1094**    | +0.0840          |
| Mean separation across folds | ~+0.075         | +0.103         | +0.079           |
| Frozen threshold             | 0.66            | 0.36           | 0.64             |
| Gate Result                  | REJECTED (size) | **PASSED ✅**  | **PASSED ✅**    |

**Attribution: LightGBM does the load-bearing work, HMM is a tradeoff.**

The RF→LightGBM swap alone is sufficient to clear the integrity-fixed
gate: PF 2.22, 116 OOS trades, separation +0.084 on all three folds. The
HMM adds an axis of variation rather than pure improvement:

- HMM-on lowers the Devil's optimal threshold (0.64 → 0.36) because the
  per-instrument regime posteriors give the booster a sharper conditional
  signal — probabilities spread wider, lower threshold captures the bulk.
- That lower threshold means more trades reach the gate (231 vs 116) at
  lower per-trade conviction (PF 1.85 vs 2.22).
- Separation gap is materially better with HMM (+0.109 vs +0.084) — the
  Devil *discriminates* winners from losers more cleanly — but the
  extra-marginal trades pulled in by the threshold drop are weaker than
  the high-conviction core, dragging PF down.

**Net: ship LightGBM-only as the production default.** PF 2.22 is a
0.85× margin over the 1.20 floor (vs LGB+HMM's 0.65×), so a 10% adverse
drift on live data still leaves it inside the gate. The HMM stays in the
codebase behind `RETRAIN_USE_HMM=1` — verified to plumb cleanly, but not
load-bearing on this data slice. Reasonable next experiment for the HMM
is to use it as a hard-filter regime gate (only trade in specific
states) rather than soft features, where its discrimination might cash
out as PF rather than trade-volume.

**The signal we were chasing was real all along — the gate just couldn't
see it through the RF model's high selectivity.** Both LightGBM variants
clear the 100-trade floor because LightGBM's probabilities spread over a
wider range than RF's, lowering the optimal threshold and surfacing
trades the RF Devil rejected at 0.66.

## Risk & follow-ups

- **Single-window result.** Both passing runs are one retrain each on one
  365-day window.
  Before betting capital, re-run with a different random seed and on a
  shifted window (e.g. 30-day-earlier endpoint) to confirm the gate pass
  isn't a window-specific artifact. The integrity-fix audit noted that
  even the same parameters produced different numbers across runs on
  different windows.
- **HMM convergence warnings (when HMM enabled).** Several `Model is not
  converging` warnings fired during HMM fits (e.g. `Delta is -14.913` on
  GBP_NZD). The `monitor_.converged` flag still reports True because the
  tolerance threshold is met, but the negative log-likelihood deltas
  suggest the optimizer is bouncing near a local maximum. Not blocking
  for the LightGBM-only ship, but if HMM mode is revisited (hard-filter
  experiment), worth bumping `n_iter` from 50 → 200 or trying
  `covariance_type="full"` (more parameters, more data needed).
- **HMM as a hard regime filter is the natural follow-up experiment.**
  Soft HMM features gave the booster more proposal volume but lower PF.
  Using the HMM's most-likely-state output as a *binary trade/no-trade
  filter* (e.g. only trade in the high-volatility state) might cash the
  +0.03 separation advantage out as PF rather than trade volume.
- **PF 2.22 is "passing" not "comfortable".** The gate floor is 1.20, so
  the shipped (LightGBM-only) model has a 0.85 PF margin of safety. A 10%
  adverse drift on live data (slippage, news regimes outside training
  distribution) leaves it inside the gate, but the model is not a
  high-conviction edge. Recommend live-monitoring a paper-trade run before
  promoting to a real-money account.
- **Inference-path HMM scoring.** `predict_regime_probs` calls
  `hmm.predict_proba(X)` on the symbol's recent feature window every bar.
  For an HMM, `predict_proba` runs forward-backward smoothing over the
  entire input — fine on a rolling buffer of ~60-260 bars, but worth
  benchmarking on live trading machines to confirm it doesn't add
  noticeable latency to the inference path.
- **Devil training set is still tiny.** 3,827 Angel-approved rows out of
  2.9M raw bars (0.1%) is the production Devil's training population.
  Even at 18× the gate-clearing trade volume, the Devil sees a sparse
  slice of the feature space. Future work: a less restrictive Angel
  threshold (currently 0.40) to enrich Devil training, or a different
  Angel-target definition.

## Files touched

- `src/core/retrainer.py` (RF→LightGBM swap, BASE/FEATURE_COLS split,
  HMM integration in `validate_candidate()` fold loop and full-data block,
  new 7th return element, `promote_or_reject()` gains `hmm_models` kwarg)
- `src/strategies/concrete_strategies/ml_strategy.py` (drop hardcoded
  feature_names in favor of `model.feature_names_in_`, load HMM artifact
  when model expects regime features, apply HMM in `_generate_features`)
- `src/ml/regimes/__init__.py` (new)
- `src/ml/regimes/hmm_regime.py` (new — fit/predict/save/load helpers)
- `Pipfile` (`hmmlearn = "*"`)
- `Pipfile.lock` (regenerated)
