---
type: handoff
date: 2026-05-22
time: PDT
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: User request — brief Gemini CLI (Antigravity) on the current state of the ML pipeline since the V5 forex pivot
head: 9d6752f25e10d80ed323398aca515cbb78c529a7
scope: report-only
related:
  - refactors/2026-05-14_ml-pipeline-refactor.md
  - stops/2026-05-14_ml-pipeline-push-blocker.md
files_touched: []
---

# MODEL-TO-MODEL HANDOFF

**FROM:** Claude Opus 4.7 (Claude Code, repo-resident execution agent)
**TO:** Gemini (Gemini CLI, Antigravity — repo-aware architect peer)
**RE:** ML pipeline state since V5 forex pivot — what changed, what didn't, what still blocks
**HEAD:** `9d6752f` on `feature/v5-history-prime`
**WORKING TREE:** clean except `M src/data/oanda_provider.py` (uncommitted)
**TONE:** peer-to-peer, no filler

---

## TL;DR

No model architecture changes. The V3.3 Angel/Devil meta-labeled Random Forest pair is still production (`models/angel_latest.pkl`, `models/devil_latest.pkl`, also `src/ml/models/{angel,devil}_rf_model.joblib`). All ML-pipeline work since 2026-05-14 has been plumbing: thread-safety, data-loss fixes, hot-path latency, naming hygiene, schema-drift guards, plus a legacy-class swap (`FeatureEngineer` → `FeaturePipeline`) in the analysis tools, and strategy-timeframe parameterization to support OANDA M5 vs M1 streaming.

Bottom line for you: **the pipeline is now structurally sound; the open question is at the strategy/risk seam, not the ML seam.** Specifically, the `min_sl_pct` chop floor for forex is now overridden but the override is empirical, not derived — and that's the area where I'd want your architectural read.

---

## 1. Confirmed model inventory (verified `2026-05-22`)

| Artifact | Path | Origin | Status |
|---|---|---|---|
| Angel RF (regime/setup detector) | `models/angel_latest.pkl` | V3.3 era | Loaded by `MLStrategy.__init__` |
| Devil RF (meta-labeler, takes Angel prob as input) | `models/devil_latest.pkl` | V3.3 era | Loaded by `MLStrategy.__init__` |
| Mirror copies | `src/ml/models/angel_rf_model.joblib`, `src/ml/models/devil_rf_model.joblib` | V3.3 era | Per-package paths used by some tests/scripts |
| Devil threshold | `models/thresholds.json` → `devil_threshold` key | Retrainer writes this | Read by `MLStrategy` at init |
| Angel threshold | Hard-coded in strategy | Manual | `ANGEL_THRESHOLD` constant |

Feature contract — 18 features, ordered, from `MLStrategy.__init__` (`src/strategies/concrete_strategies/ml_strategy.py:140` ff). Devil receives those 18 + `angel_prob` appended → 19-feature input. See §4 below for the hot-path code.

---

## 2. The Five Fixes (commit `98be438`, report `llm_reports/refactors/2026-05-14_ml-pipeline-refactor.md`)

All five landed together on branch `fix/ml-pipeline-refactor`, merged to `main` 2026-05-14. Verified still in place on HEAD `9d6752f`.

### Fix 1.4 — Subset-aware `clean_data` (`src/ml/feature_pipeline.py:44–66`)

**Problem:** `FeaturePipeline.clean_data()` called `df.drop_nulls()` on the entire DataFrame. Sparse fundamental columns (quarterly financials, NaN outside reporting windows) wiped otherwise-valid rows where the 18 ML features were fine.

**Fix verified at file:line:**
```python
# src/ml/feature_pipeline.py:44
def run(self, df: pl.DataFrame, feature_cols: Optional[List[str]] = None) -> pl.DataFrame:
    ...
    return self.clean_data(df, feature_cols=feature_cols)

# src/ml/feature_pipeline.py:54
@staticmethod
def clean_data(df: pl.DataFrame, feature_cols: Optional[List[str]] = None) -> pl.DataFrame:
    ...
    subset = [c for c in feature_cols if c in df.columns] if feature_cols else None
    return df.drop_nulls(subset=subset) if subset else df.drop_nulls()
```

**Caller status:** `MLStrategy._generate_features()` now passes `feature_cols=self.feature_names`. `core/retrainer.py:520` still calls `FeaturePipeline.clean_data(df)` **without subset** — flagged as risk-follow-up in the original report; not yet addressed. **For your review:** is whole-frame drop_nulls in the retrainer actually safe given the feature columns are dense (technical indicators only, no fundamentals on the V5 path)? My read says yes for V5, no for V4 investor, but worth a peer check.

### Fix 2.2 — Thread-safe model hot-reload (`src/strategies/concrete_strategies/ml_strategy.py`)

**Problem:** `_check_model_updates()` called `angel_trainer.load(...)` without synchronization. `MLStrategy` is shared across symbols; `generate_signals()` runs inside `asyncio.to_thread()`. Two symbols hot-reloading concurrently → joblib mid-load + predict on a half-mutated estimator.

**Fix verified:**
- `src/strategies/concrete_strategies/ml_strategy.py:23` — `import threading`
- `:80` — `self._reload_lock = threading.Lock()` in `__init__`
- `:246` — `with self._reload_lock:` wrapping the `angel_trainer.load(...)` + mtime update block

Lock scope includes `n_jobs = 1` override and mtime mutation so the transition is atomic.

**For your review:** no stress test exists. Smoke test only. If you have a strong opinion on whether a two-threads-loading-while-disk-swaps test is worth writing, name it — I'll execute.

### Fix 2.7 — Killed the pandas roundtrip in inference (the hot-path one)

**Problem:** Both `MLStrategy.generate_signals()` (line 350-area pre-fix) and `LiveOrchestrator._run_inference()` (line 1202-area pre-fix) were doing this every bar:
```python
import pandas as pd
meta_features = pd.DataFrame(latest_features_df.to_numpy(), columns=self.feature_names)
meta_features["angel_prob"] = angel_prob
devil_prob = self.devil_trainer.predict_proba(meta_features)[0, 1]
```
sklearn doesn't need a DataFrame for `predict_proba` — any array-like works. The pandas allocation (index, dtype validation, block manager) was thrown away immediately. ~0.3–0.5 ms/bar wasted, GC churn on a per-tick hot path.

**Fix verified:**
- `src/strategies/concrete_strategies/ml_strategy.py:362-363`:
  ```python
  angel_prob_col = np.array([[angel_prob]])
  meta_features = np.hstack([latest_features, angel_prob_col])
  ```
- `src/execution/live_orchestrator.py:1203-1204` — identical pattern.
- `import pandas as pd` fully removed from `ml_strategy.py`. `numpy` import added at `live_orchestrator.py:72`.

**Latency math (eyeballed):** for OANDA M5 streams the savings are noise. For a tick-driven watchdog reacting on the raw-tick hook (`src/data/oanda_provider.py`), every ms matters. So this is a real win on the scalper path, marginal on the bar-close path.

### Fix 2.8 — Quintile → top-k rename (`scripts/investor_feature_pipeline.py`)

Pure naming hygiene. `_top_quintile_label` → `_top_k_label`, `target_top_quintile` → `target_top_k`. The function uses `pd.qcut(valid, q=5, ...)` but `q` was always parameterizable. Verified at `scripts/investor_feature_pipeline.py:89` (function), `:234` (column write), and 11 downstream references throughout the same file.

**Breaking change for downstream:** any Jupyter notebook or BI tool reading `target_top_quintile` will break. Grep found no in-repo references outside the script. **For your review:** if you know of any out-of-repo consumers (the user's own notebooks, dashboards) that I can't see, flag it.

### Fix 2.6 — Feature schema drift guard at load time

Inserted at `src/strategies/concrete_strategies/ml_strategy.py:166-172`:
```python
model_features = self.angel_trainer.feature_names_in_
if model_features is not None and list(model_features) != self.feature_names:
    raise RuntimeError(
        f"Feature schema drift detected: strategy features {self.feature_names} "
        f"do not match angel model features {list(model_features)}"
    )
```

**Current state:** dormant. Production Angel model was trained on raw numpy → `feature_names_in_` returns `None` → guard skips. **Activation condition:** next retrain that passes a DataFrame to sklearn. The retrainer at `src/core/retrainer.py` would need verification on this point — I haven't traced whether `fit()` is called with a DataFrame or ndarray.

**For your review (action item, peer-to-peer):** can you confirm whether `core/retrainer.py` currently passes a DataFrame or ndarray to `model.fit()`? If ndarray, the guard will stay dormant after retrain too, which defeats the point. This is the kind of seam where a precision bug would hide.

---

## 3. The Follow-on Refactor (commit `9d6752f`, 2026-05-21)

Title: *"parameterize strategy timeframes, update Oanda integration, and replace legacy FeatureEngineer with FeaturePipeline in analysis tools"*

Three things bundled:

### 3a. Strategy timeframes parameterized

`run_oanda.py` gained `--granularity N` (default 5 minutes). The strategy now takes `timeframe`, `htf_timeframe`, `warmup_period` constructor args:

```python
# run_oanda.py
htf_tf = "30m" if args.granularity == 5 else "5m"
warmup_pd = 300 if args.granularity == 5 else 260
strategy = MLStrategy(
    timeframe=args.granularity,
    htf_timeframe=htf_tf,
    warmup_period=warmup_pd,
)
```

**Why this matters for V5:** the V3.3 Angel/Devil models were trained on a specific bar resolution (V3.3-era was 1m equities; the user is now resurrecting them for OANDA forex M5). The timeframe param lets the strategy compute features at whatever timeframe is being streamed, but **the model itself doesn't know it's running on different bars than it was trained on**. That's a silent assumption.

**For your review (critical):** is the user planning to retrain the Angel/Devil pair on OANDA M5 forex data, or run the equities-era models on forex bars? The repo's set up to do either, but the second case is a research-grade hack with execution-grade consequences. I haven't seen a retrain on V5 data yet.

### 3b. OANDA integration (the `min_sl_pct` override — STALE MEMORY ALERT)

`run_oanda.py:130`:
```python
risk_profile = RiskProfile(min_sl_pct=0.00002, round_precision=5)
```

**Memory says** (`project_a3_chop_filter_blocker.md`): "`min_sl_pct=0.0015` equities-era floor rejects ~100% of forex M1 signals; blocks all V5 trades pending Gemini ruling."

**Reality on HEAD:** the workaround is committed (`d92202f` — "inject custom risk profile into Oanda orchestrator to adjust SL percentage for forex volatility"). The override is `0.00002` (~0.23 pips on EUR/USD per the inline comment), which is empirically tuned, not derived.

**For your review (this is the live question):** the override is functional but the magnitude was chosen by inspection. The legitimate architectural question is whether `min_sl_pct` should be:
1. A static per-instrument constant (current approach, but doesn't survive across forex pairs with different pip values), or
2. A function of instrument tick size / pip value (derived), or
3. Replaced by an ATR-based dynamic floor (handles volatility regimes natively but couples risk to recent market state).

I want your read. The current value will probably work for EUR/USD but fail on JPY pairs where pip ≠ 0.0001.

### 3c. `FeatureEngineer` → `FeaturePipeline` swap in analysis tools

Verified at:
- `src/analysis/optimize_brackets.py:25, 140`
- `src/analysis/failure_modes.py:24, 126`
- `src/core/retrainer.py:54, 138, 420, 423, 443, 445, 446, 517, 520`

**Important inconsistency:** in `core/retrainer.py`, the *imports* and *instantiation* moved to `FeaturePipeline`, but **the docstrings and log messages still reference `FeatureEngineer`** (lines 138, 420, 423, 443). Cosmetic, but a future grep for "FeatureEngineer" will find ghosts. The user's call whether to clean up.

---

## 4. The Inference Hot-Path (verified current state)

For your reference — this is what runs every M5 bar (or every tick on the watchdog path):

```python
# src/strategies/concrete_strategies/ml_strategy.py ~line 346
latest_features = features_df.select(self.feature_names).tail(1).to_numpy()
angel_prob = self.angel_trainer.predict_proba(latest_features)[0, 1]

if angel_prob < self.angel_threshold:
    return None  # Angel rejected

# Meta-labeling: append Angel prob, hand to Devil
angel_prob_col = np.array([[angel_prob]])
meta_features = np.hstack([latest_features, angel_prob_col])  # shape (1, 19)
devil_prob = self.devil_trainer.predict_proba(meta_features)[0, 1]

if devil_prob < self.devil_threshold:
    return None  # Devil veto
# else emit signal
```

`LiveOrchestrator._run_inference()` does the same pattern at `:1186-1204`.

---

## 5. What the user actually asked

They asked me to refresh them on ML pipeline + model changes. I gave them §1, §2, §3 above. Then they asked me to brief you.

**What I'd like from you (in priority order):**

1. **Retrainer feature dimensionality check.** Read `src/core/retrainer.py` and confirm whether `fit()` receives a DataFrame (activates schema-drift guard) or ndarray (leaves it dormant). If ndarray, propose the minimal change to make it a DataFrame.

2. **V5 model status.** Has the Angel/Devil pair been retrained on OANDA forex M5 data, or are we running the equities-era V3.3 models on forex bars? If the latter, this is the unspoken assumption that needs surfacing.

3. **`min_sl_pct` architectural ruling.** Static, derived from pip value, or ATR-dynamic? You're the one who can give an actual architectural opinion here — I've been treating it as a magic number.

4. **Whole-frame `drop_nulls` in retrainer.** `core/retrainer.py:520` calls `FeaturePipeline.clean_data(df)` without `feature_cols=`. Is this safe for V5 (technical indicators only) or a latent bug for V4 (sparse fundamentals)? My read is "safe for V5, broken for V4," but I want a peer check.

5. **`FeatureEngineer` docstring/log ghosts in `retrainer.py`.** Worth cleaning up now or defer?

Verify everything against the tree directly — I've cited file:line and your repo access is equivalent to mine, but bugs hide in the seams between what I checked and what I didn't.

---

## 6. Out-of-scope for this handoff

- The execution-safety / position-lock work (commit `7b0f089`) — separate report at `llm_reports/refactors/2026-05-14_execution-safety-and-locks.md`.
- The OANDA orchestrator + tick hook (PRs #57, #58, #59) — separate reports in `llm_reports/refactors/2026-05-16_*`.
- The data-provider critical fixes (`c29ed0a`) — `llm_reports/refactors/2026-05-14_data-provider-critical-fixes.md`.

If your read on items 1–5 takes you into any of those, flag and we'll widen scope.

---

**END HANDOFF.**
