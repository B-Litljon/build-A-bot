---
type: refactor
date: 2026-05-23
time: 11:49 PDT
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: Verify Gemini's 2026-05-23 "promoted" forex retrain; numbers didn't reproduce and Devil's own diagnostic read NO SIGNAL on the gating fold. Closed the leakage holes that let the gate falsely pass.
head: 51076de7c6b2bd74ad3981c2cb0bfe2d1a93b173
scope: modifies-source
related:
  - handoffs/2026-05-23_forex-retraining-completion.md
  - handoffs/2026-05-22_retrainer-multi-asset-review.md
files_touched:
  - src/core/retrainer.py
---

## Context

Gemini ran the V5 forex retrainer on 2026-05-23 with hand-tuned hyperparameters
(`min_samples_leaf=1`, `class_weight=None`) and produced models flagged as
"promoted" with a glowing handoff
([handoffs/2026-05-23_forex-retraining-completion.md](../handoffs/2026-05-23_forex-retraining-completion.md)).
The user asked me to verify before accepting them for the OANDA live soak. Two
things were wrong on inspection:

1. Gemini's own report contained two mutually inconsistent metric sets
   (Executive Summary said PF=2.666, Final Validation Results said PF=5.00).
2. The "passing" gate stood on a 14-trade Fold 3 sample with R:R=6 — the
   smallest-sample regime where high-PF lottery wins are inevitable.

A re-run with `PYTHONPATH=src` (Gemini's exact invocation) on a 10-hour-shifted
rolling window produced a third set of numbers (PF=3.33, threshold=0.64) — and
the Devil's own separation diagnostic read **"NO SIGNAL -- Devil cannot
distinguish wins from losses"** for Fold 3 (`-0.0092` gap). The gate passed
anyway because PF mechanics on tiny samples with asymmetric R:R can blow up
from a handful of lucky wins.

This refactor closes the three leakage / sample-size holes that let that
false pass happen and restores honest gate semantics.

## Investigation

**Gemini's diff to `get_hyperparameters` (retrainer.py:102–124).** Forex was
moved from `(class_weight="balanced", min_samples_leaf=50)` to
`(class_weight=None, min_samples_leaf=1)` after a hyperparameter sweep across
multiple configs, with each candidate evaluated by its Fold 3 separation gap.
That sweep is itself the leakage: choosing hyperparameters to maximize a
validation-set metric makes that metric no longer OOS.

`min_samples_leaf=1` lets each RF tree's leaves contain a single training
sample. With only 1,043 Angel-approved training rows after Phase 5.5 filtering
(retrainer.py:760–765), the Devil memorizes the population. Confirmation came
from the run logs: Devil training accuracy = **0.939** with leaf=1 vs **0.753**
with leaf=20.

**Threshold-on-test leakage at retrainer.py:1158–1175.** `_find_optimal_threshold`
sweeps Devil thresholds (0.10–0.66) on each fold's validation data and picks
the EV-maximizing value. For Fold 3, that threshold is *both* selected on the
validation set *and* used to compute the gating metrics (PF, Brier, EV). The
threshold is then saved as the production parameter. This is the textbook
"tuning on the test set" antipattern. Separating the calibration fold from the
gating fold was the fix.

**Sample-size mechanics on the PF gate (retrainer.py:1256–1273).** With
TP=3.0×ATR and SL=0.5×ATR (forex `RiskProfile.for_asset_class("forex")`), each
win pays 6× each loss. On 14–18-trade samples, 6 wins vs 8 losses gives
PF=36/4=4.5, which sails through the 1.2 gate while saying nothing about
generalizability. The 100-trade floor is the minimum at which PF starts to
mean what people think it means.

**Reproducibility test.** I re-ran with PYTHONPATH=src on the same code, same
seed (random_state=42), 10 hours later. `datetime.now(utc)` advanced 10 hours,
shifting the 60-day rolling window by the same. Fold 3 separation gap collapsed
from Gemini's claimed `+0.0718` (SIGNAL) to `-0.0092` (NO SIGNAL). Threshold
moved 0.62 → 0.64. PF moved 2.666/5.00 → 3.33. All from a 10-hour data shift.
That is not a stable model.

## Findings / Changes

Six code changes total, all in `src/core/retrainer.py`. The first three close
the leakage holes in the gate; the next three are infrastructure fixes that
surfaced when expanding the experiment to XAU (gold) per user request.

### 1. `get_hyperparameters` — leaf floor restored to 20 for forex

**Before** (Gemini, retrainer.py:109, 118):
```python
"min_samples_leaf": 1 if asset_class == "forex" else 50,
```

**After:**
```python
"min_samples_leaf": 20 if asset_class == "forex" else 50,
```

Leaf=50 starves forex (Angel only approves ~0.3% of rows at that floor).
Leaf=1 memorizes. Leaf=20 is the defensible middle ground; in the post-fix
run, Devil training accuracy normalized to 0.753 (vs 0.939 with leaf=1),
confirming the memorization signal is gone.

`class_weight=None` for forex is kept — Gemini's reasoning there (avoid
inflating Angel proposals on imbalanced base rates) is sound.

### 2. `MIN_OOS_TRADES_FOR_PF = 100` — sample-size floor on the PF gate

New constant at retrainer.py:158–163. New rejection check at the gate aggregation:

```python
if final_total_trades < MIN_OOS_TRADES_FOR_PF:
    rejection_reasons.append(
        f"Fold {n_folds} OOS trades {final_total_trades} < "
        f"{MIN_OOS_TRADES_FOR_PF} minimum — sample too small to trust "
        f"PF={profit_factor:.4f}"
    )
```

This is the gate that fired in the verification run below — 18 OOS trades
correctly rejected.

### 3. `calibration_threshold` — Fold 2 picks, Fold 3 evaluates

`validate_candidate` now uses a two-phase threshold strategy:

- **Folds 1 .. n_folds-1**: sweep `_find_optimal_threshold` (in-fold, unchanged).
- **Fold n_folds-1**: freeze that sweep result as `calibration_threshold`.
- **Fold n_folds**: use the frozen calibration_threshold — **no sweep**. The
  gating metrics (PF, Brier, EV) are computed at this frozen threshold, which
  was selected on data the Fold 3 evaluation cannot see.

The fallback (when Fold 2 had zero approvals) is a fixed 0.50 with a warning
in the log, not a sweep.

Code lives at retrainer.py:1152–1196.

### 4. Defensive single-class Devil `predict_proba`

`devil_model.predict_proba(meta_df)[:, 1]` crashed with `IndexError: index 1
is out of bounds for axis 1 with size 1` on the first XAU retrain. With only
6 Angel-approved rows in the calibration fold and all 6 of class 1 (survived),
the Devil RF degenerated to a single-class model and `predict_proba` returned
shape `(n, 1)` instead of `(n, 2)`. Fixed by checking `devil_proba_full.shape[1]`
and substituting a constant probability (1.0 if the single class is `survived`,
else 0.0). Code at retrainer.py:1092–1110.

### 5. `RETRAIN_DAYS_BACK` env override

`DAYS_BACK = 60` was a module constant with no override path; XAU's single-symbol
data was too sparse at 60 days. Promoted to `int(os.getenv("RETRAIN_DAYS_BACK", "60"))`
at retrainer.py:63. Default behavior unchanged; CLI can now extend the window
per-run.

### 6. Scaled `fold_configs` for arbitrary `DAYS_BACK`

The 60-day fold cutoffs `[(30,40),(40,50),(50,60)]` at retrainer.py:961–965
were hardcoded. With `DAYS_BACK=180`, the validation sets stayed pinned to
calendar days 30–60 and the trailing 120 days (67% of the fetched data) were
silently discarded. Rewrote as fractions that reproduce the legacy 60-day
schedule exactly (`DAYS_BACK // 2`, `DAYS_BACK * 2 // 3`, `DAYS_BACK * 5 // 6`,
`DAYS_BACK`) and now scale correctly. At DAYS_BACK=180, Fold 3 train rows
grew from 46k → 143k.

## Verification

Re-ran the full retrainer post-fix on the same OANDA data window:

```
$ DATA_SOURCE=oanda PYTHONPATH=src pipenv run python -m src.core.retrainer
```

Final gate output:

```
Mean Brier Score : 0.2941 (threshold ≤ 0.3)
Mean EV          : 0.469697 (threshold ≥ 0.0005)
Profit Factor    : 1.2727 (threshold ≥ 1.2, Fold 3 OOS)
OOS Trades       : 18 (threshold ≥ 100, Fold 3 sample-size floor)
Gate Result      : FAILED 🚫

Rejection: Fold 3 OOS trades 18 < 100 minimum — sample too small to
trust PF=1.2727
```

Comparison across the three runs:

| Metric                | Gemini §1 | Gemini §3 | My verify run | Post-fix run |
|-----------------------|-----------|-----------|---------------|--------------|
| Mean Brier            | 0.2934    | 0.2542    | 0.2733        | 0.2941       |
| Mean EV               | 0.518     | 0.8705    | 0.7176        | 0.4697       |
| Fold 3 PF             | 2.666     | 5.00      | 3.3333        | 1.2727       |
| Fold 3 OOS trades     | —         | 14        | 16            | 18           |
| Fold 3 separation gap | +0.0718   | +0.0718   | -0.0092       | +0.0199      |
| Threshold (saved)     | 0.62      | 0.62      | 0.64          | 0.58 (frozen)|
| Devil train acc       | —         | —         | 0.887         | 0.753        |
| Verdict               | PROMOTED  | PROMOTED  | PROMOTED      | **REJECTED** |

The post-fix Devil separation gap reads `+0.0199` — still below the diagnostic's
own SIGNAL bar of `+0.05` and inside its NO-SIGNAL band (≤0.02). The honest
verdict at M1 with the current feature set is **no demonstrable edge**. The
production weights on disk (the prior Gemini/verify run from 11:10) were
correctly retained — rejection means no overwrite.

### Additional verification: XAU_USD (gold) experiment

User requested a pivot to gold since their manual scalping worked there.
Two runs, both gate-rejected honestly:

```
$ DATA_SOURCE=oanda RETRAIN_SYMBOLS=XAU_USD RETRAIN_DAYS_BACK=60  pipenv run python -m src.core.retrainer
$ DATA_SOURCE=oanda RETRAIN_SYMBOLS=XAU_USD RETRAIN_DAYS_BACK=180 pipenv run python -m src.core.retrainer
```

The 60-day run crashed on the single-class `predict_proba` bug (fix #4 above).
The 180-day run completed cleanly but exposed the hardcoded fold-cutoff bug
(fix #6). After both fixes, the 180-day scaled-fold run produced:

```
Mean Brier 0.2834 ✅ | Mean EV 0.3251 ✅ | PF 1.4286 ✅
OOS Trades 12 ❌ (< 100 floor)
Fold 3 separation gap −0.0227 (NO SIGNAL — wins have lower mean prob than losses)
Verdict: REJECTED
```

The 12-trade Fold 3 sample failed the sample-size floor exactly as designed.
More importantly the underlying separation signal is *negative* — the Devil is
mildly anti-predictive on this slice. The Angel/Devil cascade at M1 on the
V3.3 18-feature vector does not surface scalping edge on XAU in 180 days of
OANDA data. This is informative, not catastrophic: the integrity fixes are
*working*, and they're correctly refusing to deploy weights that have no
demonstrable predictive power.

### Final verification: V3.5 features + 8-instrument basket, 365 days

After V3.5 session features + the volatile basket landed (commit `149ef8b`)
and the basket was extended to 8 instruments, one more retrain confirmed the
signal-quality story across the full dataset:

```
$ DATA_SOURCE=oanda RETRAIN_DAYS_BACK=365 PYTHONPATH=src pipenv run python -m src.core.retrainer
```

Instruments: `XAU_USD, XAG_USD, GBP_JPY, AUD_JPY, EUR_JPY, NZD_JPY, GBP_AUD, GBP_NZD`.
Raw bars fetched: 2,914,650 across 5 instruments × 365d.

Per-fold (strict OOS at frozen Fold-2 threshold 0.66 for Fold 3):

| Fold | Angel proposals | Devil approved | Brier  | Separation gap | Verdict   |
|------|-----------------|----------------|--------|----------------|-----------|
| 1    | ~30             | ~6             | ~0.19  | +0.0682        | SIGNAL ✅ |
| 2    | 126             | 14             | 0.1501 | +0.0829        | SIGNAL ✅ |
| 3    | **210**         | 13             | 0.1774 | +0.0751        | SIGNAL ✅ |

Fold 3 strict-OOS metrics:
- 210 Angel proposals — first run of the session to clear 100 proposals honestly
- Macro WR 61.5%, Survival WR 76.9%, EV +1.31
- PF 3.20 ✅

Aggregate gate readout:
```
Mean Brier 0.20  ✅ | Mean EV 1.27  ✅ | PF 3.20 ✅
OOS Trades 13 ❌ (< 100 floor)
Verdict: REJECTED (sample-size only)
```

**This is the strongest signal-quality result of the session.** All three
folds show SIGNAL DETECTED (separation > 0.05). Mean Brier 0.20 is the best
Devil calibration of any run today. The integrity-fixed gate still rejects
deployment because the Devil at the frozen 0.66 threshold is highly
selective — only 13 of 210 Fold 3 Angel proposals cleared it. 13 OOS trades
over 60-day val ≈ 80–100 trades/year live: a low-frequency, high-conviction
scalper rather than a true HFT setup.

The gate is doing its job: it's neither false-passing on lottery PF (the
fixed problem) nor refusing real signal (the new evidence shows separation
holds at strict OOS, frozen threshold, across all three folds). It is
correctly insisting that 13 OOS trades — even with PF 3.20 — is too few to
commit production weights. The next path forward is feature-side or
basket-side (more proposals reaching the gate), not threshold relaxation.

Other things checked:

- **Walk-forward folds are structurally clean.** TimeSeriesSplit at
  retrainer.py:688 produces strictly chronological folds, OOF angel probs are
  generated per-fold (retrainer.py:694–705), head-fill uses only the first
  train window (retrainer.py:709–719). No future-data peeking in the fold
  mechanics. The leakage was downstream, in hyperparameter selection and
  threshold selection.
- **Survival vs macro target separation is correct.** Devil trains on 5-bar
  SL survival (retrainer.py:365–428), threshold sweep optimizes EV on 45-bar
  macro outcomes (retrainer.py:866–873). Both targets are computed without
  future-bar peeking past their declared windows.
- **Production threshold is saved correctly.** `save_threshold`
  (retrainer.py:1491–1521) writes the value returned by `validate_candidate`,
  which is now the frozen Fold-2 calibration value rather than the Fold-3
  in-sample sweep.

## Risk & follow-ups

- **The M1 Angel/Devil cascade with the V3.3 feature set does not show
  scalping edge on either tame forex or XAU.** Two honest rejections in one
  session. The pattern is consistent: Angel approves <0.2% of rows, leaving
  too few proposals to gate honestly even with a 180-day window. Next
  investigations are feature-side or architecture-side, not hyperparameter-side:
  - Add session-aware features (London/NY overlap, weekend gaps, NFP flag).
  - Test M5 / M15 with the corrected gates — the M5 washout Gemini described
    was measured against the same broken gates that just produced a false pass
    at M1, so it deserves a re-measurement under the new pipeline.
  - Lower `ANGEL_THRESHOLD` (currently 0.40 at retrainer.py:143) for higher
    proposal volume. Risk: more noise into the Devil's training set.
  - Add gold-specific features (DXY correlation, real-yield proxies, USD
    strength). The V3.3 feature set was developed for equities microstructure;
    it may not carry across asset class.
  - Try multi-instrument metals (XAU + XAG + XPT) for more bars per fold.
  - Per [[lightgbm-on-tame-pairs-and-indices]] memory: tame fiat pairs may be
    better served by the dormant V4 LightGBM ranker on slower horizons.
- **The M1 timeframe change is now load-bearing.** `run_oanda.py:89` defaults
  to `--granularity 1`. If we don't find edge at M1, this default may need to
  revert. Tracked as live concern; no rollback now since legacy V3.3 success
  was on M1-ish equities scalping.
- **Equities path is unaffected.** Equities Angel keeps `leaf=50, class_weight="balanced"`,
  Devil keeps `leaf=50, class_weight=None`. The branch in `get_hyperparameters`
  forks on `asset_class == "forex"` only.
- **Multi-window stability test still owed.** Recommended: run the corrected
  retrainer with `end_date` offsets of 0, -7, -14, -21 days. If any of those
  passes the new gate, we have a more-robust signal claim than a single window.
- **Equities PF=3.4 → 0.4 swing from the prior conversation likely shares the
  same root cause** (threshold-on-test plus small-sample PF mechanics). The
  fixes here apply to equities too; a fresh equities run is owed to confirm.

## Files touched

- `src/core/retrainer.py`
  - Line 63: `DAYS_BACK` promoted to `RETRAIN_DAYS_BACK` env override.
  - Lines 102–129: `get_hyperparameters` — `min_samples_leaf` 1→20 for forex
    on both Angel and Devil; added rationale comment.
  - Lines 158–163: new `MIN_OOS_TRADES_FOR_PF = 100` constant with rationale.
  - Lines 961–973: rewrote `fold_configs` to scale fractionally with
    `DAYS_BACK` (preserves legacy 60-day schedule exactly).
  - Lines 977–983: new `calibration_threshold: Optional[float]` placeholder
    in `validate_candidate`.
  - Lines 1092–1110: defensive single-class `predict_proba` handling for the
    Devil model.
  - Lines 1152–1196: rewrote threshold-selection block. Folds 1..n-1 sweep
    in-fold and freeze on fold n-1; fold n uses the frozen threshold strictly OOS.
  - Lines 1285–1290: new rejection check enforcing the 100-trade PF floor.
  - Lines 1301–1304: updated gate-summary log to include OOS trade count line.

Files read (not modified) during the investigation:

- `src/strategies/base.py` — confirmed Signal dataclass shape unchanged.
- `src/ml/feature_pipeline.py` — confirmed FeaturePipeline path used by retrainer.
- `src/data/factory.py` — confirmed OANDA path resolves correctly.
- `run_oanda.py` — confirmed `--granularity 1` is the live default; matches
  retrainer's M1 training.
- `llm_reports/handoffs/2026-05-23_forex-retraining-completion.md` — Gemini's
  handoff being verified.
