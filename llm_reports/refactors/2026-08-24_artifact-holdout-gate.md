---
type: refactor
date: 2026-08-24
time: 14:00 PDT
agent: Kimi K3 (with Kimi K2.7-code-highspeed — two sessions worked this brief concurrently; this report reconciles both)
model: kimi-k3
trigger: m2m brief — the promotion gate validated fold models but shipped a full-data retrain (llm_reports/m2m/2026-08-24_overfit-gate-and-the-holdout-fix.md)
head: c7d5a93
scope: modifies-source
related:
  - m2m/2026-08-24_overfit-gate-and-the-holdout-fix.md
  - recons/2026-08-23_behavior-matrix-and-the-trend-high-hole.md
files_touched:
  - src/core/retrainer.py
  - tests/test_holdout_gate.py
  - src/core/README.md
  - tests/README.md
  - GLOSSARY.md
  - scripts/diagnose_5yr_holdout.py
---

# Artifact holdout gate — and what it says about the current model

## Context

The m2m brief (2026-08-24) established that the promotion gate measured fold
models but shipped a full-data retrain: the served artifact
(`models/forex_m15_wide`, trained to 2026-08-09) scored **65.3% in-sample and
17.6% out-of-sample** (Fisher p = 1.0e-4). The brief ordered a holdout that
judges the artifact actually served, plus honest-gate answers for the 2-year
and 5-year configs. This report is the reply: what was built, and what the
honest gate said.

## What was built

The six non-negotiable properties from the brief, as implemented in
`src/core/retrainer.py`:

1. **Carved first.** `_split_holdout` splits raw fetched bars by timestamp —
   chronologically last 18% (`RETRAIN_HOLDOUT_FRAC`, default 0.18; 0 disables)
   — before any feature engineering. The boundary bar belongs to the holdout.
2. **Folds on the remainder only.** Fold boundaries now scale with the actual
   span of the input frame (`validate_candidate`), so the same expanding 3-fold
   shape runs unchanged on the remainder. The frozen calibration threshold
   mechanism (Devil threshold swept on fold n−1, applied frozen to fold n) is
   untouched.
3. **The final model trains on the remainder too.** `main()` passes only the
   remainder into `engineer_features_and_labels` → `validate_candidate` → the
   final refit. The served artifact never sees a holdout row.
4. **The artifact is gated on the holdout.** After the fold gate passes, the
   holdout slice is engineered separately and scored with the frozen production
   threshold, tradeable-only, same bars as the fold gate (Brier ≤ 0.30, EV ≥
   0.0005, PF ≥ 1.20, scaled trade floor). Failing flips
   `report.gate_passed` to False and the artifact is rejected with the reason
   recorded. Passing folds is now necessary but not sufficient.
5. **Recorded in `metadata.json`.** `holdout.used/fraction/start_date/
   end_date/brier_score/expected_value/win_rate/profit_factor/trades/
   angel_proposed_trades/bypass_reason`, following the existing
   `sl_atr_multiplier`/`behavior_veto` pattern. A served model can always be
   checked against what it earned.
6. **Bypass is loud.** `RETRAIN_HOLDOUT_FRAC=0` or an empty holdout logs a
   ⚠️ warning at carve time and again at gate time, and the metadata records
   `used: false` with the reason (`disabled`, `empty holdout`, or
   `fold gate failed — no artifact to score`).

A format-string bug (`%.1%%` instead of `%.1f%%`) in the HOLDOUT METRICS log
line was fixed so the line prints correctly; the metrics themselves were always
recorded in `metadata.json`.

## Verification

- `PYTHONPATH=src:. python -m pytest -q` → **310 passed** (299 base + 11 new in
  `tests/test_holdout_gate.py`).
- `python -m compileall -q src/` clean.
- The central invariant is tested, not asserted: remainder and holdout frames
  are engineered separately and `rem_features.timestamp.max() <
  hold_features.timestamp.min()` must hold (`TestHoldoutNeverInTraining`).
- Split disjointness, chronological ordering, fraction accuracy, tradeable-only
  scoring (XAU/XAG excluded), and metadata recording (pass + bypass) all have
  unit tests using mock classifiers — no real training required.
- The live soak (`models/forex_m15_wide`) was untouched throughout; both
  evaluation runs wrote to side dirs via `RETRAIN_MODEL_DIR`.
- Every number in "What the honest gate says" was re-derived from the logs or
  the artifact's own `metadata.json`, not copied between drafts: 2-year
  holdout metrics from `models/forex_m15_holdout_2yr/metadata.json`, 5-year
  fold metrics and the rejection reason from `logs/holdout_5yr.log`, the
  diagnostic from `logs/diagnose_5yr_holdout.log`. The 54,119/135,953
  engineered-row counts come from the two runs' own "Final dataset" lines —
  an earlier draft gave both holdouts the same count, which was wrong.
- The diagnostic trained genuinely fast (~10s for 624K rows) because it ran
  alone on 12 cores; the gate runs' hour-plus wall times were two
  `n_jobs=-1` LightGBM jobs contending. Identical fetched row counts
  (982,841) and engineered counts (624,194) across both runs confirm the
  same window and pipeline.

## What the honest gate says

Both runs pinned to `RETRAIN_END_DATE=2026-08-09`, M15 bars, 1h HTF, the live
8-symbol basket, current brackets (2.0×/4.0× ATR), thresholds Angel 0.40 /
Devil 0.66. Logs: `logs/holdout_2yr.log`, `logs/holdout_5yr.log`. Only the
2-year run produced a side model dir (`models/forex_m15_holdout_2yr`, not
served); the 5-year run was rejected before saving.

### 1. The 2-year config (`RETRAIN_DAYS_BACK=730`)

Holdout: 54,119 rows engineered, **2026-03-29 → 2026-08-07** (18% of the raw
window, 71,931 rows before the chop veto).

| metric | fold gate | artifact on holdout |
|---|---:|---:|
| Brier | 0.1576 (mean, ≤ 0.30) | 0.2352 |
| EV | 1.4756 (mean, ≥ 0.0005) | 1.1716 |
| Profit factor | 2.2857 (Fold 3, ≥ 1.20) | 1.8286 |
| Win rate (macro) | 53.3% (Fold 3) | 47.8% |
| Trades | 266 pooled / 60 Fold 3 | 134 (floor ≈ 41.8) |
| Result | **PASSED** | **PASSED** |

The 2-year config passes an honest holdout. The gap between fold PF (2.29) and
holdout PF (1.83) is real but both clear the bar; the overfitting gap the old
gate missed is modest at this horizon, not the 65%→18% collapse seen on the
shorter-window served model.

### 2. The 5-year config (`RETRAIN_DAYS_BACK=1825`) — REJECTED by the fold gate, but see the diagnostic

Holdout reserved: 176,705 rows, **2025-09-13 → 2026-08-07** (135,953 rows
after engineering) — never scored by the gate, because no artifact survived
to score. Run: `logs/holdout_5yr.log`, exit 2; nothing saved (no
`models/forex_m15_holdout_5yr` exists, by design).

| metric | value | bar | verdict |
|---|---:|---:|---|
| Mean Brier | 0.1425 | ≤ 0.30 | ✓ |
| Mean EV | 1.5459 | ≥ 0.0005 | ✓ |
| Fold 3 PF | 1.4211 (macro WR 41.5%, 65 trades) | ≥ 1.20 | ✓ |
| Pooled OOS trades | **221** | ≥ 232 | ✗ |

Rejected on the pooled-trade floor, eleven trades short: "sample too small to
trust PF=1.4211". Two weak-confidence tells inside the folds: Fold 3's
threshold sweep bottomed out at **0.10** (the optimizer found no
discriminating cut and approved nearly everything), and per-fold trade counts
(116/40/65) shrank as training windows grew — more history made the Angel
*more* selective, not better.

**The diagnostic the brief's question 2 really asks for.** The gate run
produced no artifact, so `scripts/diagnose_5yr_holdout.py` replicates the
pipeline exactly (same pinned window, same carve, final refit on the
remainder, frozen Fold-3 threshold 0.10) and scores the holdout anyway — no
promotion, nothing saved. Reproduce:
`DATA_SOURCE=oanda RETRAIN_TIMEFRAME_MINUTES=15 RETRAIN_DAYS_BACK=1825 RETRAIN_END_DATE=2026-08-09 python scripts/diagnose_5yr_holdout.py`
(log: `logs/diagnose_5yr_holdout.log`).

| metric | fold gate said | would-be artifact on holdout |
|---|---:|---:|
| Brier | 0.1425 | **0.0859** |
| EV | 1.5459 | 1.6786 |
| Profit factor | 1.4211 | **2.8000** |
| Win rate (macro) | 41.5% | **58.3%** (49W/35L) |
| Trades | 221 pooled | 84 (floor 42) |

The would-be 5-year artifact would have **passed the holdout gate
comfortably** — the best honest numbers of any config tested. The only thing
between this config and promotion is a sample-size floor calibrated for the
old 60-day fold schedule. 58.3% against a 33.3% break-even on 84 trades is
binomially significant (p < 1e-5), but 84 trades is still a modest sample —
exactly the tension the floor exists to adjudicate.

### 3. Capacity control vs no signal — the read

- **The catastrophic 65.3%→17.6% split conflated two things**, and this work
  separates them. The measurement bug (full-data artifact judged on data it
  trained on) is fixed: the honest 2-year artifact holds 47.8% / PF 1.83 on
  untouched data, a fold→holdout gap of ~20%, not a collapse. The August 2026
  live collapse happened *after* every window tested here ends (2026-08-07) —
  it is a regime break, and no gate can see past its own end date.
- **The Devil is capacity pointed at noise.** Its Fold-3 separation gap in
  the 2-year run is −0.0081 (mean prob 0.9257 on wins vs 0.9338 on losses):
  out-of-sample, the meta-labeler cannot distinguish wins from losses at all.
  Every passing metric rides on the Angel's selectivity. If capacity control
  is tried anywhere, it is there — shrink the Devil (fewer leaves, larger
  `min_child_samples`) or ablate it entirely and let the Angel + frozen
  threshold trade. That is a concrete, gate-evaluable experiment.
- **Capacity control is not the binding constraint on the Angel.** The 2-year
  config generalizes at current settings (n_estimators=200, lr=0.05,
  max_depth 10/8, num_leaves 63/31, min_child_samples 20). The 5-year
  config's artifact is *better* on the holdout, not worse — more data did not
  buy overfitting, it bought selectivity.
- **The live risk is regime dependence, not capacity.** The gate now honestly
  certifies in-regime generalization. What it cannot certify is next month's
  regime — the 9,183 graded live decisions and the 17.6% August record are
  the evidence that the regime broke. That argues for regime-aware gating or
  faster retraining cadence, not regularization.

(An earlier draft of this section cited a "60-day sanity run (PF 1.05, pooled
trades 177 < floor 225)". No such run exists in `logs/` — the claim was
unsupported and has been removed rather than left to be quoted later.)

## Risk & follow-ups

- The gate is expected to start rejecting models it used to pass — per the
  brief, that is the point, not a bug to tune away. The 5-year result is an
  example: it used to pass the old gate (full-data retrain, no holdout) but
  fails the honest gate on sample size.
- **The pooled-trade floor was calibrated for the 60-day fold schedule.**
  With folds now scaling to multi-year spans, the Angel's selectivity
  tightens and pooled fold trades fall even though the artifact itself earns
  PF 2.80 on an 84-trade holdout. Whether the floor should scale with window
  length — or whether the artifact-holdout score should carry more weight
  than pooled fold counts at long horizons — is now the live gate-design
  question. Do not simply lower the floor; that is the bar moving to fit the
  candidate.
- **The served model is now formally unvalidated.** `models/forex_m15_wide`
  predates the holdout gate and its live August record (17.6%) is worse than
  what the honest 2-year artifact earned in-regime (47.8%). The 2-year side
  artifact (`models/forex_m15_holdout_2yr`) is the only model with an honest
  scorecard. Whether to promote it to service is a human decision — it passed
  the gate, but the regime it was validated in ended 2026-08-07 and the live
  evidence says the regime has changed since.
- Two Kimi sessions worked this brief concurrently (the implementation and
  the evaluation runs were shared state in the working tree). The retrainer's
  WR log line now multiplies by 100 for display (`src/core/retrainer.py:2976`)
  — an uncommitted one-line fix from the other session, included in this
  report's commit.
- `models/forex_m15_holdout_2yr` is an experiment side dir; it is not served
  and can be deleted once a promotion decision is made. (No
  `models/forex_m15_holdout_5yr` exists — that run was rejected before
  saving, which is itself the gate working.)
