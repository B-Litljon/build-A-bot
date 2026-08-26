---
type: refactor
date: 2026-08-24
time: 19:45 PDT
agent: DeepSeek V4 Pro
model: deepseek-v4-pro
trigger: "m2m brief: the holdout gate is correct but its verdict flips with the clock (llm_reports/m2m/2026-08-24_holdout-stability-fixes.md)"
head: 7d68aaf
scope: modifies-source
related:
  - m2m/2026-08-24_holdout-stability-fixes.md
  - audits/2026-08-24_holdout-gate-audit.md
  - refactors/2026-08-24_artifact-holdout-gate.md
files_touched:
  - src/core/retrainer.py
  - tests/test_holdout_gate.py
  - scripts/diagnose_5yr_holdout.py
  - scripts/run_stability_batch.sh
  - GLOSSARY.md
  - src/core/README.md
  - scripts/README.md
  - tests/README.md
---

# The holdout verdict no longer depends on the clock

## Context

The brief (from Claude Opus, 2026-08-24) ordered three fixes to Kimi K3's
artifact holdout gate, all verified leak-free by the audit: (1) make the
verdict stable — the same configuration passed at holdout PF 1.444 and failed
at 0.982 an hour apart; (2) purge the ~360 mislabelled rows at the
remainder/holdout boundary; (3) score the holdout even when the fold gate
fails. It also demanded the fixed gate be demonstrated at three or more
pinned endpoints for the 2-year and 5-year configs, and a judgement on whether
the holdout can be decisive at this data volume at all. "Nothing passes a
stable gate" was declared a publishable answer, and tuning until something
passes was forbidden.

Everything the brief said to preserve was preserved. The no-leak property is
untouched (the remainder purge only removes rows), the Devil threshold is
still frozen from fold n-1 and passed into the holdout, `feature_stats` still
comes from the remainder, and a failing holdout still sets
`report.gate_passed=False` which `promote_or_reject` honours.

## Finding 1 fix — gate the holdout PF on its exact confidence bound

The root cause named in the audit: the holdout floor was 42 trades
(`300 × 0.18 × 0.776`) against the fold gate's 233. Raising the floor would
just move the cliff; raising `HOLDOUT_FRAC` spends training data; multi-window
promotion multiplies retrain cost. The fix chosen: keep the floor's job but
make it continuous and exact.

The verdict now gates on the **Clopper-Pearson one-sided 95% lower bound** of
the macro win rate, mapped through the PF formula
(`_holdout_pf_lower_bound`, `src/core/retrainer.py:1844`). A promotion
requires the holdout to exclude break-even at 95% confidence. Consequences,
by construction:

- For independent trades, a true break-even artifact passes with probability
  ≤ 5% at **any** trade count. The 45-bar macro walks overlap in price, so
  the bound is conservative in practice rather than a literal coverage
  guarantee — the safe direction for a promotion gate. The old
  point-estimate gate's false-pass rate was measured at the audit's sample
  sizes and is **47-52%** — a coin flip
  (`P(Bin(n, 0.375) ≥ ⌈0.375n⌉)` for n = 42, 55, 62, 82, 134 → 0.526, 0.509,
  0.470, 0.519, 0.479).
- Sample size enters through the bound, not a cliff: a perfect 3-for-3
  holdout is rejected (bound 1.1666 < 1.2) while a perfect 4-for-4 passes
  (1.7941). Clopper-Pearson rather than Wilson because Wilson under-covers
  below ~40 trades — exactly the regime that used to flip.

On the audit's own three windows, the stable gate is unanimous where the old
gate flipped:

| window end | trades | point PF | old verdict | 95% lower bound | stable verdict |
|---|---:|---:|---|---:|---|
| 2026-08-24 ~22:00 | 62 | 1.444 | PASS | 0.9111 | **FAIL** |
| 2026-08-24 ~23:00 | 82 | 0.982 | FAIL | 0.6442 | **FAIL** |
| 2026-08-23 (pinned) | 55 | 1.333 | PASS | 0.8112 | **FAIL** |

The verdict no longer depends on which hour the retrain runs. Brier and EV
keep their point bars — both were stable across the audit's windows — and NaN
metrics (zero-trade holdouts) fail loudly rather than passing vacuously
(`_holdout_verdict`, `src/core/retrainer.py:1892`). The trade floor was
removed from the holdout path entirely; the fold gate's pooled floor is
untouched. `HOLDOUT_PF_CONFIDENCE = 0.95` is deliberately not env-tunable: the
promotion gate is not a knob.

## Finding 2 fix — purge the unresolvable tail on both sides

The remainder's last `max_hold` (45) bars per symbol resolve their macro
labels against bars that now live in the holdout; every one reads
"timeout → loss" regardless of what price did. `_tail_cutoff_by_symbol`
(`src/core/retrainer.py:1927`) derives the per-symbol cutoff from the **raw**
series — the walk's true domain, before the chop veto removes rows — and
`_purge_boundary_tail` (`:1950`) drops engineered rows at or after it. Logged
per run: the six demonstration runs dropped 200-305 remainder rows each (of
≤ 360 possible; the veto had already removed the rest), against 500k-600k
engineered rows — the ~0.2% the audit predicted.

The same purge applies to the holdout's own tail before scoring
(`_score_artifact_holdout`, `:1978`): those rows' walks run off the end of
the fetched window and would otherwise read as guaranteed losses, a
pessimistic bias on the gate's own evidence. The effect is measurable, not
cosmetic: on the pinned 08-09 2-year window the unpurged holdout read PF
1.8286 on 134 trades (47.8% WR, the recon's number), of which 24 trades sat
in the unresolvable tail — 20 of them guaranteed "losses". Purging moves the
honest number to PF 2.40 on 110 trades. The head-side embargo stays as the
audit found it — indicator warm-up consumes the holdout's first ~2 days via
NaN cleaning, which now has a comment stating that this is the deliberate
behaviour rather than an accident.

## Finding 3 fix — the holdout is scored even when folds fail

`main()` no longer ties holdout scoring to `report.gate_passed`. When the
fold gate fails, the Fold 3 models are scored on the holdout and the metrics
are logged and recorded with `diagnostic_only=True` on the
`ValidationReport.holdout`; the fold verdict stands and the holdout cannot
rescue it. A rejected candidate now produces the one comparison that mattered
— fold-gate numbers next to holdout numbers — and the 5-year config no longer
needs `scripts/diagnose_5yr_holdout.py` to see its own holdout (that script
still exists for the *final artifact* score, which the fold-fail path cannot
produce). `metadata.json` additionally records `wins`, `pf_lower_bound`,
`pf_confidence`, and `purged_tail_rows` — but not `diagnostic_only`, which is
unreachable there: metadata is only written on promotion, and promotion
implies the fold gate passed.

## The demonstration — six full gate runs, three pinned endpoints

`scripts/run_stability_batch.sh` runs the fixed gate exactly as production
runs it (same entry point, same env, side `RETRAIN_MODEL_DIR`s), for the
2-year config (`RETRAIN_DAYS_BACK=730`) and the 5-year config (1825), each
pinned to 2026-08-09, 2026-08-16, and 2026-08-23. Runs are sequential, one
at a time: this box has 6 physical cores and any two-way overlap collapses
LightGBM fit speed ~100× once both jobs are mid-refit (3.5s → 9+ min per
fit, measured), so solo runs are both faster (~3-4 min each) and keep every
pin measured under identical conditions. The 2yr@08-09 fold metrics match
the recon's bit-for-bit (PF 2.2857, 128/56, pooled 266, floor 232), so the
before/after comparison is like-for-like.

| config | pin | fold gate | holdout artifact | 95% lower bound | stable verdict |
|---|---|---|---|---|---|
| 2yr | 08-09 | PASS (pooled 266 ≥ 232) | PF 2.4000, WR 54.5%, 110t (60w) | 1.7218 | **PASS** |
| 2yr | 08-16 | PASS (pooled 290 ≥ 232) | PF 2.5833, WR 56.4%, 110t (62w) | 1.8516 | **PASS** |
| 2yr | 08-23 | PASS (pooled 340 ≥ 232) | PF 2.4000, WR 54.5%, 110t (60w) | 1.7218 | **PASS** |
| 5yr | 08-09 | PASS (pooled 238 ≥ 232) | PF 3.2353, WR 61.8%, 89t (55w) | 2.2159 | **PASS** |
| 5yr | 08-16 | FAIL (230 < 232) | PF 2.0000, WR 50.0%, 88t (Fold 3 diag) | 1.3767 | **REJECT (fold floor)** |
| 5yr | 08-23 | FAIL (227 < 233, PF 1.1304) | PF 2.4500, WR 55.1%, 89t (Fold 3 diag) | 1.6897 | **REJECT (fold floor)** |

The holdout verdict — the thing this brief asked to stabilise — is
**unanimous across all six windows and both configs**: PASS everywhere, with
95% lower bounds of 1.38-2.22 against the 1.2 bar. No clock dependence. The
08-09 and 08-23 2yr windows happen to show identical trade counts (110t,
60w); the windows are distinct (7-day shift, different Brier/EV), so this is
coincidence — the Devil's selectivity is stable — not a fetch bug.

The two 5yr rejections are the **fold gate's** knife-edge, not the
holdout's: pooled trades sit at 238 / 230 / 227 around the 232 floor as the
noise-driven threshold sweep (fold 2: 0.64 / 0.10-range) shifts approvals by
a handful. The recon flagged exactly this ("the pooled-trade floor is now
the live gate-design question") and it is outside this brief. Finding 3's
diagnostic earns its keep here: both rejected 5yr candidates are scored on
the holdout anyway, and both would have cleared it (bounds 1.38 / 1.69) —
the fold floor, not the artifact, is what is rejecting 5-year configs, two
trades and five trades short.

## What the stable gate says (section 8 answers)

**The 2-year config passes a stable gate, at all three pinned endpoints,
with margin.** Holdout PF point estimates 2.40 / 2.58 / 2.40 on ~110 trades;
the 95% lower bounds are 1.72 / 1.85 / 1.72, comfortably above 1.2. The
verdict would survive a 30% haircut in observed performance. Note the
tail-purge correction is doing real work: the recon's unpurged holdout on
the same 08-09 window read PF 1.8286 on 134 trades (47.8%) — 24 of those
trades sat in the last 45 bars of the fetched window, 20 of them
guaranteed "timeout → loss" labels, and removing them moves the honest
number to PF 2.40 on 110 trades. The old holdout was pessimistically biased
by roughly 0.6 PF, not optimistically.

**The 5-year config is holdout-clean but fold-floor-blocked.** Its artifact
clears the holdout bound at all three pins (2.22 / 1.38 / 1.69), but the
fold gate admits it at only one of three (238 vs 230 vs 227 pooled against
~232). Promotion on a single unpinned run would still be wrong — for the
fold gate's reasons now, not the holdout's.

**Can the holdout be decisive at this data volume?** Yes for strong edges,
no for marginal ones. The decisiveness frontier — the minimum trade count at
which an artifact whose observed win rate equals a given point PF clears the
95% bound at 2:1 — is:

| point PF | win rate | minimum trades to clear |
|---:|---:|---:|
| 1.20 (break-even) | 37.5% | never |
| 1.50 | 42.9% | 230 |
| 1.83 | 47.8% | 64 |
| 2.00 | 50.0% | 43 |
| 2.80 | 58.3% | 20 |

A holdout carved at 18% of a 2-year window yields ~110 trades, so it can
certify a true PF ≥ ~1.8 and never certifies a true PF of 1.5 (230 trades is
beyond any practical 18% carve). At the audit's 365-day volume (55-82
trades) the same arithmetic explains the instability: even a PF-1.8-strength
edge sits right at the 64-trade boundary, and the observed ~1.33-1.44 PFs
would need 333+ trades to clear — they were never going to. The holdout is a
decisive instrument for the configs that actually have edge (2yr, 5yr) and a
correctly conservative one for those that do not (365-day). Promotion at
short horizons needs a different kind of evidence — more window, not a
looser bar.

## Verification

- `PYTHONPATH=src:. python -m pytest -q` → **325 passed** (310 prior + 15 new:
  exact CP bound values, verdict decisions including NaN handling, boundary
  purge mechanics, metadata fields, `TestPermanentLeakGuard` — the audit's
  instrumented leak check made permanent — and `TestMainWiringLeakGuard`,
  which runs `main()` end to end with a recording `refit_models` and asserts
  no training frame contains a holdout timestamp).
- `python -m compileall -q src/` clean.
- Independent review (separate agent, uncommitted diff against the brief):
  verdict SAFE — all six preserved properties confirmed, CP arithmetic
  recomputed and matched to 6 decimals, no constraint or dead-end violations.
  Its full findings were folded in: the coverage language notes the
  overlapping-walk caveat, `diagnostic_only` no longer claimed in
  `metadata.json`, the diagnose script applies the boundary purge (and no
  longer carries an unused variable), the main()-wiring test closes the
  leak-guard gap, the batch script's rejection exit-code line now prints
  under `set -e`, the `_tail_cutoff_by_symbol` docstring states the
  short-symbol rationale honestly, `pf_lower_bound`'s None-vs-0.0 comment is
  corrected, and an empty engineered holdout under HMM no longer reaches
  `predict_regime_probs`.
- 2-symbol smoke run exercised every new path (purge, fold-fail diagnostics,
  CI verdict) end to end; rejected with exit 2, nothing written.
- Demonstration layout: six sequential solo runs (one at a time, LightGBM
  default threads, ~3-4 min each). Parallel layouts were tried and measured
  worse — a 3-way/4-OMP-thread layout ran fits ~300× slower and a 2-way
  layout collapsed ~100× once both jobs were mid-refit — so the final
  numbers come from solo runs only, which also keeps every pin measured
  under identical conditions. The 2yr@08-09 fold metrics match the recon's
  bit-for-bit (PF 2.2857, pooled 266, floor 232), confirming fidelity.
- Promoted side artifacts verified: `models/forex_m15_stability_2yr_20260809/
  metadata.json` carries the full CI evidence (`wins` 60, `pf_lower_bound`
  1.7218, `pf_confidence` 0.95, `purged_tail_rows` 315).
- The live soak and `models/forex_m15_wide` were untouched throughout.

## Risk & follow-ups

- **The fold gate's pooled-trade floor (~232) is now the binding constraint
  for the 5-year config, and it is itself knife-edged**: pooled trades came
  in at 238 / 230 / 227 across the three pins around a 232 floor, with a
  noise-driven fold-2 threshold sweep (0.64 at one pin, ~0.10 at another)
  deciding which side of the line each run lands. The holdout clears at all
  three pins either way (bounds 2.22 / 1.38 / 1.69), so this is the fold
  gate rejecting the 5-year config two trades short — the recon flagged the
  floor as a live gate-design question, and this demonstration confirms it
  is now the instability that matters. Deliberately left alone here: moving
  the floor without a new evidential justification is bar-moving.
- **A pass still needs a human.** The stable gate certifies in-regime
  generalization; the August live collapse happened after every window
  tested here ends, and no gate can see past its own end date.
- **The 2-year side artifacts are promotion-ready evidence.** All three
  pinned 2-year runs promoted into `models/forex_m15_stability_2yr_*` with
  holdout bounds 1.72-1.85. Whether to serve them remains a human decision
  (mirroring the recon's retained `models/forex_m15_holdout_2yr`); the four
  side dirs are deliberately kept for that decision and can be removed with
  `rm -rf models/forex_m15_stability_*` once it is made.

## Files touched

- `src/core/retrainer.py` — CI-bound verdict (`_holdout_pf_lower_bound`,
  `_holdout_verdict`, `HOLDOUT_PF_CONFIDENCE`), tail purge
  (`_tail_cutoff_by_symbol`, `_purge_boundary_tail`), fold-fail diagnostics
  (`_score_artifact_holdout`), restructured Phase 4.5, extended
  `HoldoutMetrics` + metadata, Glossary block.
- `tests/test_holdout_gate.py` — +15 tests, including the permanent leak
  guards.
- `scripts/diagnose_5yr_holdout.py` — scores via `_score_artifact_holdout`,
  prints the stable-gate verdict.
- `scripts/run_stability_batch.sh` — the reproducible 6-run demonstration.
- `GLOSSARY.md`, `src/core/README.md`, `scripts/README.md`, `tests/README.md`
  — documentation for the new terms and behaviour.
