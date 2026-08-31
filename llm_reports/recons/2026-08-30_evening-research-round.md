---
type: recon
date: 2026-08-30
agent: K3 (kimi-k3, via dsh)
trigger: "User: knock out all four queued items while the soak accumulates evidence — tests, behavior research, orchestrator cleanup, sensitivity sweep"
related:
  - recons/2026-08-29_new-gate-old-vs-trim-matrix.md
  - recons/2026-08-29_trim-100x15-promotion.md
  - recons/2026-08-23_behavior-matrix-and-the-trend-high-hole.md
---

# Evening research round (2026-08-30): four items while the soak bakes

State at start: trim 100x15 serving the soak since 14:05 PT 2026-08-30
(models/forex_m15_wide), branch feat/trim-100x15-retrain through 94a8ed7.

## A — The gate rebuild is now pinned by permanent tests

`tests/test_dynamic_thresholds.py` (14 tests; suite 343 → 357, all passing):
the Angel-bar sweep across compressed / constant / tiny-frame / empty
distributions plus determinism; `_devil_min_child` (extracted from
`refit_models` into a pure helper for testability — same maths, no behavior
change); CP lower-bound reference values, including the audit's canonical
shape (3/3 perfect must NOT pass the 1.2 bar, 4/4 must); and a
recorder-driven `validate_candidate` run proving the calibrated threshold and
both CP bounds land on the ValidationReport.

Found while writing it, worth knowing: `tests/` is a package containing a
`tests/execution/` directory. Putting `tests/` itself on `sys.path` (the
obvious way to cross-import a sibling test module) **shadows the real
`execution` package** and kills core imports mid-collection. Sibling imports
must go through the package: `from tests.test_holdout_gate import ...`.
Documented in the new test file's header.

## B — Behavior matrix: `trend_high` still loses, but the edge has a home

New driver `scripts/capture_behavior_ledger.py` runs the retrainer's real
walk-forward with the opt-in OOS ledger on, tags bars with the causal
behavior labels, and scores via `analysis/behavior_matrix.py`. Raw outputs:
`logs/behavior_ledger_{shipped200x63,trim100x15,mid150x31}.parquet` and
`logs/behavior_matrix_*.csv` (2yr window through 2026-08-28).

The `trend_high` cell (the only INFORMATIVE cell for both trim configs):

| config | n | WR | net EV | PF net | 95% CI |
|---|---|---|---|---|---|
| shipped 200x63 | 16 | 0.50 | +0.17R | 1.26 | [−0.58, +0.92] (thin) |
| **trim 100x15 (live)** | 31 | 0.29 | −0.46R | 0.51 | [−0.94, +0.02] |
| mid 150x31 | 37 | 0.32 | −0.36R | 0.60 | [−0.84, +0.13] |

Verdict: the 2026-08-23 recon's directional finding **survives the rebuild** —
runaway-trend bars are still where 2x-stops die before 4x-targets land. The
trim models bleed less (PF 0.51/0.60 vs the old 0.30–0.44) but it's still the
worst cell in each table. The veto rationale is intact; its urgency fell with
the margin, and enabling it still requires the live-side gate that does not
exist. The mirror finding is more constructive: the trim model's positive edge
concentrates in `mixed_high` (+0.77R net, PF 2.93, the only cell anywhere
whose bootstrap CI excludes zero) and `range_high` — high volatility that is
NOT trending. Coherent with the feature set (range_coil_10, Bollinger).
Ledger populations include the untradeable metals (matches the 2026-08-23
recon's population); thin cells are flagged, not hidden, and they dominate
these tables — treat every non-`trend_high` row as suggestive.

## C — LiveOrchestrator feature-schema landmine removed

`src/execution/live_orchestrator.py` (the legacy/quarantined Alpaca path, not
the soak) hardcoded the pre-2026-08-29 18-feature list — including the five
dropped dead features, minus the session flags. Fed to a 17-feature model,
its `select()` crashes every bar. It now reads the served model's own
`feature_names_in_` through the strategy (hot-reload-refreshed), with the
calibrated-Angel-bar caveat written in for any future revival. The three test
mocks that silently depended on the hardcoded list now expose a
frame-derived `feature_names`. 12/12 orchestrator tests pass.

## D — Proposal-rate knob is a smooth frontier, not a cliff

`RETRAIN_MIN_ANGEL_PROPOSALS` swept at 150 / 300 (default) / 600 through the
real gate, 100x15, 2yr unpinned (all three PASSED both gates):

| min proposals | final Angel bar | pooled PF lb | pooled trades | holdout PF [lb] trades |
|---|---|---|---|---|
| 150 | 0.4008 | 1.58 (48/90) | 90 | 6.57 [3.08] 30 |
| **300** | 0.3833 | **1.62 (69/132)** | 132 | 4.55 [2.40] 36 |
| 600 | 0.3587 | 1.29 (83/183) | 183 | 3.30 [2.00] 53 |

Loosening the bar roughly doubles trade volume while per-trade evidence thins
— and the default sits at the evidence peak on both measures. The promoted
model's pass is robust to the knob; this was a measurement, not a retune.
Side dirs `models/sweep_mp{150,600}/` retained.

## Net state

Nothing about this round changes the served model or its evidence; the soak
remains the confirmation experiment. The one live-relevant number from this
round for expectation-setting: the trim model's proposals run ~half metals,
which this account does not trade — so live frequency lands around one trade
every few days, and its quality should track the tradeable-only gate evidence
(holdout PF lb 2.40).
