---
type: refactor
date: 2026-09-21
time: 16:40 PDT
agent: opencode
model: deepseek-flash
trigger: "Implement the plan in llm_reports/handoffs/2026-09-21_feature-lab-plan.md (Brandon: 'there is a new set of instructions in llm_reports/handoffs')."
head: 14fe328
scope: "new src/lab package + one extraction in core/retrainer/_features.py; no models/, config/, or live path touched. Uncommitted in the working tree at time of writing — the plan's commit split is suggested at the end."
related:
  - handoffs/2026-09-21_feature-lab-plan.md
  - recons/2026-09-14_session-evidence-and-options.md
  - recons/2026-09-21_lab-v3-base-control.md
  - recons/2026-09-21_lab-spread-table-control.md
  - recons/2026-09-21_lab-microstructure.md
files_touched:
  - src/lab/ (new package: __init__, spec, registry, features, data, frames, gate, backtest, experiments, report, specs, cli, README)
  - src/lab/artifact.py (addendum: served-artifact replay)
  - src/lab/backtest.py (addendum: shared _replay_models + run_artifact_backtest)
  - src/lab/experiments.py, report.py, cli.py, __init__.py, README.md (addendum)
  - src/core/retrainer/_features.py (apply_labels_and_veto extracted)
  - src/core/retrainer/__init__.py (facade re-export + glossary)
  - src/README.md
  - GLOSSARY.md
  - tests/test_lab_spec.py, test_lab_registry.py, test_lab_frames.py, test_lab_gate.py, test_lab_backtest.py, test_lab_report.py (new, 38 tests)
  - tests/test_lab_artifact.py (addendum, 12 tests)
  - tests/README.md
  - llm_reports/recons/2026-09-21_lab-v3-base-control.md (+ edition: seed runs)
  - llm_reports/recons/2026-09-21_lab-spread-table-control.md, 2026-09-21_lab-microstructure.md
  - llm_reports/recons/2026-09-21_lab-served-artifact-baseline.md (addendum, new)
  - analysis_cache/lab_frames/ (gitignored frame cache, 107 MB after the 4 runs)
---

# The feature lab is built — and its control run reproduces the production frame exactly

## What was built

`src/lab/` now implements the plan's loop: **`FeatureSpec` → frame → retrainer
gate → live-gated backtest → recon report**. A candidate feature is one
`BaseFeatureGenerator` class registered under a name; nothing in
`FeaturePipeline`, `core/retrainer`, `src/execution`, or `run_oanda.py` changes
to score it. The lab is offline-only — no live module imports it.

```bash
PYTHONPATH=src:. python -m lab.cli run --name v3_base_control
PYTHONPATH=src:. python -m lab.cli run --spec specs/my_feature.py
```

Exit codes mirror the retrainer (0 passed / 2 rejected / 1 error). Frames are
cached on the spec's `content_hash` (spread-table bytes + veto env included), so
changing a lookback cannot produce a stale hit. Reports land in
`llm_reports/recons/<date>_lab-<slug>.md`; the three seed runs above are already
there and are the first three recons dated 2026-09-21.

## The one production-adjacent change

`engineer_features_and_labels` fused (a) a hardcoded generator list, (b) label
construction, and (c) the rice of vetoes. The lab needs (b)+(c) with (a)
swapped, so the second half was extracted:

```python
# before: one function, no seam
engineer_features_and_labels(df, sl_mult, tp_mult, max_hold, survival_bars,
                             htf_timeframe, angel_mult, risk_profile, alpha_table)
    -> (df, base_cols, chop_veto_rate)

# after: the wrapper keeps its exact behaviour; the shared half is callable
apply_labels_and_veto(df, feature_cols, *, sl_mult, tp_mult, max_hold,
                      survival_bars, angel_mult, risk_profile, alpha_table)
    -> (df, chop_veto_rate)
```

Order is preserved verbatim (Angel target → macro → survival → excursions →
chop/behavior veto → cleanup), because the bracket walk must see the contiguous
price path before any veto drops rows. `base_cols` is now computed in the
wrapper and passed in, which is what lets the lab clean on its own feature set.
Everything else composes existing code; no other `src/` file changed.

## Verification

- **Frame parity, exact.** `tests/test_lab_frames.py` asserts the lab's
  `feature_sets=("v3_base",)` frame equals `engineer_features_and_labels` +
  production's Phase-3a tail purge, row for row, with and without the spread
  table.
- **Calibration against the known measurement.** The control frame is
  **226,909 rows + 83 tail-purged = 226,992** — the exact engineered-frame count
  the 2026-09-14 work reports on the same cached basket — and its pooled base
  rate is **0.25399** against the known **0.2540**.
- **Full suite:** `576 passed, 17 subtests passed` (was 538/6; +38 lab tests),
  `python -m compileall -q src/` clean.
- **Live state untouched:** `soak.service` active, PID 362086, before and after
  every run.

## The three seed runs (all correctly FAIL the gate)

| run | rows | feats | chop veto | pooled trades | base rate | edge over random | backtest net EV |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v3_base_control` | 226,909 | 17 | 23.7% | 30 | 0.2540 | **+0.1793** | +0.255R (flat toll) |
| `spread_table_control` | 174,730 | 18 | 41.2% | 40 | 0.2495 | +0.1005 | +0.425R (measured toll) |
| `microstructure` | 226,909 | 21 | 23.7% | 101 | 0.2540 | +0.0628 | +0.165R (flat toll) |

Two readings that matter:

1. **The +0.179 edge on the control is the known small-sample artifact, not a
   new signal.** `edge_over_random` is in win-rate units, and the EV-maximising
   Angel bar confines the gate to 30 pooled trades — the same "+0.179 on 30
   trades" that the 2026-09-14 session explicitly withdrew. The lab reproduces
   the artifact it is supposed to reproduce; the reports now carry a Caveats
   section that says so whenever the trade count is thin.
2. **The spread-table experiment finally ran cleanly** (the 2026-07-07 run was
   confounded by a window shift): the table thins the basket hard (chop veto
   23.7% → 41.2%, frame −23%), the gate's evidence is no better
   (PF lb 0.773 → 0.582), and the backtest improves only because its model
   population differs — not a promotable result. The microstructure seed raises
   trade counts without raising the edge, and fails on EV too.

## Deviations from the plan (all deliberate)

1. **`apply_labels_and_veto` takes `feature_cols`.** The plan's sketch did not;
   the function cleans on the model-facing columns, and the lab's columns are
   not `BASE_FEATURE_COLS`.
2. **One `GeometrySpec`, not separate label and evaluation brackets.** Splitting
   them lets the gate score an EV computed under a bracket the labels were never
   built for; the failure is silent, so v1 refuses to expose the footgun.
3. **No artifact holdout in v1.** `_score_artifact_holdout` engineers its slice
   with the production feature list, so it cannot score a candidate set without
   a second extraction; the plan's own recommendation was to keep the target
   axis fixed first. Documented in `src/lab/README.md` and in every generated
   report.
4. **`run_frontier` deferred.** `scripts/angel_bar_frontier.py` already answers
   that question on the cached basket; porting it needs per-fold OOF
   probabilities that `validate_candidate` does not expose.

One correction to the plan: it specified `Signal.raw_sl_distance =
raw_atr * sl_mult`. The actual contract (`MLStrategy`, the rule-based library,
and `RiskManager.calculate_bracket`, which multiplies raw ATR itself) is raw
ATR in price units. The lab implements the real contract and
`tests/test_lab_backtest.py` pins it — a pre-multiplied stop would silently
double every bracket.

## Suggested commit split (nothing is committed)

1. `refactor(retrainer): extract apply_labels_and_veto from engineer_features_and_labels`
   — `_features.py` + `__init__.py` (+ the parity test could ride here).
2. `feat(lab): offline feature lab scored by the retrainer gate`
   — `src/lab/`, `tests/test_lab_*.py`, docs.
3. `docs: lab reports for the three seed runs` — the generated recons.

---

## Addendum (20:05) — served-artifact replay, and the baseline run

The user asked for the *current model* through the lab as a baseline. The gate
path retrains fresh fold models, so the lab could not answer that: a `replay`
verb was added.

- **New `src/lab/artifact.py`**: `load_served_artifact` reads `OANDA_MODEL_DIR`
  (default `models/forex_m15_wide`) — Angel/Devil pkls, `threshold.json` with
  live precedence (`MLStrategy`'s), `metadata.json` — and pins the fit-time
  feature order (`feature_names_in_`, CatBoost's `feature_names_`; a schema-less
  estimator is refused because numpy predict is positional).
  `replay_served_artifact` prepares the same content-hashed frame, scores the
  whole frame (`predict_probabilities`), and splits population/replay at the
  artifact's recorded holdout window.
- **`backtest.py`**: the per-symbol replay was factored into one `_replay_models`
  body; `run_model_backtest` (gate) and `run_artifact_backtest` (artifact) both
  call it, so the two paths cannot drift. `LabModelStrategy` takes optional
  `angel_feature_cols`/`devil_feature_cols` for the artifact's order.
- **`experiments.py`**: `prepare_frame` extracted from `run()` and shared.
- **`report.py`**: `render_artifact_report` / `write_artifact_report` (slug
  defaults to `served-artifact-<spec>`, so a replay cannot overwrite a gate
  report); the honesty box states the replay is not a gate and is mostly
  in-sample.
- **`cli.py`**: `python -m lab.cli replay --name v3_base_control
  [--model-dir DIR]` (exit 0 = completed replay).
- **Tests**: `tests/test_lab_artifact.py` (12) — threshold precedence, the
  CatBoost spelling, schema-less refusal, column-order scoring, gate-path
  parity, window split, report caveats. Suite **588 passed / 17 subtests**,
  compileall clean.

**Run**: `PYTHONPATH=src:. python -m lab.cli replay --name v3_base_control
--report-slug served-artifact-baseline` → 2.6 s on the cached frame; report at
`recons/2026-09-21_lab-served-artifact-baseline.md` with the Interpretation
filled. Headline: 63 proposals / 56 approvals in 730 days at the served bars
(0.028% of bars), **zero** in the 2,691 bars after the artifact's training data
ends; the recorded promotion holdout (36 trades / 69.4% / PF 4.55) is not
reproduced (12 approvals / 33.3% / PF 1.00 on the same window). The three seed
recons' `## Interpretation` sections are filled, and the KB topic
`build-a-bot-soak` carries the state.

`files_touched` for this addendum: `src/lab/artifact.py` (new),
`src/lab/backtest.py`, `src/lab/experiments.py`, `src/lab/report.py`,
`src/lab/cli.py`, `src/lab/__init__.py`, `src/lab/README.md`,
`tests/test_lab_artifact.py` (new), `tests/README.md`, `GLOSSARY.md`,
`llm_reports/recons/2026-09-21_lab-{v3-base-control,spread-table-control,microstructure}.md`,
`llm_reports/recons/2026-09-21_lab-served-artifact-baseline.md` (new).

