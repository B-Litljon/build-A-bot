---
type: handoff
date: 2026-09-22
time: 00:30 PDT
agent: Claude (architect sub-agent)
model: kimi-k3
trigger: "Brandon: 'we already implemented the last plan since you and I had last spoke. scout the work, tell me what you think. we would need a whole new handoff.' Plus his standing question: can the lab screen feature interactions (the polypharmacy problem)."
head: 14fe328b0b918555b706895a33c8165462436e59
scope: read-only
related:
  - handoffs/2026-09-21_feature-lab-plan.md
  - refactors/2026-09-21_feature-lab-built.md
  - recons/2026-09-21_lab-v3-base-control.md
  - recons/2026-09-21_lab-spread-table-control.md
  - recons/2026-09-21_lab-microstructure.md
  - recons/2026-09-21_lab-served-artifact-baseline.md
---

# Feature lab v2 — assessment of what was built, and the ablation + architecture handoff

## Context

The v1 plan (`handoffs/2026-09-21_feature-lab-plan.md`) was implemented (opencode /
deepseek-flash, 2026-09-21 evening): `src/lab/` exists, uncommitted at time of
writing, and four recon reports are already filed. This handoff is the architect's
review of that work, plus the design for the two things the user asked about and
v1 does not yet do: **feature-interaction screening** (his polypharmacy analogy)
and the **architecture axis**. Written read-only; soak verified active and
untouched before and during (`soak.service`, PID 362086).

## Assessment — did the build get it right?

Verdict: **yes, unusually right for a first-pass research harness, and its numbers
are trustworthy.** The implementation even caught and fixed an error in the
original plan (below). An independent line-by-line audit (every file in `src/lab/`,
all seven `test_lab_*.py`, the two production diffs) found **zero BLOCKERs**,
two RISKs, and a set of NITs. The two things that determine whether a lab lies
are both correct *and* pinned by non-vacuous tests:

1. **Frame parity.** `tests/test_lab_frames.py:102-111` asserts
   `assert_frame_equal(lab_frame, production_frame, check_row_order=True,
   check_column_order=True)` with and without the spread table. Production frame
   is built with `engineer_features_and_labels` + the real `_tail_cutoff_by_symbol`
   / `_purge_boundary_tail`. And the calibration run against *real* data confirms
   it independently: the control frame is **226,909 rows + 83 tail-purged = the
   known 226,992**, pooled base rate **0.25399 vs the known 0.2540**. This is not
   a unit test asserting nothing — it is the production contract, verified.

2. **Row alignment.** `tests/test_lab_backtest.py:97-106` uses a
   `FeatureEchoModel` whose probability *is* the row's feature value; any
   off-by-one between bars and the post-clean frame would surface as a wrong
   score. It passes with exact values (0.9 then 0.5). The single most likely
   silent-corruption bug the original plan called out is genuinely pinned.

### The plan errors the build caught

- **F-P1 (the big one).** The original plan's §4 pseudocode specified
  `raw_sl_distance = raw_atr * sl_mult` — that would double the bracket, because
  `RiskManager.calculate_bracket` multiplies raw ATR by the multipliers itself.
  The build emits raw ATR (`test_emits_raw_atr_not_a_multiplied_stop`), which is
  correct. My plan was wrong; the build was right.
- **F-P2.** One `GeometrySpec` instead of separate label/eval brackets — a split
  would let the gate score an EV under a bracket the labels were never built for.
  Refusing that footgun is the right call.

### What the lab has already produced (four recons, all correctly FAIL)

| run | frame rows | feats | pooled trades | edge_over_random | verdict |
|---|---:|---:|---:|---:|---|
| `v3_base_control` | 226,909 | 17 | 30 | +0.1793 | FAIL (PF lb 0.7727 < 1.2) |
| `spread_table_control` | 174,730 | 18 | 40 | +0.1005 | FAIL (PF lb 0.5824) |
| `microstructure` | 226,909 | 21 | 101 | +0.0628 | FAIL (EV −0.078) |
| `served-artifact replay` | 226,909 | 17 | 42 (backtest) | n/a | replay, no verdict |

Three substantive results worth Brandon's attention:

- **The +0.179 control edge is the withdrawn small-sample artifact, deliberately
  reproduced.** `edge_over_random` is in win-rate units and the EV-max Angel bar
  confines the gate to 30 trades; the reports now auto-emit a Caveats section
  stating this. The lab reproduces the artifact it is supposed to reproduce —
  that is calibration, not signal.
- **The 2026-07-07 spread-table asymmetry is now cleanly closed.** With measured
  alphas on: chop veto 23.7% → 41.2%, frame −23%, and the gate's evidence *worsens*
  (PF lb 0.773 → 0.582). The cost side is measured; it does not create edge.
- **The served-artifact replay is the most damning number in the set.** At its
  own pinned bars (Angel 0.3833 / Devil 0.44) the served artifact proposes on
  **63 of 226,909 bars (0.028%)** and produces **zero proposals in the 2,691 bars
  after its training data ends (2026-08-29 → 09-07)**. The soak's 0 fills are the
  bar saying no, not an execution failure. And the recorded promotion holdout
  (36 trades / 69.4% / PF 4.55) is **not reproduced** by the very pkls it claims
  to describe (15 proposals / 12 approvals / 33.3% WR / PF 1.00 on the same
  window). Whether that gap is slice engineering or provenance is the lab's
  outstanding measurement — needs the metals bars cached for the full window.

## The two RISKs the audit found (both fixable without touching production)

### R1 — the frame cache hash does not cover generator-internal constants

`FeatureSpec.content_hash()` (`src/lab/spec.py:167-185`) hashes the spec fields,
the spread-table bytes, the risk-profile fingerprint, and 13 frame-affecting env
keys. It does **not** hash what is inside a registered generator. Concretely:
`_jsonable` reduces a generator instance to `_generator_id(gen) = module.qualname`
(`spec.py:188-196`), so:

- editing `LabMicrostructureFeatures._WINDOW = 20 → 21` changes no spec field and
  no hash → **a stale cached frame is silently reused**, while the README and the
  refactor report both claim "changing a single lookback is automatically a fresh
  frame";
- two instances of the same extra-generator class with different constructor args
  collide on the cache key.

This is the exact failure mode the plan's cache contract was designed to prevent,
and the claim has drifted from the mechanism. It will not fire while only the
three seeds exist, but it is a live hazard the moment iterative feature work
begins (someone edits a class attribute, re-runs, and reads yesterday's frame as
today's).

**Fix (v2 work item W1):** require every registered family to declare a
`version: int` (and `register_feature` to require it), and fold
`(name, version, resolved constructor state)` into the hash; for
`extra_generators`, hash `dataclasses.asdict(gen)`-or-`vars(gen)` alongside the
class id, and **raise** when a generator's state is not hashable/serializable
rather than silently hashing only its class. Prohibition per the architect
standards: no silent fallthrough to a "best-effort" hash.

### R2 — the backtest re-derives feature lists instead of using the gate's own

`run_model_backtest` (`src/lab/backtest.py:240-253`) discards
`gate.angel_features`/`gate.devil_features` and re-derives them from
`frame.feature_cols`. Correct today only **by construction**: `run_gate` passes
`feature_cols` straight through and `refit_models` returns `feature_cols` /
`feature_cols + ["angel_prob"]`. It silently breaks the moment HMM features are
enabled (`_gate.py:606` appends `HMM_OUTPUT_COLS` to the model-facing list) or a
future spec diverges `feature_columns()` from what the gate trains on. The
`GateResult` already carries the right lists.

**Fix (W2):** thread `gate.angel_features`/`gate.devil_features` through
`run_model_backtest` into `LabModelStrategy`, and add a test that constructs a
gate result whose feature lists contain a column absent from `frame.feature_cols`
(a synthetic HMM stand-in) and asserts the backtest uses the gate's list.

### NITs worth folding in while the files are open

- `stack_bars` has a dead two-identical-branches `if/else` (`frames.py:76-79`).
- `use_spread_table` defaults `False` where the plan said `True` — correct for the
  control, but the deviation should be a conscious default in the README (a new
  user gets the flat-toll arm without asking).
- The `np.random.seed(42)` the gate sets at `_gate.py:600` mutates *global* numpy
  state in the caller's process (inherited, not lab-introduced) — worth a one-line
  "the lab inherits the gate's global-RNG seeding" note in the lab README.

## v2 work items (the actual handoff)

### W3 — Ablation (`lab ablate`): the polypharmacy question

The user is right that single-feature-at-a-time answers the wrong question. The
mechanically correct answer is an ablation study: the "effect" of family *X in
the presence of the rest* is exactly `edge(full cocktail) − edge(cocktail − X)`,
and any other definition (a lone-feature run, a univariate correlation) measures
a population the deployed model never sees.

**Design.** New module `src/lab/ablate.py` + a CLI verb:

```bash
PYTHONPATH=src:. python -m lab.cli ablate --name <spec>   # full + one-minus-each-family
```

`ablate(spec)` expands to N+1 `FeatureSpec`s: the base spec unchanged, plus one
per registered family in `spec.feature_sets` with that family omitted. The
**labels/veto/geometry/data are identical across every variant**; only
`feature_sets` differs. Because all variants share the same underlying labeled
frame computation and differ only in the model-facing column list, the frames
cache efficiently (W1's richer hash makes this precise), and each variant costs
only its gate run (~8-11 s measured, from the recons).

**The interaction-versus-main-effect subtlety (resolved).** There are two kinds
of "does X matter":
1. *Main effect* — run `feature_sets=("x",)` alone.
2. *Effect in situ* — run full and full-minus-x (ablation).

The polypharmacy worry is (2), and it already subsumes (1): if X's main effect is
zero but removing it from the cocktail *hurts*, X was a silent load-bearing
interaction partner; if X's main effect is large but the ablation delta is zero,
X's information was already carried by something else in the cocktail. The
ablation delta is the honest number. A lone-feature table is a diagnostic, not
evidence, and the report should present it as such.

**Statistical honesty guardrail (mandatory).** On a 30-40 pooled-trade population
(the v3_base regime), an ablation delta of a few hundredths of a win-rate point is
indistinguishable from sampling noise, and the existing Caveats machinery already
says so for a single run. The ablation report must emit, per variant: pooled
trades, pooled wins, `edge_over_random`, both PF lower bounds, **and a
Clopper-Pearson-style interval on the delta itself** so "family X adds +0.03" is
presented as "+0.03 ± (wide interval), consistent with zero" rather than a point.
A family should only be *dropped* on ablation evidence when the *point* delta is
negative AND the trade count is not thin. This mirrors the gate's own
conservatism (it gates on the PF lower bound, not the estimate).

**SHAP (per-family, post-hoc) — the second half of the answer.** Ablation says
*whether* a family matters; per-family SHAP on a gate-passing model says *how*
credit is shared between correlated features (e.g. `natr_14` vs `bb_width_pct`).
This is the follow-on, not a v2 requirement: it needs a promoted (or near-pass)
model to be meaningful, and the seed gate verdicts are all FAIL. Note on
dependency hygiene: the served model is LightGBM but candidate artifacts include
CatBoost pickles (per the repo's `feature_names_in_`/`feature_names_` dual-read
discipline); if SHAP is added, prefer a model-agnostic explainability path
(permutation importance on the frame) over a LightGBM-specific `shap.TreeExplainer`,
so the architecture axis doesn't strand the diagnostic.

### W4 — The architecture axis (the "if features don't work" branch)

The gate already isolates the estimator behind `MODEL_FAMILY` and
`make_classifier` (`_common.py:410-415`), and the lab's `GateConfig.model_family`
already flows to it (`gate.py`). So **"run the identical spec under LightGBM and
CatBoost" is zero new lab code** — it is two runs differing in one field, and it
should be the first architecture experiment, not an afterthought:

```bash
PYTHONPATH=src:. python -m lab.cli run --name v3_base_control   # lightgbm (default)
MODEL_FAMILY=catboost python -m lab.cli run --name v3_base_control  # same frame, catboost
```

The shared content hash means both runs reuse the same frame (R1's fix must not
let `model_family` pollute the *frame* hash — it affects the model, not the
features; that field belongs in the gate-run key, not the frame key).

A genuinely novel architecture (a sequence/temporal model, a different label
estimator) is outside `make_classifier`'s sklearn-style `fit/predict_proba`
contract and is a **v3** question, deliberately. The honest sequencing the edge
budget work implies: (a) ablation to stop paying for dead families, (b) the
LightGBM↔CatBoost A/B to know whether the incumbent estimator is the bottleneck,
(c) only then a new estimator family. The lab should not grow a bespoke
architecture harness before (a) and (b) are measured.

### W5 — Commit discipline (nothing is committed yet)

The working tree carries the whole lab + the `_features.py` extraction as
uncommitted changes. The refactor report proposes a sensible 3-commit split;
endorsed, with two adjustments:

1. `refactor(retrainer): extract apply_labels_and_veto from engineer_features_and_labels`
   — `_features.py` + `__init__.py` + `tests/test_lab_frames.py` (the parity test
   belongs *here*, proving the extraction is behavior-preserving).
2. `feat(lab): offline feature lab scored by the retrainer gate` — the rest of
   `src/lab/`, remaining `tests/test_lab_*.py`, `src/README.md`, `GLOSSARY.md`,
   `tests/README.md`.
3. `docs: lab reports for the four seed/baseline runs` — the recons.

Do this **before** starting W1/W2, so the RISK fixes land as their own small
diffs on top, reviewable in isolation. Per repo rules, commit only when Brandon
asks; this handoff recommends the split but does not perform it.

## Verification

- Soak state checked at session start and end: `systemctl --user is-active
  soak.service` → `active`; PID 362086 uptime grew across the session (the lab is
  in the working tree only, the running process imported nothing new).
- The audit's two highest-stakes claims were re-read in primary source, not
  trusted: parity assertions at `tests/test_lab_frames.py:102-111` and the
  echo-model alignment test at `tests/test_lab_backtest.py:97-106` (both quoted
  above); the content-hash generator-collision at `src/lab/spec.py:188-196`
  (verbatim: `_generator_id` returns only `module.qualname`).
- All four recons read in full; numbers in the table above are taken from their
  Gate/Frame sections directly.
- No source was written in this session; the only file created is this handoff.

## Risk & follow-ups

- **R1 is the one to fix before any iterative feature work.** The moment someone
  edits a lookback constant inside a registered generator, the cache lies. W1 is
  small, test-covered, and production-free.
- **The recorded-holdout provenance question is the lab's biggest open
  measurement.** The served artifact's claimed promotion holdout (PF 4.55 on 36
  trades) is not reproduced by the artifact itself on the same window (PF 1.00).
  Settling it needs the metals bars (XAU/XAG) cached for the full 730-day window
  so `_score_artifact_holdout`'s exact slice can be replayed. That is a data fetch,
  not code; it is queued, not done.
- **Do not drop families on the seed ablation evidence.** v3_base's four natural
  families (base/htf/session/cost) have not yet been ablated, and even the v2
  ablation runs will sit on 30-40 pooled trades — enough to see a large effect,
  not a small one.
- The Devil being inert at its pinned bar (56/63 raw approvals pass) is unchanged
  by all of this and remains the open 2026-09-14 proposal (train the Devil on the
  macro label); it is a retrainer change, not a lab change.

## Files touched

- Created: `llm_reports/handoffs/2026-09-22_feature-lab-v2.md` (this file).
- Read (for the next agent): the scout's full audit of `src/lab/`;
  `src/lab/spec.py:160-204`, `tests/test_lab_frames.py:88-117`,
  `tests/test_lab_backtest.py:80-109`; the four recons of 2026-09-21;
  `refactors/2026-09-21_feature-lab-built.md`.
