---
type: handoff
date: 2026-09-21
time: 14:35 PDT
agent: Claude (architect sub-agent)
model: kimi-k3
trigger: User request — "make a feature lab, a package we can use to quickly try out various features and backtest them on recently old data. Build out the plan and I'll look through it when I return."
head: 14fe328b0b918555b706895a33c8165462436e59
scope: read-only
related:
  - recons/2026-09-14_session-evidence-and-options.md
  - recons/2026-09-20_reprice-wide-geometry-band-analysis.md
  - refactors/2026-09-16_code-downsizing-lane-deletions-and-retrainer-split.md
---

# Feature Lab — architectural plan (no source yet)

## Context

The measured reality (from [[build-a-bot-edge-budget]], verified through
2026-09-20): the M15 fiat model's real selectivity is **≈ +0.045R**, every cost
side and market axis has been measured and closed, and **the only untested lever
is a genuinely different feature/target design**. The user's ask is the tool for
that axis: a lab to iterate on candidate features fast and backtest them on
recently-old data, and — if features don't move the number — a clean
architecture-swap harness.

This file is the plan and the contract. **No source was written** in this
session — the OANDA soak is live under `soak.service` (verified below), and the
point of doing it as a handoff is that Brandon reads and signs off on the shape
before any `src/` change lands near a running bot.

## Live-state verification (performed before any planning)

```
$ systemctl --user show soak.service -p ActiveState,SubState
ActiveState=active / SubState=running
$ ps -o pid,lstart,cmd -p $(pgrep -f 'run_oanda.py --daemon')
PID 362086  STARTED Sun Sep 20 14:05:00 2026
  run_oanda.py --daemon --env practice --granularity 15
```

Soak is up (post-weekend cold start, as designed). Decision: this session
stays read-only, delivers the design only. When it is built, the lab is an
**offline package** — nothing in it imports from or is imported by
`src/execution/` or the boot path of `run_oanda.py`, so even a broken lab
cannot wedge the live process.

## Investigation

Two scout passes (feature pipeline + backtest machinery) were mapped, then every
load-bearing claim was re-read in the actual files before designing on it:

| Claim relied on | Verified at |
|---|---|
| Retrainer package facade re-exports the whole gate pipeline | `src/core/retrainer/__init__.py:384-444` (`FoldMetrics`, `ValidationReport`, `HoldoutMetrics` from `_types`; `engineer_features_and_labels` from `_features`; `validate_candidate`, `_macro_base_rate` from `_gate`; `refit_models` from `_train`) |
| The canonical bracket-walker, SL-first conservative, entry at close | `src/core/retrainer/_labels.py:27-89` `_compute_devil_targets_atr` |
| Feature+label assembly entry point and its signature | `src/core/retrainer/_features.py:37-47` `engineer_features_and_labels` |
| The offline backtester signature, per-instrument toll, live-gate hook | `src/analysis/strategy_backtester.py:145-158` `run_backtest(...)` |
| Offline basket loader + parquet caching | `src/analysis/build_strategy_matrix.py:237-304` `load_basket` |
| Forex bracket/risk profile: 2.0×/4.0×, `spread_k_base=3.0`, `spread_atr_alpha=0.15` | `src/execution/risk_manager.py:376-397` `RiskProfile.for_asset_class("forex")` |
| `alpha_overrides` hook signature | `src/execution/risk_manager.py:418-436` `RiskManager.__init__` |
| The single promotion seam used everywhere else | `soak.service` declares the served model dir (single source of truth — per CLAUDE.md, and re-asserted here as the lab's boundary) |

Local, offline, no-API history that already exists and is usable as the lab's
default dataset (verified with `ls` + the loaders):

- `analysis_cache/strategy_matrix/*_M15.parquet` — **6 fiat pairs, ~730 days**
  (2024-09-08 → 2026-09-08), the exact cache `evaluate_barriers.py` runs on.
  This is "recently old data" and is the default. 18 files, M15/M60/M240.
- `logs/graded_decisions.parquet` — **19,367 rows, 2026-07-31 → 2026-09-18**,
  8 symbols. The live decision ledger; a *replay* substrate, not a training one.
- `data/cache/ab_catboost/*_M15_60d_20260908.parquet` — includes XAU/XAG for the
  metals-control arm (60-day M15 only).
- `config/spread_alphas_m15.json` — the per-instrument cost table the
  backtester/label veto need (GBP_NZD 0.8929 … AUD_JPY 0.3372).

## The plan

### 0. Design decisions up front (the answers I chose — challenge these)

1. **New top-level package `src/lab/`** — a deliberate peer of `src/analysis/` and
   `src/ml/`, not a subfolder of either. Rationale: it *orchestrates* both plus
   `src/core/retrainer`, and per SRP a package that composes three existing ones
   should not live inside one of them. It is **offline-only**: nothing in
   `src/execution`/`run_oanda` imports it.
2. **Candidate features are `BaseFeatureGenerator`s, no exceptions.** The seam
   already exists (`src/ml/core/interfaces.py:26-31`) and both sides already
   consume a list of them. Writing a new feature = writing one class.
   `FeaturePipeline.run()` (`feature_pipeline.py:70-77`) is reused unchanged.
3. **Reuse labels + gate, never reimplement.** The verdict for "is this feature
   set better" is the retrainer's own gate (`validate_candidate`), which now
   reports `edge_over_random` (added 2026-09-14, `_gate.py:1160-1187`), so a lab
   run and a production retrain answer the same question with the same ruler.
   That is the whole point of the lab: **comparability with the served model's
   training numbers**, which no ad-hoc backtest metric would give.
4. **Two metrics, always, on every experiment:**
   - `edge_over_random` (pooled) — **primary decision metric**, the one number
     the edge-budget work says matters.
   - `net_ev_r` from `strategy_backtester.run_backtest` with
     `spread_alphas=config/spread_alphas_m15.json` — the honest cost-side check.
   A feature set only "wins" if it moves `edge_over_random` without being an
   artifact of a cost side the gate does not see.
5. **The cache contract for speed.** Feature frames are expensive
   (`engineer_features_and_labels` over ~50k bars × 6 symbols). The lab caches
   them as parquet keyed by a **content hash of the spec**, not by name, so a
   rerun of an unchanged spec is a reload, not a recompute.

### 1. Package layout

```
src/lab/
├── __init__.py            # facade, re-exports the public API only
├── README.md              # layer-2 doc: per-file what/imports/reads-writes
├── spec.py                # FeatureSpec, GateConfig: frozen dataclasses
├── registry.py            # @register_feature decorator + get_generators(spec)
├── data.py                # bars cache: load_basket wrapper + parquet cache
├── frames.py              # build_frame(spec, bars) -> (features_df, feature_cols, chop_veto_rate)
├── gate.py                # run_gate(features_df, ...) -> ValidationReport; run_frontier(...)
├── backtest.py            # run_model_backtest(...) — model-aware, gates live
├── experiments.py         # ExperimentRunner: spec -> {frame, gate report, backtest report}; content-hash cache of frames
├── cli.py                 # python -m src.lab ...
└── report.py              # emit llm_reports/recons/<date>_<spec-slug>.md skeleton
```

### 2. `spec.py` — the thing a user writes

```python
@dataclass(frozen=True)
class FeatureSpec:
    name: str                                   # slug used for cache + report
    symbols: tuple[str, ...] = DEFAULT_TRADEABLE_6  # no metals by default
    granularity: int = 15
    days_back: int = 730
    feature_sets: tuple[str, ...] = ("v3_base",)    # names into registry
    label: LabelSpec = LabelSpec()                   # survival/macro/horizon/mults
    geometry: GeometrySpec = GeometrySpec(2.0, 4.0, 45)  # the SERVED one by default
    use_spread_table: bool = True                    # bake cost_ratio from config
    extra_generators: tuple[BaseFeatureGenerator, ...] = ()  # explicit escape hatch
    def content_hash(self) -> str: ...   # sha of the *resolved* spec (incl. versions)
```

Everything that must be true about a run — what data, what features, what
labels, what bracket — is one frozen, hashable object. The hash is the cache
key, so "change one lookback constant" is automatically a fresh frame, never a
stale hit.

### 3. `frames.py` — the heart

```python
def build_frame(spec: FeatureSpec, bars: dict[str, pl.DataFrame]) -> FrameResult:
    stacked = prepare(bars)                  # concat + sort ["symbol","timestamp"] + tag symbol
    pipeline = FeaturePipeline(feature_generators=get_generators(spec))
    df = pipeline.run(stacked)               # candidate + base features, then clean
    df = engineer_labels_only(df, spec)      # NEW: labels/veto WITHOUT re-running features
    return FrameResult(df, spec.feature_cols(), chop_veto_rate)
```

**The one real refactor this requires.** Today `engineer_features_and_labels`
(`_features.py:106-280`) fuses (a) a hardcoded generator list, (b) label
construction, (c) chop veto. The lab needs (b)+(c) with (a) swapped. Options,
in preference order:

- **Option 1 (preferred): extract `apply_labels_and_veto(df, *, sl_mult, tp_mult,
  max_hold, survival_bars, angel_mult, risk_profile, alpha_table)`** from
  `engineer_features_and_labels` so both the retrainer (Option 1 keeps its exact
  behaviour as a thin wrapper) and the lab call the same labeling path. Small,
  test-covered (existing tests pin the frame), preserves the 17-feature contract
  when `feature_sets=("v3_base",)`.
- Option 2: the lab re-builds the current pipeline list from a registry alias
  `("v3_base",)` and calls `engineer_features_and_labels` on a frame whose
  candidate columns are **already present** — requires `_features.py` to tolerate
  pre-added columns. More fragile; only if Option 1's refactor is judged too
  invasive under a live soak.

**Ordering constraint inherited from the repo:** generators run per symbol so
one instrument's history never feeds another's indicators (the `group_by` in
`V3BaseFeatures`), and label/veto runs **after** features because
`_compute_devil_targets_atr` needs `natr_14`. The lab's `prepare()` enforces
per-symbol isolation by construction (loop per symbol, concat at the end), the
same shape as `_features.py`.

### 4. `backtest.py` — honest, model-aware, gated

`BacktestModelStrategy(BaseStrategy)` — a thin `BaseStrategy` whose
`generate_signals` runs the **trained candidate models** (LightGBM by default)
over the already-computed frame and returns a `Signal` per bar with
`raw_sl_distance = raw_atr * sl_mult` and `metadata` carrying the probabilities.

It is run through `analysis.strategy_backtester.run_backtest(..., risk_manager=
RiskManager.for_asset_class("forex"), spread_alphas=<config table>)` — so the
backtest includes the **live Gate A/B/C vetoes** and the **measured per-instrument
toll**, not an idealised copy. This is the piece earlier ad-hoc scripts
(`reprice_band_geometry.py`'s local `walk()`, the 60-geometry sweep) lacked.

**A deliberate nuance:** `run_backtest` currently walks a *raw bar frame* and
lets the risk manager own the bracket. A model-aware lab strategy needs the
feature row. Design: the lab strategy holds a reference to the *aligned*
features DataFrame (same row order as bars after cleaning; `FrameResult` carries
both) and `generate_signals` reads row `i`. Documented failure mode to pin in
tests: if cleaning dropped rows, the strategy must consume the post-clean frame
or the indices skew — this is the single most likely silent-corruption bug, so
it gets a dedicated test.

### 5. Evaluation flow (the actual "try a feature" loop)

```
python -m src.lab run --spec specs/my_feature.py
    -> frames.build_frame            (cached on content-hash)
    -> gate.run_gate                 -> ValidationReport (Brier/EV/PF/edge_over_random per fold + holdout)
    -> gate.run_frontier             (optional: angel-bar sweep, same trick as angel_bar_frontier)
    -> backtest.run_model_backtest   -> BacktestReport(net_ev_r, PF, drawdown, gate_rejections)
    -> report.skeleton               -> llm_reports/recons/2026-MM-DD_<spec>.md
```

The gate is the same function the production retrainer calls — including the
holdout discipline and `_score_artifact_holdout` at the frozen production
threshold. That means a lab "PASS" means "would have promoted," no translation
needed.

### 6. Architecture axis (the user's "if that doesn't work")

The gate already isolates the estimator via `MODEL_FAMILY` ("lightgbm" default,
"catboost" wired, `make_classifier` at `_common`). The lab exposes this as
`GateConfig.model_family`, so the *same* spec can be scored under both
estimators with zero other change. A genuinely different architecture (e.g. a
sequence model) is out of scope for the lab's first version — but because the
lab's verdict is produced by `validate_candidate`, swapping the model means
writing a new `make_classifier`-compatible trainer, not a new lab.

### 7. Tests (`tests/test_lab_*.py`)

The repo's conventions are already set (`tests/README.md`): deterministic
seeds, no network, orchestrators built via `__new__`, and **failure-path-first**.
The lab's tests, in that spirit:

1. `test_spec.py` — frozen spec; same spec ⇒ same `content_hash`; one constant
   change ⇒ different hash.
2. `test_registry.py` — duplicate-name raise; generator round-trip; a spec
   referencing a missing feature fails loudly (prohibitions: no silent drops).
3. `test_frames.py` — **`engineer_features_and_labels` vs lab-built frame for
   spec `feature_sets=("v3_base",)` produce identical 17-feature frames** (the
   train/serve-parity contract, pinned); row-alignment between bars and
   features after cleaning; tail purge matches `_tail_cutoff_by_symbol`.
4. `test_gate.py` — deterministic ValidationReport on a synthetic frame; a
   zero-skill feature set yields `edge_over_random ≈ 0` (regression test for the
   telemetry doing its job).
5. `test_backtest.py` — model-aware strategy emits signals with correct
   `raw_sl_distance`; **the post-clean row-alignment test**; spread table
   applied per symbol.

Run per CLAUDE.md: `PYTHONPATH=src:. <venv python> -m pytest -q tests/test_lab_*.py`
and `python -m compileall -q src/` before any commit.

### 8. Deliverable sequencing (safe under a live soak)

1. **This plan** (read-only, this file). ← you are here
2. `spec.py + registry.py + tests` — pure offline, zero production imports.
3. `frames.py` — needs the Option-1 refactor of `_features.py`; do it with the
   soak running is fine (the retrainer is training-only, not in the boot path),
   but as its own commit with `test_frames.py` proving parity.
4. `gate.py + backtest.py + cli.py` — the loop.
5. First three seed specs to prove the lab answers the question (below).

### 9. First experiments the lab should run (already informed by the closed axes)

Because the toll budget is known, seeds should probe *signal*, not cost:

1. **`v3_base` control** — the served 17 features, default geometry. Must
   reproduce `edge_over_random ≈ +0.045` and the existing report numbers. This
   is the lab's calibration run; if it doesn't reproduce, the lab is wrong, not
   the model.
2. **`spread_table_control`** — same features but `use_spread_table=True` (the
   one configured-but-unused asymmetry; its 2026-07-07 experiment was confounded
   and never re-run — this closes it cleanly).
3. **One genuinely new feature family** — e.g. a `V3MicrostructureFeatures`
   generator (bar-shape/return-autocorrelation/range-position over short
   windows), as the template for how a new family gets registered and scored.

## Findings / Decisions (summary)

- **F1.** The feature lab belongs in a new `src/lab/` package, offline-only.
- **F2.** Candidate features are `BaseFeatureGenerator`s registered by name —
  zero edits to the pipeline or the training/live symmetry to add one.
- **F3.** The verdict metric is the retrainer's own `validate_candidate` gate
  (`edge_over_random` primary, Brier/EV/PF/holdout the guardrails) plus a
  model-aware `run_backtest` with live gates and per-instrument cost — never a
  bespoke metric that would not be comparable to the served model's numbers.
- **F4.** One small refactor is required to make labeling reusable
  (`apply_labels_and_veto` extracted from `engineer_features_and_labels`);
  everything else composes existing code.
- **F5.** Architecture swap is a first-class knob (`model_family`), not a future
  rewrite — the gate already isolates it.

## Verification

Every citation above was read in the file this session (lines quoted in the
Investigation table). No source was written; `git status` remains clean except
this report. The soak's state was verified before planning and is untouched:
`soak.service` active, PID 362086.

## Risk & follow-ups

- **Live-bot proximity.** When `frames.py` is implemented, the `_features.py`
  refactor is the only production-adjacent change. Keep it in its own commit,
  run the full suite, and confirm `soak.service` is unaffected (it is
  training-side, not boot-side, but CLAUDE.md's rule stands: anything that could
  crash `src/execution/` is a money-losing bug — this refactor cannot reach it
  by construction, which the facade's import graph should pin in review).
- **Frame memory.** 730 days × 6 symbols × ~30+ candidate features in polars is
  fine; if a user adds many families, `build_frame` should support per-family
  incremental caching (later iteration, not v1).
- **The lab's verdict is only as honest as the data.** The default cache ends
  2026-09-08; a "refresh data" path exists (`load_basket` falls back to OANDA
  and caches), so newer "recently old" windows are one flag away.
- **Open question for Brandon:** whether the lab should also own the *target*
  axis (specifying new label definitions beyond survival/macro), or whether
  that stays a retrainer change. Recommendation: keep targets fixed in v1 —
  the label vocabulary is the riskiest place to introduce silent leakage, and
  the edge-budget work says features are the more promising lever.

## Files touched

- Created: `llm_reports/handoffs/2026-09-21_feature-lab-plan.md` (this file).
- Read during investigation (for the next agent): everything in the
  Investigation table above, plus `src/ml/feature_pipeline.py`,
  `src/ml/core/interfaces.py`, `src/strategies/base.py`, and file lists of
  `src/ml/`, `src/analysis/`, `analysis_cache/strategy_matrix/`.
