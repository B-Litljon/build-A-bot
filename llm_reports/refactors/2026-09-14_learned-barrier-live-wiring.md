---
type: refactor
date: 2026-09-14
time: 00:55 PDT
agent: DeepSeek Harness (dsh, web)
model: deepseek-v4-flash
trigger: "Phase 2 of the quantile-MAE barrier plan (llm_reports/handoffs/2026-09-14_quantile-mae-barriers-and-catboost-plan.md): wire BarrierEstimator predictions into ml_strategy.py and risk_manager.py so live orders can attach dynamic MAE/MFE distances"
head: f69595d39e1366e226c2b0cddb9aab609f77279
scope: modifies-source
files_touched:
  - src/ml/barriers/estimator.py
  - src/ml/barriers/README.md
  - src/ml/README.md
  - src/strategies/concrete_strategies/ml_strategy.py
  - src/strategies/base.py
  - src/strategies/README.md
  - src/strategies/concrete_strategies/README.md
  - src/execution/risk_manager.py
  - src/execution/oanda_forex_orchestrator.py
  - src/execution/README.md
  - src/analysis/strategy_backtester.py
  - src/analysis/README.md
  - GLOSSARY.md
  - tests/test_barriers.py
  - tests/test_ml_strategy_guards.py
  - tests/test_risk_manager.py
  - tests/README.md
related:
  - handoffs/2026-09-14_quantile-mae-barriers-and-catboost-plan.md
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
---

# Learned barrier geometry, wired into live execution — and the gate that still says no

## Context

The handoff assigns Harness B the sidecar schema: "Design the `RiskManager` /
`MLStrategy` sidecar for dynamic barrier loading during live inference." This is
that, implemented and tested — plus three measurements that changed what the
right design is.

The ask was for live orders to attach dynamic MAE/MFE distances instead of the
static `2.0×`/`4.0×` ATR bracket. What the tree actually contains today is a
fitted estimator (Phase 1) with **no producer** writing artifacts and **no
consumer** reading them, and a Phase 1 promotion gate that has not passed. So
the work split three ways: build the live path, make it impossible for it to
turn itself on, and measure whether turning it on would be a good idea. It would
not, yet.

## Investigation

Read in full: `src/ml/barriers/{estimator,labels}.py`,
`src/strategies/concrete_strategies/ml_strategy.py`,
`src/execution/risk_manager.py`, the bracket call site
(`oanda_forex_orchestrator.py:1379`), `run_oanda.py`'s model-dir wiring,
`scripts/evaluate_barriers.py`, `tests/test_ml_strategy_guards.py`, and the
glossaries in both modules.

Four facts decided the design.

**1. The Phase 1 gate fails.** `scripts/evaluate_barriers.py` on this machine
(CatBoost backend, 297k labelled rows over three expanding folds):

```
fold 1: pinball learned=0.4816 static=1.9407 BEAT  coverage=0.953 ok
fold 2: pinball learned=0.5065 static=2.0235 BEAT  coverage=0.934 ok
fold 3: pinball learned=0.8457 static=2.4576 BEAT  coverage=0.905 UNDER-COVERED
VERDICT: FAIL — static bracket stays (promotion blocked, prior weights stand)
```

Note what this does and does not say: the learned stop beats the static constant
on pinball loss on **every** fold, by 2×–4×, and the gate fails *only* on fold
3's coverage (0.905 against the 0.93 floor). The handoff's premise — LightGBM's
failure was fixed by CatBoost's monotone constraints — is confirmed; what blocks
promotion now is out-of-sample coverage drift in the most recent regime, not a
refused fit.

**2. `admissible` is False for every bar in the entire evaluation basket.** The
estimator flags an output tradeable when `rr >= rr_floor` (default 2.0), where
`rr = Q_MFE(0.50)/Q_MAE(0.95)`. Measured distribution of `rr` over the same
cached basket:

```
fold 1: n=74352 adm=0.0000 rr med=0.279 p90=0.283 max=0.285
fold 2: n=74352 adm=0.0000 rr med=0.296 p90=0.305 max=0.306
fold 3: n=74352 adm=0.0000 rr med=0.303 p90=0.305 max=0.305
```

A median favourable excursion divided by a 95th-percentile adverse excursion is
**structurally below 1** for a near-random walk, while the floor of 2.0 was
written for the static 4.0/2.0 *payoff*. The two numbers are not the same
quantity. Consequence for the wiring: had I made "inadmissible → stand down" or
even "inadmissible → static bracket" the live rule, the feature would have been
dead on arrival for 100% of bars, and it would have looked like a wiring bug.

**3. The learned stop is much wider than the constant.** Median `q_mae` across
folds is 9.5–10.1 ATR against the static 2.0×, i.e. ~5× the stop distance.

**4. The OANDA path does not size by risk.** `GLOSSARY.md:235` and
`src/execution/README.md` both record it and it is still true:
`run_oanda.py` drives this orchestrator with a fixed `units_per_trade = 1000`,
and `RiskManager.calculate_quantity` — where the $50 floor and 2%-of-equity
sizing live — is never called on the live forex path. On the Alpaca paths a 5×
wider stop means a 5× smaller position for the same risk. Here it means **5× the
loss per stop-out**, with nothing to compensate.

Facts 2 and 4 are why this ships default-off, with the width change logged, and
why the report says plainly: do not enable it.

## Findings / Changes

### The sidecar schema (Harness B's deliverable)

**Artifacts** — three files in the model directory, beside the Angel/Devil
pickles, written by `BarrierEstimator.save(model_dir, horizon)`
(`src/ml/barriers/estimator.py:255`) and read by `BarrierEstimator.load`
(`:323`):

| File | Contents |
|---|---|
| `barriers_mae.pkl` | the stop-side quantile model (joblib) |
| `barriers_mfe.pkl` | the target-side quantile model (joblib) |
| `barriers_meta.json` | `feature_cols`, `tau_mae`, `tau_mfe`, `rr_floor`, `horizon`, `backend`, `family`, `monotone_increasing`, `trained_at` |

**Write order is load-bearing.** Both pickles are replaced first, the meta last,
and `_reload_barriers_if_changed` (`ml_strategy.py:706`) triggers on the META's
mtime **alone**. A bar that sees a new meta therefore reads a complete matching
pair; a pickle replaced on its own is invisible rather than silently mixed —
the same hazard `_pair_mixed` already defends against for Angel/Devil.

**Payload** — `Signal.metadata[BARRIER_GEOMETRY_KEY]`, i.e.
`"barrier_geometry"`, defined once in `strategies/base.py:61`:

```python
{"source": "barrier",
 "sl_atr_mult": float, "tp_atr_mult": float,   # NATR multiples
 "rr": float, "admissible": bool,
 "tau_mae": float, "tau_mfe": float, "backend": str}
```

The multipliers are **NATR multiples** — the identical units of
`RiskProfile.sl_atr_multiplier` / `tp_atr_multiplier`. That is the whole
integration: the learned quantiles substitute for the two constants at the
multiplier step, so the gates, the rounding and the sizing stay on one code path
and cannot tell where a distance came from. Verified by test:
`payload["sl_atr_mult"] * atr_abs == BarrierOutput.raw_sl_distance`, to 9 places.

**Why the key lives in `strategies/base.py`** rather than in the strategy: three
layers need the same string and only one may own it. `execution/risk_manager.py`
is deliberately numpy-only (its README says so; importing `ml.barriers` would
drag polars and joblib in), and `analysis/strategy_backtester.py` keeps execution
imports out of module scope by design. `strategies.base` is the cheapest edge
that pulls nothing but polars. `risk_manager` takes the payload as an *argument*
and never needs the key at all.

**Switch and failure modes**

- `BARRIER_GEOMETRY_ENABLED` (or `MLStrategy(use_barriers=True)`), **default
  OFF**. Off, `MLStrategy` behaves exactly as it did before the sidecar existed —
  which is what makes it safe for the running soak to restart onto this tree.
- Enabled but artifacts missing/unparsable/horizon-mismatched/feature-vocab
  mismatched → **boot error**, never a silent fall back to static. Same
  philosophy as the `cost_ratio` and HMM guards: an operator who asked for
  learned stops and got static ones cannot tell.
- A *failed promotion* mid-run keeps the previously loaded estimator and alerts;
  it never disables and never half-swaps.
- A prediction failure on a bar (bad frame, raising model) falls back to static
  for that bar and logs. A telemetry-shaped bug in a sidecar must not sit on the
  order path.

### Per file

**`src/ml/barriers/estimator.py`** — added `save`/`load` + the three filename
constants; `horizon_` restored from the meta; `allow_writing_files: False` on the
CatBoost params (`:422`) so a fit stops writing a `catboost_info/` training log
into whatever directory the evaluator was launched from (it dirtied the tracked
tree during Phase 1 work — see Risk & follow-ups). Before: the no-lightgbm
fallback returned a closure, i.e. the degraded backend **could not be pickled**
and was therefore unservable live; it is now the module-level `_BinnedLadder`
(`:127`), still callable so every existing caller and test is unchanged.

**`src/strategies/concrete_strategies/ml_strategy.py`** — `use_barriers`
constructor arg resolved from the env switch; `_load_barriers` (`:603`) with
horizon and feature-vocabulary enforcement; `_barrier_geometry` (`:660`)
producing the payload from the live feature frame's tail (row-independent models,
so one row is enough); `_reload_barriers_if_changed` (`:706`) on the meta mtime;
the geometry attached at `:1183` and folded into the existing agreement log line
and `bar` event (`geometry=`, `sl_atr_mult=`, `tp_atr_mult=`, `barrier_rr=`,
`barrier_admissible=`, null when static so a consumer's schema is stable).

**`src/execution/risk_manager.py`** — `calculate_bracket(..., barrier=None)`
(`:366`); `_barrier_multipliers` (`:436`) validates and either substitutes or
declines with a critical log; `last_geometry_source` (`"static"` / `"barrier"`)
as per-call provenance; `BARRIER_WIDTH_WARN = 1.0` (`:205`) logs a WARNING when
the learned stop is ≥2× off the static one, naming the fixed-unit consequence.
Every gate still runs, and now runs on the *substituted* stop — so "does this
stop pay the spread" is asked of the distance actually being placed.

**`src/execution/oanda_forex_orchestrator.py:1406`** — passes
`(signal.metadata or {}).get(BARRIER_GEOMETRY_KEY)` through, and logs
`geometry=` on the bracket line. A pure pass-through: this layer does not decide
learned vs static.

**`src/analysis/strategy_backtester.py:277`** — the same hand-off, so an offline
replay of a barrier-enabled model measures the brackets the bot would actually
place. Inert for the library strategies (they attach no payload).

**Docs** — the three-layer system updated in the same change: module glossaries
in all four touched source modules, `src/ml/README.md` (the `barriers/`
subpackage was never listed), `src/ml/barriers/README.md` (backends, persistence
contract, the promotion verdict table), both strategies READMEs,
`src/execution/README.md`, `src/analysis/README.md`, and `GLOSSARY.md` (new
"MAE / MFE" and "learned barrier geometry" entries).

### Two deliberate non-changes

**No veto on `admissible`** — see Investigation finding 2. It travels as
telemetry. Making it a gate belongs behind a `rr_floor` recalibrated for the tau
pair, as a fourth gate.

**No clamp on how wide a learned stop may be.** A conditional quantile is
allowed to differ from a constant, and an arbitrary "0.25×–4× of static" band
would be a threshold nobody measured — precisely the mistake the repo's rails
warn about ("don't use textbook PSI thresholds"). Loud logging instead, and the
hazard is written into the report and the glossary.

## Verification

**Unit + integration tests.** 25 new tests; full suite green:

```
$ PYTHONPATH=src:. <venv>/bin/python -m pytest -q
472 passed, 6 warnings, 6 subtests passed in 22.37s
```

```
$ PYTHONPATH=src:. <venv>/bin/python -m compileall -q src/
COMPILEALL OK
```

New coverage: exact save/load round-trip equality and the meta-last write order;
refusal on an incomplete set, a meta without a horizon, empty `feature_cols`;
the binned fallback pickling; the switch defaulting off; boot refusal on all
three mismatch classes (missing artifacts, horizon, feature vocabulary); the
NATR-multiple ↔ price-distance identity; prediction-failure fallback; the
meta-mtime promotion swapping the estimator; a broken promotion keeping the
previous one; payload substitution not compounding with the profile; Gate A and
Gate B firing on the substituted stop; seven malformed payload shapes degrading
to static; provenance resetting per call.

**End-to-end on real bars** (scratch script, not committed): real GBP_JPY M15
from `analysis_cache/`, the live `FeaturePipeline`, the served
`models/forex_m15_wide` pair symlinked into a temp dir, and a CatBoost barrier
fitted on the live feature frame. Both switches, same slice:

```
[OFF] signal at end=1500: True | barrier key present: False
[ON ] signal at end=1500: True
  payload: sl_atr_mult=15.0755 tp_atr_mult=2.5907 rr=0.172 admissible=False backend=catboost
  static  bracket: (0.98652, 1.97304) source= static
  barrier bracket: (7.43611, 1.27791) source= barrier gate= none
  identical to static: False
  sl width ratio learned/static: 7.54x
```

with the width warning firing as designed:

```
[GBP_JPY] learned barrier stop 15.075x vs static 2.000x (7.54x wider) — fixed-unit
path does not size by risk, so this scales the loss per stop-out; rr=0.172 admissible=False
```

Two caveats, stated so nobody over-reads that number: the 7.54× is a *scratch*
single-symbol fit on 3.9k rows, not the Phase 1 artifact (whose basket median is
~5×); and both signals came from the real Angel/Devil at the same bar, so the
switch changes geometry only, not the decision.

**Phase 1 re-run after the CatBoost `allow_writing_files` change** reproduced the
verdict fold-for-fold (FAIL, fold 3 coverage 0.905) and left `git status
catboost_info/` empty.

**The live soak was not touched.** It is still the same process
(`run_oanda.py --daemon --env practice --granularity 15`, `soak.service active`),
`models/forex_m15_wide/` still holds only its five original files with Aug 29
mtimes, and no `barriers_*` artifact exists anywhere under `models/` — so with
the switch off the live path is bit-identical to before.

## Risk & follow-ups

1. **Do not enable this live yet.** The Phase 1 gate fails on fold 3 coverage,
   and enabling it on the OANDA path multiplies loss per stop-out by the width
   ratio (measured basket median ≈ 5×) because that path is fixed-unit. The
   ordering that makes sense: either get coverage over 0.93 (full feature
   vocabulary instead of the evaluator's two-feature subset is the obvious next
   try) or route the forex path through `calculate_quantity` first.
2. **No producer exists.** Nothing in `src/core/retrainer.py` writes barrier
   artifacts, so enabling the switch today fails at boot by design. Phase 3
   (retrainer integration) is the blocker, and it also has to decide what
   `metadata.json` says about brackets once the Devil's labels and the served
   geometry disagree — `_validate_metadata`'s bracket check assumes static
   multiples and was deliberately left alone.
3. **`rr_floor` needs recalibrating** for the `(Q_MFE(0.50), Q_MAE(0.95))` pair
   before `admissible` can mean anything live.
4. **Concurrent-agent commit.** While this work was in progress, commit
   `f69595d` ("feat: add ML-based trading strategy framework, risk manager, and
   OANDA forex orchestrator", 00:27:59) swept up these source and test changes
   (and CatBoost's `catboost_info/` log churn) from the other active harness —
   message does not describe the contents. The documentation edits in this
   report are **still uncommitted**. Not reworded or reverted: history is not
   rewritten unasked.
5. **`catboost_info/` should probably not be tracked at all.** It is generated
   training output; the estimator no longer writes it, but the committed copy
   remains and will churn again if any other code path fits CatBoost from the
   repo root.
6. **Handoff Phase 2 (H4 vs M15 timeframe)** is untouched by this report — the
   evidence that CatBoost edges out LightGBM on H4
   (`recons/2026-09-13_h1-h4-catboost-ab.md`) still argues for evaluating
   learned barriers at H4, where the spread toll is diluted. The wiring is
   timeframe-agnostic; only the artifact's declared horizon matters.

## Files touched

- `src/ml/barriers/estimator.py` — `97-99` (filenames), `127-152`
  (`_BinnedLadder`), `190` (`horizon_`), `255-321` (`save`), `323-390` (`load`),
  `422` (`allow_writing_files`).
- `src/strategies/concrete_strategies/ml_strategy.py` — `97-120` (glossary),
  `166` (env switch), `237`/`407-417` (`use_barriers`), `603-658`
  (`_load_barriers`), `660-704` (`_barrier_geometry`), `706-745`
  (`_reload_barriers_if_changed`), `900` (reload hook), `1124-1190` (payload +
  log + event).
- `src/strategies/base.py` — `61` (`BARRIER_GEOMETRY_KEY`), glossary `27-34`.
- `src/execution/risk_manager.py` — glossary `104-134`, `199-205` (geometry
  constants + warn threshold), `366` (signature), `405-425` (substitution),
  `436-533` (`_barrier_multipliers`).
- `src/execution/oanda_forex_orchestrator.py` — glossary `134-141`, `1406`
  (pass-through), `1420-1427` (bracket provenance log).
- `src/analysis/strategy_backtester.py` — `277-284`.
- `tests/test_risk_manager.py` (`TestBarrierGeometry`), `tests/test_barriers.py`
  (`TestPersistence`), `tests/test_ml_strategy_guards.py`
  (`TestBarrierSidecar`, plus the `TestHotReloadSeams._bare` fixture gaining the
  new constructor state).
- `tests/README.md` — per-file counts and a note on the substitution property.
- Docs: `src/ml/barriers/README.md`, `src/ml/README.md`, `src/strategies/README.md`,
  `src/strategies/concrete_strategies/README.md`, `src/execution/README.md`,
  `src/analysis/README.md`, `GLOSSARY.md` (bracket multipliers corrected to
  2.0×/4.0× while adding the new entries).
