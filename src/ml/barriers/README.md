# `src/ml/barriers`

Quantile-regression barrier geometry: instance-specific stop/target distances
predicted per bar, replacing the static ATR-multiple bracket.

## Files

### `labels.py`

`compute_excursions(df, horizon)` appends MAE/MFE labels (NATR-scaled, clipped
at 0) to an OHLC frame, with a `resolvable` flag; the last `horizon` bars are
null-labeled and must be dropped before fitting, and so is any row whose
forward window contains even one null high/low (a gap bar would otherwise
produce a finite, understated excursion marked resolvable). Short-direction
labels are the long labels mirrored (short MAE ≡ long MFE), so no separate
labeling is needed. `pinball_loss` is the quantile objective. Imported by the
retrainer research path, `estimator.py`, and the live strategy's horizon check.

### `estimator.py`

`BarrierEstimator` — two quantile regressions (MAE τ=0.95 stop-side, MFE τ=0.50
target-side) on the same feature vocabulary Angel/Devil see. `BarrierOutput`
carries price-unit distances for the Signal contract plus the NATR-space
quantiles and implied reward:risk for telemetry and the pre-trade veto. A
predict-time failure mode is load-bearing: it is `admissible=False` for
essentially every bar at the shipped `rr_floor=2.0` (measured rr 0.28–0.30 on
all three evaluation folds, 2026-09-14), because that ratio scores
`Q_MFE(0.50)/Q_MAE(0.95)` while the floor was written for the static 4.0/2.0
payoff — so the flag is served as telemetry and is deliberately **not** a live
veto.

Three backends, selected by `family` (`BARRIER_FAMILY`, default `catboost`):

- **CatBoost** (`loss_function="Quantile:alpha=τ"` with
  `monotone_constraints` on `natr_14`/`vol_rel`). Monotone by construction,
  which is why it is the default; the fit-time audit still runs as a
  redundant check.
- **LightGBM** (`objective="quantile"`). Its quantile objective **rejects**
  `monotone_constraints` outright, so monotonicity is enforced by audit
  instead: a fitted model whose stop response falls more than `_AUDIT_TOL_ATR`
  below its running peak is refused, and the fit raises rather than returning
  an inverted geometry.
- **Binned fallback** (no libraries installed): empirical tau-quantiles per
  `natr_14` quartile bin, isotonised at fit so it is monotone by construction.
  Wrapped in `_BinnedLadder` — a module-level handle rather than the closure
  the first version used, because a locally-defined function cannot be
  pickled, which made the degraded backend unservable live.

`used_catboost_` / `used_lightgbm_` / `backend_` record which one a fit landed
on, and `load()` restores them.

**Persistence is the live sidecar contract.** `save(model_dir, horizon,
verdict=None)` writes three files — `barriers_mae.pkl`, `barriers_mfe.pkl`,
`barriers_meta.json` — each atomically, **pickles first and the meta last**.
`load(model_dir)` refuses an incomplete set, a meta without `feature_cols`, or a
meta without a `horizon`; it returns an estimator carrying `horizon_` (which the
caller checks against the execution lifetime) and `verdict_`.

**The verdict is the artifact's own reason to be served.** `verdict` takes the
output of `scripts/evaluate_barriers.py` (`BARRIER_VERDICT_OUT=<path>`, or the
`VERDICT_JSON` line it always prints) and records it beside the weights, because
a retrain's barrier hook fires off the Angel/Devil gate rather than off the
barrier gate — so artifacts can exist whose Phase 1 gate **failed**, and nothing
in the artifact would say so. `_validate_verdict` refuses to *write* a verdict
without a boolean `passed` (an unreadable verdict-shaped blob looks evidenced,
which is worse than none), and passes every other field through verbatim.

**`q_mae_scale` is the calibrated stop.** The raw `Q_MAE(0.95)` under-covers the
most recent regime (see the table below), so `save(..., q_mae_scale=s)` records an
inflation factor for the STOP side only, applied in `predict()` before `rr` and
`admissible` are computed and refused when non-positive. A positive scalar cannot
reorder the response, so it is compatible with the monotonicity guarantee by
construction. A per-decile calibration (cumulative-maxed to stay non-decreasing)
took the gate from `0.951/0.937/0.912` to **`0.954/0.974/0.942` — all three folds
clearing both criteria** — at the cost of 0.03–0.04 pinball at the two folds that
were already passing, because it over-covers them.

The live policy lives with the consumer: `MLStrategy._load_barriers` **refuses**
an artifact whose recorded verdict is not PASS, and serves one with **no**
recorded verdict with a warning (absence is "unknown", which is the state of
every artifact written before the field existed). `static_baseline_loss` scores
the incumbent 2.0×/4.0× constants in the same pinball units — the Phase 1
promotion comparison.

## Reads/writes

Reads: bars + `natr_14` from the feature pipeline vocabulary.
Writes: `models/<dir>/barriers_mae.pkl` + `barriers_mfe.pkl` +
`barriers_meta.json` (atomic, meta last) — via `save()`. As of 2026-09-14 the
live loader is wired (`MLStrategy`, behind `BARRIER_GEOMETRY_ENABLED`) and the
retrainer's promotion hook writes the artifact set (`RETRAIN_LEARN_BARRIERS`,
default on), so a model dir carries barriers as soon as a retrain passes the
**Angel/Devil** gate. Whether they may be *served* is a separate question, and
the answer lives in the artifact: see the verdict above and the promotion gate
below.

## Horizon contract

The label horizon (default 45) must equal the live `max_hold` limit; see the
glossary in `labels.py`. The live side enforces it: `MLStrategy._load_barriers`
raises when `horizon_` disagrees with `DEFAULT_HORIZON`.

## Promotion gate (status 2026-09-14)

`scripts/evaluate_barriers.py` is the Phase 1 gate: three expanding
chronological folds over the cached 6-pair M15 basket (297k labelled rows,
730 days). Its verdict is what authorises serving learned geometry, and as of
2026-09-14 it says **FAIL**:

```
fold 1: pinball learned=0.4816 static=1.9407 BEAT  coverage=0.953 ok
fold 2: pinball learned=0.5065 static=2.0235 BEAT  coverage=0.934 ok
fold 3: pinball learned=0.8457 static=2.4576 BEAT  coverage=0.905 UNDER-COVERED
VERDICT: FAIL — static bracket stays (promotion blocked, prior weights stand)
```

Verified by running it on this machine (CatBoost backend, `OMP_NUM_THREADS=4`).
The learned model beats the static constant on pinball loss on **every** fold —
by 2×–4× — and the gate fails only on fold 3's MAE coverage (0.905 against a
0.93 floor), i.e. the stop is slightly too tight in the most recent regime
under a two-feature evaluation vocabulary (`ppo`, `natr_14`). Do not flip the
live switch on the strength of the pinball column alone.

Since 2026-09-14 the verdict is also machine-readable: the script prints a
`VERDICT_JSON` line and writes the same object to `BARRIER_VERDICT_OUT` when
that env var is set, which is what `save(verdict=...)` embeds so the load path
can refuse an artifact that failed this gate.

### Does the learned bracket PAY? Measured 2026-09-14: it dilutes the toll, it does not add edge

Coverage and pinball say the quantile is honest; neither says the bracket earns.
The replay `evaluate_barriers.py` advertises in its own docstring ("realised-R
replay … same gates, same cost table") was built as a scratch harness and run:
barrier fitted on the older 75% of the pooled timeline, per-decile calibrated on
the last 20% of that window, then the same out-of-sample bars replayed twice
through `analysis.strategy_backtester.run_backtest` (gap fills, timeouts at the
realised move, the three live gates, per-instrument spread alphas, `max_hold=45`).

| arm | trades | win rate | gross | net | toll/trade |
|---|---:|---:|---:|---:|---:|
| static 2.0×/4.0× | 2653 | 0.284 | +0.040R | **−0.207R** | 0.246R |
| learned (sl 12.37×, tp 2.74×) | 1761 | 0.549 | +0.000R | **−0.040R** | 0.040R |
| constant 10.25×/2.74× (no model) | 1746 | 0.533 | −0.001R | **−0.049R** | 0.049R |
| matched (472 bars both arms traded) | 472 | 0.271 / 0.574 | +0.052 / +0.018 | −0.205 / **−0.023** | — |

Stop-outs collapse 57.1% → 3.6% and the win rate doubles, but gross expectancy
does **not** improve (+0.052 → +0.018R on matched bars). The entire net gain is
cost dilution: the toll per trade falls by 6.1× against a 6.2× wider stop — the
2026-08-08 finding ("widening dilutes a fixed cost over more risk") reproduced at
6× the width.

⚠️ **And a constant wide bracket captures all of it.** A fixed 10.25×/2.74×
payload (no model) matches the learned per-bar quantiles to within 0.004R: same
toll (0.0485R), same win rate (0.533 vs 0.550), same net (−0.049R vs −0.046R).
The fitted stop response is nearly flat (~8.5–10.25 ATR across the whole
distribution), and a conditional model that predicts a constant cannot beat the
constant. **The value is the bracket width, not the learning** — and width is a
`RiskProfile` setting, testable with no new machinery, which also keeps the
Devil's labels (built from the profile multiples at `retrainer.py:1469`) in
step with what is served. Sweeping `tau_mfe` does not help either: gross
expectancy stays ~zero at every setting from 0.50 to 0.85 as win rate and payoff
cancel. Both arms lose money; the entry population is unconditioned (a stub
firing every bar), so these levels measure the bracket, not the model's
selection. On the model-selected population the question is **unmeasurable** —
14 of 56,728 bars pass both stages, i.e. 11 trades and a ±0.87R confidence
interval — while at relaxed Angel bars the wide bracket's advantage (+0.08 to
+0.15R net) holds and every arm is still negative. See
`llm_reports/recons/2026-09-14_bracket-population-and-live-trade-rate.md`.

### The proxy is not why it fails (measured 2026-09-14)

`evaluate_barriers.py` fits its own estimator on `EVAL_FEATURES = ["ppo",
"natr_14"]` — a two-feature compromise documented in its docstring — so a fair
objection is that it does not evaluate the artifact the retrainer writes. It was
run both ways, at both timeframes:

| run | fold 1 | fold 2 | fold 3 | verdict |
|---|---|---|---|---|
| M15, 2-feature proxy | 0.953 | 0.934 | **0.905** | FAIL |
| **M15, full 17-feature artifact shape** | 0.951 | 0.937 | **0.912** | FAIL |
| H4, 2-feature proxy | **0.928** | **0.912** | **0.850** | FAIL |

(the full-vocabulary run used `engineer_features_and_labels` on the cached
basket, per-symbol labels at `horizon=max_hold=45`, `BASE_FEATURE_COLS`,
`generate_time_decay_weights`, the evaluator's own folds — 226,909 resolvable
rows)

A per-decile conformal rescale of the stop (`q_mae_scale`, fitted on a recent
slice that precedes the scored fold, cumulative-maxed so the response stays
non-decreasing) clears the gate at all three folds — 0.954/0.974/0.942 — which is
the evidence that the gate is passable at all. It over-covers the two folds that
already passed, costing 0.03–0.04 pinball there.

14× the features moved fold-3 coverage by 0.7 of a point and did not move the
verdict, and coverage falls monotonically across folds (0.951 → 0.937 → 0.912)
while pinball loss *beats* the static constant everywhere. So the failure is not
the evaluator's vocabulary and not a bad fit: the learned `Q_MAE(0.95)` is
calibrated ~4 points loose on the newest regime at both timeframes. Remedies
worth trying are producer-side (a conformal-style scalar rescale of `q_mae`
fitted on a recent holdout — which preserves the monotone response — or a higher
`tau_mae`), not consumer-side.
