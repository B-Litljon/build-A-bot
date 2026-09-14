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

**Persistence is the live sidecar contract.** `save(model_dir, horizon)` writes
three files — `barriers_mae.pkl`, `barriers_mfe.pkl`, `barriers_meta.json` —
each atomically, **pickles first and the meta last**. `load(model_dir)` refuses
an incomplete set, a meta without `feature_cols`, or a meta without a
`horizon`; it returns an estimator carrying `horizon_`, which the caller checks
against the execution lifetime. `static_baseline_loss` scores the incumbent
2.0×/4.0× constants in the same pinball units — the Phase 1 promotion
comparison.

## Reads/writes

Reads: bars + `natr_14` from the feature pipeline vocabulary.
Writes: `models/<dir>/barriers_mae.pkl` + `barriers_mfe.pkl` +
`barriers_meta.json` (atomic, meta last) — via `save()`. As of 2026-09-14 the
live loader is wired (`MLStrategy`, behind `BARRIER_GEOMETRY_ENABLED`) but
**nothing in the retrainer writes these yet**; the only producer is the Phase 1
evaluation/research path, so a live model dir has no barrier set unless someone
puts one there deliberately.

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
