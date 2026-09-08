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
retrainer research path and `estimator.py`.

### `estimator.py`

`BarrierEstimator` — two LightGBM quantile regressions (MAE τ=0.95 stop-side,
MFE τ=0.50 target-side) on the same feature vocabulary Angel/Devil see.
`BarrierOutput` carries price-unit distances for the Signal contract plus
NATR-space quantiles and implied reward:risk for the pre-trade veto.
Volatility monotonicity is enforced by a fit-time AUDIT (`_audit_monotone`)
because LightGBM's quantile objective rejects `monotone_constraints` outright;
a fitted model whose stop response falls more than `_AUDIT_TOL_ATR` below its
running peak is refused. The no-LightGBM path falls back to binned empirical
quantiles whose per-natr-bin ladder is isotonised at fit, keeping that path
monotone by construction. `used_lightgbm_` records which backend a fit landed
on. `static_baseline_loss` scores the incumbent 2.0×/4.0× constants in the
same pinball units — the Phase 1 promotion comparison. Imported by the Phase 1
evaluation script (to be wired into `src/core/retrainer.py` research flow in
Phase 1's second half).

## Reads/writes

Reads: bars + `natr_14` from the feature pipeline vocabulary.
Writes (planned, Phase 1 second half): `models/<dir>/barriers_mae.pkl` +
`barriers_mfe.pkl`, atomic, alongside `threshold.json` — not yet wired to the
live path.

## Horizon contract

The label horizon (default 45) must equal the live `max_hold` limit; see the
glossary in `labels.py`.