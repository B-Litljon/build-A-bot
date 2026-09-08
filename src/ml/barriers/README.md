# `src/ml/barriers`

Quantile-regression barrier geometry: instance-specific stop/target distances
predicted per bar, replacing the static ATR-multiple bracket.

## Files

### `labels.py`

`compute_excursions(df, horizon)` appends MAE/MFE labels (NATR-scaled, clipped
at 0) to an OHLC frame, with a `resolvable` flag; the last `horizon` bars are
null-labeled and must be dropped before fitting. Short-direction labels are
the long labels mirrored (short MAE ≡ long MFE), so no separate labeling is
needed. `pinball_loss` is the quantile objective. Imported by the retrainer
research path and `estimator.py`.

### `estimator.py`

`BarrierEstimator` — two LightGBM quantile regressions (MAE τ=0.95 stop-side,
MFE τ=0.50 target-side) on the same feature vocabulary Angel/Devil see.
`BarrierOutput` carries price-unit distances for the Signal contract plus
NATR-space quantiles and implied reward:risk for the pre-trade veto. Falls
back to binned empirical quantiles when LightGBM is absent. `static_baseline_loss`
scores the incumbent 2.0×/4.0× constants in the same pinball units — the Phase 1
promotion comparison. Imported by the Phase 1 evaluation script (to be wired
into `src/core/retrainer.py` research flow in Phase 1's second half).

## Reads/writes

Reads: bars + `natr_14` from the feature pipeline vocabulary.
Writes (planned, Phase 1 second half): `models/<dir>/barriers_mae.pkl` +
`barriers_mfe.pkl`, atomic, alongside `threshold.json` — not yet wired to the
live path.

## Horizon contract

The label horizon (default 45) must equal the live `max_hold` limit; see the
glossary in `labels.py`.