# `src/ml/regimes`

Market-regime detection — inferring the hidden "mode" the market is in (quiet
drift, trending, violent) and handing the classifier its confidence in each one
as extra inputs.

⚠️ **Experimental and off by default.** Only active when `RETRAIN_USE_HMM=1`.

## Files

### `hmm_regime.py`
Fits one 3-state Gaussian hidden Markov model **per symbol** on
`(log_return, natr_14)` — direction-of-move and size-of-move. Price level is
deliberately excluded so regimes stay comparable across instruments and eras.
The fitted model's posterior probabilities become three feature columns,
`hmm_state_0_prob` … `hmm_state_2_prob`.

Things that will bite you if you assume otherwise:

- **The states are not named or interpreted.** "State 0" has no fixed meaning
  across symbols or across runs — it's whatever the fit converged on.
- **Symbols with under `MIN_FIT_ROWS` (200) rows are skipped**, and skipped or
  failed symbols get a uniform 1/3 across all three columns — a deliberately
  uninformative value the classifier can ignore, rather than a gap that would
  drop the row.
- **Leakage rule:** fit on training-fold rows *only*, then score both training
  and validation frames with that fitted model. Fitting on everything first
  would let a validation fold's own future inform its regime labels.

- **Imports from repo:** none.
- **Imported by:** `src/core/retrainer.py` (fits and saves),
  `src/strategies/concrete_strategies/ml_strategy.py` (loads for live
  inference).
- **Writes:** a joblib dict of per-symbol models, saved atomically next to the
  Angel and Devil pickles so all three artifacts travel together.

### `__init__.py`
Package marker with a one-line description. (This is the only `src/ml`
subpackage that has one.)
