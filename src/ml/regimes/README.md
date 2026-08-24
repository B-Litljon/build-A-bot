# `src/ml/regimes`

Market-regime detection — working out what "mode" the market is in (quiet
drift, trending, violent).

Two unrelated approaches live here, and they are **not** interchangeable:

1. **`hmm_regime.py`** — a *statistical* detector. Infers hidden states nobody
   labelled, and feeds its confidence to the classifier as extra inputs.
   ⚠️ Experimental and off by default; only active when `RETRAIN_USE_HMM=1`.
2. **`behavior_tagger.py`** — a *deterministic* labeller. Assigns each bar a
   plain-language behavior name for **analysis**, not for the model to consume.
   Always on when called; nothing in the live path depends on it yet.

See the root [GLOSSARY.md](../../../GLOSSARY.md) for the "regime" name
collision — it means a volatility band in `risk_manager.py` and a hidden state
in this folder.

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

### `behavior_tagger.py`
Labels each sealed bar along two axes — volatility (`low`/`normal`/`high`) and
trend (`range`/`mixed`/`trend`) — producing composite labels like `trend_high`
and `range_low`. These are the cells the behavior matrix scores model variants
over: "which configuration earns its keep when the market looks like *this*".

The one thing that matters here is **causality**. Both axes are the percentile
rank of the current bar inside a *trailing* window, which is the same estimator
the live Gate B runs in `RiskManager._evaluate_dynamic_gates` — same window
(260 bars), same finite-filtering, same cold-start rule (60 bars). That is what
lets a label mean the same thing in a backtest and in the bot.

⚠️ **Do not "simplify" this to `reinforcement_voter.calculate_atr_regimes`.**
That function takes its p33/p67 cut points over the *whole* frame, so a bar's
label depends on bars that had not happened yet. Fine for a post-hoc report;
fatal for anything conditioning a decision, and impossible to reproduce live.
`tests/test_behavior_tagger.py` pins causality by prefix-invariance (truncating
the future must not change any earlier tag) and pins the estimator by driving
the real `RiskManager` and checking the rank predicts Gate B's verdict.

- **Imports from repo:** none (numpy only).
- **Imported by:** `tests/test_behavior_tagger.py`. No production caller yet —
  it is the first phase of the behavior-matrix tool.
- **Reads/writes:** nothing.

### `__init__.py`
Package marker with a one-line description. (This is the only `src/ml`
subpackage that has one.)
