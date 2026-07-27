# `src/analysis`

Offline diagnostics. **Nothing here runs during live trading** — every file is
a standalone script (`python -m src.analysis.<name>`) that answers one "why is
the model behaving like this?" question against saved data.

Nothing imports anything here; they're all run by hand.

⚠️ **Read these with a date in mind.** They target the *legacy Alpaca equities*
stack: the root-level `models/angel_latest.pkl` (not the current
`models/forex/`, `models/forex_m15/` layout), the 5-symbol equities basket, and
in places fixed-percentage brackets rather than the volatility-scaled ones now
used. Those old artifacts still exist on disk, so the scripts *run* — they're
just diagnosing the old models unless you repoint them.

See the root [GLOSSARY.md](../../GLOSSARY.md) for domain terms (angel/devil,
bracket, NATR, drift, OOS).

## Files

### `failure_modes.py`
Answers **"*how* are the losers losing?"** — stopped out instantly, bled out
slowly, or timed out flat. The distinction matters because each points at a
different fix: a too-tight stop, no real edge, or too short a hold. Its
`FAST_SL_CUTOFF = 3` marks a stop hit within 3 bars as "fast", the signature of
an entry that was wrong immediately rather than one that drifted.

- **Imports from repo:** `ml.feature_pipeline`, `ml.features.v3_features`.
- **Reads:** `models/angel_latest.pkl`, `models/devil_latest.pkl` (fallbacks
  under `src/ml/models/`), `data/oos_bars.parquet`, `data/raw/`.
- **Writes:** console only.

### `optimize_brackets.py`
Holds the model's signals fixed and sweeps 100 bracket geometries
(4 stop widths × 5 target widths × 5 hold limits) to ask what *would* have made
the most of them.

> ⚠️ The obvious trap: sweeping 100 combinations against one fixed history and
> keeping the winner is curve-fitting. Treat the result as a hypothesis to test
> on fresh data, not a setting to adopt. `MIN_TRADES = 10` blunts the worst of
> it by discarding combinations with too few trades to mean anything.

- **Imports from repo:** `ml.feature_pipeline`, `ml.features.v3_features`,
  `execution.live_orchestrator`.
- **Reads:** model pickles + OOS bars. **Writes:** console only.

### `optimize_threshold.py` — oldest file here
Sweeps the decision cut-off (0.30–0.50). On imbalanced data the default 0.50 is
rarely right: if 1 bar in 20 is a genuine opportunity, demanding 50% confidence
rejects nearly everything.

Predates Angel/Devil — single model, fixed ±% brackets, and its **own hardcoded
`FEATURE_COLS`** that may no longer match the pipeline. Uses a **date** split
(`SPLIT_DATE = 2024-01-01`), not a random one, which is correct for time series.

The live equivalent of this job is `retrainer._find_optimal_threshold`, which
sweeps per retrain and writes `threshold.json`.

- **Imports from repo:** none. **Reads:** `data/processed/training_data.parquet`,
  `src/ml/models/rf_model.joblib`. **Writes:** console only.

### `reinforcement_voter.py`
Splits performance by **volatility band** instead of reporting one average — a
model can look fine overall while being badly wrong in exactly the conditions
that matter. `CALIBRATION_TOLERANCE = 0.20` flags any band where predicted and
actual win rates diverge by more than 20%.

Historical significance: this is where `live_orchestrator`'s
`ATR_KILL_SWITCH_THRESHOLD = 0.5204` came from — the high-volatility band where
calibration broke down.

> Note "regime" here means a **volatility band**, not the hidden-state model in
> [`src/ml/regimes/`](../ml/regimes/). Same word, two meanings.
>
> ⚠️ It expects `data/signal_ledger.parquet`, while `src/core/resolver.py`
> writes `data/signal_ledger.csv`. Different formats, same base name — they are
> not interchangeable.

The current drift tool is `scripts/probe_model.py`, which compares live feature
distributions against a saved training snapshot and needs no replay data.

- **Imports from repo:** none.
- **Reads:** `data/evaluation_results.parquet`, `data/signal_ledger.parquet`,
  `data/oos_bars.parquet`. **Writes:** `data/drift_report.json`.

### `__init__.py`
Empty package marker.
