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

### `behavior_matrix.py` — the exception to the warning above
Scores candidate **configurations** per market behavior: rows are behavior tags
from `ml/regimes/behavior_tagger.py` (`trend_high`, `range_low`, …), columns are
candidates, cells carry trade count, win rate, expectancy and profit factor with
bootstrap intervals. Answers "which configuration earns its keep when the market
looks like *this*", and `recommend()` names the winner per behavior — or returns
`None`, which is a real answer when the evidence is thin.

Unlike its neighbours it is **current, not legacy**: every trade it scores is a
Devil-approved OOS trade captured from the retrainer's own expanding
walk-forward via `validate_candidate(oos_ledger=...)`. That means the chop veto,
the labels and the feature pipeline are identical to the ones the promotion gate
sees — there is no second, drifting scorer. It deliberately does *not* build on
`replay_test.py` / `evaluate_performance.py`, which are Alpaca-era and hardcode
the equities setup.

Two rules it enforces so results stay honest:

- **Cost is not optional.** Cells are reported net of a per-trade spread toll in
  R, with gross beside it. Gross PF is the trap this project has already been
  caught by (gross 1.373 → net 1.004 on the shipped model). Because the toll is
  a constant subtracted from every trade, it never changes the *ranking* of
  cells — only where the zero line falls.
- **Thin cells are flagged, not hidden.** Below `MIN_CELL_TRADES` (30) a cell is
  marked uninformative and `recommend()` skips it. A recommender that always
  recommends is a random number generator with a nice interface.

⚠️ **Lookback must be set with `RETRAIN_DAYS_BACK`**, not by passing `days_back`
to the fetch. The fold schedule is derived from the module constant, so a longer
fetch without that env var silently trains on the oldest 60 days and discards
the rest. `validate_candidate` now warns when the frame's span exceeds it.

- **Imports from repo:** `ml.regimes.behavior_tagger`.
- **Imported by:** `tests/test_behavior_matrix.py`. Driven by hand.
- **Reads/writes:** nothing directly; the caller supplies the captured ledger.

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

### `optimize_brackets.py` — ⚠️ import-dead
Holds the model's signals fixed and sweeps 100 bracket geometries
(4 stop widths × 5 target widths × 5 hold limits) to ask what *would* have made
the most of them.

> ⚠️ **Does not currently import**: it references `get_alpaca_client`, removed
> from `core/retrainer.py` on 2026-05-22 (59a1125). Any bracket numbers it
> ever produced predate the M15 era — the live M15 brackets actually come from
> `RiskProfile.for_asset_class("forex")`, not from this script.

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
