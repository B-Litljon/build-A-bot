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

### `decision_grader.py` — how the SERVED model actually decides
Grades the live bot's recorded decisions against what price did next. The bot
writes an `ev="bar"` record to `logs/events-*.jsonl` for **every** bar it scores
(~500/day); only ~1 a week becomes a fill, so judging the model on fills throws
away 99.9% of its own evidence. This refetches the following bars, walks the
same ATR bracket the model was trained against, and joins the outcome back on.

Reports: **calibration** (when it says 0.30, does it win 30%?), a **threshold
sweep** on live probabilities, and a **per-behavior breakdown**.

First run, 2026-08-24 on 9,183 graded decisions, found the headline: the model's
confidence is *inverted at the top*. It wins 28-32% in its low bands (random
entry = 29.3%) but only **6.7% at the 0.40 bar it actually trades on**. The
Angel's own direction target is hit 33.3% on those bars vs 15.8% baseline — so
the model is RIGHT about direction and the bracket loses anyway, stopping out
73.3% of the time versus 56.0% elsewhere.

Two rules it keeps: decisions too recent for the 45-bar walk are **dropped, not
guessed**, and simulated fills flatter reality by roughly the spread toll, so
read it as a relative measure.

- **Imports from repo:** none at module level (the caller supplies graded bars).
- **Imported by:** `tests/test_decision_grader.py`, `run_decision_grader.sh`.
- **Reads:** `logs/events-*.jsonl`. **Writes:** nothing (the runner writes the
  report and `logs/graded_decisions.parquet`).
- **Scheduled:** weekly via cron, Sundays 12:00 PT. Read-only toward the soak.

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

### `strategy_backtester.py`
The strategy-agnostic scorer, added 2026-09-02. Everything else offline in this
repo is welded to the Angel/Devil model path; this takes any `BaseStrategy`,
walks it bar by bar over a Polars frame, and emits a trade ledger that
`behavior_matrix.score_ledger` can consume.

Two conventions that will mislead you if skipped:

- **`macro_win` is bracket resolution, not profitability.** It is 1 only when
  the target was reached. A timeout is 0 no matter where price ended, matching
  `retrainer._compute_devil_targets_atr`.
- **`gross_r` / `net_r` are the realised move** over the stop distance. So a
  trade that timed out slightly up has `macro_win = 0` and a small positive
  `gross_r`. The two columns answer different questions and are *meant* to
  disagree on timeouts.

Execution realism, all three from the 2026-09-02 review:

- a bar that **gapped** past a level fills at the open, not the level
  (`sl_gap` / `tp_gap`) — filling at the level books a price that never traded;
- a **timeout** pays its realised move, not the bracket's nominal payoff. The
  original booked ±payoff on sign alone, which inflated win rate by 10–12 points
  on every library strategy;
- when a `RiskManager` is passed, the **live gates run** (C time, B regime,
  A cost) and vetoed signals land in `gate_rejections` rather than being traded.
  Gate B alone vetoes the bottom 20% of the volatility window — roughly 60% of
  the tagger's `*_low` band — so a gateless run scores a population the live bot
  would never take.
- a strategy's learned barrier payload
  (`Signal.metadata["barrier_geometry"]`) is **passed through to
  `calculate_bracket` exactly as the live orchestrator passes it**, so replaying
  a barrier-enabled model measures the brackets the bot would actually place.
  Inert for every library strategy, which attach no payload.

`spread_alphas` gives a per-instrument toll mirroring Gate A's own proxy; the
shipped alphas span 0.072–0.903, a 12.6× range one flat constant cannot cover.

It deliberately does **not** import `core.retrainer`, keeping LightGBM, scipy
and the model artifacts out of its import graph — that is also what makes it
safe to run beside the live soak.

- **Imports from repo:** `strategies.base`, `analysis.behavior_matrix`
  (`DEFAULT_TOLL_R`, `_profit_factor`). `execution.risk_manager` is a
  `TYPE_CHECKING`-only annotation — the instance is passed in by the caller.
- **Imported by:** `walk_forward_tuner.py`, `build_strategy_matrix.py`, tests.
- **Reads/writes:** nothing directly; the caller owns the frame and the ledger.

### `walk_forward_tuner.py`
Grid search over strategy parameters, scored strictly out-of-sample across
expanding chronological folds. The point is to resist "tune until it fits" —
parameters are chosen on train windows and judged on windows they never saw.

- Validation slices are **prefixed with warmup bars** of prior history. Slicing
  at exactly `val_start` meant the backtester spent each fold's whole warmup
  window unable to trade, silently discarding those bars and resetting trailing
  state that live never resets.
- A **fresh strategy instance per fold**, so no fitted state can cross a fold
  boundary.
- `robust` requires the **Clopper-Pearson lower bound** on the pooled OOS win
  rate to clear the bracket's break-even rate, `1 / (1 + payoff)`. It previously
  computed that bound and ignored it, so a lucky 31-trade run read as robust on
  its point estimate alone.

- **Imports from repo:** `analysis.strategy_backtester`,
  `analysis.behavior_matrix`, `strategies.base`.
- **Imported by:** `build_strategy_matrix.py`, tests.
- **Reads:** a bars frame from the caller. **Writes:** JSON/CSV when run as a
  CLI.

### `build_strategy_matrix.py`
Scores the strategy library per market behavior and emits a routing table.
**This is the only legitimate source of a routing table** — the files in
`config/` named `*.example.json` are hand-authored templates, and the real one
carries `_generated_by` / `_generated_at` / `_basket` provenance fields so the
two can never again be confused.

Runs over a **basket**, not one symbol: nine behavior tags × five strategies
needs far more trades than one pair produces, and a cell built from a single
pair measures that pair rather than that regime. It defaults to the six
tradeable crosses and deliberately excludes XAU/XAG, which are broker-dead yet
made up 38–60% of prior model picks.

Three things it does that the first version did not, each of which changes the
answer:

- runs the **live gates** and reports the veto funnel per strategy;
- charges **measured per-instrument spread costs**;
- scores cells from the ledger's **realised R** rather than re-deriving them
  from a binary win flag times a flat toll — which had made every metric in a
  cell an affine transform of the win rate.

It also trims every frame to the window they all cover. A legacy parquet in
`data/raw` spanned a different two years from a fresh fetch, and pooling those
would have put two market eras in one cell.

`--sl-mult` / `--tp-mult` / `--max-hold` expose bracket geometry as a swept
axis; comparing two geometries is how you separate "no edge" from "cut off too
early" (see `llm_reports/recons/2026-09-02_strategy-library-behavior-matrix.md`,
finding 7).

- **Imports from repo:** `analysis.strategy_backtester`,
  `analysis.behavior_matrix`, `ml.regimes.behavior_tagger`, the strategy
  registry, and `execution.risk_manager` (only when gates are on).
- **Reads:** `config/spread_alphas_m15.json`, cached bars under
  `analysis_cache/strategy_matrix/`, else fetches via `data.factory`.
- **Writes:** a routing JSON at `--output`, a matrix CSV at `--matrix-out`, and
  the bar cache.

### `__init__.py`
Empty package marker.
