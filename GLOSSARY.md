# GLOSSARY

The vocabulary of this codebase, defined once. Every module's `Glossary:`
docstring section points here for domain terms rather than re-explaining them.

**Three layers of documentation:**

| Layer | Where | What it tells you |
|---|---|---|
| 1 | this file | What each folder is for, and what the recurring words mean |
| 2 | `README.md` in each code folder | Per file: what it is, what it imports, what imports it, what it reads/writes |
| 3 | `Glossary:` in each module docstring | The meaningful identifiers *inside* that file |

**Related docs:** [`table-o-content.md`](table-o-content.md) is the "Haynes
manual" — a narrative tour of how the system works and how it got here. This
file is the dictionary; that one is the tour. (Note it references three files
that no longer exist: `src/core/trading_bot.py`, `main.py`, `src/main.py`.)
[`docs/`](docs/) holds three older design notes.

---

# Part 1 — Repository map

## What actually runs

Two live products share this repo, plus a lot of scaffolding around them:

- **V5 OANDA forex bot** — the intraday bot. Trades currency pairs and
  metals on OANDA, decides on each sealed bar, enforces its own stops in
  software. *Currently running* as a practice-account soak
  (`run_oanda.py --daemon --env practice --granularity 15`), running as the
  `soak.service` systemd user unit and kept alive by `soak_watchdog.sh` in
  cron.
- **V4 equities investor** — a monthly stock ranker on Alpaca. Picks 8 names,
  equal-weighted, max 2 per sector, rebalances once a month from cron. Holds
  for weeks; never watches a tick.

Everything else is either supporting infrastructure (data, features, training,
tests) or dormant experiments kept for reference.

## Top-level

| Path | Role | When it runs |
|---|---|---|
| [`src/`](src/README.md) | All library code. See its README for the package map. | training + live |
| [`scripts/`](scripts/README.md) | Hand-run tools: the whole V4 investor, model diagnostics, calibration, paper launchers. | manual + monthly cron |
| [`tests/`](tests/README.md) | 163 tests. No network, no broker, no real models. | CI / manual |
| `models/` | Trained model artifacts. Subdirectories are separate models: `forex/`, `forex_m15/`, `forex_swing/`, plus legacy root-level `angel_latest.pkl` and `dt_*` (day-trade experiment) and `v4_investor_lgbm.txt`. | read live |
| `data/` | Bars, ledgers, and processed datasets (`raw/`, `processed/`, `cache/`). Mostly gitignored. | training + analysis |
| `config/` | Configuration not in code — currently just a baked spread-cost table. | training + live |
| `logs/` | Run logs, including the multi-day `soak_*.log` files that calibration is mined from. | live |
| `docs/` | Three older design documents (architecture, day-trade model, an RSI/Bollinger strategy). | reference |
| `llm_reports/` | Written reports of work done, filed by category (`audits/`, `handoffs/`, `refactors/`, `recons/`, `stops/`). See its README for the convention. | reference |
| `m2m_prompts/` | The other half of that ledger: the briefs that *requested* the work. | reference |
| [`dashboard/`](dashboard/README.md) | Read-only web view of the soak: Rust (axum) API + TypeScript front-end. Reads `logs/events-*.jsonl`, `logs/status.json`, and OANDA REST. Cannot affect trading. | manual |
| `viz/` | Empty. | — |
| `src/autopilot/`, `src/research/` | **No source files on this branch** — only stale `__pycache__`. | — |

**Root-level Python** (entry points and ad-hoc tools):

| File | What it does |
|---|---|
| `run_oanda.py` | ⚠️ Launcher for the **currently-running** forex bot. |
| `run_live.py` | Launcher for the Alpaca equities/crypto scalper (not live). |
| `run_factory.py` | Launcher for the Factory path (Alpaca crypto). |
| `chop_ab_test.py` | Controlled A/B of the chop filter: same cached data, one variable changed. |
| `trading_mcp.py` | Read-only MCP server exposing soak observability as tools, plus two guarded control tools. |
| `backtest_60.py` | ⚠️ **Broken** — calls a constructor signature and a method that no longer exist. |

**Root-level shell:**

| File | What it does |
|---|---|
| `run_soak.sh` | Launches the forex soak. |
| `soak.service` | systemd **user** unit the soak actually runs as (symlinked into `~/.config/systemd/user/`). Declares the served model dir, and records how the process died — `systemctl --user status soak.service`. Needs `loginctl enable-linger`. |
| `soak_watchdog.sh` | Cron, every 5 min: starts `soak.service` if the soak died, and logs a post-mortem of the previous run. **Kill switch: `touch soak.off` *before* stopping it**, or it comes back within 5 minutes. |
| `run_pipeline.sh` | The offline loop: harvest → replay → grade → check drift → retrain if needed. Branches on exit codes. |
| `run_investor_rebalance.sh` | Monthly investor cron wrapper. |
| `run_chop_ab.sh` | Wrapper for the chop A/B harness. |

## `src/` packages

| Package | Role | When |
|---|---|---|
| [`data/`](src/data/README.md) | Every vendor adapter, behind one interface. `DATA_SOURCE` picks which. Providers **return empty rather than raising**, so one dead symbol can't abort a fetch. | training + live |
| [`ml/`](src/ml/README.md) | The feature factory: bars → the numbers models see. Training and live run the *same* generators in the same order. | training + live |
| [`strategies/`](src/strategies/README.md) | The decision. Bars in, `Signal` or `None` out. No broker, no sizing. | live |
| [`execution/`](src/execution/README.md) | Brokers, orders, position state, software stops. Three orchestrators for three markets. | **live** |
| [`core/`](src/core/README.md) | Two unrelated things: shared types + the Discord notifier, **and** `retrainer.py`, the entire training and promotion pipeline. | mixed |
| [`utils/`](src/utils/README.md) | Bar aggregation. One live module. | live |
| [`analysis/`](src/analysis/README.md) | Offline diagnostics, run by hand. Targets the legacy Alpaca stack. | never live |
| [`day_trading/`](src/day_trading/README.md) | Dormant "V4.0" 5-minute experiment. Fully self-contained, `dt_`-prefixed artifacts. | never live |

---

# Part 2 — Domain glossary

## The two-stage model

**Angel** — stage one. Tuned for *recall*: catch as many real opportunities as
possible, tolerating false alarms. It **proposes**. Fires above the Angel
threshold — calibrated per retrain from out-of-fold scores since 2026-08-29
(`ANGEL_THRESHOLD` env var pins the old fixed bar, default 0.40).
*(`src/core/retrainer.py`)*

**Devil** — stage two. Tuned for *precision*, and trained only on the bars the
Angel already liked: of these candidates, which are actually worth taking. It
**vetoes**. Two stages exist because one model tuned for both jobs does neither
well. *(`src/core/retrainer.py`)*

> For non-technical audiences these are called **Thing 1** and **Thing 2**.
> Code and internal docs keep Angel/Devil.

**Meta-labeling** — the name for that arrangement: a second model that judges
the first model's signals rather than the market directly.

**threshold** — the confidence level above which a stage acts. The Devil's is
not fixed; the retrainer tunes it per model and saves it to `threshold.json`,
which the live strategy loads at startup and on hot reload. The Angel's is
likewise tuned per model since 2026-08-29: calibrated from out-of-fold
probabilities (`_find_optimal_angel_threshold`) so the proposal bar tracks
the model's own score distribution, then pinned into `threshold.json` — a
deployed pair always runs at the bar its Devil and brackets were fitted for.
The `ANGEL_THRESHOLD` env var (default 0.40, `src/core/thresholds.py`) is the
fallback and selects the old fixed-bar mode when explicitly set.

**angel_prob / devil_prob** — each stage's output probability, carried in a
signal's metadata and shown in Discord alerts.

## Bars and time

**bar / candle** — one time slice of price: **OHLCV** = open, high, low, close,
volume. *(`src/utils/bar_aggregator.py`)*

**sealed bar** — a bar whose time window has fully elapsed and will never change.
Strategies only ever act on sealed bars; acting on a forming bar means acting on
a number that is still moving.

**M1 / M15** — 1-minute and 15-minute bars. The live soak runs M15.

**HTF (higher timeframe)** — the slower chart, read for context alongside the
fast one. Columns prefixed `htf_`. *(`src/ml/features/v3_features.py`)*

**`available_at` / lookahead guard** — the subtlest correctness rule in the
repo. A 5-minute bar stamped 12:00 is not *finished* until 12:05, so using it at
12:01 is reading the future. Each slow bar's timestamp is pushed forward one
full timeframe before joining. Get this wrong and backtests look brilliant while
live trading fails. *(`V3HTFFeatures`)*

**lookahead / leakage** — any way information from the future reaches a
decision that could not have known it. The repo guards against it in at least
four separate places: `available_at`, the 45-day fundamentals lag, the 60-day
investor embargo, and out-of-fold probabilities.

**warm-up** — pre-loading enough history that indicators can compute before the
first live decision. 260 bars is the usual figure (a 50-period average on
5-minute bars needs ~250 one-minute bars).

**history seam** — the junction where replayed warm-up history meets the live
stream. They overlap in time, so the orchestrator tracks where history ended to
avoid double-counting. *(`oanda_forex_orchestrator.py`)*

**window floor** — rounding a timestamp down to its clock window (12:34 → 12:30
for 5-minute bars), so bars align to the wall clock rather than to whenever data
happened to arrive. *(`LiveBarAggregator`)*

**forward fill / synthetic flat candle** — a placeholder bar (OHLC = previous
close, volume 0) inserted when the feed drops data, keeping the series evenly
spaced so indicator maths stays valid.

**sessions** — `session_asia` / `session_london` / `session_ny` /
`session_overlap`, 0/1 flags from the UTC hour. The overlap (12:00–16:00 UTC,
London and New York both open) is the busiest window. They exist because
identical indicator values mean different things at 03:00 than at 14:00.

## Prices, volatility, and cost

**bid / ask / spread** — the price you can sell at, the price you can buy at,
and the gap between them. The spread is the toll you pay to enter and exit.

**mid price** — (bid + ask) / 2. What bars are built from on OANDA.

**pip** — the standard small unit of currency-pair movement: 0.0001, or 0.01 for
anything quoted in yen. Meaningless for gold, which is why metals use a percent
floor instead. *(`risk_manager.py`)*

**ATR (average true range)** — typical bar movement in price terms.

**NATR** — the same thing as a **percent of price**. This is the volatility unit
almost everything scales from — bracket widths, the cost gate, position size —
because a percentage lets one model span gold near \$2,700 and a currency pair
near 1.09. `atr_abs = close × natr_14 / 100`.

**bracket / SL / TP** — the stop-loss and take-profit levels placed either side
of an entry. Sized as multiples of NATR, so they adapt to how much the
instrument is actually moving. Equities default 0.5× / 3.0× (a 6:1 payoff);
**the forex profile overrides this to 1.0× / 2.0×** (2:1) — always check the
profile rather than the module constants.

**alpha / `spread_atr_alpha`** — trading cost expressed as a fraction of a
typical move. Dimensionless: 0.07 means the toll is 7% of a normal move (cheap);
0.93 means 93% (effectively untradeable). The placeholder is 0.15; real
per-instrument values are measured live and baked by
`scripts/bake_spread_alphas.py`.

**`cost_ratio`** — that same cost inequality turned into a model *feature*, so
the model can see cost rather than having expensive setups silently filtered out
behind it. *(`V3CostFeatures`)*

**notional** — the total value of a position. Capped absolutely
(`max_notional_cap`), and floored at \$50 so no dust position is opened.

**risk per trade** — 2% of account equity. Position size is derived from this
and the stop distance, not set as a fixed quantity.

## The gates (the "chop filter")

Applied *after* a strategy already wants to trade. A model can be perfectly
right about direction and still lose money if the cost of trading exceeds the
move — that is what these prevent. *(`src/execution/risk_manager.py`)*

**chop veto / chop filter** — the collective name for all three gates.

**Gate A (cost)** — reject when the stop distance is smaller than the cost of
trading: `sl_dist < k_eff × spread`.

**Gate B (regime)** — reject when current volatility sits in the bottom 20% of
its own recent 260-bar window. A market too quiet to move can't reach the target
before the hold limit expires.

**Gate C (time)** — reject everything inside the daily rollover blackout
(default 16:55–17:30 **New York time**, so it tracks daylight saving instead of
drifting an hour twice a year), when spreads briefly blow out roughly tenfold.

**k_eff / coupling** — the effective safety multiple on cost. Scales with
volatility above the median, in either direction ("tighten" = more cost
discipline as volatility rises, "loosen" = less), and is clipped at ≥ 1.0 so a
passing trade's cost can never exceed its own stop distance.

**cold start** — with less than 60 bars of volatility history the regime gate
stands down, so a just-restarted bot isn't frozen by a half-filled buffer.

**kill switch** — a blunt override: `ATR_KILL_SWITCH_THRESHOLD` (0.5204 NATR)
skips bars judged too violent to trade, regardless of what the models think.

**symmetry contract** — the gate logic is *shared* between live execution and
training (`retrainer._compute_chop_veto_mask`), so the model only ever learns
from bars the live bot would actually take. Changing a gate on one side without
the other is a silent correctness bug.

## Telemetry

The machine-readable half of the bot's output, added 2026-07-31 so the
dashboard reads facts rather than regex-scraping a log written for humans.
*(`src/core/events.py`, consumed by `dashboard/`)*

**event stream** — `logs/events-YYYY-MM-DD.jsonl`, one JSON object per line,
append-only. Every line carries `ts` (UTC ISO-8601) and `ev` (the kind); see
`dashboard/README.md` for the per-kind fields.

**bar event** — one record per *evaluation*, carrying that bar's angel and
devil probabilities and the outcome. The heartbeat summarises 30 bars; this is
the bar itself, and it is the first per-bar probability record this project has
kept.

**status snapshot** — `logs/status.json`, the bot's "right now": positions,
counters, config, last bar per symbol. Replaced atomically (temp + rename) each
bar, so a reader never sees a partial file.

**opt-in telemetry** — nothing is written until `events.configure()` is called.
Importing a strategy in a test or backtest must never append to the live bot's
logs. `EVENTS_ENABLED=0` disables it outright.

**best-effort** — the sink never raises, never blocks (bounded queue + daemon
writer; a full queue DROPS events), and is never called from the tick path.
Losing telemetry always beats stalling a bar.

## The entry guards

Distinct from the gates above: the gates ask "is this trade worth its cost?",
these ask "should we be taking *this* trade *now*, given what we just did and
what we already hold?" They live in the orchestrator, not the RiskManager, and
have no training-side mirror — they constrain sequencing and concentration,
not the merit of a setup. *(`src/execution/oanda_forex_orchestrator.py`,
both added 2026-07-30)*

**post-exit cooldown** — a symbol cannot be re-entered for a set window after
its position closes (`OANDA_REENTRY_COOLDOWN_SECONDS`, default one bar period;
0 disables). A fresh stop is evidence the read was wrong, and the next bar is
too soon to re-litigate it. Blocks *opening* only — reversing a position that
is still open is a different act and stays allowed.

**currency leg** — one side of an instrument, signed by direction. Long
GBP_JPY is `+GBP, −JPY`; short is the mirror. XAU_USD decomposes the same way,
which is what makes a metals position and a fiat cross comparable.

**correlated-exposure cap** — the maximum number of open positions sharing the
same signed currency leg (`OANDA_MAX_PER_CURRENCY`, default 2; 0 disables).
Long GBP_JPY + long AUD_JPY + long NZD_JPY is not three trades, it is one
short-yen bet at triple size — which is exactly how it behaved on 2026-07-30,
when all three lost together.

**reservation (`_pending_entries`)** — an entry that has passed the cap but
whose fill has not returned yet, counted as though already held. Without it,
two signals evaluated on the same bar both see the pre-trade world and both
pass a cap they jointly breach.

## Training

**feature** — one input column the model sees. The live set is 22 columns, or 23
with `cost_ratio`. *(`src/ml/features/v3_features.py`)*

**target / label** — the "right answer" a model is trained to predict. Two
exist here:
- **macro target** — replay each bar forward up to 45 bars: did the target or
  the stop come first? The stop is checked first, so a bar touching both counts
  as a loss.
- **survival target** — did price avoid the stop for the next 5 bars? This is
  what the Devil actually trains on, because its inputs describe a 1–5 minute
  horizon and asking it about a 45-bar outcome was an unlearnable mismatch.

**walk-forward validation** — train on the past, score on the future it never
saw, then roll forward and repeat. **Expanding window** means each fold's
training set grows. Never a random split: shuffling time series lets a model
learn from its own future.

**OOS (out-of-sample)** — data the model never trained on. The only data a
performance claim means anything on.

**OOF (out-of-fold)** — probabilities produced by models that never saw the row
being scored. Used to pick the Devil's training rows, so it doesn't inherit the
Angel's overconfident in-sample opinions.

**embargo** — a deliberate gap between training and test windows. The V4
investor's target looks 60 days ahead, so a training window's tail overlaps the
test window's future; a 60-day gap removes the overlap.

**time-decay weights** — weighting recent rows more heavily (factor 0.95) so the
model leans toward current market behaviour.

**promotion gate** — the pass/fail that decides whether new models overwrite the
live ones. Failing is a *healthy* outcome, not a crash: the retrainer exits 2 and
the previous weights stay in place.

**holdout** — a chronologically last slice of data carved off before any feature
engineering, used to score the actual artifact that gets served. Passing the
fold gate is necessary but not sufficient; the served model must also earn its
bars on the holdout. Recorded in `metadata.json` so a deployed model can be
checked against what it actually earned. *(`src/core/retrainer.py`)*

**artifact holdout gate** — the additional pass/fail applied to the final model
on the holdout, using the same Brier/EV bars as the fold gate, the frozen
production threshold, and — for profit factor — the Clopper-Pearson **lower
confidence bound** on the macro win rate rather than the point estimate
(a PF on 55-82 trades flips with the clock; the bound is the fix). When the
fold gate fails, the holdout is still scored for diagnostics (Fold 3 models,
`ValidationReport.holdout.diagnostic_only`); the fold verdict stands either
way. Disabled by `RETRAIN_HOLDOUT_FRAC=0`; when disabled or empty the metadata
records the bypass explicitly. *(`src/core/retrainer.py`)*

**Clopper-Pearson bound** — the one-sided lower confidence bound on a binary
win rate (`Beta(1-confidence; wins, losses+1)` quantile), mapped through
the profit-factor formula for the holdout gate. Chosen over the Wilson
approximation because Wilson under-covers below ~40 trades — precisely the
sample sizes that used to flip the verdict. Exact for independent trades;
the 45-bar macro walks overlap in price, so in practice the bound is
conservative rather than a literal coverage guarantee — the safe direction
for a promotion gate. *(`src/core/retrainer.py`)*

**unresolvable tail** — the last `max_hold` bars per symbol of a raw slice,
whose bracket walk runs off the end of the frame and resolves "timeout →
loss" no matter what the price actually did. Those labels are systematically
wrong, so the engineered remainder and holdout each drop them after
engineering (the **boundary purge**); the cutoffs are derived from the raw
series because the walk needs the contiguous pre-veto path. *(`src/core/retrainer.py`)*

**lift over random vs lift over benchmark** — two different questions, and for a
long time the investor only asked the first. "Better than guessing" is measured
against the base rate (a top-*quintile* target makes random guessing score 0.20).
"Better than doing nothing" is measured against equal-weighting the whole
universe. A model can pass the first and fail the second — the 2026-07-03
investor promotion did exactly that — so the investor's gate now requires both.

**benchmark gate** — the investor's lift-over-benchmark check
(`scripts/investor_train_model.py`). Simulates the basket actually deployed
(top 8, max 2 per sector) against equal-weighting all 96 names. The bar that
binds is a **t-statistic** on the monthly excess, not the excess itself: that
quantity carries a standard error of roughly 60 basis points over ~30 months,
so point estimates are nearly uninformative on their own. The shipped model is
the worked example — +47.3 bps/month reads as substantial and is t = 0.75.

**effective sample size** — how much independent evidence a set of overlapping
measurements really contains. The gate re-measures at five fold alignments, but
those share nearly all their rows and their excess series correlate 0.71, so
they amount to about 2.2 independent samples. Re-slicing the same data guards
against a lucky boundary; it does not manufacture statistical power. The same
caution applies to the horizon study's "held at 5 of 5 alignments".

## Scoring

**Brier score** — mean squared error between predicted probabilities and what
actually happened. 0 is perfect; lower is better. This is the *calibration*
test: does "70% confident" actually mean right about 70% of the time.

**profit factor (PF)** — gross wins ÷ gross losses. 1.0 is break-even.

**EV (expected value)** — average return per trade.

**win rate** — fraction of trades that won. Descriptive only; a high win rate
with tiny wins and huge losses is worthless, which is why PF and EV are gated on
instead.

**max drawdown** — the worst peak-to-trough fall in the equity curve; how much
pain the strategy puts you through, not where it ends up.

**P@K / lift** — investor-side. Of the top K picks, what fraction really landed
in the top quintile. Since the target *is* a quintile, random guessing scores
0.20, so gates threshold on **lift** (P@K ÷ 0.20) rather than raw precision.

**NDCG** — a whole-ordering score that rewards good names placed near the top.

## Diagnosing a quiet model

**drift** — the model's inputs or its calibration have moved away from what it
was trained on.

**PSI (Population Stability Index)** — one number for how far a feature's live
distribution has moved from its training distribution.

**null calibration** — ⚠️ **the reason the textbook PSI thresholds (0.10 / 0.25)
are not used here.** Those assume independent samples. Market bars are heavily
autocorrelated, so any short window sits in a narrow slice of the full range and
scores high PSI even when nothing is wrong. Instead, the system measures what
PSI *ordinary training windows* produce, and only calls drift when live PSI
beats that null's upper tail. *(`src/ml/feature_stats.py`)*

**SHAP / TreeSHAP** — exact per-feature attribution: how much each input pushed
the model's opinion up or down. Turns "the model is quiet" into "the model is
quiet *because* momentum features are suppressing it".

**DRIFT vs HONEST** — the two verdicts from `scripts/probe_model.py`. DRIFT
means retraining may help; HONEST means the model is working correctly in an
edge-less stretch and retraining would be chasing noise.

**regime** — ⚠️ **two different meanings.** In `risk_manager.py` and
`reinforcement_voter.py` it means a *volatility band*. In `src/ml/regimes/` it
means a *hidden market mode* inferred statistically by a Gaussian HMM (states
are unnamed — "state 0" has no fixed meaning across symbols or runs). Unrelated.

**behavior tag** — a plain-language label for what the market was doing at one
bar, along two axes: volatility (`low`/`normal`/`high`) and trend
(`range`/`mixed`/`trend`), combined into names like `trend_high` or
`range_low`. A third state, `cold`, means the trailing window was not yet warm
enough to judge — it is excluded from analysis, never pooled. Produced by
`ml/regimes/behavior_tagger.py`. Distinct from both meanings of *regime* above:
it is computed for **analysis**, is never fed to the model, and is deliberately
causal (trailing window only) so a tag means the same thing offline and live.

**behavior matrix** — the table the tagger exists to produce: behavior tag ×
candidate configuration, scored on trade count, win rate, expectancy and profit
factor. Answers "which configuration earns its keep in *this* kind of market".
Its output is a hypothesis to test, not a promotion decision — per-cell samples
are thin and the scoring is in-sample unless walk-forward.

**candidate** — one evaluation configuration in the behavior matrix: a model
directory, a threshold pair, bracket widths, and a chop-gate config. Not new
strategy code — every candidate runs the same strategy with different settings.

## Live operation

**orchestrator** — the object that owns the running loop: receive bars, run the
strategy, place orders, watch positions. One per broker/market.

**event loop** — the single asyncio thread that owns all mutable state. Model
inference and REST calls are pushed to worker threads (`asyncio.to_thread`)
because they'd otherwise stall the price feed.

**software SL/TP** — stops and targets enforced *by the bot process*, not by the
broker. The consequence is that **the process being alive is a safety
requirement** — a dead bot means an unwatched position. Almost every piece of
hardening in `src/execution/` follows from this.

**watchdog** — ⚠️ **two different things.** (1) The in-process loop that checks
price against stop/target and closes the position. (2) `soak_watchdog.sh`, the
cron job that restarts the whole bot if it died.

**heartbeat** — ⚠️ also two. (1) OANDA's keepalive message every ~5 seconds;
60 seconds of silence means the feed is dead. (2) `MLStrategy`'s periodic log of
recent Angel probabilities, so an operator can see the model is alive during
long no-trade stretches.

**liveness / stale stream** — silence on the feed is treated as failure. On a
stale stream the bot reconnects **and flattens** — holding a position you can't
see is worse than holding nothing.

**flatten** — close everything.

**net position** — one signed number per instrument (positive long, negative
short, zero flat). No separate lots, because US rules require FIFO closing and
forbid holding both directions at once; one signed number makes that impossible
to violate by construction. *(`oanda_order_manager.py`)*

**reconcile on boot** — on startup, ask the broker what's actually open. A
restart must adopt reality, not assume it's flat, or a position left by a
crashed process runs with nothing watching its stop.

**hot reload** — the live strategy compares model file timestamps each bar, so a
retrain lands without a restart.

**atomic write** — every model artifact is written to a temp name then renamed
into place, so the hot reloader can never read a half-written file.

**state machine** — the per-symbol lifecycle: FLAT → PENDING → IN_TRADE →
PENDING_EXIT → COOLING → FLAT. **Cooling** is a 5-minute pause after any close,
so one choppy stretch can't cause repeated re-entries.

**SymbolContext** — all per-symbol runtime state in the Alpaca orchestrator.
Owned exclusively by the event loop; worker threads get immutable snapshots and
return frozen results. Enforced by a regression test.

**soak** — a long unattended run on the practice account, used to gather
evidence (spread measurements, gate telemetry) rather than to make money.

**paper / practice vs live** — simulated money vs real. Both brokers default to
simulated; real money requires an explicit opt-out.

**SPREAD_CALIB** — the periodic log line recording measured spread costs per
instrument. Mined by `scripts/bake_spread_alphas.py` into a cost table.

## Artifacts

Written into a model directory (`models/forex_m15/`, etc.) — all atomically, all
travelling together so a model always matches its assumptions:

| File | What it is |
|---|---|
| `angel_latest.pkl` / `devil_latest.pkl` | The two trained models. |
| `threshold.json` | The Devil threshold tuned for *this* model, plus (since 2026-07) the pinned Angel bar the pair was trained at. |
| `metadata.json` | Asset class, timeframe, symbols, training date, and holdout metrics (or bypass reason). Read at boot to confirm the bot trades what the model was trained on and to check what the artifact earned on untouched data. |
| `feature_stats.json` | Training feature distributions + null calibration, for drift probing. |
| `spread_alphas.json` | The cost table this model was trained against. |
| `hmm_latest.pkl` | Per-symbol regime models, when that experiment is on. |

Under `data/`:

| File | What it is |
|---|---|
| `oos_bars.parquet` | Harvested bars the offline loop runs on. |
| `signal_ledger.parquet` | Signals recorded by `replay_test.py`. ⚠️ A `signal_ledger.csv` also exists; `core/resolver.py` reads the CSV while everything else uses the parquet. |
| `resolved_ledger.csv` | Signals graded win/loss. |
| `evaluation_results.parquet` | Scored performance. |
| `drift_report.json` | Per-volatility-band calibration analysis. |
| `active_trades.json` | Open-trade state, so a restart can recover. |

## Discord personas

Alerts are tagged by sender so the source is obvious at a glance:
**Build-A-Bot Executive** (Alpaca path), **Build-A-Bot V5 Forex** (OANDA
path), **The Accountant** (retraining verdicts and drift alerts).
*(`src/core/notification_manager.py`)*
