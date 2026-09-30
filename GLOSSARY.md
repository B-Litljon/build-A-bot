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
| [`tests/`](tests/README.md) | 538 tests. No network, no broker, no real models. | CI / manual |
| `models/` | Trained model artifacts. Live: `forex_m15_wide/` (served by the soak) with `forex_m15_wide_backup_20260829/` as rollback, `forex_h4_catboost/` (gated-out candidate, kept for the audit). Root-level `dt_*` is the day-trade experiment and `v4_investor_lgbm.txt` the monthly investor model. The ~24 one-off experiment dirs were pruned 2026-09-17. | read live |
| `data/` | Bars, ledgers, and processed datasets (`raw/`, `processed/`, `cache/`). Mostly gitignored. | training + analysis |
| `config/` | Configuration not in code — currently just a baked spread-cost table. | training + live |
| `logs/` | Run logs, including the multi-day `soak_*.log` files that calibration is mined from. | live |
| `docs/` | Older design documents (architecture, day-trade model, an RSI/Bollinger strategy) plus `GEMINI_NOTEBOOK_STUDY_NOTES.md`, a current sectioned architecture primer written 2026-09-18 for long-context study/notebook use. | reference |
| [`llm_reports/`](llm_reports/README.md) | Written reports of work done, filed by category (`audits/`, `handoffs/`, `refactors/`, `recons/`, `stops/`); `m2m-prompts/` inside also holds the model-to-model briefs and live threads, including the pre-2026-09 ledger threads moved from the removed top-level `m2m_prompts/`. See its README for the convention. | reference |
| [`dashboard/`](dashboard/README.md) | Read-only web view of the soak: Rust (axum) API + TypeScript front-end. Reads `logs/events-*.jsonl`, `logs/status.json`, and OANDA REST. Cannot affect trading. | manual |
| `viz/` | Empty. | — |
| `src/autopilot/`, `src/research/` | **No source files on this branch** — only stale `__pycache__`. | — |

**Root-level Python** (entry points and ad-hoc tools):

| File | What it does |
|---|---|
| `run_oanda.py` | ⚠️ Launcher for the **currently-running** forex bot. |
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

## `src/` packages

| Package | Role | When |
|---|---|---|
| [`data/`](src/data/README.md) | Every vendor adapter, behind one interface. `DATA_SOURCE` picks which. Providers **return empty rather than raising**, so one dead symbol can't abort a fetch. | training + live |
| [`ml/`](src/ml/README.md) | The feature factory: bars → the numbers models see. Training and live run the *same* generators in the same order. | training + live |
| [`strategies/`](src/strategies/README.md) | The decision. Bars in, `Signal` or `None` out. No broker, no sizing. | live |
| [`execution/`](src/execution/README.md) | Brokers, orders, position state, software stops. Three orchestrators for three markets. | **live** |
| [`core/`](src/core/README.md) | Two unrelated things: shared types + the Discord notifier, **and** `retrainer/`, the entire training and promotion pipeline. | mixed |
| [`utils/`](src/utils/README.md) | Bar aggregation. One live module. | live |
| [`analysis/`](src/analysis/README.md) | Offline diagnostics, run by hand. Targets the legacy Alpaca stack. | never live |
| [`lab/`](src/lab/README.md) | The feature lab: candidate features scored by the retrainer's own promotion gate. Built 2026-09-21; v2 (state-hashed cache, gate-list threading, `ablate`, estimator A/B) 2026-09-23. | never live |

---

# Part 2 — Domain glossary

## The two-stage model

**Angel** — stage one. Tuned for *recall*: catch as many real opportunities as
possible, tolerating false alarms. It **proposes**. Fires above the Angel
threshold — calibrated per retrain from out-of-fold scores since 2026-08-29
(`ANGEL_THRESHOLD` env var pins the old fixed bar, default 0.40).
*(`src/core/retrainer/`)*

**Devil** — stage two. Tuned for *precision*, and trained only on the bars the
Angel already liked: of these candidates, which are actually worth taking. It
**vetoes**. Two stages exist because one model tuned for both jobs does neither
well. *(`src/core/retrainer/`)*

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
**the forex profile overrides this to 2.0× / 4.0×** (2:1) — it ran 1.0× / 2.0×
until 2026-08-08. Always check the profile rather than the module constants.

**MAE / MFE (maximum adverse / favourable excursion)** — how far a trade ran
against you, and how far in your favour, over a fixed forward window, expressed
as a multiple of the entry bar's ATR. The training targets of the barrier
models. *(`src/ml/barriers/labels.py`)*

**learned barrier geometry / quantile barrier** — a per-bar replacement for the
bracket's static multiples, learned as conditional quantiles of the forward
excursion: the stop is `Q_MAE(0.95)`, the 95th-percentile adverse walk a trade
tolerates, and the target `Q_MFE(0.50)`, the median favourable walk. Both are
NATR multiples, so they substitute for `sl_atr_multiplier` /
`tp_atr_multiplier` and everything downstream — the gates, rounding, sizing —
is unchanged. Travels on `Signal.metadata["barrier_geometry"]`
(`BARRIER_GEOMETRY_KEY`) and is **OFF by default**, and an artifact that records
a FAILED **promotion verdict** in its `barriers_meta.json` is refused outright at
boot rather than merely switched off
(`BARRIER_GEOMETRY_ENABLED=1` to serve it). Two gates stand in front of it: the
switch, and the Phase 1 promotion verdict in `scripts/evaluate_barriers.py`,
which as of 2026-09-14 **fails** (fold 3 coverage 0.905 against a 0.93 floor)
even though the learned stop beats the static constant on pinball loss on every
fold. *(`src/ml/barriers/`)* — see GLOSSARY-worthy caveat in
`src/execution/README.md`: the OANDA path trades fixed 1000 units and does not
size by risk, so a wider learned stop is a proportionally larger loss per
stop-out.

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
and the stop distance, not set as a fixed quantity. ⚠️ **Not enforced on the
live forex path** (verified 2026-09-09): `run_oanda.py` drives the OANDA
orchestrator with a fixed `units_per_trade = 1000`
(`oanda_forex_orchestrator.py:28-29`), and `RiskManager.calculate_quantity`
(where the \$50 floor lives) is never called by it. The equity-derived sizing
and the notional floor applied to the deleted Alpaca/Factory paths (2026-09-16)
only; git history has them.

### Lane 4 research terms (index-option variance premium, 2026-09-24)

**variance premium** — the empirical tendency of an index option's *implied*
volatility (what the premium prices in) to exceed the *realised* volatility the
underlying later delivers. Selling richly-implied volatility and hedging (or
defining the risk away) harvests that structural risk-transfer spread. The
canonical short-volatility edge; unlike bracket trading it profits from a
premium, not from predicting direction. *(Lane 4 brief §2)*

**IVR (implied-volatility rank)** — the percentile rank of today's ATM implied
volatility within its own trailing 252-trading-day history, 0–100. A short premium
entry trigger wants IVR elevated (sold when vol is rich). Computed as the IV of
the option whose |delta| is closest to 0.50 at 30–45 DTE, per day. *(Lane 4 brief §5.1)*

**iron condor** — sell one out-of-the-money put spread *and* one OTM call spread,
same expiry. The short wings sit at Δ∈[0.20, 0.30] in this lane. Wins when the
underlying pins the range between the wings; defined-risk by construction.

**credit spread** — sell one option, buy a further-out same-type option to cap the
loss; you collect (credit) more premium than you spend on the hedge. A *put credit
spread* is the long-vol-directional-free expression this lane was to use on the low side.

**defined-risk** — the maximum loss is fixed at entry (spread width − credit) rather
than open-ended, so sizing is exact and no stop-out surprise exists. Required for any
short-vol structure this repo would run.

**inverted chop veto** — the repo-integration mandate reusing
`_compute_chop_veto_mask` (`src/core/retrainer/_labels.py:163`) *backwards*: a
rangebound bar that the **directional** strategy vetoes is exactly the bar a **short-vol**
strategy wants, so `chop_friendly = chop_veto_mask` becomes a positive entry requirement
(IVR ∈ [25, 30]). Not a new filter — the same mask, read as "favourable" instead of "drop".

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

**feature** — one input column the model sees. As of the 2026-08-29 trim the
served pair is **17 columns** (`BASE_FEATURE_COLS`; the Devil gets 18, +`angel_prob`).
The 22/23-column count from earlier generations was cut to 17 when 5 features
were dropped in the trim retrain — read `retrainer.BASE_FEATURE_COLS`, not this
paragraph. *(`src/ml/features/v3_features.py`)*

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
checked against what it actually earned. *(`src/core/retrainer/`)*

**artifact holdout gate** — the additional pass/fail applied to the final model
on the holdout, using the same Brier/EV bars as the fold gate, the frozen
production threshold, and — for profit factor — the Clopper-Pearson **lower
confidence bound** on the macro win rate rather than the point estimate
(a PF on 55-82 trades flips with the clock; the bound is the fix). When the
fold gate fails, the holdout is still scored for diagnostics (Fold 3 models,
`ValidationReport.holdout.diagnostic_only`); the fold verdict stands either
way. Disabled by `RETRAIN_HOLDOUT_FRAC=0`; when disabled or empty the metadata
records the bypass explicitly. *(`src/core/retrainer/`)*

**Clopper-Pearson bound** — the one-sided lower confidence bound on a binary
win rate (`Beta(1-confidence; wins, losses+1)` quantile), mapped through
the profit-factor formula for the holdout gate. Chosen over the Wilson
approximation because Wilson under-covers below ~40 trades — precisely the
sample sizes that used to flip the verdict. Exact for independent trades;
the 45-bar macro walks overlap in price, so in practice the bound is
conservative rather than a literal coverage guarantee — the safe direction
for a promotion gate. *(`src/core/retrainer/`)*

**unresolvable tail** — the last `max_hold` bars per symbol of a raw slice,
whose bracket walk runs off the end of the frame and resolves "timeout →
loss" no matter what the price actually did. Those labels are systematically
wrong, so the engineered remainder and holdout each drop them after
engineering (the **boundary purge**); the cutoffs are derived from the raw
series because the walk needs the contiguous pre-veto path. *(`src/core/retrainer/`)*

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

## The feature lab

**feature lab** — `src/lab/`, the offline harness for asking whether a candidate
feature set would *promote*. It composes the `ml` generators, the retrainer's
labels and gate, and the strategy backtester; nothing in `src/execution/` or
`run_oanda.py` imports it. *(`src/lab/README.md`)*

**FeatureSpec** — one frozen, hashable object describing a lab experiment: bars,
feature families, bracket geometry, label knobs, cost-table switch, estimator
family. Frozen so a change is a new spec, never a mutation.

**content hash** — the spec's SHA-256 (16 hex chars), used as the frame-cache
key. It covers the spread table's bytes, the frame-affecting environment, every
registered family's (name, version) pair, and the RESOLVED state of any extra
generator — a stale cache hit is impossible by construction. It deliberately
EXCLUDES `gate.model_family`/`n_folds` (run provenance: they change the
estimator, not the frame), which is what lets the W4 estimator A/B reuse one
cached frame. *(`src/lab/spec.py`)*

**family version** — the required `version: int` (>= 1) every
`register_family`/`register_feature` call must pass. Folded into the content
hash: bump it whenever anything inside the family's generators changes frame
contents (a lookback constant, a formula), or cached frames from the old
behaviour are silently reused. *(`src/lab/registry.py`)*

**generator state** — an extra (unregistered) generator's resolved
`__dict__`, hashed alongside its class id into the content hash. Two instances
of one class with different constructor args hash differently; a generator
whose state is not JSON-serializable raises from `content_hash()` rather than
degrading to a class-name-only hash. *(`src/lab/spec.py`)*

**ablation (lab ablate)** — `lab.ablate`, the feature-INTERACTION question:
edge(full cocktail) − edge(cocktail − X) per registered family, with only
`feature_sets` varying across the N+1 arms (labels, veto, geometry, data and
cost table identical). Each delta carries a Clopper-Pearson interval on the
underlying win-rate difference; a delta consistent with zero on thin pooled
trades is never a drop decision. *(`src/lab/ablate.py`)*

**estimator A/B (MODEL_FAMILY arm)** — running one identical spec under two
estimator families via the retrainer's `MODEL_FAMILY` seam
(`core/retrainer/_common.py:410`): `MODEL_FAMILY=catboost python -m lab.cli
run --name <seed>`. The env-selected family is the run's arm (the CLI's
precedence: `--model-family` flag > env > the spec's declared family), the
frame is shared by construction (the frame hash excludes the estimator), and
the frame's feature-list contract is the gate's `GateResult` lists. Measured
and closed 2026-09-23: CatBoost scored worse than random on the same frame.

**feature family** — a registered name -> generators + model-facing columns. A
new candidate feature is one `BaseFeatureGenerator` class plus a registration;
nothing else in the repo changes.

**DSR / CSCV PBO / HLZ (deflated Sharpe, probability of backtest overfitting,
haircut Sharpe)** — the multiple-testing statistics in `src/lab/stats.py`,
shared across the 2026-09-24 quant-lane dispatches (Lanes 1–5) under one pinned
signature set so every lane reports them identically. `deflated_sharpe_ratio`
is the probability the observed Sharpe beats the expected max of N null trials
(Bailey & Lopez de Prado 2014, with the skew/kurt variance correction that
tightens it under negative-skew strategies like short-vol); `cscv_pbo` is the
combinatorially-symmetric CV PBO (Bailey et al. 2014; zero-skill matrix → 0.5
exactly, via a degenerate-tie branch); `hlz_haircut_sharpe` is the HLZ 2016
multiple-testing haircut, and `hlz_se` is the skew/kurt standard error behind
the "HLZ t > 3.0" gate. *(`src/lab/stats.py`)*

**edge over random** — the gate's telemetry: pooled fold win rate minus the
macro bracket's **base rate** (what a random long entry won on the same
tradeable bars). In *win-rate units*, not R. A positive PF is worth nothing
unless this is positive — and at the EV-maximising Angel bar the approved
population is thin enough that a large positive edge is usually small-sample
noise. Read it beside `pooled_oos_trades`. *(`src/core/retrainer/_gate.py`)*

**base rate** — the fraction of random long entries that would win under the
same bracket. 0.254 on the M15 fiat basket over the cached 730-day window.

**served-artifact replay** — `lab.artifact`, the lab's second baseline question:
*what would the model the bot currently runs have done on this frame?* It loads
the `OANDA_MODEL_DIR` pair, pins its own `threshold.json` bars, and replays it
through the same live-gated backtester as a candidate — no gate, no retrain, no
PASS/FAIL. The report splits the frame at the artifact's recorded holdout window
(the only rows it never saw); everything else is in-sample. *(`src/lab/artifact.py`)*

**frame** — in the lab, the engineered + labelled + vetoed table the gate and
backtest both consume (`FrameResult.df`). Building it and scoring it must use
the same rows; the parity test pins that `feature_sets=("v3_base",)` reproduces
`engineer_features_and_labels` row for row, plus the production tail purge.
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

**candidate** — one evaluation configuration in the behavior matrix: a name,
the bracket multiples (`sl_mult` / `tp_mult`) and a training lookback. Not new
strategy code — every candidate runs the same strategy with different settings.

*(Corrected 2026-09-02: this entry previously described a model directory,
threshold pair and chop-gate config, none of which `behavior_matrix.Candidate`
has ever carried.)*

## The strategy library and the router

**lazy strategy registry** — `STRATEGIES` in
`src/strategies/concrete_strategies/__init__.py`. Maps a config name to a
strategy class, but resolves on lookup rather than at import: listing the names
loads nothing, so the live bot's import graph contains only the strategy it
actually serves. An eagerly-importing registry meant a typo in an unused
research strategy could stop the bot booting.

**strategy library** — the set of ordinary, non-ML strategies in
`src/strategies/concrete_strategies/`: moving-average cross, RSI mean
reversion, Bollinger breakout, Donchian breakout, momentum. Each is bars in,
a trade or a decline out. They exist to be compared *against each other per
behavior tag*. ⚠️ As of 2026-09-02 **none has a measured edge** — all five
scored negative net expectancy on GBP_JPY M15.

**regime router** — the "master" strategy (`regime_router.py`). Tags the
current bar's behavior, looks the label up in a routing table, and delegates to
the named strategy or declines. It is a third meta stage above Angel/Devil:
Angel proposes, Devil vetoes, the router decides *which proposer to listen to*.

**routing table** — the router's whole intelligence: a JSON map from behavior
tag to strategy name, or to `null`. ⚠️ The tables in `config/` are
**hand-authored templates** (`*.example.json`) with no empirical basis; only
`build_strategy_matrix.py` produces a real one, and `run_oanda.py` refuses to
run the router without an explicit `--routing-config`.

**stand down** — the router returning `None`: no trade, deliberately. Three
causes, all intended: the trailing window is `cold`, the tag maps to `null`, or
the named strategy is unknown. Standing down is the router's most important
output — one that always picks something will trade into regimes where nothing
has an edge.

**R** — a trade's result expressed in multiples of its own stop distance. A
2:1 bracket that reaches its target pays +2R; one that stops out pays −1R. The
unit the whole repo reports expectancy in, because it is comparable across
instruments and price levels.

**realised R** — R computed from the price actually filled at, rather than from
the bracket's nominal payoff. Matters at the two edges: a trade that **timed
out** pays only the small distance it actually moved (booking it at ±full
payoff on the sign of the move inflated win rate by 10–12 points until
2026-09-02), and a trade that **gapped** through its stop loses more than 1R,
which is a real loss the nominal convention hid.

**gap fill** — an exit where the bar *opened* beyond the bracket level rather
than trading through it. The fill is the open, not the level; booking the level
credits a price that never traded. Recorded as `sl_gap` / `tp_gap`.

**gate funnel** — the count of signals a strategy proposed versus the count the
live gates actually admitted, per gate. A result in its own right: a regime
whose picks are mostly gate-vetoed is not reachable live, however good the
surviving trades look.

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
