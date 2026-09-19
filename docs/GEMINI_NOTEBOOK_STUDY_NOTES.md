# Build-A-Bot — Architecture Study Notes

**Purpose:** a structured, sectioned reference intended to be pasted into Gemini
Notebook (or any long-context study tool) so the architecture can be interrogated
conversationally. Every section is self-contained and clearly labelled. Facts here
were read from the working tree on **2026-09-18** (branch `feat/quantile-mae-barriers`,
HEAD `b8fe4c9`), not from older docs.

> **Read this first — the repo's own docs are partly stale.** The root `readme.md`
> describes "Universal Scalper V3.4" on Alpaca; that lane is gone. `table-o-content.md`
> ("Haynes manual", 2026-07-04) is a good narrative but references three files that
> no longer exist. `GLOSSARY.md` is current. When this document and an older doc
> disagree, this one reflects the code as of the date above.

---

## 1. What this system is, in one page

Two independent trading products share one repository, one ML feature factory, and
one Discord notifier:

| | **V5 OANDA Forex Bot** | **V4 Equities Investor** |
|---|---|---|
| Broker | OANDA v20 (practice/live) | Alpaca (paper) |
| Asset | FX crosses + metals | 96 US large-caps, all 11 GICS sectors |
| Cadence | Continuous intraday (M1/M5/M15) | Monthly rebalance (1st of month, cron) |
| Process | Long-running async daemon | One-shot synchronous script |
| Model | LightGBM Angel/Devil meta-labeling | LightGBM LambdaRank |
| Entry point | `run_oanda.py` | `scripts/portfolio_orchestrator.py` |
| Model dir | `models/forex_m15_wide/` (served) | `models/v4_investor_lgbm.txt` |
| State | **currently soaking on practice** | paper, monthly cron |

They are architecturally isolated — different broker SDKs, model files, event loops,
rate limits. Running both in parallel is safe. The only shared runtime surface is the
Discord webhook.

**The one-sentence mechanism of the forex bot:** every sealed bar, two LightGBM models
(Angel proposes, Devil vetoes) decide whether to trade; a `RiskManager` then vetoes
trades whose cost is too high relative to the move (the "chop filter"); the trade is
submitted to OANDA; and a tick-by-tick software watchdog — not the broker — enforces
the stop and target. **A dead process means an unwatched open position**, and most of
the engineering hardening follows from that single fact.

**The central empirical finding (2026-09-14, still the standing conclusion):** the bot
is *not* misconfigured. Its measured edge is roughly **+0.045R**, while any bracket
geometry needs roughly **+0.09R** to overcome the spread toll. No configuration of this
basket/timeframe/model is promotable, and the promotion gate refusing new models is
correct. Widening stops *dilutes* the cost; it does not create edge. See §13.

---

## 2. Repository map

```
build-A-bot/
├── run_oanda.py                 # ⚠️ LAUNCHER FOR THE LIVE BOT
├── run_soak.sh                  # sets PYTHONPATH, sources .env, execs run_oanda.py
├── soak.service                 # systemd USER unit the soak runs as
├── soak_watchdog.sh             # cron every 5 min: relaunch if dead
├── run_pipeline.sh              # offline loop: harvest→replay→grade→drift→retrain
├── trading_mcp.py               # read-only MCP server + 2 guarded control tools
├── GLOSSARY.md                  # the dictionary (current)
├── table-o-content.md           # the "Haynes manual" tour (partly stale)
├── src/                         # all library code
│   ├── data/                    # vendor adapters behind one interface
│   ├── ml/                      # feature factory: bars → model inputs
│   ├── strategies/              # bars in → Signal|None out
│   ├── execution/               # brokers, orders, software stops (LIVE)
│   ├── core/                    # shared types + retrainer/ (training pipeline)
│   ├── utils/                   # bar aggregation (legacy Alpaca path only)
│   └── analysis/                # offline diagnostics, run by hand
├── scripts/                     # hand-run tools + the whole V4 investor
├── tests/                       # 538 tests, no network/broker/real models
├── models/                      # trained artifacts (gitignored)
├── data/                        # bars, ledgers, processed datasets (gitignored)
├── config/                      # spread alpha tables, example routing tables
├── logs/                        # soak logs, events-*.jsonl, status.json
├── dashboard/                   # Rust (axum) read-only API + TS frontend
└── llm_reports/                 # audits / refactors / recons / handoffs / stops
```

**Import convention (a real trap):** entry points prepend `src/` to `sys.path`, so
modules import as `data.factory`, not `src.data.factory`. The offline pipeline tends to
use the `src.` prefix; live code does not. Both work depending on `PYTHONPATH`. Run tests
with `PYTHONPATH=src:.`.

---

## 3. Runtime data flow

```
provider → LiveBarAggregator* → FeaturePipeline → MLStrategy (Angel→Devil)
                                                        ↓ Signal
                                          RiskManager (bracket + chop veto)
                                                        ↓
                                              Orchestrator → broker
                                                        ↓
                                          tick watchdog (software SL/TP)
```
\* OANDA seals its own bars; `LiveBarAggregator` only serves the legacy Alpaca lane.

At the file level (forex path):

1. `data/oanda_provider.py::OandaMarketProvider` — REST history + streaming quotes,
   builds bars from ticks itself.
2. `ml/feature_pipeline.py::FeaturePipeline` — runs generator classes in order.
3. `strategies/concrete_strategies/ml_strategy.py::MLStrategy` — Angel/Devil.
4. `strategies/base.py::Signal` — direction, entry price, raw stop distance, metadata.
5. `execution/risk_manager.py::RiskManager` — turn distance into bracket, or veto.
6. `execution/oanda_forex_orchestrator.py::OandaForexOrchestrator` — owns the loop.
7. `execution/oanda_order_manager.py::OandaOrderManager` — net-position order truth.

**The symmetry rule that governs everything:** training and live inference must run the
*same* feature generators, in the *same* order, with the *same* cleaning. A feature
computed even slightly differently in the two places produces a model that validates
well and loses money, silently, with no error message. This is why the pipeline is a
shared import rather than two implementations, and why the chop veto is mirrored in the
retrainer (`_compute_chop_veto_mask`) rather than only living in execution.

---

## 4. The data layer (`src/data/`)

Three layers:

1. **Contracts** — `market_provider.py` (`MarketDataProvider`), `fundamentals.py`,
   `macro.py`, `enums.py`, `timeframe.py`. Rule: **no vendor SDK imports**.
2. **Adapters** — `oanda_provider.py`, `alpaca_provider.py`, `polygon_provider.py`,
   `yahoo_provider.py`, and `providers/` (SimFin, yfinance, composite).
3. **Factory** — `factory.py`, reads env vars and returns the right adapter.

**Behavioral rule to internalize: providers return empty, they do not raise.** A dead
symbol or vendor outage yields an empty DataFrame, so one bad instrument cannot abort a
fetch of forty.

**Canonical bar shape:** six columns — `timestamp` (microsecond UTC), `open`, `high`,
`low`, `close`, `volume`, all Float64.

### OANDA provider (`oanda_provider.py`) — the live forex feed
- OANDA streams individual *quotes*, not bars, so this module builds bars from ticks.
- Hardening that keeps the soak alive: 20s read-inactivity timeout (OANDA heartbeats
  every ~5s), `seconds_since_last_message` liveness property, `force_disconnect()`.
- Exposes a raw `tick_callback` used to measure live spreads — runs inline on the stream
  thread and **must not block**.
- `get_tradeable_instruments()` — used at boot to drop symbols the account can't trade;
  returns empty set on failure meaning "unknown", never "nothing tradeable".
- ⚠️ `volume` is **tick count, not traded size**; OANDA doesn't report real volume.
  Training uses the same proxy, so the two agree.

### Factory dispatch
- `get_market_provider()` reads `DATA_SOURCE`: `alpaca` (default), `polygon`,
  `yahoo` (paper only), `oanda` (forex).
- `get_fundamental_provider()` reads `FUNDAMENTAL_SOURCES`: an ordered chain —
  first non-empty result wins. `simfin` (needs `SIMFIN_API_KEY`), `yfinance`, `none`.
- Unknown values **raise** rather than defaulting, so a typo can't trade the wrong market.

### Fundamentals/macro providers (`src/data/providers/`)
Only the equities investor reads these. `CompositeFundamentalProvider` chains sources,
resolving first-non-empty **per method call**. `SimFinFundamentalProvider` bulk-downloads
`general`/`banks`/`insurance` partitions and renames columns to Yahoo shapes.
`yf_macro.py` has two documented, uncorrected Yahoo traps: `10Y_YIELD` (`^TNX`) is
reported **×10** (4.2% → 42.0), and `2Y_YIELD` (`^IRX`) is actually the **13-week
T-bill**, misnamed.

---

## 5. The feature layer (`src/ml/`)

This folder turns bars into numbers. It does not decide trades and does not talk to a
broker. The rule: **same generators, same order, same cleaning in training and live.**

Assembly:
```
bars → [V3BaseFeatures, V3SessionFeatures, V3CostFeatures?, V3HTFFeatures]
     → (target generator, training only) → clean_data → model-ready frame
```

### `features/v3_features.py` — the vocabulary of the system
| Class | Adds | Why |
|---|---|---|
| `V3BaseFeatures` | 14 single-bar columns: momentum, volatility, position-in-range, candle shape | Core view of one bar; computed **per symbol** so instruments don't leak |
| `V3SessionFeatures` | `session_asia/london/ny/overlap` 0/1 flags from UTC hour | Same indicator means different things at 03:00 vs 14:00 |
| `V3CostFeatures` | `cost_ratio` | Lets the model *see* trading cost instead of it being silently filtered behind it; no-op without a cost table |
| `V3HTFFeatures` | `htf_rsi_14`, `htf_trend_agreement`, `htf_vol_rel`, `htf_bb_pct_b` | What the slower chart says, joined onto each fast bar |

**The lookahead guard (`available_at`)** is the subtlest correctness rule in the repo: a
5-minute bar stamped 12:00 is not finished until 12:05, so using it at 12:01 reads the
future. Each slow bar's timestamp is pushed forward one full timeframe before joining.
Break it and backtests look brilliant while live trading fails.

### `feature_pipeline.py`
Runs ordered generators, then `clean_data`, which converts both NaN **and infinity** to
null before dropping incomplete rows. Infinity is the non-obvious half — it arises when a
perfectly flat bar makes a normalising denominator zero.

### `feature_stats.py` — "is it broken or just quiet?"
Computes training-time feature distributions and compares live distributions later,
**without retraining**. The load-bearing idea is **null calibration**: textbook PSI
cutoffs (0.10/0.25) assume independent samples, but market bars are autocorrelated, so
short windows score high PSI even when nothing is wrong. This module measures what PSI
*ordinary training windows* produce and only calls drift when live PSI beats that null's
upper tail. Stats stored pooled and per-symbol.

### `barriers/` — learned bracket geometry (experimental, off by default)
Quantile regressions that predict a per-bar stop and target instead of the static
`2.0×/4.0×` NATR bracket:
- stop = `Q_MAE(0.95)` — the 95th-percentile adverse walk a trade tolerates
- target = `Q_MFE(0.50)` — the median favourable walk
- `labels.compute_excursions` computes MAE/MFE per symbol; null for incomplete windows
  (including windows containing a gap bar).
- Backends: CatBoost (default; quantile loss accepts `monotone_constraints`), LightGBM
  (rejects them, so monotonicity is enforced by an audit), and a binned fallback.
- Persistence contract: `barriers_mae.pkl` + `barriers_mfe.pkl` + `barriers_meta.json`,
  **pickles first, meta last** (so a reader seeing a new meta sees a complete pair).
- **Serving gated twice:** `BARRIER_GEOMETRY_ENABLED=1` on the strategy side, and the
  artifact must not record a FAILED promotion verdict. The Phase 1 gate
  (`scripts/evaluate_barriers.py`) currently **FAILS** (fold 3 MAE coverage 0.905 vs a
  0.93 floor), so no serving artifact exists yet.
- **Measured conclusion:** the learned bracket beats the static constant on pinball loss
  on every fold, but a *constant* wide bracket captures all of the net gain — the value
  is bracket *width* (cost dilution), not the learning. See §13.

### Subpackages
- `core/interfaces.py` — three ABCs: `BaseFeatureGenerator`, `BaseTargetGenerator`,
  `BaseTrainer`. `predict_proba` returning probabilities (not labels) is what makes
  tunable thresholds possible.
- `targets/v3_targets.py` — ⚠️ legacy; only `feature_pipeline.main()` uses it.
  Production labels live in `core/retrainer/_labels.py`.
- `trainers/v3_rf_trainer.py` — ⚠️ misleading name; production models are LightGBM.
  `.load()` unpickles whatever is on disk, so the live strategy constructs a
  `V3RandomForestTrainer` and loads a LightGBM model into it.
- `regimes/hmm_regime.py` — experimental 3-state Gaussian HMM per symbol on
  `(log_return, natr_14)`, off unless `RETRAIN_USE_HMM=1`. States are unnamed.
- `regimes/behavior_tagger.py` — deterministic labeller (`trend_high`, `range_low`, …),
  causal/trailing-window only, for **analysis**, never fed to the model.

---

## 6. The decision layer (`src/strategies/`)

A strategy takes a DataFrame and returns a `Signal` or `None`. It holds no broker
connection, places no orders, knows nothing about sizing. `None` is the normal return.

### `base.py`
`BaseStrategy` + `Signal`. **This is the only `Signal` class** as of 2026-09-16 (the
Alpaca path's `core.signal.Signal` was deleted).
- `direction` — `"long"`/`"short"` (plain string)
- `entry_price` — latest sealed bar's close, not the fill
- `raw_sl_distance` — a **price distance**, not a level/percentage
- `raw_tp_distance` — ⚠️ **written but never read**; `RiskManager` owns target sizing
- `metadata` — diagnostics, plus one real contract key: `BARRIER_GEOMETRY_KEY`

`BARRIER_GEOMETRY_KEY = "barrier_geometry"` lives here, beside the Signal, because the
producer (strategy), consumer (RiskManager, which must stay numpy-only) and offline
replay (backtester) all need the same string.

### `ml_strategy.py::MLStrategy` — the brain
The two-stage Angel/Devil decision-maker. Design constraints:
- Feature pipeline **imported** from `src/ml`, never reimplemented; generator order
  matches the retrainer.
- Feature list read from the model's own `feature_names_in_`, not hardcoded.
- Devil threshold **overwritten** from the model's `threshold.json`.
- Constructor **raises at boot** if the model expects `cost_ratio` or regime features
  whose sidecars are missing.

Runtime behaviors:
- **Hot reload** — compares model file mtimes each bar; a retrain lands without restart.
- **Stale-bar guard** — if cleaning dropped the *newest* bar, return `None` rather than
  score an older bar against the current price.
- **Heartbeat** — every 15 bars logs recent Angel probabilities.
- **Bracket validation** — `_validate_metadata` raises if the model's stored
  `sl_atr_multiplier`/`tp_atr_multiplier` disagree with the live `RiskProfile`.

### The strategy library + router
Five rule-based strategies (`sma_crossover`, `rsi_mean_reversion`,
`bollinger_breakout`, `donchian_breakout`, `momentum`) plus `regime_router`.
- ⚠️ **None has a measured edge** — all five scored negative net expectancy on GBP_JPY
  M15. `run_oanda.py` logs a warning if you select one.
- `regime_router.py` tags each bar's behavior and looks it up in a routing table.
  Three stand-down paths: cold window, label maps to `null`, unknown strategy.
- The tables in `config/*.example.json` are **hand-authored templates, not measurements**.
  `run_oanda.py` refuses `--strategy regime_router` without an explicit
  `--routing-config`. Only `analysis/build_strategy_matrix.py` produces a real one.
- `__init__.py::STRATEGIES` is a **lazy registry** (a `Mapping`, resolves on lookup).
  `MLStrategy` is eager; selecting one strategy must not import the others, or a typo in
  an unused research strategy would stop the live bot booting.

---

## 7. Risk & the chop filter (`src/execution/risk_manager.py`)

`RiskProfile` (tunable numbers) + `RiskManager` (bracket sizing, position sizing, veto).
Applied **after** a strategy already wants to trade. The veto is the important part: a
model can be right about direction and still lose money if trading cost exceeds the move.

### Bracket sizing
Forex profile overrides the module defaults. **Always read the profile, not the
constants.** The forex pair was `1.0×/2.0×` until 2026-08-08, then doubled to
`2.0×/4.0×` (a 2:1 payoff) specifically to dilute the spread toll.
- `min_sl_pips = 2.0` (FX stop floor), `min_sl_pct_metals = 0.0001` (gold/silver floor,
  because a 0.0001 pip is meaningless on a price near $2,700)
- `round_precision = 5` for forex (quotes to 5 decimals)

### The three gates (collectively the "chop filter")
| Gate | Name | Rejects when |
|---|---|---|
| **A** | cost (`GATE_SPREAD`) | `sl_dist < k_eff × spread` — the stop is too tight relative to the toll |
| **B** | regime (`GATE_REGIME`) | current volatility is in the bottom 20% of its own 260-bar window |
| **C** | time (`GATE_TIME`) | timestamp is in the daily rollover blackout (default 16:55–17:30 **New York**) |

- **Gate C is checked first and is anchored to America/New_York**, so it tracks daylight
  saving instead of drifting an hour twice a year. Spreads blow out ~10× at the rollover.
- **`spread_k_base = 3.0` for forex** — algebraically a **toll cap**: it admits a trade
  only when the spread eats at most `1/k` (33%) of the stop. Was 1.5 (67%) until
  2026-08-08. `spread_k_coupling` can scale `k_eff` with volatility.
- **Cold start** — with fewer than 60 bars of volatility history, Gate B stands down, so
  a just-restarted bot isn't frozen by a half-filled buffer.
- **Kill switch** — `ATR_KILL_SWITCH_THRESHOLD` (0.5204 NATR) skips bars judged too
  violent, regardless of the models.
- Each gate is toggleable via `RISK_*_GATE_ENABLED`; master kill is
  `RISK_CHOP_FILTER_ENABLED`.
- `last_veto_gate` records which gate fired, so soak logs answer "why didn't it trade?"
- **Symmetry contract:** gate logic mirrors `retrainer._compute_chop_veto_mask`.
  Change one side without the other and the model learns from setups it will never see.

### `scheduled_market_pause`
Returns `PAUSE_WEEKEND` / `PAUSE_DAILY_ROLLOVER` / `None`, anchored to New York. Exists
because forex stops ticking at the daily rollover and over the weekend, and the liveness
watchdog used to read that legitimate silence as an outage — producing 12,674 CRITICALs
in one weekend and, worse, flattening positions held across the rollover. The watchdog
now returns early during a scheduled pause. Holidays are deliberately **not** covered (a
wrong calendar is worse than none).

### Learned barrier substitution (2026-09-14)
`calculate_bracket` takes an optional `barrier` payload. When present and usable, the
per-bar NATR multiples **replace** the profile multipliers (they do not compound). Gates,
rounding and sizing all still see a distance and cannot tell its origin.
`last_geometry_source` records `"static"` or `"barrier"`. No clamp on stop width (a
learned quantile can legitimately be ~10 ATR); widening beyond 2× the static width logs
a WARNING because the OANDA path trades *fixed* 1,000 units and a wider stop is a
proportionally larger loss.

---

## 8. The live orchestrator (`src/execution/oanda_forex_orchestrator.py`, ~2,500 lines)

⚠️ **This is the currently-running bot.** Two clocks run at once:
- **Slow path** — a bar seals → features → model → maybe a trade. Every 15 min in the soak.
- **Fast path** — every incoming quote checked against the open position's SL/TP.
  Runs on the provider's stream thread and **must return in <50 µs with no blocking I/O**.

### Key methods
| Method | Role |
|---|---|
| `run()` | Signal handlers, boot reconcile, subscribe, prime history, launch stream + liveness, block on shutdown |
| `_on_tick` | **Hot path.** Captures spread, checks SL/TP against bid/ask, dispatches close off-thread |
| `_on_bar` | **Cold path.** Seam dedup, buffer append, regime/ATR update, spread sampling, `generate_signals` |
| `_evaluate_and_trade` | Applies entry guards, computes bracket, submits order, records state |
| `_watchdog_close` | Closes a breached position with exponential-backoff retry; on total failure parks it as `CLOSE_FAILED` and fires a manual-intervention Discord alert |
| `_reconcile_on_boot` | Adopts reality before trading; any position found at boot is an orphan and flattened; failed sync aborts startup |
| `_prime_history` | Fills buffers from REST (last ~5 days) and seeds regime/ATR state |
| `_check_stream_liveness` | 60s of silence → pause-aware (see above) else reconnect + flatten |
| `_catch_up_missed_bars` / `_backfill_seam_bar` | Score bars that sealed during an outage; before these existed ~8% of bars were lost |
| `_reconcile_unverified_entries` | Settles `ENTRY_UNRECONCILED` positions on the liveness loop; never resolved by assumption |

### Position state machine
`FLAT → PENDING → IN_TRADE → PENDING_EXIT → COOLING → FLAT`
(Orchestrator states: `OPEN`, `PENDING_CLOSE`, `REVERSING`, `CLOSE_FAILED`,
`ENTRY_UNRECONCILED`.)

### Entry guards (added 2026-07-30, born from one morning's trades)
- **Post-exit cooldown** — no re-entry for one bar period after a close
  (`OANDA_REENTRY_COOLDOWN_SECONDS`). Blocks *opening* only; flipping stays allowed.
- **Correlated-exposure cap** — max positions sharing the same **signed currency leg**
  (`OANDA_MAX_PER_CURRENCY`, default 2). Long GBP_JPY + long AUD_JPY are one short-yen
  bet; on 2026-07-30 a third joined and all three lost together.
- **Reservations (`_pending_entries`)** — in-flight entries counted as held, so two
  signals on one bar can't both pass a cap they jointly breach.

### Other hardening
- **`_drop_untradeable_symbols`** — drops configured symbols the account can't trade
  (XAU/XAG on this account). Deliberately fail-open: only a successful lookup may drop
  anything.
- **Reconnect backoff** — jittered exponential 5s–60s, reset after 120s healthy. The cap
  sits below the 60s flatten threshold.
- **Spread calibration (`SPREAD_CALIB`)** — samples real spread once per sealed bar, off
  the fast path; `scripts/bake_spread_alphas.py` consumes it.
- **Fixed size** — `units_per_trade = 1000`. This path does **not** use equity-based
  sizing; `RiskManager.calculate_quantity` is never called by it.

### `oanda_order_manager.py`
Tracks the **net signed position** per instrument, never lots, because US (NFA) rules
require FIFO closing and forbid opposing positions. One signed number makes those
impossible to violate by construction. **A failed close leaves local state untouched**
(believing you're flat while holding is far worse). `submit_target_position` expresses
orders as "end at N units", and retries an ambiguous failure only after re-reading the
broker position — the re-sync, not the phrasing, makes the retry safe.

---

## 9. The training pipeline — "The Cure V2" (`src/core/retrainer/`)

A package since 2026-09-16 (was a 4,366-line file). Submodules: `_common` (config),
`_types` (dataclasses), `_data` (fetch + holdout carve), `_labels` (targets), `_features`
(engineering), `_train` (fitting), `_thresholds` (search), `_gate` (validation),
`_persist` (promotion + atomic writes), `_pipeline` (`main()`). `__init__.py` re-exports
every historical name. **Patch tests at the owning submodule, not the facade.**

Run as `python -m src.core.retrainer`. Exit codes: `0` promoted, `1` execution error,
`2` trained but **rejected** (a healthy outcome, not a crash).

### The shape of a run (`_pipeline.main()`)
1. **Phase 1** — resolve provider + asset config (`get_asset_config(DATA_SOURCE)`).
2. **Phase 2** — fetch history.
3. **Phase 2a** — **carve the holdout FIRST**, before any feature engineering. The
   chronologically last slice (~18%). Nothing may see across this boundary.
4. **Phase 2.5** — optional per-instrument spread-cost table.
5. **Phase 3** — engineer features/labels on the remainder only.
6. **Phase 3a** — **boundary purge**: drop the last `max_hold` bars per symbol, whose
   macro walk runs off the frame and reads "timeout → loss" regardless of truth.
7. **Phase 4** — `validate_candidate`: 3-fold expanding walk-forward, scored strictly OOS.
8. **Phase 4.5** — artifact holdout gate: score the served model on the untouched holdout.
9. **Phase 5** — `promote_or_reject`: only if **both** the fold gate and holdout gate pass.

### Angel/Devil, precisely
- **Angel** — stage one, tuned for **recall**: catches as many real opportunities as
  possible, tolerating false alarms. It **proposes**.
- **Devil** — stage two, tuned for **precision**, trained only on the bars the Angel
  already liked. It **vetoes**.
- This arrangement is called **meta-labeling**. The Devil gets `angel_prob` as an 18th
  feature.
- **Two Devil targets:** the **survival** target (trained on; avoid the stop for 5 bars)
  vs the **macro** target (45-bar bracket walk; used for EV/PF gate evaluation).
  ⚠️ `RETRAIN_DEVIL_LABEL="macro"` is the validated fix — the shipping survival-trained
  Devil is *anti-informative* about the served bracket (AUC 0.4722 vs 0.5839).

### Labels (`_labels.py`)
- **macro** — replay each bar forward up to `MAX_HOLD_BARS` (45); did target or stop come
  first? Stop checked first, so a bar touching both is a loss.
- **survival** — did price avoid the stop for the next `SURVIVAL_BARS` (5)?
- **chop-veto mask** — vectorised copy of the live veto, so the model learns only from
  bars the live bot would take.

### Gate thresholds
- Brier ≤ 0.30 (raised from 0.25; the survival label's ~45% base rate puts a do-nothing
  classifier near 0.25)
- EV ≥ 0.0005 (R)
- Profit Factor ≥ 1.2
- Pooled OOS trade floor: 30, scaled down by the chop-veto drop rate
- **Profit factor is gated on the Clopper-Pearson lower confidence bound**, not the point
  estimate — a PF on 55–82 trades flips with the clock, and the bound is the fix. CP over
  Wilson because Wilson under-covers below ~40 trades.

### Dynamic thresholds (`_thresholds.py`)
- **Angel bar** — since 2026-08-29 calibrated per refit from OOF score quantiles
  (`_find_optimal_angel_threshold`), maximizing EV subject to `MIN_ANGEL_PROPOSALS`
  (300). Pinned into `threshold.json`; the `ANGEL_THRESHOLD` env var (default 0.40)
  selects the old fixed mode.
- **Devil bar** — `_find_optimal_threshold` sweeps cutoffs to maximize EV subject to a
  minimum trade count.
- **Devil `min_child_samples`** auto-scales to a tenth of the Angel-approved population
  (capped, floored at 5), because a split needs ≥2× min_child rows and the Angel-side 80
  collapsed a Devil trained on dozens of rows into a constant.

### The holdout gate and edge-over-random
- **Artifact holdout gate** — the served model's own score on data it never saw.
  `diagnostic_only=True` when the fold gate already failed; the fold verdict stands.
- **EDGE OVER RANDOM** (added 2026-09-14) — `_macro_base_rate` computes what a random
  long entry would have won on the same bars under the same bracket. Every gate log
  prints it per fold and pooled. **It is telemetry, deliberately not gated on** — because
  a PF lower bound can be cleared by a zero-skill model in a high-base-rate regime, and a
  skilled model rejected in a low-base-rate one.
- **`RETRAIN_DAYS_BACK`** must be set to change lookback; the fold schedule derives from
  the module constant, so a longer fetch without the env var silently trains on the
  oldest 60 days and discards the rest.
- **`RETRAIN_MODEL_DIR`** redirects a run to a side directory without touching promoted
  models. `_is_production_model_dir` suppresses every "now live" Discord claim when set.

### Learned barriers ride along
With `RETRAIN_LEARN_BARRIERS=1` (default), a passing retrain also fits the two quantile
regressions and persists the barrier artifacts. ⚠️ **That hook fires off the Angel/Devil
gate, not the barrier gate**, so artifacts can exist whose own barrier gate failed.
`RETRAIN_BARRIER_VERDICT` is how the barrier evidence travels into `barriers_meta.json`.

---

## 10. Telemetry (`src/core/events.py`) and the dashboard

### `events.py`
The machine-readable half of the bot's output: `emit()` appends one JSON object per line
to `logs/events-YYYY-MM-DD.jsonl`; `write_status()` replaces `logs/status.json`
atomically. Three load-bearing properties, because it runs **inside the live process**:
1. never raises (every entry point swallows its own exceptions)
2. never blocks (bounded queue + daemon writer; a full queue **drops** events)
3. never called from the tick path (`tests/test_events.py` checks this at source level)

Opt-in: nothing is written until `configure()` is called; `EVENTS_ENABLED=0` disables.

### Event kinds
| `ev` | Key fields |
|---|---|
| `boot` | pid, symbols, granularity, units, cooldown_s, max_per_ccy, thresholds |
| `bar` | sym, bar_ts, close, angel, devil, outcome, proposed — **every evaluation** |
| `entry` | sym, dir, units, entry, sl, tp, angel, devil |
| `exit` | sym, units, dir, entry, sl, tp, reason, attempt (no exit price) |
| `gate_veto` | sym, gate, spread, regime, time, devil_approved |
| `guard_block` | sym, guard (`cooldown`/`exposure`), detail, remaining_s, total |
| `calib` | sym, n, alpha_emp, med_spread_pct, med_baseline_natr |
| `stream` | kind (`disconnect`/`seam_catchup`/`seam_backfill`), sym, delay_s/age_s |
| `heartbeat` | sym, median, p75, max, proposed, n_bars, threshold |

Volume is low — roughly 800 `bar` events/day across eight instruments.

### `dashboard/`
Rust (axum) API + TypeScript/Vite frontend. **Read-only; cannot affect trading.** Three
data sources: `status.json` (is it alive/what it holds), `events-*.jsonl` (what it's
doing), **OANDA REST** (the only authority on fills and realized P&L — the bot's log
knows what it *asked* for, not what it *got*). Working routes: `/api/health`,
`/api/status`, `/api/events`. To build: read-only OANDA client, derive, more routes, SSE.

### `log_filters.py`
`TruncatingFilter` caps over-long records and flattens to one line. Exists because OANDA
sits behind Cloudflare: during weekend maintenance the API answers 502/520 with a ~96 KB
styled HTML page, and a 2026-08-16 soak wrote **447 MB across 909k lines**, 99% of it
Cloudflare markup. Attached to the root *handler* so third-party loggers are covered.

---

## 11. The V4 Equities Investor (`scripts/`)

A separate product: a **monthly** cross-sectional ranker. Pipeline:
```
investor_data_miner → investor_feature_pipeline → investor_train_model
                              ↓
                    portfolio_orchestrator  (monthly cron, places orders)
```

- **`investor_universe.py`** — 96 tickers across 11 GICS sectors + sector map. Every name
  chosen to have a full 5-year history so walk-forward folds aren't unbalanced.
- **`investor_data_miner.py`** — merges daily prices, quarterly financials, daily macro.
  **`_FUNDAMENTAL_LAG_DAYS = 45`** is the most important number: filing takes weeks, so
  fundamentals are shifted forward 45 days before joining. Without it the model ranks
  using earnings nobody had seen.
- **`investor_feature_pipeline.py`** — momentum (63/126/252d), 1-month reversal, trailing
  vol, quality ratios, macro, within-date cross-sectional rank-normalisation, and the
  target **`target_top_quintile`** (top fifth of the universe over the next 60 days) —
  a *relative* question, which is what makes this a ranker.
- **`investor_train_model.py`** — LightGBM LambdaRank, expanding walk-forward,
  `EMBARGO_DAYS = 60` (the target looks 60 days ahead; the gap prevents overlap). Gates on
  **lift**, not raw precision. `GATE_P8_MIN_LIFT` is the one that matters because the
  orchestrator deploys `TOP_K=8`. The **benchmark gate** simulates the deployed basket
  against equal-weighting the universe; `INVESTOR_GATE_BENCH_T` (default 2.0) binds.
  Fails closed.
- **`portfolio_orchestrator.py`** — `TOP_K=8` at equal 12.5% weight, `SECTOR_CAP=2`,
  `REBALANCE_DEADBAND=0.005`, `EQUITY_BUFFER=0.99`. Steps 1–2 run as subprocesses so a
  crash can't leave the process half-updated.
- **Cron:** `30 16 1 * *` (16:30 PT on the 1st) runs `run_investor_rebalance.sh`.

---

## 12. Offline analysis (`src/analysis/`)

Nothing here runs during live trading. ⚠️ Most target the **legacy Alpaca equities
stack** (root-level `models/angel_latest.pkl`, fixed-percentage brackets). The exception
is `behavior_matrix.py`, which is current.

- **`behavior_matrix.py`** — scores candidate **configurations** per behavior tag, cells
  carry trade count/win rate/expectancy/PF with bootstrap intervals. **Cost is not
  optional** (cells reported net of a per-trade toll in R). Below `MIN_CELL_TRADES` (30)
  a cell is uninformative and `recommend()` skips it.
- **`decision_grader.py`** — grades the **served** model's recorded decisions against what
  price did next. Uses the `ev="bar"` records (~500/day; only ~1/week becomes a fill).
  First run on 9,183 graded decisions found the headline: the model's confidence is
  **inverted at the top** — it wins 28–32% in low bands (random = 29.3%) but only 6.7% at
  the 0.40 bar it trades on. The model is right about direction; the bracket loses anyway.
- **`strategy_backtester.py`** — strategy-agnostic scorer. Key conventions: `macro_win` is
  bracket resolution (target reached), not profitability; `gross_r`/`net_r` are the
  realized move over the stop distance. Realism: gap fills at the open, timeouts pay the
  realized move, and when a `RiskManager` is passed the **live gates run**.
- **`walk_forward_tuner.py`** — grid search scored strictly OOS; `robust` requires the
  Clopper-Pearson lower bound to clear break-even.
- **`build_strategy_matrix.py`** — the **only legitimate source of a routing table**.
- **`failure_modes.py`** — how losers lose: stopped instantly (`FAST_SL_CUTOFF = 3`), bled
  slowly, or timed out.
- **`optimize_brackets.py`** / **`optimize_threshold.py`** — older sweepers; the former is
  import-dead; the latter predates Angel/Devil.

### Diagnostics & calibration (`scripts/`)
- **`probe_model.py`** — "the bot hasn't traded in days, is it broken?" Distinguishes
  **DRIFT** (inputs moved; retraining may help) from **HONEST** (inputs normal; setups
  genuinely absent). Uses per-feature/instrument PSI + TreeSHAP, read against the null
  calibration, never raw textbook cutoffs.
- **`angel_bar_frontier.py`** — "could this model pass its own gate at ANY bar?" Reference
  run: **0 of 6 points satisfy both criteria**; verdict **unreachable**.
- **`bake_spread_alphas.py`** — parses `SPREAD_CALIB` lines into a per-instrument cost
  table. ⚠️ `--denomination-minutes` is load-bearing: alphas measured on M15 are not valid
  for M1.
- **`generate_feature_stats.py`** — backfills `feature_stats.json`; the end-date argument
  **must be the model's training date** or it manufactures drift.
- **`evaluate_barriers.py`** — the Phase 1 barrier promotion gate (currently FAIL).
- **`run_catboost_ab.py`**, **`run_h4_candidate.py`**, **`run_stability_batch.sh`** — A/B
  and stability runs; nothing writes under `models/` unless deliberately promoted.

---

## 13. The measured edge budget (the most important finding)

Established 2026-09-14; the evidence lives in the repo's live thread and in
`llm_reports/recons/2026-09-14_session-evidence-and-options.md`.

**The conclusion:** the bot is not misconfigured. Its measured edge is roughly a third of
what the market requires, and every lever tested acts on the cost side instead.

| quantity | value | currency |
|---|---|---|
| model's real selectivity | ≈ **+0.045R** | +1.5pp win rate over base rate at 2:1 |
| static bracket's toll | ≈ **0.25R** | per-trade spread over a 2×ATR stop |
| a wide bracket's toll | ≈ **0.04R** | same measure over a ~10×ATR stop |
| **what any bracket geometry needs** | ≈ **0.09R** | best random-entry cell of a 60-geometry sweep |

**So widening the stop does not create edge — it lowers the cost below the signal.** The
wide arm measures ≈break-even (−0.049R) against the static arm's −0.207R. The gap is a
factor of two to three on the *edge* side: find something worth ~3pp of win rate or stand
down.

### The four research axes and their verdicts
- **Bracket geometry / target definition — CLOSED.** 60-geometry sweep: **0 of 60 cells
  positive at random entries**; gross expectancy tracks break-even within ±0.002R.
- **Direction — CLOSED, against the hypothesis.** Shorts were the untested lever;
  measured long beats short in every period of both configs.
- **A different market (crypto) — CLOSED.** 0 of 27 geometries positive at random entries
  at D1 and 0 of 27 at H4.
- **Non-fiat instruments — OPEN, and a HUMAN decision.** Metals had a 47.8–56.6% long base
  rate in 2025 (genuinely favourable) but `UNTRADEABLE_SYMBOLS` excludes XAU/XAG because
  this practice account can't trade them.
- **A genuinely different feature/target design — UNTESTED.** The only axis left that
  could *raise* the +0.045R rather than lower the toll.

### Related measured facts
- The M15 model's edge over random is real but tiny (+0.011 to +0.019 over base rate), and
  **at the thin top the EV-maximizing calibration selects, the edge turns negative**.
- The H4 CatBoost candidate is not promotable three ways; its newest 18% of data was never
  scored (`scripts/run_h4_candidate.py` carves a holdout and never evaluates it).
- The learned barrier's net gain is **entirely cost dilution** — a constant wide bracket
  (10.25×/2.74×) matches the learned per-bar quantiles to within 0.004R.

---

## 14. Operational scaffolding

### The soak
- **`run_soak.sh`** — sources `.env`, sets `PYTHONPATH=src:.`, uses the pipenv venv python
  (system python 3.14 lacks deps), logs to `logs/soak_<ts>.log`, PID in `/tmp/soak.pid`.
  Gzips logs >30 days, deletes gzips >180 days.
- **`soak.service`** — systemd **user** unit. `OANDA_MODEL_DIR=models/forex_m15_wide` is
  the **single source of truth for the served model**, and it must match the brackets in
  the checked-out tree (a model trained on one bracket and served under another is
  train/serve skew). `Restart=no` — `soak_watchdog.sh` owns *when* to launch.
  `KillMode=mixed` + `TimeoutStopSec=90` let the SIGTERM flatten finish.
- **`soak_watchdog.sh`** — cron every 5 min: starts the soak if dead, logs a post-mortem.
  ⚠️ **Kill switch: `touch soak.off` BEFORE stopping it**, or it resurrects within 5 min.
  Its liveness is historically pgrep-only (a wedged event loop with open positions looks
  alive); a `status.json` freshness check was the recommended fix.
- **Lingering must stay on** (`loginctl enable-linger`) or the unit dies with the session.

### The MCP server (`trading_mcp.py`)
Read-only observability tools plus two guarded control tools (start/stop the soak) behind
a **two-step confirm token**. `tests/test_trading_mcp.py` pins that a wrong, missing, or
reused token cannot act — an AI assistant can't start/stop the bot by accident.

### Safety rails (repo conventions)
- **Software stops, not broker brackets** — the live process being alive is a safety
  requirement.
- **Model artifacts written atomically** (temp file + rename) because the live strategy
  hot-reloads them.
- A position that won't close is **parked** (`CLOSE_FAILED`), never forgotten.
- On boot, **ask the broker**; never assume flat.

---

## 15. Testing (`tests/`)

**538 tests + 6 subtests, all passing** (2026-09-16 snapshot). Run:
```bash
PYTHONPATH=src:. python -m pytest -q
```
`pyproject.toml` sets `testpaths = ["tests"]` as a guard against root-level `test_*.py`
probe scripts with import-time side effects (one once posted to the live Discord webhook).

**No network, no broker, no real models.** Every test stubs its dependencies; orchestrators
are even constructed via `__new__` to skip heavy `__init__`.

Most tests target **failure paths, not happy paths**, because stops are enforced in
software: the dangerous states are "we think we're flat but aren't" and "we tried to close
and it didn't work." Key files:
- `test_risk_manager.py` (35) — gates and barrier substitution; a regression here silently
  starts taking trades the system was built to refuse.
- `test_oanda_forex.py` (31) — failure paths: rapid-breach ticks close once, close not
  called synchronously in tick, watchdog failure parks, boot reconcile aborts on failed sync.
- `test_holdout_gate.py` (26) — holdout leakage guards and the confidence-bound verdict.
- `test_ml_strategy_guards.py` (27) — stale-bar guard, threshold pinning, barrier sidecar.
- `test_events.py` (11) — the telemetry sink never raises/blocks/is on the tick path.
- `test_barriers.py` (39), `test_dynamic_thresholds.py` (14), `test_entry_guards.py` (15),
  `test_stream_liveness.py` (9), `test_trading_mcp.py` (9), and others.

---

## 16. Things that will mislead you (curated trap list)

1. **`V3RandomForestTrainer` holds a LightGBM model.** `.load()` unpickles whatever is on
   disk; production has been LightGBM since 2026-05-23. It reads both
   `feature_names_in_` and CatBoost's `feature_names_`.
2. **`Signal.raw_tp_distance` is written but never read.** `RiskManager` owns target sizing.
3. **Bracket multipliers differ by asset class.** Module constants say 0.5×/3.0×; the
   forex profile overrides to 2.0×/4.0×. Always read the profile.
4. **"regime" means two things** — a volatility band (`risk_manager`, `reinforcement_voter`)
   vs an HMM hidden state (`ml/regimes/`). Unrelated.
5. **"watchdog" means two things** — the in-process stop monitor vs the cron restarter.
6. **"heartbeat" means two things** — OANDA's keepalive vs the strategy's periodic log.
7. **Don't use textbook PSI thresholds** (0.10/0.25). Use the null calibration in
   `feature_stats.json`.
8. **`resolver.py` is orphaned** — reads `signal_ledger.csv` while everything else uses
   `signal_ledger.parquet`. `run_pipeline.sh` Phase 3 runs `evaluate_performance.py`.
9. **`optimize_brackets.py` is import-dead** — references a function removed 2026-05-22.
10. **`backtest_60.py` (root) is broken** — calls a constructor signature and method that
    no longer exist.
11. **`src/autopilot/` and `src/research/` have no source** — only stale `__pycache__`.
12. **`utils/risk_management.py` is a 0-byte stub** — the real logic is in
    `execution/risk_manager.py`.
13. **One `Signal` class now** — `strategies.base.Signal`. `core.signal.Signal` was deleted.
14. **`volume` from OANDA is tick count, not traded size.**
15. **`fetch_training_data`** is a function in `core/retrainer/_data.py`; the same-named
    dead module in `src/data/` was deleted 2026-09-16.
16. **The root `readme.md`** describes the deleted V3.4 Alpaca scalper.
17. **`_GRANULARITY_PROFILES` matters**: `get_asset_config`'s default HTF pairing assumes
    M1 (`"5m"`); an M15 caller reusing `cfg` for feature engineering silently trains on the
    wrong HTF features.

---

## 17. Configuration & artifact reference

### Live model directory (`models/forex_m15_wide/`, served)
- `angel_latest.pkl`, `devil_latest.pkl` — the two LightGBM models
- `threshold.json` — Devil threshold **and** the pinned Angel bar (`0.44` / `0.3833`)
- `metadata.json` — asset class, timeframe, trained symbols, training date, holdout
  metrics. Current: 8 symbols (XAU/XAG, GBP_JPY, AUD_JPY, EUR_JPY, NZD_JPY, GBP_AUD,
  GBP_NZD), M15, HTF 1h, `sl/tp = 2.0/4.0`, 730-day lookback, holdout PF lower bound 2.3987.
- `feature_stats.json` — training distributions + null calibration for drift probing
- `spread_alphas.json` — the cost table (when the cost experiment is on)
- `barriers_{mae,mfe}.pkl` + `barriers_meta.json` — learned geometry (when enabled)
- `hmm_latest.pkl` — per-symbol regime models (when that experiment is on)

### Key environment variables
| Var | Effect |
|---|---|
| `OANDA_MODEL_DIR` | redirect the live artifact set to a side model |
| `RETRAIN_MODEL_DIR` | redirect a training run to a side directory |
| `OANDA_ENV` | `practice` (default) or `live` |
| `OANDA_UNITS` | position size (default 1000) |
| `DATA_SOURCE` | `oanda` / `alpaca` / `polygon` / `yahoo` |
| `RISK_*_GATE_ENABLED` | toggle Gates A/B/C; `RISK_CHOP_FILTER_ENABLED` masters |
| `RETRAIN_DAYS_BACK` | training lookback window |
| `RETRAIN_HOLDOUT_FRAC` | holdout fraction (default 0.18; 0 disables) |
| `RETRAIN_DEVIL_LABEL` | `survival` (default) or `macro` |
| `RETRAIN_LEARN_BARRIERS` | fit and save barrier artifacts (default on) |
| `RETRAIN_BARRIER_VERDICT` | path to the barrier promotion verdict |
| `BARRIER_GEOMETRY_ENABLED` | serve learned brackets (default off) |
| `RETRAIN_USE_HMM` | add HMM regime features |
| `RETRAIN_SPREAD_TABLE` | path to a measured cost table |
| `EVENTS_ENABLED` | telemetry off switch |
| `LOG_MAX_CHARS` | log-truncation cap (0 disables) |

---

## 18. Suggested study questions for the notebook

1. Trace one M15 bar end-to-end: from an OANDA tick that completes the bar, through
   feature generation, Angel and Devil inference, the three gates, order submission, and
   finally the tick watchdog that can close it. Which components run on the event loop and
   which are offloaded, and why?
2. Why is software-enforced SL/TP a safety architecture rather than just an implementation
   detail? Enumerate every piece of hardening that follows from it.
3. What exactly is meta-labeling, and why does the Devil train only on Angel-approved
   bars? What would go wrong with one model tuned for both recall and precision?
4. Explain the "symmetry contract" between `RiskManager`'s gates and
   `retrainer._compute_chop_veto_mask`. What silent failure appears if one side changes
   without the other?
5. The system measures ~+0.045R of edge and needs ~0.09R. Explain the R currency, the
   spread toll, and why widening the bracket from 2× to ~10× moves net expectancy from
   −0.207R toward break-even without adding any predictive skill.
6. Why is the Clopper-Pearson lower bound used instead of the point-estimate profit factor
   for the promotion gates? What behavior of small-sample PF does it correct?
7. Why is the holdout carved **before** any feature engineering, and what leakage does the
   boundary-tail purge prevent? What is an "unresolvable tail"?
8. What is the lookahead guard (`available_at`) and what does a backtest look like if it is
   broken? Give the 12:00/12:01 example.
9. Compare the Angel and Devil thresholds as they are chosen today versus the old fixed
   constants. Why did score compression motivate calibrating the Angel bar from OOF
   quantiles?
10. The learned barrier gate currently FAILS even though the learned stop beats the static
    constant on pinball loss on every fold. Explain how both can be true, and what
    `q_mae_scale` calibration does to the verdict.
11. Which "regime" and "watchdog" meaning is live in a given file? Find one place where
    confusing them would produce a wrong conclusion.
12. Why does the strategy registry resolve lazily, and what specific failure does the eager
    version cause for the live bot?
13. Walk the promotion pipeline's decision points: fold gate, edge-over-random, artifact
    holdout gate. Which are gated on and which are telemetry only, and what is the argument
    for each choice?
14. Two products share this repo. What do they actually share, and where is that sharing
    enforced (or deliberately avoided)?

---

*End of study notes. Source of truth remains the code and `GLOSSARY.md`; if this document
and the working tree disagree, read the code and update this file.*
