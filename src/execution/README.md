# `src/execution`

Where decisions become orders. Everything here runs **live**: connect to a
broker, hold state about open positions, submit and close orders, and enforce
stops in software.

The folder contains **three orchestrators for three different brokers/markets**,
which is the main thing to get straight before reading any of it:

| File | Market | Status |
|---|---|---|
| `oanda_forex_orchestrator.py` | OANDA forex | ⚠️ **currently running live** (M15 practice soak) |
| `live_orchestrator.py` | Alpaca equities + crypto | Working, tested, not the live bot |
| `factory_orchestrator.py` | Alpaca (Factory path) | Smallest/clearest; good place to start reading |

All three share the same shape: **bar seals → run the model on a worker thread
→ if a signal comes back and we're flat, size it and send the order → a
separate watchdog loop closes the position when price crosses a level.**

Two rules hold across all of them:

- **Software stops, not broker brackets** (except Alpaca equities, which do get
  a real broker-side bracket). The consequence is that *the process being alive
  is a safety requirement* — a dead bot means an unwatched position. That's why
  the OANDA path flattens everything when its feed goes quiet, and why
  `soak_watchdog.sh` exists.
- **Model inference never runs on the event loop.** It's CPU-bound and would
  stall the feed, so it's offloaded with `asyncio.to_thread`.

See the root [GLOSSARY.md](../../GLOSSARY.md) for domain terms (bracket, chop
veto, NATR, spread, sealed bar, watchdog, warm-up).

## Files

### `risk_manager.py` — **the Shield**
`RiskProfile` (all the tunable numbers) and `RiskManager` (bracket sizing,
position sizing, and the **chop filter**). Applied *after* a strategy has
already decided it wants to trade.

The veto is the important part — a model can be right about direction and still
lose money if the cost of trading exceeds the move. Three gates:

- **Gate A (cost)** — reject when the stop distance is smaller than the cost of
  trading (`sl_dist < k_eff × spread`).
- **Gate B (regime)** — reject when volatility sits in the bottom 20% of its own
  recent window. Too quiet to reach the target before the hold limit.
- **Gate C (time)** — reject everything in the daily rollover window (default
  16:55–17:30 **New York**, so it tracks daylight saving rather than drifting an
  hour twice a year), when spreads briefly blow out ~10×.

`last_veto_gate` records which gate fired, so soak logs answer "why didn't it
trade?" with a specific constraint.

> **Symmetry contract:** `coupled_keff` and the gate logic are mirrored by
> `retrainer._compute_chop_veto_mask`, so the model only ever trains on bars the
> live bot would actually take. Change a gate here without changing the training
> side and the model learns from setups it will never be offered.

Note the forex profile overrides the bracket multipliers to **1.0× / 2.0×**
(2:1), not the 0.5/3.0 defaults.

- **Imports from repo:** none (numpy only — deliberately dependency-light).
- **Imported by:** `src/core/retrainer.py`, `factory_orchestrator.py`,
  `oanda_forex_orchestrator.py`, `__init__.py`, `run_factory.py`,
  `run_oanda.py`, `chop_ab_test.py`, `scripts/generate_feature_stats.py`,
  `scripts/run_paper_live.py`, `scripts/smoke_test.py`,
  `tests/test_risk_manager.py`, `tests/test_cost_feature.py`.
- **Data artifacts:** none directly; reads per-instrument costs passed in from
  the model dir's `spread_alphas.json`.

### `oanda_forex_orchestrator.py` (1621 lines) — ⚠️ the live bot
`OandaForexOrchestrator`. Two clocks run at once, and most of the design
follows from that:

- **Slow path** — a bar seals, features are computed, the model is asked, a
  trade may open. Every 15 minutes in the current soak.
- **Fast path** — every incoming quote is checked against the open position's
  stop and target. Runs on the provider's stream thread and **must return in
  under 50 µs with no blocking I/O**, or it stalls the feed.

Things worth knowing:

- **History seam** (`_last_hist_ts`, `_seam_crossed`, `_last_scored_ts`) —
  warm-up history and the live stream overlap in time; these prevent
  double-counting at the junction. After every (re)prime,
  `_catch_up_missed_bars` scores the newest sealed bar if it landed while the
  stream was down and is still fresh (`SEAM_CATCHUP_MAX_AGE_SECONDS`; default
  one bar period). Before this existed, signals sealing during an outage were
  silently lost — ~6 of 15 would-be signals in the 2026-07 soak.
  Its other half, `_backfill_seam_bar`, handles the bar that was *in flight*
  when the stream died: the stream's copy is incomplete so it is dropped, but
  it has sealed, so the complete version is re-fetched from REST and scored
  (`SEAM_BACKFILL_ATTEMPTS`, `SEAM_BACKFILL_RETRY_DELAY`). Measured cost
  without it: 5 lost evaluations per symbol in the first 16h of the
  2026-07-28 soak, ~8% of bars. Both seam paths emit `_emit_status` after
  scoring (added 2026-09-10) so `status.json` stays fresh through stream
  outages — the watchdog reads its mtime, and a recovering soak used to look
  stale and get restarted mid-recovery.
- **Reconnect backoff** — jittered exponential, 5s base / 60s cap, reset after
  120s of healthy streaming (`OANDA_RECONNECT_*`). The cap sits below the
  liveness watchdog's 60s flatten threshold, so backing off never leaves a
  position unwatched longer than the stall response already permits.
- **Entry guards** (added 2026-07-30, both born from one morning's trades) —
  a **post-exit cooldown** (`OANDA_REENTRY_COOLDOWN_SECONDS`, default one bar
  period) keeps a symbol from being re-entered on the bar after its own stop,
  and a **correlated-exposure cap** (`OANDA_MAX_PER_CURRENCY`, default 2)
  limits how many open positions may share the same *signed currency leg* —
  long GBP_JPY and long AUD_JPY are two short-JPY bets, and on 2026-07-30 a
  third one joined them and all three lost together. In-flight entries are
  reserved in `_pending_entries` so two signals on the same bar cannot both
  pass the cap before either fills. The cooldown does not block *flipping* an
  already-open position; set either knob to 0 to disable it.
- **Unknown entry outcomes** (added 2026-08-29) — an entry order can go out,
  fail ambiguously, and leave the broker unreadable, so whether the position
  exists is genuinely unknown. It is parked as `ENTRY_UNRECONCILED` rather
  than discarded as a zero fill: counted by the exposure caps, but ignored by
  `_on_tick`, since running a software stop against a position that may not
  exist could close something the account doesn't hold.
  `_reconcile_unverified_entries` settles it on the liveness loop — flat
  drops the record, open promotes it to `OPEN` with the bracket rebuilt from
  the stored distances around the price actually filled. A failed sync leaves
  it parked for the next pass; it is never resolved by assumption.
- **`_reconcile_on_boot`** — asks the broker what's actually open before
  trading. A restart must adopt reality, not assume it's flat, or a position
  left by a crashed process runs with nothing watching its stop.
- **`_drop_untradeable_symbols`** — at boot, drops configured symbols the
  account isn't permitted to trade. `XAU_USD`/`XAG_USD` are in the trained
  basket but not on this account, so every metals signal became a submitted
  order, an `INSTRUMENT_NOT_TRADEABLE` rejection and a stack trace that reads
  like a real fault. Deliberately **fail-open**: only a successful instrument
  lookup may drop anything, because an API blip that silently muted the whole
  basket would be far worse than the odd rejection. Dropping *everything*
  aborts startup instead of running a bot that can't place an order.
- **Liveness watchdog** — 60s of silence (OANDA heartbeats every ~5s) means the
  feed is dead, so it reconnects **and flattens exposure**. Correct, given
  software-enforced stops.
- **Spread calibration (`SPREAD_CALIB`)** — samples the real spread once per
  sealed bar, off the fast path, so the placeholder cost assumption (α = 0.15)
  can be replaced with measured per-instrument values. This is what
  `scripts/bake_spread_alphas.py` consumes.
- **Fixed size** — `units_per_trade = 1000`. This path does *not* use
  `calculate_quantity`; only the bracket logic is shared.

- **Imports from repo:** `core.notification_manager`, `data.oanda_provider`,
  `execution.oanda_order_manager`, `execution.risk_manager`,
  `strategies.concrete_strategies.ml_strategy`.
- **Imported by:** `run_oanda.py`, `scripts/bake_spread_alphas.py`,
  `tests/test_oanda_forex.py`, `tests/test_stream_liveness.py`,
  `tests/test_entry_guards.py`.
- **Data artifacts:** none written directly; logs to `logs/soak_*.log`.

### `oanda_order_manager.py`
`OandaOrderManager` — the truth about what's currently held. Tracks the **net
signed position** per instrument, never individual lots, because US (NFA) rules
require FIFO closing and forbid opposing positions in the same pair. One signed
number makes those rules impossible to violate by construction.

One asymmetry is deliberate: **a failed close leaves local state untouched.**
Believing you're flat while holding a position is far more dangerous than the
reverse. `submit_target_position` expresses orders as "end at N units" rather
than "buy N", and retries an ambiguous failure (timeout, 5xx, reset) only after
re-reading the broker position — the re-sync, not the phrasing, is what makes
the retry safe. Failures split on two questions: any 4xx was rejected before
execution (so the position cannot have moved), but only a body carrying a
reject reason is *permanent*. A business reject returns immediately; a bare
401/403/429 is a transient blip and is retried. The order client carries an
HTTP timeout so a half-open socket cannot hang the entry path.

- **Imports from repo:** none (oandapyV20 only).
- **Imported by:** `oanda_forex_orchestrator.py`, `run_oanda.py`,
  `tests/test_oanda_entry.py`, `tests/test_oanda_forex.py`,
  `tests/test_execution_safety.py`.

### `live_orchestrator.py` (2475 lines) — Alpaca dual-stream
`LiveOrchestrator`. Trades equities and crypto in one event loop, with a Rich
terminal dashboard. Not the live bot; `__init__.py` deliberately doesn't export
it.

**Threading contract** — the thing most likely to bite you here: the event loop
is the *sole owner* of every mutable field on `SymbolContext`. Worker functions
receive immutable snapshots and return frozen results (`InferenceOutcome`,
`EntryOrderResult`) that the loop applies. Thread functions never hold a
`SymbolContext` reference. Enforced by a regression test that instruments
`SymbolContext.__setattr__` and asserts every write lands on the loop thread.

Other specifics: a `SymbolState` machine (FLAT → PENDING → IN_TRADE →
PENDING_EXIT → COOLING → FLAT) with a 5-minute cooling-off so one choppy
stretch can't cause repeated re-entries; a volatility kill switch that skips
bars above 0.5204 NATR regardless of what the models say; a Smart Clock Gate
blocking equities outside market hours while crypto always passes; and a cached
HTF feature snapshot with cold/warm paths, read once into a local before
offloading so the warm path can't mix two periods.

- **Imports from repo:** `core.notification_manager`, `core.signal`,
  `ml.feature_pipeline`, `ml.features.v3_features`,
  `strategies.concrete_strategies.ml_strategy`, `utils.bar_aggregator`.
- **Imported by:** `run_live.py`, `src/analysis/optimize_brackets.py`,
  `tests/execution/test_live_orchestrator.py`.
- **Writes:** `active_trades.json` (so a restart can recover open trades).

### `factory_orchestrator.py` (191 lines) — start here
`FactoryOrchestrator`. The smallest complete trading loop in the repo and the
clearest illustration of the shared shape.

Two details worth carrying to the bigger files: it **re-checks
`active_positions` after acquiring the lock**, because sizing awaits broker
calls during which another coroutine could have entered; and its watchdog reads
the **last sealed bar's close**, so exits can lag by up to a bar — the OANDA
orchestrator improves on this by checking live quotes.

- **Imports from repo:** `data.feed`, `execution.enums`,
  `execution.risk_manager`, `strategies.concrete_strategies.ml_strategy`,
  `utils.bar_aggregator`.
- **Imported by:** `run_factory.py`, `__init__.py`, `scripts/smoke_test.py`,
  `scripts/run_paper_live.py`, `tests/verify_warmup.py`.

### `enums.py`
`OrderSide`, `OrderType`, `TimeInForce` — broker-agnostic order vocabulary.
Adapters translate these into broker types; no broker SDK imports allowed here.
The live paths use market orders only.
- **Imported by:** `factory_orchestrator.py`.

### `__init__.py`
Exports only `FactoryOrchestrator` and `RiskManager`. Both `LiveOrchestrator`
and `OandaForexOrchestrator` are **intentionally excluded** so importing this
package doesn't pull in the heavy orchestrators; entry-point scripts import them
by path.
