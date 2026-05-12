---
type: refactor
date: 2026-05-10
time: 02:10 PDT
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: Post-hoc documentation of commit 7addd18 (V5 forex pivot — OANDA market data + order manager wiring)
head: 7addd18c85f38c299eaa194ea2c74862c0b006cf
scope: modifies-source
files_touched:
  - src/data/oanda_provider.py
  - src/execution/oanda_order_manager.py
  - scripts/portfolio_orchestrator.py
  - src/data/factory.py
  - scripts/investor_feature_pipeline.py
  - Pipfile
  - Pipfile.lock
imported_from: OANDA_INTEGRATION_REPORT_2026-05-10.md
---

# OANDA Integration Report — V5 Forex Scalper Foundation

- **Date:** 2026-05-10
- **Agent:** Claude Opus 4.7
- **Trigger:** Post-hoc documentation of commit `7addd18` (V5 forex pivot — OANDA market data + order manager wiring)
- **HEAD commit:** `7addd18c85f38c299eaa194ea2c74862c0b006cf` (2026-05-10 01:42:23 -0700)
- **Files created:**
  - `src/data/oanda_provider.py` (337 lines)
  - `src/execution/oanda_order_manager.py` (199 lines)
  - `scripts/portfolio_orchestrator.py` (548 lines — V4 equities, see §Scope-creep note)
- **Files modified:**
  - `src/data/factory.py` (+19/-1)
  - `scripts/investor_feature_pipeline.py` (+82/-40 — `--inference` mode)
  - `Pipfile`, `Pipfile.lock` (added `oandapyv20==0.7.2`)

---

## 1. Context

The V5 forex scalper pivot (project memory `project_v5_forex_pivot.md`) requires two new broker-coupled components:

1. A `MarketDataProvider` adapter for OANDA's v20 REST + Streaming API that fits the existing `src/data/market_provider.py` ABC — so the V5 path can plug into the same factory `DATA_SOURCE` switch as Alpaca/Polygon/Yahoo.
2. A position-state manager that complies with U.S. NFA rules enforced by OANDA: **FIFO** (oldest-position-first close ordering) and **no hedging** (no simultaneous long+short on the same instrument).

The architectural decision flagged as "pending 2026-05-09" in memory — sibling orchestrator vs broker abstraction extraction vs standalone OrderManager — was resolved by going with **standalone `OandaOrderManager`** scoped to *state + close only*. Per the docstring at `src/execution/oanda_order_manager.py:13-15`: _"Scope of this module: state + close. Entry methods, fill-stream consumers, and watchdog wiring live in separate modules."_ Entry/exit submission and the broker-side fill stream remain to be built.

## Pre-flight

```
$ git status --short
(clean)
$ git log -1 --oneline
7addd18 feat: integrate OANDA API with new market data provider and order manager
```

---

## 2. Changes

### 2.1 `src/data/oanda_provider.py` *(new)*

Concrete `OandaMarketProvider(MarketDataProvider)` backed by `oandapyV20`. Implements the three ABC methods:

| Method | Behavior |
|--------|----------|
| `get_active_symbols(limit)` | `accounts.AccountInstruments` — returns up to `limit` tradable instruments in API order (no volume ranking on OANDA). |
| `get_historical_bars(symbol, timeframe_minutes, start, end)` | Pages `instruments.InstrumentsCandles` with `from`+`count` (max 5 000 candles/page), price=`M` (mid), filters `complete=False` candles. Returns `pl.DataFrame` matching `_BAR_SCHEMA` (mirrors `polygon_provider._BAR_SCHEMA`). |
| `subscribe(symbols, callback)` + `run_stream()` | Subscribes to `pricing.PricingStream`, aggregates bid/ask mid-price ticks into fixed-duration bars (`OANDA_STREAM_GRANULARITY_MIN`, default 1m). `run_stream()` blocks; `stop_stream()` flips a `threading.Event`. |

Symbol normalization (`_to_oanda_symbol`) accepts `EUR/USD`, `EURUSD`, or `EUR_USD` and emits `EUR_USD`.

Granularity map `_GRANULARITY` covers `M1, M2, M4, M5, M10, M15, M30, H1, H2, H3, H4, H6, H8, H12, D` — request for any other minute count raises `ValueError`.

OANDA returns nanosecond-precision timestamps (`2024-01-01T00:00:00.000000000Z`) which `datetime.fromisoformat` rejects; `_parse_iso` truncates to seconds before parsing (`oanda_provider.py:68-78`).

**Tick-bar aggregation** (`_handle_tick` at `oanda_provider.py:144-186`): mid price is `(best_bid + best_ask) / 2`; `volume` field is **tick count, not trade volume** — explicitly called out in the docstring and worth re-emphasizing for downstream feature engineering.

### 2.2 `src/execution/oanda_order_manager.py` *(new)*

`OandaOrderManager` — net-position state manager. Holds at most one signed net position per instrument (`_net_positions: Dict[str, int]`, `_avg_entry_prices: Dict[str, float]`).

| Method | Behavior |
|--------|----------|
| `get_net_position(instrument)` | Signed integer units (positive=long, negative=short, 0=flat). |
| `get_average_entry_price(instrument)` | Broker-reported average entry price; 0.0 when flat. |
| `sync_position(instrument)` | Calls `positions.PositionDetails`, reads both `long.units` and `short.units`, asserts FIFO/no-hedging by selecting whichever side is non-zero. Updates internal cache. |
| `close_position(instrument)` | FIFO-compliant flatten via `positions.PositionClose` with `{"longUnits": "ALL"}` or `{"shortUnits": "ALL"}`. Returns True if a close was submitted, False if already flat or on error (state untouched on error to allow retry). |

**`PositionCloseRequest` workaround** (`oanda_order_manager.py:155-158`): `oandapyV20.contrib.requests.PositionCloseRequest` rejects `Units("ALL")` with `ValueError: incorrect units: ALL`. The fix bypasses the contrib helper and POSTs the raw `data` dict to `positions.PositionClose` directly, which the REST endpoint accepts. Worth keeping the comment — this will recur the next time someone reaches for the contrib helper.

### 2.3 `src/data/factory.py` *(modified)*

Adds `DATA_SOURCE=oanda` branch. Reads `OANDA_ENV` (default `"practice"`), `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_STREAM_GRANULARITY_MIN` (default `1`) and constructs `OandaMarketProvider`. Updated the unknown-source error to list `oanda` alongside `alpaca, polygon, yahoo`.

### 2.4 `Pipfile` / `Pipfile.lock`

Added `oandapyv20 = "*"` (pinned to `0.7.2`). No transitive churn beyond what `oandapyV20` requires.

### 2.5 Scope-creep note — V4 equities work bundled in the same commit

The commit message advertises only the OANDA work, but two V4-equities artifacts ship with it:

- **`scripts/portfolio_orchestrator.py`** (548 lines, new). A monthly cron-driven rebalance for the V4 LightGBM ranker: subprocess-runs the V4 data miner + feature pipeline, loads `models/v4_investor_lgbm.txt`, takes top-K (default 2) of `UNIVERSE = [AAPL, MSFT, NVDA, JPM, XOM, WMT, JNJ]`, then submits Alpaca `MarketOrderRequest`s with `TimeInForce.DAY`. Includes a 1% equity buffer (avoids 1¢ overdraft rejections on the second order) and a 0.5%-of-equity rebalance deadband. Sells fetch a live Alpaca quote at execute time to size fractional qty.
- **`scripts/investor_feature_pipeline.py`** — added `--inference` mode that retains the 60-day embargo window so today's row survives, writing to `data/processed/v4_inference_features.parquet` instead of the training output. The orchestrator depends on this.

Per `project_v5_forex_pivot.md`: _"The V4 investor stack ... is being deprioritized but not deleted yet."_ These additions extend rather than retire that stack. Worth flagging because (a) the commit message obscures the V4 work, and (b) the orchestrator is Alpaca-coupled and won't be the model for V5 forex execution.

---

## 3. Risk & regression table

| # | Area | Risk | Severity | Mitigation / next step |
|---|------|------|----------|------------------------|
| R1 | `OandaMarketProvider._flush_bar` (`oanda_provider.py:188-203`) | Tries `asyncio.get_running_loop()` then falls back to `asyncio.run(self._callback(bar))`. `run_stream` blocks in a sync thread, so `get_running_loop` will always raise — every bar flush therefore creates and tears down a fresh event loop. Fine in low-frequency tests; on a 1m stream across many instruments this is wasteful and may starve under load. | Medium | Run the stream in a dedicated asyncio task, or maintain a long-lived loop and `loop.call_soon_threadsafe(asyncio.create_task, ...)` from the streaming thread. Revisit before live scalping. |
| R2 | Tick-volume semantics | `volume` in streamed bars is tick count, not transacted volume — a feature engineered on this column will not match an equities-style volume signal. Documented in code, but easy to forget. | Medium | Make sure V5 Angel/Devil feature engineering treats volume as activity proxy, not size. Consider renaming the column or tagging the schema for forex bars. |
| R3 | `OandaOrderManager` — entry path missing | Class is intentionally state+close only. `submit_market_order`, fill-stream listener, watchdog (software SL/TP, FIFO enforcement on entry) are all unbuilt. The V5 strategy cannot trade end-to-end yet. | High | Next module: order entry + fill-stream consumer. Memory `project_v5_forex_pivot.md` flags software SL/TP requirement (no native conditional orders — they leak stops to OANDA's order book). |
| R4 | No OANDA tests | `find tests -name '*oanda*'` is empty. `_handle_tick` aggregation, `_parse_iso` nanosecond handling, `sync_position` long/short branching, `close_position` FIFO direction selection — all unverified except by manual inspection. | Medium | Add unit tests with mocked `oandapyV20.API.request`. The tick-bar epoch rollover is the highest-value target. |
| R5 | `_to_oanda_symbol` duplicated | Same helper in `oanda_provider.py:63-65` and `oanda_order_manager.py:27-29`. Drift risk if either file evolves the normalization rules. | Low | Hoist into a shared module (e.g. `src/execution/oanda/_symbols.py` or extend `src/data/symbols.py` if one exists). |
| R6 | Bundled V4 equities work | `portfolio_orchestrator.py` + `--inference` mode are V4 (Alpaca, lightgbm) and have nothing to do with the OANDA integration the commit advertises. Future bisects on OANDA behavior will land on a 1 197-line commit and have to disentangle V4 from V5. | Low | Already shipped — note for future commits: V4 maintenance commits should be split from V5 build commits. |
| R7 | Streaming reconnect on failure | `run_stream` catches `Exception` and exits the loop; there is no automatic reconnection. A transient TCP drop kills the data feed silently (logs, but the orchestrator loop has no signal). | Medium | Wrap `run_stream` in a supervisor with backoff, or surface a "stream-down" status on the provider for the orchestrator to react to. |
| R8 | Granularity map gaps | `_GRANULARITY` accepts only the OANDA-canonical minute counts. A scalper feature pipeline asking for `timeframe_minutes=3` (or any other unsupported value) gets a `ValueError`, not a downsample. | Low | Document the accepted values where the strategy config picks a timeframe; or add resampling for unsupported granularities. |

---

## 4. Files changed + commit status

```
$ git diff --stat 7addd18^..7addd18
 Pipfile                              |   1 +
 Pipfile.lock                         |   9 +-
 scripts/investor_feature_pipeline.py | 122 +++++---
 scripts/portfolio_orchestrator.py    | 548 +++++++++++++++++++++++++++++++++++
 src/data/factory.py                  |  19 +-
 src/data/oanda_provider.py           | 337 +++++++++++++++++++++
 src/execution/oanda_order_manager.py | 199 +++++++++++++
 7 files changed, 1197 insertions(+), 38 deletions(-)
```

Tree clean post-commit. Branch `main` is **1 commit ahead of `origin/main`** — `7addd18` has not been pushed.

## 5. Next-step shortlist (V5 path)

In rough dependency order, derived from the gaps in §3:

1. **`OandaOrderManager` entry path** — `submit_market_order(instrument, units)` returning the trade ID, plus a TransactionStream consumer that updates `_net_positions` from authoritative `ORDER_FILL` events.
2. **Software SL/TP watchdog** — a periodic task that monitors the bar stream and submits a flatten via `close_position` when the price crosses a strategy-set level. Per memory: do not use native conditional orders.
3. **V5 orchestrator** — sibling to (or replacement of) `live_orchestrator.py`, wiring `OandaMarketProvider` → strategy (Angel/Devil RF reload) → `OandaOrderManager` → watchdog.
4. **Tests** — at minimum `test_oanda_provider_tick_bars.py` and `test_oanda_order_manager_close.py` with mocked `request()`.
