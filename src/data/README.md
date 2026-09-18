# `src/data`

Everything that talks to an outside data vendor. The rest of the system never
imports a vendor SDK directly — it asks `factory.py` for a provider and codes
against the abstract contract, so switching brokers is a one-environment-variable
change (`DATA_SOURCE`).

Three layers live here:

1. **Contracts** (`market_provider.py`, `fundamentals.py`, `macro.py`,
   `enums.py`, `timeframe.py`) — abstract shapes with a strict rule: *no vendor
   SDK imports*.
2. **Adapters** (`alpaca_provider.py`, `oanda_provider.py`,
   `polygon_provider.py`, `yahoo_provider.py`, and `providers/`) — one per
   vendor, translating that vendor's quirks into the canonical shapes.
3. **The factory** (`factory.py`) — reads environment variables and hands back
   the right adapter.

A rule worth internalising: **providers return empty, they don't raise.** A
dead symbol or a vendor outage produces an empty DataFrame, so one bad
instrument can't abort a fetch of forty.

The canonical bar shape everywhere is six columns — `timestamp` (microsecond
UTC), `open`, `high`, `low`, `close`, `volume` (all Float64).

See the root [GLOSSARY.md](../../GLOSSARY.md) for domain terms (OHLCV, bid/ask,
mid price, sealed bar, warm-up, heartbeat).

> Neither this folder nor `providers/` has an `__init__.py` — they work as
> implicit namespace packages. Import prefixes are inconsistent across the repo
> (`data.market_provider` vs `src.data.market_provider`) depending on whether
> `src` or the repo root is on `PYTHONPATH`.

## Contracts

### `market_provider.py`
`MarketDataProvider` — the main abstract base class. Deliberately bundles
discovery + historical + streaming into one class, because a vendor's
credentials and rate limits are shared across all three. Also owns `_BAR_SCHEMA`,
the canonical bar shape.
- **Imports from repo:** none. **Imported by:** `factory.py`, all four market
  adapters, `src/core/retrainer.py`, `src/ml/data_miner.py`.

### `fundamentals.py` / `macro.py`
`FundamentalProvider` (company financials: sector, valuation ratios, quarterly
income statements) and `MacroProvider` (economy-wide series like VIX and
yields). Both are used by the **equities investor**, never by the forex bot.
- **Imported by:** `factory.py`, `providers/*`, `tests/test_composite_fundamentals.py`.

### `enums.py`
`AssetClass`, `AssetStatus`, `DataFeed`. Broker-agnostic by design — forex is
notably absent, because the OANDA path bypasses these.
- **Imported by:** `alpaca_provider.py`, `discovery.py`, `harvester.py`.
  (The day_trading and Alpaca-orchestrator consumers were deleted 2026-09-16.)

### `timeframe.py`
Frozen `TimeFrame(amount, unit)` plus `MIN_1` / `MIN_5` / `HOUR_1` / `DAY_1`
constants. Replaces direct use of Alpaca's own timeframe type.
- **Imported by:** `alpaca_provider.py`, `feed.py`, `harvester.py`.
  (The `fetch_training_data.py` module and the day_trading/Alpaca consumers
  were deleted 2026-09-16.)

## The switch

### `factory.py`
`get_market_provider()` reads `DATA_SOURCE`; `get_fundamental_provider()` reads
`FUNDAMENTAL_SOURCES`. Vendor SDKs are imported lazily *inside* each branch, so
running OANDA never requires the Polygon package. Unknown values raise rather
than defaulting — a typo can't quietly trade the wrong market.
- **Imports from repo:** `src.data.market_provider`, and (lazily)
  every adapter.
- **Imported by:** `src/core/retrainer.py`, `src/ml/data_miner.py`,
  `chop_ab_test.py`, `scripts/generate_feature_stats.py`,
  `scripts/investor_data_miner.py`, `tests/test_composite_fundamentals.py`.

## Market adapters

### `oanda_provider.py` — **the live forex feed**
`OandaMarketProvider`. The most operationally important file here. OANDA
streams individual quotes, not finished bars, so this module builds bars from
ticks itself. Contains the hardening that keeps the live soak alive: a
20-second read-inactivity timeout (OANDA heartbeats every ~5s, so silence means
a dead socket), a `seconds_since_last_message` liveness property the
orchestrator's watchdog polls, and `force_disconnect()` to force a reconnect.
Also exposes a raw `tick_callback` the forex bot uses to measure live spreads —
that hook runs inline on the stream thread and must not block. And
`get_tradeable_instruments()`, the account's full instrument set, which the
forex bot uses at boot to drop symbols it would only get rejected on; it returns
an empty set on failure, meaning "unknown", never "nothing tradeable".

> ⚠️ `volume` from this provider is **tick count, not traded size** — OANDA
> doesn't report real volume. Training data uses the same proxy, so the two
> agree, but don't read it as money changing hands.

- **Imports from repo:** `data.market_provider`.
- **Imported by:** `run_oanda.py`, `src/execution/oanda_forex_orchestrator.py`,
  `factory.py`, `scripts/probe_model.py`, `tests/test_oanda_tick_hook.py`,
  `tests/test_stream_liveness.py`.

### `alpaca_provider.py`
`AlpacaProvider` — US equities and crypto. Holds separate REST clients and
stream classes for each, routing on whether the symbol contains a slash
(`BTC/USD` vs `AAPL`). Uses the free IEX feed, so volume is partial.
- **Imports from repo:** `data.market_provider`, `data.enums`, `data.timeframe`.
  **Imported by:** `factory.py`.

### `polygon_provider.py`
`PolygonDataProvider`. Selectable via `DATA_SOURCE=polygon` but not used by
either live bot; kept as a working alternative.
- **Imports from repo:** `data.market_provider`. **Imported by:** `factory.py`.

### `yahoo_provider.py`
`YahooDataProvider` — **paper trading only.** No push feed exists, so
`run_stream()` polls every 60 seconds; that's the latency floor. Its
`_FALLBACK_ACTIVE` list is 15 hardcoded tickers returned when discovery fails —
a fixed stand-in, not a live "most active" ranking.
- **Imports from repo:** `data.market_provider`. **Imported by:** `factory.py`.

## A second feed abstraction

### `feed.py`
`MarketDataFeed` / `AlpacaCryptoFeed`. A *separate*, narrower contract that
coexists with `MarketDataProvider`: async, crypto-only, and adds a **warm-up**
step (pre-load recent history so indicators have enough bars before the first
live bar). Only the Factory path uses it. Contains `_resolve_index_key`, which
works around Alpaca returning `BTCUSD` when you asked for `BTC/USD`.

> ⚠️ `warmup_history` has leftover unconditional `print("[DEBUG] …")` calls
> that write to stdout on every warm-up regardless of log level. Flagged, not
> changed.

- **Imports from repo:** `data.timeframe`.
- **Imported by:** `run_factory.py`, `src/execution/factory_orchestrator.py`,
  `scripts/run_paper_live.py`, `scripts/smoke_test.py`.

## Scripts and legacy

### `harvester.py` — live (as a script)
Phase 1 of `run_pipeline.sh`. Pulls 7 days of 1-minute bars for a hardcoded
5-ticker basket and writes the file the whole offline loop runs on. Predates
the provider abstraction and calls the Alpaca SDK directly.
- **Writes:** `data/oos_bars.parquet` (overwritten each run).
- **Imported by:** nothing — run as `python -m src.data.harvester`.

### `fetch_training_data.py` — ⚠️ DELETED 2026-09-16
It was dead and misleadingly named: nothing imported it, while a live
*function* of the same name in `src/core/retrainer/_data.py` is the one actually
used (imported by `scripts/generate_feature_stats.py`, with a different basket
and output layout). Deleted in the downsizing pass; git history has it.

### `discovery.py` — ⚠️ dormant
`DiscoveryService.get_in_play_tickers()` scans the whole Alpaca universe for
stocks that gapped overnight (default: $10–$200, ≥2% gap, top 50 by volume).
Nothing imports it; the Haynes manual lists it as a legacy helper. Note the
"gap" is computed close-to-close, not previous-close-to-open. Flagged, not
removed.

## `providers/`
Fundamentals and macro adapters — see [`providers/README.md`](providers/README.md).
