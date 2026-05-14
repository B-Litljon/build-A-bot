---
type: refactor
date: 2026-05-14
time: 02:15 PDT
agent: Claude Sonnet 4.6
model: claude-sonnet-4-6
trigger: Strike list of 6 data-layer bugs across oanda_provider, polygon_provider, bar_aggregator, and market_provider (schema drift + asyncio misuse)
head: 1e09f71c6b8d3a5f2e7d9b4c0a1f8e3d6b5c2a9
scope: modifies-source
related:
  - refactors/2026-05-10_oanda-integration.md
  - refactors/2026-05-14_execution-safety-and-locks.md
files_touched:
  - src/data/market_provider.py
  - src/data/oanda_provider.py
  - src/data/polygon_provider.py
  - src/data/alpaca_provider.py
  - src/utils/bar_aggregator.py
---

# Data Provider Critical Fixes — Schema Drift, Tick Timestamp, Pagination, and asyncio

## Context

Directed refactor targeting six bugs identified by an architect LLM review of the data ingestion layer. All bugs were in the `fix/data-provider-critical` branch scope. The fixes fall into four categories:

1. **Schema drift** — `_BAR_SCHEMA` was duplicated in every provider with no shared source of truth.
2. **Tick timestamp misalignment** — OANDA streaming bars were stamped with the raw tick arrival time rather than the window start.
3. **Pagination duplicate boundary** — OANDA historical pagination re-fetched the last candle on every page.
4. **asyncio misuse** — Both OANDA and Polygon were calling `asyncio.run()` per bar from a background thread, creating a new event loop on every callback invocation.

---

## Investigation

### Fix 2.1 — Schema drift (`market_provider.py`, `oanda_provider.py`, `polygon_provider.py`, `alpaca_provider.py`)

All three concrete providers carried their own copy of `_BAR_SCHEMA`:

```python
# oanda_provider.py  (line 32)
_BAR_SCHEMA = {"timestamp": pl.Datetime(...), "open": pl.Float64, ...}

# polygon_provider.py (line 25) — identical dict, different comment
_BAR_SCHEMA = {"timestamp": pl.Datetime(...), "open": pl.Float64, ...}
```

`alpaca_provider` had no schema at all and returned bare `pl.DataFrame()` on error, which strips timezone info and column names — silent downstream failures.

### Fix 1.2 — `bar_start` timestamp (`oanda_provider._handle_tick`, line 172)

```python
# Before — bar_start was the raw tick arrival time
self._tick_bars[instrument] = {
    "bar_start": ts,   # could be 12:34:47 for a 12:30 bar
    ...
}
```

The bar epoch was computed correctly (`int(ts.timestamp()) // (gran * 60)`) but was never converted back to a floored datetime; the raw `ts` was used instead.

### Fix 1.6 — Pagination duplicate boundary (`oanda_provider.get_historical_bars`, line 280)

```python
# Before
chunk_start = last_ts   # next request starts AT last_ts → re-fetches it
```

OANDA's `from` parameter is inclusive. Re-using `last_ts` as the next chunk start guaranteed the last candle of each page appeared twice.

### Fix 1.3 — `bar_aggregator` timeframe validation (`bar_aggregator.LiveBarAggregator.__init__`, line 54)

`_window_floor` uses `ts.minute % self.timeframe` arithmetic that only produces clean clock-aligned windows when `timeframe` divides 60 evenly. A timeframe of 7 or 11 silently produced nonsensical window boundaries; there was no guard.

### Fix 2.3/2.4 — asyncio threading (`oanda_provider._flush_bar`, `polygon_provider._sync_handler`)

```python
# Before — in both providers
try:
    loop = asyncio.get_running_loop()
    loop.create_task(self._callback(bar))
except RuntimeError:
    asyncio.run(self._callback(bar))   # new event loop per bar
```

`asyncio.run()` spins up a brand-new event loop for every bar when called from a thread with no running loop — expensive and incompatible with long-running async pipelines. `loop.create_task()` only works if the caller IS on the running loop (same thread), which is not the case when `run_stream()` fires from a background thread.

---

## Findings / Changes

### `src/data/market_provider.py`

Added `_BAR_SCHEMA` class attribute and `_empty_bars()` classmethod to the base:

```python
_BAR_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
    "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
    "close": pl.Float64, "volume": pl.Float64,
}

@classmethod
def _empty_bars(cls) -> pl.DataFrame:
    return pl.DataFrame({col: [] for col in cls._BAR_SCHEMA}, schema=cls._BAR_SCHEMA)
```

### `src/data/oanda_provider.py`

- Removed module-level `_BAR_SCHEMA` dict (was lines 31–39).
- Added `timedelta` to datetime imports.
- Added `self._loop: Optional[asyncio.AbstractEventLoop] = None` to `__init__`.
- **Fix 1.2**: `bar_start` now computed from epoch: `datetime.fromtimestamp(bar_epoch * self._stream_gran * 60, tz=timezone.utc)`.
- **Fix 1.6**: `chunk_start = last_ts + timedelta(minutes=timeframe_minutes)`.
- **Fix 2.3**: `_flush_bar` now uses `asyncio.run_coroutine_threadsafe(self._callback(bar), self._loop)` when loop is running.
- **Fix 2.4**: `subscribe()` captures `self._loop` via `asyncio.get_running_loop()` (falls back to `get_event_loop()`). `stop_stream()` flushes all in-flight tick-bar accumulators before clearing state.
- All empty/error DataFrame returns updated to `self._empty_bars()`.

### `src/data/polygon_provider.py`

- Removed module-level `_BAR_SCHEMA`.
- Added `asyncio`, `Optional` imports.
- Added `self._loop` field; captured in `subscribe()`.
- **Fix 2.4**: `_sync_handler` in `run_stream()` now uses `run_coroutine_threadsafe` instead of `asyncio.run()`.
- Empty returns use `self._empty_bars()`.

### `src/data/alpaca_provider.py`

- Replaced two bare `return pl.DataFrame()` calls with `return self._empty_bars()` so error paths return a schema-conformant frame.

### `src/utils/bar_aggregator.py`

- Added validation in `__init__` (after the `timeframe < 1` check):

```python
if 60 % timeframe != 0:
    raise ValueError(f"timeframe must evenly divide 60, got {timeframe}")
```

---

## Verification

Ran an in-process smoke test via `pipenv run python -c "..."` covering all six fixes:

```
Fix 2.1 OK: _BAR_SCHEMA and _empty_bars on base class
Fix 1.3 OK: timeframe=7 rejected: timeframe must evenly divide 60, got 7
Fix 1.3 OK: timeframe=5 accepted
Fix 1.2 OK: bar_start floored to window boundary
Fix 1.6 OK: pagination advances by timedelta
Fix 2.3 OK: OANDA uses run_coroutine_threadsafe
Fix 2.4 OK: stop_stream flushes in-flight bars
Fix 2.4 OK: Polygon uses run_coroutine_threadsafe

ALL FIXES VERIFIED
```

`git diff --stat` confirmed exactly 5 files changed, 54 insertions, 53 deletions. Committed to `fix/data-provider-critical` as `c29ed0a`.

---

## Risk & follow-ups

- **asyncio loop lifetime**: `self._loop` is captured at `subscribe()` time. If the caller tears down and recreates an event loop between `subscribe()` and `run_stream()`, `_loop` will be stale. In practice the V5 pipeline calls both from the same async context, so this is low risk, but worth documenting.
- **Alpaca schema conformance**: `alpaca_provider.get_historical_bars` returns `pl.from_pandas(df)` on the happy path, which may produce different column dtypes than `_BAR_SCHEMA` depending on what Alpaca sends. A downstream cast/validation step should be added if strict schema enforcement is needed live.
- **OANDA `stop_stream` flush race**: The flush in `stop_stream()` calls `_flush_bar` synchronously. If `run_coroutine_threadsafe` is used and the target loop is being torn down concurrently, the future may silently drop. Low priority for current paper-trading use.

---

## Files touched

- `src/data/market_provider.py` — lines 31–42 added (`_BAR_SCHEMA`, `_empty_bars`)
- `src/data/oanda_provider.py` — lines 17, 29–39 removed, 107–130 (`__init__`), 155–158 (`bar_start` fix), 193–196 (`_flush_bar`), 273 (pagination), 281–286 (empty returns), 295–302 (`subscribe`), 335–339 (`stop_stream`)
- `src/data/polygon_provider.py` — lines 10–14, 48–51 (`__init__`), 132–153 (empty returns), 155–167 (`subscribe`), 182–199 (`run_stream`/_sync_handler)
- `src/data/alpaca_provider.py` — lines 88, 105 (empty returns)
- `src/utils/bar_aggregator.py` — lines 56–57 (timeframe validation)
