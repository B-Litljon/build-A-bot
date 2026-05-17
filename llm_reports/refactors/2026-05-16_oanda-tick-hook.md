---
type: refactor
date: 2026-05-16
time: 11:51 PDT
agent: Kimi K2.6
model: moonshotai/kimi-k2.6
trigger: Strike 2 — expose raw-tick hook in OandaMarketProvider for sub-second watchdog reaction
head: b43e8f16ec0f9faf786c72d300a251bf183950a9
scope: modifies-source
related:
  - refactors/2026-05-10_oanda-integration.md
files_touched:
  - src/data/oanda_provider.py
  - tests/test_oanda_tick_hook.py
---

# OANDA Raw-Tick Hook — `tick_callback`

## Context

The V5 forex scalper's watchdog needs to react to price moves in **sub-second** time, but `OandaMarketProvider` only emits completed bars (1-minute by default). There is no event path between the raw OANDA PricingStream tick and the strategy layer. This strike adds an optional `tick_callback` to `subscribe()` that fires synchronously on every valid PRICE tick **before** bar aggregation, so the watchdog can detect SL/TP breaches inline without waiting for bar close.

Architect constraints (frozen contract):
- Extend `subscribe()` signature with `tick_callback: Optional[Callable] = None`; existing callers unaffected.
- Invocation signature is exactly `tick_callback(symbol, bid, ask)`.
- Fire after parsing `instrument`/`bid`/`ask` from the message but **before** bucketing logic (mid computation, epoch rollover, `_flush_bar`).
- Synchronous on the stream thread; callee must return in <50 µs and do NO blocking I/O.
- Skip non-PRICE msgs (HEARTBEAT) exactly as existing aggregation code does.
- Do NOT silently swallow `tick_callback` exceptions: log `ERROR` with symbol, continue stream.
- Do NOT change bar aggregation behaviour when `tick_callback` is `None`.

## Investigation

**Files examined:**

- `src/data/oanda_provider.py` — read end-to-end to understand stream lifecycle and `_handle_tick` aggregation (lines 135–189).
- `llm_reports/refactors/2026-05-10_oanda-integration.md` — reviewed R1 (asyncio loop per bar flush) and R7 (streaming reconnect) to confirm the hook must be synchronous and non-blocking.

**Key observations:**

- `run_stream()` (line 332) is a **blocking** `for msg in self._client.request(req)` loop; it filters `msg.get("type") == "PRICE"` before calling `_handle_tick(msg)` (line 326). HEARTBEAT messages never reach `_handle_tick`.
- `_handle_tick` already has an outer `try/except` that logs `WARNING` on parse errors and returns early when `bids` or `asks` are empty (lines 147–149).
- `_flush_bar` (line 182) creates a fresh `asyncio` event loop on every call when `get_running_loop()` raises (line 193–196). A blocking watchdog callback inside `_handle_tick` would stall the stream and starve the loop. The contract's <50 µs / no-I/O rule is a hard requirement.
- `_tick_bars` state is mutated **after** mid computation and epoch check (lines 167–189). The hook must sit between bid/ask parsing and mid computation so it sees the raw spread.

**Hypothesis testing:**

- Confirmed that `subscribe()` is called before `run_stream()`; `self._tick_callback` is safe to store as an instance attribute and reference inside `_handle_tick` without additional locks (the stream is single-threaded).
- Verified that a `MagicMock` injected as `tick_callback` inside `_handle_tick` can be asserted for call order against a patched `_flush_bar` — proving the hook fires before bar rollover.

## Findings / Changes

### `src/data/oanda_provider.py`

1. **Added `_tick_callback` storage** (line 114):
   ```python
   self._tick_callback: Optional[Callable] = None
   ```

2. **Extended `subscribe()`** (lines 302–330):
   ```python
   def subscribe(
       self,
       symbols: List[str],
       callback: Callable,
       tick_callback: Optional[Callable] = None,
   ) -> None:
       """
       Register *callback* for real-time bar updates.
       ...
       Parameters
       ----------
       tick_callback : callable, optional
           Invoked synchronously on every raw tick as
           ``tick_callback(symbol, bid, ask)``.  Must return in <50 µs and
           perform **no blocking I/O**; any exception is logged and the
           stream continues.
       """
       self._symbols = [_to_oanda_symbol(s) for s in symbols]
       self._callback = callback
       self._tick_callback = tick_callback
       self._tick_bars = {}
       ...
   ```

3. **Added raw-tick hook inside `_handle_tick()`** (lines 155–165):
   ```python
   bid = float(bids[0]["price"])
   ask = float(asks[0]["price"])

   # Raw tick hook — callee must return in <50 µs and do NO blocking I/O
   if self._tick_callback is not None:
       try:
           self._tick_callback(instrument, bid, ask)
       except Exception as e:
           logger.error(
               "OandaMarketProvider tick_callback error for %s: %s",
               instrument,
               e,
               exc_info=True,
           )

   mid = (bid + ask) / 2.0
   bar_epoch = int(ts.timestamp()) // (self._stream_gran * 60)
   ...
   ```

   Placement is **after** bid/ask parsing, **before** mid computation and bar aggregation. The inner `try/except` isolates callback errors from the outer `_handle_tick` exception handler so a watchdog bug never kills ingestion.

### `tests/test_oanda_tick_hook.py` *(new)*

Five mocked test cases exercising `_handle_tick` directly with synthetic OANDA PRICE message dicts:

| Test | Scenario | Asserts |
|------|----------|---------|
| `test_tick_callback_receives_correct_args` | Valid PRICE msg | `tick_callback` called once with `(instrument, bid, ask)` |
| `test_tick_callback_fires_before_bar_flush` | Tick causes bar rollover | `tick_callback` call count == 1 **before** `_flush_bar` is entered (verified via `side_effect` assertion) |
| `test_no_tick_callback_bar_path_unchanged` | `subscribe` without `tick_callback` | Bar state (open/high/low/close/volume) updated exactly as before; second tick in same epoch updates high/low/close correctly |
| `test_missing_bids_asks_no_tick_callback` | HEARTBEAT-shaped msg (no bids/asks) | `tick_callback` not called; `_tick_bars` remains empty |
| `test_tick_callback_exception_logged_continues` | `tick_callback` raises `RuntimeError` | `ERROR` log contains `tick_callback error for EUR_USD`; bar state still created despite exception |

## Verification

1. **New tick-hook tests:**
   ```bash
   $ pipenv run python -m unittest tests.test_oanda_tick_hook -v
   test_missing_bids_asks_no_tick_callback ... ok
   test_no_tick_callback_bar_path_unchanged ... ok
   test_tick_callback_exception_logged_continues ... ok
   test_tick_callback_fires_before_bar_flush ... ok
   test_tick_callback_receives_correct_args ... ok
   ----------------------------------------------------------------------
   Ran 5 tests in 0.004s
   OK
   ```

2. **Regression on existing tests:**
   ```bash
   $ pipenv run python -m unittest tests.test_execution_safety -v
   test_alpaca_rebalance_gate ... ok
   test_oanda_close_position_fill_parsing ... ok
   test_oanda_partial_fill_behavior ... ok
   ----------------------------------------------------------------------
   Ran 3 tests in 0.005s
   OK
   ```

3. **Syntax checks:**
   ```bash
   $ pipenv run python -m py_compile src/data/oanda_provider.py
   $ pipenv run python -m py_compile tests/test_oanda_tick_hook.py
   ```

No regressions in bar aggregation, stream lifecycle, or existing safety tests.

## Risk & follow-ups

| # | Risk | Severity | Mitigation / next step |
|---|------|----------|------------------------|
| R1 | **Blocking callee stalls stream** — If the watchdog (or any future consumer) performs I/O inside `tick_callback`, the single-threaded `run_stream()` loop blocks and bar flushes are delayed. | High | Enforce the <50 µs contract in watchdog code; use `asyncio.create_task` or a `ThreadPoolExecutor` for any outbound network call. Document this explicitly in the watchdog handoff. |
| R2 | **Tick callback exceptions are not surfaced upstream** — They are logged but swallowed. A crashing watchdog will not stop the strategy; the scalper may continue trading without stop-loss protection. | Medium | The watchdog module should maintain its own health metric (e.g., last_tick_timestamp) and the orchestrator should flatline the position if the watchdog goes stale. |
| R3 | **Bid/ask vs mid price** — The hook receives raw bid/ask; the bar aggregation uses mid. The watchdog must be aware that SL/TP levels should probably reference mid (or the relevant side for the current position), not bid or ask in isolation. | Low | Document in watchdog contract: `tick_callback(symbol, bid, ask)` — caller decides which price to use based on position direction. |
| R4 | **No async tick_callback support** — The hook is sync only. If the watchdog later wants an `async` handler, it would need a different mechanism (e.g., `asyncio.Queue` fed from the sync hook). | Low | The sync design is intentional (minimal latency). An async wrapper can be added later without changing this module's interface. |
| R5 | **HEARTBEAT filtering is in run_stream, not _handle_tick** — If `_handle_tick` is ever called directly with a HEARTBEAT dict (e.g., in a future unit test), the `msg["instrument"]` KeyError is caught by the outer `try/except` and logged as a parse warning. This is harmless but noisy. | Low | Move the `type == "PRICE"` guard into `_handle_tick` if direct invocation becomes common. |

## Files touched

| File | Lines | Change |
|---|---|---|
| `src/data/oanda_provider.py` | 114 | Added `self._tick_callback: Optional[Callable] = None` |
| `src/data/oanda_provider.py` | 302–330 | Extended `subscribe()` with `tick_callback` param, docstring, and storage |
| `src/data/oanda_provider.py` | 155–165 | Added raw-tick hook in `_handle_tick()` between bid/ask parsing and bar aggregation |
| `tests/test_oanda_tick_hook.py` | 1–138 | New file: 5 test cases covering args correctness, ordering, no-op default, HEARTBEAT skip, and exception resilience |
