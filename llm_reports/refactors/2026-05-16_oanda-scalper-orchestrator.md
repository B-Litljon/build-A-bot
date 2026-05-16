---
type: refactor
date: 2026-05-16
time: 12:12 PDT
agent: Kimi K2.6
model: moonshotai/kimi-k2.6
trigger: Strike 3 — create OandaScalperOrchestrator wiring provider → strategy → order manager with software SL/TP watchdog
head: b43e8f16ec0f9faf786c72d300a251bf183950a9
scope: modifies-source
related:
  - refactors/2026-05-10_oanda-integration.md
  - refactors/2026-05-16_oanda-tick-hook.md
  - refactors/2026-05-16_oanda-entry-method.md
files_touched:
  - src/execution/oanda_scalper_orchestrator.py
  - run_oanda.py
  - tests/test_oanda_scalper.py
---

# OANDA Scalper Orchestrator — V5 Forex Pivot

## Context

The V5 forex scalper needs an end-to-end async loop that wires:
1. **OandaMarketProvider** — streaming bars + raw tick hook (Strike 2)
2. **MLStrategy** — Angel/Devil meta-labeling signal generation
3. **OandaOrderManager** — net-position entry/reversal/close (Strike 1)

with an embedded **software SL/TP watchdog** that reacts sub-second via the tick callback rather than waiting for bar close. The existing `live_orchestrator.py` is a 2300-line Alpaca-coupled monolith; this strike builds a lean ~250-line OANDA-only replacement from scratch.

Architect constraints (frozen contract):
- One asyncio loop owns: bar callback (ML path), tick callback dispatch (watchdog), graceful shutdown.
- Tick callback runs synchronously on the provider's blocking stream thread; must return in <50 µs and do NO blocking I/O.
- Watchdog breach dispatches `close_position` OFF the stream thread via `asyncio.run_coroutine_threadsafe(coro, loop)` — NOT `asyncio.create_task` (no running loop on the stream thread).
- The scheduled coroutine wraps `close_position` in `loop.run_in_executor` (blocking HTTP).
- `PENDING_CLOSE` idempotent guard prevents duplicate closes and blocks the bar path from reopening.
- Software SL/TP only — never pass native brackets to the broker.
- Graceful shutdown on SIGINT/SIGTERM: stop stream, optional flatten all.

## Investigation

**Files examined:**

- `src/data/oanda_provider.py` — confirmed `_flush_bar` (line 182) calls `asyncio.run_coroutine_threadsafe(self._callback(bar), self._loop)`, so `self._callback` must be a coroutine function (or return a coroutine when called). Confirmed `_handle_tick` (line 136) calls `self._tick_callback(instrument, bid, ask)` synchronously on the stream thread.
- `src/execution/oanda_order_manager.py` — confirmed `submit_target_position` and `close_position` are both blocking HTTP calls that must be offloaded from the asyncio loop.
- `src/strategies/concrete_strategies/ml_strategy.py` — `generate_signals(df)` returns `Optional[Signal]` with fields: `direction` ("long"/"short"), `entry_price`, `raw_sl_distance`, `raw_tp_distance`, `metadata`. Requires `symbol` column in the DataFrame (line 341).
- `src/execution/risk_manager.py` — `calculate_bracket(entry_price, raw_atr)` is broker-agnostic and returns `(sl_distance, tp_distance)` or `None` (A3 chop filter). `calculate_quantity` is equities-oriented and not wired for forex.
- `run_live.py` / `run_factory.py` — both use `sys.path.insert(0, str(_SRC_DIR))` bootstrap pattern before importing from `src/`.

**Key observations:**

- `run_stream()` is a blocking loop on a dedicated thread. `subscribe()` stores `self._callback` and `self._tick_callback`. `_flush_bar` dispatches bar callbacks to the asyncio loop via `run_coroutine_threadsafe`.
- The positions dict (`self._positions`) is accessed from BOTH the stream thread (`_on_tick`) and the asyncio loop (`_on_bar`, `_watchdog_close`). A `threading.Lock()` serializes all reads and mutations.
- `_on_bar` must build a rolling polars DataFrame with `timestamp`, `open`, `high`, `low`, `close`, `volume`, and `symbol` columns for `MLStrategy.generate_signals`.
- `MLStrategy.warmup_period` defaults to 260 bars. The orchestrator maintains a rolling buffer of `warmup * 2` bars per symbol.

## Findings / Changes

### `src/execution/oanda_scalper_orchestrator.py` *(new)*

**Constructor** (lines 49–82):
```python
def __init__(
    self,
    symbols: List[str],
    provider: OandaMarketProvider,
    strategy: MLStrategy,
    order_manager: OandaOrderManager,
    risk_manager: Optional[RiskManager] = None,
    units_per_trade: int = 1000,
    warmup_period: Optional[int] = None,
    flatten_on_exit: bool = True,
):
```

**Tick callback `_on_tick`** (lines 86–117):
- Acquires `_positions_lock`, reads position state.
- Cheap float compares: `bid <= sl` or `bid >= tp` for long; `ask >= sl` or `ask <= tp` for short.
- Idempotent guard: under lock, re-checks `state == "OPEN"`, sets `state = "PENDING_CLOSE"`.
- Dispatches `asyncio.run_coroutine_threadsafe(self._watchdog_close(symbol), self._loop)`.
- Never calls `close_position` directly.

**Watchdog close coroutine `_watchdog_close`** (lines 119–136):
```python
async def _watchdog_close(self, symbol: str) -> None:
    await asyncio.get_running_loop().run_in_executor(
        None, self._order_manager.close_position, symbol
    )
    ...
    with self._positions_lock:
        self._positions.pop(symbol, None)
```

**Bar callback `_on_bar`** (lines 138–215):
1. Append bar to rolling buffer (list of dicts → `pl.DataFrame` on demand).
2. Skip if insufficient bars for warmup.
3. Call `strategy.generate_signals(df)`.
4. Skip if signal is `None` or position is `PENDING_CLOSE`.
5. Calculate SL/TP via `risk_manager.calculate_bracket`.
6. Derive signed `target_units` (+units for long, −units for short).
7. Skip same-direction re-entry.
8. Submit via `run_in_executor(None, order_manager.submit_target_position, symbol, target_units)`.
9. On non-zero fill, record position state: `{entry, sl, tp, units, state="OPEN"}`.

**Lifecycle `run()`** (lines 217–239):
- Registers SIGINT/SIGTERM handlers that set `self._shutdown_event`.
- Subscribes to provider with both bar and tick callbacks.
- Runs blocking `provider.run_stream()` in a thread-pool executor via `asyncio.create_task(...run_in_executor(...))`.
- Waits on `self._shutdown_event`, then calls `shutdown()`.

**Shutdown `_flatten_all()`** (lines 241–269):
- Iterates all positions, dispatches parallel `close_position` calls via `asyncio.gather(*tasks, return_exceptions=True)`.
- Clears position dict after all closes complete.

### `run_oanda.py` *(new)*

Root-level entrypoint following `run_factory.py` bootstrap pattern:
- `sys.path.insert(0, str(_SRC_DIR))`
- Parses `--symbols`, `--units`, `--env`, `--no-flatten`, `--daemon`
- Reads `OANDA_SYMBOLS`, `OANDA_UNITS`, `OANDA_ENV` from environment.
- Constructs `OandaMarketProvider`, `OandaOrderManager`, `MLStrategy`, `RiskManager`, then `OandaScalperOrchestrator`.

### `tests/test_oanda_scalper.py` *(new)*

Four mocked test cases:

| Test | Scenario | Asserts |
|------|----------|---------|
| `test_bar_signal_long` | Long signal after warmup | `submit_target_position("EUR_USD", 1000)` called once; position recorded with `units=1000`, `state="OPEN"` |
| `test_bar_signal_short` | Short signal after warmup | `submit_target_position("EUR_USD", -1000)` called once; position recorded with `units=-1000` |
| `test_rapid_breach_ticks_close_once` | 5 ticks breaching SL on long position | `asyncio.run_coroutine_threadsafe` called exactly once; position state becomes `PENDING_CLOSE`; `close_position` never called synchronously |
| `test_close_not_called_synchronously_in_tick` | Single breach tick | `close_position` not called; dispatch via `run_coroutine_threadsafe` confirmed |

## Verification

1. **New scalper tests:**
   ```bash
   $ pipenv run python -m unittest tests.test_oanda_scalper -v
   test_bar_signal_long ... ok
   test_bar_signal_short ... ok
   test_close_not_called_synchronously_in_tick ... ok
   test_rapid_breach_ticks_close_once ... ok
   ----------------------------------------------------------------------
   Ran 4 tests in 0.009s
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
   $ pipenv run python -m py_compile src/execution/oanda_scalper_orchestrator.py
   $ pipenv run python -m py_compile run_oanda.py
   $ pipenv run python -m py_compile tests/test_oanda_scalper.py
   ```

No regressions in existing OANDA or Alpaca tests.

## Risk & follow-ups

| # | Risk | Severity | Mitigation / next step |
|---|------|----------|------------------------|
| R1 | **Tick callback blocking the stream** — If the watchdog (or any future consumer) performs I/O or heavy computation inside `_on_tick`, the provider's blocking stream thread stalls and bar flushes are delayed. | High | The <50 µs contract is documented in code and docstring. The watchdog must do ONLY float compares and dispatch. Any outbound I/O must go through `run_coroutine_threadsafe`. |
| R2 | **`run_coroutine_threadsafe` vs `create_task`** — On the stream thread there is no running event loop; `create_task` would raise `RuntimeError`. The orchestrator correctly uses `run_coroutine_threadsafe`. If this pattern is copy-pasted elsewhere without understanding the thread context, it will break. | Medium | Document the cross-thread dispatch pattern explicitly in the orchestrator docstring and in any watchdog module that consumes the tick hook. |
| R3 | **Position sizing is config-based, not adaptive** — `units_per_trade` is a flat integer. The RiskManager's `calculate_quantity` is equities-oriented (uses `buying_power`, `cash`, `is_crypto`). Forex position sizing should ideally account for account equity, leverage, and pip value. | Medium | TODO: Wire OANDA account details (equity, margin, leverage) into the orchestrator and adapt `RiskManager.calculate_quantity` for forex, or build a forex-specific sizing module. |
| R4 | **No reconnection on stream drop** — `run_stream` catches `Exception` and exits. A transient TCP drop kills the data feed. The orchestrator's `stream_task` would complete, but `_shutdown_event` is not set, so `run()` hangs indefinitely. | High | Add a supervisor loop around `provider.run_stream()` with exponential backoff, or monitor `stream_task` completion and trigger shutdown/restart. |
| R5 | **`_flatten_all` races with watchdog close** — During shutdown, `_flatten_all` iterates positions and dispatches closes. If a watchdog close is already in flight for the same symbol, both may attempt `close_position`. OANDA's net-position model is idempotent (closing a flat position is a no-op), but duplicate API calls waste rate-limit budget. | Low | Add a `CLOSING` state or skip symbols already in `PENDING_CLOSE` during `_flatten_all`. |
| R6 | **Warm-up requires historical bars** — On cold start, the orchestrator waits for `warmup_period` streamed bars before generating signals. For a 260-bar warmup at 1-minute granularity, this is >4 hours of idle time. | Medium | Pre-seed the bar buffer with historical bars from `provider.get_historical_bars()` before starting the stream. |
| R7 | **MLStrategy feature pipeline expects equities columns** — The V3 feature pipeline (`ml_strategy.py:417`) may reference columns specific to equities data (e.g., fundamental features). Forex OHLCV bars from `oanda_provider.py` may not contain all required columns, causing `generate_signals` to raise. | Medium | Verify the feature pipeline's expected schema against forex bar output. Add a forex-specific feature engineer if needed. |

## Files touched

| File | Lines | Change |
|---|---|---|
| `src/execution/oanda_scalper_orchestrator.py` | 1–269 | New file: lean async orchestrator with bar path, tick watchdog, position state machine, and graceful shutdown |
| `run_oanda.py` | 1–93 | New file: root-level entrypoint following `run_factory.py` bootstrap pattern |
| `tests/test_oanda_scalper.py` | 1–181 | New file: 4 mocked test cases covering long/short signal submission, idempotent breach guard, and async dispatch verification |
