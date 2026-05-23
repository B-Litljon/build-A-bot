---
type: refactor
date: 2026-05-20
time: 14:50 PDT
agent: Gemini CLI (Chief Architect, 3.5 Flash)
model: gemini-3.5-flash
trigger: implement stream retry/backoff loop on disconnect
head: d92202f10568b363794c6faaecc699c42ba744dc
scope: modifies-source
files_touched:
  - src/execution/oanda_scalper_orchestrator.py
  - src/data/oanda_provider.py
---

# Stream Resiliency retry/backoff loop

## Context

The 4-hour 2026-05-20 soak on `feature/v5-history-prime` successfully fired ~25 trades, confirming execution mechanics. However, it disconnected due to `OandaMarketProvider stream error: Response ended prematurely`—plausibly a flow-rate soft limit. 

This work implements **Strike 3** to wrap the OANDA pricing stream in a retry-with-backoff loop. On disconnect, the orchestrator sleeps for 5s, resets the seam state, re-primes historical bars, and resumes the stream.

## Investigation

In `src/data/oanda_provider.py` (lines 357-358), disconnects and general stream errors were captured by a broad `except Exception as e:` block and logged as `OandaMarketProvider stream error`. This swallowed the exception and completed the thread pool executor cleanly, causing the daemon to run indefinitely with a dead stream.

To enable the orchestrator to detect disconnects, `run_stream` must propagate exceptions up to the thread wrapper.

In `src/execution/oanda_scalper_orchestrator.py` (lines 387-389), the stream was launched via a one-shot `asyncio.to_thread` call:
```python
self._stream_task = asyncio.create_task(
    asyncio.to_thread(self._provider.run_stream)
)
```
This task is now wrapped in a loop coroutine `_stream_with_retry`.

## Findings / Changes

### `src/data/oanda_provider.py`
- Removed the broad `except Exception` block in `run_stream` so disconnects and other runtime exceptions propagate upward.
- Maintained the `KeyboardInterrupt` catch so user-initiated shutdowns exit cleanly.

### `src/execution/oanda_scalper_orchestrator.py`
- Defined `_stream_with_retry` coroutine.
- Wraps `await asyncio.to_thread(self._provider.run_stream)` in a `while not self._shutdown_event.is_set():` loop.
- Catches stream exceptions, waits 5s, clears internal bar buffers, clears seam crossing flags, and triggers `_prime_history()` to backfill any missed bars during disconnect.
- Swapped the one-shot task creation in `run()` to invoke `self._stream_with_retry()`.

## Verification

### 1. Compile Checks
Verified that both modified files compile cleanly without any syntax errors:
```bash
python3 -m py_compile src/execution/oanda_scalper_orchestrator.py src/data/oanda_provider.py
```

### 2. Smoke Launch
Tested start-up, historical priming, and OANDA stream subscriptions using the main daemon command:
```bash
pipenv run python3 run_oanda.py --daemon
```

## Risk & follow-ups

- **Inverted R:R / Spread bleeding:** Under M1 granularity, a 0.5x ATR stop loss (typically ~0.5 pips) sits inside the spread. This soak exercises mechanics (orders open and close rapidly) but will lose spread. A pivot from M1 to M5 is planned as a separate refactor to escape the sub-spread stop trap.
- **Retry Backoff:** If OANDA is down for an extended period, the bot will retry every 5 seconds. If necessary, a future refactor can implement exponential backoff.

## Files touched

- [oanda_scalper_orchestrator.py](file:///home/tha_magick_man/mystuf/development/build-A-bot/src/execution/oanda_scalper_orchestrator.py#L365-L395)
- [oanda_provider.py](file:///home/tha_magick_man/mystuf/development/build-A-bot/src/data/oanda_provider.py#L349-L359)
