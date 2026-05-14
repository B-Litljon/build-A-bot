---
type: refactor
date: 2026-05-14
time: 01:30 PDT
agent: Gemini CLI
model: gemini-2.0-flash-thinking-exp
trigger: Directives to improve execution safety and thread-safety (Fixes 1.5, 1.7, 2.5)
head: c29ed0a4b6f15668c88a9a5eb67b919b9777177a
scope: modifies-source
related:
  - audits/2026-05-10_polars-and-logic.md
files_touched:
  - scripts/portfolio_orchestrator.py
  - src/execution/oanda_order_manager.py
  - tests/test_execution_safety.py
---

# Execution Safety and Thread-Safe Position Management

## Context
This update addresses critical safety concerns in the execution layer, specifically protecting manual trades from automated liquidation, ensuring accurate position tracking during OANDA fills, and preparing the execution state for concurrent access from async fill streams.

## Investigation & Rationale

### Finding 1.5: Manual Trade Protection
In `scripts/portfolio_orchestrator.py`, the `execute_rebalance` function was designed to liquidate any position not found in the `top_k` list. However, this logic was overly broad, as it would also close manual trades or long-term holdings not managed by the bot. 
**Fix:** Added a gate so that only symbols present in the bot's configured `UNIVERSE` are candidates for automated liquidation.

### Finding 1.7: Authoritative OANDA State
The `close_position` method in `src/execution/oanda_order_manager.py` previously assumed that a close request would always succeed in flattening the position entirely and immediately zeroed the local state. This is risky in the event of partial fills or API errors.
**Fix:** The logic now parses the `longOrderFillTransaction` and `shortOrderFillTransaction` objects in the OANDA response to determine the *actual* number of units filled, updating the local `_net_positions` incrementally.

### Finding 2.5: Thread-Safety for Async Fills
The V5 Forex Pivot requires a move toward asynchronous fill processing. The `OandaOrderManager` state (`_net_positions`, `_avg_entry_prices`) was not thread-safe, posing a risk of race conditions when the fill stream and the orchestrator access state concurrently.
**Fix:** Introduced `threading.RLock` to serialize all reads and mutations of the internal state.

## Findings / Changes

### `scripts/portfolio_orchestrator.py`
- Updated the list comprehension for `to_liquidate` to include an `s in UNIVERSE` check.
- Improved logging to reflect that only universe symbols are being considered for liquidation.

### `src/execution/oanda_order_manager.py`
- Added `import threading`.
- Initialized `self._state_lock = threading.RLock()` in `__init__`.
- Wrapped `get_net_position`, `get_average_entry_price`, `sync_position`, and `close_position` in `with self._state_lock:` blocks.
- Refactored `close_position` to extract `units` from the fill transactions in the OANDA response and apply the delta to the local state.

### `tests/test_execution_safety.py`
- New test suite added to verify:
    - Rebalance gate logic (manual trade protection).
    - OANDA fill parsing (correct state update on full fill).
    - OANDA partial fill behavior (correct residual state).

## Verification

Tests were executed using the project's `pipenv` environment:

```bash
$ pipenv run python tests/test_execution_safety.py
Loading .env environment variables...
...
Ran 3 tests in 0.015s

OK
```

## Risk & follow-ups
- **Risk:** Low. The changes are surgical and protected by unit tests. The use of `RLock` prevents deadlocks for re-entrant calls within the same thread.
- **Follow-up:** The next phase involves integrating the `oandapyV20` streaming endpoints to feed this thread-safe manager in real-time.

## Files touched
- `scripts/portfolio_orchestrator.py` - Gated liquidation logic.
- `src/execution/oanda_order_manager.py` - Added RLock and response parsing.
- `tests/test_execution_safety.py` - New unit tests.
