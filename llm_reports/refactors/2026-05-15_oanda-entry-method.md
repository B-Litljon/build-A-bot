---
type: refactor
date: 2026-05-15
time: 14:50 PDT
agent: Kimi K2.6
model: kimi-k2.6
trigger: Strike 1 — add submit_target_position() entry/reversal path to OandaOrderManager
head: b43e8f1
scope: modifies-source
related:
  - refactors/2026-05-10_oanda-integration.md
files_touched:
  - src/execution/oanda_order_manager.py
  - tests/test_oanda_entry.py
---

# OANDA Entry Path — submit_target_position()

## Context

The V5 forex scalper pivot is missing the entry/reversal method in `OandaOrderManager`. The class already tracks net positions and can `close_position()`; this strike adds `submit_target_position()` so the strategy can move from flat to long, long to short, or any signed delta in a single broker call.

Architect constraints (frozen contract):
- Single method signature: `submit_target_position(symbol: str, target_units: int) -> dict`
- Signed delta market order via `oandapyV20.endpoints.orders.OrderCreate`
- No native stopLossOnFill / takeProfitOnFill (software SL/TP only)
- Parse `orderFillTransaction` for ACTUAL filled units; never trust requested values
- State mutation only after successful fill parsing; API errors leave state untouched
- OANDA US/NFA netted accounts: trust broker-side FIFO, do not implement client-side lot tracking

## Investigation

Read `src/execution/oanda_order_manager.py` end-to-end to mirror existing `_state_lock` discipline and fill-parsing patterns from `close_position()` (lines 150–225).

Key observations:
- `close_position` acquires `_state_lock` twice: once to read `net`, again after the API call to mutate state using the ACTUAL fill (`total_filled = units_l + units_s`).
- `_avg_entry_prices` is cleared only when the resulting net is exactly zero.
- `get_net_position` normalizes the symbol internally, so `submit_target_position` can safely call it with the raw symbol.

Verified `oandapyV20` is available in the pipenv environment (`pipenv --venv` returns a valid path). Existing tests (`tests/test_execution_safety.py`) use `@patch("oandapyV20.API")` and patch endpoint classes (`PositionClose`) to inject mock response dicts — pattern copied exactly.

## Findings / Changes

### `src/execution/oanda_order_manager.py`

1. **Added import** (`line 24`):
   ```python
   import oandapyV20.endpoints.orders as v20_orders
   ```

2. **Added `submit_target_position`** (lines 228–319):
   - Normalizes symbol via `_to_oanda_symbol()`.
   - Acquires `_state_lock` to compute `delta = target_units - current_net`.
   - Early no-op return when `delta == 0` (zero API calls, zero state changes).
   - Submits `MARKET` order with `units=str(delta)` and `instrument=oanda_symbol`. No SL/TP attached.
   - Parses `orderFillTransaction` for:
     - `fill_units` (signed, actual filled amount)
     - `fill_price`
     - `tradesClosed` list → sum of absolute closed units
     - `tradeOpened` dict → absolute opened units
   - Second `_state_lock` acquisition to update state:
     - `new_net = old_net + fill_units` (robust for partial fills, closes, opens, reversals)
     - Clears `_avg_entry_prices` when `new_net == 0`
     - **Adding to same-direction position**: weighted average of old avg and opened-leg price
     - **Reduction**: preserves old average price
     - **Reversal / fresh open**: uses `tradeOpened.price` (or `fill_price` fallback)
   - Returns `{'filled': total_filled, 'avg_price': fill_price, 'closed_units': closed_units, 'opened_units': opened_units}`
   - Exception handler logs ERROR and returns zero-dict without mutating state.

### `tests/test_oanda_entry.py` *(new)*

Five mocked test cases using `unittest.mock.patch` on `oandapyV20.API` and `oandapyV20.endpoints.orders.OrderCreate`:

| Test | Scenario | Asserts |
|------|----------|---------|
| `test_flat_to_long` | Flat → 100 long | `filled=100`, `opened_units=100`, state net=100, avg=1.085 |
| `test_long_to_short_reversal` | Long 100 → Short 50 (delta −150) | `filled=150`, `closed_units=100`, `opened_units=50`, state net=−50, avg=1.09 |
| `test_delta_zero_noop` | Target equals current net | `OrderCreate` never called, state untouched |
| `test_api_error_leaves_state` | `request()` raises Exception | Returns zero-dict, net and avg unchanged |
| `test_add_to_existing_position_weighted_avg` | Long 100 → Long 200 at higher price | Weighted average = 1.085 computed correctly |

## Verification

```bash
$ pipenv run python -m unittest tests.test_oanda_entry -v
Ran 5 tests in 0.004s
OK

$ pipenv run python -m unittest tests.test_execution_safety -v
Ran 3 tests in 0.005s
OK
```

Both new and existing OANDA tests pass. No regressions in `close_position` fill parsing or partial-fill behavior.

## Risk & follow-ups

1. **Partial fill / IOC handling** — The current implementation uses a plain `MARKET` order (OANDA defaults to IOC). If a partial fill occurs, `fill_units` will reflect the actual filled amount and `new_net` will correctly represent the residual. However, the unfilled portion of the delta will NOT be automatically re-submitted. The caller (V5 orchestrator) must detect `new_net != target_units` and retry if desired.
2. **FOK explicit flag** — If the strategy later requires all-or-nothing semantics, add `"timeInForce": "FOK"` to `order_data["order"]`. The current code already handles FOK rejects gracefully (no `orderFillTransaction` → error return, no state mutation).
3. **Average price on reduction** — When OANDA reports `tradeReduced` (partial close, same direction), the code keeps the old average price. This is correct for a single netted trade because FIFO closes oldest lots first, and with only one trade the remaining units retain the same average. If OANDA ever splits the position into multiple trades on a netted account, this assumption may need revisiting.
4. **Watchdog / software SL-TP** — `submit_target_position` is stateless about stops. The V5 scalper still needs a watchdog module that monitors price and calls `close_position()` when software SL/TP levels are breached. (Flagged in `refactors/2026-05-10_oanda-integration.md` as R3.)

## Files touched

- `src/execution/oanda_order_manager.py` — added `import v20_orders` (line 24) and `submit_target_position()` method (lines 228–369)
- `tests/test_oanda_entry.py` — new file, 172 lines, 5 test cases
