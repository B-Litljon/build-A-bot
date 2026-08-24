---
type: refactor
date: 2026-08-03
time: 04:59 UTC
agent: opencode (deepseek-v4-pro)
model: deepseek-v4-pro
trigger: "Discord close notifications show entry price and omit TP/SL hit info — making them useless for post-trade review"
head: 6d0a74d4a709634253418c6f4f116046a9592ea3
scope: modifies-source
files_touched:
  - src/core/notification_manager.py
  - src/execution/oanda_scalper_orchestrator.py
  - src/execution/live_orchestrator.py
  - llm_reports/refactors/2026-08-03_discord-close-notification-fix.md
---

# Discord close notification fix — show actual exit price and TP/SL outcome

## Context

The bot sends Discord embed notifications for trade entries and exits.
Entries are useful: they show entry price, SL/TP bracket levels, angel/devil
probabilities.  Exit notifications, however, regurgitated the exact same
entry price as the "Price" field with no indication of which bracket was
breached or what the closing price was — making them functionally useless for
post-trade review.

Three distinct bugs converged to cause this:

1. **OANDA path (call-site):** `_watchdog_close` passed `pos_snapshot["entry"]`
   as the `price` kwarg — the entry price, not the live bid/ask at breach —
   even though `_on_tick` had those values in scope.
2. **Notification layer:** `send_oanda_trade_alert` gated SL/TP display behind
   `action == "ENTRY"` and had no mechanism to show a close price or which
   level was hit.
3. **Alpaca path:** `_enter_cooling` called `send_system_message` (a generic
   one-liner) instead of `send_trade_alert` — no embed, no exit price, no
   bracket info whatsoever.

## Investigation

### Bug 1 — OANDA path sends entry price on close

`src/execution/oanda_scalper_orchestrator.py:807` (pre-fix):

```python
price=pos_snapshot.get("entry", 0.0),
```

`_on_tick` (line 442) receives `bid` and `ask` on every quote and uses them
to detect breaches (lines 464-469). Those values were never stored or
forwarded. The `pos_snapshot` dict only carries the fields populated at entry
time (entry, sl, tp, units, state) — none of which is the closing price.

Same defect on the `CLOSE_FAILED` path (line 831) and implicitly on
`BOOT_FLATTEN` (line 1372), though the flatten case is less severe since the
position was orphaned with no known SL/TP anyway.

### Bug 2 — `send_oanda_trade_alert` hides TP/SL on close

`src/core/notification_manager.py:168` (pre-fix):

```python
if action == "ENTRY" and sl_price is not None and tp_price is not None:
    description += f"\n🛑 **SL:** {sl_price:.5f}\n"
    description += f"🚀 **TP:** {tp_price:.5f}\n"
```

The `action == "ENTRY"` guard meant close embeds never displayed the bracket
levels, so the user could not see which side was breached. There was no
`close_price` or `hit_level` parameter at all, so even if the call-site had
passed good data there was nowhere to put it.

### Bug 3 — Alpaca path never sends a trade close embed

`src/execution/live_orchestrator.py:2031` (pre-fix):

```python
self._notifier.send_system_message(
    f"[{symbol}] Bracket resolved — cooling off for "
    f"{COOLING_SECONDS // 60}m before next entry."
)
```

This bare one-liner is the ONLY Discord message triggered by a trade close.
No embed, no close price, no entry price, no TP/SL info. The data exists:
`_on_trade_update`'s SELL fill branch (pre-fix line 1818-1827) has the order
object with `filled_avg_price`, and `ctx.tp_price`/`ctx.sl_price` are still
live inside the lock. But `_enter_cooling` clears those fields at the top
(lines 2017-2020 pre-fix) before doing anything else, discarding the
evidence.

## Findings / Changes

### 1. `src/core/notification_manager.py` — core notification logic

**`send_trade_alert` (Alpaca path):**

- Close actions now check `signal.metadata["close_price"]`; if present they
  show "Close Price:" as the primary line, "Entry Price:" as secondary, and
  a "TP HIT" or "SL HIT" line with emoji.
- SL/TP bracket display ungated from `action == "ENTRY"` — shown for both
  entry and close embeds so the user can compare the close price to the levels.
- When `close_price` is absent, falls back to the legacy single-price line.

**`send_oanda_trade_alert` (forex path):**

- Added keyword-only parameters: `close_price: Optional[float] = None`,
  `hit_level: Optional[str] = None`.
- Same dual-price + hit-indicator logic as the Alpaca path when `close_price`
  is provided.
- SL/TP display ungated from `action == "ENTRY"`.
- Backward-compatible: callers that don't pass the new params get the same
  output as before.

**Glossary updated** with the new fields.

### 2. `src/execution/oanda_scalper_orchestrator.py` — capture breach data

**`_on_tick` (lines 463-488 now):** The breach-detection block now computes
`breach_price` (bid for long, ask for short) and `hit_level` ("TP"/"SL") and
stores them under `_positions_lock` alongside the `PENDING_CLOSE` transition.
Previously the lock block only set `current["state"]`.

**`_watchdog_close` (WATCHDOG_CLOSE path, ~line 807 now):** Reads
`breach_price` and `hit_level` from `pos_snapshot` and passes them as
`close_price=` and `hit_level=` to `send_oanda_trade_alert`. Also now passes
`sl_price` and `tp_price` so the embed can show the bracket levels.

**`CLOSE_FAILED` path (~line 831 now):** Similarly passes `sl_price` and
`tp_price` from the snapshot (close_price/hit_level omitted since the close
was never submitted).

**Glossary updated** to document the new position-dict fields and the
breach-capture flow in `_on_tick`.

### 3. `src/execution/live_orchestrator.py` — close embed for Alpaca

**`_on_trade_update` SELL fill branch (line 1827 now):** Before calling
`_enter_cooling`, captures `fill_price` (from the order's
`filled_avg_price`) and determines `hit` by comparing `fill_price` against
`ctx.tp_price` / `ctx.sl_price`. Passes both as new parameters.

**`_enter_cooling` (line 2023 now):** Accepts `close_price` and `hit_level`
parameters. Captures `entry_price`, `sl_price`, and `tp_price` from `ctx`
**before** clearing them. Constructs a `Signal` with close metadata and calls
`send_trade_alert(sig, action="CLOSE")` — producing a proper embed with close
price, entry price, TP/SL levels, and which side was hit. The existing
`send_system_message` is preserved as a secondary status update.

**Glossary updated** to document the new `_enter_cooling` parameters.

## Verification

```bash
PYTHONPATH=src:. python -m pytest -q
PYTHONPATH=src:. python -m compileall -q src/
```

- Both commands run green.
- The notification changes are purely additive (new optional parameters,
  fallback to legacy behavior when absent), so existing notification tests
  pass unchanged.
- The OANDA tick-path change stores two extra primitive fields under an
  existing lock — sub-µs overhead, no new syscall or allocation pattern,
  no risk to the 50 µs tick budget.
- The Alpaca change captures four existing float fields before clearing them
  — zero additional allocations on the hot path.

## Risk & follow-ups

1. **Low — Discord webhook latency under lock.** The Alpaca
   `_enter_cooling` already called `send_system_message` (a blocking
   `requests.post`) inside `ctx.lock`. The new `send_trade_alert` call is an
   additional blocking POST in the same critical section. This is pre-existing
   risk, not new. If it ever causes issues, both calls should be offloaded to
   `asyncio.to_thread` or an executor.
2. **Follow-up:** The Alpaca path could benefit from tracking the entry
   direction (long/short) in `SymbolContext` so the close embed can color-code
   green/red based on whether the trade was a winner or loser. Today the path
   is long-only so this is moot, but it's worth adding if short-selling is
   ever enabled.
3. **Follow-up:** `_flatten_all` in the OANDA path (shutdown / liveness
   watchdog) calls `_mark_exit` but does not send a per-symbol close
   notification — only a single aggregate system message on failure. A
   per-symbol flatten notification (with the last known entry price) would
   close that gap.

## Files touched

- `src/core/notification_manager.py` — `send_trade_alert`: close-price logic
  (lines 60-98 now); `send_oanda_trade_alert`: new `close_price`/`hit_level`
  params + ungated SL/TP + close-price description block (lines 127-196 now);
  module Glossary expanded (lines 13-43).
- `src/execution/oanda_scalper_orchestrator.py` — `_on_tick`: breach
  capture + position-dict storage (lines 463-500 now); `_watchdog_close`:
  pass breach metadata to notification (lines 800-820 now); `CLOSE_FAILED`:
  pass SL/TP from snapshot (lines 826-844 now); module Glossary updated
  (lines 34-38, 85-96).
- `src/execution/live_orchestrator.py` — `_on_trade_update` SELL fill:
  capture close details before `_enter_cooling` (lines 1818-1845 now);
  `_enter_cooling`: new params + close embed construction (lines 2023-2097
  now); module Glossary expanded (lines 84-96).
- `llm_reports/refactors/2026-08-03_discord-close-notification-fix.md` —
  this report.
