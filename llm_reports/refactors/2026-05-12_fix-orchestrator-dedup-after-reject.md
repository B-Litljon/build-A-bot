---
type: refactor
date: 2026-05-12
time: 22:10 PDT
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: Latent bug discovered while splitting Gemini's smuggled-in change out of the Finding #1.1 fix
head: 020d910104fde5523efcc32a8479b145009dcaa9
scope: modifies-source
related:
  - refactors/2026-05-12_fix-orchestrator-dead-path.md
files_touched:
  - src/execution/live_orchestrator.py
  - tests/execution/test_live_orchestrator.py
---

# Clear `last_client_order_id` After Terminal Rejection

## Context

While reviewing a prior Gemini-authored attempt at fixing Finding #1.1
(orchestrator dead path), an additional change was found bundled into that
commit: `ctx.last_client_order_id = None` inside the
`canceled`/`expired`/`rejected` branch of `_on_trade_update`. The change
was not part of the audit finding and not mentioned in Gemini's report
narrative.

On review the change is correct on its own merits, but mixing it with the
finding #1.1 fix obscured each commit's intent. This commit lands the
dedup-clear separately with its own rationale and test.

## Investigation

Traced the signal-submission path through `_handle_signal`:

- `src/execution/live_orchestrator.py:1294` constructs
  `client_order_id = f"{safe_sym}_{ts_iso}"` from `symbol` plus the signal
  timestamp formatted at second resolution.
- `:1296` — `if client_order_id == ctx.last_client_order_id:` returns early
  with a "Duplicate client_order_id — skipping" warning.

Without clearing `last_client_order_id` on a terminal rejection, the
sequence:

1. Signal at `T` produces `client_order_id = "BTCUSD_20260512T220300"`.
2. Order is rejected (e.g. insufficient buying power, broker maintenance).
3. State returns to `FLAT`, but `last_client_order_id` still holds the
   rejected order's id.
4. A retry signal that happens to produce the same `client_order_id`
   (same symbol, same second-resolution timestamp) is silently dropped at
   `:1296`.

In practice (`ts_iso` at second resolution) the collision window is
narrow, but it is reachable: a watchdog-driven re-entry attempt that
fires on the same second the prior order was rejected would be eaten by
the dedup gate.

## Findings / Changes

### `src/execution/live_orchestrator.py:1614-1616`

Inside the existing `canceled`/`expired`/`rejected` branch, after the
other field resets, added:

```python
# Clear so a same-bar retry signal isn't silently dropped by the
# dedup gate in _handle_signal.
ctx.last_client_order_id = None
```

The `state in (PENDING, IN_TRADE, PENDING_EXIT)` guard above this block
already restricts the reset to terminal events that actually had an order
attached, so this clear only happens when an entry was in flight.

### `tests/execution/test_live_orchestrator.py`

Added `test_rejected_clears_client_order_id`. Sets up a `SymbolContext`
in `PENDING` with `last_client_order_id` populated, fires a `rejected`
event, and asserts both `state == FLAT` and `last_client_order_id is None`.

## Verification

```
$ pipenv run python -m unittest tests.execution.test_live_orchestrator -v
...
Ran 6 tests in 0.010s
OK
```

All six tests (five from the prior commit plus the new one) pass.

## Risk & follow-ups

- **Risk:** Low. The change is a single field reset inside an existing
  state-guarded branch and only fires on terminal rejection events.
- **Follow-up:** No related work queued. Finding #1.2 (OANDA timestamp
  skew) remains next on the 2026-05-10 audit list.

## Files touched

- `src/execution/live_orchestrator.py:1614-1616` — added
  `ctx.last_client_order_id = None` inside the cancel/expire/reject branch.
- `tests/execution/test_live_orchestrator.py` — added
  `test_rejected_clears_client_order_id`.
