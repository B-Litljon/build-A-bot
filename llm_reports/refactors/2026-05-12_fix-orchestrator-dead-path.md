---
type: refactor
date: 2026-05-12
time: 22:05 PDT
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: Re-do of Finding #1.1 fix from the 2026-05-10 audit after the prior Gemini attempt introduced a SELL partial_fill regression
head: df269198fbde5b1a9cbb12f13ef0f381942c2735
scope: modifies-source
related:
  - audits/2026-05-10_v5-orchestrator-dead-path.md
files_touched:
  - src/execution/live_orchestrator.py
  - tests/execution/__init__.py
  - tests/execution/test_live_orchestrator.py
---

# Fix Orchestrator Dead Path (Finding #1.1)

## Context

The 2026-05-10 audit flagged a critical bug in `_on_trade_update`: a second
`elif event_type == "fill":` branch was shadowed by an earlier
`if event_type in ("fill", "partial_fill"):` block. Watchdog-driven SELL fills
arrived with `event_type == "fill"`, matched the first branch, but failed its
inner `ctx.state == SymbolState.PENDING` check (state was `PENDING_EXIT`), so
the symbol stayed stuck in `PENDING_EXIT` forever — never reaching `COOLING`
or returning to `FLAT`.

A prior Gemini attempt at this fix (now reverted) merged the two branches but
allowed `partial_fill` SELL events to call `_enter_cooling` directly. That
created a fresh regression: the first partial SELL fill would null out
`entry_price`/`entry_qty`/`sl_price`/`tp_price` and persist `COOLING` state
to disk while the broker still had open quantity for the same order.

## Investigation

Reading `src/execution/live_orchestrator.py:1575-1632` at `HEAD` confirmed
the shadow pattern from the audit:

- `:1578` — `if event_type in ("fill", "partial_fill"):` (BUY entry path,
  guarded by `state == PENDING`)
- `:1625` — `elif event_type == "fill":` (SELL → COOLING path, unreachable
  because `fill` already matched at `:1578`)

Cross-checked `_enter_cooling` at `:1770` to verify its contract: it sets
`state = COOLING`, nulls `entry_price`/`entry_qty`/`sl_price`/`tp_price`,
persists via `_save_state`, and schedules a cooling timer. The docstring
requires the caller to hold `ctx.lock`, which the new SELL branch does.

Also reviewed `_handle_signal` at `:1296` to understand the dedup gate that
makes commit 2 (separate report) necessary — that change is intentionally
not in this commit.

## Findings / Changes

### `src/execution/live_orchestrator.py:1575-1614` (replaced 1575-1632)

Merged the two fill branches into one. Within the unified block, dispatch on
`order_side` and `event_type`:

- `BUY` + `state == PENDING` → `PENDING → IN_TRADE` (unchanged behavior; the
  `state == PENDING` guard still makes second-partial BUY fills no-op).
- `SELL` + `event_type == "fill"` + `state in (IN_TRADE, PENDING_EXIT)` →
  `await self._enter_cooling(ctx)`. **`partial_fill` is intentionally excluded**
  so a partial sell does not drop us into `COOLING` while the remainder is
  still working at the broker.
- `SELL` + `event_type == "partial_fill"` → log only
  (`"Partial SELL fill — awaiting terminal fill"`), no state transition.

The unreachable `elif event_type == "fill":` at the previous `:1625-1632` is
removed.

### `tests/execution/__init__.py` (new)

Empty package marker so `python -m unittest tests.execution.…` works.

### `tests/execution/test_live_orchestrator.py` (new)

Five state-machine tests built on `unittest.IsolatedAsyncioTestCase` (the
convention already in use by `tests/verify_warmup.py`; pytest is not
installed in the project's pipenv). `SymbolContext` and `LiveOrchestrator`
are constructed via `__new__` to skip their heavy `__init__` chains;
`_enter_cooling`, `_log_activity`, and `_notifier` are mocked so the tests
exercise the state machine in isolation.

- `test_sell_fill_pending_exit_enters_cooling` — the audit-finding repro.
- `test_sell_fill_in_trade_enters_cooling` — symmetric coverage for the
  bracket-TP/SL path.
- `test_sell_partial_fill_does_not_enter_cooling` — guards the regression
  introduced by the prior Gemini attempt.
- `test_sell_fill_then_partial_does_not_double_cool` — defense in depth: a
  late `partial_fill` arriving after the order has cooled must not call
  `_enter_cooling` again. Exercises the actual state gate (manually
  transitions `ctx.state = COOLING` after the mocked first call) rather
  than relying on `assert_called_once`.
- `test_buy_fill_pending_enters_in_trade` — no-regression check on the BUY
  entry path.

## Verification

```
$ pipenv run python -m unittest tests.execution.test_live_orchestrator -v
...
Ran 5 tests in 0.010s
OK
```

All five tests pass against the new implementation. Manual diff review
confirmed the unreachable `elif` block is removed and both BUY/SELL fills
are dispatched from a single top-level conditional.

The architect-AI brief specified `pytest tests/execution/test_live_orchestrator.py -v`,
but the project's pipenv does not have pytest installed and `Pipfile`'s
`[dev-packages]` section is empty. Rather than introduce a dependency, the
tests use stdlib `unittest`, matching `tests/verify_warmup.py`. To run them:
`pipenv run python -m unittest tests.execution.test_live_orchestrator -v`.

## Risk & follow-ups

- **Risk:** Low for the stated bug — the fix is a one-condition split.
  Partial-fill handling for BUY remains pre-existing: the first BUY partial
  transitions `PENDING → IN_TRADE` and sets `entry_qty` to the partial qty;
  subsequent partials no-op because the inner `state == PENDING` guard
  fails. This means `entry_qty` can under-count the eventual filled total
  on the BUY side. That latent issue predates this work and is out of scope
  for finding #1.1; flagging it here for a future report.
- **Follow-up 1:** Commit 2 (separate report
  `2026-05-12_fix-orchestrator-dedup-after-reject.md`) clears
  `ctx.last_client_order_id` in the cancel/expire/reject branch so a retry
  signal isn't silently deduplicated. The prior Gemini attempt smuggled
  that change into this fix; splitting it out keeps each commit's intent
  legible.
- **Follow-up 2:** Finding #1.2 (OANDA timestamp skew) is the next
  high-priority item from the 2026-05-10 audit.

## Files touched

- `src/execution/live_orchestrator.py:1575-1614` — unified fill handler;
  removed unreachable SELL branch at previous `:1625-1632`.
- `tests/execution/__init__.py` — new (empty).
- `tests/execution/test_live_orchestrator.py` — new; 5 state-machine tests.
