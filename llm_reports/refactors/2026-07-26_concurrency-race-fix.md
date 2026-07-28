---
type: refactor
date: 2026-07-26
time: 22:28 PDT
agent: Claude Fable 5
model: claude-fable-5
trigger: Brief "fix async bug" — thread-safety violations in live_orchestrator break the SymbolContext ownership contract
head: 23ea1da52bcfdcd103777d358e411645f2af5bc8
scope: modifies-source
files_touched:
  - src/execution/live_orchestrator.py
  - tests/execution/test_live_orchestrator.py
---

# Concurrency race fix — restore the SymbolContext ownership contract in live_orchestrator

## Context

`src/execution/live_orchestrator.py` (the Alpaca dual-stream scalper) runs an
asyncio event loop that owns all mutable trading state and offloads CPU-bound
inference and REST order submission to the thread pool via `asyncio.to_thread`.
The `SymbolContext` docstring stated the ownership contract: worker threads
only read the immutable `history_df` snapshot and never mutate `SymbolContext`.
The code violated that contract in four places (plus one the brief missed).
The brief demanded the contract be restored via **message passing**
(snapshot in, frozen result out, event loop applies all mutations) — no locks.

All findings were re-verified against the working tree at HEAD `23ea1da`
before editing; every line number in the brief matched exactly.

## Investigation

Line references below are **pre-fix** (at `23ea1da`) unless marked "now".

### Race 1 — `ctx.htf_cache`: two unsynchronized writers on different threads (core race)

- `_run_inference` (thread pool, dispatched at 1017) read the cache at
  1066–1073 and **wrote** it on the cold path at 1094
  (`ctx.htf_cache = HTFCache.from_features_df(...)`).
- `_prime_htf_cache` wrote it **from the event loop** at 1880 (call sites
  2007 daemon path, 2137 interactive path).
- The warm path did five separate `ctx.htf_cache.X` dereferences (1103,
  1109–1114). `HTFCache` was a mutable dataclass.

**Failure scenario:** the loop swaps `ctx.htf_cache` between the warm path's
first dereference (`sealed_at` for the debug log) and the last
(`htf_bb_pct_b`). The feature row then contains HTF scalars from **two
different 5-minute epochs** — e.g. `htf_rsi_14` from the old seal,
`htf_bb_pct_b` from the new one. The model silently scores a feature vector
that never existed in any market state; no exception, no log, just corrupted
inference on the exact bars where the cache rolls over (once every 5 minutes,
i.e. the moments most likely to coincide with regime shifts). The reverse
interleaving (thread's stale cold-path write landing after the loop's fresh
prime) reinstalls an expired cache and poisons every warm-path bar until the
next 5-minute boundary.

### Race 2 — `ctx.last_price`: thread overwrites fresh data with stale data

- The loop wrote the freshest tick at 979 (`ctx.last_price = float(bar.close)`)
  — this is the price `_universal_watchdog_loop` (1678–1679) uses to decide
  SL/TP exits.
- `_run_inference` later overwrote it at 1178 with the **sealed bar's** close —
  up to a minute stale by the time inference finishes.

**Failure scenario:** price drops through the SL between the bar sealing and
the inference thread finishing. The loop's tick handler has already stored
the fresh (breaching) price; the inference thread then overwrites it with the
sealed close from up to 60 s earlier, which is back above the SL. The
watchdog's next 1 s poll evaluates the stale price and does **not** exit; the
position stays open past its stop until the next tick arrives. On a fast move
that is real money lost strictly to the race.

### Race 3 — `ctx.sl_price`: order-submission thread mutates watchdog state

- `_submit_entry_order` (thread pool) grabbed the context via
  `ctx_ref = self._contexts[symbol]` and wrote the floored SL directly
  (1420–1421), racing loop-side writes made under `ctx.lock` (1309 signal
  path, 1331 rollback path).

**Failure scenarios:** (a) the watchdog briefly monitors the **unfloored** SL
(written at 1309) while the thread is still working — an SL exit can fire at
the pre-floor level the position was *not* sized against; (b) if the
submission ultimately fails, the loop's rollback (`ctx.sl_price = None` at
1331) can interleave with the thread's write, leaving a stale SL on a FLAT
symbol; (c) the write bypassed `ctx.lock` entirely, so no ordering guarantee
existed at all.

### Race 4 (minor) — thread-side dashboard writes

`ctx.last_atr` (1179) and `ctx.last_conviction` (1226) written from the
inference thread. Cosmetic state, same contract violation.

### Race 5 — **not in the brief**: `ctx.entry_qty` + thread-side `_save_state()`

`_submit_entry_order` also did (1522–1524):

```python
ctx = self._contexts[symbol]
ctx.entry_qty = qty
self._save_state()
```

That is a thread-side `SymbolContext` write **and** a thread-side
`_save_state()` call that iterates every context's `state`/`sl_price`/
`tp_price`/`entry_qty` while the loop may be mutating them. Fixed with the
same pattern (qty travels back in the result; the loop applies it and calls
`_save_state()`).

### Confirmed already-correct (untouched)

- All `ctx.state` transitions: loop-only, under the per-symbol `asyncio.Lock`.
- The `history_df` clone at 1005 (now 1044).
- `_contexts` dict: never structurally modified after `__init__`.
- `_prime_htf_cache`: runs on the loop during warm-up — a legitimate owner
  write, unchanged.

## Findings / Changes

All changes in `src/execution/live_orchestrator.py`; design is message-passing,
zero new locks, zero behavior change on non-racy paths.

### Why message passing instead of locking

A `threading.Lock` around `htf_cache` would fix tearing but not staleness
(last-writer-wins between the thread's old-epoch cache and the loop's fresh
prime is still wrong), would have to be held across the warm path's five
dereferences, and would put a blocking primitive inside the event loop's
callbacks. Making the thread functions pure — snapshot in, frozen result out,
single owner applies mutations — removes the shared-mutable-state problem
instead of managing it, is trivially testable (see the regression test), and
matches the architecture the docstring already claimed.

### New frozen result types (lines 303–332 now)

```python
@dataclass(frozen=True)
class InferenceOutcome:
    signal: Optional[Signal] = None
    new_htf_cache: Optional[HTFCache] = None
    last_atr: Optional[float] = None
    last_conviction: Optional[float] = None

@dataclass(frozen=True)
class EntryOrderResult:
    success: bool
    adjusted_sl: Optional[float] = None
    submitted_qty: Optional[float] = None
```

`HTFCache` itself is now `@dataclass(frozen=True)` (line 231) so the snapshot
crossing the thread boundary is immutable; a repo-wide grep confirmed no code
ever mutated an instance's fields (all refreshes construct new instances).

### `_on_bar` (loop) — snapshot before, apply after

Before:
```python
signal_result: Optional[Signal] = await asyncio.to_thread(
    self._run_inference, symbol, history_snapshot
)
if signal_result is None:
    return
await self._handle_signal(ctx, signal_result)
```

After (1055–1075 now):
```python
htf_snapshot: Optional[HTFCache] = ctx.htf_cache
outcome: InferenceOutcome = await asyncio.to_thread(
    self._run_inference, symbol, history_snapshot, htf_snapshot
)
if outcome.new_htf_cache is not None:
    ctx.htf_cache = outcome.new_htf_cache
if outcome.last_atr is not None:
    ctx.last_atr = outcome.last_atr
if outcome.last_conviction is not None:
    ctx.last_conviction = outcome.last_conviction
if outcome.signal is None:
    return
await self._handle_signal(ctx, outcome.signal)
```

The `is not None` guards preserve the old per-path dashboard semantics
exactly: `last_atr` updates on angel-reject/devil-veto/signal paths but not on
kill-switch/null-row/error paths; `last_conviction` only on the signal path.

### `_run_inference` (thread) — now a pure function

- New signature: `(self, symbol, history_df, htf_cache: Optional[HTFCache])
  -> InferenceOutcome` (1082 now).
- The `ctx = self._contexts[symbol]` lookup is gone; every `ctx.htf_cache`
  read/write became the local `htf_cache` parameter / `new_htf_cache` local.
- `new_htf_cache` is initialized **before** the `try` (1105 now) and carried
  on **every** return path — null-row filter, kill switch, angel reject,
  devil veto, signal, and the exception handler — so a cold-path recompute is
  never lost to an early return and the warm-path optimization survives
  rejects (the brief's critical detail #3).
- The `ctx.last_price = current_price` write was **deleted outright** (the
  loop's tick-time write is always fresher — intentional per the brief; this
  also fixes Race 2 by removing the stale writer entirely).
- `ctx.last_atr` / `ctx.last_conviction` writes became outcome fields.

### `_handle_signal` (loop) — applies the order result

Before: `success = await asyncio.to_thread(_submit_entry_order, ...)` with
only a failure branch. After (1389–1413 now):

```python
result: EntryOrderResult = await asyncio.to_thread(
    self._submit_entry_order, sig, client_order_id,
)
if result.success:
    async with ctx.lock:
        if result.adjusted_sl is not None:
            ctx.sl_price = result.adjusted_sl
        ctx.entry_qty = result.submitted_qty
    self._save_state()
else:
    # unchanged rollback-to-FLAT branch
```

### `_submit_entry_order` (thread) — returns instead of writing

- Return contract changed `bool` → `EntryOrderResult`.
- The SL-floor branch sets `adjusted_sl = sl_price` (1507 now) instead of
  writing `self._contexts[symbol].sl_price` (Race 3 fixed).
- The `ctx.entry_qty = qty` + `_save_state()` block was removed (Race 5
  fixed); qty travels back as `submitted_qty`.
- All local sizing/guard logic, logging, Discord alerts, and the
  `sig.metadata` enrichment are unchanged (the Signal object is only touched
  by one thread at a time by construction, so metadata writes are safe).

### Docstrings

`SymbolContext`'s contract (354 now) rewritten to match reality: every field
loop-owned; threads get snapshots and return frozen records; thread functions
must never hold a `SymbolContext` reference. `_run_inference`,
`_submit_entry_order`, `HTFCache`, and the module-header diagram updated to
the new contracts.

## Sibling audit

### `src/execution/oanda_scalper_orchestrator.py` — **no violation; no changes**

Different architecture from the brief's premise: it has no `_contexts` /
`SymbolContext` at all, and inference (`generate_signals`, line 497) runs
**on the event loop**, not in a thread. Its genuinely cross-thread state is
handled correctly:

- `_positions` is accessed under `self._positions_lock` (a `threading.Lock`)
  at every site on both the tick thread (`_on_tick`, 218–244) and the loop
  (`_on_bar` 513/583, `_watchdog_close` 375/402/421, `_flatten_all` 953/993).
- `_latest_spread` / `_latest_spread_ts` are written lock-free from the tick
  thread by explicit documented design (single writer, GIL-atomic dict
  assignment, staleness-checked reader) — acceptable, not a tear risk.
- Blocking broker calls go through `run_in_executor` into
  `OandaOrderManager`, which guards all of its own state with an internal
  `threading.RLock` (`_state_lock`, oanda_order_manager.py:80, held at every
  mutation site).

This is the module the currently-running M15 soak executes — deliberately
left byte-identical so a watchdog restart cannot pick up unreviewed code.

### `src/execution/factory_orchestrator.py` — **no violation; no changes**

All mutable orchestrator state (`active_positions`, `aggregators`) lives on
the loop; the only thread-side work is `strategy.generate_signals` over a
cloned history (line 97–101) and REST calls. The thread receives a snapshot
and returns a signal — already the message-passing shape.

One shared-object caveat spanning both siblings: `MLStrategy` mutates itself
during `generate_signals` (hot-reload of models/threshold/spread-table, and
per-symbol heartbeat counters). The hot-reload writes are guarded by its own
`_reload_lock` (ml_strategy.py:99); heartbeat state is keyed per symbol, and
in the factory orchestrator concurrent `generate_signals` calls only occur
for **different** symbols. Not a defect today; noted so nobody adds a shared
per-strategy accumulator without checking the factory's threaded call path.

## Verification

Environment: project virtualenv
`~/.local/share/virtualenvs/build-A-bot-A3hTUWzK` (`pytest 9.0.3`),
`PYTHONPATH=src:.`.

- **Baseline (before any edit):** `pytest -q` → **112 passed**, 0 failed.
  No pre-existing failures.
- **After the fix:** `pytest -q` → **114 passed**, 0 failed (112 original +
  2 new).

New regression tests in `tests/execution/test_live_orchestrator.py`
(`TestThreadOwnership`):

1. `test_full_signal_cycle_writes_only_on_loop_thread` — patches
   `SymbolContext.__setattr__` to record `(attribute, thread_ident)` for
   every write, then drives the **real** `_on_bar → _run_inference →
   _handle_signal → _submit_entry_order` cycle (models, feature pipeline,
   and Alpaca REST mocked at the I/O boundary; synthetic features chosen so
   the MIN_SL_PCT floor fires and the cash-cap sizing path runs). Asserts
   every recorded write happened on the event-loop thread, plus a vacuity
   guard that the cycle actually wrote all eight previously-racy/loop fields
   (`last_price`, `state`, `htf_cache`, `last_atr`, `last_conviction`,
   `sl_price`, `tp_price`, `entry_qty`), the floored SL landed
   (`ctx.sl_price == 99.85`), and `_save_state` ran once.
2. `test_htf_cache_propagates_on_angel_reject` — same harness with the Angel
   rejecting: asserts no order is submitted **and** the cold-path HTF cache
   still reaches `ctx` (the brief's "warm-path optimization must survive
   early returns" requirement), `last_atr` updated, `last_conviction` not.

**Negative proof:** a deliberate thread-side write
(`self._contexts[symbol].last_atr = -1.0`) was temporarily injected into
`_run_inference`; both ownership tests failed with the offending
attribute/thread listed, confirming the instrumentation detects regressions.
The injection was removed and the full suite re-run green (114 passed).

Nothing was committed or pushed; the working tree is left for review.

## Risk & follow-ups

Items found during the work that the brief missed (none fixed beyond #1,
which was in-pattern):

1. **Fixed, flagged:** the thread-side `ctx.entry_qty` write + `_save_state()`
   call in `_submit_entry_order` (Race 5 above) — the brief's list stopped at
   `sl_price`.
2. **Pre-existing, not fixed — persistence gap:** `_save_state()` only
   serializes `IN_TRADE` symbols, but it is only *called* at entry-submission
   time (symbol still `PENDING` → excluded from the payload) and at cooling
   (symbol removed). The `PENDING → IN_TRADE` fill transition in
   `_on_trade_update` never calls `_save_state()`, so a freshly filled trade
   is typically **absent** from `active_trades.json` until some other
   symbol's entry/exit triggers a save. A restart mid-trade would then not
   restore SL/TP for it. One-line fix (call `_save_state()` after the BUY-fill
   transition), but it is a behavior change outside this brief's scope.
3. **Pre-existing, not fixed — benign qty race:** the pre-seeded
   `entry_qty = submitted_qty` can overwrite the authoritative `filled_qty`
   if the fill event is processed before `_handle_signal` resumes after the
   submission `await`. This exact hazard existed before the refactor (the
   thread's write could land after the fill handler's); the fix preserves the
   old semantics deliberately. Now that both writers are on the loop, a
   `if ctx.state == SymbolState.PENDING` guard around the pre-seed would
   close it cleanly — cheap follow-up.
4. **Pre-existing, not fixed — cross-thread deque append:**
   `_log_activity` is called from thread-pool functions
   (`_submit_entry_order`, `_submit_manual_exit`) and appends to
   `self._activity_log` while the dashboard loop iterates it; a
   `RuntimeError: deque mutated during iteration` is theoretically possible
   in interactive mode. Cosmetic-path only; the clean fix is returning
   activity messages in the result objects too.
5. **Operational note:** the report path requested by the brief
   (`llm_reports/2026-07-26_concurrency_race_fix.md`, repo root) was adjusted
   to `llm_reports/refactors/2026-07-26_concurrency-race-fix.md` per the
   folder's README convention (category folder, kebab-case).

## Files touched

- `src/execution/live_orchestrator.py` — module docstring diagram (~22);
  `HTFCache` frozen + docstring (231–258); new `InferenceOutcome` /
  `EntryOrderResult` dataclasses (303–332); `SymbolContext` ownership
  docstring (350–360); `_on_bar` snapshot/apply (1055–1075);
  `_run_inference` signature, purity, per-path outcome returns (1082–1345);
  `_handle_signal` result application (1389–1413); `_submit_entry_order`
  return-contract change and removal of all context writes (1420–1631).
- `tests/execution/test_live_orchestrator.py` — imports + `_make_ctx`
  extension (dashboard/htf fields); new helpers `_make_history_df`,
  `_make_features_df`, `_make_inference_orch`; new `TestThreadOwnership`
  class (2 tests) at end of file.

Read (audit, unmodified): `src/execution/oanda_scalper_orchestrator.py`,
`src/execution/factory_orchestrator.py`, `src/execution/oanda_order_manager.py`
(lock sites), `src/strategies/concrete_strategies/ml_strategy.py`
(reload-lock scope, heartbeat state).
