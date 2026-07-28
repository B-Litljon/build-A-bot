---
type: refactor
date: 2026-07-27
time: 13:40 PDT
agent: Claude Fable 5
model: claude-fable-5
trigger: "Brandon: fix the trade-persistence gap in live_orchestrator.py — three symptoms at one seam, adopt the ownership rule that whoever writes authoritative state persists it"
head: 3fa770c49c8f81aef2fdb289c9edae53e7b52f47
scope: modifies-source
related:
  - refactors/2026-07-26_concurrency-race-fix.md
  - refactors/2026-07-27_three-layer-glossary.md
files_touched:
  - src/execution/live_orchestrator.py
  - tests/execution/test_live_orchestrator.py
---

## Context

`_save_state()` serializes only symbols in `IN_TRADE` to `active_trades.json`;
`_load_state()` reads that file on startup and re-injects SL/TP for any symbol
Alpaca still shows open. **For crypto the software watchdog is the only exit
mechanism**, so a position missing from that file restarts with nothing
enforcing its stop.

Brandon's brief identified three symptoms at that one seam and specified the
remedy: an ownership rule that *whoever writes authoritative state persists it*.
The persistence gap was originally surfaced (but left unfixed, as out of scope)
in the 2026-07-26 concurrency race-fix report.

Work happened on `fix/persistence-gap`, branched from `docs/glossary` @
`3fa770c` per Brandon's answer to the branch-base question — chosen because the
brief's line numbers were verified against that tip, and because
`live_orchestrator.py`'s `Glossary:` section (which CLAUDE.md requires updating)
exists only there.

## Investigation

### Gate 1 — branch state

`git status` clean. `docs/glossary` (11 commits) unmerged; `main` at `b47dade`.
`3fa770c` confirmed as the current tip of `docs/glossary`, so the brief's line
numbers applied as written. Branch base was Brandon's call and was asked before
any edit.

### Gate 2 — does the running soak import this file?

**Verified empirically, not assumed.** PID 10385 runs
`run_oanda.py --daemon --env practice --granularity 15` from this working tree.
Importing exactly what `run_oanda.py` imports (its lines 60–66), transitively:

```
modules loaded: 2109
live_orchestrator in graph: NO -- absent
execution pkg __init__ imports: ['execution.enums', 'execution.risk_manager',
  'execution.factory_orchestrator', 'execution.oanda_order_manager',
  'execution.oanda_scalper_orchestrator']
```

A grep of every module in that chain found exactly one mention of
`live_orchestrator` anywhere: a docstring line in `execution/__init__.py`
stating it is deliberately excluded. **Editing this file cannot reach the soak,
even on a watchdog restart.** Gate passed.

### Gate 3 — line numbers re-verified

All confirmed at the stated locations before editing: `_save_state` :620 with
the `IN_TRADE` filter :632, `_load_state` :650, the `_handle_signal`
seed + save :1488/:1490, the BUY branch's `state == PENDING` requirement :1748,
the cancel branch :1785–1805.

Two things the brief asked me to establish rather than assume:

**The race window is real.** `ctx.state = PENDING` is set at :1458 *inside* the
lock acquired at :1434; that lock is released before
`await asyncio.to_thread(self._submit_entry_order, …)` at :1474. So a fill event
can be processed on the loop during the await, before `_handle_signal` resumes
at :1485.

**Alpaca's fill fields are cumulative.** Read from the installed SDK
(`alpaca/trading/models.py:192-194, 230-231`): `filled_qty` and
`filled_avg_price` are attributes of the `Order` model — the order's current
aggregate state, not a per-event delta — and `filled_avg_price` is by definition
an average *across* fills. Relying on cumulative semantics is sound. (Caveat
worth stating plainly: I verified the SDK's field semantics, not live
partial-fill behaviour against the API.) Note they are typed
`Optional[Union[str, float]]`; the existing `float(… or 0.0)` coercion already
handles the string case correctly.

## Findings / Changes

Ownership rule adopted: **whoever writes authoritative position state persists
it.** `_save_state` filters to `IN_TRADE`, so any caller running while the
symbol is `PENDING` is structurally incapable of recording that trade — which
is precisely what the old code did.

### Symptom 1 — new trades were never persisted

`_handle_signal` seeded the qty and called `_save_state()` while the symbol was
still `PENDING`, so the `IN_TRADE` filter dropped it; the BUY-fill branch that
*does* reach `IN_TRADE` never saved at all.

**Before** (`_handle_signal`):
```python
    ctx.entry_qty = result.submitted_qty
# Persist immediately so a restart mid-trade can recover
self._save_state()          # no-op for this trade: still PENDING
```
**After** — the call is deleted and the reason recorded in place, with
persistence moved to the fill branch:
```python
# Lock released — persist the authoritative state written above.
# Whoever writes authoritative state persists it: a BUY fill so a
# restart can re-inject SL/TP, and a cancel/expire/reject so the
# entry leaves the file the moment it leaves IN_TRADE.
# (Terminal SELL fills persist inside _enter_cooling.)
if persist_needed:
    self._save_state()
```

### Symptom 2 — fast fills persisted the wrong qty

If the fill landed during the submission await, :1488 overwrote the
authoritative `filled_qty` with the *requested* `submitted_qty`.

**After** — the seed is guarded, so it can only ever fill an empty slot:
```python
# Seed the qty ONLY while still PENDING. ctx.lock is released
# across the await above, so the fill event may already have
# been processed and written the authoritative filled_qty —
# which differs from the requested qty on a partial fill or
# fractional rounding. Never clobber it with the request.
if ctx.state == SymbolState.PENDING:
    ctx.entry_qty = result.submitted_qty
```

### Symptom 3 — partial-then-terminal BUY fills lost the final qty

The branch required `ctx.state == PENDING`. A `partial_fill` flipped state to
`IN_TRADE`, so the terminal `fill` skipped the branch entirely.

**Before:** `if order_side == "BUY" and ctx.state == SymbolState.PENDING:`
**After:** accepts both states, transitions only from `PENDING`, and refreshes
the cumulative values every time:
```python
if order_side == "BUY" and ctx.state in (
    SymbolState.PENDING,
    SymbolState.IN_TRADE,
):
    was_pending = ctx.state == SymbolState.PENDING
    if was_pending:
        ctx.state = SymbolState.IN_TRADE
    ctx.entry_price = float(getattr(order, "filled_avg_price", 0.0) or 0.0)
    ctx.entry_qty = float(getattr(order, "filled_qty", 0.0) or 0.0)
    persist_needed = True
```
Existing logging and the `_log_activity` "Filled @ …" entry are preserved
exactly on the `PENDING` path; the new `IN_TRADE` refresh path gets its own
`logger.info` rather than reusing a now-inaccurate "State -> IN_TRADE" line.

### Requirement 2 — cancel/expire/reject now persists

`persist_needed = True` added in that branch so the entry leaves the file at the
moment it leaves `IN_TRADE`.

### Design note on `persist_needed`

The brief specified persisting *after* releasing `ctx.lock`. A single flag is
declared before the `if/elif` chain and one write happens after it, which keeps
blocking file I/O out of the critical section and produces at most one write per
event. `_save_state`'s synchronous I/O was left alone as instructed.

### Docs

Per CLAUDE.md's maintenance rule, the module `Glossary:` gained the
**persistence ownership rule**, `persist_needed`, and an expanded `STATE_FILE`
entry spelling out the crypto consequence. `_save_state`'s own docstring was
corrected — it claimed it was "called after a successful market buy", which was
never true in the sense that mattered — and now carries an explicit NOTE that
calling it while a symbol is PENDING is a no-op for that symbol.

## Verification

```
python -m compileall -q src/execution/live_orchestrator.py   → OK
Full suite BEFORE (at 3fa770c):  114 passed
Full suite AFTER:                118 passed
```

Zero pre-existing failures; the four new tests are the entire delta.

**One existing test legitimately failed and was corrected, not weakened.**
`TestThreadOwnership::test_full_signal_cycle_writes_only_on_loop_thread` ended
with `orch._save_state.assert_called_once()` — an assertion that *pinned the
bug*, requiring the no-op save to happen. It now asserts `assert_not_called()`
with the reason recorded inline. This is the one behavioural change visible to
an existing test, and it is the intended one.

**Negative proof.** The new tests were run against the pre-fix source
(`git stash push src/execution/live_orchestrator.py`). All four failed, plus the
corrected ownership assertion:

```
FAILED …TestThreadOwnership::test_full_signal_cycle_writes_only_on_loop_thread
FAILED …TestTradePersistence::test_buy_fill_persists_trade_with_authoritative_qty
FAILED …TestTradePersistence::test_cancel_removes_entry_from_state_file
FAILED …TestTradePersistence::test_fast_fill_during_await_is_not_clobbered
FAILED …TestTradePersistence::test_partial_then_terminal_fill_records_final_qty
5 failed, 7 passed
```

Each test genuinely catches its symptom rather than passing vacuously.

**The tests assert on the real artifact.** `TestTradePersistence` binds the
*real* `_save_state` and `chdir`s into a temp directory (`STATE_FILE` is a
relative path), so assertions read the actual `active_trades.json` the live bot
would write — not "a mock was called". Coverage:

| Test | Pins |
|---|---|
| `test_buy_fill_persists_trade_with_authoritative_qty` | File is empty while PENDING, then contains the symbol with the event's `filled_qty` (93.5, deliberately ≠ the submitted 95.0) and the floored SL. |
| `test_fast_fill_during_await_is_not_clobbered` | Patches `asyncio.to_thread` so the fill is processed on the loop *during* the submission await; final `ctx.entry_qty` and the file both carry 93.5. |
| `test_partial_then_terminal_fill_records_final_qty` | `partial_fill` 40 → terminal `fill` 100 @ 101.5; ctx and file both carry the final cumulative values. |
| `test_cancel_removes_entry_from_state_file` | An `IN_TRADE` symbol receiving `canceled` is removed from the file. |

## Risk & follow-ups

1. **`_enter_cooling` persists while holding `ctx.lock`** (`:2022`; its docstring
   requires the caller hold the lock). The rule adopted here writes *after*
   releasing it. The SELL-close path therefore does blocking file I/O inside the
   critical section — pre-existing, correct in outcome, and explicitly outside
   this brief's scope, so untouched. Worth aligning if you ever revisit
   `_save_state`'s I/O.
2. **`entry_price` is never persisted.** `_save_state` writes only
   `sl_price`/`tp_price`/`qty`, so a trade restored by `_load_state` has
   `entry_price = None`. I traced every read: it is only used at `:1800`,
   `:1805`, `:1814`, all inside the BUY-fill branch immediately after assignment.
   So this is **harmless today** — nothing reads it on a restored path — but it
   is a latent gap the moment anything wants a restored position's entry price
   (P&L, dashboard). Not fixed; adding a field changes the file schema.
3. **`_load_state` does restore `state = IN_TRADE`** (`:730`), so a restored
   position is genuinely watched — I checked, because the fix would be worth
   much less otherwise.
4. **A partial BUY fill now persists the partial qty.** That is the intended
   behaviour (better than persisting nothing), but it means a crash between a
   partial and its terminal fill restores a position whose recorded qty is
   smaller than what Alpaca actually holds. The watchdog would then close only
   the recorded amount. Fully resolving that needs reconciliation against
   Alpaca's real position size on load — the OANDA orchestrator already does
   this (`_reconcile_on_boot`); this one only cross-references *which* symbols
   are open, not *how much*. That is a real gap, larger than this brief, and I
   would treat it as the natural next piece of work.
5. Nothing was committed or pushed. The tree is left for review on
   `fix/persistence-gap`.

## Files touched

- `src/execution/live_orchestrator.py` — 83 insertions, 23 deletions.
  Module docstring `Glossary:` (~:130–145); `_save_state` docstring (~:620–636);
  `_handle_signal` seed guard and removed save (~:1495–1518);
  `_on_trade_update` `persist_needed` declaration, BUY branch, cancel branch,
  and the post-lock write (~:1770–1875).
- `tests/execution/test_live_orchestrator.py` — 189 insertions, 3 deletions.
  Imports (`json`, `tempfile`, `Path`, `STATE_FILE`); module docstring
  `Glossary:`; corrected `_save_state` assertion in `TestThreadOwnership`
  (~:346); new `TestTradePersistence` class (4 tests).
