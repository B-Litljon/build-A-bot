# `tests/execution`

Tests for the execution layer's most subtle contracts. See the parent
[`../README.md`](../README.md) for how to run the suite and what it protects.

## Files

### `test_live_orchestrator.py` (8 tests)
Two suites in one file.

**State machine** (original) — exercises `_on_trade_update`'s transitions
(FLAT → PENDING → IN_TRADE → PENDING_EXIT → COOLING) without Alpaca clients,
models, or websockets. `SymbolContext` and `LiveOrchestrator` are built via
`__new__` to bypass their heavy `__init__` chains; only the attributes the
handler reads get populated.

**Thread ownership** (added 2026-07-26 with the concurrency fix) — pins that the
asyncio event loop is the *sole writer* of `SymbolContext` state.

The technique is worth reusing: `SymbolContext.__setattr__` is patched to record
`(attribute, thread id)` on every write; the test then drives a real
sealed-bar → inference → signal → order cycle and asserts every recorded write
happened on the loop thread. That catches a whole **class** of bug rather than
one instance — any future code that mutates context from a worker thread fails
these tests, whether or not anyone thought to test that specific field.

Two details worth copying into other tests:

- **Vacuity guard** — it also asserts all eight expected fields were actually
  written. Without it, a test that accidentally exercised nothing would still
  pass, and a passing assertion over an empty set proves nothing.
- **Negative proof** — the instrumentation was validated by temporarily
  injecting a thread-side write and confirming both tests *failed*, then
  removing it. An assertion never observed failing isn't evidence that it works.

Fixture note: `natr_14 = 0.05` is chosen so the volatility kill switch passes
*and* the minimum-stop floor fires, exercising both paths in a single cycle.

- **Imports from repo:** `execution.live_orchestrator`.
- **Data artifacts:** none — all inputs synthetic, all dependencies mocked.

### `__init__.py`
Empty package marker.
