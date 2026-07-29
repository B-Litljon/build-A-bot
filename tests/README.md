# `tests/`

**136 tests, all passing.** Run with:

```bash
PYTHONPATH=src:. python -m pytest -q
```

`pyproject.toml` sets `testpaths = ["tests"]` so a bare `pytest` only collects
from here — a deliberate guard, because root-level `test_*.py` probe scripts
have historically had import-time side effects (one posted to the live Discord
webhook on import).

> ⚠️ **`verify_warmup.py` is never run.** It contains a real test case, but
> collection also requires the default `test_*.py` filename pattern and this
> file doesn't match. The suite reports 136 collected and this one isn't among
> them. Renaming it to `test_warmup.py` would include it. Flagged, not changed.

**No network, no broker, no real models.** Every test stubs its dependencies —
`SymbolContext` and `LiveOrchestrator` are even constructed via `__new__` to
skip their heavy `__init__` chains. Running the suite against a live soak is
harmless.

## What the suite is actually protecting

Most of these tests target **failure paths, not happy paths**, and the reason is
structural: stops and targets are enforced *in software by the bot process*. So
the dangerous states aren't "the model was wrong" — they're "we think we're flat
but aren't" and "we tried to close and it didn't work". Read the suite with that
lens and the emphasis makes sense.

See the root [GLOSSARY.md](../GLOSSARY.md) for domain terms.

## Files

| File | Tests | Covers |
|---|---:|---|
| `test_composite_fundamentals.py` | 18 | Provider chaining: first non-empty wins; a raising source is a miss, not an error |
| `test_risk_manager.py` | 16 | The bracket floors and all three chop gates |
| `test_oanda_scalper.py` | 30 | The live bot's control flow — mostly failure paths |
| `test_feature_stats.py` | 12 | The stats artifact and the PSI maths |
| `test_stream_liveness.py` | 9 | What happens when the price feed goes silent |
| `test_cost_feature.py` | 9 | The per-instrument cost feature and veto alphas |
| `test_trading_mcp.py` | 9 | The MCP two-step confirm-token safety flow |
| `execution/test_live_orchestrator.py` | 12 | State machine + **thread-ownership regression** + persistence ownership |
| `test_oanda_entry.py` | 5 | Net-position arithmetic |
| `test_oanda_tick_hook.py` | 5 | The raw tick callback contract |
| `test_execution_safety.py` | 3 | Rebalance deadband, fill parsing, partial fills |
| `test_retrainer_output_dir.py` | 3 | `RETRAIN_MODEL_DIR` isolation |
| `test_ml_strategy_guards.py` | 5 | The stale-bar guard + threshold.json pinning |
| `verify_warmup.py` | (1, **not collected**) | Warm-up injection |

### Tests worth understanding before changing anything

**`test_risk_manager.py`** — pins the rules that decide whether a trade is
allowed *at all*. A regression here doesn't raise; it silently starts taking
trades the system was built to refuse. Note `test_cold_start_bypasses_regime_gate`:
with too little history the regime gate must stand down rather than veto
everything, or a just-restarted bot freezes.

**`test_oanda_scalper.py`** — the failure-path collection.
`test_rapid_breach_ticks_close_once` (quotes arrive far faster than a close
completes, so a burst must produce *one* close),
`test_close_not_called_synchronously_in_tick` (the tick callback runs on the
stream thread and must dispatch, not block),
`test_watchdog_close_failure_retries_then_parks` (a position that won't close is
*parked*, not forgotten — forgetting it means an open position nothing is
watching), and the `test_boot_reconcile_*` group (on startup, ask the broker
what's really open; a failed sync **aborts** rather than proceeding blind).
The `test_seam_catchup_*` group pins the reconnect-gap fix: a bar that sealed
while the stream was down is scored exactly once if fresh — stale bars and
already-scored bars are skipped, and an evaluation error must not escape (it
would kill the reconnect loop). This is the fix for the 2026-07 soak losing
~half its signals in re-prime gaps. The `test_seam_backfill_*` group covers
its other half — the bar in flight when the stream died is re-fetched from
REST and scored, with retries for REST lag and a quiet give-up that never
raises inside the bar callback.

**`test_oanda_tick_hook.py`** — `test_tick_callback_exception_logged_continues`
is the important one: an exception escaping the callback would kill the price
feed, which with software stops means an unwatched position.

**`execution/test_live_orchestrator.py`** — the thread-ownership suite added
with the 2026-07-26 concurrency fix. It patches `SymbolContext.__setattr__` to
record `(attribute, thread id)` for every write, drives a real bar→inference→
order cycle, and asserts every write landed on the event-loop thread. That
catches a whole *class* of bug rather than one instance. Two details worth
copying elsewhere: a **vacuity guard** (assert all eight fields were actually
written, so a test that exercised nothing can't pass), and the fact that the
instrumentation was **negative-proofed** — a thread-side write was temporarily
injected to confirm both tests fail, then removed. An assertion never observed
failing isn't evidence.

**`test_retrainer_output_dir.py`** — small but load-bearing. `RETRAIN_MODEL_DIR`
is the isolation mechanism for experiments; if it leaked, a side experiment
would overwrite the promoted model a live bot hot-reloads, silently swapping the
running strategy's brain for an unvalidated candidate.

**`test_trading_mcp.py`** — pins that a wrong, missing, or reused confirm token
cannot act, i.e. an AI assistant can't start or stop the live bot by accident.

### `__init__.py` / `execution/__init__.py`
Empty package markers.
