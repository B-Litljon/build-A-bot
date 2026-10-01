# `tests/`

**642 tests + 17 subtests, all passing** (2026-10-01, after the rollover-bar
exclusion and the pipette-quantization serving toggle landed; 2 environmental
errors in `TestStaleFeatureGuard` when run inside a lane worktree — untracked
model pkls exist only in the main checkout). Run with:

```bash
PYTHONPATH=src:. python -m pytest -q
```

`pyproject.toml` sets `testpaths = ["tests"]` so a bare `pytest` only collects
from here — a deliberate guard, because root-level `test_*.py` probe scripts
have historically had import-time side effects (one posted to the live Discord
webhook on import).

> `verify_warmup.py` (a filename-mismatched Alpaca-lane test, never
> collected) and `execution/test_live_orchestrator.py` were deleted with the
> dormant Alpaca scalper lane on 2026-09-16.

**No network, no broker, no real models.** Every test stubs its dependencies,
and orchestrators are even constructed via `__new__` to skip their heavy
`__init__` chains. Running the suite against a live soak is harmless.

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
| `test_risk_manager.py` | 35 | The bracket floors, all three chop gates, and the learned-barrier geometry substitution |
| `test_oanda_forex.py` | 31 | The live bot's control flow — mostly failure paths |
| `test_feature_stats.py` | 12 | The stats artifact and the PSI maths |
| `test_stream_liveness.py` | 9 | What happens when the price feed goes silent |
| `test_entry_guards.py` | 15 | Post-exit cooldown + the correlated-exposure cap |
| `test_events.py` | 11 | The telemetry sink: never raises, never blocks, never on the tick path |
| `test_cost_feature.py` | 9 | The per-instrument cost feature and veto alphas |
| `test_rollover_bar_exclusion.py` | 13 | The pre-featurization NY-rollover bar exclusion: DST contract (July/January), end-exclusive boundary, the flag off restores the old rows, the window comes from `get_blackout_window_et` (no third copy), missing-timestamp/unparseable-window safe no-ops, and the feature-contamination observable (neighbor features change once the bars are dropped) |
| `test_oanda_pipette_rounding.py` | 15 | The `OANDA_ROUND_MID_TO_PIPETTE` emission quantizer: default-OFF inertness (emitted bar bit-identical, internal state untouched), JPY 3dp / GBP_AUD 5dp, half-up ties, the construction-time flag snapshot (no mid-stream env flip), and the full `_handle_tick` → `_flush_bar` pipeline both ways |
| `test_trading_mcp.py` | 9 | The MCP two-step confirm-token safety flow |
| `test_oanda_entry.py` | 5 | Net-position arithmetic |
| `test_oanda_tick_hook.py` | 5 | The raw tick callback contract |
| `test_execution_safety.py` | 3 | Rebalance deadband, fill parsing, partial fills |
| `test_retrainer_output_dir.py` | 3 | `RETRAIN_MODEL_DIR` isolation |
| `test_holdout_gate.py` | 26 | Holdout split, artifact scoring, metadata recording, the confidence-bound verdict, the boundary-tail purge, and the permanent leak guards |
| `test_dynamic_thresholds.py` | 14 | The 2026-08-29 gate rebuild: OOF Angel-bar calibration, Devil min_child auto-scaling, CP fold-evidence bounds |
| `test_retraining_notification.py` | 9 | The retrain Discord embed: a gate pass is not a deployment |
| `test_ml_strategy_guards.py` | 27 | The stale-bar guard, threshold.json pinning, sidecar reload seams, and the learned-barrier sidecar (boot refusal, units contract, promotion swap, promotion-verdict refusal) |
| `test_alpaca_timeframe.py` | 7 | Bar-size → Alpaca timeframe mapping; regression test for the defect that made H4/D1 requests impossible (minute amounts cap at 59, Day/Week take amount 1 only) |
| `test_base_rate_benchmark.py` | 6 | The gate's edge-over-random benchmark: the macro base rate is the mean on the population given, non-finite outcomes are dropped, an unlabelled frame returns nan (never 0.0), and the report carries it per fold and pooled |
| `test_devil_label_switch.py` | 6 | `RETRAIN_DEVIL_LABEL`: default preserves the shipping label, `macro` selects the validated one, typos warn and fall back, read per call |
| `test_barriers.py` | 39 | Excursion labels, the quantile estimator, the monotone audit, and the artifact contract (save/load, promotion verdict, stop calibration) |
| `test_lab_spec.py` | 8 | `FeatureSpec`: frozen, validated, and content-hashed — every field that changes a frame must move the hash |
| `test_lab_registry.py` | 9 | Feature-family registry: unknown names raise, declared columns are what the model sees, the seed family is per-symbol and causal |
| `test_lab_frames.py` | 5 | The parity contract — `v3_base` frame == `engineer_features_and_labels` + production tail purge, with and without the spread table |
| `test_lab_gate.py` | 5 | Gate wrapper wiring: exact args, `RETRAIN_DEVIL_LABEL` scoped and restored, MODEL_FAMILY mismatch fails loudly |
| `test_lab_backtest.py` | 6 | Model-aware strategy: raw-ATR units, row alignment, thresholds, live-gate funnel, per-symbol toll |
| `test_lab_artifact.py` | 12 | Served-artifact replay: `threshold.json` precedence, CatBoost's `feature_names_`, schema-order refusal, gate-path parity, window split |
| `test_lab_report.py` | 5 | The recon emitter's caveats (failed gate, thin population, cost table off) and its atomic write |

### Tests worth understanding before changing anything

**`test_risk_manager.py`** — pins the rules that decide whether a trade is
allowed *at all*. A regression here doesn't raise; it silently starts taking
trades the system was built to refuse. Note `test_cold_start_bypasses_regime_gate`:
with too little history the regime gate must stand down rather than veto
everything, or a just-restarted bot freezes.

`TestBarrierGeometry` in the same file (added 2026-09-14) covers the learned
bracket substitution, and one property there is the one to keep: a payload
**replaces** the profile multipliers, it does not compound with them
(`test_payload_does_not_compound_with_the_profile`), and the gates must be asked
about the substituted distance
(`test_gate_a_asks_about_the_substituted_stop`). Both failure modes are silent —
a doubled bracket is just "a wider stop", and a gate reading the static distance
admits trades whose real stop is eaten by the spread.

**`test_oanda_forex.py`** — the failure-path collection.
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


**`test_retrainer_output_dir.py`** — small but load-bearing. `RETRAIN_MODEL_DIR`
is the isolation mechanism for experiments; if it leaked, a side experiment
would overwrite the promoted model a live bot hot-reloads, silently swapping the
running strategy's brain for an unvalidated candidate.

**`test_holdout_gate.py`** — the central invariant of the artifact holdout:
holdout rows are carved first and never enter training, the split is
chronological and disjoint, and `metadata.json` records either the holdout
metrics or an explicit bypass reason. Since the 2026-08-24 stability brief it
also pins the verdict itself: the PF bar gates on the Clopper-Pearson
lower bound (`TestHoldoutPfConfidenceBound`, `TestHoldoutVerdict` — the
audit's PASS/FAIL/PASS windows are FAIL/FAIL/FAIL under the bound), the
boundary tail purge drops exactly the last `max_hold` bars per symbol
(`TestBoundaryTailPurge`), `TestPermanentLeakGuard` is the audit's
instrumented leak check made permanent (a recording `refit_models` stand-in
asserts no validate_candidate training frame contains a holdout timestamp),
and `TestMainWiringLeakGuard` runs `main()` end to end so a regression in
Phase 3a/4.5 wiring fails here. The tests use small mocks so they never
train real models, but they exercise the same indexing paths the production
gate uses.

**`test_dynamic_thresholds.py`** — pins the three 2026-08-29 gate fixes as
units, because each was originally discovered as a silent failure in a gate
log: the Angel bar calibrated from OOF score quantiles (compressed /
constant / tiny-frame / empty distributions, determinism), the Devil
min_child auto-scaler (a tenth of the approved population, capped, floored —
a split is always possible), and the CP lower bound's pinned reference values
(3/3 perfect must fail the 1.2 bar, 4/4 must pass). One trap is written into
the file header: never put `tests/` itself on `sys.path` to reach sibling
test modules — `tests/execution/` then shadows the real `execution` package
and core imports die mid-collection; import siblings as `tests.test_*`.

**`test_retraining_notification.py`** — the other half of that isolation. The
files stayed isolated on 2026-08-17, but the *alert* did not: a side experiment
posted "✅ PROMOTED — New models passed all validation gates and are now live"
to Discord, indistinguishable from a real production promotion. Nothing had
gone live. These tests pin that a run redirected by `RETRAIN_MODEL_DIR` is
reported as a side candidate, in a different colour, naming the directory it
actually wrote to. Discord is the only channel this system uses to reach a
human, so a false alarm there costs real trust.

**`test_trading_mcp.py`** — pins that a wrong, missing, or reused confirm token
cannot act, i.e. an AI assistant can't start or stop the live bot by accident.

**`test_investor_benchmark_gate.py`** — covers the investor's lift-over-benchmark
gate: the sector cap it simulates, the month-pairing that turns model scores into
realised returns (including the holding period that legitimately closes *outside*
the test window), and that an unmeasurable model **fails closed** rather than
sliding through. One test asserts the gate's `TOP_K`/`SECTOR_CAP` still equal the
orchestrator's — if the deployed basket shape changes, the gate must fail loudly
rather than quietly measure a basket nobody trades.

### `__init__.py` / `execution/__init__.py`
Empty package markers.
