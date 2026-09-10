---
type: refactor
date: 2026-09-09
time: 15:50 PDT
agent: z
model: kimi-k3 (via dsh, ollama-cloud route)
trigger: Implementing the ranked list from the 2026-09-08 audit, in order, easiest first
head: 6390128db7ec566f816c9dae2551641c591d74b2
scope: modifies-source
related:
  - audits/2026-09-08_high-benefit-fixes-ranked.md
files_touched:
  - CLAUDE.md
  - GLOSSARY.md
  - soak.service
  - soak_watchdog.sh
  - config/spread_alphas_m15.json
  - src/core/feedback_loop.py
  - src/core/retrainer.py
  - src/data/oanda_provider.py
  - src/evaluate_performance.py
  - src/execution/oanda_forex_orchestrator.py
  - src/execution/oanda_order_manager.py
  - src/replay_test.py
  - src/strategies/concrete_strategies/ml_strategy.py
  - tests/test_cost_feature.py
  - tests/test_execution_safety.py
  - tests/test_holdout_gate.py
  - tests/test_htf_pairing.py (new)
  - tests/test_ml_strategy_guards.py
  - tests/test_oanda_forex.py
  - tests/test_retrainer_output_dir.py
  - tests/test_risk_manager.py
  - tests/test_stream_liveness.py
---

# Safety & training-integrity pass — items 1–20 of the 2026-09-08 audit

## Context

The 2026-09-08 audit (`llm_reports/audits/2026-09-08_high-benefit-fixes-ranked.md`)
produced a ranked list of high-benefit fixes. This refactor implements items 1–20
in that order — the Tier 0 data/doc fixes, all of Tier 1 and Tier 2, and the
Tier 3 money-safety bundle — leaving only the two retrain-gated Tier 4 items
deferred (see Risk & follow-ups) and the Tier 5 research decisions. Every
change is tested; the live soak (PID at start: 3365348, then 5597 after
Brandon's morning restart) runs the OLD code until its next restart, by design
— nothing here hot-reloads except model artifacts.

## Investigation → Changes

### Tier 0 — data and docs
- **Stale artifacts retired to reversible stashes** (moves, not deletes):
  `data/_retired_2026-09-09/` (signal_ledger.csv, drift_report.json,
  evaluation_results.parquet, oos_bars.parquet, training_data.parquet,
  forex_365d_1m cache) and `models/_legacy_equities_2026-05/` (the root
  angel/devil pkls `replay_test.py` used to grade). The offline tools that
  referenced them now fail loudly instead of silently grading 5-month-old
  data.
- **Spread alphas re-baked** from the live soak log: GBP_JPY 0.3307,
  AUD_JPY 0.3372, EUR_JPY 0.3600, NZD_JPY 0.5591, GBP_AUD 0.6952 (was a stale
  0.5834), GBP_NZD 0.8929, n=710 each. Serving the table is deferred to the
  next retrain (symmetry contract — see below).
- **Docs**: GLOSSARY risk-per-trade entry now states the live forex path uses
  fixed 1000 units (2%/floor apply to Alpaca/Factory only); feature count
  corrected 22/23 → 17/18; CLAUDE.md test command now names the venv python
  and the real count; soak.service's warning updated to describe the actual
  danger (copying a wrong-bracket pair INTO forex_m15_wide) — and the bracket
  check it promises now exists (item 8).

### Tier 1 — small guards
- **`oanda_forex_orchestrator._on_bar`**: early-return when
  `_shutdown_event` is set (a shutdown-flushed partial bar can no longer open
  a post-flatten position), and bars for instruments outside the normalized
  basket are ignored instead of raising KeyError inside the unawaited loop
  future.
- **`oanda_provider._handle_tick`**: build-then-flush — a raising `_flush_bar`
  can no longer wedge an instrument's bar pipeline forever.
- **`feedback_loop.py`**: RESOLVED_PATH now points at
  `data/evaluation_results.parquet` (the file Phase 3 actually writes); the
  loader derives `outcome` from `exit_type == "WIN"`, EV is computed from
  `pnl_r` (R-multiples) with the legacy constants as a fallback for old
  files. The drift→retrain branch of `run_pipeline.sh` is alive again.
- **`evaluate_performance.py`**: brackets now come from
  `RiskProfile.for_asset_class("forex")` (2.0×/4.0×) and the intra-bar exit
  check is SL-first — the offline grade now describes the same trade the
  trainer labels and the bot takes.
- **`retrainer.save_threshold`**: full-precision bars (was `round(x, 4)`,
  drifting the live Angel bar ~5e-5 looser than its calibration population)
  and a UTC `updated_at`.
- **`MLStrategy._validate_metadata`**: raises when metadata's
  sl/tp multipliers mismatch the asset-class profile — the train/serve-skew
  tripwire soak.service always wanted. (Lazy import of RiskProfile: a
  module-level import cycles through `execution/__init__.py`.)
- **Tests added**: `TestGateCBlackout` (7 cases: EDT/EST windows, boundaries,
  midnight wrap, naive-UTC, disabled profile, production default — this gate
  had zero coverage) and `tests/test_htf_pairing.py` (pins
  `run_oanda._GRANULARITY_PROFILES` ≡ `retrainer._HTF_FOR_TIMEFRAME`).

### Tier 2 — small patches
- **`soak_watchdog.sh`**: staleness check added to the already-running branch
  — status.json older than 20 min for 3 consecutive ticks (market-open only,
  run older than the threshold) logs + ntfy, then restarts the unit. Weekend
  blackout now computed in America/Los_Angeles; dead NOT_BEFORE gate deleted;
  crash-loop state moved from /tmp to logs/; post-launch check sleeps 15s.
  Live-exercised: ran the real tick with the soak healthy — exit 0, no output.
- **Gate C mirrored in training**: `_compute_chop_veto_mask` now vetoes
  NY-rollover bars (same window semantics as `RiskManager._in_blackout`,
  DST-correct via America/New_York conversion, naive timestamps treated as
  UTC). Two bugs found while doing it: polars `dt.hour()*3600` overflows
  Int8 (cast to Int64), and the per-symbol loop ASSIGNED
  `veto[idx] = gate_a|gate_b`, clobbering the Gate C mask (now ORs).
  `TestGateCMirror` pins winter/summer veto rows.
- **Hot-reload consistency bundle** (`ml_strategy.py` + `retrainer.py`):
  threshold.json and spread_alphas.json reload on their OWN mtimes every bar
  (previously only when a pickle also changed — a promotion could pin the new
  pair to the old bars forever); a pair-seam state machine
  (`_pair_mixed`/`_pair_pending`) stands signal generation down when one half
  of the Angel/Devil pair advances without the other and clears when the
  other half lands; `save_models` now serializes both pkls to temp files
  BEFORE the two back-to-back replaces. `TestHotReloadSeams` covers the
  state machine including the inverted-pending bug it caught.
- **`_prime_history`**: never raises (a raise there killed the reconnect
  loop permanently); bounded retries with backoff (env OANDA_PRIME_ATTEMPTS/
  OANDA_PRIME_BACKOFF); CRITICAL log when a symbol primes empty after
  retries (previously ~2.75 days of silent no-trading); naive-tz bars abort
  the symbol instead of the loop; shutdown's stream-wait now swallows a
  raising stream task so the SIGTERM flatten always runs.
- **`replay_test.py`**: pointed at the SERVED model dir (OANDA_MODEL_DIR,
  default models/forex_m15_wide), reads both threshold.json keys, FEATURE_NAMES
  := retrainer.BASE_FEATURE_COLS (the old hand list omitted the four
  session_* flags and would crash on the served schema), HTF from the
  granularity pairing, and a zero-signal run deletes the stale ledger before
  the early return.
- **Cooldown race**: `_evaluate_and_trade` re-checks the post-exit cooldown
  under the second lock acquisition — a stop-out landing between the two
  lock sections can no longer let the symbol re-enter on the bar that
  stopped it.

### Tier 3 — money-safety
- **Close verification** (`oanda_order_manager.close_position`): the return
  value is now "broker VERIFIED flat" — a follow-up `sync_position()` runs
  after every close request, the cache-flat path verifies before no-op'ing
  (it can read 0 while the broker holds an unverified fill), and an
  unverifiable state raises OrderCloseError. Partial fills return False
  (verified still open) so callers retry the remainder.
- **`_watchdog_close`**: honors the verified result; False = retry.
- **`_flatten_all`**: clears only VERIFIED-flat records; failures park
  CLOSE_FAILED instead of the old unconditional `_positions.clear()`;
  ENTRY_UNRECONCILED records stay for the reconciler; symbols with in-flight
  `_pending_entries` are skipped (a close mid-delta can double the fill).
- **`_retry_failed_closes`**: new — every 10s liveness pass retries
  CLOSE_FAILED closes; success pops the record and marks the exit. A failed
  stop-exit is no longer terminal for the whole run.
- **Boot reconcile**: treats verified-still-open the same as a failed
  flatten (refuse to start).
- **Liveness blind windows** (`oanda_provider` + `_check_stream_liveness`):
  the provider now tracks `_last_price_msg` and `_stream_down_since` (set in
  run_stream's finally, cleared on message). The liveness check flattens on
  EITHER stream-down past threshold (the old `age is None` no-op was the
  state during every real outage) or price-silence while heartbeats flow;
  alerts once per incident via `_liveness_alert_fired`. The pinned
  `test_stream_not_running_no_action` was consciously updated to the new
  contract, plus two new flatten-on-down/silence tests.
- **Entry integrity**: after submit, `position_units != target_units` parks
  the entry ENTRY_UNRECONCILED (a flatten racing the delta-based order leaves
  2× units — recording it would mis-track size and mis-reserve the cap);
  brackets re-anchor on the real fill price preserving the approved
  distances (the same fix `_reconcile_unverified_entries` already had for
  parked entries, now on the normal path).

### Tier 4 — retrain-gated (the two that shipped in code, deferred in effect)
- **OOF leakage** (`retrainer.refit_models`): TimeSeriesSplit now runs on a
  chronological permutation (`np.argsort(timestamp)`) with probabilities
  written back to original indices — previously the symbol-blocked frame made
  each val fold whole symbol blocks and the Devil's `angel_prob` meta-feature
  came from models trained on other symbols' FUTURE. The head-fill comment
  now honestly says in-sample-for-the-head, no-future-leakage.
- **Decay weights** (`generate_time_decay_weights`): optional `timestamps`
  parameter — weights rank by dense timestamp rank instead of row index
  (row index encoded basket position; reordering RETRAIN_SYMBOLS changed the
  model). Legacy no-timestamp path preserved.
- **Fold/holdout EV semantics**: EV now uses the MACRO approval rate with
  macro R:R on both the holdout scorer and the per-fold metrics (the old
  survival-rate × macro-R:R mixture overstated expectancy — served artifact
  recorded 1.58 vs a 0.0005 bar).

## Verification

- `PYTHONPATH=src:. <venv>/bin/python -m pytest -q` → **437 passed, 5
  skipped** (was 411/5 at audit time; +26 from the new Gate C, HTF-pairing,
  hot-reload-seam, close-verification, liveness, entry-integrity and
  chronological-OOF tests).
- `python -m compileall -q src/` clean.
- `feedback_loop` smoke-tested against the stashed parquet (loads, grades,
  drift verdict produced).
- The real `soak_watchdog.sh` tick was run with the soak healthy: exit 0,
  no spurious restart/alert.
- Live soak re-checked after the pass: active, status fresh, flat, counters
  at zero — it continues to run the pre-change code until its next restart.

## Risk & follow-ups

- **Nothing here takes effect live until the soak restarts.** The watchdog
  relaunches from the working tree, so the NEXT restart (manual or watchdog)
  picks up all of it. If you restart manually: `touch soak.off` first or the
  watchdog races you.
- **Gate C mirror + spread table + OOF fixes all change the training
  population** — they land with the NEXT retrain. Do not cherry-pick a
  retrain before reviewing the new chop-veto counts in its log.
- The new liveness flatten fires when the stream is down >60s WITH positions —
  during a real OANDA outage it will close positions (by design: software
  stops are blind without prices). The reconnect backoff (max 60s) plus
  prime normally stays under the threshold; a healthy reconnect cycle should
  not trip it. Watch for flatten noise after the next deploy.
- **Deferred (audit items 12-fold-curse, 21–24)**: folds 1–2 still report
  EV at their own swept threshold (winner's curse on 2 of 3 folds — the
  meaningful part, the EV-gate semantics, is fixed); the economics question,
  strategy-library edge, Alpaca-path revival, and ops-structure items
  (systemd timer for the watchdog, run_soak singleton guard, /tmp state
  files, log pruning) are unstarted.
- The served `threshold.json` still carries the OLD calibrated Angel bar —
  expected; it re-derives on the next retrain from honest OOF probabilities.
- `clientRequestID` dedup on entry retries was deliberately NOT implemented
  (OANDA semantics unverified); the settle-delay alternative also remains.
