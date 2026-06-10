---
type: refactor
date: 2026-06-09
time: 17:55 PDT
agent: Claude Fable 5
model: claude-fable-5
trigger: User approved a 4-work-package plan to fix the 2026-06-09 full-repo audit findings
head: 8891711
scope: modifies-source
related:
  - audits/2026-06-09_full-repo-audit.md
files_touched:
  - src/execution/oanda_order_manager.py
  - src/execution/oanda_scalper_orchestrator.py
  - src/data/oanda_provider.py
  - src/strategies/concrete_strategies/ml_strategy.py
  - src/execution/risk_manager.py
  - src/core/notification_manager.py
  - src/ml/feature_pipeline.py
  - run_oanda.py
  - pyproject.toml
  - Pipfile
  - tests/test_oanda_scalper.py
  - tests/test_oanda_tick_hook.py
  - tests/test_risk_manager.py
  - tests/test_stream_liveness.py
  - tests/test_ml_strategy_guards.py
---

# Audit hardening — V5 OANDA scalper (4 commits on fix/audit-hardening)

## Context

The same-day audit (`audits/2026-06-09_full-repo-audit.md`) found that
every critical issue in the V5 live path shared one shape: a transient
failure silently converts into an unmonitored live position, because
SL/TP enforcement is software-only. User rulings before implementation:
flatten unknown broker positions on boot; keep the 2.0-pip chop floor
default (A3 telemetry decides); give metals a percent-based floor; soak
is down, so behavior-changing fixes may land freely.

Branch: `fix/audit-hardening`, four logical commits (WP1–WP4), audit
report committed first at `dd43063`.

## Investigation

All design detail is in the audit report. One additional discovery made
during WP1 testing: the pytest suite was posting to the **live Discord
webhook**. `LiveOrchestrator.__init__` calls `load_dotenv()`
(live_orchestrator.py:420), which loads `.env` — including
`DISCORD_WEBHOOK_URL` — into `os.environ` for the remainder of the
pytest process; every later test that constructed a real
`NotificationManager` then posted for real. The orchestrator tests had
been sending ENTRY alerts on every suite run, and the new WP1 tests
briefly sent CLOSE_FAILED / manual-intervention alerts before the cause
was found (operator informed). Also verified `oandapyV20` supports
`request_params={"timeout": ...}` on the API client and
`PricingStream.terminate()` → `StreamTerminated`, and that
`requests`' read timeout on a streaming response means *inactivity*
timeout — the correct semantics for heartbeat-backed stall detection.

## Findings / Changes

### Commit 1 — `3a34ed2` execution safety (C1, C2, C4, H1, M5)

- `oanda_order_manager.py`: `close_position` raises new `OrderCloseError`
  on broker failure (was: swallow + return False, indistinguishable from
  already-flat). `sync_position` returns a success bool.
  `submit_target_position` return dict gains `position_units` /
  `position_avg_price` (authoritative post-trade net) alongside the raw
  fill fields.
- `oanda_scalper_orchestrator.py`: `_watchdog_close` retries with
  exponential backoff (`OANDA_CLOSE_MAX_ATTEMPTS`, default 5; 1/2/4/8/16s),
  pops the position only on confirmed close, parks it as `CLOSE_FAILED`
  with a manual-intervention alert when exhausted (was: pop in `finally`
  + log "completed" even on failure). New `_reconcile_on_boot()` syncs
  every symbol before trading, flattens orphans, aborts startup if
  broker state can't be verified. New `_notify()` pushes all Discord
  posts off the event loop. Entry guard skips any non-OPEN state.
  Position records use `position_units`/`position_avg_price` (raw
  `filled` double-counted reversals). `_flatten_all` escalates exit
  failures to a manual-intervention alert.
- Orchestrator accepts an injectable `notifier`; tests inject a mock.

### Commit 2 — `fbc0d8d` stream liveness (C3, L8)

- `oanda_provider.py`: API client constructed with
  `request_params={"timeout": OANDA_STREAM_TIMEOUT_SECONDS}` (default
  20s) — converts silent TCP stalls into exceptions the existing
  `_stream_with_retry` reconnect already handles (OANDA heartbeats
  ~5s, so 20s of silence is dead). Tracks last-message age including
  heartbeats (`seconds_since_last_message`), exposes
  `force_disconnect()` (terminate, best-effort) and `reset_stop()`.
- Orchestrator: `_liveness_watchdog` task probes every 10s; past
  `OANDA_STREAM_STALE_SECONDS` (default 60) it flattens open positions
  (REST is a separate connection), alerts, and forces a reconnect.

### Commit 3 — `630fc0c` signal integrity (H2, H3, M3, M7)

- `ml_strategy.py`: skips the signal when the feature frame's tail
  timestamp ≠ raw frame's tail (clean_data dropped the newest bar —
  previously scored stale features against the current price). `symbol`
  initialized before the try (except-path NameError).
- `oanda_provider.py`: tick aggregator opens bars with `volume: 1`
  (was 0) — restores parity with training data where OANDA candle
  volume is full tick count. **Live `vol_rel` distribution shifts
  slightly; landed at a soak boundary.**
- Orchestrator: flips mark the old position `REVERSING` under the lock
  before submit (tick watchdog only fires on OPEN, so no mid-flip close
  on stale SL/TP); state restored to OPEN on submit failure; a flip
  netting to flat pops the record.

### Commit 4 — `8891711` config & hygiene (H4, M2, M4, M6, L1, L3–L7)

- `pyproject.toml` (new): `testpaths = ["tests"]` — bare `pytest` no
  longer imports root probe scripts (`test_discord.py` posts to the live
  webhook at import). `pytest` added to Pipfile dev-packages.
- `risk_manager.py`: metals (XAU/XAG/XPT/XPD) use percent-of-price floor
  `RISK_METALS_MIN_SL_PCT` (default 0.0001 = 0.01%, the relative scale
  of the 2-pip JPY-cross floor). Pip floor was a no-op on metals.
- `run_oanda.py`: default basket = `trained_on_symbols` from
  models/forex/metadata.json (was EUR/USD — not in the trained basket);
  warns per out-of-basket symbol.
- `ml_strategy.py`: Devil hot-reload under `_reload_lock`;
  `feature_names` refreshed from the reloaded Angel; Angel/Devil schema
  consistency check with critical alert on mismatch.
- Misc: junk root file deleted, `logging.basicConfig` moved out of
  feature_pipeline import path, tz-naive prime bars raise (was assert),
  `utcnow()` → `now(timezone.utc)`, `_load_threshold` warning names the
  real path.

## Verification

- Full suite after each commit; final: **50 passed** (was 26), via the
  pipenv venv and via bare `pytest` (confirming testpaths scoping).
- New tests: watchdog retry/park/recover, CLOSE_FAILED entry block, boot
  reconcile (flatten / no-op / abort), reversal accounting, REVERSING
  tick suppression, failed-flip state restore, flip-to-flat pop, stream
  staleness (4 orchestrator + 5 provider cases), stale-feature guard
  against the real promoted models, metals floor pass/veto + fiat
  non-regression.
- `_trained_basket()` verified to return the 8-instrument basket from
  metadata.json.
- 5-minute practice-env smoke (`run_oanda.py --daemon`, SIGTERM at end):
  boot reconciliation, history priming, stream + liveness watchdog,
  graceful shutdown. Result recorded in the PR.

## Risk & follow-ups

- **Volume init change (M3)** shifts live `vol_rel` slightly toward the
  training distribution; watch early soak telemetry rather than mixing
  mid-soak data.
- **Metals floor** is newly active (default 0.01% of price) — it can now
  veto XAU/XAG entries where the old code never did. The A3 rejection
  counter covers it; tune `RISK_METALS_MIN_SL_PCT` if the veto rate is
  material.
- `CLOSE_FAILED` positions do not auto-retry after the 5th attempt by
  design (manual intervention) — consider a slow background retry loop
  if soak shows transient broker errors lasting > ~30s.
- Out of repo: `~/.config/systemd/user/universal-scalper.service`
  ExecStart still points at a nonexistent `.venv` (audit L2).
- Not addressed (per rulings/audit): chop-floor default (telemetry
  decides), retrainer chop-filter asymmetry, V3/V4 equities path, OOF
  head-fill (L10).

## Files touched

See frontmatter. Key line anchors: watchdog retry loop
`oanda_scalper_orchestrator.py` (`_watchdog_close`), boot reconcile
(`_reconcile_on_boot`), liveness (`_check_stream_liveness`); provider
timeout/liveness `oanda_provider.py` (`__init__`, `run_stream`);
stale-feature guard `ml_strategy.py` (`generate_signals`); metals floor
`risk_manager.py` (`calculate_bracket`, `_is_metal_symbol`).
