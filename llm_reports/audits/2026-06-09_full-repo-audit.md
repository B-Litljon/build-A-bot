---
type: audit
date: 2026-06-09
time: 17:06 PDT
agent: Claude Fable 5
model: claude-fable-5
trigger: User asked for a full repo audit — critical issues, small bugs, edge cases
head: f6330db4def0f82dafc42b08845ff6aaa2544c50
scope: read-only
related:
  - audits/2026-05-24_lightgbm-pilot-soak-readiness.md
  - refactors/2026-05-25_configurable-chop-floor.md
---

# Full repo audit — V5 OANDA scalper path + supporting modules

## Context

Brandon asked for a whole-repo audit: critical issues, small bugs, edge
cases. The active money path is the V5 OANDA forex scalper
(`run_oanda.py` → `OandaScalperOrchestrator` → `MLStrategy` →
`OandaOrderManager`), so that path was read line-by-line. The retrainer,
risk manager, feature pipeline, and notification manager were read in
full or in load-bearing part. The dormant V3/V4 equities path
(`live_orchestrator.py`, 2,392 lines) got only a targeted skim — it is
disabled and dormant per project state.

## Investigation

Read in full: `run_oanda.py`, `src/execution/oanda_scalper_orchestrator.py`,
`src/execution/oanda_order_manager.py`, `src/data/oanda_provider.py`,
`src/strategies/concrete_strategies/ml_strategy.py`,
`src/execution/risk_manager.py`, `src/core/notification_manager.py`,
`src/strategies/base.py`, `src/ml/feature_pipeline.py`,
`src/ml/features/v3_features.py`, `src/ml/trainers/v3_rf_trainer.py`.
Read in part: `src/core/retrainer.py` (asset config, target generation,
`refit_models`, `main`). Verified the promoted model's feature schema by
loading `models/forex/angel_latest.pkl` / `devil_latest.pkl` directly
(LGBMClassifier; 22 base features, Devil = base + `angel_prob` appended
last — matches live construction order). Checked git history for the
chop-floor lineage (`d92202f` → `d40827a` → `f6330db`), checked `.env`
keys (redacted) and systemd units, and ran the test suite.

## Findings / Changes

### Critical — live money path

**C1. Failed watchdog close silently orphans a position.**
`OandaOrderManager.close_position` (`oanda_order_manager.py:218-226`)
catches all exceptions and returns `False` — it never raises. The
watchdog (`oanda_scalper_orchestrator.py:163-186`) ignores the bool
return, so its `except` branch is unreachable for API failures: a failed
close logs `"Watchdog close completed"`, and the `finally` block pops the
symbol from `self._positions` regardless. Net effect: if the close HTTP
call fails (timeout, 5xx, rate limit), the position stays open at OANDA
with **no SL/TP monitoring at all** (brackets are software-only by
design) and no local record it exists. The Discord alert also reports a
successful close. Fix: have the watchdog check the return value (or make
`close_position` raise), keep the position in `PENDING_CLOSE` on failure,
and retry with backoff.

**C2. No broker reconciliation at startup.**
The orchestrator never calls `OandaOrderManager.sync_position` —
`sync_position` exists but has zero call sites in the live path. After a
crash (or C1), a restart boots with empty `_positions` and empty
`_net_positions`: the pre-existing broker position is unprotected, and
the next signal computes `delta = target - 0`, **stacking units on top of
the unknown position**. Fix: call `sync_position` for every symbol during
`run()` startup and either adopt or flatten non-flat positions.

**C3. Silent stream stall leaves positions unprotected indefinitely.**
`run_stream` (`oanda_provider.py:332-356`) iterates the PricingStream
with no read timeout and no heartbeat watchdog. OANDA sends HEARTBEAT
messages every ~5s; the code discards them without tracking arrival time.
If the TCP connection stalls without raising (silent half-open
connection), the `for msg in ...` loop blocks forever, the reconnect
wrapper in `_stream_with_retry` never fires, and — because SL/TP is
software-only — open positions have no stop. Fix: track
last-message-received time (heartbeats included) and have a watchdog task
on the asyncio loop force-reconnect (and optionally flatten) if the
stream goes quiet for >N seconds.

**C4. Blocking Discord webhooks run on the event loop.**
`send_oanda_trade_alert` and `send_system_message` use synchronous
`requests.post(timeout=5)` (`notification_manager.py:146,79`) and are
called directly from `_on_bar` and `_watchdog_close` — both coroutines on
the event loop. A slow/unreachable webhook stalls the loop up to 5s,
during which queued watchdog closes and bar processing for all symbols
wait. For a 1-minute scalper with tick-driven exits this is a real exit
latency hazard. Fix: wrap posts in `run_in_executor` (or a fire-and-forget
thread).

### High

**H1. Reversal records 2× the actual position size.**
On a flip (long +1000 → short signal), `submit_target_position` trades
delta −2000 and returns `filled = abs(fill_units) = 2000`
(`oanda_order_manager.py:306,349`). The orchestrator records
`actual_units = -2000` (`oanda_scalper_orchestrator.py:339`) when the
true net is −1000. Sign is right so the watchdog still works and
`close_position("ALL")` still flattens, but local state, logs, and
Discord alerts misreport size 2×. The returned `avg_price` is also the
whole-order fill price rather than the opened leg (the manager's internal
`_avg_entry_prices` handles this correctly; the return dict doesn't).
Fix: return `opened_units`-based values for position recording — the dict
already contains `opened_units`; the orchestrator just ignores it.

**H2. Race between signal entry and watchdog close → state drift.**
`_on_bar` checks `PENDING_CLOSE` early (`oanda_scalper_orchestrator.py:256-263`),
then spends 100ms+ in feature gen + order submission. A tick can breach
SL in that window: the watchdog closes the *old* position and pops the
dict, while `_on_bar`'s in-flight order fills and overwrites
`_positions[symbol]` with `state: OPEN`. Depending on interleaving at the
broker, you can end up flat-at-broker/OPEN-locally or vice versa.
Low probability per trade but nonzero on every entry near a stop. A
monotonic position epoch (or re-checking state after the fill, under the
lock, and syncing on mismatch) would close it.

**H3. Stale-feature signal when the latest bar is dropped by clean_data.**
`FeaturePipeline.clean_data` drops any row with null/NaN/Inf in the
feature subset (`feature_pipeline.py:72-73`). In `generate_signals`,
features come from `features_df.tail(1)` but `current_price` and
timestamp come from the *raw* df tail (`ml_strategy.py:406-410`). If the
newest bar produced a null/Inf feature (flat-volatility bar, division
guard miss), the model silently scores the *previous* bar's features
against the *current* price and trades on it. Fix: compare the feature
row's timestamp to the raw tail timestamp and return None on mismatch.

**H4. Bare `pytest` from the repo root fires live side effects.**
`test_discord.py` posts to the production Discord webhook at module
import (module-level code, no `__main__` guard); `test_alpaca.py` /
`test_speed.py` similarly hit real APIs on import. There is no
`pytest.ini`/`pyproject.toml`, so plain `pytest` collects root
`test_*.py` files and *imports* them — sending a webhook ping and burning
API calls. `pytest tests/` is safe; bare `pytest` is not. Fix: add
`testpaths = tests` config, or rename the root scratch scripts (they are
manual probes, not tests).

### Medium

**M1. Chop-filter floor default is 10× the empirically validated value.**
History: `d92202f` set the working floor to `min_sl_pct=0.00002`
(~0.2 pips on EUR/USD) after 0.15% blocked everything; `d40827a`
(2026-05-22) replaced it with `min_sl_pips=2.0` — 10× tighter — and
`f6330db` made it env-tunable. `.env` does **not** set
`RISK_FOREX_MIN_SL_PIPS`, so live runs at 2.0 pips. For the actual traded
basket (metals + JPY crosses) this is plausible, but it is exactly what
the A3 rejection counter (`82dbdba`) was built to watch. The
`project_a3_chop_filter_blocker` memory still describes the 0.00002
state and is stale.

**M2. Pip size is meaningless for metals — chop filter is a no-op there.**
`_is_forex_symbol("XAU_USD")` → True (6 alpha chars), quote "USD" → pip
0.0001 (`risk_manager.py:97-106`). Floor = 2 × 0.0001 = 0.0002 on gold at
~$2,700 — never vetoes. Same for XAG_USD. Half the trained basket is
metals, so the chop filter effectively only exists for the JPY/AUD/NZD
crosses. If that's intended, document it; otherwise metals need a
percent-based floor.

**M3. Live/training volume skew in the tick aggregator.**
`_handle_tick` initializes a new bar with `volume: 0` and increments only
on subsequent ticks (`oanda_provider.py:178-191`), so live bar volume =
tick_count − 1 and single-tick bars have volume 0. Training data uses
OANDA REST candle `volume` = full tick count. `vol_rel` and `htf_vol_rel`
are model features, so quiet-session bars see systematically deflated
values vs. training. One-line fix: initialize `volume: 1`.

**M4. Default symbol isn't in the trained basket, and metadata doesn't catch it.**
`run_oanda.py` defaults to `EUR/USD`, but `models/forex/metadata.json`
shows the promoted model trained on XAU/XAG/JPY-cross/GBP-cross
instruments only. `_validate_metadata` checks `asset_class` but not
instruments, so launching with defaults silently trades an
out-of-distribution pair. Fix: default `OANDA_SYMBOLS` to the trained
basket, and/or warn when a requested symbol is absent from
`trained_on_symbols`.

**M5. Watchdog Discord alert lies on failure (corollary of C1).**
`_watchdog_close` sends `action="WATCHDOG_CLOSE"` with reason "SL or TP
breach detected" even when the close failed; price field is the *entry*
price, labeled "Price" — misleading during incident review.

**M6. Hot-reload races and non-atomic pair swap.**
`_check_model_updates` reloads Angel under `_reload_lock` but Devil
without it (`ml_strategy.py:333`), and `generate_signals` never takes the
lock — today it's safe only because everything runs on one event-loop
thread. Separately, Angel and Devil are checked/swapped independently:
if a retrain's two `os.replace` calls straddle a bar, one bar can be
scored by new-Angel + old-Devil. Also, `self.feature_names` is *not*
refreshed on reload — a retrain that changes the feature space (e.g.
enabling HMM) would mispredict until restart rather than fail loudly.

**M7. Stray `except`-path NameError in `generate_signals`.**
The handler at `ml_strategy.py:499` logs `f"[{symbol}] ..."`, but
`symbol` is assigned inside the try (line 414). Any exception before that
line (e.g. a missing feature column in `features_df[self.feature_names]`)
raises `NameError` *inside the except block*, masking the original error.

### Low / hygiene

- **L1.** Stray junk file at repo root: `ystemctl --user disable --now
  universal-scalper.service` (captured `systemctl status` output from a
  March shell-redirect typo). Safe to delete.
- **L2.** `~/.config/systemd/user/universal-scalper.service` ExecStart
  points at `.venv/bin/python`, which doesn't exist (project venv is the
  pipenv one under `~/.local/share/virtualenvs/`). Service is disabled;
  if ever re-enabled it will fail to start.
- **L3.** `pytest` is not in `Pipfile` `[dev-packages]` — the suite (26
  tests, all passing) can't run from a fresh `pipenv install --dev`. I
  installed pytest into the pipenv venv during this audit.
- **L4.** `feature_pipeline.py:24` calls `logging.basicConfig` at import
  — configures root logging as a side effect for every consumer,
  including the live bot.
- **L5.** `_prime_history` uses `assert row["timestamp"].tzinfo is not
  None` (`oanda_scalper_orchestrator.py:406`) — vanishes under `python -O`;
  should be an explicit check if it matters.
- **L6.** `datetime.utcnow()` deprecation in
  `notification_manager.py:139` (warning visible in the test run).
- **L7.** `_load_threshold`'s warning message says
  `models/threshold.json` but the actual path is
  `models/<asset_class>/threshold.json`.
- **L8.** `_stream_with_retry` reaches into the provider's private
  `_stop_event` (`oanda_scalper_orchestrator.py:446`); deserves a public
  `reset()`.
- **L9.** `run_oanda.py --granularity` accepts any int but only values in
  the provider's `_GRANULARITY` map work; e.g. `3` degrades to cold
  warm-up (history fetch raises, is caught and logged as a warning).
- **L10.** Retrainer OOF "head fill" (`retrainer.py:790-805`) scores the
  first-fold rows with a model trained on those same rows — in-sample
  angel_prob for the head ~17% of Devil training data. Acknowledged in
  comments; time-decay weights mitigate.
- **L11.** `stop_stream` flushes in-flight bars from the caller's thread
  while the stream thread may still be mutating `_tick_bars` — benign
  shutdown-only race.

### Confirmations (things that look wrong but aren't)

- `direction="long"` hardcoded in `MLStrategy` is consistent with the
  retrainer's long-only target construction (`SL = close − …`,
  `TP = close + …`). The orchestrator's short-side handling is dead but
  correct code.
- Training/live bracket multipliers are symmetric: both sides source
  `RiskProfile.for_asset_class` (`retrainer.py:91,107-108`). The known
  asymmetry remains only the A3 chop filter itself (documented, telemetry
  in place).
- Devil feature ordering matches: training uses
  `feature_cols + ["angel_prob"]`; live appends `angel_prob` last to a
  frame ordered by the model's own `feature_names_in_`.
- `save_models` uses temp-file + `os.replace` — hot-reload can't read a
  torn pickle.
- `.env` is git-ignored and not tracked; no secrets in the repo
  (157 tracked files checked).
- HTF `available_at` join correctly prevents lookahead live: the
  in-progress HTF bucket's `available_at` is in the future, so the
  backward as-of join skips it.
- Seam logic deliberately drops one possibly-good bar after reconnect —
  conservative, fine.

## Verification

- Test suite: `26 passed` via the pipenv venv
  (`~/.local/share/virtualenvs/build-A-bot-A3hTUWzK`), 3.66s. One
  RuntimeWarning (`_watchdog_close` never awaited) comes from a mocked
  test, not production code.
- Model schema claims verified by loading both pickles with joblib and
  printing `feature_names_in_`.
- Chop-floor lineage verified via `git show d92202f` / `d40827a` /
  `git log -S`.
- `.env` keys listed with values redacted; `git ls-files` and
  `git check-ignore` confirmed untracked.
- C1 traced by reading both sides of the call: `close_position` catch-all
  return False + watchdog's unused return + `finally: pop`.

## Risk & follow-ups

1. C1–C4 should be fixed before the next soak leg, C1/C2 especially —
   they convert a transient API failure into an unmonitored live
   position. Together they'd make a tight, single-PR hardening pass.
2. Decide M1 (floor default) with soak telemetry; update or retire the
   stale `project_a3_chop_filter_blocker` memory.
3. M3 (volume skew) is a one-line fix but changes live feature
   distribution — fix it and note it in the soak log rather than
   mid-soak silently.
4. H4 is a 3-line `pyproject.toml` fix; do it before anyone runs bare
   `pytest`.

## Files touched

None (read-only audit). Files read:

- run_oanda.py (full)
- src/execution/oanda_scalper_orchestrator.py (full)
- src/execution/oanda_order_manager.py (full)
- src/execution/risk_manager.py (full)
- src/data/oanda_provider.py (full)
- src/strategies/concrete_strategies/ml_strategy.py (full)
- src/strategies/base.py (full)
- src/core/notification_manager.py (full)
- src/ml/feature_pipeline.py (full)
- src/ml/features/v3_features.py (full)
- src/ml/trainers/v3_rf_trainer.py (full)
- src/core/retrainer.py (lines 83-180, 375-900, 1709-1834)
- test_discord.py, test_alpaca.py (headers)
- models/forex/{metadata,threshold}.json, Pipfile, .env (keys only)
