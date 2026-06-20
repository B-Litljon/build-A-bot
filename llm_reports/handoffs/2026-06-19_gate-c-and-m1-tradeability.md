---
type: handoff
date: 2026-06-19
time: 02:31 PDT
agent: Claude Opus 4.8
model: claude-opus-4-8
trigger: User asked for a full-chat handoff so a fresh instance can continue — spans a suspected-bug recon, the spread-calibration soak, the Gate A investigation, the Gate C fix, and the M1-tradeability finding.
head: 4ff55c722b1b862120016996182e28bd9c1ab362
scope: modifies-source
related:
  - refactors/2026-06-14_dynamic-hybrid-chop-floor.md
files_touched:
  - src/execution/risk_manager.py
  - src/execution/oanda_scalper_orchestrator.py
  - src/core/retrainer.py
---

# Gate C (rollover blackout) + the M1-tradeability finding — session handoff

**Authored 2026-06-19 02:31 PDT · Claude Opus 4.8 (`claude-opus-4-8`).**
**⚠️ The Gate C code below is UNCOMMITTED in the working tree on `feature/dynamic-chop-floor` — HEAD is still `4ff55c7`.**

## Context
The session opened with the user suspecting a "bug." It was **not code**: a prior Claude
instance had told the user it was "monitoring the soak" when that soak (`soak_2026-06-10_1934.log`)
had cleanly ended **2026-06-12 20:42 PDT**. A confabulation seeded by a stale "active soak /
babysitting" memory description. The user killed that instance and started fresh. Lesson saved as
memory `feedback_no_confabulated_monitoring` (verify a job is alive via ps/pidfile/log-mtime before
ever claiming it runs). **Working agreement for this project: on-demand checks only — the user asks,
the agent verifies live and reports; NO autonomous loops / scheduled wakeups / idle token burn.**

The active work: a **V5 OANDA forex paper soak** (practice account, 8-instrument volatile basket,
M1) whose purpose was to **calibrate the placeholder `RiskProfile.spread_atr_alpha = 0.15`** from
real spread data via the `SPREAD_CALIB` sink (commit `4ff55c7`). Mid-session the soak revealed the
bot generates signals but the risk gates veto ~100% of them — which drove the Gate A investigation,
the Gate C fix, and ultimately the structural tradeability finding below.

## Investigation
- **Suspected import bug:** retrainer crashed with `ModuleNotFoundError: No module named 'data.feed'`.
  Root cause = mixed import conventions (`src.`-prefixed in core/ml vs **bare** `from data.feed import`
  in execution/strategies/~22 files). The wrappers set `PYTHONPATH=src:.` which resolves both; the
  crash was a missing-env invocation, not broken code. Latent footgun (see `reference_soak_runbook`).
- **Gate A trace** (`risk_manager.py` `_evaluate_dynamic_gates`): the `Gate A (cost/live)` veto uses
  the **live spread = `ask - bid`** (`oanda_scalper_orchestrator.py` tick handler), same **absolute
  price units** as `sl_dist = raw_atr × sl_atr_multiplier`. The alarming `spread=0.388` on GBP_JPY
  was a **real ~38-pip spread** — the signals fired at the **5 PM ET rollover (21:05 UTC)** when
  spreads blow out ~10×. **Not a units bug.**
- **Parquet check** (`data/cache/forex_365d_1m_6199416cc1.parquet`, 2.92M bars): the 16:55–17:30 ET
  blackout covers **1.60% of training bars**; the real OANDA daily candle gap is only **~4 min
  (17:00–17:03 ET)**. NOTE: my first two measurement scripts were buggy (reported "0%" then
  "16:00–17:59 empty"); a clean per-NY-hour histogram corrected it — *verify-by-rerun caught it.*
- **The key derivation:** `alpha_emp ≡ spread / ATR`. The cost gate passes iff
  `sl_dist ≥ k_eff · spread` → `sl_mult · ATR ≥ k_eff · alpha · ATR` → **`alpha ≤ sl_mult / k_eff`**.
  With `sl_mult = 1.0`, `k_eff = 1.5` → **tradeable only when `alpha ≲ 0.667`.**

## Findings / Changes
**Findings:**
1. **Gate C works; the 100% veto was correct, not a mistuned gate.** The early vetoes were
   rollover-timed (toxic spreads); the gate rightly refused.
2. **THE BIG ONE — on M1 the spread ≈ the 1-bar ATR for the fiat crosses, so the cost gate is
   structurally unclearable.** Converged `alpha_emp` (~29h soak):

   | instrument | alpha_emp | tradeable on M1 (≤0.667)? |
   |---|---|---|
   | XAU_USD | 0.30 | ✅ |
   | XAG_USD | 0.52 | ✅ |
   | EUR_JPY | 1.25 | ❌ |
   | GBP_JPY | 1.26 | ❌ |
   | AUD_JPY | 1.36 | ❌ |
   | GBP_AUD | 2.35 | ❌ |
   | NZD_JPY | 2.50 | ❌ |
   | GBP_NZD | 3.50 | ❌ |

   **Only the 2 metals clear the gate; the 6 fiat crosses cannot on M1.** This explains the
   zero-trade behavior across every soak. `alpha` is volatility-regime-dependent (majors ranged
   ~1.0 active-session to ~1.6 calm). The `0.15` placeholder was wildly low.

**Changes (Gate C — time-of-day blackout, UNCOMMITTED):**
- `risk_manager.py`: `RISK_TIME_GATE_ENABLED` (default on) + `RISK_BLACKOUT_ET` (default
  `16:55-17:30`) env knobs; `GATE_TIME`; `_NY_TZ` + `_parse_blackout_et`; `RiskProfile.blackout_start/
  end` + `time_gate_enabled` property; `_in_blackout()` (DST-correct via `America/New_York`); Gate C
  check at the top of `_evaluate_dynamic_gates` (before A/B); `timestamp` threaded through
  `calculate_bracket`.
- `oanda_scalper_orchestrator.py`: `GATE_TIME` import, `_time_gate_rejections` counter,
  `timestamp=datetime.now(timezone.utc)` passed to `calculate_bracket`, `time=` added to the
  bracket-rejected telemetry line.
- `retrainer.py`: `TODO(symmetry)` in `_compute_chop_veto_mask` — Gate C blackout is **not yet
  mirrored in training** (deferred; only ~1.6% of bars, near the academic end of the project's
  `<1% / >5%` asymmetry rule).

## Verification
- **31 tests pass** (`test_risk_manager.py` + `test_oanda_scalper.py`); no regressions.
- **DST sanity:** 17:05 ET fires in summer (21:05 UTC) *and* winter (22:05 UTC); the summer-UTC hour
  (21:05) correctly does NOT fire in winter → window tracks NY time, not a fixed UTC hour.
- **Gate C validated LIVE:** fired `2026-06-18 14:09 PDT` (17:09 ET) on a GBP_AUD rollover signal,
  bucketed correctly as `time=1`.
- **Structural finding cross-checked on real signals:** the `2026-06-18 17:37 PDT` cluster vetoed 4
  signals (EUR/GBP/AUD/NZD_JPY) at **normal** spreads (2.2–3.9 pips) because `sl_dist < 1.5·spread`
  — confirming the `alpha > 0.667` derivation outside the rollover.

## Risk & follow-ups  (← START HERE, next agent)
1. **DECISION NEEDED — how to make the 6 crosses tradeable** (or accept metals-only):
   (a) wider stops `sl_mult ≥ ~1.5·alpha` (~1.8 for GBP_JPY) — changes R:R;
   (b) **higher timeframe M5/M15** — ATR grows faster than spread, effective alpha drops below 0.667
       (likely the cleanest fix);
   (c) refocus the basket on low-alpha instruments (metals already qualify).
2. **Commit the Gate C change** — it's uncommitted on `feature/dynamic-chop-floor`.
3. **Next deliberate retrain** should bundle: the lever decision (1), the Gate C training-symmetry
   `TODO(symmetry)`, and the harvested `spread_atr_alpha` per-instrument values (table above; decide
   blended-24h vs active-session — units already verified comparable). Do NOT retrain mid-soak (hot-
   swaps the model). Build a candidate without clobbering prod via the `chop_ab_test.py` /
   `validate_candidate` path (writes nothing to `models/forex/`).
4. **Soak wind-down:** scheduled `at` job (Fri 2026-06-19 13:45 PDT → SIGTERM by pidfile; `atrm` to
   cancel) to beat the Friday close and avoid weekend no-tick churn. Calibration logs:
   `soak_2026-06-16_2344.log` (18h converged) + `soak_2026-06-17_2136.log` (current, Gate C run).

## Files touched
- `src/execution/risk_manager.py` — Gate C: imports, env constants, `GATE_TIME`/`_NY_TZ`/
  `_parse_blackout_et`, `RiskProfile.blackout_start/end` + `time_gate_enabled`, `_in_blackout()`,
  Gate C in `_evaluate_dynamic_gates`, `timestamp` in `calculate_bracket`.
- `src/execution/oanda_scalper_orchestrator.py` — `GATE_TIME` import, `_time_gate_rejections`,
  timestamp into `calculate_bracket`, `time=` telemetry.
- `src/core/retrainer.py` — `TODO(symmetry)` in `_compute_chop_veto_mask`.
- Read: `data/cache/forex_365d_1m_6199416cc1.parquet`; memory `project_v5_soak_2026-06-16`,
  `project_a3_chop_filter_blocker`, `reference_soak_runbook`, `feedback_no_confabulated_monitoring`.

## Addendum — events after authoring (2026-06-19 ~16:10 PDT)
1. **Soak wound down cleanly on schedule.** The `at` job fired at 13:45 PDT (queue now empty); the
   log (`soak_2026-06-17_2136.log`) shows `OandaScalperOrchestrator shutdown complete`, account flat,
   **0 executed trades**. Final converged alphas (n≈2,360) confirm the §Findings table — metals
   tradeable (**XAU 0.32, XAG 0.53**), all six crosses not (**GBP_JPY 1.38 … GBP_NZD 3.58**). Final
   veto tally: **7 Devil-approved signals, all vetoed (5 cost / 1 regime / 1 time)**. → §Risk item 4
   is DONE, not "scheduled."
2. **New: read-only MCP observability server `trading_mcp.py` (repo root, UNCOMMITTED).** Wraps the
   manual check-ins as 3 typed tools — `soak_status`, `calibration`, `gate_activity` — usable by any
   MCP client (Claude Code / OpenCode / Agent SDK). `mcp[cli]` is installed in the venv; register via
   `claude mcp add trading-ops --scope project -- <venv-python> /mnt/storage/mystuf/development/build-A-bot/trading_mcp.py`
   (venv python: `/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python`).
   Strictly read-only/additive — does not touch the bot. **Priority unchanged: get the strategy
   trading (§Risk item 1) before building more harness.**
3. **§Risk "START HERE" decision is unchanged** — pick the lever for the 6 crosses (user leaning
   metals-only; weigh the sample-size-floor caveat from `project_a3_chop_filter_blocker`).
