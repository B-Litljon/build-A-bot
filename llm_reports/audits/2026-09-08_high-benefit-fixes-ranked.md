---
type: audit
date: 2026-09-08
time: 23:58 PDT
agent: z
model: kimi-k3 (via dsh, ollama-cloud route)
trigger: User asked for a project assessment and a ranked list, easiest-first, of high-benefit improvements (bug fixes, data changes)
head: 6390128db7ec566f816c9dae2551641c591d74b2
scope: read-only
related:
  - audits/2026-08-24_holdout-gate-audit.md
  - m2m/2026-08-29_trim-100x15-retrain-and-promotion.md
  - recons/2026-09-02_strategy-library-no-edge.md
---

# High-benefit fixes for the bot, ranked easiest → hardest

## Context

Brandon asked for an assessment of the whole project and a list, ordered from
easiest to hardest, of changes that would hugely benefit the bot — bug fixes or
data changes. The M15 forex soak is live on the OANDA practice account
(soak.service, PID 3365348, up since 2026-08-30, `models/forex_m15_wide`),
stops are software-enforced by the running process, and nothing was modified
in producing this report.

## Investigation

Four parallel read-only audits, each verified by reading the code (and, where
load-bearing, running it):

1. **Execution layer** — `oanda_forex_orchestrator.py` (2098 lines),
   `oanda_order_manager.py`, `risk_manager.py`, `live_orchestrator.py`,
   `run_oanda.py`, `soak_service`/`run_soak.sh`/`soak_watchdog.sh`.
2. **Decision path** — `ml_strategy.py`, the strategy library +
   `regime_router.py`, `v3_features.py`, `hmm_regime.py`, `behavior_tagger.py`,
   `oanda_provider.py`, `bar_aggregator.py`, `thresholds.py`, `events.py`,
   plus byte-level inspection of the served pickles (Angel = 17 features =
   `BASE_FEATURE_COLS`; Devil = those + `angel_prob`).
3. **Training pipeline** — `retrainer.py` (3867 lines), `resolver.py`,
   `feedback_loop.py`, `replay_test.py`, `evaluate_performance.py`,
   `ml/barriers/`, `run_pipeline.sh`. The Clopper-Pearson bound and the
   TimeSeriesSplit fold composition were reproduced numerically.
4. **Ops/runtime** — journalctl, watchdog.log, 9.5 days of the live soak log,
   `status.json` freshness, data/ and models/ artifact state with mtimes,
   spread-table provenance, dashboard/trading_mcp, and a full test run.

Live facts established: 9.5 days up (one clean, deliberate restart for the
Aug-29 model deploy); ~340 broker-side stream errors all self-healed;
**zero trades in 9.5 days** — all 3 Devil-approved signals were gate-vetoed,
and the Angel has not been near its bar (live max ≈0.27 vs threshold 0.3833);
test suite = **411 passed / 5 skipped** under the project venv
(pipenv python 3.12), but fails collection under system python 3.14
(missing deps — the documented command is environment-misleading).

## Findings / Changes (the ranked list)

Severity tags: **[M]** money-losing · **[S]** safety (unwatched-position risk) ·
**[C]** correctness · **[D]** data · **[H]** hygiene. Effort = realistic
implementation time including tests, not just lines.

### Tier 0 — command/data/doc-level, minutes

**1. Retire the stale offline artifacts. [D]** `data/signal_ledger.csv`
(Feb 20 — read only by the self-declared-orphaned `src/core/resolver.py:73`),
`data/drift_report.json` (Feb 20 — still carries a 6.5-month-old
`"safety_switch_recommended": true` alarm at equities era), plus
`oos_bars.parquet` / `evaluation_results.parquet` (both Feb 20). Meanwhile
replay/eval use `data/signal_ledger.parquet` (Mar 13). The trap: anyone who
runs `python -m src.core.resolver` grades 5-month-old signals against current
bars and **writes `data/resolved_ledger.csv` — the exact file
`feedback_loop.py:73` reads** → a garbage drift verdict. Also archive the root
`models/angel_latest.pkl` / `devil_latest.pkl` (May-23 equities pair that
`replay_test.py` loads) so the replay arm fails loudly instead of pretending
to grade the live model. Confirm-before-delete per house rules.

**2. Re-bake the spread table — and notice what is actually served. [D][M]**
`scripts/bake_spread_alphas.py` last ran **Jul 7**; the resulting
`config/spread_alphas_m15.json` never reached the model dir, so **live Gate A's
stale-spread proxy runs on flat `spread_atr_alpha = 0.15`**
(`risk_manager.py:250`) while the soak's own SPREAD_CALIB lines measure
**0.34–0.89** (EUR_JPY 0.38, GBP_AUD 0.69, GBP_NZD 0.89). That understates the
toll 2.5–6×, always in the too-permissive direction (it only bites in
stale-spread windows — fresh tick spreads override). The Jul table itself is
mostly still accurate (within each symbol's own 9-day drift band) **except
GBP_AUD: 0.584 baked vs 0.675 live floor — ~16% permissive.** One command to
re-bake; *serving* it changes Gate A without retraining, so pair it with the
next retrain (`RETRAIN_SPREAD_TABLE`) to preserve the symmetry contract.

**3. Fix docs that now mislead. [H]** `GLOSSARY.md:310` says the live feature
set is 22/23 columns — the served pair is **17/18** (5 features dropped in the
Aug-29 trim); `CLAUDE.md` says "200 tests" (416 collect; and it needs the
venv python); `soak.service`'s "MUST NOT serve `models/forex_m15`" warning
references a directory that no longer exists (good — but the warning should
say the danger is *copying an old model into* `forex_m15_wide`), and
`table-o-content.md` references three deleted files. Also: the glossary's
"2% equity risk per trade" and "$50 notional floor" are **not enforced on the
live forex path** — `units_per_trade` is fixed at 1000 (`run_oanda.py:143-147`,
`oanda_forex_orchestrator.py:28-29`) and `RiskManager.calculate_quantity`
(where the floor lives, `risk_manager.py:535-536`) is never called by it;
deliberate per the glossary, but the docs oversell the sizing. Commit or
ignore ` M logs/graded_decisions.parquet` and the 5 untracked research
outputs, and silence the test warning
`coroutine '_watchdog_close' was never awaited`.

### Tier 1 — one-to-ten-line source fixes, no retrain

**4. Guard the shutdown flush. [S][M]** `oanda_provider.py:495-500` pushes the
partial, unsealed bar into `_on_bar` on `stop_stream()`; during SIGTERM the
orchestrator can *open a new position after `_flatten_all`'s snapshot* — the
process exits with an unwatched live position (orchestrator `:1091`). One
line: `if self._shutdown_event.is_set(): return` at the top of `_on_bar`,
plus ignore symbols not in the constructor basket (`:1115-1117` currently
crashes the bar pipeline on a KeyError, swallowed by the future). Same file
area: reorder `_handle_tick` (`oanda_provider.py:243-257`) to build-then-flush
so a raising flush can't wedge an instrument's bar pipeline forever.

**5. Revive the dead drift→retrain trigger. [D]** `run_pipeline.sh` Phase 3
writes `data/evaluation_results.parquet`; Phase 4 `feedback_loop.py:73`
hardcodes `RESOLVED_PATH = data/resolved_ledger.csv` — a file *nothing writes*
(it does not exist on disk). The offline loop's retrain branch can therefore
**never fire; it can only exit 1.** One line (plus a column map:
`exit_type WIN/LOSS` → outcome) turns the documented harvest→replay→grade→
drift→retrain loop back on.

**6. Make the offline grader grade the instrument live actually trades. [C]**
`evaluate_performance.py:71-72` hardcodes the **equities** brackets 0.5×/3.0×
(live forex is 2.0×/4.0× and the file's own docstring says they must match),
and `:309-317` checks **TP before SL** intra-bar while the trainer and
resolver check SL first — both-touch bars (common at 2.0/4.0 M15) are graded
WIN offline and LOSS in training. Two small edits: multipliers from
`RiskProfile.for_asset_class(...)`, swap the two `if`s. Its conviction filter
(`:229-233`) also reads the stale Feb-20 `drift_report.json` (item 1).

**7. Threshold sidecar precision + timestamp. [C]** `retrainer.py:3466` rounds
the pinned Angel bar (`0.38330523…` → `0.3833`), so the live bar sits ~5e-5
looser than the calibration population; `:3467` writes a naive-local
`updated_at` while `metadata.json` is UTC (the served dir's threshold.json
looks 7 h older than its own models). Two one-liners.

**8. Enforce brackets in `_validate_metadata`. [C][S]** `ml_strategy.py:327-357`
checks only `asset_class`. The retrainer writes `sl_atr_multiplier`/
`tp_atr_multiplier` into metadata precisely so train/serve skew is detectable
— nothing compares them. ~6 lines: raise when present-and-mismatched vs
`RiskProfile`. This is the guard for the exact failure the soak.service
comment warns about (serving a 1.0×/2.0×-trained model under a 2.0×/4.0× tree).

**9. Test pins for two silent-drift surfaces. [C]** (a) Zero tests cover Gate
C's NY-timezone blackout/DST logic (`risk_manager.py:467-482`) — the one
timezone-sensitive gate. (b) `run_oanda.py:94-98 _GRANULARITY_PROFILES` and
`retrainer.py:413 _HTF_FOR_TIMEFRAME` are hand-copied duplicates — a two-line
test importing both locks the HTF pairing that every `htf_*` feature depends
on.

### Tier 2 — small patches (~15–40 lines each + tests), same-day

**10. Watchdog staleness check. [S]** `soak_watchdog.sh:61` checks pgrep only.
A wedged event loop (deadlock in `_positions_lock`, stream stalled without
disconnect) satisfies pgrep forever while positions sit unenforced — the
failure CLAUDE.md calls the worst case. The dashboard already computes
freshness (`dashboard/status.rs`); the watchdog should too: on weekday ticks,
if `logs/status.json` mtime is stale > ~15 min, log + ntfy; escalate to
restart after N stale ticks. ~15 lines of shell. Bundle: fix the `crontab`
**PAM lockout** found during the audit (the schedule is currently invisible/
uneditable from the CLI — the watchdog fires, but only the journal proves it;
moving to a systemd user timer per the `mem-watchdog.timer` template fixes
both).

**11. Mirror Gate C in training. [C][D]** Symmetry-contract violation:
`_compute_chop_veto_mask` (`retrainer.py:1190-1269`) mirrors Gates A and B
exactly (verified line-by-line against `risk_manager.py`) but **has no Gate
C** — the module's own TODO admits it. Live vetoes every 16:55–17:30 NY entry
(DST-correct); the Devil trains on rollover entries live never takes, and the
fold/holdout PF that promotes models includes the worst-fill window. ~15
vectorized lines (`timestamp.dt.convert_time_zone("America/New_York")` vs
`profile.blackout_start/end`, OR into the veto), automatically covering folds
and holdout.

**12. Hot-reload consistency bundle. [C][M]** Three coupled seams around model
promotion: (a) `ml_strategy.py:540-583` re-reads `threshold.json` / spread
table **only when a pickle's mtime also changed** — the retrainer writes
pickles *then* thresholds, so a bar landing mid-promotion makes the new pair
run at the old bars *forever* (persistent, silent). Fix: track each sidecar's
own mtime. (b) `ml_strategy.py:472-511` reloads Angel and Devil independently;
a bar between the two `os.replace`s trades a **mixed generation**, and the
consistency check (`:527-538`) only compares feature-name lists — a same-schema
stale Devil passes silently. Fix: pair-seam stand-down flag, return None until
both land. (c) Retrainer write order (`retrainer.py:3252-3253, 3328-3353`):
dump both pkls, write threshold.json, *then* the two replaces back-to-back.

**13. Harden `_prime_history`. [S]** `oanda_forex_orchestrator.py:1709-1730`:
an empty/raise during history prime is logged and skipped — with buffers just
cleared on reconnect, that means ~**2.75 days of silently trading nothing**
(260-bar M15 warmup) while liveness looks fine (the provider returns empty
for any API error and never raises: `oanda_provider.py:381-387`). Compounding:
the naive-tz raise at `:1726-1730` is unguarded at `:1875` — one throw kills
the reconnect task permanently (process alive → watchdog never relaunches), and
at shutdown `wait_for(self._stream_task)` (`:2018-2026`) can re-raise and skip
the SIGTERM flatten. Fix: bounded retry + CRITICAL-on-empty + guard the raise
+ wrap the shutdown stream-wait.

**14. Point `replay_test.py` at the served model. [D]** It loads the May-23
root pkls, the module `ANGEL_THRESHOLD` constant (0.40) instead of the pinned
0.3833, `V3HTFFeatures(timeframe="5m")` (live is 1h), and its `FEATURE_NAMES`
omits the four `session_*` flags in `BASE_FEATURE_COLS` — even aimed at the
served model it would **crash** on a missing column. Fixes: model dir from
`OANDA_MODEL_DIR`, both keys from that dir's `threshold.json`,
`FEATURE_NAMES := retrainer.BASE_FEATURE_COLS`, HTF from granularity. (Half of
this is fixed for free by item 1's archival.) Also `:475-477`: a zero-signal
run should delete the stale ledger before the early return, not leave it for
the grader.

**15. Fix the post-exit cooldown race. [C]** `oanda_forex_orchestrator.py:1288-1304`
reads the position snapshot once, then evaluates the cooldown at `:1303`
against the stale copy; a tick-watchdog stop-out in between lets the symbol
re-enter on the very bar that stopped out — the exact 2026-07-30 triple-JPY
pattern the guard exists to prevent. Re-check `_last_exit_ts` inside the
second lock block (`:1392-1414`).

### Tier 3 — half-day, money-safety (touch the tick/close paths; test)

**16. Verify closes; never strand a failed exit. [M][S]** Three linked defects:
(a) `_watchdog_close` (`:822-842`) pops its record on `close_position`'s True,
but `close_position` returns True after *submit* — a cancelled FOK (halt,
rollover blowout — exactly when breaches happen) means the live position loses
both its software stop and its tracking record. Fix: sync + require verified
net==0 before popping; treat cache-flat as "verify first, not success"
(`oanda_order_manager.py:327-376`). (b) `_flatten_all` (`:2086-2098`) clears
**all** records even when closes failed — then keeps trading, possibly over
the unwatched broker position. Fix: clear only successes, park failures
(CLOSE_FAILED exists), skip symbols with active `_pending_entries`, leave
`ENTRY_UNRECONCILED` for the reconciler. (c) CLOSE_FAILED is terminal for the
run (`:877-904`) — a failed stop-exit is never retried for the remaining life
of the process (days). Fix: retry CLOSE_FAILED closes from the liveness loop
(every 10 s, `:1920-1937`). One ~30 s OANDA blip today = an unenforced real
position for days.

**17. Close the liveness blind windows. [S][M]** The documented invariant
"dead feed ⇒ hold nothing" is not enforced on any reachable path: the
provider's `finally` nulls `_last_stream_msg`, and `_check_stream_liveness`
returns early on None (`:1894-1896`) — so every real disconnect (the common
case; the 20 s read timeout fires before the 60 s staleness threshold) silently
disables the flatten backstop. If REST is also down, backoff loops forever:
no ticks, no flatten, no alert, watchdog sees a live process. Second window:
the age counts heartbeats, so "connection alive, prices missing" never flattens.
Fix: provider tracks `stream_down_since`; orchestrator flattens (or minimally
ntfy-alerts) when positions exist and stream-down > `_stream_stale_seconds`;
measure price silence, not message silence. Note: `test_stream_liveness.py:102`
pins the current no-op — fix means consciously changing that test. This is the
single largest live-safety design gap found.

**18. Re-anchor brackets on the actual fill. [M]** Brackets are sized from the
signal's bar-close (`:1323-1344`) while the real fill (`avg_price`,
`:1491-1519`) differs — small on the stream path, material on catch-up/backfill
entries evaluated up to 15 min late (extreme: stop on the wrong side of the
market — a mode `_reconcile_unverified_entries` already acknowledges and fixes
for parked entries at `:1037-1042`). Fix: recompute `sl/tp` from
`result["position_avg_price"]` preserving the approved distances, and verify
`position_units == target_units` post-submit (else park). Related race
(`oanda_order_manager.py:434-444` + orchestrator #6 in the detail): an in-flight
entry racing a flatten can leave 2× units; and a post-ambiguous re-sync can
read pre-fill state → same-delta retry → double fill — add a settle delay or
carried `clientRequestID` (verify OANDA's duplicate-ID semantics first).

### Tier 4 — small code, but retrain-gated

**19. Fix the OOF leakage in the Devil's training data. [C][D] — highest
training-correctness impact found.** `fetch_training_data` returns the frame
sorted `["symbol","timestamp"]` (`retrainer.py:970`); `refit_models` then runs
`TimeSeriesSplit` over **row index** (`:1649, :1655`) believing rows are
chronological. Verified numerically: each val fold is one or two *whole symbol
blocks*; the "OOF" Angel probabilities for late-listed symbols come from models
trained on **other symbols' full history including dates after the scored row**.
On correlated FX pairs that is genuine future leakage, and it contaminates
exactly the rows the Devil trains on (`:1755-1794`) *and* the out-of-fold
distribution `_find_optimal_angel_threshold` calibrates the production bar from
(`:1725`, pinned into threshold.json) — while live the Devil receives honest
scores. The 2026-09-06 graded-decision report shows the top Angel band
inverting (0.40+ → 13.2% win on n=38) — whether that inversion is a leakage
artifact is a **hypothesis**, but the fix is ~6 lines regardless:
chronologically permute (`np.argsort(timestamp)`) around the OOF loop, write
probs back to original indices; the head-fill model at `:1675-1690` scores its
own training rows (fix the comment/re-use). Same root cause: **time-decay
weights encode basket position, not recency** (`:1517-1542` `weights[::-1]` on
row index → symbol 0's *newest* bar gets 0.1 weight, symbol 7's oldest-era
ordering gets 1.0) — derive from timestamp rank instead. Landing the benefit
requires a retrain; the fix should go in *before* the next one.

**20. Fix the fold-gate's EV semantics. [C]** `retrainer.py:2971-2978` (and
holdout `:2207-2208`) compute "EV" from the **5-bar survival** win rate mapped
through the **45-bar macro** R:R — survival is not the complement of macro
loss, so the value is inflated and the gate is near-vacuous (served artifact:
recorded EV 1.5833 vs bar 0.0005; it cannot fail). `_find_optimal_threshold`
already does it correctly with macro targets (`:1901-1908`). Also Folds 1–2
sweep their threshold on the same window their EV is reported on
(`:2829-2840`) — winner's curse on 2 of 3 folds; freeze from Fold 1 like Fold 3
already is. Same retrain batch as item 19.

### Tier 5 — decisions/research, not fixes

**21. The economics question is now the binding constraint.** The graded
live-decision ledger (14,634 decisions, 2026-07-31 → 2026-09-04) shows net EV
negative at *every* Angel threshold (0.2 → −0.27R … 0.5 → −1.1R), base rate
26.1% vs the 2:1 bracket's 33.3% break-even (net of 0.10R toll), and a 9.5-day
soak with zero proposals (max angel ≈0.27 vs bar 0.3833). Some of this may be
explained by item 19's calibration contamination, but the quantile-barrier
module (`ml/barriers/`, merged 2026-09-07) exists precisely to re-ask "what
excursion does this market actually offer" — nothing live consumes it yet.
Deciding where the next evidence comes from (retrained calibrated Angel vs
barrier-driven bracket redesign) outranks any single code fix above Tier 3.

**22. The strategy library has no measured edge.** All routing cells are
`null` (stand-down) on both the H1 and H4 matrices (untracked
`config/regime_routing_forex_H{1,4}.json`, Sep 7) — matching commit 7007687.
The router is correctly refusing to trade; what it needs is one strategy with
a positive edge in one regime, which is research (stage briefs in
`llm_reports/m2m-prompts/`), not a patch.

**23. Dormant Alpaca path (`live_orchestrator.py`). [S if revived]** Failed
manual exit strands PENDING_EXIT forever; exit-order cancel/expire events
clobber live positions to FLAT while wiping SL/TP; shutdown cancels orders but
never flattens (stops are software-only there too); no PENDING timeout.
Do not revive this path live without addressing those.

**24. Ops structure, low urgency.** `run_soak.sh` has no singleton guard
(double-launch = double trading) and `/tmp/soak.pid` is PID-recyclable
(`trading_mcp.py:98` bare `os.kill(pid,0)` can report a dead soak alive);
watchdog crash-loop state lives in /tmp and never escalates (dying every 10
min = silent relaunch every 15 min forever); `logs/soak_*.log` accumulates
forever; the Discord promotion embed never shows the holdout block (the gate's
strongest evidence) and quotes the point PF where the gate judges the CP bound;
`run_pipeline.sh:222-223`'s "now live" echo names a path nothing serves.
Also cosmetic: `logs/events-*.jsonl` and the dead `NOT_BEFORE` gate
(`soak_watchdog.sh:68`), local-TZ weekend blackout assertion (`:76-79`).

**24b. Boot-input hygiene. [H]** Live-path env parsing is bare
`int()`/`float()` with no validation (`oanda_forex_orchestrator.py:298-308,
316-323, 336-338, 348, 419-422, 434-452`; `risk_manager.py:264-275`) — one bad
env value = a boot crash-loop held down 15 min per retry by the brake; and a
typo'd `RISK_BLACKOUT_ET` **silently disables Gate C** (fail-open,
`risk_manager.py:184-186`, the rollover blackout switches off with only a log
line as evidence). Also: `RiskProfile()` used as a mutable default argument
(`risk_manager.py:299-301` — one shared instance across default-constructed
RiskManagers; inert today, a footgun), and `_notify`'s executor futures are
never awaited (`oanda_forex_orchestrator.py:465-467` — a payload-construction
bug would vanish silently). Also confirmed by reading: `backtest_60.py` is
exactly as dead as its own docstring says (MLStrategy signature changed;
`get_order_params` no longer exists) — delete or leave flagged; and
`models/*` *directory* mtimes mislead (dirs are replaced, not written in
place — trust `metadata.json.trained_at`, never the dir mtime).

## What was verified CORRECT (do not re-audit)

- Entry retry/re-sync (c5e7e83): 4xx-vs-permanent-vs-ambiguous classification,
  delta-recompute after sync, final-attempt fill catch. 16 tests.
- Boot reconcile (orphan flatten, refuse-on-unverifiable), seam dedup
  (`_last_scored_ts` set before evaluation on all three paths), entry guards
  (cooldown/exposure/reservation under one lock), tick-watchdog idempotency
  and breach sides (long=bid, short=ask), HTTP timeouts (30 s REST / 20 s
  stream), events.py never-raises contract. All tested.
- Lookahead guard `available_at` (HTF stamp = bar start + full TF; asof join);
  training/live feature symmetry (same generators same order; live reads
  `feature_names_in_`); stale-bar guard; mid-price consistency (stream mid ==
  training `price:"M"`); no synthetic-flat-candle contamination on the forex
  path (that's Alpaca-only wiring); session flags UTC-correct.
- Gate A/B mirror training↔live (percentile-rank, cold-start, coupled k_eff
  clip ≥1.0, Wilder NATR reconstruction), Gate C DST correctness live-side
  (untested but correct), behavior tagger causality, router stand-downs on
  unknown names.
- Per-file atomic writes for every artifact (temp + `os.replace`), hot-reload
  mtime-retry semantics, failed-retrain cannot promote, holdout discipline
  (carved pre-engineering, frozen thresholds, diagnostic-only on fold fail),
  Clopper-Pearson math (reproduced the served artifact's recorded bound
  exactly: 25/36 → PF_lb 2.3987), boundary purge off-by-one clean, walk-forward
  folds date-based not row-based (only the OOF refit loop is index-based).
- MLStrategy is long-only (`direction="long"` hardcoded `:797`) — all short-side
  orchestrator paths are dead code in production.

## Verification

- Test suite: `PYTHONPATH=src:. <venv>/bin/python -m pytest -q` → **411 passed,
  5 skipped** (23 collection errors under system python 3.14 are missing deps,
  not failures).
- Numeric checks reproduced by the auditing agents: CP lower bound re-derived
  from the served artifact's own raw evidence to 4 decimals; TimeSeriesSplit
  fold composition on the symbol-blocked frame confirmed on synthetic data;
  time-decay weight misassignment computed concretely.
- Live state cross-checked: journalctl (one clean restart in 14 days), SPREAD_CALIB
  alphas vs `config/spread_alphas_m15.json` (table vs live 9-day drift bands),
  status.json freshness (5.5 min at read time), watchdog.log, models/ mtimes
  vs `metadata.json.trained_at`.
- Every defect above carries a file:line the reader re-verified against HEAD
  6390128; where two auditors hit the same defect independently (shutdown
  flush, cooldown race), the citations are merged.

## Risk & follow-ups

- **None of Tier 1–3 should ship while a position is open** if the restart is
  manual — use the normal watchdog cycle; `soak.off` before stopping.
- Item 19's retrain will change the calibrated Angel bar; expect the post-fix
  threshold and the live 9.5-day zero-proposal anomaly to be re-evaluated
  together (fresh OOF probs will move the bar).
- Items 2/11 change the trade population the model sees — do them *with* a
  retrain, not alone, or create a new training/live asymmetry in the other
  direction.
- The OANDA `clientRequestID` dedup semantics (item 18) must be confirmed
  against OANDA docs before relying on them.
- The `test_stream_not_running_no_action` pin (item 17) documents deliberate
  behavior — its author should co-sign the semantics change.

## Files touched (read)

`src/execution/{oanda_forex_orchestrator,oanda_order_manager,risk_manager,live_orchestrator}.py`,
`src/data/{oanda_provider,factory}.py`, `src/strategies/concrete_strategies/{ml_strategy,regime_router,__init__}.py`,
`src/strategies/base.py`, `src/ml/features/{v3_features,feature_pipeline}.py`,
`src/ml/regimes/{hmm_regime,behavior_tagger}.py`, `src/ml/barriers/{labels,estimator,evaluate_barriers}.py`,
`src/core/{retrainer,resolver,feedback_loop,thresholds,events,notification_manager}.py`,
`src/replay_test.py`, `src/evaluate_performance.py`, `src/analysis/build_strategy_matrix.py`,
`src/utils/bar_aggregator.py`,
`run_oanda.py`, `run_soak.sh`, `soak.service`, `soak_watchdog.sh`, `trading_mcp.py`,
`run_pipeline.sh`, `scripts/{bake_spread_alphas,run_decision_report}.py`,
`config/spread_alphas_m15.json`, `config/regime_routing_forex{,_H1,_H4}.json`,
`models/forex_m15_wide/{metadata.json,threshold.json}`, `logs/{status.json,soak_2026-08-30_1405.log,watchdog.log,decision_report_2026-09-{06,08}.txt,strategy_matrix_H{1,4}.csv}`,
`tests/` (full suite run), `dashboard/` scaffold, `GLOSSARY.md`, `CLAUDE.md`,
plus journalctl/systemd state and the live process table.
