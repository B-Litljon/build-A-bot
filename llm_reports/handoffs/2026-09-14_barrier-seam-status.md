---
type: handoff
date: 2026-09-14
time: 13:32 PDT
agent: DeepSeek Harness (dsh, web)
model: deepseek-v4-flash
trigger: "Learned quantile barriers: producer and consumer halves were built in parallel by two agents in this checkout. This is the pointer file — the live seam thread is where the work is."
head: f0d65089756646b6e389796b96d1a057939bc67f
scope: read-only
related:
  - m2m-prompts/2026-09-14_barrier-live-seam.md
  - refactors/2026-09-14_learned-barrier-live-wiring.md
  - recons/2026-09-14_sidecar-h4-and-risk-sizing.md
  - handoffs/2026-09-14_quantile-mae-barriers-and-catboost-plan.md
files_touched: []
---

# Barrier seam — read the thread first, then this

**The live two-agent thread is
[`llm_reports/m2m-prompts/2026-09-14_barrier-live-seam.md`](../m2m-prompts/2026-09-14_barrier-live-seam.md).**
Eight blocks, append-only, every claim marked VERIFIED with its command and real
output. This file exists only because that folder is indexed under "live threads"
in `m2m-prompts/README.md` and `CLAUDE.md`, and an agent whose loop reads
`handoffs/` would otherwise never see it. **Do not add analysis here — append to
the thread.** A one-way report cannot coordinate two writers, which is the whole
reason the thread exists.

## What is true now (2026-09-14, verified on this machine)

**Wiring is done on both halves, and both switches are OFF.**

- Consumer side: `BarrierEstimator.save/load` (three-file artifact set, meta last),
  the `Signal.metadata["barrier_geometry"]` payload in NATR multiples,
  `RiskManager.calculate_bracket(barrier=…)` substitution, the orchestrator and
  backtester pass-throughs, hot-reload on the meta's mtime.
- Gate enforcement: a recorded FAILED promotion verdict in `barriers_meta.json` is
  **refused at boot**; no verdict serves with a warning; the producer records one
  from `RETRAIN_BARRIER_VERDICT` (path to `evaluate_barriers.py`'s
  `BARRIER_VERDICT_OUT`).
- Calibrated stop: `q_mae_scale`, recorded in the meta, applied to the stop side
  only, refused when non-positive.

**The feature is not promotable, and the reasons are measured, not asserted:**

1. The promotion gate FAILS — fold-3 coverage 0.912 with the full 17-feature
   artifact shape, 0.905 with the evaluator's 2-feature proxy, **0.850 on H4**.
   It beats the static constant on pinball loss on every fold at every timeframe.
2. The shortfall is a **conditional** miscalibration: worst in the *quiet*
   volatility deciles (0.877 in decile 1), not in the fat tail. A global conformal
   rescale cannot fix it (0.926); a per-decile one can (0.954/0.974/0.942).
3. The monotone constraint is **not** the cause. Dropping it fails fold 3 anyway
   (0.927) and destroys the served-distance monotonicity guarantee — verified by
   per-symbol ladder probe, in price units.
4. **The learned bracket does not create edge.** Realised-R replay: stop-outs
   57.1% → 3.6% and win rate 28.4% → 54.9%, but gross expectancy does not improve
   (+0.052 → +0.018R matched) and every net improvement is cost dilution — the
   toll falls 6.1× against a 6.2× wider stop. Both arms lose money.

## What to do next

**The decision view is one page:
[`recons/2026-09-14_session-evidence-and-options.md`](../recons/2026-09-14_session-evidence-and-options.md)**
— the answer to the original question, the six remaining options with their measured
costs, and the recommendation. Read that first; this section is the agent queue.

1. **Hold every switch off.** `BARRIER_GEOMETRY_ENABLED` and `RISK_SIZING_ENABLED`
   both stay at their defaults. Nothing measured in this session justifies enabling
   either, and the unit-scaling arithmetic is only needed by a configuration with no
   positive expectancy.
2. **If a retrain runs:** set `RETRAIN_DEVIL_LABEL=macro` (validated: AUC 0.4722 →
   0.5839, window-stable, and it removes a stage that is *anti-correlated* with the
   outcome it filters at r = −0.166) **and `RETRAIN_LEARN_BARRIERS=0`** (otherwise a
   rejected run still writes unverdicted barrier artifacts into the served
   directory). Expect rejection on the trade-count backstop (13 approvals against a
   floor of 23) rather than on PF — better evidence, less of it. Re-derive
   `BRIER_THRESHOLD`; its rationale is label-specific.
3. **Both remaining levers are measured and dead — do not re-open them.**
   Basket expansion does not buy trade count (30 → 28, the bar self-adjusts) and
   re-targeting the Angel bar dissolves the trade-count wall only by admitting a
   population with negative EV (**0 of 6 points** satisfy both gate criteria; best
   PF lower bound 0.7271 against 1.2; binding rejection becomes EV < 0.0005).
   **The gate is unsatisfiable at any bar** for this model, which is a statement
   about the model, not the configuration. The EV-maximising calibration is what
   *confines* the system to the only population that is not obviously negative.
   What changes the outcome is a better model or a different market hypothesis.
   ⚠️ Trap found while testing: `get_asset_config("oanda")` returns
   `htf_timeframe="5m"` unless `RETRAIN_TIMEFRAME_MINUTES` is set — its default
   assumes M1. An M15 caller reusing `cfg` for feature engineering silently gets the
   wrong HTF pairing; mirror `run_oanda.py`'s `_GRANULARITY_PROFILES`.
4. **Unclaimed items:** narrow/verify the `use_barriers` bracket-mismatch bypass
   (it now *reports* the Devil-label skew instead of hiding it, and the ask to make
   it conditional is still open), and fix or delete
   `scripts/test_sidecar_retrain.py` (asserts `sl_price_distance`, a key that exists
   nowhere, and treats `generate_signals` — which returns `Signal | None` — as
   returning a list).
5. **Superseded — do not act on these.** Each was measured and died; re-deriving
   them costs a session:
   - the two-feature evaluator vocabulary as the cause of the gate failure;
   - the monotone volatility constraint as the cause (dropping it fails fold 3 too
     **and** loses the served-distance monotonicity guarantee);
   - `tau_mfe` as the lever (gross expectancy ~zero at 0.50–0.85);
   - the bar-0.30/wide cell as a positive configuration (dead on the holdout);
   - "the top Angel band un-inverts once probabilities are honest" (compared
     different rows; on identical rows the served and a freshly retrained model both
     invert it and correlate at 0.907);
   - "the served model is damaged" (refuted by that same 0.907 correlation);
   - the learned geometry as a better stop (**+0.0037R** over a constant wide
     bracket).


## The H4 CatBoost lane (audit Stage 2) — measured, 2026-09-14

**Three of four gate criteria pass; one period fails.** Metals-scored arm of the
lane's own `scripts/run_h4_candidate.py`, 229 approvals: Brier 0.1741, mean EV
+0.345R, pooled PF 95% lower bound **1.4482** (bar 1.2), and **fold 3 = 18/61 wins
(29.5% against a 33.3% break-even), PF lower bound 0.5006**.

| fold | span | trades | win | metals share (win) | fiat (win) |
|---|---|---:|---:|---|---|
| 1 | 2025-07-11 → 2025-10-13 | 45 | 0.489 | 31% (0.429) | 0.516 |
| 2 | 2025-10-14 → 2026-01-16 | 123 | 0.561 | 74% (0.593) | 0.469 |
| 3 | 2026-01-19 → 2026-04-08 | 61 | **0.295** | **98%** (0.300) | 1 trade |

- **The metals framing is refuted, twice over (both were mine).** PF lower bounds
  from the ledger's own counts: metals 1.3700 (78/165, 47.3% win), fiat 1.2057
  (31/64, **48.4%** win), pooled 1.4482, folds 1+2 1.8116, fold 3 0.5006. Fiat wins
  *more* often; metals look stronger only because there are 165 of them, and the
  confidence bound tightens with n. The original run's 1.2057 is exactly the fiat
  subset's bound. **No metals edge, no metals artifact — a 47–48% win rate on 229
  approvals with one hostile period.**
- The recon's +0.5484R/+0.4062R **reproduce exactly**; my round-20 "unreproduced"
  claim was an error (different configuration), retracted.
- **CatBoost beats LightGBM here** (EV +0.345 vs −0.036; PF lb 1.4482 vs 0.9875), so
  Stage 2's premise holds where the lane runs.
- **Fold 3 answered (round 24): two events, neither a configuration problem.**
  Capturing Angel *proposals* shows **every proposal survived the Devil** (its frozen
  threshold is 0.10 — the second stage filters nothing in this lane), and:
  fiat proposals go 31 → 32 → **1** while XAG goes 11 → 79 → 48; the Angel's median
  proposal confidence barely moves (0.25–0.28 every fold); and fold-3 metals win
  30.0% against fold-2's 59.3%. So the Angel stopped proposing fiat (score
  distribution) *and* its metals stopped winning (market). The gate fails on the
  second.
- ⚠️ **Structural fragility:** proposals sit at 0.25–0.28 against a ~0.25 bar — a
  knife-edge band inside the Angel's own score distribution, so a 1–2 point shift
  reshuffles which symbols trade. Same mechanism as the EV-max bar parking at the top
  of the quantile grid: the certified population is thin and marginal by construction.
- **The newest data was being discarded (round 25).** `run_h4_candidate.py:75` carves
  an 18% holdout and **never calls `_evaluate_holdout`**, so a failing fold gate throws
  the chronologically last 18% away — the opposite of the retrainer's documented
  discipline. Scored (2026-05-18 → 2026-09-11, 2,375 rows, 27 proposals of which 23 are
  metals): win rate 0.222 (metals scored) / 0.250 (default, 4 scoreable), EV −0.33R /
  −0.25R, PF lower bound 0.23 / 0.03. **Below break-even in both.**
- **Front-loaded, not proven decay:** folds 1+2 win 54.2% on 168 trades; fold 3
  (29.5%, 61) and the holdout (22.2%, 27) pool to 27.3% on 88 trades,
  P(≤ that | break-even) = 0.14 — suggestive, not significant. But it means the pooled
  PF lower bound 1.4482 is carried by the older half of history while the two most
  recent periods are both below break-even.
- **Why 2026 ≠ 2025: the base rate (round 26).** Macro base rate per period: P1 metals
  **0.566**, P1 fiat 0.312, P1 all 0.388; P2 all 0.297; P3 all 0.298 (P3 metals 0.218).
  Long metals in 2025 won 56.6% against a 33.3% break-even — the regime alone cleared
  the gate. Against each period's own base rate: **P1 metals model 0.571 vs 0.566 →
  edge +0.005, p=0.50 (no selectivity at all)**; **P1 fiat 0.492 vs 0.312 → +0.180,
  p=0.0021 (real edge, 63 trades)**; P2 0.295 vs 0.297 (tracks random); P3 0.222 vs
  0.298 (below).
- **The gate now reports it (round 27, implemented).** `_macro_base_rate` +
  `EDGE OVER RANDOM` logging per fold and pooled; `FoldMetrics.base_rate`,
  `ValidationReport.pooled_base_rate`/`.edge_over_random`; 6 tests in
  `tests/test_base_rate_benchmark.py`. **Verdict unchanged** (telemetry). Measured with
  it — and then corrected the reading (round 28). Sweeping the bar with the edge column
  (`scripts/angel_bar_frontier.py`): the M15 Angel's edge is **+0.011 to +0.019 over
  base on 3,314–55,957 trades (5 of 6 points positive)**, but it lives at **population**
  bars and turns **negative** at the thin top the EV-max bar selects (−0.019, n=912,
  win 0.235). So the earlier "+0.179 on 30 trades" for M15 is **withdrawn as noise**.
  The helper independently reproduces the 2026-09-08 report's M15 base rate (0.2540 vs
  24.4%).
- **Direction is closed (round 29), against the hypothesis.** Every bracket is long-only,
  so shorts were the last structural lever. Measured market-wide: **longs beat shorts in
  every period of both configs** (M15: 0.298/0.287, 0.300/0.281, 0.295/0.280,
  0.298/0.296; H4: 0.344/0.256, **0.478/0.202**, 0.316/0.299, 0.363/0.312). A
  short-capable system would have done worse. It also explains the lane's window: a long
  base rate of 47.8% against 20.2% for shorts — a long-only bracket inside a long-only
  regime. **In 2026 both directions sit at ~30% against a 33.3% break-even in both
  configs**, so stand-down is the market's verdict, not a concession.
  **The target definition is closed (round 30) and the requirement is priced.** A
  model-free sweep of 60 geometries (226,369 bars each) for a RANDOM long entry, net of
  Gate A's spread proxy: **0 of 60 cells positive**; best is an 8×/1×/90 bracket at
  **−0.087R**; the live 2/4/45 config is **−0.371R** (16th of 60) because a 2×ATR stop
  carries the largest toll tier. Gross expectancy tracks break-even to within a rounding
  error (+0.002R over 226k bars at 4.0/2.0/90) — the market is efficient for these
  brackets, so re-aiming finds nothing. **The requirement: ~0.09R of edge; the model has
  ~0.045R — a factor of two to three, on the edge side, where no lever this session
  acted.**
  **The crypto axis is closed too (round 31):** 0 of 27 geometries positive at random
  entries at D1 *and* 0 of 27 at H4 (Alpaca bars, 6 majors, the 2026-08-11 recon's own
  costs). D1 toll measured 0.0497R at a 2×ATR stop — confirming the recon's 6.6% — and
  D1's gross expectancy at random entries is **negative** (−0.021R best; forex's best was
  +0.002R). Buy-and-hold in the same window: BTC +283%, SOL +460%, LTC −37%.
  Also fixed en route: `AlpacaProvider` built `TimeFrame(n, Minute)` for everything, and
  Alpaca caps minute amounts at 59 — **H4/D1 were impossible and failed silently as "no
  data"**. Now `_timeframe_for(minutes)` (Day/Week accept amount 1 only, so 2880 raises
  locally) + 7 tests.
  **Remaining axes:** non-fiat instruments (metals base rate 47.8–56.6%, untradeable here
  — a human decision) and a genuinely different feature/target design — the only untested
  axis left, and the one that would have to be worth ~3pp of win rate.
- **Toll budget (the session's central quantitative conclusion):** model selectivity
  **≈ +0.05R** vs a static toll of **≈ 0.25R** and a wide-bracket toll of **≈ 0.04R**
  (currencies labelled: gate win-rate units vs backtester R). The M15 config is
  **edge-too-small-to-pay-the-toll**, not edgeless. Every lever tested moves the cost
  side; none moves the edge side. The open question for the next agent is the one nobody
  has answered: **can anything raise the +0.05R?**
- ⚠️ **Gate gap, general and not lane-specific:** a gate scoring ABSOLUTE win rate/PF
  passes a zero-skill model in a high-base-rate regime and rejects a skilled one in a
  low-base-rate regime. **The missing score is edge over the period's own base rate** —
  one column of the frame (`devil_target_macro`). Applies to the M15 soak too.
- **Decision: do not promote the H4 candidate.** Its metals approvals (the bulk) had no
  selectivity; its one real edge (fiat, +18pp) is 63 trades in a window that has ended.
  The lane's best use now is as evidence for the base-rate gap and as a *fiat-pair*
  investigation.
- ⚠️ Reading a gate log from this lane: `FoldMetrics.win_rate` is the **survival**
  rate while macro counts sit beside it, so a fold prints `WR=100.0% (16/31)`.
  One-line fix at `retrainer.py:3096`. The lane's script also builds the per-fold OOS
  ledger with `fold`/`symbol`/`macro_win` in it — dumping it to parquet makes these
  questions minutes rather than rounds.

## Working rules for this checkout (they are not optional)

Two agents share this tree. Append to the thread rather than writing a report
nobody reads; **re-read it immediately before appending**; say so if you edit a
file the other agent may be mid-edit in; and verify the tree before believing a
note — including one you wrote. Commits here are frequently made by whoever
finishes last, with a message that describes only their own half.
