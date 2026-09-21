---
type: recon
date: 2026-09-14
time: 16:10 PDT
agent: DeepSeek Harness (dsh, web)
model: deepseek-v4-flash
trigger: "Sixteen rounds of measurement on the learned-barrier feature produced a decision, not a promotion. This is the short version: what the evidence says, what each remaining option costs, and what I would not do."
head: f0d65089756646b6e389796b96d1a057939bc67f
scope: read-only
related:
  - recons/2026-09-14_served-artifact-provenance.md
  - recons/2026-09-14_bracket-population-and-live-trade-rate.md
  - m2m-prompts/2026-09-14_barrier-live-seam.md
  - handoffs/2026-09-14_barrier-seam-status.md
  - audits/2026-09-08_high-benefit-fixes-ranked.md
files_touched: []
---

# The barrier feature, decided — and the six options that remain

## Context

The ask was to wire learned quantile MAE/MFE barriers into live execution. That
wiring is built, tested and **off**. Deciding whether to switch it on took sixteen
rounds of measurement, because four separate hypotheses had to be killed first
(this round's predecessors are in the seam thread, which is append-only and keeps
every refutation). This file is the decision view: one page, with the numbers that
made each choice, and links to the long form.

## The answer to the original question

**Static stops were not the problem.** The learned geometry's entire benefit is
cost dilution, and a *constant* wide bracket reproduces it:

| arm | trades | win | gross | net | toll/trade |
|---|---:|---:|---:|---:|---:|
| static 2.0×/4.0× | 2653 | 0.284 | +0.040R | −0.207R | 0.246R |
| learned per-bar quantiles | 1785 | 0.550 | +0.003R | −0.046R | 0.041R |
| **constant 10.25×/2.74× (no model)** | 1746 | 0.533 | −0.001R | −0.049R | 0.041R |

The quantile model's conditioning is worth **+0.0037R per trade** over a constant
— i.e. nothing — because its stop response is nearly flat (~8.5–10.25 ATR across
the distribution). Two further controls: raising `tau_mfe` from 0.50 to 0.85 keeps
gross expectancy at ~zero at every setting (win rate and payoff cancel), and no
cell on a held-out window is positive net of cost.

## The options, with what each one measurably costs

| # | option | measured consequence |
|---|---|---|
| 1 | **Do nothing; let the soak accumulate** | The wall is evidence, not configuration: 30 validation trades over 2 years, 13–38 top-band rows over 45 live days, 11–109 trades per holdout cell. Only time or a better model moves it. Current live rate: ~1 approval per few days, **0 fills in the 15 days since promotion** — which is the design working, not a fault. |
| 2 | **Retrain with `RETRAIN_DEVIL_LABEL=macro`** (validated fix, switch exists) | PF lower bounds improve to **1.4927 pooled / 1.4376 fold-3** (from 0.7727 / 0.5965), EV +0.37 → **+0.93** — and the gate then fails on a *different* criterion, **13 approvals against a floor of 23**. Better evidence, less of it. Re-derive `BRIER_THRESHOLD` (its rationale is label-specific). |
| 3 | **Retrain unchanged** (survival Devil, default) | Rejected on PF lower bound, exactly as today; the Angel bar would move 0.3833 → **0.3564** (proxy), i.e. 0.027 in the same tail. Not an unlock. |
| 4 | **Widen the bracket in `RiskProfile`** (≈10×/2.7×) | The only lever with a measured expectancy effect: toll 0.246R → 0.041R per trade, net improvement +0.08 to +0.18R/trade across four populations. **No cell positive out of window**, and on the live fixed-1000-unit path it needs `RISK_SIZING_ENABLED` to be expressible in dollars at all. It would also keep the Devil's labels in step with what is served — which the learned path cannot. |
| 5 | **Lower the Angel bar** to trade more | The live ledger's threshold sweep is **negative at every bar** (0.20 → −0.316R … 0.50 → −1.1R) against a 25.3% base rate and a 33.3% break-even. More trades, not better ones. |
| 6 | **Expand the basket** to buy sample | ❌ **Measured and refuted.** Six extra vol-carrying crosses (`CAD_JPY, CHF_JPY, EUR_AUD, EUR_CAD, AUD_NZD, GBP_CHF`) doubled the engineered rows (226,992 → 452,433) and left the approved trade count unchanged (**30 → 28**), with a worse PF lower bound (0.7727 → 0.5260, win rate 43.3% → 35.7%). **Mechanism:** the pinned Angel bar is re-calibrated per refit to *maximise EV*, so it self-adjusts to keep taking only the top of the distribution — it moved **up** (0.3564 → 0.3743) as the basket grew. The trade count is capped by the calibration objective, not by the instrument list. |
| 7 | **Change the calibration objective** (bar chosen for population, not for EV) | ❌ **Measured and refuted.** Bar as the top X% of training OOF scores: the trade-count wall dissolves (912 → 55,957 pooled trades) and the evidence collapses — **0 of 6 points** satisfy both gate criteria, win rate never exceeds 0.273 against a 0.333 break-even, the PF lower bound never exceeds **0.7271** against 1.2, and the binding rejection becomes **EV < 0.0005** at every point. The two criteria have no intersection. |

## Recommendation

1. **Hold both switches off.** Nothing measured this session justifies enabling
   learned barriers (`BARRIER_GEOMETRY_ENABLED`) or unit scaling
   (`RISK_SIZING_ENABLED`) on evidence — the scaling is right on arithmetic, but it
   exists to serve a configuration with no positive expectancy.
2. **If a retrain is run at all, flip the Devil's label** (option 2). It is one env
   var, it is validated (AUC 0.4722 → 0.5839, stable across windows), and it
   removes a stage that is currently *anti-correlated* with the outcome it filters
   (r = −0.166). Expect the gate to reject for want of trades rather than for want
   of evidence — that is progress, and it should be read that way.
3. **Both remaining levers are now measured and dead.** Basket expansion (option 6)
   does not buy trade count — 30 → 28, because the bar self-adjusts — and changing
   the calibration objective (option 7) dissolves the trade-count wall only by
   admitting a population with negative EV: **0 of 6 points satisfy both gate
   criteria**, best PF lower bound 0.7271 against a 1.2 bar. The gate is
   **unsatisfiable at any bar** for this model on this basket, which is a statement
   about the model, not the configuration.
   **Consequence: there is no configuration work left to do here.** Every lever
   tested this session — bracket geometry, τ pair, calibration method, Devil label,
   Angel bar, basket composition — fails, and the EV-max calibration turns out to be
   what *confines* the system to the only population that is not obviously negative.
   What changes the outcome is a better model or a different market hypothesis, not
   another sweep.
4. **Do not** promote the learned geometry, re-tune `tau_mfe`, chase the top Angel
   band, expand the basket, re-target the Angel bar, or read "0 fills in 15 days" as
   a bug. Each of those has a measurement in the thread saying why — and the last
   three of them were tested *after* being recommended, and refuted.
5. **The gate is correct and should stay strict.** It refuses at every setting
   because there is no population that is both large enough and positive. Loosening
   it — the 23-trade backstop, the 1.2 PF lower bound, the EV bar — is the one
   action that would produce a "pass" without producing an edge.

## The active lane (audit Stage 2, H4 CatBoost) — where it actually stands

`scripts/run_h4_candidate.py`, analysed at the level of its per-fold OOS ledger.

**The lane clears three of four gate criteria** (metals-scored arm, 229 approvals):
Brier 0.1741, mean EV +0.345R, pooled PF 95% lower bound **1.4482** against a 1.2
bar. Its single failure is **fold 3** — 18/61 wins, 29.5% against a 33.3% break-even,
PF lower bound 0.5006.

**Fold composition, which is the diagnosis:**

| fold | span | trades | macro win | metals share (win) | fiat (win) |
|---|---|---:|---:|---|---|
| 1 | 2025-07-11 → 2025-10-13 | 45 | 0.489 | 31% (0.429) | 0.516 |
| 2 | 2025-10-14 → 2026-01-16 | 123 | 0.561 | 74% (0.593) | 0.469 |
| 3 | 2026-01-19 → 2026-04-08 | 61 | **0.295** | **98%** (0.300) | 1 trade |

**The metals framing is refuted (mine, twice).** PF lower bounds recomputed from the
ledger's own counts with the gate's own helper: metals only **1.3700** (78/165,
47.3% win), fiat only **1.2057** (31/64, **48.4% win**), pooled 1.4482, folds 1+2
1.8116, fold 3 0.5006. Fiat pairs win *more* often than metals; metals only look
stronger because there are 165 of them to fiat's 64, and the confidence bound
tightens with n at a fixed win rate. The original run's 1.2057 is exactly the fiat
subset's bound, confirming the mechanism. So: no metals edge, no metals artifact —
**a 47–48% macro win rate across 229 approvals, with one hostile period.**

**Fold 3's question is answered (round 24), and it is two events, neither of them a
configuration problem.** Capturing the Angel's *proposals* (by wrapping
`_capture_oos_ledger`) shows **every proposal in every fold survived the Devil** —
its frozen threshold is 0.10, so in this lane the second stage filters nothing and the
composition is entirely the Angel's:

| fold | fiat proposals | XAG proposals | XAU proposals | metals win |
|---|---:|---:|---:|---:|
| 1 | 31 | 11 | 3 | 0.429 |
| 2 | 32 | 79 | 12 | **0.593** |
| 3 | **1** | 48 | 12 | **0.300** |

1. **The Angel's confidence barely moved** (median proposal prob 0.25–0.28 in every
   fold; fold 3's XAG is *higher* than fold 2's).
2. **What moved is which symbols proposed anything**: fiat proposals 31 → 32 → **1**.
   In Jan–Apr 2026 the Angel has essentially no fiat conviction.
3. **And the metals it did propose stopped winning**: 59.3% → 30.0%.

So fold 3 is "the model stopped proposing fiat" (score distribution) **and** "the
metals it proposed stopped working" (market). The gate fails on the second, and
neither is reachable by tuning a threshold or a basket.

**And the newest evidence was invisible until scored (round 25).** The lane's script
carves an 18% holdout (`run_h4_candidate.py:75`) and **never calls
`_evaluate_holdout`**, so when the fold gate fails the chronologically last 18% is
silently discarded — a divergence from the retrainer's documented discipline, which
scores it as a diagnostic. Scored, it spans **2026-05-18 → 2026-09-11** (2,375 rows,
27 proposals of which 23 are metals):

| config | scored | wins | win rate | EV | PF lower bound |
|---|---:|---:|---:|---:|---:|
| default | 4 | 1 | 0.250 | −0.250R | 0.0258 |
| metals scored | 27 | 6 | 0.222 | −0.333R | 0.2259 |

Both are well below the 33.3% break-even. **The honest reading is front-loaded, not
proven decay**: folds 1+2 (Jul 2025 → Jan 2026) win 54.2% on 168 trades, while fold 3
(29.5%, 61 trades) and the holdout (22.2%, 27 trades) pool to 27.3% on 88 trades —
P(≤ that | true break-even) = 0.14, suggestive but not significant. What it does
establish is that the pooled PF lower bound of **1.4482 is carried by the older half
of the lane's history**, and that the two *most recent* periods are both below
break-even. Promote on that and you promote on evidence from more than eight months
ago while ignoring the freshest trades.

**Why 2026 looks unlike 2025: the base rate (round 26).** Macro base rate for every
resolvable bar, per period: **P1 metals 0.566**, P1 fiat 0.312, P1 all 0.388; P2 metals
0.294, fiat 0.298, all 0.297; P3 metals 0.218, fiat 0.326, all 0.298. Long metals in
2025 won 56.6% against a 33.3% break-even — **the regime alone cleared the gate.**

Measured against each period's own base rate:

```
P1 metals: model 60/105 = 0.571 vs base 0.566 -> edge +0.005  p=0.50   (NO edge)
P1 fiat  : model 31/ 63 = 0.492 vs base 0.312 -> edge +0.180  p=0.0021 (real edge)
P2 all   : model 18/ 61 = 0.295 vs base 0.297 -> edge -0.002          (tracks random)
P3 all   : model  6/ 27 = 0.222 vs base 0.298 -> edge -0.076          (below random)
```

So the lane's pooled PF lower bound of 1.4482 was earned by metals trades that carried
**zero selectivity** (57.1% against a 56.6% base) inside a favorable regime, plus one
real edge: **fiat pairs in the 2025 window, +18 points over base on 63 trades
(p = 0.0021**, surviving Bonferroni for the six cells examined). Fold 3's 29.5% is
exactly its period's 29.7% base — the model did nothing wrong; the period offered
nothing.

⚠️ **The gap this exposes in the gate, which is not specific to this lane:** a gate
scoring **absolute** win rate and PF will pass a zero-skill model whenever the market's
base rate is high (P1 metals: 56.6% base vs a 33.3% break-even — any long bracket
clears a 1.2 PF bound). It would equally reject a skilled model in a 20%-base period.
**The missing score is edge over the period's own base rate** — one column of the frame
(`devil_target_macro`), and the only quantity that separates skill from weather. This
applies to the M15 soak as much as to this lane.

**Implemented, not just recommended (round 27).** The gate now reports it:
`_macro_base_rate` computes the benchmark on each fold's own tradeable bars, logs
`EDGE OVER RANDOM` per fold and pooled, and carries `FoldMetrics.base_rate` /
`ValidationReport.pooled_base_rate` / `.edge_over_random`. **Verdict unchanged** — it is
telemetry so the decision can see the difference between skill and weather (6 tests in
`tests/test_base_rate_benchmark.py`).

Two real frames, measured with it:

| frame | base rate | edge over random | trades |
|---|---:|---:|---:|
| M15 fiat (`analysis_cache`, 226,992 rows) | 0.2540 | **+0.1793** | 30 |
| H4 + metals (lane cache, 14,082 rows) | 0.2949 | **+0.0122** | 1,078 |

The H4 lane's edge is ~zero and *well measured* (+0.012 on 1,078 trades), agreeing with
the ledger decomposition above. The helper also independently reproduces the 2026-09-08
report's own M15 base rate (0.2540 vs its 24.4%) computed by different code.

**The M15 arm's +0.179 on 30 trades is withdrawn (round 28).** Sweeping the bar with the
new edge column (`scripts/angel_bar_frontier.py`) shows the M15 Angel has a *small real
edge* — **+0.011 to +0.019 over base on 3,314–55,957 trades, five of six points
positive** — but it lives at **population** bars, and at the thin top (the region the
EV-maximising calibration selects) the edge turns **negative** (−0.019 on 912 trades,
win rate 0.235, the worst point). The live bar sits beyond even that.

### The synthesis: a toll budget

| quantity | value | source |
|---|---:|---|
| the model's real selectivity | **≈ +0.05R** | +1.5pp win rate at 2:1 (1pp ≈ 0.03R), M15 fiat, gate currency |
| the static bracket's toll | **≈ 0.25R** | per-trade spread over a 2×ATR stop (backtester + per-instrument alphas) |
| the wide bracket's toll | **≈ 0.04R** | same measure over a 10×ATR stop — ~6× dilution |

**The M15 configuration is not edgeless; it is edge-too-small-to-pay-the-toll**: ~0.05R
of signal against ~0.25R of cost. Widening the bracket does not create edge — it lowers
the cost below the signal, which is why the wide arm measured ≈ break-even (−0.049R)
against the static arm's −0.207R. Every lever tested this session moves the **cost**
side; none moves the **edge** side. The question worth handing on is therefore the one
nobody has answered: *can anything raise the +0.05R?*

### Direction is closed, and against the hypothesis (round 29)

Every bracket here is **long-only**, so the untested structural lever was the short
side. Measured market-wide, same walk convention, per period:

| period | M15 long base | M15 short base | H4 long base | H4 short base |
|---|---:|---:|---:|---:|
| 2024-09 → 2025-06 (pre) | 0.298 | 0.287 | 0.344 | 0.256 |
| 2025-07 → 2026-01 (folds 1+2) | 0.300 | 0.281 | **0.478** | **0.202** |
| 2026-01 → 2026-04 (fold 3) | 0.295 | 0.280 | 0.316 | 0.299 |
| 2026-05 → 2026-09 (holdout) | 0.298 | 0.296 | 0.363 | 0.312 |

**Longs beat shorts in every period of both configs** — a short-capable system would have
done worse. Two things it explains: the H4 lane's "promotable" window gave a long base
rate of **47.8%** against the short side's **20.2%**, i.e. a long-only bracket inside a
long-only regime (block 28's finding from the other side, and the cleanest statement of
why that pooled PF is not skill); and in 2026 **both directions sit at ~30% against a
33.3% break-even in both configs** — the market offers nothing to this bracket design.

**The target definition is closed too (round 30), and it prices the requirement.**
A model-free sweep of 60 geometries (stop × target × horizon, 226,369 bars each) for what
a *random* long entry earns net of Gate A's own spread proxy:

| stop | target | horizon | base rate | break-even | gross R | toll R | **net R** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.0× | 1.0× | 90 | 0.870 | 0.889 | −0.021 | 0.066 | **−0.087** |
| 4.0× | 2.0× | 90 | 0.668 | 0.667 | **+0.002** | 0.132 | −0.130 |
| 2.0× | 4.0× | 45 *(live)* | 0.341† | 0.333 | −0.108 | 0.263 | **−0.371** |

**0 of 60 cells is positive at random entries.** Gross expectancy tracks break-even to
within a rounding error (+0.002R over 226k bars at 4.0/2.0/90) — the market is efficient
for brackets of this shape, so re-aiming the bracket finds nothing. The live config ranks
16th of 60 and near the bottom on net, because a 2×ATR stop carries the largest toll tier.

**The requirement, priced:** the best random-entry cell costs ~0.09R per trade, and the
model's measured selectivity is ~0.045R (+1.5pp of win rate at 2:1 — itself measured at
2/4/45, so geometry-specific). Even crediting the model with its best edge at the best
geometry leaves ≈ −0.04R. **The gap is a factor of two to three on the edge side: find
something worth ~3pp of win rate, or stand down.**

**The crypto axis is closed too (round 31).** The 2026-08-11 recon priced the toll by
timeframe (D1 = 6.6%, cheaper than anything in forex) and left one question — whether
edge exists there. Measured on Alpaca bars for 6 majors with the recon's own costs:
**0 of 27 geometries positive at random entries at D1 and 0 of 27 at H4.** D1's toll
measures 0.0497R at a 2×ATR stop, confirming the recon's figure on independent data —
and it is still not enough, because D1's *gross* expectancy at random entries is
**negative** (−0.021R at best; forex's best cell was +0.002R). Same window buy-and-hold:
**BTC +283%, SOL +460%**, LTC −37%, AVAX −41% — the benchmark problem the recon warned
about, visible in the same data.

⚠️ **And a real bug found on the way, now fixed:** `AlpacaProvider.get_historical_bars`
built `TimeFrame(n, TimeFrameUnit.Minute)`, which Alpaca rejects above 59 — so **H4 and
D1 requests were impossible** and failed inside the method's broad `except`, surfacing as
"no data" rather than an error. The recon's headline recommendation was unservable. Fixed
with `_timeframe_for(minutes)` (15→15Minute, 60→1Hour, 240→4Hour, 1440→1Day; raises
locally for 90/2880, since Day/Week accept amount 1 only) + `tests/test_alpaca_timeframe.py`.

**Remaining axes, all outside this basket:** non-fiat instruments (metals carried a
47.8–56.6% long base rate, untradeable on this account — a human decision); a different
market (crypto/trend — the repo's stage-6 brief, base rate never measured here); a
genuinely different feature/target design. **Do not spend more rounds on this basket:**
four periods, two timeframes, two directions, 60 geometries, ~250k bars, five model
configurations — and the market's own ceiling is −0.09R against a model contribution that
would need to be twice what it is.

† the live-config base rate is for its horizon-matched walk; the row is included for the
net comparison, not the base-rate column.

**Decision: do not promote the H4 candidate** — its metals approvals had no
selectivity, and its one real edge rests on 63 fiat trades in a window that has ended.
Judge the lane on the edge column from now on; on that column it has nothing.
The lane's best use now is as the evidence for the base-rate gap, and as a fiat-pair
investigation rather than a metals one.

⚠️ **Structural fragility worth carrying:** proposals' median `angel_prob` is
**0.25–0.28 against a bar of ~0.25** — the certified population is a knife-edge band
inside the body of the Angel's own score distribution, so a one- or two-point shift
reshuffles which symbols trade at all. That is the same mechanism as the EV-maximising
bar parking at the top of the quantile grid: the population is thin and marginal by
construction.

Also worth recording: **CatBoost beats LightGBM in this configuration** (EV +0.345 vs
−0.036; PF lower bound 1.4482 vs 0.9875), so Stage 2's premise holds where the lane
runs. And the recon's +0.5484R/+0.4062R **reproduce exactly** — my round-20
"unreproduced" claim was an error (I ran a different configuration), retracted.

## What is verified, and what is only inferred

**Re-runnable:** `scripts/angel_bar_frontier.py` reproduces the frontier that makes
option 7's refutation (and therefore the "no configuration work left" conclusion)
checkable after any future retrain — exit 2 when the gate is unreachable at every
bar. The other harnesses behind this document were scratch files and did not
survive the session; their measurements are recorded here and in the seam thread.

**Verified on this machine:** every number above (commands and outputs are in the
seam thread and the two companion recons); the soak is untouched (same PID, five
files in its model dir, no barrier artifacts anywhere under `models/`); both
switches default off; the retrainer's fixed calibration and gate are in the code
while no artifact in the tree postdates them.

**Inferred, not verified:** that option 7 helps (nobody has run it — it is a design
proposal with a mechanism behind it, not a result); that the live
rate stays at ~1 approval per few days (it is measured over 15 days); that the
proxy pre-flights match what a real retrain would produce (they use the cached
basket, no holdout, `alpha_table=None`).

**Retracted during this session** (recorded so nobody re-derives them as
findings): the two-feature evaluator vocabulary as the cause of the gate failure;
the monotone constraint as the cause; a statistically positive bar-0.30/wide cell;
and "the top Angel band un-inverts once probabilities are honest" — that last one
compared different rows and is refuted on identical rows, where the served and a
freshly retrained model both invert the top band and correlate at 0.907.
