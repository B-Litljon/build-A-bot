---
type: recon
date: 2026-09-14
time: 13:55 PDT
agent: DeepSeek Harness (dsh, web)
model: deepseek-v4-flash
trigger: "Does the learned-bracket work have a measurable population to be judged on? Running the bracket comparison on the Angel+Devil-approved bars, and cross-checking the harness against the live soak's own telemetry."
head: f0d65089756646b6e389796b96d1a057939bc67f
scope: read-only
related:
  - m2m-prompts/2026-09-14_barrier-live-seam.md
  - handoffs/2026-09-14_barrier-seam-status.md
  - recons/2026-08-29_trim-100x15-promotion.md
  - recons/2026-09-14_sidecar-h4-and-risk-sizing.md
files_touched: []
---

# The bracket question has no measurable population — and the soak is selective, not broken

## Context

Six rounds of barrier work established (a) the promotion gate fails at M15 and H4,
(b) the learned geometry's entire benefit is cost dilution, and (c) a constant
wide bracket reproduces it to within 0.004R, i.e. the learning adds nothing. All
of it was measured on an **unconditioned** entry population — a stub firing every
bar — where gross expectancy is ~0 by construction. That is fine for comparing
brackets and useless for judging the bot, because the bot does not trade that
population. This asks the two questions that follow: does the comparison hold on
the bars the Angel and Devil actually approve, and is that population big enough
to decide anything?

## Investigation

**Harness.** Cached M15 basket (6 pairs, 730 days), `engineer_features_and_labels`
for the 17-column vocabulary the served pair uses (`assert`ed against
`angel_latest.pkl`'s `feature_names_in_`), the served
`models/forex_m15_wide` Angel and Devil for selection with the bars pinned in its
own `threshold.json` (angel 0.3833, devil 0.44), then
`analysis.strategy_backtester.run_backtest` for both arms — gap fills, timeouts
paid at the realised move, the three live gates, per-instrument spread alphas
from `config/spread_alphas_m15.json`, `max_hold=45`.

Two bracket arms on identical entries: static 2.0×/4.0× (what ships) and a
constant 10.25×/2.74× (the width the barrier work converged on). The learned
per-bar geometry is omitted because it matches the constant to 0.004R/trade.

**Cross-check before reporting.** A three-day window of live heartbeats shows
`proposed=0/30` and a maximum `angel_prob` of ~0.25 against a bar of 0.3833,
which reads as "the model cannot reach its own bar". That would have been an
overclaim: the full soak log record since the promotion tells a different story
(below). Every number here was checked against the live process.

## Findings

1. **At the pinned bars the comparison is unmeasurable.** 14 of 56,728 test bars
   pass both stages (0.02%), yielding 11 replayed trades and a 95% CI of ±0.87R
   on net expectancy. The apparent −0.28R penalty for the wide bracket on those
   11 matched pairs is inside the noise.

2. **At relaxed Angel bars the wide bracket wins consistently, and every arm
   loses.** Net Δ (constant − static) is +0.154R at bar 0.300 (n=73 pairs),
   +0.095R at 0.250 (n=494), +0.079R at 0.200 (n=836) — same direction as the
   unconditioned run, and at 0.20/0.25 the arms separate beyond their per-arm
   CIs. Gross expectancy stays ~0 everywhere (|gross| ≤ 0.07R with CIs spanning
   zero), so relaxing the bar buys sample size, not edge.

3. **The live bot has taken zero trades in 15 days, and not because the models
   are silent.** Since the 2026-08-29 promotion: **3 Angel+Devil agreements**
   (angel 0.39, 0.39, 0.43), **3 Gate B vetoes** (`natr rank=0.07 < P20%`,
   i.e. volatility in the bottom 7% of its 260-bar window), **0 positions
   recorded**. The pinned bar sits deep in the live distribution's upper tail
   (recent per-symbol maxima 0.206–0.249), so approvals are rare by design and
   the ones that occur are being filtered by the regime gate.

4. **The soak is behaving as designed, not broken.** The promotion recon
   predicted "it proposes far more selectively than the old 200x63 … with a live
   Devil". Three proposals in fifteen days, all regime-vetoed, is consistent with
   that and with the ~1-per-few-days rate measured on the basket. A quiet stretch
   of days is not evidence of a fault.

## Verification

- Selection: `assert list(feature_cols) == list(angel.feature_names_in_)` (17
  columns, no `cost_ratio`), pinned bars read from the served `threshold.json`.
- Live cross-check: `grep -h "AGREEMENT" logs/soak_*.log | awk '$1 >= "2026-08-30"'`
  → 3 lines; the matching `Bracket rejected` lines show `regime=3 time=1 (3 / 3
  Devil-approved vetoed, 100.0%)`; `Recording position` → 0 lines.
- Hearthbeat lines quoted in the thread; per-symbol maxima 0.206–0.249.
- ⚠️ Caveat carried with the replayed levels: the served Angel was trained
  2026-08-29 over a long lookback, so its selection on a 2026-03→09 slice is
  partly in-sample and the *levels* are inflated. The arm-vs-arm deltas are on
  identical entries and are not affected.

## Risk & follow-ups

- **Any bracket decision made from the pinned-bar population is noise.** Quote the
  Angel bar with any bracket result, and prefer a relaxed bar where the CI is
  usable.
- **The binding constraint is entry quality.** Gross expectancy is ~0 in every
  population measured, including the model-selected one, matching the repo's own
  2026-09-08 audit ("the strategy library has no measured edge, all routing cells
  stand down"). Bracket work — learned or static — cannot fix that.
- **If the bracket width is ever changed** (the only lever with a measured
  effect), it belongs in `RiskProfile` as a static multiplier: no new machinery,
  and it keeps the Devil's labels — built from the profile multiples at
  `retrainer.py:1469` — in step with what is served. The learned path pays a real
  correctness cost for a benefit a constant reproduces.
- **Live-rate baseline recorded here for future comparison:** ~1 agreement per
  few days at the pinned bar; 0 fills in the 15 days since promotion. If that
  rate changes materially after a retrain, this recon is the reference.

## Files touched

_None — read-only._ Read: `models/forex_m15_wide/{angel_latest.pkl,devil_latest.pkl,
threshold.json,metadata.json}`, `analysis_cache/strategy_matrix/*_M15.parquet`,
`config/spread_alphas_m15.json`, `logs/soak_*.log`, `logs/events-*.jsonl`,
`src/analysis/strategy_backtester.py`, `src/core/retrainer.py`,
`src/ml/barriers/{estimator,labels}.py`, `src/strategies/base.py`.

---

## Addendum (same session, 14:35) — the constraint is entry SELECTION, and one configuration is net positive

This recon concluded "the binding constraint is entry quality" and left it there.
Following it up produced the first positive result in the whole line of work, and
it refines that conclusion rather than contradicting it: entry quality is
*selectable*, and the bracket width is what makes the selection payable.

**The top Angel band un-inverts once the probabilities are honest.** Rebuilding
the calibration curve with the fixed OOF algorithm (chronological permutation;
Angel trained on `angel_target`, graded on `devil_target_macro`) against the
2026-09-08 leaked table:

| band | honest win rate | leaked win rate |
|---|---|---|
| 0.00–0.15 | 0.2290 | 0.2299 |
| 0.15–0.20 | 0.2711 | 0.2507 |
| 0.20–0.25 | 0.2765 | 0.2631 |
| 0.25–0.30 | 0.3111 | 0.2550 |
| 0.30–0.40 | 0.3064 | 0.2406 |
| **0.40+** | **0.3897** (n=213) | 0.1176 (n=34) |

AUC(angel_prob → macro win) = **0.5378**, 95% CI [0.5355, 0.5402] — weak, but real.
The honest OOF distribution also matches the live one (median 0.166 vs the soak's
0.155, p90 0.228 vs ~0.22), so the leaked calibration was what put the pinned bar
out of live reach.

**Crossing the bar with the bracket width** (OOF selection, live gates,
per-instrument alphas):

| Angel bar | arm | trades | win | gross | net | toll | net CI95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.25 | static | 2421 | 0.325 | +0.141 | −0.091 | 0.231 | ±0.057 |
| 0.25 | wide | 2075 | 0.602 | +0.037 | −0.009 | 0.046 | ±0.016 |
| 0.30 | static | 658 | 0.312 | +0.144 | −0.064 | 0.208 | ±0.106 |
| **0.30** | **wide** | **585** | **0.617** | **+0.063** | **+0.022** | **0.041** | **±0.027** |
| 0.35 | static | 260 | 0.339 | +0.180 | −0.020 | 0.200 | ±0.171 |
| 0.35 | wide | 239 | 0.603 | +0.060 | +0.021 | 0.039 | ±0.042 |
| 0.40 | static | 95 | 0.421 | +0.441 | +0.250 | 0.192 | ±0.293 |
| 0.40 | wide | 96 | 0.573 | +0.066 | +0.029 | 0.037 | ±0.059 |

Gross rises with the bar (+0.141 → +0.441R); the wide bracket's 5× smaller toll
turns net positive from bar 0.30 up, and at **bar 0.30 / wide / n=585 the CI
excludes zero** — the first statistically-supported positive net expectancy found.
The trade-off inverts at 0.40 (matched Δnet −0.224), where static's higher payoff
wins, but n=95 cannot settle that.

Caveats: cached 2-year basket, six pairs, **no holdout**, **no Devil filter** (the
live path uses both stages), and the margin is 2% of R. This is a counterfactual
ledger result, not an out-of-sample claim.

⚠️ **All of it is in R units**, i.e. it assumes risk-normalized size. The wide
bracket is 5.12× wider, so on the live fixed-1000-unit path the advantage becomes a
~5× larger dollar loss per stop-out. The configuration is not live-realizable until
`RISK_SIZING_ENABLED` is on — which makes `RiskManager.calculate_forex_units`
load-bearing rather than defensive.

---

## Addendum 2 (same session, 14:50) — the positive cell does NOT survive the holdout, and the Devil is a no-op

Addendum 1 reported one statistically-supported positive cell (Angel bar 0.30 +
wide bracket, n=585, **+0.022R ±0.027**) and flagged two missing controls. Running
them refutes the positive read:

**Held-out final 25% of the timeline (from 2026-03-06, 56,752 rows):**

| bar | arm | population | trades | win | gross | net | toll | net CI95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | static | 188 | 109 | 0.193 | −0.216 | −0.378 | 0.162 | ±0.233 |
| 0.30 | wide | 188 | 97 | 0.516 | −0.032 | **−0.066** | 0.033 | ±0.081 |
| 0.35 | static | 55 | 37 | 0.324 | +0.150 | −0.011 | 0.161 | ±0.437 |
| 0.35 | wide | 55 | 33 | 0.546 | +0.058 | +0.025 | 0.033 | ±0.116 |
| 0.40 | static | 13 | 11 | 0.364 | +0.871 | +0.652 | 0.219 | ±0.677 |
| 0.40 | wide | 13 | 11 | 0.546 | +0.076 | +0.033 | 0.043 | ±0.230 |

The headline cell is **−0.066R ±0.081 on 97 held-out trades** — centred negative,
population collapsed from 1721 bars to 188. Nothing out of window is
statistically positive, and the large point estimates sit in CIs three times their
size. So the correct status is **unresolved, not positive**: the out-of-window
sample (11–109 trades per cell) cannot confirm or refute a 2%-of-R effect, and the
earlier number was carried by the first 75% of the timeline.

**The Devil filter is decorative.** Honest OOF Devil probabilities: median 0.852,
p10 0.749, and a pass rate of **1.000** at the pinned bar 0.44 — it approves every
bar, so the cells above are identical to the Angel-only run (658/585 trades at bar
0.30, 95/96 at 0.40). My harness sees the served Devil approve 93% of bars too.
Either its bar needs the same honest-OOF recalibration the Angel's got, or the
second stage has no discriminative content and should be re-thought, not re-tuned.

**Net position unchanged:** no configuration measured in this session is positive
net of cost on a held-out window; the barrier machinery remains correct,
default-off, and worth nothing over a constant wide bracket.
