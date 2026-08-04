---
type: recon
date: 2026-08-03
time: 18:40 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "Look for a better target — does a different prediction goal carry more signal than 'top quintile of 60-day forward return'?"
head: 6523254
scope: read-only
related:
  - 2026-08-02_investor-ebitda-gap-and-rebalance-policy.md
---

# V4 Investor: the prediction horizon is the load-bearing knob

## Context

The 2026-08-02 recon found the stock picker does not beat equal-weighting the
96-name universe out of sample. Of the four follow-up directions, Brandon
chose: **look for a better target**.

The shipped model is trained to rank stocks by "will this land in the best
fifth of 60-trading-day forward returns." Everything else about the pipeline
— features, folds, LightGBM parameters, the top-8 sector-capped selection —
is held fixed here. Only the label changes.

Every variant is scored on the **same** yardstick: realized money from a
monthly rebalance versus equal-weighting all 96, never on its own label's
precision. Each label makes its own metric easy; only the money is comparable.

## Investigation

Harness rebuilt from scratch (`investor_target_lab.py`, `investor_horizon_sweep.py`,
`investor_jitter.py` in the session scratchpad — yesterday's was lost with its
session). Mirrors `scripts/investor_train_model.py`: TRAIN_DAYS=504 expanding,
EMBARGO_DAYS=60, TEST_DAYS=60, LambdaRank, identical hyperparameters, greedy
top-8 with SECTOR_CAP=2. `ebitda_margin` dropped per yesterday's finding 1, so
13 features throughout.

Six targets, all reduced to the same binary top-quintile form so only the
label's *content* varies:

| Variant | What it asks the model to prefer |
|---|---|
| `fwd60_baseline` | biggest 60-day gain (what ships today) |
| `fwd21_monthmatch` | biggest 21-day gain (matches the holding period) |
| `fwd60_trailrisk` | 60-day gain per unit of *known* volatility |
| `fwd60_sharpe` | 60-day gain per unit of *realized forward* volatility |
| `fwd60_pain` | 60-day gain, docked by the worst dip along the way |
| `fwd60_graded` | same content as baseline, 0–4 relevance instead of 0/1 |

Then a horizon sweep (5/10/21/42/60/90/120 days) and — because the first
sweep was non-monotonic — a fold-boundary jitter test, sliding the
walk-forward start by 0/7/14/21/28 trading days.

## Findings

### 1. Shorter horizons are better, and the gradient is smooth. (severity: HIGH — actionable)

Mean monthly excess over equal-weight, averaged across the five fold
alignments:

| Horizon | Mean excess | Sign at each of 5 alignments | Spread (sd) |
|---|---|---|---|
| 5 days | **+110 bps** | + + + + + | 36 |
| 10 days | **+110 bps** | + + + + + | **18** |
| 21 days | +54 bps | + + + + + | 25 |
| 60 days (**shipped**) | **−17 bps** | − − − − + | 13 |

This is the smooth, monotone relationship the guardrail sweep lacked
yesterday. The shipped 60-day target is the worst of the four and the only
one that is negative.

⚠️ **These numbers hold the embargo at 60 days for every horizon**, which
handicaps the short ones. See finding 6 — the gradient survives the fix but
the magnitudes shrink and the "10 days is the sweet spot" reading does not.

The single-alignment sweep looked like noise (38.8 → 42.9 → 29.7 → 30.5 →
21.8 → 18.5 → **39.4**), and I nearly called it that. The 120-day spike was
one lucky alignment; under jitter the 5→60 range is orderly.

### 2. At 10 days the basket beats the benchmark on every axis. (severity: HIGH)

Base alignment, 32 months out of sample:

| | CAGR | Sharpe | Max DD | Turnover |
|---|---|---|---|---|
| 10-day target | **42.9%** | **2.39** | **−5.0%** | 45% |
| 60-day target (shipped) | 21.8% | 1.50 | −13.9% | 44% |
| Equal-weight all 96 | 23.1% | 2.03 | −5.9% | — |

**Turnover is flat across horizons** (45% vs 44%), which matters: the
short-horizon advantage is not bought with extra trading, so unmodelled costs
cannot differentially explain it.

Against 1,000 random sector-capped 8-name baskets (median 22.4% CAGR, 95th
percentile 34.0%): the shipped model at 21.8% sits **below the median of
random**, the 10-day variant at 42.9% sits **above the 95th percentile**.

### 3. It is not yet statistically established. (severity: moderate — read before acting)

The 10-day edge is +132 bps/month with a standard error of 64:

| Horizon | Mean excess | 95% CI | t |
|---|---|---|---|
| 10 days | +132 bps | **[+6, +258]** | 2.05 |
| 21 days | +47 bps | [−67, +161] | 0.81 |
| 60 days | −6 bps | [−109, +97] | −0.12 |

The interval clears zero by a hair, on 32 months, after I looked at thirteen
target/horizon combinations. Bonferroni on that many looks wants t ≈ 2.8.
What carries more weight than the t-statistic is the 5-of-5 sign consistency
under jitter and the orderly gradient in finding 1 — but neither is proof.

The return is also **fat-tailed**: only 53% of months are positive; the mean
is carried by three months of +576, +953 and +1073 bps. Dropping the best
three months takes it from +132 to **+56 bps** — still positive, where the
same haircut takes the shipped 60-day target from −6 to **−62 bps**. So the
short-horizon variant survives the haircut and the shipped one does not, but
"survives" is the honest verb, not "wins."

### 4. Risk-adjusting the target does not help; the label's form does not matter. (severity: low)

| Variant | CAGR | vs benchmark 23.1% |
|---|---|---|
| `fwd60_pain` | 29.7% | +51 bps/mo |
| `fwd60_sharpe` | 22.4% | −0 bps/mo |
| `fwd60_graded` | 22.8% | +1 bps/mo |
| `fwd60_trailrisk` | 18.7% | −32 bps/mo |

Teaching it to prefer a smooth path (`fwd60_sharpe`) or to discount jumpy
names (`fwd60_trailrisk`) does nothing or hurts — notable, since the model's
volatility appetite was the suspected disease yesterday. Treating the label
as five graded buckets instead of a yes/no is a wash: **label form is not the
lever, horizon is.**

`fwd60_pain` is the one leftover worth a second look — it matched the 21-day
variant at the base alignment — but it was **never run through the jitter
test**, so it sits exactly where the 120-day spike sat before jitter deflated
it. Treat as unmeasured, not as a finding.

### 5. Seed sensitivity is untestable as configured. (severity: informational)

Three random seeds returned byte-identical results. With no bagging and no
feature subsampling, LGBMRanker is deterministic — `random_state` is inert.
Anyone reading seed agreement as evidence of stability (I ran it expecting
exactly that) is reading noise-free arithmetic. Fold-boundary jitter is the
substitute used above.

### 6. Corrected: giving short horizons their real training data shrinks the effect. (severity: HIGH — supersedes the magnitudes above)

Findings 1–3 hold the embargo at 60 days for every variant, so no version
gets a thinner leakage barrier than another. Comparable, but not what any of
them would actually deploy under: **the gap only needs to be as long as the
label reaches.** A 10-day label dated day 554 is fully resolved by day 564,
still before a test window opening at day 565. Those rows are legitimately
trainable, and the fixed-60 rule discarded ~50 days of them per fold.

Re-run with `train_last ≤ test_start − horizon`, test windows unchanged
(`investor_proper_embargo.py`):

| Horizon | Fixed 60d embargo | Proper embargo | Signs across 5 alignments |
|---|---|---|---|
| 5 days | +110 bps (sd 36) | **+98 bps (sd 21)** | + + + + + |
| 10 days | +110 bps (sd 18) | **+81 bps (sd 48)** | + + + + + |
| 21 days | +54 bps (sd 25) | **+26 bps (sd 38)** | − + + + + |
| 60 days | −17 bps (sd 13) | **−17 bps (sd 13)** | − − − − + |

The 60-day row is byte-identical, as it must be — at that horizon the two
rules are the same arithmetic. That doubles as a regression check on the new
code path.

I expected the extra data to *strengthen* the short horizons. It did the
opposite. Three consequences:

- **The gradient survives**: 98 → 81 → 26 → −17, still monotone in horizon,
  still positive at every alignment for 5 and 10 days. This is the finding.
- **"10 days is the sweet spot" does not survive.** Its tightness under the
  fixed embargo (sd 18) was an artifact of the handicap; properly configured
  it is the *least* stable short horizon (sd 48) and 5 days is the tighter.
  Claim "shorter is better," never a specific number.
- **The effect size is smaller than findings 2–3 state**: ~+80–100 bps/month,
  t ≈ 1, not +132 bps at t=2.05. The CI in finding 3 is correspondingly
  optimistic.

Why fresher training rows hurt is **unexplained**. At sd 21–48 across five
alignments the +110 vs +81 gap is well inside noise, so it may be nothing.
Recorded rather than rationalised. Note also that this adds a second
researcher degree of freedom (embargo rule × horizon), which widens the
multiple-comparisons problem in finding 3 rather than easing it.

### 7. Revision 2026-08-04 — the gradient is not as orderly as findings 1 and 6 claim

Re-examined on request, with two tests that should have come first.

**The smooth gradient is an artifact of averaging correlated slices.** Findings
1 and 6 average five fold alignments. Those alignments share nearly all their
rows; their monthly excess series correlate **0.71**, so five of them carry
about **2.2 independent samples**. At any single alignment the ordering is
ragged — at offset 0 with the proper embargo, restricted to the 32 months all
horizons share:

| Horizon | Mean excess | SE | t | p |
|---|---|---|---|---|
| 5 days | +127.9 bps | 63.5 | 2.01 | 0.053 |
| 10 days | +76.7 bps | 58.7 | 1.31 | 0.201 |
| 21 days | **−24.2 bps** | 47.0 | −0.51 | 0.611 |
| 60 days | −1.1 bps | 52.5 | −0.02 | 0.984 |

The 21-day target lands **below** the 60-day one. The monotone 98 → 81 → 26 →
−17 in finding 6 only appears after averaging; it is not visible in any single
slice.

**The paired test does not rescue power.** Comparing horizons against each
other on the same months should cancel the common market move. It does not
help — the baskets hold different names (monthly-return correlation 0.66–0.77),
so idiosyncratic variance dominates:

| Comparison | Mean | SE | t | p | months won |
|---|---|---|---|---|---|
| 5d − 60d | +128.9 bps | 62.4 | 2.07 | 0.047 | 21/32 |
| 10d − 60d | +77.8 bps | 48.9 | 1.59 | 0.122 | 19/32 |
| 21d − 60d | −23.1 bps | 49.7 | −0.46 | 0.646 | 16/32 |

**What actually survives:** the 5-day target beats both the benchmark
(p = 0.053) and the 60-day target (p = 0.047) at nominal significance — as the
best of thirteen combinations examined, which does not survive any correction
for that search. The 10-day result does not reach significance. The 21-day
result is negative here.

The durable statement is narrower than findings 1–6 imply: **a 5-day target
looks better than the shipped 60-day one, at borderline nominal significance
on 32 months of one bull market, selected from thirteen candidates.** That is a
hypothesis worth testing on data it was not chosen on — not a finding.

## Verification

- Every forward-looking target hand-checked against manually computed values
  at three separate rows per symbol before any model was trained.
- Caught and fixed a real bug mid-build: the first forward-window
  implementation rolled over a `shift(-60)` series, which silently blanked
  the first 59 rows of every symbol. Rewritten to roll on the reversed
  series; coverage went from 108,960 to 114,720 rows, matching the plain
  60-day return exactly.
- Fold boundaries are identical across every variant — the only thing that
  differs between two runs is the label.
- Embargo held at 60 days for all horizons, which is conservative for the
  short ones (a 10-day label needs only a 10-day gap), so no variant is
  advantaged by a shorter leak barrier.
- The baseline reproduces yesterday's headline independently: no edge over
  equal-weight (21.8% vs 23.1% here; 24.9% vs 24.9% yesterday — the gap is
  `ebitda_margin` being dropped and a month-end rather than daily rebalance
  convention).

**Not verified / limits:** no transaction costs or slippage (turnover parity
means this does not change the *ranking* of horizons, but every absolute
number is optimistic). Single 32-month window, one bull regime, and the
universe is survivorship-clean by construction — all three inherited from
yesterday and all three still binding. `fwd60_pain` untested under jitter.

## Risk & follow-ups

1. **Do not ship a 10-day retrain on this evidence alone.** It is one
   window and a best-of-thirteen selection.
   A **10-day embargo** is legitimate for a 10-day label and worth adopting —
   it retains more training rows — but it is *not* the cure for n=32 that an
   earlier draft of this report claimed. The out-of-sample ceiling is set by
   history length, not by the embargo: (1255 dates − 504 to train the first
   model − embargo) / 21 ≈ **32.9 months at a 60-day embargo vs 35.3 months at
   10 days**, i.e. 11 folds → 12. Three extra months, not "far more folds."
   The real cure for the sample size is more history, which is a separate
   piece of work (re-mine back to ~2015; see 2026-08-02 follow-up 5).
2. **A faster rebalance is a separate knob worth testing.** The basket is
   held ~21 days while the best-performing label predicts 10. That mismatch
   working *at all* suggests testing a two-week rebalance directly, rather
   than assuming the label alone should carry it.
3. **Run `fwd60_pain` through jitter** before it gets quoted as a result.
4. **Yesterday's gate finding is unchanged and still the priority.** The
   retrain gate tests lift-over-random, and the shipped model scores below
   the median of random 8-name baskets while passing it. Whatever target
   wins, the gate needs the equal-weight benchmark before it can approve
   anything honestly.
5. Nothing was retrained, promoted, or deployed. The live monthly cron and
   the current model are untouched.

## Files touched

None in the repo except this report. Harness lives in the session scratchpad
(`investor_target_lab.py`, `investor_horizon_sweep.py`, `investor_jitter.py`,
plus `target_lab_results.json`, `horizon_sweep_results.json`,
`jitter_results.json`) and should be moved into `scripts/` if this line of
work continues.
