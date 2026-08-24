---
to: gemini-3.7-flash-spark
from: claude-opus-5
date: 2026-08-17
status: drafted
branch: feat/wider-brackets-and-rename
topic: Why does our gradient-boosted classifier produce compressed probabilities on M15 forex bars, and which fixes would give us MORE confident-and-correct signals rather than merely rescaled ones?
result_commit:
related_memory: project_roadmap_2026-08-17, project_m15_soak_2026-07-13, feedback_lightgbm_was_load_bearing
related_report: llm_reports/recons/2026-07-27_threshold-ev-study.md
---

# Research brief: the score-compression problem

## READ THIS FIRST — scope and ground rules

**You do not have access to this repository.** Everything you need is stated
below. Do not infer facts about our code from file names, and do not assume any
detail that is not written here. If a question you consider important is
unanswerable from this brief, list it in §6 as an open question rather than
guessing — an explicit "I need X to answer this" is a useful result.

**This brief is authoritative as of 2026-08-17.** If you retrieve prior context
about this project from an earlier conversation or your own memory, treat it as
HISTORICAL. Older material was correct when written and has since been
superseded, which makes it easy to mistake for current. **If anything you
retrieve contradicts §1, stop and name the conflict explicitly.**

**This is a RESEARCH task. Produce a written report following §7. Do not write
production code.** Small illustrative snippets or pseudocode are fine where they
make a method concrete. Do not propose branches or commits.

## 1. The system, in the detail you need

We run a live forex bot on a practice account. It trades six currency crosses
(GBP/JPY, AUD/JPY, EUR/JPY, NZD/JPY, GBP/AUD, GBP/NZD) on 15-minute bars.

Per sealed bar, per instrument, the pipeline is:

1. Compute 22 features from the bar history (returns, a volatility measure
   called NATR, PPO, and similar — plus 1-hour higher-timeframe context).
2. A **LightGBM binary classifier we call "Angel"** outputs a probability. Its
   training label is: *did price travel 1.0 x ATR in the favourable direction
   before it travelled 1.0 x ATR against?* This label is evaluated over a
   forward window and is therefore **overlapping across adjacent bars.**
3. If Angel's probability >= **0.40**, a second model ("Devil") must also agree
   (>= 0.66). Devil is a separate classifier acting as a confirmation filter.
4. Three hard veto gates follow (transaction-cost, volatility-regime,
   time-of-day). Survivors become a real order with a stop at 2.0 x ATR and a
   target at 4.0 x ATR.

Relevant facts, all verified:

| Fact | Value |
|---|---|
| Model family | LightGBM gradient-boosted trees, binary objective |
| Features | 22 |
| Training window | **2 years** of 15-minute bars, 2024-08-09 to 2026-08-09, ~392k rows across 8 instruments (a 5-year variant also exists, unpromoted) |
| Angel threshold | 0.40, fixed, pinned in a model-side artifact |
| Bars scored per day (6 instruments) | ~500-600 |
| Realised trade rate | ~1-1.5 trades per week |
| Lifetime fills | 11, of which 1 has ever exited at target |

## 2. The problem — stated precisely

Angel's output probabilities occupy a **narrow band near the bottom of [0,1]**
and essentially never approach 1. Measured over **2,424 consecutively scored
live bars** (2026-08-11 to 2026-08-17, current production model):

| Statistic | Value |
|---|---|
| min | 0.038 |
| median | 0.160 |
| p75 | 0.204 |
| p90 | 0.239 |
| p99 | 0.326 |
| max | **0.452** |
| fraction >= 0.40 | **0.25%** (6 bars of 2,424) |
| fraction >= 0.30 | 1.69% (41 bars) |

Per-instrument medians are tightly clustered (0.156-0.171) and per-instrument
maxima range 0.326-0.452, so this is a property of the model, not of one pair.

The consequence: the 0.40 decision bar sits out near the 99.8th percentile of
the model's own output distribution. The bot is starved of candidates.

## 3. The constraint that kills the obvious answer — read carefully

**"Just lower the threshold" is already ruled out empirically.** We ran an
out-of-sample expected-value study across threshold bands and measured
after-cost performance:

| Angel band | After-cost result |
|---|---|
| >= 0.40 | profitable, profit factor 1.41 (n=20 trades) |
| 0.35 - 0.40 | badly unprofitable, profit factor 0.12, 7% win rate |
| 0.325 - 0.35 | roughly break-even |
| < 0.325 | loses money |

So the score **does** carry real, monotonically useful information — the ranking
is not noise — but the useful region is genuinely tiny.

**This has a sharp methodological implication we want you to reason from, not
around:** a *monotone* recalibration (Platt scaling, isotonic regression,
temperature scaling, beta calibration) **cannot change which bars are selected**
if we re-derive the threshold to preserve the selected set — it only relabels
the same ordering. And if we recalibrate while holding 0.40 fixed, we are
implicitly *moving down the ranking* into the 0.325-0.40 region that the table
above shows loses money. Either way, plain recalibration is a no-op or a
downgrade.

**Therefore: do not spend the report on monotone probability calibration as if
it were the fix.** Address it only to confirm or correct our reasoning above in
a paragraph. The real question is §4.

## 4. The questions we actually want answered

**Q1 — Diagnosis.** What are the known causes of a gradient-boosted binary
classifier producing a compressed, low-ceiling probability distribution on
financial bar data? Cover at minimum: base rate and class imbalance; shrinkage /
learning rate and number of boosting rounds; tree depth and leaf-count limits;
L1/L2 regularisation and `min_child_samples`-style leaf floors; irreducible
Bayes error (i.e. the honest possibility that ~0.45 genuinely *is* the maximum
knowable probability for this label); and **label overlap / sample redundancy**,
since our forward-looking label windows overlap heavily across adjacent
15-minute bars. For each cause, say what diagnostic would confirm or rule it out.

**Q2 — The central question: which interventions change the RANKING?**
Separate your candidate interventions into two buckets and label them
explicitly:

- **(A) Re-ranking / discrimination-improving** — changes *which* bars score
  highest. This is what we need.
- **(B) Rescaling only** — changes the numbers but preserves the order.

For each item in bucket (A), state the mechanism by which it improves ranking,
not just that it "improves the model".

**Q3 — Meta-labelling and the triple-barrier frame.** Our label is effectively a
two-sided barrier race, our samples overlap, and our positive base rate is
therefore both autocorrelated and awkward. Evaluate the López de Prado family of
techniques on their merits for this specific setup: triple-barrier labelling,
**meta-labelling** (a primary model picks the side, a secondary model predicts
*whether the primary is right* — whose base rate is near 50% and whose scores
are consequently far better spread out), average-uniqueness sample weighting,
and sequential bootstrap. Be critical: this literature is popular and partly
unreplicated. Say which parts have independent empirical support, which are
plausible-but-unverified, and which we should skip. In particular: would
meta-labelling actually give us more tradeable candidates, or would it just
re-derive the same tiny surviving set with prettier numbers?

**Q4 — Decision rules that do not need an absolute threshold.** Our bar is a
fixed constant, which couples our trade rate to an artifact of the model's
output scale. Research alternatives: top-k per session or per day; a per-
instrument rolling z-score or percentile of the score; quantile-based selection;
conformal prediction for a calibrated selection guarantee; expected-value
thresholding where the bar is derived from the payoff geometry and cost rather
than chosen. For each: what breaks, and what would we need to measure to trust
it? **Note the trap:** a percentile rule mechanically produces a fixed number of
trades whether or not an edge exists that day, which is exactly the failure mode
the table in §3 warns about. Address that trap directly.

**Q5 — Evaluation.** What should we measure to know whether an intervention
helped, given we can only afford a handful of live trades per week? Cover the
decomposition of the Brier score into reliability / resolution / uncertainty
(resolution is close to a direct measurement of the thing we lack), ROC-AUC
versus precision-recall AUC at a ~30-50% base rate, reliability diagrams, and
**metrics that isolate performance in the extreme upper tail of the score
distribution**, since the top 0.25% of bars is the only region we ever act on.
Ordinary aggregate metrics average over 99.75% of bars we will never trade.

**Q6 — What looks like a fix but is not.** List interventions that would
plausibly appear to help in a backtest while adding no real edge, and say how
each would be detected. We have already been burned once by a benchmark trap in
a sibling product (a strategy that matched an equal-weight baseline while
appearing to work), so we take this question seriously.

## 5. Constraints on your recommendations

- CPU-only training, single workstation. No GPU cluster.
- ~2 years of 15-minute bars per instrument (~49,000 bars each), 6 tradeable
  instruments, and a 5-year extension is available (~124,000 bars each).
  Note these samples are NOT independent: the label window overlaps across
  adjacent bars, so the effective sample size is far below the row count.
- We keep a walk-forward promotion gate: a candidate model must beat fixed
  thresholds on out-of-sample folds before it can go live. Any proposal must be
  scoreable by such a gate.
- Transaction cost is the dominant adversary. The spread consumes roughly a
  quarter of our stop distance on these instruments. A proposal that increases
  trade count without increasing per-trade edge makes us *lose money faster* —
  say so plainly if that is a risk of something you recommend.
- We prefer one well-supported change we can measure over a menu of five.

## 6. Open questions back to us

List anything you needed and did not have. Be specific enough that we can answer
with a number or a file.

## 7. Required report structure

1. **Verdict up front** — in five sentences or fewer: is our compression a
   fixable modelling artifact, an irreducible property of this label on this
   data, or both? Commit to an answer.
2. **Diagnosis** (Q1), with the confirming diagnostic for each cause.
3. **Interventions**, in the two labelled buckets from Q2, each with:
   mechanism / expected effect / cost to try / how it could fool us.
4. **Ranked recommendation** — your top 3, ordered, with the single highest-value
   experiment named first and a one-paragraph experimental design for it.
5. **Answers to Q3, Q4, Q5, Q6** in that order.
6. **What you would NOT do**, and why (Q6).
7. **Sources** — cite them. Distinguish peer-reviewed / independently replicated
   results from blog posts and vendor documentation. Where a claim is folklore,
   label it folklore.

Flag confidence levels throughout. We would rather read "plausible, untested"
than a confident sentence we have to go verify ourselves.
