---
type: handoff
date: 2026-09-06
time: 14:00 PDT
agent: Claude Sonnet 5
model: claude-sonnet-5
trigger: "Brandon wants to hand a detailed architecture summary to an external web-based research agent to explore other strategies and ML algorithms, and didn't know where to start."
head: 7007687bb57dc82943f60e05c3f21a245bdffbe2
branch: feat/strategy-library-and-scorer
scope: read-only
related:
  - recons/2026-09-02_strategy-library-behavior-matrix.md
  - recons/2026-08-11_timeframe-study-h1-killed.md (see MEMORY project_timeframe_study_h1_killed)
  - recons/2026-08-11_crypto-expansion-feasibility.md
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
  - handoffs/2026-06-19_gate-c-and-m1-tradeability.md
files_touched: []
---

# Architecture & research brief — for an external research agent

**Authored 2026-09-06 14:00 PDT · Claude Sonnet 5 (`claude-sonnet-5`).**

Purpose: a self-contained primer to paste into a separate (web-based) research
agent so it can propose new strategies / ML approaches for this codebase
without re-deriving ground this project has already covered. Not a report of
work done this session — no code or config was touched to produce it.

=====================================================================
1. WHAT THIS IS
=====================================================================

A personal algo-trading codebase with two live-ish products sharing one
repo:

- V5 OANDA FOREX BOT — an intraday bot trading currency pairs (and
  nominally metals, though the broker account can't actually place
  metals orders) on 15-minute bars. Currently running as an unattended
  practice-account "soak" (paper money, for evidence-gathering, not
  profit) under a systemd user service, auto-restarted by a cron
  watchdog. This is the primary focus of recent research.

- V4 EQUITIES INVESTOR — a monthly stock ranker on Alpaca. Picks 8
  names from a 96-stock universe, equal-weighted, max 2 per sector,
  rebalances once a month via cron. Holds for weeks; never watches a
  tick. Currently dormant/paper.

Everything else is supporting infrastructure (data adapters, feature
engineering, training/promotion pipeline, offline analysis tools,
tests) or dormant experiments kept for reference (a day-trading
5-minute experiment, an Alpaca crypto "Factory" path, a deleted
feature-research lab, a deleted London-breakout strategy).

=====================================================================
2. CORE DATA FLOW (the forex bot)
=====================================================================

```
OANDA feed -> LiveBarAggregator -> FeaturePipeline -> MLStrategy
                                                          |
                                                  Signal or None
                                                          |
                                                    RiskManager
                                            (position sizing + 3 gates)
                                                          |
                                                     Orchestrator
                                                          |
                                                     OANDA broker
```

Training runs the identical middle section (feature pipeline, gate
logic) against saved history — this symmetry between train-time and
live-time is treated as the single most important invariant in the
codebase. Any change to a gate, a feature, or a cost assumption has to
be made in both places or the model quietly learns from a different
population than it trades on.

=====================================================================
3. THE DECISION LAYERS (three stacked stages)
=====================================================================

STAGE 1 — Angel/Devil (two-stage ML, "meta-labeling")
  - Angel: a classifier tuned for RECALL. Proposes candidate trades.
    Fires above a threshold (~0.40, now calibrated per-model from
    out-of-fold scores rather than fixed).
  - Devil: a second classifier, trained ONLY on bars Angel already
    liked, tuned for PRECISION. Vetoes weak candidates from among the
    proposals. Its threshold is tuned per retrain and saved with the
    model.
  - Rationale: one model tuned for both recall and precision does
    neither well; splitting the job works better in practice.
  - Current model family: LightGBM (swapped from RandomForest
    2026-05-23 — this swap was what got a model past the promotion
    gate; RF's narrower probability distribution kept failing
    sample-size checks even with reasonable Brier/PF numbers).
  - Feature set: ~22-23 columns per bar (momentum, volatility,
    position-in-range, candle shape, session flags for
    Asia/London/NY/overlap, an optional cost-ratio feature, and
    higher-timeframe context features with a strict lookahead guard).
  - Targets: a "macro" label (did price hit target or stop first,
    replayed bar-by-bar up to 45 bars forward) and a "survival" label
    (did price avoid the stop for the next 5 bars — this is what Devil
    actually trains on, because a 45-bar-ahead question was an
    unlearnable mismatch for a model whose inputs describe a 1-5
    minute horizon).

STAGE 2 — RiskManager's "chop filter" (three rule-based gates, applied
AFTER a strategy already wants to trade)
  - Gate A (cost): reject if the stop distance is smaller than
    k_eff × spread. This is the single most important gate in the
    system — see section 5.
  - Gate B (regime/volatility): reject if current volatility sits in
    the bottom 20% of its own trailing 260-bar window (too quiet to
    reach target before the hold limit). This gate does almost all the
    vetoing work in practice.
  - Gate C (time): reject everything in a daily rollover blackout
    window (~16:55-17:30 New York time) when spreads blow out ~10x.
  - These gates are mirrored exactly on the training side
    (retrainer._compute_chop_veto_mask), so the model only ever learns
    from bars the live bot would actually be allowed to take.

STAGE 3 — the regime router (newest, built 2026-09-02, currently
inert)
  - Tags the current bar's market behavior along two axes: volatility
    (low/normal/high) and trend (range/mixed/trend) -> 9 named tags
    like "trend_high", "range_low", plus a "cold" (not-yet-warmed-up)
    state that's excluded from analysis.
  - Looks the tag up in a routing table (JSON: tag -> strategy name or
    null) and either delegates to a named classical strategy or
    "stands down" (declines to trade).
  - The only legitimate routing table is machine-generated by
    build_strategy_matrix.py; hand-authored template tables exist but
    are explicitly marked as unvalidated examples, and the live bot
    refuses to run the router without an explicit table passed in.
  - CURRENT STATE: the generated table is nine stand-downs and
    nothing else, because the underlying strategy library has no
    measured edge anywhere (see section 5). The router itself is
    considered correct/working; it simply has nothing worth routing to
    yet.

=====================================================================
4. THE NON-ML STRATEGY LIBRARY (built 2026-09-02, for comparison
   against the ML model, not currently used live)
=====================================================================

Five classical technical-analysis strategies, each a simple
bars-in/signal-out rule with no ML:
  - SMA crossover
  - RSI mean reversion
  - Bollinger breakout
  - Donchian breakout
  - Momentum

They share a common strategy interface (BaseStrategy: takes a
DataFrame of bars, returns a Signal or None — no broker knowledge, no
sizing) and a lazy registry so listing available strategies doesn't
import anything unused.

=====================================================================
5. WHAT'S ALREADY BEEN TRIED — KEY FINDINGS (read this before
   proposing new strategies, so you don't re-derive it)
=====================================================================

** THE CENTRAL, REPEATEDLY-CONFIRMED FINDING: transaction cost (the
bid/ask spread), not lack of predictive skill, is what's killing edge
in this system at this timeframe. **

This has now been reached independently at least three separate ways:

(a) Strategy library sweep (2026-09-02): 5 strategies x 9 regime tags
    = 45 cells, 28,492 trades, on the 6 tradeable forex crosses (gold
    and silver excluded — the broker account can't actually trade
    them, despite being 38-60% of what earlier, buggier evaluations
    scored). ZERO of 45 cells had positive net expectancy; 42/45 were
    statistically significant on the LOSING side. Win rates ranged
    19.8%-38.5% against a 33.3% break-even rate for the 2:1 bracket
    used. But BEFORE cost, gross expectancy per strategy ranged only
    +0.007R to -0.062R — essentially a coin flip. The entire loss
    (-0.09R to -0.62R per cell) is transaction cost. Doubling the
    target and hold time (2x/8x/180 bars vs 2x/4x/45) to test whether
    trends were being cut off early did NOT help — gross expectancy
    barely moved. Conclusion: the bracket geometry isn't the problem;
    there's no edge to clip.

(b) Stop-width study (2026-08-08): live forex stops were widened
    (1.0x -> 2.0x ATR) and targets widened (2.0x -> 4.0x ATR),
    reasoning about whether stops were "too tight." Finding: stops
    were not too tight in isolation, but the spread was eating ~40% of
    the stop distance either way. Gate PF after modeling cost dropped
    from a superficially good 1.373 gross to ~1.0-1.5 net depending on
    version — break-even to marginal.

(c) Crypto feasibility recon (2026-08-11): considered as a new asset
    class. Data layer already exists (Alpaca crypto feed). Finding:
    fees (not spread) dominate — 0.15-0.25% round trip vs BTC's
    0.118% typical spread — making 15-minute-bar crypto trading
    arithmetically dead even at zero fees (115% cost-to-move ratio).
    HOWEVER: daily-bar crypto trend showed a much better cost ratio
    (6.6% toll vs every forex timeframe tried) — flagged as the
    strongest untried idea in the whole recon (see section 7).

Other load-bearing findings:

- TREND_HIGH REGIME RELIABLY LOSES, four independent times, across
  both the ML model and the entire classical strategy library. Best
  strategy there (RSI mean reversion) still nets -0.154R over 2,094
  trades. This looks like a structural market property at M15 under
  this cost structure, not a fixable gap in any one model.

- TIMEFRAME STUDY (2026-08-11): slower bars cut the cost toll sharply
  (M15 26.7% -> H1 12.3% -> H4 5.0%) but the ML feature set's edge
  ALSO decayed on slower bars — H1 gate failed outright (PF 0.82, win
  rate below its own random-entry base rate). A 5-year M15 lookback
  variant passed the gate at PF 1.72 gross (vs the shipped 2-year
  config's 1.37), isolating training-window length from timeframe as
  a separate, real lever. This candidate was never promoted to live.
  IMPORTANT CAVEAT: the classical-strategy-library sweep in (a) above
  has NOT yet been re-run at H1/H4 — only the ML feature set was
  tested at slower timeframes. This is flagged in the repo as the
  single cheapest remaining experiment.

- DECISION GRADER FINDING (2026-08-24, unresolved puzzle): grading
  ALL ~9,183 scored bars (not just the ~1/week that became fills)
  found the model's calibration is INVERTED at the top of its
  confidence range. It wins 28-32% in its low-confidence bands
  (roughly at the random-entry base rate) but only 6.7% at the 0.40
  threshold it actually trades on. Yet the model's own directional
  call is RIGHT 33.3% of the time on those bars vs a 15.8% baseline —
  it's actually correct about direction more often when confident, but
  the trade still loses because the position gets stopped out 73.3% of
  the time at high confidence vs 56.0% elsewhere. This suggests the
  BRACKET GEOMETRY, not the model's directional skill, is the problem
  specifically in the high-confidence regime — an open, unexplained
  finding worth investigating directly.

- V4 INVESTOR (equities, monthly ranker): a 2.4-year walk-forward
  found NO measured edge over the naive benchmark (equal-weighting the
  same universe) — 24.9% CAGR either way, Sharpe 1.52 vs 2.18 for the
  benchmark (P(beat)=52%, i.e. a coin flip). BUT a follow-up horizon
  study found shortening the prediction target from 60 days to 10 days
  produced a smooth, monotonic improvement (+110bps/month at 10-day vs
  -17bps/month at the shipped 60-day label), consistent across
  fold alignments — not yet statistically significant (t=2.05, n=32,
  best-of-13 comparisons) but the cleanest untried lever on the
  equities side.

- Metals-only model (2026-06-19) was tried and rejected (weak
  separation, PF 1.05) — a dedicated XAU/XAG model isn't viable; the
  existing basket model just gets restricted to them when tradeable.

- The gate/threshold/cost-modeling infrastructure has been through
  several rounds of hardening: realised-R accounting (a trade that
  times out books its ACTUAL small move, not the bracket's nominal
  payoff — the earlier convention inflated apparent win rates by
  10-12 points across every strategy tested), gap-fill realism
  (a bar that opens past a stop/target fills at the open price, not
  the level), and per-instrument measured spread costs (a single flat
  cost assumption was replaced with empirically baked, per-pair
  values ranging 0.072-0.903 — a 12.6x range one constant can't
  represent).

=====================================================================
6. INFRASTRUCTURE ALREADY BUILT FOR TESTING NEW STRATEGY IDEAS
   (use these instead of rebuilding a backtester from scratch)
=====================================================================

- BaseStrategy interface (src/strategies/base.py): the contract any
  new rule-based or ML strategy needs to satisfy — bars DataFrame in,
  a Signal (or None) out, nothing about broker/sizing.

- strategy_backtester.py (src/analysis/): a STRATEGY-AGNOSTIC scorer.
  Takes any BaseStrategy, walks it bar-by-bar over historical data,
  and produces a trade ledger with realised-R accounting, gap-fill
  realism, and (optionally) the live RiskManager's three gates applied
  so vetoed signals are excluded rather than counted as trades. This
  is the tool to plug a new strategy into for an apples-to-apples
  comparison against what's already been measured.

- behavior_matrix.py: scores any set of "candidates" (parameter
  configurations, not necessarily new code — could also score a new
  strategy's ledger) against the 9 market-behavior tags, with
  bootstrap confidence intervals and a minimum-trade-count floor (30)
  before it will recommend anything. Deliberately returns "no
  recommendation" when evidence is thin rather than always picking a
  winner.

- walk_forward_tuner.py: grid search over strategy parameters with
  proper out-of-sample discipline (fresh strategy instance per fold,
  validation slices prefixed with warmup bars so trailing state
  doesn't leak across fold boundaries) and a "robust" mode requiring a
  Clopper-Pearson lower confidence bound on win rate to clear
  break-even — not just a point estimate, which flips easily on
  small trade counts.

- build_strategy_matrix.py: the full pipeline — runs the whole
  strategy library across the whole basket, tags every bar's regime,
  applies live gates and measured costs, and emits both a routing
  table and a raw matrix CSV. This is what produced the finding in
  section 5(a). It exposes bracket geometry (--sl-mult, --tp-mult,
  --max-hold) as swept parameters, and takes --days-back /
  --granularity so re-running at H1/H4 (the flagged next experiment)
  is one command.

- src/core/retrainer.py: the ML training + promotion pipeline for the
  Angel/Devil models — 3-fold expanding walk-forward validation, a
  chronologically-last holdout scored with Clopper-Pearson bounds
  (not point estimates), atomic artifact writes, and an explicit
  "healthy failure" exit code when a candidate model doesn't clear the
  gate (this happens often and is treated as correct behavior, not a
  bug).

- src/ml/features/v3_features.py: the feature vocabulary (momentum,
  volatility, candle shape, session flags, cost-ratio, higher-
  timeframe features with a lookahead guard). Any new ML approach
  built on this codebase's data would likely start from this feature
  set rather than reinventing one.

- scripts/probe_model.py + src/ml/feature_stats.py: a no-retraining
  diagnostic that distinguishes "model is drifting" from "model is
  correctly quiet because there's genuinely no signal right now" using
  PSI with a NULL-CALIBRATED threshold (textbook PSI cutoffs don't
  work on autocorrelated market bars — this tool measures what PSI a
  normal training window produces and only flags drift when live PSI
  exceeds that null's upper tail) plus TreeSHAP attribution.

- scripts/decision_grader.py: grades every bar the live bot evaluated
  (not just the rare fills) against what price actually did next —
  this is how the calibration-inversion finding in section 5 was
  found, and is the tool to re-run against any new model.

=====================================================================
7. WHERE THE UNTRIED, EVIDENCE-BACKED OPPORTUNITY ACTUALLY IS
   (prioritized, in the order the codebase's own recon docs rank them)
=====================================================================

1. RE-RUN THE CLASSICAL STRATEGY LIBRARY AT H1/H4, NOT JUST M15. The
   cost toll falls sharply on slower bars (this is proven for the ML
   feature set) but nobody has run the same 5-strategy x 9-regime
   sweep at H1/H4 yet. One command
   (build_strategy_matrix.py --granularity 60/240). Flagged in the
   codebase itself as "the cheapest remaining shot."

2. DAILY-BAR CRYPTO TREND ON BTC/ETH. The one asset-class/timeframe
   combination measured so far with a genuinely favorable cost ratio
   (6.6% toll vs 26.7%+ for every forex timeframe tried). Scoped but
   never built. Would need to be benchmarked against buy-and-hold BTC,
   not against zero (a prior equities effort made exactly this mistake
   — beating "doing nothing" is a different, harder bar than beating
   random guessing).

3. THE DECISION-GRADER CALIBRATION-INVERSION PUZZLE (section 5). The
   model is directionally MORE right at high confidence but the
   bracket's stop-out rate is also higher there — this smells like a
   volatility-mismatched bracket (confident bars may be higher-
   volatility bars where a fixed-multiple stop is proportionally
   tighter relative to noise, or vice versa). Investigating this
   directly, rather than trying new strategies, might unlock the
   existing model's real edge.

4. SHORTEN THE V4 EQUITIES PREDICTION HORIZON. 10-day labels showed a
   clean, monotonic improvement over the shipped 60-day label but
   remains underpowered (t=2.05, best-of-13 comparisons — needs a
   pre-registered, non-cherry-picked confirmation run rather than
   another parameter sweep).

5. GRID-SEARCH RSI MEAN REVERSION SPECIFICALLY. It's the only
   classical strategy with non-negative GROSS expectancy in the
   library sweep. walk_forward_tuner.py is built and has proper
   robustness checks; a grid search over its parameters has not been
   run. Expected (by the codebase's own prior analysis) to still fail
   once cost is applied, but running it would close the loop rather
   than leave it assumed.

6. SEPARATE LONG AND SHORT IN THE STRATEGY MATRIX. All scoring so far
   pools both directions per strategy per regime. A rule that works
   one way and not the other would average out to the observed zero.
   Not yet tested.

7. ALTERNATIVE MODEL FAMILIES / ARCHITECTURES FOR ANGEL-DEVIL. LightGBM
   is the only booster that's ever cleared the promotion gate (RF was
   tried and rejected on distribution-width grounds, not accuracy).
   Genuinely unexplored: other gradient boosters, probability
   recalibration methods (Platt scaling / isotonic regression) applied
   post-hoc to Angel's output, or replacing the fixed-ATR-multiple
   bracket with a learned/quantile-regression stop-target sizing model
   instead of a constant multiplier — which ties directly into finding
   3 above.

=====================================================================
8. HARD CONSTRAINTS FOR ANY NEW RESEARCH (violate these and the
   result will look great and be worthless)
=====================================================================

- NO LOOKAHEAD. Any feature or label that uses information not
  actually available at decision time will make backtests look
  brilliant and lose money live. The codebase's own worst historical
  bug class. If a higher-timeframe or slower signal is joined onto a
  faster bar series, its timestamp must be pushed forward by a full
  timeframe before joining.

- COST MUST BE MODELED PER-INSTRUMENT, NEVER AS A FLAT CONSTANT. A
  single "spread" number across a basket has repeatedly hidden bad
  results (metals' cost profile is wildly different from fiat crosses,
  and pooling them distorted every early evaluation).

- BOOK REALISED OUTCOMES, NOT NOMINAL BRACKET PAYOFFS. A trade that
  times out pays what it actually moved, not ±the bracket's designed
  R multiple. A trade whose exit bar gapped past the stop/target fills
  at the open, not the level. Getting this wrong inflates apparent win
  rate by 10+ points and has fooled this project before.

- STATISTICAL DISCIPLINE: n≥30 per reported cell/cohort minimum,
  bootstrap or Clopper-Pearson confidence intervals rather than point
  estimates for anything gating a decision, and awareness that
  overlapping/re-sliced fold windows are NOT independent samples (five
  fold alignments in this codebase's own investor gate turned out to
  carry the statistical weight of about 2.2 independent samples, once
  their correlation was measured).

- "BETTER THAN RANDOM" IS NOT "BETTER THAN THE OBVIOUS ALTERNATIVE."
  For trading strategies, compare against a realistic benchmark (buy-
  and-hold for an asset, equal-weighting for a basket, a flat
  break-even rate for a fixed R:R bracket) — not against zero or
  against random guessing. A prior equities model in this project
  passed every "beats random" gate while losing to the trivial
  benchmark.

- DO NOT MODIFY ANYTHING UNDER models/ OR TOUCH THE LIVE SOAK
  PROCESS. Research/backtesting work should be read-only with respect
  to the currently-running practice bot; promoting anything to live is
  a separate, human-approved step.

=====================================================================
9. SUMMARY FOR THE RESEARCH AGENT
=====================================================================

This is a mature, well-instrumented, well-tested trading research
codebase (~390+ tests) that has already run a large number of careful,
statistically disciplined experiments and found that, at the current
timeframe (M15) and asset class (forex majors/crosses), transaction
cost — not lack of directional skill — is the dominant reason nothing
clears its promotion gate. The most credible next moves are (a) change
the timeframe or asset class to get a better cost ratio (H1/H4 forex,
or daily-bar crypto trend) rather than inventing new M15 entry rules,
and (b) investigate the specific calibration-inversion puzzle in the
existing ML model, which suggests there may be real directional skill
being wasted by a mismatched bracket rather than genuinely absent.
Anyone proposing new classical technical-analysis rules at M15 on
these forex pairs should be aware that five independent rule families
have already been tested there and landed at zero net expectancy for
the same underlying reason.
