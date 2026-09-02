---
type: handoff
date: 2026-09-01
time: 19:27 PDT
agent: DeepSeek V4 Pro (dsh)
model: deepseek-v4-pro
trigger: "Brandon's vision for a multi-strategy, regime-routed crypto bot on Binance; summarize the plan so a larger model can do the detailed planning."
head: 51646e92f0edd0a124c18f041b377071f348eaa9
scope: read-only
related:
  - recons/2026-08-11_crypto-expansion-feasibility.md
  - recons/2026-08-22_market-behavior-classifier-and-algorithm-recommender.md
  - recons/2026-08-23_behavior-matrix-and-the-trend-high-hole.md
---

# Multi-strategy regime-routed crypto bot on Binance: plan summary

## Context

Brandon wants to move beyond the single unproven strategy the repo currently
runs. His vision, in his words, is three layers:

1. **A library of common strategies** — standard, non-ML algorithms (moving
   average cross, RSI mean-reversion, Bollinger breakout, Donchian, momentum,
   etc.).
2. **A "master" model that routes them by market regime** — picks the right
   algorithm for each asset in whatever "market weather" is current.
3. **A "tuner" model that dials each algorithm's parameters in** — tweaks params
   until the strategy "fits just right."

Target market: **crypto, on Binance** (not Alpaca), chosen because crypto never
closes. Brandon clarified that "trade everything" does **not** mean always be in
a position — it means *the strategy changes depending on the market*. That
regime-conditional switching is the whole point of the idea.

This report is a **summary of the plan as discussed**, not the detailed plan
itself. Brandon is handing the detailed planning to a larger model; this
document is the handoff that frames it.

## Investigation

Read the repo's documentation and the three prior recons that already cover
most of this ground. The key discovery: **this idea is not greenfield — the
repo already carries most of the parts, and prior recons have already scoped
the hard questions.**

**The strategy contract already fits the library.** `BaseStrategy.generate_signals(df)
-> Signal | None` (`src/strategies/base.py:78`) is exactly the interface a
library of non-ML strategies needs: bars in, a trade or a decline out, no
broker, no sizing. Each new strategy is a ~50-line subclass. The `STRATEGIES`
name→class registry already exists (`src/strategies/concrete_strategies/__init__.py:13`)
but holds a single entry (`ml_strategy`); runtime selection is hardcoded in
`run_oanda.py:234`, not config-driven.

**The regime layer already exists, twice.** `src/ml/regimes/behavior_tagger.py`
produces causal `trend_high` / `range_low` labels *specifically* to answer
"which configuration earns its keep in this kind of market" — that is the
routing question. `src/ml/regimes/hmm_regime.py` is the statistical version
(3-state Gaussian HMM, currently off in production). The behavior tagger is
deliberately causal (trailing window only) so a label means the same thing
offline and live.

**The router is meta-labeling, which the repo already runs.** Angel/Devil is
already a two-stage meta arrangement (`GLOSSARY.md:100-115`). A regime router
is a third stage: given regime + each strategy's signal, pick which (if any) to
act on.

**Prior art, in order of relevance:**

- `recons/2026-08-11_crypto-expansion-feasibility.md` — the crypto cost study.
  Concluded (for **Alpaca**) that fast crypto is arithmetically dead (M15 toll
  115% at taker rates), daily crypto is the cheapest thing measured anywhere
  (6.6% toll), and long-only crypto must be scored against buy-and-hold BTC or
  we manufacture a false positive. **These findings are Alpaca-specific and do
  not automatically carry to Binance** — see Risks.
- `recons/2026-08-22_market-behavior-classifier-and-algorithm-recommender.md` —
  a four-phase plan (behavior tagger → candidate set → behavior matrix +
  recommender → housekeeping) for exactly the routing layer, scoped as an
  offline diagnostic. Phases 1–3 are largely built now.
- `recons/2026-08-23_behavior-matrix-and-the-trend-high-hole.md` — the behavior
  matrix over model *variants* (not new strategies), and the finding that
  `trend_high` is the hole no current config earns its keep in.

## Findings / Changes

**The plan, as discussed, in four phases:**

1. **Strategy library.** A handful of non-ML strategies, each a `BaseStrategy`
   subclass, each backtested standalone against the existing walk-forward
   harness before it touches anything live. Low risk, high value, unblocks
   everything downstream.

2. **Behavior matrix over the strategies.** The existing `behavior_tagger` +
   matrix machinery already scores "which config earns its keep in which
   regime." Extend it from model *variants* to the new strategy *types*. This
   is the router's training data, and it is mostly built.

3. **Router.** Start as a rule table (regime → strategy, with an explicit
   "trade nothing" cell) before it is a model. A model can replace the table
   once the table proves the concept. The router's most important output is
   *"none"* — a router that always picks something will trade into regimes
   where no strategy has edge.

4. **Tuner.** Last, and only as **walk-forward parameter search** (tune on a
   past window, validate on a future window it never saw, roll forward), never
   "fit until perfect." "Dial it in until it fits just right" is overfitting by
   definition; the repo's whole training vocabulary (walk-forward, OOS,
   promotion gate, Clopper-Pearson bounds, effective sample size, null
   calibration) exists to resist exactly that instinct.

**Caveats raised in discussion, to carry into the detailed plan:**

- **"Trades everything in any weather" is the wrong target.** No strategy has
  edge in every regime. The honest router stands down when nothing has edge.
- **More strategies stress the existing entry guards harder.** Post-exit
  cooldown, correlated-exposure cap, and per-currency concentration
  (`GLOSSARY.md:278-306`) exist because "trade everything" blew up on
  2026-07-30 when three correlated yen shorts lost together. A router fanning
  out across many strategies will hit those guards more, not less.
- **The tuner is the dangerous layer.** The legitimate version is walk-forward
  param search; the naive version is overfitting. This is the single most
  important thing for the larger model to get right.

## Verification

Read-only session; no code written, nothing deployed, the forex soak untouched.
Citations are file:line and were read directly (not via a sweep). The three
prior recons were read in full. Git head at time of writing:
`51646e92f0edd0a124c18f041b377071f348eaa9`.

## Risk & follow-ups

1. **Binance changes the cost calculus, and the prior recon is Alpaca-specific.**
   The 2026-08-11 recon's headline findings — no shorting, 0.15%/0.25%
   maker/taker fees, "fast crypto is arithmetically dead" — were measured on
   Alpaca. Binance differs materially: it offers **perpetual futures** (which
   *are* shortable and marginable, removing the "half the signals are unusable"
   constraint) and **much lower fees** (roughly an order of magnitude below
   Alpaca's taker rate, lower still with BNB or VIP tiers). This is the single
   most important thing for the larger model to verify first, because it may
   reopen the fast-crypto question the Alpaca recon closed. **Do not port the
   Alpaca toll table to Binance; re-measure.**
2. **The benchmark problem carries over.** Long-only crypto competes against
   buy-and-hold BTC. Any crypto work must be scored against buy-and-hold BTC
   and an equal-weight basket from the first experiment, or we manufacture the
   same false positive the V4 investor did (`recons/2026-08-11`, finding 5).
   Futures shorting softens but does not remove this.
3. **Two live products share this repo.** The OANDA forex bot and the Alpaca
   equities investor are both live. A crypto/Binance path is a *third* product;
   it must not disturb the running soak (`soak.service`, hot-reloads
   `models/forex/`).
4. **Thin cells.** The behavior matrix will have regimes × strategies ×
   instruments; treat any cell with n < ~30 trades as uninformative
   (`recons/2026-08-22`, risk 1).
5. **Symmetry contract.** The regime tagger must be computable identically at
   training time and live, or the matrix is fiction (`GLOSSARY.md:246-249`).

## What to do next (for the larger model)

1. **Verify the Binance cost/venue facts first** — fee schedule, futures vs
   spot, shorting/margin availability, funding rates on perps, API surface
   (REST + WebSocket). Re-measure the toll table; do not reuse Alpaca's.
2. **Decide the venue shape** (spot vs perp futures) before anything else, since
   it determines whether shorting is available and what the cost floor is.
3. **Produce the detailed plan** for the four phases above, with the tuner
   explicitly scoped as walk-forward param search, and the router's "trade
   nothing" cell made first-class.
4. **Flag any contradiction** between this summary and the repo's current state
   rather than acting on it — this summary was written against head
   `51646e9` and the repo moves fast.

## Files touched

Read only:

- `GLOSSARY.md` (full)
- `src/strategies/base.py`, `src/strategies/README.md`
- `src/strategies/concrete_strategies/` (listing), `src/ml/regimes/README.md`
- `llm_reports/README.md`, `llm_reports/_TEMPLATE.md`, `llm_reports/m2m/README.md`
- `llm_reports/recons/2026-08-11_crypto-expansion-feasibility.md`
- `llm_reports/recons/2026-08-22_market-behavior-classifier-and-algorithm-recommender.md`
