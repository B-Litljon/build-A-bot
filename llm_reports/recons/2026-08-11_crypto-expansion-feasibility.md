---
type: recon
date: 2026-08-11
time: 22:27 PDT
agent: "Claude Opus 5"
model: "claude-opus-5"
trigger: "Brandon: start researching expansion into crypto; deliver a ranked recommendation."
head: b76f375eafe9d69e920c617fdcd559a41958b293
scope: read-only
related:
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
---

# Crypto expansion: what the cost structure allows, and what to build first

## Context

Brandon asked to begin researching a move into crypto and wanted the output as
a ranked recommendation. This report answers one question first, because it
constrains every other choice:

> **At what speed, if any, can we trade crypto profitably given what it costs
> to transact?**

That framing comes straight from today's forex work. The timeframe study
(`2026-08-11`, see memory) established that our forex system is not limited by
model quality but by the toll — the share of a trade's risk consumed by
transaction cost — and that slowing bars down cuts the toll but destroys the
edge. Crypto deserves the same lens before any modelling.

## Investigation

**Existing reuse — better than expected.** Crypto plumbing already exists and
was never wired to a strategy:

- `src/data/enums.py:26` — `AssetClass.CRYPTO`
- `src/data/feed.py:65` — `AlpacaCryptoFeed`, warm-up history + live stream
- `src/data/alpaca_provider.py:52` — a dedicated `crypto_client`; the provider
  already routes on symbol format (`"BTC/USD"` vs equities)

What does *not* exist: crypto is absent from `_DEFAULT_TICKERS_BY_CLASS`
(`src/core/retrainer.py:216`, only `forex` and `equities`), and
`_asset_class_for_source` (`:225`) returns `"equities"` for anything that is
not OANDA. `RiskProfile.for_asset_class` has no crypto branch, so crypto would
silently inherit the equities profile. So: the **data layer is ready, the
model/risk layer is not**.

**Data access verified live**, not assumed — 8 pairs, 15-minute bars, 45 days,
via our existing Alpaca credentials: 4,320 bars per symbol against a theoretical
maximum of 4,320. Crypto trades 24/7, giving **35,040 M15 bars/year vs forex's
24,960** — about 40% more observations per calendar year, which directly helps
the statistical-power constraint that has dogged the forex work.

**Fees are the whole story.** Alpaca crypto is a maker/taker schedule, tier 1
(under $100k/30d): **0.15% maker / 0.25% taker**. Verified against Alpaca's
published schedule rather than recalled. Crypto is also **spot only — not
shortable, not marginable**.

## Findings

**1. The fee dwarfs the spread, inverting the forex situation. (severity: HIGH
— this determines everything)**

Quoted spreads on the majors are genuinely tight — BTC 0.118%, ETH 0.114%,
LINK 0.161%. But a round trip adds 0.50% in taker fees, four times the spread
on BTC. In forex the spread *is* the cost; in crypto the commission is.

Toll (round-trip cost as a share of a 2xATR stop), 8-pair average:

| | M15 | H1 | H4 | D1 |
|---|---|---|---|---|
| taker 0.25%/side | **115%** | 44.3% | 20.7% | **6.6%** |
| maker 0.15%/side | 87.0% | 33.4% | 15.6% | 5.0% |
| zero fees (unreachable) | 44.6% | 17.1% | 8.0% | 2.5% |

*Forex reference: M15 toll 26.5%, which nets PF 1.18 — the system we run.*

**2. Fast crypto is arithmetically dead, not merely marginal.** A toll above
100% means the round trip costs more than the entire stop distance: you lose
more than a full stop-out even when the trade is a winner. Note the zero-fee
row — even if fees vanished entirely, M15 crypto (44.6%) would still be worse
than the forex M15 we already struggle to make money on (26.5%). This is not
fixable by venue shopping or by better execution.

**3. The forex playbook does not port, for three independent reasons.** Any one
would be sufficient: (a) the toll above; (b) no shorting, so half of a
direction-predicting model's signals are unusable; (c) today's H1 finding —
our feature set loses all skill as bars slow, and it reached *below* its own
random-entry base rate at hourly. Reasons (a) and (c) are a vice: cost pushes
us slower, skill decays slower.

**4. Daily crypto is cheaper than anything in forex. (the actual opportunity)**
D1 toll of 6.6% at taker rates beats the best forex figure we measured
anywhere (11.9%, H1). Cost simply stops being the binding constraint at daily
frequency. Whether *edge* exists there is entirely unknown and is the thing to
test.

**5. The benchmark problem is severe and we have already been burned by it.**
Long-only crypto competes against just holding the asset. 3-year buy & hold:

| | CAGR | Sharpe | worst drawdown |
|---|---|---|---|
| BTC | 29.5% | 0.55 | 53.1% |
| ETH | 0.8% | 0.01 | 67.6% |

The V4 investor produced 24.9% CAGR against an equal-weight benchmark's 24.9%
— no edge — precisely because the retrain gate tests lift-over-random and never
lift-over-benchmark. **Any crypto work must be scored against buy-and-hold BTC
and an equal-weight basket from the first experiment**, or we will manufacture
the same false positive.

**6. The BTC/ETH dispersion is the most promising signal in this report.** Over
the same 3 years BTC returned 29.5%/yr and ETH 0.8%/yr. Choosing *which* coin
to hold was worth far more than any entry timing — and choosing is a long-only
operation, so the no-shorting constraint costs nothing.

## Ranked recommendation

Ranked by (expected edge net of cost) x (probability it survives honest
validation) / (implementation cost).

**1. Volatility-targeted trend following on BTC/ETH, daily. — DO THIS FIRST.**
Hold when the trend is up, sit in cash when it is not, size the position so
risk is constant rather than the dollar amount. Toll is negligible at daily
frequency and turnover is low. Time-series momentum is among the better-
documented effects in this asset class, so we are not betting on a novel idea.
Be honest about the payoff: the realistic win is cutting a 53-68% drawdown to
something survivable, *not* beating BTC's return. Cheapest to falsify — 3 years
of daily bars is a single API call (already pulled), and the backtest is an
afternoon. Kill criterion: fails to improve risk-adjusted return over
buy-and-hold BTC.

**2. Cross-sectional coin selection, weekly or monthly (V4-ranker style).**
Rank a crypto universe, hold the top handful, rebalance slowly. Reuses the V4
investor's LightGBM ranker and rebalance machinery, is inherently long-only,
and targets the dispersion in finding 6 — the largest effect measured here.
Ranked second only because V4 has never demonstrated edge over its benchmark in
equities; we would be reapplying an approach that has not yet worked for us.
Worth doing after (1), and only with a benchmark-relative gate.

**3. H4 bracket system — a slowed port of the forex model.** Toll 20.7% is
comparable to the forex M15 we run today, and it reuses the most existing code.
But long-only discards half the signals, and the H1 result predicts the skill
will not be there. Low expected value despite low effort. Only attractive if
(1) and (2) both fail and we want to salvage the existing stack.

**4. M15/H1 scalping — do not build.** Toll 44-115%. Dead on arithmetic before
any model is trained.

**Recommended first step:** a daily-bar backtest of (1) on BTC and ETH scored
against buy-and-hold, with fees charged at the taker rate. Small enough to
finish quickly, and decisive either way.

## Verification

- Crypto data pull executed live against Alpaca with the repo's credentials;
  bar counts reconciled against the 24/7 theoretical maximum (4,320/4,320).
- Spreads sampled from live quotes, not estimated.
- ATR computed with the same Wilder definition the bot uses, median-aggregated,
  over 120d (M15), 365d (H1/H4) and 1,095d (D1).
- Fee schedule and the no-shorting/no-margin constraint taken from Alpaca's
  published documentation, deliberately not from model memory.
- Reuse claims are file:line citations, each opened and read.

## Risk & follow-ups

- **Tolls use *current* spreads.** Crypto spreads widen sharply in stress,
  exactly when a trend system trades. Treat the D1 figures as a floor.
- **Alpaca is one venue.** Lower-fee venues exist; none change finding 2, since
  even zero fees leave M15 unviable. Venue shopping only matters if we later
  want H4.
- **Three years is one crypto cycle.** A daily backtest over 2023-2026 covers
  roughly one drawdown and one recovery — thin for confident conclusions.
- **No code was written and nothing was deployed.** The forex soak was untouched
  throughout.
- Follow-up worth its own report: charging spread inside `validate_candidate`
  so the retrain gate scores net rather than gross — an open item from
  `refactors/2026-08-08_wider-brackets-retrain-and-rename.md` that would serve
  crypto and forex alike.

## Files read

- `src/data/enums.py` (AssetClass), `src/data/feed.py:1-120` (AlpacaCryptoFeed)
- `src/data/alpaca_provider.py:1-95` (dual stock/crypto clients)
- `src/core/retrainer.py:216-266` (asset-class config, tickers, env knobs)
- `src/execution/risk_manager.py:255-285` (per-asset-class risk profiles)
- `llm_reports/recons/2026-08-08_stop-width-and-the-spread-toll.md`

Scratch analysis scripts live in the session scratchpad (`crypto_probe.py`,
`crypto_toll.py`, `crypto_final.py`); they are throwaway and not committed.
