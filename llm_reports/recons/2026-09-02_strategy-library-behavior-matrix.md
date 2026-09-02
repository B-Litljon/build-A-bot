---
type: recon
date: 2026-09-02
time: 14:10 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "Generate a real routing table from measurement, replacing the hand-authored templates."
head: 51646e92f0edd0a124c18f041b377071f348eaa9
branch: feat/trim-100x15-retrain
scope: modifies-data
related:
  - refactors/2026-09-02_strategy-scorer-execution-realism.md
  - recons/2026-08-23_behavior-matrix-and-the-trend-high-hole.md
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
files_touched:
  - config/regime_routing_forex.json
  - logs/strategy_matrix.csv
  - logs/strategy_matrix_roomy.csv
  - analysis_cache/strategy_matrix/
  - src/analysis/build_strategy_matrix.py
---

# The strategy library has no edge in any regime, and the reason is the toll

## Context

The question: **which plain, non-ML strategy earns its keep in which kind of
market?** That is the router's entire premise — if the answer is "different ones
in different regimes", a router is worth building; if it is "none of them
anywhere", the router has nothing to route.

Until today the repo had two hand-authored routing tables presented as answers.
They were written twelve seconds after the backtester and before any measurement
existed (see the companion refactor). This run replaces them with a measurement.

## Investigation

Six tradeable crosses — AUD_JPY, EUR_JPY, GBP_AUD, GBP_JPY, GBP_NZD, NZD_JPY —
on M15, over the two years all six cover in common (**2024-09-02 → 2026-06-26**,
~49,700 bars each after alignment). XAU_USD and XAG_USD were excluded
deliberately: they are broker-dead, yet made up 38–60% of prior model picks,
which is how earlier scores got distorted.

Five strategies (SMA crossover, RSI mean reversion, Bollinger breakout, Donchian
breakout, momentum) × nine behavior tags = 45 cells, **28,492 trades**.

Conditions, all of which differ from the first pass:

- exits book **realised R**, so a timeout pays what it actually moved;
- gapped bars fill at the **open**, not the level;
- the **live gates** run (A cost, B regime, C time blackout);
- costs are the **measured per-instrument spread alphas**, not a flat constant;
- cells need **n ≥ 30** and a bootstrap CI excluding zero to win a regime.

Note on window alignment: the local `data/raw/GBP_JPY_M15.parquet` spans
2023-06-22 → 2026-06-26, a *different* two years from the fresh fetches. Pooling
those would have put "GBP_JPY in 2023" and "EUR_JPY in 2025" in the same cell as
though they were the same regime. Every frame is now trimmed to the common
window before scoring.

## Findings

**1. Not one cell in forty-five has positive expectancy. (severity: decisive)**

```
cells = 45     informative (n>=30) = 45     significant = 42
cells with positive net expectancy        =  0
cells whose CI upper bound even clears 0  =  3
n per cell 39-2094      total trades 28,492
win rate 19.8% - 38.5%          (break-even for this 2:1 bracket: 33.3%)
net expectancy -0.619R to -0.090R
```

This is not a thin-data result. Forty-two of the forty-five cells are
statistically significant — on the losing side. The three that are not
significant are merely too noisy to prove, and are negative in expectation
anyway.

The routing table therefore contains **nine stand-downs and nothing else**.

**2. The strategies are coin flips before cost, and the toll is the whole
loss. (severity: HIGH — this is the real finding)**

Pooled per strategy, trade-weighted:

| strategy | trades | win rate | **gross R** | net R |
|---|---|---|---|---|
| RSI mean reversion | 5,017 | 24.8% | **+0.007** | −0.230 |
| Donchian breakout | 7,345 | 26.5% | **−0.002** | −0.253 |
| Bollinger breakout | 6,917 | 26.0% | **−0.019** | −0.272 |
| Momentum | 5,137 | 25.3% | **−0.024** | −0.277 |
| SMA crossover | 4,076 | 23.4% | **−0.062** | −0.309 |

Before cost these strategies are indistinguishable from zero — the gross column
spans +0.007 to −0.062 across 28,000 trades. Essentially the entire loss, −0.23
to −0.31 R per trade, is transaction cost.

That is the same conclusion the 2026-08-08 stop-width study reached by a
completely independent route, and the same one the crypto feasibility recon
reached for a different asset class: **cost dominates predictive skill in this
system.** Three unrelated measurements now agree.

**3. Low volatility is where the toll hurts most. (severity: MEDIUM)**

Pooled per regime:

| regime | trades | win rate | net R |
|---|---|---|---|
| range_high | 3,643 | 24.4% | −0.193 |
| mixed_high | 3,748 | 23.4% | −0.222 |
| trend_high | 5,834 | 21.9% | −0.224 |
| mixed_normal | 3,568 | 27.3% | −0.284 |
| range_normal | 3,986 | 27.6% | −0.290 |
| range_low | 1,980 | 31.1% | −0.291 |
| trend_normal | 3,121 | 26.0% | −0.312 |
| mixed_low | 1,683 | 27.3% | −0.384 |
| trend_low | 929 | 26.7% | −0.405 |

The ordering inverts win rate. Low-volatility regimes have the *best* win rates
(range_low 31.1%, the highest of any band) and the *worst* net results, because
a fixed spread is a larger share of a smaller move. High-volatility regimes win
less often and lose less money. This is the toll arithmetic showing up as a
regime effect, and it is a caution against reading win rate as quality anywhere
in this repo.

**4. `trend_high` remains the hole, for the fourth independent time.**

The 2026-08-23 matrix found `trend_high` the one cell where the ML model
reliably loses. It has now been measured again with five completely different
strategies: 5,834 trades, **21.9% win rate** against a 33.3% break-even, net
−0.224R. The best any strategy manages there is RSI mean reversion at −0.154R
over 2,094 trades — significant, and still losing.

Nothing in this library fills the hole. It is looking less like a gap in the
model and more like a property of the market at M15 under this cost structure.

**5. The least-bad cells, for the record.**

| regime | strategy | n | win rate | net R | 95% CI |
|---|---|---|---|---|---|
| mixed_high | RSI mean reversion | 433 | 25.9% | −0.090 | [−0.216, +0.034] |
| range_low | RSI mean reversion | 39 | 38.5% | −0.133 | [−0.585, +0.324] |
| trend_high | RSI mean reversion | 2,094 | 22.6% | −0.154 | [−0.206, −0.100] |

RSI mean reversion is the least-bad strategy and the only one with positive
gross expectancy. If anything here deserves a follow-up it is that one — but its
best cell still has a negative point estimate, and the one cell with a genuinely
promising win rate (range_low, 38.5%) has n=39 and an interval wide enough to
drive a bus through.

**6. The gates removed 9,400 signals, almost all on volatility.**

| strategy | vetoed | regime (Gate B) | time (Gate C) |
|---|---|---|---|
| Donchian breakout | 2,818 | 2,582 | 236 |
| Bollinger breakout | 2,710 | 2,397 | 313 |
| Momentum | 1,908 | 1,727 | 181 |
| SMA crossover | 1,106 | 1,033 | 73 |
| RSI mean reversion | 858 | 755 | 103 |

Gate A (cost) fired zero times — unsurprising, since these strategies size their
stop from raw ATR exactly as `MLStrategy` does, and the 2026-08-08 widening to
2.0× put the stop comfortably clear of a 33% toll cap. Gate B did essentially all
the work. Had the gates been skipped, every `*_low` cell would have described a
population roughly 60% of which the bot would refuse to trade.

**7. Giving winners room does not help. The bracket was not the problem.
(severity: HIGH — this closes the question)**

The obvious objection to finding 2 is that a 4×ATR target with a 45-bar hold
truncates exactly the tail trend-following lives on, so "no edge" might really be
"cut off too early". That is cheap to test, so it was tested: the same sweep at
**2.0×/8.0× over 180 bars** — four times the target, four times the hold.

| strategy | gross R @ 2×/4×/45 | gross R @ 2×/8×/180 | delta |
|---|---|---|---|
| RSI mean reversion | +0.007 | +0.005 | −0.002 |
| Donchian breakout | −0.002 | −0.014 | −0.012 |
| Momentum | −0.024 | −0.024 | +0.000 |
| Bollinger breakout | −0.019 | −0.026 | −0.007 |
| SMA crossover | −0.062 | −0.073 | −0.012 |

Gross expectancy does not move. Every strategy is flat or slightly worse with
four times the room, and still zero cells out of forty-five turn positive on net.

Note the trap this run also illustrates: mean win rate **rises** from 25.4% to
32.4% under the roomier bracket, because a 4:1 payoff only needs 20% to break
even. Read as a two-point payoff that looks like a large improvement. It is not
— realised R is unchanged, because the extra "wins" are timeouts booking small
actual moves. This is precisely why the ledger separates `macro_win` from
`net_r`, and precisely the illusion the original scorer was built on.

So the entry rules carry no information at M15 on these pairs. The bracket
geometry was not clipping an edge; there is no edge to clip.

## Verification

Read-only toward `models/` (`git status --short models/` empty). The soak was
running throughout and is still `active`. 391 tests pass.

The bars are cached under `analysis_cache/strategy_matrix/` so the run is
reproducible against the exact data that produced it:

```bash
PYTHONPATH=src:. python -m analysis.build_strategy_matrix \
  --days-back 730 --granularity 15 \
  --output config/regime_routing_forex.json \
  --matrix-out logs/strategy_matrix.csv
```

The generated table carries its own provenance (`_generated_by`,
`_generated_at`, `_basket`, `_gates_applied`, `_significance_required`,
`_min_cell_trades`) — the absence of exactly this metadata is what made the
hand-authored predecessors indistinguishable from findings.

## Risk & follow-ups

1. **Two bracket shapes, not all of them.** 2.0×/4.0×/45 and 2.0×/8.0×/180 were
   both measured (finding 7) and agree. That covers the "cut off too early"
   objection, not every possible geometry — a much tighter or much wider stop is
   untested.
2. **Default parameters only.** No tuning was run. `walk_forward_tuner.py` is
   built and now honest, but a grid search over five strategies has not been
   done; today's answer is about these strategies at their default settings.
3. **One timeframe.** M15 only. The 2026-08-11 timeframe study found the toll
   falls sharply on slower bars while forex skill decayed with them — but that
   was measured on the ML feature set, not on these rules. H1/H4 is untested
   here.
4. **Long and short pooled.** The library emits both directions; the matrix does
   not separate them. A strategy could plausibly work one way and not the other.
5. **The router is built and correct but has nothing to do.** Serving a table of
   nine stand-downs is equivalent to not trading. That is the honest state, and
   it should stay that way until a cell earns its place.
6. **Do not read this as "non-ML strategies don't work".** It says these five
   rules, at default parameters, on one bracket shape, at M15, on six crosses,
   do not overcome this cost structure. The gross-expectancy result suggests the
   binding constraint is cost, not the rules — which points at bracket geometry
   and timeframe, not at better entry logic.

## Recommended next step

The bracket question is answered, so the honest conclusion is: **these five
rules, at default parameters, carry no signal at M15 on these pairs, and no
routing table can rescue that.** The router should keep its nine stand-downs.

Three things are worth doing next, in order of expected value:

1. **Change the timeframe, not the rules.** Gross expectancy sits at zero and
   the loss is entirely toll, so the lever is cost, not entry logic. The
   2026-08-11 study measured the forex toll falling from 26.5% at M15 to 12.3%
   at H1 and 5.0% at H4. That study also found the *ML* feature set losing its
   skill on slower bars, but these are different rules and have not been tried
   there. Re-running this exact matrix at H1 and H4 is one command and is the
   cheapest remaining shot.
2. **Tune, but only to falsify.** `walk_forward_tuner.py` is built and now
   honest (its `robust` flag requires a Clopper-Pearson bound clearing
   break-even). A grid over RSI mean reversion — the one strategy with
   non-negative gross — would establish whether default parameters were simply
   unlucky. Expect it to fail; run it so the failure is on record rather than
   assumed.
3. **Separate long from short.** The matrix pools both directions. A rule that
   works one way and not the other would average to the zero we observe.

What is **not** worth doing is adding more strategies of the same kind at the
same timeframe. Five independent rule families all landing at zero gross is not
five unlucky draws; it is a statement about M15 on these pairs.
