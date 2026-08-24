---
type: recon
date: 2026-08-08
time: "21:10 PDT"
agent: Claude Opus 5
model: claude-opus-5
trigger: "11 of 13 live fills exited on the stop. Are the stops too tight for M15 noise, and what should the multiple be?"
head: 8ce0f16
scope: read-only
related:
  - recons/2026-07-27_threshold-ev-study.md
files_touched:
  - analysis_cache/2026-08-08_stop_geometry/stop_geometry.py
  - analysis_cache/2026-08-08_stop_geometry/toll_detail.py
  - analysis_cache/2026-08-08_stop_geometry/stop_sweep.py
---

# The stops are not too tight. The spread toll relative to them is too big.

## The question

The live record is 13 fills, 2 wins, 11 exits on the stop. The working
hypothesis was that a 1x-ATR stop sits inside ordinary 15-minute noise and is
being hit by wiggle rather than by the trade being wrong. The ask: measure it,
then pick a multiple, rather than guessing one.

## How the stop is set today

`ml_strategy.py:772` emits raw volatility (`natr_14` converted to an absolute
price distance). `risk_manager.py:331` multiplies it: `sl = 1.0 x ATR`,
`tp = 2.0 x ATR`, both from `RiskProfile.for_asset_class("forex")`
(`risk_manager.py:249`). `oanda_scalper_orchestrator.py:1131` hangs the prices
off the fill. Nothing adapts per instrument, per hour, or per setup.

(The module constants read 0.5x/3.0x. Those are the equity defaults and are
overridden for forex — the trap already flagged in CLAUDE.md.)

## Why the model replay could not answer this

The obvious study — replay history, sweep the multiple, read the money — dies
on sample size. Reusing the 2026-07-27 EV study's cached scoring and varying
the stop reproduces that study **bit for bit** at 1.0x/2.0x (n=4, WR 75.0%,
avg +0.0231%, PF 2.96 — identical to its live-faithful row), which validates
the harness. But the production model finished training 2026-07-02, so the
genuinely-unseen window is ~5 weeks and contains **four** live-faithful
trades. Nothing can be concluded from four, and fetching more history does not
help: older bars are in-sample by construction.

So the question was re-based on something that does not have that ceiling.
Whether a 1x stop is hit before a 2x target is a property of **price**, not of
the model. It can be measured on every bar in the cache — 40,650 hypothetical
entries across the six tradeable crosses, 2026-04-20 to 2026-07-28 — instead
of only the four the model liked.

## Findings

### 1. The bracket geometry is fair at every width. (severity: HIGH — refutes the hypothesis)

Every bar as a hypothetical long entry, resolved SL-first on same-bar
collisions (conservative), 192-bar cap:

| Stop | Stop-hit | Target-hit | Gross R | Median bars held |
|---|---|---|---|---|
| 0.75x | 67.1% | 32.9% | −0.013 | 3 |
| **1.00x (live)** | **66.6%** | **33.4%** | **+0.001** | **5** |
| 1.50x | 66.0% | 33.8% | +0.017 | 9 |
| 2.00x | 65.7% | 33.7% | +0.022 | 16 |
| 3.00x | 64.2% | 32.5% | +0.026 | 33 |

A 2:1 bracket on driftless price hits the stop 2/3 of the time by pure
geometry. The measured rate at the live setting is **66.6% against a
theoretical 66.7%**. There is no excess noise-kill to rescue. Gross R of
+0.001 on a random entry is also the sanity check that the simulator is
unbiased — a no-edge entry earns nothing, as it must.

**Widening the stop does not improve the odds of the bracket.** That was the
hypothesis and it is wrong.

### 2. The spread eats 40% of the stop distance. (severity: HIGH — this is the real defect)

Among bars passing the live regime and blackout gates:

| Stop | Median toll (spread ÷ risk) | Win rate needed to break even |
|---|---|---|
| **1.00x (live)** | **40.2%** | **46.7%** |
| 1.25x | 34.2% | 44.7% |
| 1.50x | 29.0% | 43.0% |
| 2.00x | 21.8% | 40.6% |
| 2.50x | 17.4% | 39.1% |
| 3.00x | 14.5% | 38.2% |

At the live setting every trade opens roughly **four-tenths of a stop-loss in
the hole**. A 2:1 bracket needs 33.3% wins to break even for free; this one
needs 46.7%. The model must be 40% better than the geometry just to pay the
ferryman.

This is the mechanism by which a wider stop helps, and it is **not** the one
we assumed: it does not win more often, it dilutes a fixed cost across more
risk. 1.0x -> 2.0x cuts the toll from 40% to 22% and the required win rate
from 46.7% to 40.6%.

### 3. Gate A is a toll cap, and it is set very loose. (severity: HIGH — cheapest fix)

Gate A is `sl_mult * natr >= k * spread` with `k = spread_k_base = 1.5`. That
is algebraically **`toll <= 1/k = 66.7%`**. The live system will accept a
trade whose spread consumes two-thirds of its stop distance.

This is a separate knob from the stop multiple, it needs no retrain, and it
attacks the defect directly rather than sideways.

### 4. Two instruments cannot pay for themselves at 1x. (severity: moderate)

Median toll as a fraction of risk, at the live 1.0x setting:

| Symbol | Median toll | Bars clearing Gate A |
|---|---|---|
| AUD_JPY | 31.8% | 100% |
| GBP_JPY | 33.9% | 100% |
| EUR_JPY | 35.4% | 100% |
| GBP_AUD | 51.6% | 93.1% |
| NZD_JPY | 53.8% | 89.3% |
| **GBP_NZD** | **60.9%** | **28.9%** |

GBP_NZD is structurally untradeable at this stop width — 61% of its risk goes
to the spread and 71% of its bars are already vetoed. NZD_JPY at 53.8%
explains the 2026-08-04 fill directly: a 7-pip stop on a 0.072 ATR, into which
2.4 pips of entry slippage landed, degrading the intended 2:1 payoff to 1.24:1
before the trade drew a breath.

### 5. The live record is not yet distinguishable from chance. (severity: moderate — read before concluding)

Against the measured 66.6% geometric stop rate, 13 trades should produce ~8.7
stops. We observed 11.

- P(>= 11 stops in 13 | fair geometry) = **0.138**
- P(<= 2 wins in 13 | 33.4% baseline) = **0.138**

Neither clears any reasonable bar. The "stops are killing us" reading is
**consistent** with the record but not **established** by it. What can be said
without a significance test is finding 2: the toll is 40% of risk, and that is
a measured structural fact, not an inference from 13 trades.

## Recommendation

Two changes, in this order, because the first is free:

1. **Raise `spread_k_base` from 1.5 to ~3.0.** Caps the toll at 33% of risk
   instead of 67%. One line, no retrain, no change to the model's meaning. It
   removes exactly the trades that cannot pay for themselves — most of
   GBP_NZD, the worst of NZD_JPY and GBP_AUD — and leaves the three JPY majors
   essentially untouched (their median tolls are already ~32-35%).
2. **Then widen the stop to 2.0x ATR**, holding the 2:1 payoff (target 4.0x).
   Toll 40% -> 22%, required win rate 46.7% -> 40.6%.

⚠️ **Step 2 requires a matched retrain and step 1 does not.** The Devil's
training label *is* "does the target get hit before the stop" — move the stop
and its labels answer a question that no longer exists. `retrainer.py:223`
reads the same `RiskProfile`, so the change propagates on its own, but only on
a retrain. Shipping a wider stop against the current Devil would be a
train/serve skew of exactly the kind the 2026-08-02 investor recon caught.

Also note step 2 changes the strategy's character: median time-to-resolution
goes from 5 bars (~75 min) to 16 (~4h). It stops being a scalper. That is a
product decision, not a parameter.

## Risk & follow-ups

1. Everything here is **long-only**, matching the live strategy. Short
   behaviour is unmeasured.
2. The toll uses the soak's **median** measured spreads. Live spreads widen
   exactly when the model wants to trade, so 40% is a floor, not a ceiling.
3. Entry slippage is **not** modelled — the sim enters at the bar close, live
   enters at the next fill. The 2026-08-04 NZD_JPY fill lost 34% of its ATR to
   that gap. Real tolls are worse than the table.
4. The 192-bar hold cap starts binding above 3x (11% timeouts at 4x), so the
   usable range for widening is 1.5x-3.0x.
5. Nothing was changed, retrained, promoted, or deployed. The soak is
   untouched and still running the current 1.0x/2.0x brackets.

## Files touched

No `src/` changes. Harness and cached results in
`analysis_cache/2026-08-08_stop_geometry/` (`stop_geometry.py` — the noise
floor; `toll_detail.py` — the toll distribution and Gate A; `stop_sweep.py` —
the model-conditional replay that ran out of sample size, kept because it
reproduces the 2026-07-27 study exactly and is the harness to re-run after any
retrain).
