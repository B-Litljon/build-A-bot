---
type: handoff
date: 2026-06-26
time: 14:32 PDT
agent: Claude Opus 4.8
model: claude-opus-4-8
trigger: First implementation step of the London-breakout pivot — build the deterministic GBP/JPY breakout rule and backtest it honestly with realistic costs. Hand the real numbers back to Gemini for the next iteration.
head: 82dbdba114a7f1ac8c2c5cd20b933eece665dd96
scope: modifies-source
related:
  - handoffs/2026-06-26_pivot-to-london-breakout.md
files_touched:
  - src/strategies/concrete_strategies/london_breakout.py
  - backtest_london.py
---

# London-Open breakout on GBP/JPY — first backtest (honest result: no edge after costs)

## Context
The 2026-06-26 pivot: drop the ML sprawl, ship ONE simple hardcoded rule. Gemini's
spec was a London-Open volatility breakout on GBP/JPY. This task = build it
(deterministic, vectorized, no ML) and backtest it **with realistic costs** to find
out whether a small edge survives. A clean "no edge" is a valid, valuable answer.

Confirmed with Brandon before building: **15-minute bars**, **fixed UTC session
windows** (no BST shift in v1), **branch off main** (`feature/london-breakout`).

## Investigation
- Verified the OANDA provider before relying on it: `get_historical_bars("GBP_JPY",
  15, start, end)` (`src/data/oanda_provider.py:250`) returns a polars OHLCV frame
  with UTC tz-aware timestamps, pages 5000 candles/call, M15 supported. Fetched
  **74,862 real M15 bars, 2023-06-22 → 2026-06-26** (cached to
  `data/raw/GBP_JPY_M15.parquet`).
- The live `BaseStrategy` interface (`src/strategies/base.py`) is single-bar and
  live-oriented; a vectorized backtest needs its own self-contained path, so the new
  module owns its stop/target and does NOT subclass it yet (live wiring is a later
  step, per the TP-distance ruling).

**Rule as built:** Asian range = high/low of `00:00–07:00 UTC`. During
`07:00–15:00 UTC`, take the first break of each side (long at the Asian High, short at
the Asian Low). 1:2 RR. Two stop variants tested: **opposite** (far Asian edge) and
**midpoint**. Entry filled at the broken edge (resting stop order); if neither stop
nor target is hit by 15:00 UTC, close at the session-end bar. Cost applied as a flat
round-trip pip charge, swept at 0/3/5 pips. One long + one short max per day.

## Findings / Changes

**Headline: the edge does NOT survive realistic costs.** Best case (opposite stop) is
a coin-flip at an optimistic 3-pip cost and loses money at a realistic 5-pip cost —
and 5 pips is conservative for the GBP/JPY *open*, exactly when spread blows out.

`opposite` stop variant (the only one with a pulse):

| window | cost | trades | win% | PF | maxDD(pips) | exp(pips) |
|--------|------|--------|------|------|-------------|-----------|
| full   | 0    | 924    | 48.8 | 1.15 | 974   | +3.20 |
| full   | 3    | 924    | 47.1 | 1.01 | 1471  | +0.20 |
| full   | 5    | 924    | 45.5 | 0.92 | 2042  | −1.80 |
| **OOS**| 0    | 291    | 50.2 | 1.23 | 610   | +3.79 |
| **OOS**| 3    | 291    | 48.1 | 1.04 | 895   | +0.79 |
| **OOS**| 5    | 291    | 45.4 | 0.94 | 1170  | −1.21 |

`midpoint` stop variant **loses across the board** (full-period PF 0.97 / 0.82 / 0.74
at 0/3/5 pips) — tighter stops get hit far more often than the 1:2 target pays for.

**The most important structural finding — the "1:2 RR" is mostly cosmetic.** Of 924
trades (opposite), only **59 (6%) hit the 1:2 target**; **236 (26%) hit the stop**;
**629 (68%) closed at the London session end** having hit neither. So this is really
an *"enter on the break, exit at 15:00 UTC"* strategy. The thin positive expectancy at
zero cost comes from those session-end exits averaging +0.26R (a small intraday
drift), NOT from the breakout running to a 2:1 target. That drift is far too small to
clear a 3–5 pip toll.

**Engine verified (not flattering itself):** target exits are exactly +2R, stops
exactly −1R, session-end exits fall between (−0.98R to +1.89R). Hand-checked one trade
(2023-10-06) against raw bars: Asian range, entry at the edge, first-break timing, and
target-fill bar all match.

**Sizing on <$400 is feasible, but with a fat tail.** Median stop 61 pips (worst 606).
At reference GBP/JPY 213.5, USD/JPY 161.8: risking 1% ($4) on a 61-pip stop ≈ 1,059
units, ~$28 margin at 50:1 — fine. The worst-case 606-pip stop day is the real danger;
one of those at careless size hurts.

## Verification
- `PYTHONPATH=src:. backtest_london.py --years 3 --oos-months 12` ran end-to-end on
  74,862 real bars.
- Exit-reason R-multiples confirm correct mechanics (target=+2R, stop=−1R exactly).
- Manual trade trace matched raw bars (no lookahead).
- Soak untouched: PID 2168603 alive (4d13h, metals-only); only two new files added; no
  ML imports.

## Risk & follow-ups
- **Entry fill is optimistic** — filled exactly at the range edge. Real London-open
  breakouts gap *through* the level, so true fills are worse. This makes the negative
  conclusion **stronger**, not weaker.
- Flat per-trade cost understates reality (open spread is variable and worst at entry).
- Verdict for the M2M loop with Gemini: **as specified, no durable edge after costs.**
  Next ideas to discuss (don't silently pursue): a filter (only trade wide-enough or
  narrow-enough Asian ranges; skip news days), an earlier London cutoff, or accepting
  this is a "session-drift" play and testing that hypothesis directly rather than a 2:1
  breakout. See the m2m handback for the questions posed to Gemini.

## Files touched
- `src/strategies/concrete_strategies/london_breakout.py` — new, pure polars/numpy rule
  engine (`generate_trades`, `compute_asian_ranges`, `summarize`).
- `backtest_london.py` — new standalone harness (fetch+cache, grid, OOS split, sizing).
- `data/raw/GBP_JPY_M15.parquet` — cached data (not source).
