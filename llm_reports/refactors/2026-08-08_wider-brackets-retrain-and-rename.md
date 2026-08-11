---
type: refactor
date: 2026-08-08
time: "23:05 PDT"
agent: Claude Opus 5
model: claude-opus-5
trigger: "Ship the cost-gate tighten, widen the brackets, retrain to match, and rename the strategy since it is no longer a scalper."
head: 8ce0f16
branch: feat/wider-brackets-and-rename
scope: modifies-code
related:
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
files_touched:
  - src/execution/risk_manager.py
  - src/core/retrainer.py
  - src/execution/oanda_scalper_orchestrator.py -> src/execution/oanda_forex_orchestrator.py
  - tests/test_oanda_scalper.py -> tests/test_oanda_forex.py
  - tests/test_risk_manager.py
  - CLAUDE.md, table-o-content.md, GLOSSARY.md, + ~40 docs/READMEs
---

# Wider brackets, a matched retrain, and the end of "scalper"

## What shipped

1. **Gate A tightened.** `spread_k_base` 1.5 -> 3.0 for forex. The gate is
   `sl_dist >= k * spread`, i.e. a cap admitting a trade only when the spread
   eats at most `1/k` of the stop. The cap moved from 67% to 33%.
2. **Brackets widened.** `sl_atr_multiplier` 1.0 -> 2.0, `tp_atr_multiplier`
   2.0 -> 4.0. The 2:1 payoff is deliberately unchanged; per the 2026-08-08
   recon, widening does not improve the bracket's odds (stop-hit rate is ~66.6%
   at every width against 66.7% for a fair 2:1) — it dilutes the spread toll.
3. **A retrain matched to the new brackets**, promoted into
   `models/forex_m15_wide/` — a SIDE directory. Nothing live was touched.
4. **`OandaScalperOrchestrator` -> `OandaForexOrchestrator`**, module and test
   file renamed with it; ~48 files of prose. Named by venue+asset rather than
   style, because the horizon has now changed twice.

189 tests pass (185 + 4 new). `compileall` clean. Boot smoke against the new
candidate loads thresholds, drops the untradeable metals, and starts.

## The first retrain was REJECTED, and the reason matters

Attempt 1 failed the gate outright: Brier 0.4146 (ceiling 0.30) and **49 pooled
OOS trades against a floor of 232**. The Devil trained on 852 rows with a 97.7%
survival rate — no variance to learn from, and "NO SIGNAL" separation.

Root cause: **the Angel's label was welded to the execution stop multiple.**

```python
angel_target = close.shift(-3) > close + sl_mult * ATR_abs
```

The Angel answers a DIRECTION question over 3 bars (45 minutes). The stop
multiple belongs to EXECUTION, and the trade has `max_hold` = 45 bars to
resolve. Coupling them meant doubling the stop silently doubled the move the
Angel had to predict in 45 minutes — a far rarer event. Angel positives
collapsed from ~16% to 3.8% of rows, and the whole funnel starved.

**Fix:** `_ANGEL_ATR_MULT_BY_CLASS = {"forex": 1.0, "equities": 0.5}`, plumbed
through as `angel_mult` and overridable via `RETRAIN_ANGEL_ATR_MULT`. The
parameter defaults to `sl_mult` when not passed, so every existing caller keeps
its old behaviour. These defaults hold each asset class's Angel exactly where
it was, so the ONLY thing this retrain changed is the bracket the Devil is
scored against — which is what was intended.

Attempt 2, with the Angel restored: Angel positives 24,095 (from 5,745).

## Gate result — and the number the gate does not test

| | Attempt 1 | Attempt 2 | Shipped model |
|---|---|---|---|
| Mean Brier (<= 0.30) | 0.4146 ❌ | **0.2182** ✅ | 0.2956 |
| Profit factor (>= 1.2) | 2.7273 | **1.3735** ✅ | 1.54 |
| Pooled OOS trades (>= 232) | 49 ❌ | **486** ✅ | 340 |
| Verdict | REJECTED | **PASSED** | promoted 2026-07-02 |

⚠️ **The gate scores profit factor GROSS.** It is
`(wins x tp_mult) / (losses x sl_mult)` with no spread anywhere in it. Re-scored
against the median toll measured in today's recon:

| | Macro WR | Toll | PF gross | **PF after spread** |
|---|---|---|---|---|
| New candidate (2.0x/4.0x) | 40.7% | 21.8% | 1.373 | **1.004** |
| Shipped model (1.0x/2.0x) | 43.5% | 40.2% | 1.540 | **0.878** |

So the change does what it was designed to do — it moves the strategy **from
losing to break-even** — but it does not make it profitable. The new model's
40.7% macro win rate sits almost exactly on the 40.6% break-even this bracket
requires. That is a real improvement and an honest ceiling, both.

This is the same defect the 2026-08-02 investor recon found in a different
gate: it measures the wrong thing. A follow-up should charge the spread inside
`validate_candidate` so PF is net.

## Risk & follow-ups

1. ✅ **RESOLVED 2026-08-10 — soak re-armed and running.** The watchdog had
   `OANDA_MODEL_DIR=models/forex_m15` hardcoded in three places (the old
   1.0x/2.0x model), which against the new tree would have been train/serve
   skew. Hoisted to a single `MODEL_DIR="${SOAK_MODEL_DIR:-models/forex_m15_wide}"`
   at the top of `soak_watchdog.sh`, with a comment stating the invariant: the
   model dir must match the brackets in the checked-out tree. Verified by dry
   run (kill switch honoured; correct dir logged), then launched through the
   watchdog itself so the cron path was exercised end to end — pid 2693644, log
   `logs/soak_2026-08-10_1702.log`, thresholds angel 0.40 / **devil 0.66** read
   from the new dir, metals dropped, 6 crosses trading.
   ⚠️ Still branch-scoped: the watchdog launches from the working tree, so
   checking out a branch without the 2.0x/4.0x brackets re-opens the skew.
2. **Devil separation is weak** — gaps of +0.042 / +0.007 / +0.021 across folds
   ("WEAK", "NO SIGNAL", "WEAK"), approving 89-97% of Angel proposals. It is
   closer to a rubber stamp here than the in-sample 2026-07-27 study found.
   The swept Devil threshold also moved 0.48 -> 0.66.
3. **Gate A never binds during training** (`gate_a=0` on every symbol). Training
   uses the `spread_atr_alpha=0.15` proxy rather than real spreads, so the
   tightened k=3.0 shapes live behaviour but not the training population. The
   live gate is now stricter than the one the model trained against — a
   *conservative* asymmetry, but an asymmetry. Re-baking `spread_alphas.json`
   from an M15 soak would close it.
4. Rename left `llm_reports/`, `m2m_prompts/` and `logs/` untouched — they are
   a historical record. The legacy "Universal Scalper V3.x" Alpaca branding was
   also left alone: that system genuinely was a scalper, and renaming it would
   make the docs less accurate, not more. `src/day_trading/` was rebranded to
   "Intraday Trend Engine V4.0", which is what its own spec calls it.
5. Nothing merged, nothing pushed, nothing promoted to the live model dir.
   `models/forex_m15/` still carries its 2026-07-02 mtimes.
