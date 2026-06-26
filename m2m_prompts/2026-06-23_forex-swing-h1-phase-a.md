---
to: claude-sonnet-4-6
from: claude-opus-4-8
date: 2026-06-23
status: drafted
branch: feature/forex-swing-h1
topic: H1 forex swing model — Phase A (train + validate, no live wiring)
result_commit:
related_memory: project_soak_week_upgrade_backlog
related_report:
---

# MODEL-TO-MODEL HANDOFF — H1 Forex Swing Model (Phase A: train + validate)

**TO:** Claude Sonnet 4.6 (implementing coder)
**FROM:** Claude Opus 4.8 (planner)
**REPO:** `/mnt/storage/mystuf/development/build-A-bot`
**RUNTIME:** venv python `/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python`; run from repo root with `PYTHONPATH=src:.`. `.env` has `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`.
**BRANCH:** create `feature/forex-swing-h1` off `feature/dynamic-chop-floor` (that branch has the retrainer's side-model-isolation + the env knobs below; it is also where the live metals soak runs — do NOT modify that branch, just branch from it).

---

## Context — what and why

Our only live models are 1-minute scalpers, and on 1-min bars only metals clear the transaction-cost gate; the **6 fiat cross pairs (GBP_JPY, AUD_JPY, EUR_JPY, NZD_JPY, GBP_AUD, GBP_NZD) sit idle** because the spread eats the edge (their `alpha = spread/ATR` is 1.25–3.5 vs the ≤0.667 gate). A **1-hour swing model** fixes this: the cost gate compares spread to the typical bar move, and on H1 bars that move is far larger while the spread is unchanged, so `alpha` drops below the gate and the 6 pairs become tradeable.

An audit confirmed the data layer, cost gate, and ATR-based bracket sizing are already **timeframe-agnostic and env-driven**. The ONLY thing hardwired to 1-minute is the **trade-timing labels** (how far ahead a "good trade" is judged). Phase A retunes those for swing horizons and trains an **isolated** H1 model, then verifies it. **No live execution in this phase.**

**Scope discipline:** training only. Do NOT touch `src/execution/` or `run_soak.sh` (a live soak is running). Do NOT change the M1 production path's behavior — all new knobs must default to today's M1 values. The training run does historical REST reads + a local train into an isolated dir; it does not place trades and does not disturb the soak.

---

## TASK 1 — Expose the two hardwired label horizons as env vars (`src/core/retrainer.py`)

The retrainer already reads `RETRAIN_MAX_HOLD` from env (around line 119, pattern: `int(os.getenv("RETRAIN_MAX_HOLD", str(MAX_HOLD_BARS)))`). Mirror that exact pattern for the two horizons that are still hardcoded. **Read the surrounding code to wire them through the same call paths the constants currently flow through** (the line numbers are approximate — confirm them):

1. **Devil survival window** — `SURVIVAL_BARS = 5` (≈ `retrainer.py:199`, consumed by the survival-target function ≈ lines 475/489 as `survival_bars: int = SURVIVAL_BARS`). Add `RETRAIN_SURVIVAL_BARS` (default `5`). Thread it to the same place `SURVIVAL_BARS` is used (don't just change the constant if it's imported elsewhere — follow the actual usage).

2. **Angel lookahead** — the momentum horizon is a literal `shift(-3)` (≈ `retrainer.py:661` and `:707`, e.g. `pl.col("close").shift(-3).over("symbol")`). Introduce a module-level `ANGEL_LOOKAHEAD = int(os.getenv("RETRAIN_ANGEL_LOOKAHEAD", "3"))` and replace the literal `3` in BOTH places with `-ANGEL_LOOKAHEAD`.

**Hard requirement:** defaults are today's values (`5` and `3`), so a run with none of these env vars set produces a byte-identical M1 model. Prove this in verification step 4.

---

## TASK 2 — Train the H1 swing model (isolated)

Run the existing retrainer with the swing "recipe" (all via env — no code change beyond Task 1). Confirm each of these env vars is already honored by `retrainer.py` before running (the audit says they are: `RETRAIN_TIMEFRAME_MINUTES`, `RETRAIN_HTF_TIMEFRAME`, `RETRAIN_MAX_HOLD`, `RETRAIN_MODEL_DIR`, `RETRAIN_SYMBOLS`):

```bash
PYTHONPATH=src:. \
DATA_SOURCE=oanda OANDA_ENV=practice \
RETRAIN_TIMEFRAME_MINUTES=60 \
RETRAIN_HTF_TIMEFRAME=4h \
RETRAIN_ANGEL_LOOKAHEAD=6 \
RETRAIN_SURVIVAL_BARS=12 \
RETRAIN_MAX_HOLD=48 \
RETRAIN_SYMBOLS=XAU_USD,XAG_USD,GBP_JPY,AUD_JPY,EUR_JPY,NZD_JPY,GBP_AUD,GBP_NZD \
RETRAIN_MODEL_DIR=models/forex_swing \
<venv-python> -m src.core.retrainer
```

(`models/forex_swing` isolates the candidate — it must NOT write into `models/forex/`. If the entrypoint differs, find the actual retrainer main and use it; don't invent flags.)

Horizon values are deliberate v1 guesses (H1 → ~6h momentum confirm, ~12h stop-survival, ~2-day max hold). Report them as such.

---

## Verification (run these; paste real output)

1. **Isolation** — after training, `models/forex_swing/` contains the new model + metadata; confirm `models/forex/` file mtimes are UNCHANGED (the M1 prod model was not touched). Metadata in the swing dir shows `timeframe_minutes: 60`.

2. **Model edge (headline, honest)** — report the walk-forward validation-gate result: did it PASS or FAIL, and the real numbers (Brier, profit factor, separation/AUC, pooled OOS trade count). **A swing model is a hypothesis — if it fails the gate, say so plainly with the numbers; do not retune horizons to force a pass without flagging it.** If it fails, note which metric and your read on why.

3. **Cost-gate proof (the thesis)** — show the 6 cross pairs now clear the gate at H1. Method: for each cross pair, fetch a recent window (≈60 days) of M1 and H1 bars from OANDA, compute median true-range in price units for each, then:
   `alpha_H1 = alpha_M1 × (median_TR_M1 / median_TR_H1)`
   using these documented empirical M1 alphas (from the prior soak's spread calibration): EUR_JPY 1.25, GBP_JPY 1.30, AUD_JPY 1.36, NZD_JPY 2.50, GBP_AUD 2.35, GBP_NZD 3.50. Tabulate: `pair | alpha_M1 | TR_M1 | TR_H1 | alpha_H1 | tradeable (≤ 0.667)?`. The thesis predicts most/all 6 flip to tradeable at H1. Report honestly if any don't.

4. **No regression / no behavior change** — `PYTHONPATH=src:. <venv> -m pytest tests/ -q` stays green. AND prove the default path is unchanged: a tiny check that with none of the new env vars set, `ANGEL_LOOKAHEAD == 3` and the survival default `== 5` (e.g. import the module and assert, or a 1-symbol dry comparison). The M1 production behavior must be byte-identical when the new knobs are unset.

## Report back
- Gate result + real numbers (pass/fail honest).
- The cost-gate table (alpha_M1 vs alpha_H1 for the 6 pairs) — did the gate open?
- Confirm `models/forex/` untouched and defaults preserved.
- Any horizon you think is mis-set based on what you saw (these feed the future data-recipe search).
