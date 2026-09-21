---
to: gemini-3.5-flash
from: claude-opus-4-8
date: 2026-06-22
status: rejected
branch: feature/investor-edge-b (deleted — never merged)
topic: investor value factors (Phase B) — built in-house from price + financials we already pull
result_commit: none — REVERTED, not merged
result_notes: >
  REJECTED 2026-06-22. Gemini implemented it cleanly (book_to_price,
  earnings_yield, sales_to_price, roe; shares un-dropped; within-date
  rank-norm; lag preserved — Claude-verified, no leakage, no bug) and
  reported honestly that it REGRESSED. Re-run reproduced exactly:
  P@1 0.3217 -> 0.2950 (1.64x -> 1.51x), P@2 0.2992 -> 0.2383 (1.53x ->
  1.22x, barely clearing the 1.20x floor). The 4 value factors ranked
  HIGH in gain importance (sales_to_price 3rd, book_to_price 5th) yet OOS
  got worse — classic overfitting/noise. Likely causes: the *4 quarterly-
  annualized earnings/sales proxy is noisy, and 46 expensive mega-cap
  survivors have little value spread to exploit. Phase A (1.64x) kept;
  Phase B working changes reverted, branch deleted. Value MIGHT help with
  a cleaner build (true TTM) on a wider/cheaper universe — not worth
  chasing now. See [[project_v4_investor_dormant]].
related_memory: project_v4_investor_dormant
related_report:
---

# MODEL-TO-MODEL HANDOFF — Investor Value Factors (Phase B)

**TO:** Gemini 3.5 Flash (implementing coder)
**FROM:** Claude Opus 4.8 (planner)
**REPO:** `/mnt/storage/mystuf/development/build-A-bot`
**RUNTIME:** venv python `/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python`; run with `PYTHONPATH=src:.` from repo root. `.env` holds `SIMFIN_API_KEY`.
**BRANCH:** create `feature/investor-edge-b` off `feature/investor-edge` (which has Phase A, commit 5b46f4f).

---

## Context — why this design (read carefully, the obvious route is blocked)

Phase A added momentum/volatility/quality factors and took the ranker's Precision@1 lift from 1.05× to 1.64×. Phase B adds **value factors** ("is this stock cheap?") — historically the strongest cross-sectional equity signal.

**The obvious route is dead:** SimFin's pre-computed valuation ratios (P/E, P/B, EV/EBITDA) live in their *derived* dataset, which **is not available on our API tier** — every bulk-download of `derived` / `derived-banks` / `derived-insurance` returns HTTP 500 and there is no cached copy. Do **not** try to use `_DERIVED_LOADERS` / `get_valuation_metrics` / `_find_in_variants(sym, "derived")` — they return empty.

**The working route (verified):** build the value ratios ourselves from the income + balance statements we already download (these load fine) plus daily price. Confirmed present in `_find_in_variants("AAPL","income"|"balance")`: `Shares (Basic)`, `Shares (Diluted)`, `Net Income`, `Revenue`, `Gross Profit`, `Total Equity`, `Total Assets`, `Total Liabilities` — 19 quarters of history. **The only blocker is that `get_quarterly_financials` currently throws the share counts away.**

This is actually *better* than SimFin's pre-baked ratios: point-in-time correct (we own the 45-day lag) and uses live daily price, not a stale quarterly snapshot.

---

## TASK 1 — Stop discarding share counts (`src/data/providers/simfin_fundamentals.py`)

In `get_quarterly_financials`, `_META_COLS` (≈ lines 329-338) drops `"Shares (Basic)"` and `"Shares (Diluted)"`. **Remove those two entries from `_META_COLS`** so shares survive into the merged frame (keep dropping the other meta columns: `SimFinId`, `Currency`, `Fiscal Year`, `Fiscal Period`, `Publish Date`, `Restated Date`).

Note the income↔balance join uses `rsuffix="_balance"` (line ≈349), so income's `Shares (Diluted)` stays un-suffixed and balance's becomes `Shares (Diluted)_balance`. Downstream, **use income's un-suffixed `Shares (Diluted)`**. (`Revenue` is renamed to `Total Revenue` via `_INCOME_RENAME` — already handled.)

---

## TASK 2 — Build value + ROE factors (`scripts/investor_feature_pipeline.py`)

After the miner re-runs, the daily frame `data/raw/v4_investor_data.parquet` will carry `Shares (Diluted)`, `Net Income`, `Total Revenue`, `Total Equity` (all already 45-day-lagged → point-in-time safe). In the feature pipeline's Stage 2 (fundamental/quality block, after the existing quality ratios), add — each guarded by `if <col> in df.columns` with a graceful skip + warning, mirroring the Phase A pattern:

- `market_cap = close * Shares (Diluted)`  (current price × lagged shares — both known at time t)
- `book_to_price = Total Equity / market_cap`   ← cleanest value factor, no earnings needed
- `earnings_yield = (Net Income * 4) / market_cap`   ← quarterly earnings annualized (×4). NOTE in a comment this is a seasonality-naive proxy; a true trailing-twelve-month sum is a later refinement.
- `sales_to_price = (Total Revenue * 4) / market_cap`
- `roe = Net Income / Total Equity`   ← quality factor, no price needed

Then **rank-normalize each new factor within each date** exactly like Phase A:
```python
df[f] = df.groupby("date")[f].rank(pct=True)
```

**Add the new factors to `FACTOR_COLS` in BOTH places they are defined** (they are currently duplicated): `scripts/investor_feature_pipeline.py` AND `scripts/investor_train_model.py`. They must stay identical. (Optional cleanup if low-risk: lift `FACTOR_COLS` into `scripts/investor_universe.py` and import in both — only if it doesn't break the existing imports.)

New `FACTOR_COLS` = Phase A list **+** `["book_to_price", "earnings_yield", "sales_to_price", "roe"]`.

---

## Constraints (hard)

- **No look-ahead leakage:** within-date rank-norm only (`groupby("date")`); fundamentals are already lagged — do not undo the lag; do not use any forward/cross-date statistic.
- No changes under `src/execution/` or `run_soak.sh` (live soak running).
- Do not touch the target definition, walk-forward split, the 45-day lag, or the gate machinery.
- The 7 names already missing income/balance on the free tier (JNJ, XOM, GOOGL, T, WFC, COP, HON) will also lack value factors — that's expected; skip gracefully (NaN), do not error. (Fixing that coverage is a separate data-source decision, NOT part of this task.)

---

## Verification (run in order; paste real output)

1. **Shares survive:** after Task 1, `PYTHONPATH=src:. <venv> -c` to load `SimFinFundamentalProvider().get_quarterly_financials("AAPL")` and confirm `Shares (Diluted)` is a column.
2. **Miner + features:** `PYTHONPATH=src:. <venv> scripts/investor_data_miner.py` then `scripts/investor_feature_pipeline.py` (and `--inference`). Confirm the 4 new factors are in `FACTOR_COLS`, are in [0,1] after rank-norm, and that names with no fundamentals are NaN (not crashing).
3. **Retrain + gate:** `PYTHONPATH=src:. <venv> scripts/investor_train_model.py`. **Report new walk-forward mean P@1 / P@2 / NDCG and lift vs the Phase A baseline (P@1 0.3217, 1.64×).** If lift improves, consider nudging the gate floors up; if it does NOT improve or regresses, say so plainly — value factors are a hypothesis, not a guarantee.
4. **No regressions:** `PYTHONPATH=src:. <venv> -m pytest tests/ -q` stays green (73 tests).

## Report back
- New P@1/P@2/NDCG + lift vs Phase A's 1.64× — headline. Be honest if it doesn't help.
- Per-factor feature-importance for the 4 new value factors (did the model actually use them?).
- Confirm leakage rules (within-date norm, lagged fundamentals untouched) in one line each.
- Any names/factors skipped for missing inputs.
