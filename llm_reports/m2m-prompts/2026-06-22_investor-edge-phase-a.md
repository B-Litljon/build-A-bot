---
to: gemini-3.5-flash
from: claude-opus-4-8
date: 2026-06-22
status: verified
branch: feature/investor-edge
topic: strengthen the investor picker — bigger universe + real cross-sectional factors (Phase A)
result_commit: 5b46f4f
result_notes: >
  Committed 5b46f4f, Claude-verified by full re-run: walk-forward P@1 lift
  1.05x -> 1.64x (P@1 0.3217 vs 0.1957 base, P@2 0.2992), leakage review
  clean (within-date rank-norm, trailing/shifted windows only), 73 tests
  green. Caveats logged for follow-up: 7 names lack SimFin free-tier
  fundamentals (JNJ, XOM, GOOGL, T, WFC, COP, HON) and ride price signals
  only; universe is today's survivors (mild backtest flattery). Phase B
  (value factors P/E, P/B) to be dispatched separately by Brandon.
related_memory: project_v4_investor_dormant
related_report:
---

# MODEL-TO-MODEL HANDOFF — Strengthen the Investor Picker (Phase A)

**TO:** Gemini 3.5 Flash (implementing coder)
**FROM:** Claude Opus 4.8 (planner)
**REPO:** `/mnt/storage/mystuf/development/build-A-bot`
**RUNTIME:** venv python `/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python`; run with `PYTHONPATH=src:.` from repo root. `.env` holds `SIMFIN_API_KEY`, `ALPACA_*`.
**BRANCH:** create `feature/investor-edge` off `feature/investor-gate` (which already has the self-gate + metadata from commit ece1090).

---

## Context — the problem you are fixing

The V4 investor stock-ranker barely beats random: walk-forward Precision@1 = 0.300 vs a 0.286 positive base rate (~1.05× lift). Root cause is two compounding weaknesses:

1. **Tiny, homogeneous universe.** It ranks only 7 correlated mega-caps (`AAPL, MSFT, NVDA, JPM, XOM, WMT, JNJ`). Picking "top 2 of 7 look-alikes" has almost no signal ceiling, and "top quintile" of 7 names is just ~2 stocks.
2. **Weak features.** The only genuinely cross-sectional signal is 3 momentum columns. The macro features (VIX, 10Y yield) are **identical for every stock on a given date** → zero ranking power within a date. The fundamentals are largely **raw dollar line items** that track company *size*, not attractiveness. ~59 "features" but only ~3 carry ranking signal.

Phase A fixes both with **low data-plumbing risk** — most of the needed data is already on disk.

### What we already have (confirmed by reading the code)
- `scripts/investor_data_miner.py` pulls daily OHLCV (yfinance) + macro + **income AND balance sheet** (`SimFinFundamentalProvider.get_quarterly_financials` merges both — see `src/data/providers/simfin_fundamentals.py:311-355`). **Balance-sheet columns (Total Assets, Total Equity, Total Liabilities, etc.) are ALREADY in `data/raw/v4_investor_data.parquet`** — the feature pipeline just never uses them.
- Fundamentals are already shifted forward 45 days for point-in-time safety (`investor_data_miner.py:261-263`) — so any ratio you build from those columns is leak-safe.
- The cross-sectional target is top-quintile of 60-day forward return per date (`investor_feature_pipeline.py:89-121, 234-237`).

### What is explicitly DEFERRED to Phase B (do NOT attempt now)
Value ratios from SimFin's *derived* dataset (P/E, P/B, EV/EBITDA, ROE as a clean time series). They need a new provider method + miner join. Leave them for a follow-up handoff so we can measure Phase A's lift in isolation.

---

## TASK 1 — Expand & diversify the universe (~46 names, sector-balanced)

The universe is hardcoded in **two** places that must stay in sync: `scripts/investor_data_miner.py:56` and `scripts/portfolio_orchestrator.py:76`. **De-duplicate**: create `scripts/investor_universe.py` exposing a single `UNIVERSE: list[str]`, and import it in both files (`from investor_universe import UNIVERSE` — both scripts already `sys.path.insert` their dir / src). Keep the variable name `UNIVERSE` so `tests/test_execution_safety.py` (which imports `portfolio_orchestrator.UNIVERSE`) still resolves.

Use this starting list (all large-cap, well-covered by SimFin; banks/insurance variants are handled automatically by `_find_in_variants`). You may adjust names that turn up empty in SimFin, but **keep it sector-diversified and ~40-50 names**:

```python
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO",
    # Communication
    "GOOGL", "META", "NFLX", "DIS", "T",
    # Consumer Discretionary
    "AMZN", "HD", "MCD", "NKE", "LOW",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "COST",
    # Financials
    "JPM", "BAC", "WFC", "GS", "V", "MA",
    # Health Care
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK",
    # Energy
    "XOM", "CVX", "COP",
    # Industrials
    "CAT", "HON", "UPS", "GE",
    # Materials
    "LIN", "SHW",
    # Utilities
    "NEE", "DUK",
]
```

`TOP_K=2` portfolio holding stays as-is for now (a separate portfolio-construction decision).

---

## TASK 2 — Build real cross-sectional factors in `scripts/investor_feature_pipeline.py`

Today the feature matrix is a **deny-list** (`_EXCLUDE_COLS`, line 66) — everything not excluded becomes a feature, which lets raw dollar columns and constant macro columns leak in as noise. **Replace this with an explicit allow-list** of engineered, scale-free factors. Build these raw factors (keep the 3 existing momentum columns), then normalize (Task 3):

**A. Price-derived (free, from `close`) — add to the existing Stage 1/2:**
- `vol_60d` = trailing 60-day std of daily returns; `vol_120d` = trailing 120-day std. (Use `groupby("symbol")["close"].pct_change()` then rolling std with `min_periods=window`.)
- `mom_12_1` = 252-day return **skipping the most recent 21 days** (classic momentum, avoids short-term reversal): `close.shift(21) / close.shift(252) - 1` within each symbol.
- `reversal_1m` = 21-day trailing return (short-term reversal signal): `close / close.shift(21) - 1`.
- Keep existing `mom_3m`, `mom_6m`, `mom_12m`.

**B. Quality ratios from balance-sheet columns ALREADY in the parquet (scale-free):**
- First **inspect the actual column names** present in `data/raw/v4_investor_data.parquet` after running the expanded miner (SimFin names vary, e.g. "Total Assets", "Total Equity", "Total Liabilities", "Total Equity (Loss)"). Log what you find and map defensively (mirror the existing `if col in df.columns` guards at lines 204-216).
- `roa` = Net Income / Total Assets
- `debt_to_equity` = Total Liabilities / Total Equity
- `gross_profitability` = Gross Profit / Total Assets  (Novy-Marx quality factor)
- Keep existing margins (`gross_margin`, `operating_margin`, `net_margin`, `ebitda_margin`).
- For any input column missing, skip that ratio with a warning (don't crash); LightGBM tolerates the NaNs.

**C. Demote the dead features:** the macro columns (`VIX`, `10Y_YIELD`, `vix_sma_20`, `vix_roc_20`, `yield_sma_20`, `yield_roc_20`) and all **raw dollar fundamental columns** must NOT be in the feature matrix (macro = constant within a date; raw dollars = size-dominated). They may remain in the parquet, but the allow-list excludes them.

---

## TASK 3 — Cross-sectional normalization (the single biggest win)

A LightGBM ranker compares stocks **within one date** (one date = one query group). Raw factor magnitudes across different scales/sectors confuse that comparison. **Convert every factor to its within-date percentile rank** so the model sees each stock's *relative standing* that day.

After all raw factors are built, for each factor column `f` in the allow-list:
```python
df[f] = df.groupby("date")[f].rank(pct=True)
```
**CRITICAL — leakage:** normalize **within each date only** (`groupby("date")`). NEVER rank across the whole panel or use any cross-date / global statistic — that leaks the future return distribution into the features. Same rule for the trailing windows in Task 2A: only past data (`.shift()` / trailing `rolling`), never centered or forward windows.

Then define the explicit allow-list the trainer uses, e.g.:
```python
FACTOR_COLS = [
    "mom_3m", "mom_6m", "mom_12m", "mom_12_1", "reversal_1m",
    "vol_60d", "vol_120d",
    "roa", "debt_to_equity", "gross_profitability",
    "gross_margin", "operating_margin", "net_margin", "ebitda_margin",
]
```
Persist these (and only these, plus `date`, `symbol`, `forward_return_60d`, `target_top_quintile`) so `scripts/investor_train_model.py` trains on the allow-list. The trainer currently derives features via `_EXCLUDE_COLS` (line 66/185) — update it to use the saved `FACTOR_COLS` (e.g. read all columns minus the known non-feature set, OR have the pipeline write a sidecar list). Keep it simple: have the trainer select `[c for c in df.columns if c in FACTOR_COLS]`.

---

## Constraints (hard)

- **No look-ahead leakage** — within-date normalization only; trailing/shifted price windows only. This is the #1 risk; if in doubt, prefer the more conservative shift.
- No changes under `src/execution/` or to `run_soak.sh` (a live soak is running off that code).
- Do NOT change the target definition, the walk-forward split logic, the 45-day fundamental lag, or the gate machinery from commit ece1090.
- Keep the existing logging style and graceful-skip pattern for missing columns.

---

## Verification (run these in order; paste real output)

1. **Miner coverage:** `PYTHONPATH=src:. <venv> scripts/investor_data_miner.py` — report how many of the ~46 names returned OHLCV and fundamentals; list any that came back empty.
2. **Feature build:** `PYTHONPATH=src:. <venv> scripts/investor_feature_pipeline.py` then `--inference`. Confirm: the macro & raw-dollar columns are NOT in `FACTOR_COLS`; every factor is in [0,1] after rank-normalization; positive base rate ≈ 0.20 (true quintile now that the universe is ~46).
3. **Retrain + gate:** `PYTHONPATH=src:. <venv> scripts/investor_train_model.py`. **Report the new walk-forward mean P@1 / P@2 / NDCG and the lift over base rate.** The whole point: P@1 lift should rise meaningfully above the old 1.05×. If it does, raise the gate defaults in `investor_train_model.py` (`GATE_P1_MIN_LIFT` etc.) to a real bar (e.g. 1.3×) and confirm the new model passes.
4. **No regressions:** `PYTHONPATH=src:. <venv> -m pytest tests/ -q` stays green.

## Report back
- New walk-forward numbers (P@1/P@2/NDCG + lift) vs the old 1.05× baseline — this is the headline.
- Universe coverage (names with missing SimFin data).
- The final `FACTOR_COLS` list actually used, and any quality ratios you had to skip due to missing balance-sheet columns.
- Confirm the leakage rules (within-date norm, trailing windows) in one line each.
