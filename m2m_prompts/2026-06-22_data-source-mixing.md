---
to: claude-sonnet-4-6
from: claude-opus-4-8
date: 2026-06-22
status: verified
branch: feature/data-source-mixing
topic: mix-and-match fundamental data sources — composite provider + config-driven factory + usable Yahoo fallback
result_commit: a51148d
result_notes: >
  Committed a51148d (coder: Sonnet 4.6), Claude-verified. 91 tests green
  (18 new composite tests). Coverage proof: with FUNDAMENTAL_SOURCES=
  simfin,yfinance all 7 SimFin-orphaned names (JNJ, XOM, GOOGL, T, WFC,
  COP, HON) now return usable financials (all required columns, names
  mapped from yfinance to the contract); SimFin precedence preserved
  (AAPL still 19 quarters); default path unchanged; never-raise contract
  held. Caveat: Yahoo history shallow (~5-7 quarters vs SimFin ~19).
  Unlocks a wider, $0 universe — the next real lever.
related_memory: project_v4_investor_dormant
related_report:
---

# MODEL-TO-MODEL HANDOFF — Mix-and-Match Fundamental Data Sources

**TO:** Claude Sonnet 4.6 (implementing coder)
**FROM:** Claude Opus 4.8 (planner)
**REPO:** `/mnt/storage/mystuf/development/build-A-bot`
**RUNTIME:** venv python `/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python`; run with `PYTHONPATH=src:.` from repo root. `.env` has `SIMFIN_API_KEY`, `ALPACA_*`.
**BRANCH:** create `feature/data-source-mixing` off `feature/investor-edge` (the kept Phase A branch).

---

## Context — the goal

The investor picker is bottlenecked by data: SimFin's premium "ratios" dataset is off our (free) tier, and 7 of the 46 stocks (JNJ, XOM, GOOGL, T, WFC, COP, HON) have **no SimFin fundamentals at all** on the free tier, so they run on price signals only. We want to **mix data sources** — use one, several (with fallback), or none — at **$0**, by adding the free Yahoo financials as a backstop.

**The codebase is already set up for this.** There is a clean ABC `FundamentalProvider` (`src/data/fundamentals.py`) with three methods — `get_company_info`, `get_valuation_metrics`, `get_quarterly_financials` — and a hard contract: **every method returns an empty dict/DataFrame on failure and NEVER raises.** Two concrete impls already exist: `SimFinFundamentalProvider` (`src/data/providers/simfin_fundamentals.py`) and `YFinanceFundamentalProvider` (`src/data/providers/yf_fundamentals.py`). The "never raises / empty on miss" contract is exactly what makes chaining sources safe.

There is already a config-driven factory for *price* providers — `get_market_provider()` in `src/data/factory.py` (reads `DATA_SOURCE` env, lazy-imports the chosen provider). We mirror that pattern for fundamentals, but as a **list** instead of a single choice.

**Scope discipline:** this task is **data-source plumbing only.** Do NOT change the investor model, features, gate, or retrain it. (Aside: the model's walk-forward metric is data-snapshot-sensitive — yfinance silently revises prices — so we verify *coverage*, not model scores, here.)

---

## TASK 1 — `CompositeFundamentalProvider` (the combiner)

New file `src/data/providers/composite_fundamentals.py`. A class implementing `FundamentalProvider`, constructed from an **ordered list** of child `FundamentalProvider` instances:

```python
def __init__(self, providers: list[FundamentalProvider]): ...
```

For each of the three ABC methods: iterate children in order, return the **first non-empty result** (non-empty dict, or non-empty DataFrame). If every child returns empty, return empty (preserve the never-raise/empty contract). Log at DEBUG which source answered for a symbol. Wrap each child call in try/except returning empty on error (belt-and-suspenders — children shouldn't raise, but the composite must be bulletproof).

**The "none" case falls out for free:** `CompositeFundamentalProvider([])` returns empty for everything — exactly the "use no fundamentals" behavior (the feature pipeline already degrades gracefully to NaN). No separate Null class needed.

(v1 is **first-source-wins per symbol** — simple and solves the coverage gap. Do NOT attempt field-level merging across sources yet; note it as a future option in a docstring.)

---

## TASK 2 — `get_fundamental_provider()` factory (`src/data/factory.py`)

Mirror `get_market_provider()` (same file, lazy-import style). Read env `FUNDAMENTAL_SOURCES` (comma-separated, ordered), build each named provider, wrap in `CompositeFundamentalProvider`:

- `FUNDAMENTAL_SOURCES="simfin"` (DEFAULT — preserves today's verified behavior) → SimFin only.
- `FUNDAMENTAL_SOURCES="simfin,yfinance"` → SimFin first, Yahoo fills the gaps.
- `FUNDAMENTAL_SOURCES="yfinance"` → Yahoo only.
- `FUNDAMENTAL_SOURCES="none"` or empty → `CompositeFundamentalProvider([])`.

Registry (lazy-import each only if named, so a missing key/SDK for an unused source never breaks): `"simfin"` → `SimFinFundamentalProvider`, `"yfinance"`/`"yahoo"` → `YFinanceFundamentalProvider`. Unknown name → raise `ValueError` with the offending token (mirror factory.py's existing unknown-source error).

---

## TASK 3 — Wire the investor miner to the factory

`scripts/investor_data_miner.py:185` hardcodes `fundamental_provider = SimFinFundamentalProvider()`. Replace with `fundamental_provider = get_fundamental_provider()` (import from `data.factory`). Default env keeps current behavior; set `FUNDAMENTAL_SOURCES=simfin,yfinance` to enable the fallback. Log which sources are active at startup.

---

## TASK 4 — Make the Yahoo fallback actually USABLE (the important one)

A fallback that returns *non-empty but unusable* data is worthless. **Problem:** `YFinanceFundamentalProvider.get_quarterly_financials` (`src/data/providers/yf_fundamentals.py:111-139`) currently returns **only the income statement** (`yf.Ticker(s).quarterly_financials`, transposed) with yfinance's native column names — **no balance sheet, no share count.** But the V4 feature pipeline needs balance-sheet fields under SimFin-style names: it reads `Total Revenue`, `Gross Profit`, `Net Income`, `Total Assets`, `Total Equity`, `Total Liabilities`, and (Phase A) `Shares (Diluted)` is consumed only if Phase B value factors are on — but `roa`, `debt_to_equity`, `gross_profitability` need `Total Assets`/`Total Equity`/`Total Liabilities`.

Extend `YFinanceFundamentalProvider.get_quarterly_financials` to ALSO pull `yf.Ticker(s).quarterly_balance_sheet`, merge it with the income statement on the period-end index, and **normalize column names to the SimFin-style contract** the feature pipeline expects. yfinance's native names differ — **inspect the real columns at implement time** (`yf.Ticker("AAPL").quarterly_balance_sheet.index`) and map; likely names to map FROM (verify, don't assume):

| feature-pipeline expects | likely yfinance name |
|---|---|
| `Total Revenue` | `Total Revenue` |
| `Gross Profit` | `Gross Profit` |
| `Net Income` | `Net Income` |
| `Total Assets` | `Total Assets` |
| `Total Equity` | `Stockholders Equity` (or `Total Stockholder Equity`) |
| `Total Liabilities` | `Total Liabilities Net Minority Interest` |
| `Shares (Diluted)` | `Diluted Average Shares` (income) or `Share Issued` (balance) |

Keep the never-raise contract (wrap each fetch; if balance sheet fails, still return the income statement). Do NOT apply any lag here — the miner applies the 45-day point-in-time lag externally (`investor_data_miner.py:261-263`); leave that untouched.

---

## Constraints (hard)

- Preserve the `FundamentalProvider` contract: **never raise; empty on miss.**
- No changes under `src/execution/` or `run_soak.sh` (live soak running).
- Do NOT change the investor model/features/gate or retrain in this task — plumbing only.
- Default `FUNDAMENTAL_SOURCES=simfin` so the existing verified pipeline is byte-unchanged unless explicitly opted in.

---

## Verification (run these; paste real output)

1. **Composite unit tests (no network)** — new `tests/test_composite_fundamentals.py`: fake providers proving (a) first non-empty wins, (b) falls through to 2nd when 1st is empty, (c) all-empty → empty, (d) `CompositeFundamentalProvider([])` → empty for all three methods. Keep style consistent with existing `tests/`.
2. **Factory** — assert `get_fundamental_provider()` builds the right composite for `"simfin"`, `"simfin,yfinance"`, `"none"`, and raises on an unknown token (mock/avoid real network where possible).
3. **Coverage proof (the headline)** — for the 7 SimFin-missing names (JNJ, XOM, GOOGL, T, WFC, COP, HON), call `get_fundamental_provider()` built as `simfin,yfinance` and confirm `get_quarterly_financials` now returns **non-empty AND contains usable columns** (`Total Assets`, `Total Equity`, `Net Income`, `Total Revenue`). Report a before/after coverage count (how many of 46 now have balance-sheet data).
4. **No regressions / no silent behavior change** — `PYTHONPATH=src:. <venv> -m pytest tests/ -q` green; and with default `FUNDAMENTAL_SOURCES` unset, the miner still uses SimFin-only (confirm via the startup log).

## Report back
- Coverage before/after for the 7 missing names (did Yahoo actually fill them with *usable* balance-sheet columns?).
- The exact yfinance→contract column mapping you settled on (verified against real yfinance output, not assumed).
- Confirm the never-raise contract held and the default path is unchanged.
- Any name still uncovered by BOTH sources.
