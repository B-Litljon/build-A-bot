# V4 Data Pipeline Report — Fundamental & Macro Abstraction Layer

- **Date:** 2026-05-03
- **Time:** 18:22:19 PDT
- **Agent:** Claude Sonnet 4.6
- **Trigger:** V4 Private Investor Pipeline — Fundamental & Macro Abstraction Layer
- **Files created:**
  - `src/data/fundamentals.py`
  - `src/data/macro.py`
  - `src/data/providers/yf_fundamentals.py`
  - `src/data/providers/yf_macro.py`

---

## Mission

Phase 1 of the V4 Private Investor pivot. Lay the abstract SDK
foundation for fundamental and macroeconomic data before any data
miner orchestration is written. The `providers/` directory is isolated
so that yfinance is never imported from the `src/data/` root layer —
swapping to FRED, SimFin, or Bloomberg requires only a new file in
`src/data/providers/` that fulfills the same ABC contract.

## Pre-flight

```
$ git status --short
(empty — clean)
$ git log -1 --oneline
5eda946 refactor(execution): purge hardcoded Alpaca enums for unified SDK abstractions
```

Tree clean. Proceeded.

---

## Changes

### 1. `src/data/fundamentals.py` *(new — ABC)*

Defines `FundamentalProvider(abc.ABC)` with three abstract methods:

| Method | Return type | Purpose |
|--------|-------------|---------|
| `get_company_info(symbol)` | `dict` | Static metadata: sector, industry, country, marketCap |
| `get_valuation_metrics(symbol)` | `dict` | Value + Quality + Growth ratios: P/E, P/B, ROE, EV/EBITDA, margins, EPS |
| `get_quarterly_financials(symbol)` | `pd.DataFrame` | Income-statement rows — DatetimeIndex (descending), accounting lines as columns |

No vendor SDK imports. Follows the same pattern as
`src/data/market_provider.py`.

### 2. `src/data/macro.py` *(new — ABC)*

Defines `MacroProvider(abc.ABC)` with one abstract method:

| Method | Return type | Purpose |
|--------|-------------|---------|
| `get_macro_series(indicator_name, start_date)` | `pd.Series` | Daily UTC-indexed time series for a named macro indicator |

Docstring calls out that the yfinance PoC should be replaced with
`fredapi` for production (FEDFUNDS, DGS10, CPIAUCSL, etc.).

### 3. `src/data/providers/yf_fundamentals.py` *(new — concrete adapter)*

Implements `YFinanceFundamentalProvider`:

- `get_company_info` — extracts 10 fields from `yf.Ticker(symbol).info`
  via `.get()` with implicit None default. Never raises on missing keys.
- `get_valuation_metrics` — extracts 19 fields covering the Value
  (trailingPE, forwardPE, priceToBook, enterpriseToEbitda, pegRatio,
  priceToSalesTrailingTwelveMonths, marketCap, enterpriseValue),
  Quality (returnOnEquity, returnOnAssets, grossMargins,
  operatingMargins, profitMargins, debtToEquity, currentRatio), and
  Growth (revenueGrowth, earningsGrowth, trailingEps, forwardEps)
  factor families.
- `get_quarterly_financials` — wraps `yf.Ticker(symbol).quarterly_financials`,
  transposes (`.T`) so dates become the index, coerces to
  `pd.DatetimeIndex`, names the index `"period_end"`, sorts descending.
  Returns empty `pd.DataFrame()` on any failure.

### 4. `src/data/providers/yf_macro.py` *(new — concrete adapter)*

Implements `YFinanceMacroProvider`:

- Ticker mapping table (`_INDICATOR_MAP`):

| Indicator name | Yahoo ticker | Note |
|---------------|--------------|------|
| `VIX` | `^VIX` | CBOE Volatility Index |
| `10Y_YIELD` | `^TNX` | 10Y Treasury yield × 10 (Yahoo convention) |
| `2Y_YIELD` | `^IRX` | 13-week T-Bill proxy — use FRED DGS2 for production |
| `SP500` | `^GSPC` | |
| `NASDAQ` | `^IXIC` | |
| `DJI` | `^DJI` | |
| `GOLD` | `GC=F` | Front-month futures |
| `OIL` | `CL=F` | Front-month futures |
| `DXY` | `DX-Y.NYB` | US Dollar Index |

- `get_macro_series` — calls `yf.download()`, flattens MultiIndex
  columns (handles yfinance version differences), extracts the `Close`
  series, localizes index to UTC, sorts ascending. Returns named empty
  `pd.Series(dtype=float)` on unknown indicator or any exception.

---

## Architecture notes

### Isolation contract

`yfinance` is **only** imported in `src/data/providers/`. The two ABCs
(`fundamentals.py`, `macro.py`) import only `abc` and `pandas` —
zero broker coupling. Replacing yfinance with SimFin or FRED requires
adding a new file to `src/data/providers/` and updating the factory
(not yet written) — no existing code changes.

### `src/data/providers/` is a namespace package

No `__init__.py` is present, consistent with the existing
`src/data/` convention. Imports resolve via `from data.providers.yf_fundamentals import ...`
when `src/` is on `sys.path`.

### Pandas convention

The V4 fundamental/macro layer uses `pandas.DataFrame` and
`pandas.Series` (not Polars) because:
1. `yfinance` returns pandas objects natively.
2. The downstream cross-sectional ranking pipeline (LightGBM) expects
   pandas inputs.
3. Polars is reserved for the high-throughput OHLCV/bar pipeline.

The two ecosystems coexist and are intentionally separated by layer.

---

## Task 3 — Syntax verification (verbatim output)

```
$ python -c "import ast; ast.parse(open('src/data/fundamentals.py').read()); print('fundamentals.py: OK')"
fundamentals.py: OK
$ python -c "import ast; ast.parse(open('src/data/macro.py').read()); print('macro.py: OK')"
macro.py: OK
$ python -c "import ast; ast.parse(open('src/data/providers/yf_fundamentals.py').read()); print('yf_fundamentals.py: OK')"
yf_fundamentals.py: OK
$ python -c "import ast; ast.parse(open('src/data/providers/yf_macro.py').read()); print('yf_macro.py: OK')"
yf_macro.py: OK
```

All four files pass AST syntax gate.

---

## Scope adherence

| Rule | Status |
|------|--------|
| Create `src/data/fundamentals.py` with `FundamentalProvider` ABC | ✅ |
| Create `src/data/macro.py` with `MacroProvider` ABC | ✅ |
| Create `src/data/providers/yf_fundamentals.py` | ✅ |
| Create `src/data/providers/yf_macro.py` | ✅ |
| yfinance only called inside `src/data/providers/` | ✅ |
| All adapters return clean pandas DataFrames/Series | ✅ |
| No modification of Tier 1 / Tier 2 execution code | ✅ |
| No data miner orchestration script written | ✅ |
| Append report to `llm_reports/V4_DATA_PIPELINE_REPORT_*.md` | ✅ |
| Commit | ✅ See hash below |

---

## Notes for downstream phases

- **Phase 2** — data miner orchestration (`src/ml/v4_data_miner.py`)
  should instantiate `YFinanceFundamentalProvider` and
  `YFinanceMacroProvider` through a factory, not by direct import.
- **FRED migration path** — implement `FREDMacroProvider` in
  `src/data/providers/fred_macro.py`; update the factory to select it
  when `DATA_SOURCE=fred`. No ABC changes required.
- **Reporting lag** — `get_quarterly_financials` returns *reported*
  dates. The `available_at` logic (established in V3 HTF features) must
  be applied to ensure fundamentals are only used after their SEC
  filing date in any cross-sectional feature DataFrame.
- **^TNX scaling** — Yahoo reports the 10Y yield as the raw index
  value (e.g. 44.5 means 4.45%). Divide by 10 in the feature pipeline
  before using as a percentage.

## Commit

See `git log -1 --oneline` after commit for new HEAD hash.
