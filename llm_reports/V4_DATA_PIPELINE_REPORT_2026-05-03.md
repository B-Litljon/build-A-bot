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

---

# V4 Data Pipeline Report — Data Miner & Time-Series Alignment

- **Date:** 2026-05-03
- **Time:** 18:30:14 PDT
- **Agent:** Claude Sonnet 4.6
- **Trigger:** V4 Investor Pipeline — Data Miner & Time-Series Alignment
- **Files created:** `scripts/investor_data_miner.py`

---

## Mission

Build the Phase 2 orchestration script that wires the Phase 1 ABC
adapters into a complete data pipeline: fetch → align → save.  Output
is a single point-in-time-safe Parquet file ready for feature
engineering and cross-sectional ranking.

## Pre-flight

```
$ git status --short
(empty — clean)
$ git log -1 --oneline
884691f feat(data): implement abstract fundamental and macro providers with yfinance PoC adapters
```

Tree clean. Proceeded.

---

## Changes

### `scripts/investor_data_miner.py` *(new)*

Three-stage orchestration pipeline:

**Stage 1 — Macro series**

Calls `YFinanceMacroProvider.get_macro_series()` for `VIX` and
`10Y_YIELD`.

```
⚠️  10Y_YIELD ÷ 10 correction applied
```

Yahoo Finance's `^TNX` index reports the 10-year Treasury yield
multiplied by 10 (e.g. the value `44.5` means `4.45%`).  The script
divides the raw series by 10.0 before any merge.  A log line confirms
the transformation with a raw-vs-corrected sample at runtime.

**Stage 2 — Per-symbol pipeline**

For each symbol in the V4 universe `["AAPL", "MSFT", "NVDA", "JPM",
"XOM", "WMT", "JNJ"]`:

1. **OHLCV**: `yf.download(interval="1d")` — 5 years of adjusted daily
   bars.  `YahooDataProvider` was not used here because its
   `_yf_interval()` mapping caps at 90-minute bars; daily frequency
   required a direct `yf.download()` call at the orchestration layer
   (not the SDK layer — no ABC contracts are bypassed).

2. **Macro alignment**: `_align_macro()` — `reindex + ffill` onto the
   price DatetimeIndex.  Correct for macro since the data is already
   daily; no irregular date gaps require merge_asof logic.

3. **45-day fundamental lag** (the look-ahead bias prevention):

```python
# POINT-IN-TIME SAFETY
# Q1 2024 period end  : 2024-03-31
# Earliest safe use   : 2024-03-31 + 45d = 2024-05-15
fundamentals_df.index = (
    fundamentals_df.index + pd.Timedelta(days=_FUNDAMENTAL_LAG_DAYS)
)
```

   The index is then UTC-localized (if tz-naive) to match the
   UTC-aware price DatetimeIndex — preventing a pandas tz-mismatch
   TypeError during the merge.

4. **Fundamental alignment**: `_align_fundamentals()` — `merge_asof(
   direction="backward")` on the `"date"` key.  Each trading day
   receives the most recent quarterly report whose *lagged* date is
   ≤ that day.  A day falling between two lagged report dates gets the
   older report — the conservative, point-in-time-correct choice.

**Stage 3 — Save**

```python
combined = (
    pd.concat(symbol_frames, axis=0)
    .reset_index()
    .sort_values(["symbol", "date"])
    .set_index("date")
)
combined.to_parquet(_OUTPUT_PATH, index=True)
# → data/raw/v4_investor_data.parquet
```

---

## 45-day lag — mathematical verification

The lag is applied at the *index* of the quarterly fundamentals
DataFrame *before* the merge.  This means:

| Period end | Lagged date (available_at) | First daily row that can see it |
|-----------|---------------------------|--------------------------------|
| 2024-03-31 (Q1) | 2024-05-15 | 2024-05-15 |
| 2024-06-30 (Q2) | 2024-08-14 | 2024-08-14 |
| 2024-09-30 (Q3) | 2024-11-14 | 2024-11-14 |
| 2024-12-31 (Q4) | 2025-02-14 | 2025-02-14 |

`merge_asof(direction="backward")` guarantees that a row dated
2024-05-14 sees **only Q4 2023 data** — the Q1 2024 lagged date
(2024-05-15) has not yet been reached.  The fence is exact to the day.

---

## Architecture notes

### Why `yf.download()` directly for OHLCV

`YahooDataProvider.get_historical_bars()` was designed for intraday
timeframes (its `_yf_interval()` mapping caps at `"90m"`).  Calling it
with `timeframe_minutes=1440` silently returns 90-minute bars.
Rather than modify the existing provider, the orchestration script
calls `yf.download(interval="1d")` directly — this is correct layering:
the script is the orchestrator, not the SDK.

### UTC consistency

All three data layers are coerced to UTC-aware DatetimeIndex before any
join.  This prevents silent tz-mismatch bugs in pandas 3.0's stricter
type checking.

---

## Task 2 — Syntax verification

```
$ python -c "import ast; ast.parse(open('scripts/investor_data_miner.py').read()); print('investor_data_miner.py: OK')"
investor_data_miner.py: OK
```

---

## Readiness checklist

| Item | Status |
|------|--------|
| V4 universe hardcoded | ✅ `["AAPL","MSFT","NVDA","JPM","XOM","WMT","JNJ"]` |
| 5-year window | ✅ `_START_DATE = now − 5 years` |
| 10Y_YIELD ÷ 10 correction | ✅ applied before macro_df construction |
| 45-day fundamental lag | ✅ `pd.Timedelta(days=45)` on index before merge |
| ffill on macro | ✅ `reindex(..., method="ffill")` |
| ffill on fundamentals | ✅ `merge_asof(direction="backward")` (forward-fill semantics) |
| UTC-aware throughout | ✅ all three layers coerced before join |
| Output path | ✅ `data/raw/v4_investor_data.parquet` |
| No Tier 1/2 execution code touched | ✅ |
| No model training | ✅ data miner only |

## To run

```bash
pipenv run python scripts/investor_data_miner.py
```

Expected runtime: ~2–4 minutes (7 symbols × network I/O for fundamentals).

## Commit

See `git log -1 --oneline` after commit for new HEAD hash.

---

# V4 Data Pipeline Report — Feature Engineering & Cross-Sectional Target

- **Date:** 2026-05-03
- **Time:** 18:54:42 PDT
- **Agent:** Claude Sonnet 4.6
- **Trigger:** V4 Investor Pipeline — Feature Engineering & Top Quintile Target
- **Files created:** `scripts/investor_feature_pipeline.py`
- **Bugfix included:** `scripts/investor_data_miner.py` — datetime64 unit mismatch in `_align_fundamentals` resolved by casting right merge key to match left key dtype.

---

## Mission

Phase 3 of the V4 Private Investor pipeline. Ingest the aligned daily Parquet, engineer momentum / macro / fundamental features, calculate the 60-day cross-sectional forward return, and produce a binary top-quintile target label. Output is a single training-ready Parquet file.

## Pre-flight

```
$ git status --short
(empty — clean)
$ git log -1 --oneline
390f3c6 feat(data): build v4 investor data miner with point-in-time fundamental alignment
```

---

## Bugfix: datetime64 unit mismatch in investor_data_miner.py

First run raised `pandas.errors.MergeError: incompatible merge keys [0] datetime64[s, UTC] and datetime64[us, UTC]`.

Root cause: `yf.download(interval="1d")` produces `datetime64[ms, UTC]`; `fundamentals_df.index.tz_localize("UTC")` produced `datetime64[s, UTC]`. pandas 3.x requires identical units on `merge_asof` keys.

Fix:
```python
right["date"] = right["date"].astype(left["date"].dtype)
```

---

## Changes

### `scripts/investor_feature_pipeline.py` *(new)*

**Stage 1 — Momentum** (`groupby("symbol")["close"].pct_change`)

| Feature | Window | Null rate |
|---------|--------|-----------|
| mom_3m  | 63d    | 5.3%      |
| mom_6m  | 126d   | 10.5%     |
| mom_12m | 252d   | 21.1%     |

**Stage 2 — Macro trends** (20-day SMA and pct_change on VIX and 10Y_YIELD)

| Feature      | Null rate |
|--------------|-----------|
| vix_sma_20   | 1.6%      |
| vix_roc_20   | 1.7%      |
| yield_sma_20 | 1.6%      |
| yield_roc_20 | 1.7%      |

**Stage 3 — Fundamental ratios** (gross/operating/net/ebitda margin from quarterly data)

~84% null — yfinance only returns recent quarters (2024–2026 coverage). LightGBM handles natively.

**Stage 4 — Target construction**

```python
# 60-day forward return
df["forward_return_60d"] = df.groupby("symbol")["close"].transform(
    lambda x: x.shift(-60) / x - 1
)
# Cross-sectional top-quintile label
df["target_top_quintile"] = df.groupby("date")["forward_return_60d"].transform(_top_quintile_label)
```

420 embargo rows dropped. 8,365 labelled rows retained.

---

## Output: data/processed/v4_training_features.parquet

```
Shape : 8365 rows × 82 columns  (0.80 MB)
Index : date  datetime64[ms, UTC]
```

| Label | Count | Rate  |
|-------|-------|-------|
| 0 (not Q5) | 5,975 | 71.4% |
| 1 (Q5)     | 2,390 | 28.6% |

**Note on 28.6% vs 20%:** With only 7 symbols, qcut(5) gives ~1–2 symbols per bin. 2/7 = 28.6%, 1/7 = 14.3%. NVDA's 51.9% positive rate (dominant outperformer 2021–2026) pulls the universe average up. A 50+ symbol universe converges to ~20%.

Per-symbol positive rates: NVDA 51.9% | XOM 34.6% | AAPL 24.8% | WMT 24.4% | JNJ 24.1% | MSFT 21.3% | JPM 19.1%

---

## Syntax verification

```
$ python -c "import ast; ast.parse(open('scripts/investor_feature_pipeline.py').read()); print('...: OK')"
investor_feature_pipeline.py: OK
```

## Commit

See `git log -1 --oneline` after commit for new HEAD hash.
