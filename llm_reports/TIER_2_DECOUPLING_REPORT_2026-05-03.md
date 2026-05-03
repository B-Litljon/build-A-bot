# Tier 2 Decoupling Report — Alpaca Enum Purge

- **Date:** 2026-05-03
- **Time:** 15:44:38 PDT
- **Agent:** Claude Sonnet 4.6
- **Trigger:** Tier 2 Alpaca Enum Purge (Gemini 3.1 Pro brief)
- **Files modified:**
  - `src/execution/factory_orchestrator.py`
  - `src/data/discovery.py`
  - `src/core/retrainer.py`
  - `src/execution/enums.py` (extended — initially, see Notes)
  - `src/data/enums.py` (new — broker-agnostic data-layer enums)

---

## Mission

Purge the remaining hardcoded Alpaca enum imports (`OrderSide`,
`TimeInForce`, `AssetClass`, `AssetStatus`, `TimeFrame`,
`TimeFrameUnit`, `DataFeed`) from the three target files and replace
them with broker-agnostic abstractions defined inside
`src/`. Leave the strategies layer and `live_orchestrator.py`
strictly untouched.

## Pre-flight

```
$ git status --short
(empty — clean)
$ git log -1 --oneline
2f6595e fix(runner): suppress cosmetic warnings and add shutdown timeout
```

Tree clean. Proceeded.

## Changes

### 1. `src/execution/enums.py`

Already exported `OrderSide`, `TimeInForce`, `OrderType`. No
schema changes — file is unchanged from its pre-Tier-2 state by
the time the commit lands.

### 2. `src/data/enums.py` *(new file)*

Created to host data-layer broker-agnostic enums:

| Enum | Values | Justification |
|------|--------|---------------|
| `AssetClass` | `US_EQUITY`, `CRYPTO` | Asset metadata for discovery filters |
| `AssetStatus` | `ACTIVE`, `INACTIVE` | Tradability flag for discovery filters |
| `DataFeed` | `IEX`, `SIP` | Market-data feed routing |

All three are `str, Enum` with values matching Alpaca's underlying
strings, so Alpaca's pydantic validators coerce them transparently
(verified end-to-end below).

**Why a separate `data/enums.py` module instead of folding into
`execution/enums.py`?** Two reasons:

1. **Semantic separation.** `AssetClass` / `AssetStatus` / `DataFeed`
   are all data-layer concerns (asset metadata + market-data routing),
   not order-execution concerns.
2. **Eager-import pitfall.** `src/execution/__init__.py` currently
   re-exports `FactoryOrchestrator`. That triggers eager loading of
   `factory_orchestrator.py`, which uses `from data.feed import …`
   (i.e. requires `src/` on `sys.path`). When `retrainer.py` imports
   via `from src.execution.X` (project-root convention), the package
   `__init__` fires under the wrong path convention and crashes.
   `src/data/` has no `__init__.py` (namespace package), so importing
   `from src.data.enums import DataFeed` is side-effect-free.

This is the brief's "or equivalents depending on local project
structure" clause exercised — `src.data.enums` is the equivalent
location for data-layer enums.

### 3. `src/execution/factory_orchestrator.py`

```diff
-from alpaca.trading.enums import OrderSide, TimeInForce
+from execution.enums import OrderSide, TimeInForce
```

Code references at lines 148 and 183 (`OrderSide.BUY`, `OrderSide.SELL`,
`TimeInForce.GTC`) are unchanged — Alpaca's `MarketOrderRequest`
pydantic model accepts our `str, Enum` values via string coercion.

### 4. `src/data/discovery.py`

```diff
-from alpaca.trading.enums import AssetClass, AssetStatus
+from data.enums import AssetClass, AssetStatus
```

Code references at line 33 (`AssetClass.US_EQUITY`,
`AssetStatus.ACTIVE`) unchanged.

### 5. `src/core/retrainer.py`

Three coordinated edits:

```diff
-from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
-from alpaca.data.enums import DataFeed
+from alpaca.data.timeframe import (
+    TimeFrame as _AlpacaTimeFrame,
+    TimeFrameUnit as _AlpacaTimeFrameUnit,
+)
...
+from src.data.timeframe import TimeFrame, TimeFrameUnit
+from src.data.enums import DataFeed
```

Module-level config:

```diff
-TIMEFRAME = TimeFrame(1, TimeFrameUnit.Minute)
+TIMEFRAME = TimeFrame(1, TimeFrameUnit.MINUTE)
 DATA_FEED = DataFeed.IEX
```

Boundary translator (necessary because `TimeFrame` is a *struct*, not
an enum — Alpaca's `StockBarsRequest` pydantic validator rejects
strings for the `timeframe` field; verified):

```python
_ALPACA_TFU = {
    TimeFrameUnit.MINUTE: _AlpacaTimeFrameUnit.Minute,
    TimeFrameUnit.HOUR: _AlpacaTimeFrameUnit.Hour,
    TimeFrameUnit.DAY: _AlpacaTimeFrameUnit.Day,
    TimeFrameUnit.WEEK: _AlpacaTimeFrameUnit.Week,
    TimeFrameUnit.MONTH: _AlpacaTimeFrameUnit.Month,
}

def _to_alpaca_timeframe(tf: TimeFrame) -> _AlpacaTimeFrame:
    return _AlpacaTimeFrame(tf.amount, _ALPACA_TFU[tf.unit])
```

Call site:

```diff
 request = StockBarsRequest(
     symbol_or_symbols=ticker,
-    timeframe=TIMEFRAME,
+    timeframe=_to_alpaca_timeframe(TIMEFRAME),
     start=start_date,
     end=end_date,
     feed=DATA_FEED,
 )
```

`DATA_FEED` (a `str, Enum`) passes straight through — Alpaca's
pydantic coerces it.

The remaining `from alpaca.data.timeframe import …` line is the
aliased boundary translator; the brief's grep gate
(`from alpaca.trading.enums`) is satisfied.

## Verification — verbatim output

### AST syntax gate

```
$ python -c "import ast; ast.parse(open('src/execution/factory_orchestrator.py').read()); print('factory_orchestrator.py: OK')"
factory_orchestrator.py: OK
$ python -c "import ast; ast.parse(open('src/core/retrainer.py').read()); print('retrainer.py: OK')"
retrainer.py: OK
$ python -c "import ast; ast.parse(open('src/data/discovery.py').read()); print('discovery.py: OK')"
discovery.py: OK
$ python -c "import ast; ast.parse(open('src/data/enums.py').read()); print('data/enums.py: OK')"
data/enums.py: OK
$ python -c "import ast; ast.parse(open('src/execution/enums.py').read()); print('execution/enums.py: OK')"
execution/enums.py: OK
```

### Brief's grep gate

```
$ grep -r "from alpaca.trading.enums" \
    src/execution/factory_orchestrator.py \
    src/core/retrainer.py \
    src/data/discovery.py || echo "Clean"
Clean
```

### End-to-end import + Alpaca pydantic coercion

```
Unified abstractions: OrderSide.BUY TimeInForce.GTC AssetClass.US_EQUITY AssetStatus.ACTIVE DataFeed.IEX 1Minute
factory_orchestrator: OK
discovery: OK
MarketOrderRequest with unified enums OK -> OrderSide.BUY TimeInForce.GTC
GetAssetsRequest with unified enums OK -> AssetClass.US_EQUITY AssetStatus.ACTIVE
```

### Runner script import compatibility

```
$ pipenv run python -c "<runner module-level imports>"
All runner-side imports resolve: OK
```

The Maiden Voyage runner (`scripts/run_paper_live.py`) is unchanged
and still imports cleanly post-refactor.

## Scope rules — adherence

| Rule | Status |
|------|--------|
| Modify exactly the 3 listed files | ✅ Plus 2 abstraction modules: `src/execution/enums.py` (no net change in final commit) and `src/data/enums.py` (new) — necessary to host the unified enums the brief mandates |
| Use `src.execution.enums` / `src.data.timeframe` (or equivalents) | ✅ Used `src/execution/enums.py`, `src/data/timeframe.py`, and the equivalent-location `src/data/enums.py` per "or equivalents" clause |
| Update code references | ✅ |
| Append report to `llm_reports/TIER_2_DECOUPLING_REPORT_*.md` | ✅ This file (`git add -f`) |
| Commit on success | ✅ See hash below |
| **Not** modify `live_orchestrator.py` | ✅ Untouched |
| **Not** modify `src/strategies/` | ✅ Untouched |
| **Not** change business logic, math, risk | ✅ Only imports + boundary translation |

## Notes for downstream Tiers

- **Pre-existing path-convention split.** `retrainer.py` uses
  `from src.X` (project-root convention) while every file under
  `src/execution/` and `src/data/feed.py` uses `from data.X` /
  `from execution.X` (src-on-path convention). This split predates
  Tier 2 and is the underlying reason `from src.execution.enums`
  triggered an eager-load failure. A future tier should pick one
  convention and migrate.
- **Boundary translation pattern.** The `_to_alpaca_timeframe` helper
  in `retrainer.py` is the canonical pattern for translating unified
  *struct* abstractions to broker-specific structs. Future broker
  adapters should follow it; future enums (which coerce via string)
  do not need a translator.
- **Discovery + retrainer remain Alpaca-coupled** at the `TradingClient`
  / `StockHistoricalDataClient` / `StockBarsRequest` /
  `StockSnapshotRequest` / `GetAssetsRequest` layer. Tier 3+ should
  abstract these classes behind broker-agnostic provider interfaces
  (the pattern established by `MarketDataProvider` in `src/data/`).

## Commit

See `git log -1 --oneline` after this commit lands for the new HEAD
hash.
