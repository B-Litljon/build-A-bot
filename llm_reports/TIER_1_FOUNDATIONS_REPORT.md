# Tier 1 Foundations — Broker-Agnostic Abstraction Skeleton

**Date:** 2026-04-26  
**Agent:** Kimi K2.6 (OpenCode)  
**Branch:** `main`  
**Starting commit:** `5a99d95ea663971aca5f1919f36375b8ff14f383`  
**Final commit:** `71a56f2cdfc32285e52c1075dda926f500874141`

---

## 1. Mission Summary

Tier 1 establishes the foundational abstractions for the Build-A-Bot Factory SDK's broker-agnostic layer. This work creates four new files that define contracts and canonical types, but **nothing in the existing codebase calls through them yet**. That is intentional — Tiers 2 (adapter migration) and 3 (monolith decoupling) are deferred to future sessions.

The goal of this tier is purely structural: put the right shapes on disk so that subsequent agents have a north star. We are not modifying existing call sites, not writing concrete broker adapters, and not touching `live_orchestrator.py`. The abstractions are dormant but ready.

---

## 2. Files Created

### 2.1 `src/strategies/base.py`

| Attribute | Value |
|-----------|-------|
| **Path** | `src/strategies/base.py` |
| **Line count** | 72 |
| **Purpose** | Abstract base class for all trading strategies. Normalizes the strategy output contract via a `Signal` dataclass. |
| **Public surface** | `BaseStrategy(ABC)`, `Signal` dataclass |
| **What is NOT in this file** | No Alpaca-specific types. No `OrderParams`. No `get_order_params()` method. No execution logic. |

**Key members:**
- `Signal` — dataclass with fields: `direction: str`, `entry_price: float`, `raw_sl_distance: float`, `raw_tp_distance: float`, `metadata: Optional[Dict[str, Any]] = None`
- `BaseStrategy(ABC)` — abstract base with `generate_signals(self, df: pl.DataFrame) -> Signal` and `validate_input(self, df: pl.DataFrame) -> None`

### 2.2 `src/execution/enums.py`

| Attribute | Value |
|-----------|-------|
| **Path** | `src/execution/enums.py` |
| **Line count** | 32 |
| **Purpose** | Broker-agnostic canonical order enums. These are the SDK's internal representation of order side, duration, and type. |
| **Public surface** | `OrderSide(str, Enum)`, `TimeInForce(str, Enum)`, `OrderType(str, Enum)` |
| **What is NOT in this file** | No broker SDK imports. No order-construction logic. No `MarketOrderRequest` or equivalent. |

**Key members:**
- `OrderSide.BUY = "buy"`, `OrderSide.SELL = "sell"`
- `TimeInForce.DAY = "day"`, `TimeInForce.GTC = "gtc"`, `TimeInForce.IOC = "ioc"`, `TimeInForce.FOK = "fok"`
- `OrderType.MARKET = "market"`, `OrderType.LIMIT = "limit"`, `OrderType.STOP = "stop"`, `OrderType.STOP_LIMIT = "stop_limit"`

### 2.3 `src/data/timeframe.py`

| Attribute | Value |
|-----------|-------|
| **Path** | `src/data/timeframe.py` |
| **Line count** | 47 |
| **Purpose** | Replaces direct usage of `alpaca.data.timeframe.TimeFrame` with a vendor-neutral frozen dataclass. |
| **Public surface** | `TimeFrameUnit(str, Enum)`, `TimeFrame` dataclass(frozen=True), convenience constants |
| **What is NOT in this file** | No Alpaca imports. No historical data fetching. No bar construction. |

**Key members:**
- `TimeFrameUnit.MINUTE = "minute"`, `HOUR = "hour"`, `DAY = "day"`, `WEEK = "week"`, `MONTH = "month"`
- `TimeFrame(amount: int, unit: TimeFrameUnit)` — frozen dataclass with `__post_init__` validation (`amount > 0`) and `__str__` (e.g. `"5Minute"`, `"1Day"`)
- Convenience constants: `MIN_1`, `MIN_5`, `HOUR_1`, `DAY_1`

### 2.4 `src/data/market_provider.py`

| Attribute | Value |
|-----------|-------|
| **Path** | `src/data/market_provider.py` |
| **Line count** | 73 |
| **Purpose** | Abstract base class for historical and snapshot market data providers. Defines the contract that `src/data/factory.py` already imports but did not yet have on disk. |
| **Public surface** | `MarketDataProvider(abc.ABC)` with three abstract methods |
| **What is NOT in this file** | No streaming/live-feed methods (those belong in `feed.py`). No Alpaca imports. No concrete implementation. |

**Key members:**
- `get_historical_bars(symbols, timeframe, start, end) -> pl.DataFrame` — schema contract: `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`; timestamps must be timezone-aware UTC
- `get_latest_bar(symbol, timeframe) -> Optional[pl.DataFrame]` — single-row or None
- `is_market_open(symbol) -> bool` — asset-class-aware open/close logic

---

## 3. Cherry-Pick Details

| Attribute | Value |
|-----------|-------|
| **Source branch** | `sdk-decoupling` |
| **Source commit** | `1fad28091fa77abc46b6cfb99c2354e39c76b2e2` |
| **File** | `src/strategies/base.py` |
| **Method** | `git checkout sdk-decoupling -- src/strategies/base.py` |
| **Modified before commit?** | No — landed verbatim from the branch. |

**Verification:** The cherry-picked file contains `BaseStrategy` ABC, `Signal` dataclass, and zero Alpaca imports. All checks passed.

---

## 4. Verification Results (verbatim)

### 4.1 File existence

```text
$ ls -la src/strategies/base.py src/execution/enums.py src/data/timeframe.py src/data/market_provider.py
-rw-r--r--. 1 tha_magick_man tha_magick_man 1978 Apr 26 14:52 src/strategies/base.py
-rw-r--r--. 1 tha_magick_man tha_magick_man  683 Apr 26 14:53 src/execution/enums.py
-rw-r--r--. 1 tha_magick_man tha_magick_man 1023 Apr 26 14:53 src/data/timeframe.py
-rw-r--r--. 1 tha_magick_man tha_magick_man 1945 Apr 26 14:53 src/data/market_provider.py
```

### 4.2 Zero Alpaca imports

```text
$ grep -n "alpaca" src/strategies/base.py src/execution/enums.py src/data/timeframe.py src/data/market_provider.py
src/data/timeframe.py:3:Replaces direct usage of ``alpaca.data.timeframe.TimeFrame``. Adapters
src/data/market_provider.py:4:``alpaca_provider.py``). Do not add streaming / live-feed methods here —
```

**Note:** The only matches are in docstring references (`alpaca.data.timeframe.TimeFrame` and `alpaca_provider.py`), not import statements. No `from alpaca` or `import alpaca` lines exist in any new file.

### 4.3 Python syntax checks

```text
$ python -c "import ast; ast.parse(open('src/strategies/base.py').read())"
base.py: OK

$ python -c "import ast; ast.parse(open('src/execution/enums.py').read())"
enums.py: OK

$ python -c "import ast; ast.parse(open('src/data/timeframe.py').read())"
timeframe.py: OK

$ python -c "import ast; ast.parse(open('src/data/market_provider.py').read())"
market_provider.py: OK
```

### 4.4 No existing files modified

```text
$ git status --short
A  src/strategies/base.py
?? src/data/market_provider.py
?? src/data/timeframe.py
?? src/execution/enums.py
```

**Result:** Only new files appear. Zero `M` (modified) lines. No existing file was touched.

---

## 5. What This Enables

1. **Strategy authors can now extend `BaseStrategy`** without importing Alpaca types. The `Signal` dataclass provides a normalized output contract (`direction`, `entry_price`, `raw_sl_distance`, `raw_tp_distance`).
2. **Future broker adapters have a clear contract** via `MarketDataProvider`. Any concrete implementation (Alpaca, Polygon, Yahoo, etc.) must satisfy the `get_historical_bars`, `get_latest_bar`, and `is_market_open` methods and return Polars DataFrames with the specified columns.
3. **Order enums exist as canonical SDK types**. Tier 2 can now begin swapping `alpaca.trading.enums.OrderSide` → `src/execution.enums.OrderSide` in `factory_orchestrator.py` and other call sites.
4. **`src/data/factory.py` now has a satisfiable import**. The file already imports `from data.market_provider import MarketDataProvider`; the target now exists on disk.

---

## 6. What Is Explicitly NOT Done (Deferred Work)

| Item | Why deferred |
|------|--------------|
| `src/data/factory.py` import verification | The import target (`MarketDataProvider`) now exists, but `factory.py` itself was not modified or tested. Runtime resolution was not exercised. |
| `core/order_management.py` | Still missing on disk. This is a separate architectural decision: whether to rewrite it clean, restore from `sdk-decoupling`, or delete V1 strategy dependents. |
| Call-site migration | `factory_orchestrator.py` still imports `alpaca.trading.enums.OrderSide` directly. No existing file was modified. |
| `feed.py` ABC promotion | `MarketDataFeed` remains inside `feed.py`. A future `src/data/live_feed.py` module was not created. |
| `live_orchestrator.py` | Intentionally untouched. This is the 2,300-line Alpaca monolith; decoupling it is Tier 3 scope. |
| V1 concrete strategies | `rsi_bbands.py` and `sma_crossover.py` still depend on the missing `OrderParams` from `core.order_management`. |
| Concrete broker adapters | No `alpaca_adapter.py`, `polygon_adapter.py`, etc. were created. Tier 2 work. |
| Test coverage | No new tests were written for the abstractions. |

---

## 7. Known Risks / Things the Next Session Should Check

1. **`src/data/factory.py` runtime resolution** — Does the existing `from data.market_provider import MarketDataProvider` actually resolve in the current runtime environment? The import is syntactically valid but the call site in `get_market_provider()` was not exercised.

2. **`BaseStrategy` contract compatibility** — `BaseStrategy.generate_signals()` returns a `Signal` dataclass (direction, entry_price, raw_sl_distance, raw_tp_distance). The current `main` branch's `factory_orchestrator.py` expects `strategy.analyze()` to return `(List[Signal], float)` where `Signal` is from `core.signal` (a different shape: `symbol`, `type`, `price`, `confidence`, `timestamp`). **These are incompatible contracts.** A migration strategy needs to be decided before `BaseStrategy` is adopted by existing orchestrators.

3. **Enum string case** — The canonical values are lowercase (`"buy"`, `"gtc"`, `"market"`). Some brokers may expect uppercase (`"BUY"`, `"GTC"`). Tier 2 adapters will need to translate; the lowercase choice is a stylistic call that should be validated against the target broker SDKs.

4. **`TimeFrame.__str__` format** — Returns `"5Minute"` / `"1Day"`. Alpaca's `TimeFrame` uses `"5Min"` / `"1Day"`. Adapters will need to map this. Consider whether the SDK should standardize on a broker-neutral string representation or leave that to adapters.

5. **`is_market_open` default behavior** — The docstring says "crypto is conventionally always open," but there is no default implementation. Every concrete provider must implement this. Consider whether a mixin or default should be provided.

---

## 8. Suggested Next Steps for the Next Session

1. **Decide the `core/order_management.py` question** — Rewrite clean, cherry-pick from `sdk-decoupling`, or delete V1 dependents (`rsi_bbands.py`, `sma_crossover.py`). This blocks any strategy-layer work.
2. **Begin Tier 2: migrate `factory_orchestrator.py`** — Swap `alpaca.trading.enums.OrderSide` → `src/execution.enums.OrderSide`, `alpaca.trading.enums.TimeInForce` → `src/execution.enums.TimeInForce`, and introduce a thin `BrokerAdapter` interface for `MarketOrderRequest` construction.
3. **Promote `MarketDataFeed` ABC** — Move the abstract class out of `feed.py` into a dedicated `src/data/live_feed.py` module.
4. **Make `src/data/alpaca_provider.py` formally implement `MarketDataProvider`** — Add the `MarketDataProvider` base to the class declaration and verify method signatures align.
5. **Resolve `Signal` contract mismatch** — Decide whether to unify `core.signal.Signal` with `strategies.base.Signal`, deprecate one, or introduce a mapping layer.

---

## 9. Files for the Operator to Review

- [ ] **`src/strategies/base.py`** — Is the `Signal` shape (direction, entry_price, raw_sl_distance, raw_tp_distance) what we want long-term? Does it need `symbol`?
- [ ] **`src/data/timeframe.py`** — Are the enum values (`MINUTE`, `HOUR`, `DAY`, `WEEK`, `MONTH`) and `__str__` format (`"5Minute"`) correct?
- [ ] **`src/data/market_provider.py`** — Is the abstract contract complete? Are we missing methods (e.g., `get_quotes`, `get_trades`)?
- [ ] **`src/execution/enums.py`** — Should values be uppercase (`"BUY"`) or lowercase (`"buy"`) as the canonical form?

---

## Order Management Cleanup, Act 2 — Strategy Migration to BaseStrategy

**Date:** 2026-04-29
**Time:** 23:32:12 PDT
**Agent:** Claude Sonnet 4.6
**Trigger:** Pre-Act-2 inspection (MLSTRATEGY_INTERPRETATION_2026-04-29_1332.md) revealed that get_order_params() was dead code and the existing analyze() flow needed to migrate to BaseStrategy.generate_signals() with embedded bracket distances. This commit performs that migration on the factory path only; live_orchestrator.py is left with a known-broken state pending Tier 3 rewrite.

**Files created:**
- `src/core/order_management.py` — minimal OrderParams dataclass for backtest scenarios only

**Files modified:**
- `src/strategies/concrete_strategies/ml_strategy.py` — migrated to BaseStrategy; analyze() → generate_signals(); ATR bracket distances now computed in strategy; removed dead OrderParams / get_order_params code
- `src/strategies/concrete_strategies/ml_factory_strategy.py` — removed unused core.signal import, removed redundant self.pipeline override, removed analyze() passthrough method
- `src/execution/factory_orchestrator.py` — consumes new Signal shape; ATR fallback bug fixed; symbol added as DataFrame literal before strategy invocation
- `grid_search_backtest.py` — constructor args updated (angel_path/devil_path/angel_threshold/devil_threshold); adapted to single-signal generate_signals() return; signal field access updated for base.Signal
- `grid_search_backtest_q1.py` — same changes as grid_search_backtest.py

**Files deleted:**
- `src/strategies/strategy.py` — legacy Strategy ABC, no surviving inheritors
- `src/strategies/strategy_factory.py` — empty registry, no consumers
- `tests/test_strategy_logic.py` — placeholder stub, no real tests
- `tests/test_live_simulation.py` — placeholder stub, no real tests

**Files NOT modified (deliberately):**
- `live_orchestrator.py` logic — known broken (angel_model attribute bug); deferred to Tier 3 rewrite as its own project. No import changes were needed because it only imports MLStrategy (same module path) and does not import the deleted Strategy ABC.
- `factory_orchestrator.py` Alpaca enum imports — separate Tier 2 broker abstraction task
- All Tier 1 foundation files (base.py, enums.py, timeframe.py, market_provider.py)

### ATR sourcing
- `SL_ATR_MULTIPLIER = 0.5` — sourced from live_orchestrator.py line 186
- `TP_ATR_MULTIPLIER = 3.0` — sourced from live_orchestrator.py line 187
- `MIN_SL_PCT = 0.0015` — sourced from live_orchestrator.py line 188 (HF7 hotfix floor)
- `natr_14` (percentage-form ATR) is computed by `V3BaseFeatures` as part of the 18-feature vector. Absolute ATR is derived in-strategy as `(natr_14 / 100.0) * current_price`.

### Symbol handoff design
**Option A** — callers add a `symbol` column to the DataFrame as a Polars literal before invoking `generate_signals()`. This keeps `BaseStrategy.generate_signals(df: pl.DataFrame)` signature unchanged and avoids modifying base.py (which is locked under Tier 1). The strategy reads `df["symbol"].tail(1)[0]` and passes it through `metadata["symbol"]`.

Callers updated:
- `factory_orchestrator.py` — `history = history.with_columns(pl.lit(symbol).alias("symbol"))`
- `grid_search_backtest.py` / `grid_search_backtest_q1.py` — `hist = hist.with_columns(pl.lit(symbol).alias("symbol"))`

### Decisions made on the open questions
- `strategy_factory.py`: **deleted (Option B)** — empty `STRATEGY_REGISTRY = {}` with zero consumers across the codebase. Cleaner to delete than maintain dead infrastructure.
- Placeholder tests: **deleted (Option B)** — real ML strategy tests need model files or mocks; out of scope for Act 2. The stubs were not testing anything meaningful.
- `MLFactoryStrategy.pipeline` override: **removed** — it was a byte-for-byte duplicate of the pipeline already constructed in `MLStrategy.__init__`. Removing it eliminates confusion and defensive-code noise.

### Verification results
```
# Syntax checks
src/core/order_management.py: syntax OK
src/strategies/concrete_strategies/ml_strategy.py: syntax OK
src/strategies/concrete_strategies/ml_factory_strategy.py: syntax OK
src/execution/factory_orchestrator.py: syntax OK
src/execution/live_orchestrator.py: syntax OK
grid_search_backtest.py: syntax OK
grid_search_backtest_q1.py: syntax OK

# Structural AST checks (numpy/polars not installed in shell env)
MLStrategy bases: ['BaseStrategy']
MLStrategy is BaseStrategy subclass: OK
MLFactoryStrategy bases: ['MLStrategy']
MLFactoryStrategy is MLStrategy subclass: OK
MLStrategy has generate_signals: OK
MLStrategy lacks analyze: OK
MLStrategy lacks get_order_params: OK
All structural checks passed.

# OrderParams import
cd src && python -c "from core.order_management import OrderParams; print('OrderParams import OK')"
→ OrderParams import OK

# No surviving legacy Strategy imports
No legacy Strategy imports found

# No surviving deleted method references
No deleted method references found

# live_orchestrator parse check
live_orchestrator parses OK
```

### Known unresolved issues (flagged for future work)
- `live_orchestrator.py` still has the `angel_model` / `devil_model` attribute bug (would crash on first inference). Tier 3 rewrite required.
- Other live_orchestrator.py logic still references `Strategy` ABC patterns and constructs `core.signal.Signal` directly; full rewrite needed.
- `MLStrategy.generate_signals` only emits long signals (no short logic, matching the original code). Future enhancement.
- `OrderParams` percentage-multiplier risk model is not unified with ATR-based live execution. Backtests use one, live uses the other. Reconciling them is a future architectural decision.
- Grid-search backtests use a single model file for both Angel and Devil (`angel_path=devil_path="src/ml/models/rf_model.joblib"`) because the stale code only specified one `model_path`. This is a semantic change; if real backtests are run, separate Angel/Devil model files should be provided.

### Final commit hash
See `git log` (cannot self-reference).

---

**End of Report**

---

## Tier 1 Completion — Alpaca Streaming Port + Unified ABC

**Date:** 2026-04-29
**Agent:** Claude Sonnet 4.6
**Trigger:** The previous ABC (`market_provider.py`) defined three abstract methods — `get_historical_bars(symbols: List[str], timeframe: TimeFrame, ...)`, `get_latest_bar`, and `is_market_open` — that none of the three concrete providers implemented. Polygon and Yahoo implemented a completely different 4-method shape; AlpacaProvider had no streaming at all and did not inherit from the ABC. All three concrete classes were failing instantiation with `TypeError: Can't instantiate abstract class`. The streaming logic for Alpaca existed in `feed.py`'s `AlpacaCryptoFeed` in async form but had never been ported to `AlpacaProvider`.

**Architectural decision:** Port AlpacaCryptoFeed's async streaming logic into AlpacaProvider with a sync interface (asyncio.run-bridged), add equity streaming via StockDataStream alongside crypto, add get_active_symbols via TradingClient, then rewrite the unified ABC against the three matching concrete implementations.

**Files modified:**
- `src/data/alpaca_provider.py` — added inheritance from MarketDataProvider; added `get_active_symbols`, `subscribe`, `run_stream`; extended `__init__` to store api_key/secret_key, instantiate TradingClient, and initialize streaming state
- `src/data/market_provider.py` — rewritten entirely: replaced the mismatched 3-method ABC (with TimeFrame dependency, get_latest_bar, is_market_open) with the 4-method unified contract matching what Polygon and Yahoo already implement

**Files NOT modified:**
- `feed.py` — AlpacaCryptoFeed is now functionally redundant but left in place; deletion is a separate decision (Tier 2)
- `polygon_provider.py`, `yahoo_provider.py` — source of truth for the contract shape; untouched
- `factory.py`, `live_orchestrator.py`, `factory_orchestrator.py`, `strategies/base.py`, `execution/enums.py`, `data/timeframe.py` — all untouched

### Contract resolution

Polygon and Yahoo agree exactly on all four method signatures. No conflicts or shape mismatches were found:

| Method | Signature |
|--------|-----------|
| `get_active_symbols` | `(self, limit: int = 10) -> List[str]` |
| `get_historical_bars` | `(self, symbol: str, timeframe_minutes: int, start: datetime, end: datetime) -> pl.DataFrame` |
| `subscribe` | `(self, symbols: List[str], callback: Callable) -> None` |
| `run_stream` | `(self) -> None` |

The old ABC had `get_historical_bars(symbols: List[str], timeframe: TimeFrame, start, end=None)` — plural symbols, TimeFrame type, optional end. The concrete classes used singular symbol, int minutes, required end. The old ABC also declared `get_latest_bar` and `is_market_open` which no concrete class implemented. The new ABC drops all three of those deviations and matches the concrete implementations exactly.

### Equity vs. crypto streaming

`AlpacaProvider.subscribe()` sniffs symbols by `/` and routes to `CryptoDataStream` or `StockDataStream` accordingly. Both streams are initialized during `subscribe()` but not started — `run_stream()` calls `asyncio.run(_run_all())` where `_run_all` uses `asyncio.gather` to run both streams concurrently. This makes AlpacaProvider the only provider that natively handles both asset classes in one instance.

The `_bar_handler` is defined as an inner `async def` inside `subscribe()` and registered with Alpaca's `subscribe_bars()`. It emits a provider-agnostic dict `{symbol, timestamp, open, high, low, close, volume}` and calls `await self._callback(...)`. This matches the pattern in `feed.py`'s `AlpacaCryptoFeed.subscribe`.

### Verification results

```
AlpacaProvider: OK
PolygonDataProvider: OK
YahooDataProvider: OK

PASS: all three providers satisfy MarketDataProvider.
```

Note: verification was performed using `unittest.mock` to stub all vendor SDKs (`polars`, `alpaca`, `polygon`, `yfinance`) since the project's virtualenv packages are not installed in the CI/shell environment. All three classes instantiated without `TypeError`; the ABC compliance check is a Python runtime structural check independent of package availability.

Syntax checks (AST parse):
```
market_provider.py: OK
alpaca_provider.py: OK
```

### Still deferred (do not let this rot)

- `feed.py`'s `MarketDataFeed` ABC and `AlpacaCryptoFeed` class are now redundant. Deletion is a deliberate Tier 2 task — leave for now.
- `core/order_management.py` is still missing on disk.
- `factory.py` still uses the older provider interface; verify it still works after this change.
- No call sites migrated yet (factory_orchestrator, retrainer, etc. still import Alpaca enums directly).
- The contract uses `int` minutes for timeframe; `data/timeframe.TimeFrame` exists but is not yet wired in. Tier 2 decision.
- `data/timeframe.py` is now orphaned from the ABC — nothing imports it in the provider layer. Tier 2 decision on whether to wire it in or remove it.

### Final commit hash

See `git log` (cannot self-reference).

---

## Tier 1 Housekeeping

**Date:** 2026-04-29
**Time:** 2026-04-29 12:36:21 PDT
**Agent:** Kimi K2.6
**Trigger:** Loose ends from yesterday's Tier 1 completion — orphaned report file, unidentified Modelfile, and untested factory.py post-ABC-rewrite.

**Files modified:**
- `STOP_REPORT.md` → moved to `llm_reports/STOP_REPORT_2026-04-27.md`

**Files NOT modified:**
- `Modelfile.gemma` (deleted by operator before inspection; see investigation below)
- All `.py` source files

### STOP_REPORT.md investigation
The file documented a contract inconsistency discovered by Gemini 3.1 during an attempted fix on 2026-04-27: `AlpacaProvider` only implemented `get_historical_bars` while `PolygonDataProvider` and `YahooDataProvider` implemented four methods (`get_active_symbols`, `get_historical_bars`, `subscribe`, `run_stream`). The report was a halt-and-report because the three concrete providers did not implement the same contract at that time. It is being archived in `llm_reports/` as part of the audit trail because the inconsistency was subsequently resolved in commit `8e6d26d` (Tier 1 completion) by porting streaming logic into `AlpacaProvider` and rewriting the unified `MarketDataProvider` ABC.

### Modelfile.gemma investigation
- **File size:** N/A (deleted by operator before agent inspection)
- **File type (per `file` command):** N/A
- **Contents:** N/A
- **Assessment:** The file was deleted by the operator before this agent could inspect it. No data available for Captain B's decision.

### factory.py sanity check
```
factory.py: syntax OK
factory.py: import OK
  alpaca: returned AlpacaProvider
  polygon: returned PolygonDataProvider
  yahoo: returned YahooDataProvider
```

**Result:** No ABC violations. All three provider types instantiate correctly through `get_market_provider()`.

### Final commit hash

See `git log` (cannot self-reference).

---

## Order Management Cleanup, Act 1 — V1 Deletes (STOPPED)

**Date:** 2026-04-29
**Time:** 2026-04-29 12:47:22 PDT
**Agent:** Kimi K2.6
**Trigger:** Hybrid order-management decision (rewrite OrderParams clean + delete V1 dependents). This is the delete half; Act 2 (Sonnet) will write the new OrderParams.

**Status:** STOPPED — pre-deletion audit discovered unexpected dependents and an unresolved ABC inheritance chain. No files were deleted. No commit made.

**Files slated for deletion (none deleted yet):**
- `src/strategies/concrete_strategies/rsi_bbands.py` — V1 strategy, deprecated
- `src/strategies/concrete_strategies/sma_crossover.py` — V1 strategy, deprecated
- `src/strategies/strategy.py` — legacy Strategy ABC, superseded by BaseStrategy in base.py
- `tests/test_order_management.py` — tested a module that no longer exists

**Files NOT modified:**
- All source files untouched

### Pre-deletion dependency audit

```
=== rsi_bbands ===
./src/strategies/concrete_strategies/rsi_bbands.py:11:class RSIBBands(Strategy):
./src/strategies/concrete_strategies/__init__.py:1:from .rsi_bbands import RSIBBands
./src/strategies/concrete_strategies/__init__.py:6:    "rsi_bollinger": RSIBBands,
./src/strategies/strategy_factory.py:1:from strategies.concrete_strategies.rsi_bbands import RSIBBands
./src/strategies/strategy_factory.py:6:    "rsi_bollinger": RSIBBands,
./tests/test_live_simulation.py:17:from strategies.concrete_strategies.rsi_bbands import RSIBBands
./tests/test_live_simulation.py:110:    # Initialize RSIBBands with LOOSE parameters for testing
./tests/test_live_simulation.py:111:    strategy = RSIBBands(
./tests/test_strategy_logic.py:10:from strategies.concrete_strategies.rsi_bbands import RSIBBands
./tests/test_strategy_logic.py:110:def replay_and_collect(strategy: RSIBBands, symbol: str, df: pl.DataFrame):
./tests/test_strategy_logic.py:131:    strategy_positive = RSIBBands()
./tests/test_strategy_logic.py:144:    strategy_negative = RSIBBands()

=== sma_crossover ===
./src/strategies/concrete_strategies/sma_crossover.py:12:class SMACrossover(Strategy):
./src/strategies/concrete_strategies/__init__.py:2:from .sma_crossover import SMACrossover
./src/strategies/concrete_strategies/__init__.py:7:    "sma_crossover": SMACrossover,
./src/strategies/strategy_factory.py:2:from strategies.concrete_strategies.sma_crossover import SMACrossover
./src/strategies/strategy_factory.py:7:    "sma_crossover": SMACrossover,

=== legacy Strategy ABC ===
./src/strategies/concrete_strategies/rsi_bbands.py:3:from strategies.strategy import Strategy
./src/strategies/concrete_strategies/sma_crossover.py:9:from strategies.strategy import Strategy
./src/strategies/concrete_strategies/ml_strategy.py:33:from strategies.strategy import Strategy

=== test_order_management ===
(no output — file is truly orphaned)
```

### Strategy ABC inheritance check

```
=== ml_strategy.py inheritance ===
43:class MLStrategy(Strategy):
33:from strategies.strategy import Strategy
```

**Conclusion:** `ml_strategy.py` inherits from the OLD `Strategy` ABC (`strategies.strategy.Strategy`), NOT from `BaseStrategy` (`strategies.base.BaseStrategy`). Deleting `strategy.py` would break `ml_strategy.py` and, by extension, `ml_factory_strategy.py` (which inherits from `MLStrategy`).

### Surviving consumers of `core.order_management`

Files still importing from the missing `core.order_management` module:
- `src/strategies/concrete_strategies/ml_strategy.py` (`OrderParams`)
- `src/strategies/strategy.py` (`OrderParams`) — this file is also slated for deletion
- `grid_search_backtest.py` (`OrderParams`)
- `grid_search_backtest_q1.py` (`OrderParams`)
- `tests/test_order_management.py` (`OrderManager`, `OrderParams`) — slated for deletion

### Unexpected dependents discovered

| File | Imports | Impact |
|------|---------|--------|
| `src/strategies/concrete_strategies/__init__.py` | `RSIBBands`, `SMACrossover` | Will break if V1 strategies deleted without updating `__init__.py` |
| `src/strategies/strategy_factory.py` | `RSIBBands`, `SMACrossover` | Will break if V1 strategies deleted without updating `strategy_factory.py` |
| `grid_search_backtest.py` | `OrderParams` from `core.order_management` | Still depends on missing module; not in scope but worth flagging |
| `grid_search_backtest_q1.py` | `OrderParams` from `core.order_management` | Still depends on missing module; not in scope but worth flagging |

### Test files flagged for follow-up

- `tests/test_live_simulation.py` — imports `RSIBBands`
- `tests/test_strategy_logic.py` — imports `RSIBBands`

### Required decisions before Act 1 can proceed

1. **Migrate `ml_strategy.py` to `BaseStrategy` first?** The old `Strategy` ABC cannot be deleted until `ml_strategy.py` stops inheriting from it. Should Act 2 (Sonnet) handle this migration as part of OrderParams design, or should it be a separate pre-Act-1 task?
2. **Update `__init__.py` and `strategy_factory.py`?** These are source files that import the V1 strategies. Should they be updated in Act 1 to remove V1 references, or are they also slated for deletion?
3. **Handle `grid_search_backtest.py` and `grid_search_backtest_q1.py`?** These files import `OrderParams` from the missing module. Are they also dead code to be deleted, or do they need to be preserved and fixed?

### Final commit hash

N/A — no commit made (Act 1 stopped at audit phase).

---

## Order Management Cleanup, Act 1 (Revised) — V1 Deletion + Registry/Test Cleanup

**Date:** 2026-04-29
**Time:** 2026-04-29 13:12:35 PDT
**Agent:** Kimi K2.6
**Trigger:** Revised Act 1 after the previous attempt halted on unexpected dependents. Recon (`GRID_SEARCH_RECON_2026-04-29_1255.md`) classified grid-search backtests as stale-but-live ML tooling — not deleted here. Legacy `strategy.py` ABC also retained because `MLStrategy` still inherits from it (handled in Act 2).

**Files edited:**
- `src/strategies/concrete_strategies/__init__.py` — removed V1 registry entries (`RSIBBands`, `SMACrossover`)
- `src/strategies/strategy_factory.py` — removed V1 imports and registry entries, left empty `STRATEGY_REGISTRY` and `create_strategy` function
- `tests/test_strategy_logic.py` — removed entire V1-specific test body (RSIBBands signal/replay tests); replaced with placeholder docstring
- `tests/test_live_simulation.py` — removed entire V1-specific test body (RSIBBands live simulation with mocks); replaced with placeholder docstring

**Files deleted:**
- `src/strategies/concrete_strategies/rsi_bbands.py` — V1 strategy, deprecated
- `src/strategies/concrete_strategies/sma_crossover.py` — V1 strategy, deprecated
- `tests/test_order_management.py` — orphaned, tested missing module

**Files NOT modified (deliberately):**
- `src/strategies/strategy.py` — still in use by `MLStrategy`, deleted in Act 2
- `src/strategies/concrete_strategies/ml_strategy.py`, `ml_factory_strategy.py` — Act 2 targets
- `grid_search_backtest.py`, `grid_search_backtest_q1.py` — stale-but-live, fixed in Act 2

### Edits applied to each registry/test file

**`src/strategies/concrete_strategies/__init__.py`**
- Removed: `from .rsi_bbands import RSIBBands`, `from .sma_crossover import SMACrossover`
- Removed: `"rsi_bollinger": RSIBBands`, `"sma_crossover": SMACrossover` from `STRATEGIES` dict
- Retained: `from .ml_strategy import MLStrategy` and `"ml_strategy": MLStrategy`

**`src/strategies/strategy_factory.py`**
- Removed: both V1 imports (`RSIBBands`, `SMACrossover`)
- Removed: both V1 entries from `STRATEGY_REGISTRY`
- Retained: empty `STRATEGY_REGISTRY = {}` and `create_strategy` function

**`tests/test_strategy_logic.py`**
- Removed: `from strategies.concrete_strategies.rsi_bbands import RSIBBands`
- Removed: `build_synthetic_df`, `replay_and_collect`, `run_test`, and `if __name__ == "__main__"` block (all exclusively tested RSIBBands)
- Replaced with: module-level placeholder docstring

**`tests/test_live_simulation.py`**
- Removed: `from strategies.concrete_strategies.rsi_bbands import RSIBBands`
- Removed: `MockTradingClient`, `MockProvider`, `run_simulation`, and `if __name__ == "__main__"` block (all exclusively tested RSIBBands live simulation)
- Replaced with: module-level placeholder docstring

### Post-deletion verification

```
=== No remaining references to V1 strategy classes ===
(no output — clean)

=== No remaining imports of V1 strategy modules ===
(no output — clean)

=== ml_strategy.py and ml_factory_strategy.py still parse ===
ml_strategy.py: syntax OK
ml_factory_strategy.py: syntax OK

=== Surviving consumers of core.order_management (Act 2 targets) ===
./src/strategies/concrete_strategies/ml_strategy.py:30:from core.order_management import OrderParams
./src/strategies/strategy.py:6:from core.order_management import OrderParams
./grid_search_backtest_q1.py:16:from core.order_management import OrderParams
./grid_search_backtest.py:16:from core.order_management import OrderParams
```

**Result:** All checks passed. No RSIBBands/SMACrossover references remain. Both ML strategies parse. Surviving `core.order_management` consumers match the expected Act 2 input set.

### Surviving consumers of `core.order_management` (Act 2 input)

| File | Import | Notes |
|------|--------|-------|
| `src/strategies/concrete_strategies/ml_strategy.py` | `OrderParams` | Primary target — new OrderParams designed against this file's needs |
| `src/strategies/strategy.py` | `OrderParams` | Legacy ABC; will be deleted once `MLStrategy` migrates to `BaseStrategy` |
| `grid_search_backtest.py` | `OrderParams` | Stale ML backtest; needs constructor update + new OrderParams import |
| `grid_search_backtest_q1.py` | `OrderParams` | Stale ML backtest; same fixes as above |

### Test files that became stubs (need Act 2 re-targeting)

- `tests/test_strategy_logic.py` — now a placeholder; was entirely RSIBBands-specific
- `tests/test_live_simulation.py` — now a placeholder; was entirely RSIBBands-specific

### Final commit hash

See `git log` (cannot self-reference).

---

## Act 3 — Blocker Resolution, Path Alpha Final Commit

**Date:** 2026-05-02
**Time:** 15:26:47 PDT
**Agent:** Claude Sonnet 4.6
**Trigger:** Act 3 — Sever eager imports, verify Path Alpha, and commit
**Prior HEAD:** `323bf09` (feat(sdk): Act 2 — migrate MLStrategy to BaseStrategy, fix ATR fallback)

### Files Modified

| File | Change |
|------|--------|
| `src/execution/__init__.py` | Severed eager `LiveOrchestrator` import; now exports `FactoryOrchestrator` and `RiskManager` |
| `src/execution/factory_orchestrator.py` | Path Alpha delegation, per-symbol asyncio.Lock, crypto cash field, A3 None guard |
| `src/execution/risk_manager.py` | A3 chop filter (returns None), crypto buying-power fix, $50 min notional |
| `src/strategies/concrete_strategies/ml_strategy.py` | Removed bracket constants; emits raw ATR; Polars warning suppressed |

### Pre-flight Results

```
git status --short:
 M src/execution/factory_orchestrator.py
 M src/execution/risk_manager.py
 M src/strategies/concrete_strategies/ml_strategy.py

git rev-parse --abbrev-ref HEAD: main
git log -1 --oneline: 323bf09 feat(sdk): Act 2 — migrate MLStrategy to BaseStrategy, fix ATR fallback
```

All 3 expected modified files confirmed. HEAD matches. Tree not clean — proceeding.

### Task 1 — Eager Import Severed

`src/execution/__init__.py` before:
```python
from .live_orchestrator import LiveOrchestrator
__all__ = ["LiveOrchestrator"]
```

After:
```python
from .factory_orchestrator import FactoryOrchestrator
from .risk_manager import RiskManager
__all__ = ["FactoryOrchestrator", "RiskManager"]
```

`LiveOrchestrator` is no longer reachable via `from execution import ...`. `live_orchestrator.py` remains on disk, quarantined.

### Task 2 — Sanity Check Results

**A3 floor logic** — `self.profile.min_sl_pct = 0.0015` confirmed at `risk_manager.py:11`. A3 filter at line 33:
```python
if sl_dist < (entry_price * self.profile.min_sl_pct):
    return None
```
Status: PRESENT ✓

**Crypto cash logic** — diff confirms:
```python
bp_source = cash if is_crypto else buying_power
bp_qty = (bp_source * 0.95) / entry_price
```
Status: PRESENT ✓

**asyncio.Lock logic** — `defaultdict(asyncio.Lock)` in `__init__`; `async with self._locks[symbol]` wrapping both entry and exit order submission.
Status: PRESENT ✓

### Task 3 — Verification Results (verbatim)

```
$ python -c "import ast; ast.parse(open('src/execution/__init__.py').read())" && echo "Init Syntax: OK"
Init Syntax: OK

$ python -c "import sys; sys.path.append('src'); from execution import FactoryOrchestrator, RiskManager; print('Factory Imports: OK')"
Factory Imports: OK
```

### Smoke Test Gate Status

Previously blocked by: `ModuleNotFoundError: No module named 'polars'` triggered by eager `LiveOrchestrator` import in `__init__.py`.

After this act: factory path import succeeds cleanly. `polars` is still required to run `MLStrategy` inference — Captain B must run `pipenv install polars` (or `pipenv sync`) before executing a live smoke test.

### Architectural Invariants Carried Forward

- `Signal.raw_sl_distance` and `raw_tp_distance` carry **raw ATR** — `RiskManager` owns all multipliers
- Crypto symbols contain `"/"` — used to switch buying-power source to `account.cash`
- `RiskManager.calculate_bracket()` returning `None` = skip trade (A3 chop / low volatility)
- `$50` minimum notional — `calculate_quantity()` returns `0.0` below this threshold
- Per-symbol `asyncio.Lock` must wrap both entry and exit order submission
