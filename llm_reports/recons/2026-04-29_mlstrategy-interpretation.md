---
type: recon
date: 2026-04-29
time: 13:32:15 PDT
agent: Claude Sonnet 4.6
model: claude-sonnet-4-6
trigger: Pre-Act-2 inspection — recover the original design intent of MLStrategy before drafting the migration prompt
head: unknown
scope: read-only
imported_from: MLSTRATEGY_INTERPRETATION_2026-04-29_1332.md
---

# MLStrategy Architecture Interpretation

**Date:** 2026-04-29
**Time:** 2026-04-29 13:32:15 PDT
**Agent:** Claude Sonnet 4.6
**Trigger:** Pre-Act-2 inspection. Captain B doesn't remember the original design intent of MLStrategy and wants ground truth before we draft the migration prompt.

---

## 1. Legacy Strategy ABC (`strategy.py`)

The legacy ABC is minimal — 54 lines, three abstract members:

| Member | Type | Contract |
|--------|------|----------|
| `analyze(self, data)` | abstract method | "Analyzes market data and returns a list of Signals." Return type annotated as `List[Signal]` or `tuple[List[Signal], float]` (the docstring acknowledges both forms). |
| `get_order_params(self) -> OrderParams` | abstract method | "Returns the default OrderParams for this strategy." No logic — just a typed getter for a pre-built `OrderParams` instance. |
| `warmup_period` | abstract property → `int` | "Returns the minimum number of candles required before analysis can run." |

**Imports on the ABC:**
```python
from core.signal import Signal
from core.order_management import OrderParams
import talib
import pandas as pd
import numpy as np
import polars as pl
```

Three observations worth noting:
1. The ABC imports `talib`, `pandas`, `numpy`, and `polars` at the module level — none of which are used anywhere in the ABC itself. These are leaked imports from V1 strategy development that were never cleaned up.
2. `from core.order_management import OrderParams` — this import fails at runtime today because `core/order_management.py` doesn't exist on disk. Any attempt to import `strategy.py` currently raises `ModuleNotFoundError`.
3. The return type of `analyze()` is underspecified in the type annotation (`data` is untyped, return is `List[Signal] or tuple`). The docstring was written before the actual MLStrategy return shape was fixed.

**Conclusion for Task 1:** The legacy ABC defines a two-method interface: `analyze()` returns signals, `get_order_params()` returns risk configuration. These were intended to be separate concerns — one for "what the market is doing" and one for "how to size and bracket a trade." In practice, as we'll see, `get_order_params()` was never called by either orchestrator.

---

## 2. New BaseStrategy ABC (`base.py`)

A clean 72-line ABC written from scratch. Key differences:

**`Signal` dataclass (from `base.py`):**
```python
@dataclass
class Signal:
    direction: str          # 'long' or 'short'
    entry_price: float
    raw_sl_distance: float  # distance in price units, not a multiplier
    raw_tp_distance: float  # distance in price units
    metadata: Optional[Dict[str, Any]] = None
```

**`BaseStrategy(ABC)`:**
- Constructor: `__init__(self, **kwargs)` — stores `self.params = kwargs`, `self.name = self.__class__.__name__`
- One abstract method: `generate_signals(self, df: pl.DataFrame) -> Signal`
- One concrete method: `validate_input(self, df: pl.DataFrame) -> None`
- No `get_order_params()` — intentionally absent
- No `warmup_period` — not part of the contract

**Key structural differences from legacy `Strategy(ABC)`:**

| Aspect | `Strategy` (legacy) | `BaseStrategy` (new) |
|--------|---------------------|----------------------|
| Input | `data: Dict[str, pl.DataFrame]` (multi-symbol dict) | `df: pl.DataFrame` (single symbol DataFrame) |
| Output | `(List[Signal], float)` — list + probability scalar | `Signal` — single object |
| Risk params | `get_order_params() -> OrderParams` abstract | Not present at all |
| Signal type | `core.signal.Signal` (symbol, type, price, confidence, timestamp, metadata) | `base.Signal` (direction, entry_price, raw_sl_distance, raw_tp_distance, metadata) |
| Warmup | `warmup_period` abstract property | Not in contract |

The two `Signal` shapes are **completely incompatible** — different field names, different semantic model. `core.signal.Signal` is an event notification ("BUY happened at this price with this confidence"). `base.Signal` is a bracket specification ("enter long at X, with SL Y away and TP Z away").

---

## 3. MLStrategy Deep Read

### 3a. Constructor

**Signature:**
```python
def __init__(
    self,
    angel_path: str | Path = "models/angel_latest.pkl",
    devil_path: str | Path = "models/devil_latest.pkl",
    angel_threshold: float = 0.40,
    devil_threshold: float = 0.50,
    warmup_period: int = 260,
    angel_trainer=None,
    devil_trainer=None,
):
```

**Instance state:**
| Attribute | Type | Purpose |
|-----------|------|---------|
| `self.timeframe` | `int` | Hard-coded `1` (1-minute bars). Not used in logic. |
| `self.warmup` | `int` | Minimum candle count (from `warmup_period`) |
| `self.angel_threshold` | `float` | Minimum Angel probability to propose a trade (0.40) |
| `self.devil_threshold` | `float` | Minimum Devil probability to approve (0.50, overridden by `_load_threshold()`) |
| `self.angel_path`, `self.devil_path` | `Path` | Stored for hot-reload monitoring |
| `self.angel_trainer`, `self.devil_trainer` | `V3RandomForestTrainer` | Trainer objects wrapping sklearn `RandomForestClassifier`; models loaded via `trainer.load()` |
| `self.angel_mtime`, `self.devil_mtime` | `float` | OS modification timestamps for hot-reload comparison |
| `self.notification_manager` | `NotificationManager` | Discord alerting on hot-reload events |
| `self.pipeline` | `FeaturePipeline` | Feature generator: `[V3BaseFeatures(), V3HTFFeatures(timeframe="5m")]` |
| `self.feature_names` | `List[str]` | 18 named features (V3.4 set) |
| `self.order_params` | `OrderParams` | `OrderParams(risk_percentage=0.02, tp_multiplier=1.005, sl_multiplier=0.998, use_trailing_stop=False)` |

**Model loading:** Eager at construction. If model files don't exist at the given path, it tries the project root as a base. If the file still doesn't exist, `trainer.load()` will raise — there is no fallback or None sentinel. The constructor also immediately calls `_load_threshold()` to override `devil_threshold` from `models/threshold.json`.

### 3b. The `analyze()` / `get_order_params()` flow

**Main entry point:** `analyze(self, data: Dict[str, pl.DataFrame]) -> Tuple[List[Signal], float]`

**Input:** `data` is a dict of `{symbol_string: polars_DataFrame}`. The DataFrame must contain at least OHLCV columns (`timestamp`, `open`, `high`, `low`, `close`, `volume`) and have `len >= self.warmup_period`. The `FeaturePipeline` downstream will compute all 18 features.

**Output:** `(List[Signal], float)` where:
- `List[Signal]` — zero or one `core.signal.Signal` objects per symbol; each has `type=SignalType.BUY` (never SELL, never HOLD)
- `float` — `highest_angel_prob` — the maximum Angel probability seen across all symbols in this call, regardless of Devil outcome. Used by... nobody, in current call sites (the `_` discard in `factory_orchestrator.py:96` confirms it).

**Angel/Devil flow:**
1. For each symbol, call `_generate_features(df)` → `FeaturePipeline.run()` → 18-column Polars DataFrame
2. Extract `.tail(1)` row, convert to numpy
3. `angel_prob = angel_trainer.predict_proba(latest_features)[0, 1]` — probability of class 1 (BUY)
4. If `angel_prob < angel_threshold`: skip
5. Build `meta_features` = original 18 features + `angel_prob` column (19 total) as pandas DataFrame
6. `devil_prob = devil_trainer.predict_proba(meta_features)[0, 1]`
7. If `devil_prob < devil_threshold`: skip
8. Emit `core.signal.Signal(symbol, BUY, current_price, devil_prob, timestamp, {angel_prob, devil_prob})`

**Important:** `_generate_features()` calls `self.pipeline.run(df)` then drops null rows. The pipeline reference is `self.pipeline` which is set in `__init__`. However (see Section 4), `MLFactoryStrategy.__init__` **overwrites** `self.pipeline` after calling `super().__init__()`.

### 3c. Where does risk math happen?

**In `MLStrategy` itself:** None. Zero. The `analyze()` method produces a `core.signal.Signal` with `confidence=devil_prob` — a probability, not a bracket. The signal metadata contains only `{angel_prob, devil_prob}`. There is no SL price, no TP price, no ATR calculation, no position sizing in `MLStrategy.analyze()`.

The `self.order_params` attribute exists (constructed with `tp_multiplier=1.005`, `sl_multiplier=0.998`, `risk_percentage=0.02`) and `get_order_params()` returns it. But **none of the call sites ever call `get_order_params()`**. It is dead letter.

The risk math lives in two other places:
- **`FactoryOrchestrator._execute_buy()`**: calls `self.risk_manager.calculate_bracket(sig.price, atr)` and `self.risk_manager.calculate_quantity(...)`. Uses `RiskManager` (ATR-based).
- **`LiveOrchestrator._submit_entry_order()`**: inlines risk math directly using module-level constants (`SL_ATR_MULTIPLIER=0.5`, `TP_ATR_MULTIPLIER=3.0`, `ACCOUNT_RISK_PER_TRADE=0.02`). Also applies `MIN_SL_PCT=0.0015` floor. The ATR values come from the `Signal.metadata` dict populated in `_run_inference()`.

### 3d. The `OrderParams` interface

**Fields `MLStrategy` uses to construct `self.order_params`:**
```python
OrderParams(
    risk_percentage=0.02,
    tp_multiplier=1.005,   # 0.5% take profit (percentage multiplier)
    sl_multiplier=0.998,   # 0.2% stop loss (percentage multiplier)
    use_trailing_stop=False,
)
```

From the grid-search backtest `BOM.place_order()` — the only place `OrderParams` fields are actually consumed:
```python
risk = cap * self.order_params.risk_percentage       # float
qty = risk / sig.price                               # derived
sl = sig.price * self.order_params.sl_multiplier     # float (< 1.0)
tp = sig.price * self.order_params.tp_multiplier     # float (> 1.0)
```

**Reconstructed `OrderParams` dataclass shape:**
```python
@dataclass
class OrderParams:
    risk_percentage: float        # fraction of capital at risk (0.02 = 2%)
    tp_multiplier: float          # TP = entry_price * tp_multiplier (e.g. 1.005)
    sl_multiplier: float          # SL = entry_price * sl_multiplier (e.g. 0.998)
    use_trailing_stop: bool       # whether to use a trailing stop (unused in live code)
```

Note that the `sl_multiplier` and `tp_multiplier` in `OrderParams` are **percentage multipliers** (e.g. `0.998` = 0.2% below entry), whereas `RiskManager` and `LiveOrchestrator` both use ATR-based **absolute distances** (`SL_ATR_MULTIPLIER * atr_abs`). These are two entirely different risk parametrizations that cannot be trivially unified.

### 3e. Other observations

**Logging:** Very detailed at `DEBUG` level for rejections, `INFO` for approvals, `logger.critical()` for hot-reload events. This is a production monitoring pattern, not development scaffolding — someone was watching these logs.

**Error handling:** `analyze()` wraps each symbol in `try/except Exception as e` and `continue`s — any per-symbol crash is logged and skipped rather than bubbling up. The return value is always the same type regardless of exceptions.

**Hot-reload:** `_check_model_updates()` is called at the start of every `analyze()` call. It compares OS `mtime` of the model files and reloads if changed. This is a production feature — models are retrained periodically and the strategy picks up new weights without a restart.

**Persistent state across calls:**
- `self.angel_mtime`, `self.devil_mtime` — tracked for hot-reload
- `self.devil_threshold` — can change between calls via hot-reload
- The pipeline itself is stateless (called fresh per `analyze()` call)

**`import pandas as pd` inside the function body (line 344):** A local import inside `analyze()`. This suggests it was added later as an afterthought and the author knew it was slightly awkward but left it.

**TODO/FIXME:** None in this file.

**Broken constructor signature (⚠️ FLAG):** The grid-search backtests call `MLStrategy(model_path=..., threshold=0.50)`. The current `MLStrategy.__init__` accepts no `model_path` or `threshold` arguments — those are old parameter names from a previous version. The grid-search scripts **cannot instantiate `MLStrategy`** with the current codebase. They would raise `TypeError: __init__() got an unexpected keyword argument 'model_path'`.

---

## 4. MLFactoryStrategy

**Inheritance:** `MLFactoryStrategy(MLStrategy)` — inherits from `MLStrategy`, which inherits from `Strategy(ABC)`.

**What it adds:**
1. Forces `warmup_period=260` minimum via `kwargs.setdefault`
2. Forces V3 `RandomForestTrainer` instances via `kwargs.setdefault`
3. After calling `super().__init__(**kwargs)`, **overwrites `self.pipeline`** with a fresh `FeaturePipeline([V3BaseFeatures(), V3HTFFeatures(timeframe="5m")])`
4. Overrides `analyze()` but only to call `super().analyze(data)` and return the result unchanged

**Purpose of the factory variant:** The overwrite of `self.pipeline` after `super().__init__()` is the functional core of this class. The intent is to guarantee that the factory strategy always uses the V3 features regardless of what `MLStrategy.__init__()` constructed. In practice, `MLStrategy.__init__()` already constructs the same pipeline — so this override is **defensive/redundant**, not structurally necessary.

The class comment says "The Brain: Implementation of the ML Factory Strategy. Wraps the core MLStrategy logic with Factory-specific requirements." The "factory" in the name likely refers to the FactoryOrchestrator context, not the Factory design pattern.

**What it does NOT override:** `get_order_params()`, `_check_model_updates()`, `_load_threshold()`, `warmup_period`, `feature_names`. It's a thin wrapper with a pipeline-guarantee.

**⚠️ Surprising non-finding:** `MLFactoryStrategy` imports `Signal` from `core.signal` directly but never constructs one — the import is unused. The actual signal construction happens in the inherited `MLStrategy.analyze()`.

---

## 5. Call Sites

### 5a. `live_orchestrator.py`

**Instantiation (line 437):**
```python
self._strategy = MLStrategy(
    angel_path=angel_model_path,
    devil_path=devil_model_path,
    angel_threshold=ANGEL_THRESHOLD,
    devil_threshold=self._devil_threshold,
    warmup_period=MIN_HISTORY_BARS,
)
```

**⚠️ Critical finding — broken attribute access:** `LiveOrchestrator._run_inference()` does **not** call `self._strategy.analyze()`. Instead it reimplements the entire Angel/Devil inference inline, accessing:
```python
self._strategy.angel_model.predict_proba(...)   # line 1186
self._strategy.devil_model.predict_proba(...)   # line 1208
```

But `MLStrategy` has no `angel_model` or `devil_model` attributes. It has `self.angel_trainer` and `self.devil_trainer` (instances of `V3RandomForestTrainer`). The `.model` is one level deeper: `self.angel_trainer.model`.

**`live_orchestrator.py` is accessing `self._strategy.angel_model` which doesn't exist. This will raise `AttributeError` at runtime on the first inference call.**

This is a real breakage — not a design difference, not a migration issue, a concrete runtime crash that would happen the first time the production orchestrator tries to execute inference.

**What `live_orchestrator` does with signals:** Its own `_run_inference()` method constructs a `core.signal.Signal` with `metadata={angel_prob, devil_prob, natr_14, atr_abs, sl_price, tp_price}`. The SL and TP are embedded in the signal metadata, computed from ATR:
```python
"sl_price": round(current_price - SL_ATR_MULTIPLIER * atr_abs, 4),
"tp_price": round(current_price + TP_ATR_MULTIPLIER * atr_abs, 4),
```

`_submit_entry_order()` extracts `sl_price` and `tp_price` from `sig.metadata`, then recomputes position size from the SL distance and `ACCOUNT_RISK_PER_TRADE=0.02`. No `get_order_params()` call anywhere.

### 5b. `factory_orchestrator.py`

**Instantiation:** Takes `strategy: MLStrategy` as a constructor argument (not instantiated internally).

**How `analyze()` is called:**
```python
signal_result = await asyncio.to_thread(self.strategy.analyze, {symbol: history})
signals, _ = signal_result
```

The second return value (`highest_angel_prob`) is **explicitly discarded** (`_`). Only `signals` is used. `factory_orchestrator` is the only call site that actually calls `strategy.analyze()` correctly.

**What it does with the signal:**
```python
if symbol not in self.active_positions:
    await self._execute_buy(signals[0])
```

`_execute_buy()` uses `RiskManager.calculate_bracket(sig.price, atr)` and `RiskManager.calculate_quantity(...)`. The `atr` comes from `sig.metadata.get("atr_abs", sig.price * 0.001)`. But wait — `MLStrategy.analyze()` only puts `{angel_prob, devil_prob}` in metadata, **not** `atr_abs`. So `factory_orchestrator` falls back to `sig.price * 0.001` (0.1% of price) as a default ATR every time. **This is a silent bug**: the ATR-based bracket sizing degrades to a fixed 0.1% default because the signal never carries ATR data.

**Order submission uses Alpaca SDK enums directly:**
```python
req = MarketOrderRequest(
    symbol=symbol,
    qty=qty,
    side=OrderSide.BUY,        # alpaca.trading.enums.OrderSide
    time_in_force=TimeInForce.GTC  # alpaca.trading.enums.TimeInForce
)
```

### 5c. `grid_search_backtest.py`

**Constructor call (broken):**
```python
strategy = MLStrategy(model_path="src/ml/models/rf_model.joblib", threshold=0.50)
```

As noted above, this uses old argument names. The current `MLStrategy.__init__` will reject these with `TypeError`.

**How `OrderParams` is used:**
```python
order_params = OrderParams(
    risk_percentage=0.02,
    tp_multiplier=tp_mult,
    sl_multiplier=sl_mult,
    use_trailing_stop=False,
)
# ...
sl = sig.price * self.order_params.sl_multiplier   # percentage multiplier
tp = sig.price * self.order_params.tp_multiplier   # percentage multiplier
qty = risk / sig.price                              # where risk = capital * risk_percentage
```

The backtest instantiates its own `OrderParams` separately from the strategy — it doesn't call `strategy.get_order_params()`. The strategy is only used for `strategy.analyze()` and the backtest manages risk itself. The `OrderParams` import exists only for the backtest's `BOM` class, not to call through the strategy interface.

---

## 6. Interpretation

### Q1: Original contract

The original design intent of `MLStrategy.analyze()` was:

> "Given a dict of {symbol → recent OHLCV DataFrame}, run the Angel/Devil meta-labeling inference and return a list of approved BUY signals (typed as `core.signal.Signal`) plus the highest Angel probability seen. Each signal carries: symbol, type=BUY, entry price, conviction score (devil_prob), timestamp, and model probability metadata. Risk sizing and bracket calculation are *not* the strategy's responsibility."

The `get_order_params()` method was a vestige of the legacy ABC contract — included to satisfy the abstract method requirement but never wired into any downstream consumer. It represents a design intent that was abandoned before it was ever used.

### Q2: Strategy/execution boundary

| Decision | Who decides |
|----------|-------------|
| Whether to enter a trade | `MLStrategy.analyze()` (Angel/Devil approval) |
| What size to enter at | **Orchestrator** (`factory_orchestrator.py` via `RiskManager`; `live_orchestrator.py` inline) |
| Where to set SL/TP | **Orchestrator** (ATR-based; `live_orchestrator` embeds in signal metadata; `factory_orchestrator` delegates to `RiskManager`) |
| Time-in-force | **Orchestrator** hardcoded: `TimeInForce.GTC` (Alpaca enum directly) |
| Broker order construction | **Orchestrator** (`MarketOrderRequest` with Alpaca SDK) |

The boundary is clean on paper: the strategy says "yes/no + what price," the orchestrators handle everything else. In practice, `live_orchestrator` breaks this by bypassing `analyze()` and directly accessing model internals.

### Q3: Signal vs OrderParams mental model

**The existing `MLStrategy` code is firmly Signal-returning, not OrderParams-returning.**

`analyze()` returns `(List[core.signal.Signal], float)`. The signal carries conviction but no bracket, no position size, no order type. `get_order_params()` exists and returns an `OrderParams` instance, but it is called by nobody. The `OrderParams` stored in `self.order_params` uses percentage multipliers (`sl_multiplier=0.998`) while the actual production execution uses ATR-based absolute distances — they are different risk parametrizations and the production code chose ATR.

The existing code is not "somewhere in between." `get_order_params()` is genuinely dead code — a zombie method from the legacy ABC contract that never got wired up.

### Q4: Cost of migrating to `base.Signal`

`base.Signal` has: `direction: str`, `entry_price: float`, `raw_sl_distance: float`, `raw_tp_distance: float`, `metadata`.

`core.signal.Signal` has: `symbol: str`, `type: SignalType`, `price: float`, `confidence: float`, `timestamp: datetime`, `metadata`.

Files that would need to change:

| File | Required change |
|------|-----------------|
| `src/strategies/concrete_strategies/ml_strategy.py` | Change `analyze()` return type and signal construction; add `symbol` to `base.Signal` or pass it in metadata; rename `price` → `entry_price`; add `raw_sl_distance`, `raw_tp_distance` (currently absent); remove `confidence`, `timestamp` from signal (or move to metadata) |
| `src/execution/factory_orchestrator.py` | Update signal field access: `sig.price` → `sig.entry_price`; `sig.symbol` needs to come from somewhere; discard `sig.type` usage; update how SL/TP distances are consumed |
| `src/execution/live_orchestrator.py` | Same field renames; ATR bracket computation would also need to move into the signal |
| `src/strategies/strategy.py` | `analyze()` return type annotation update |
| `grid_search_backtest.py` | `sig.price` → `sig.entry_price`, etc. |

The `base.Signal` contract also requires `raw_sl_distance` and `raw_tp_distance` to be filled in by the strategy. This is a philosophical change: it moves bracket responsibility from the orchestrator to the strategy, which is the opposite of what the current code does. The migration cost is moderate-to-high: 4-5 files, but the semantic shift is larger than the line count suggests.

### Q5: Cost of keeping OrderParams broker-agnostic

If `OrderParams` is kept as the risk-configuration carrier (strategies return signals, but also have `get_order_params()` that orchestrators call), and we replace Alpaca enums in `factory_orchestrator.py` with `src/execution/enums.py`:

| File | Required change |
|------|-----------------|
| `core/order_management.py` | Create/rewrite with broker-agnostic fields (use `execution.enums.OrderSide` etc.) |
| `src/strategies/concrete_strategies/ml_strategy.py` | Import from new `core.order_management`; keep `get_order_params()` as-is |
| `src/execution/factory_orchestrator.py` | Replace `alpaca.trading.enums.OrderSide/TimeInForce` with `execution.enums`; update `MarketOrderRequest` construction to translate from SDK-agnostic enums to Alpaca enums |
| `grid_search_backtest.py` | Update import path only |

This approach is considerably cheaper: 3 files modified, no semantic changes to signal flow, no interface migration. The `get_order_params()` method remains vestigial unless we explicitly wire it in.

### Q6: Honest read on design intent

The original intent was a clean two-layer system: the strategy is a signal oracle (in/out decision + conviction score), and the orchestrators handle all trade mechanics. This intent is largely preserved in `MLStrategy.analyze()`, which genuinely does only one thing: run the Angel/Devil pipeline and emit a `core.signal.Signal`. The `get_order_params()` method is the one place where the original ABC's more prescriptive vision bled through — the V1 ABC imagined strategies that not only signal but also specify their own risk parameters, but this idea was abandoned in practice before it was ever wired up. 

What has accumulated over time is architectural entropy in the orchestrators, not the strategy. `LiveOrchestrator` has become a 1,700+ line monolith that reimplements the entire inference pipeline inline rather than delegating to `strategy.analyze()`, and in doing so introduced the `angel_model` attribute bug — it accesses an attribute that doesn't exist on `MLStrategy`. `FactoryOrchestrator` calls `analyze()` correctly but silently degrades to a fixed 0.1% ATR fallback because `MLStrategy` never puts `atr_abs` in the signal metadata. The strategy code is cleaner and more intentional than the orchestrator code; the debt is in the wiring, not the ML logic itself.

---

## Flags for Captain B

| Severity | File | Finding |
|----------|------|---------|
| 🔴 **Runtime crash** | `src/execution/live_orchestrator.py:1186,1208` | Accesses `self._strategy.angel_model` and `self._strategy.devil_model` which do not exist on `MLStrategy`. The correct attributes are `self._strategy.angel_trainer.model`. This will raise `AttributeError` on the first inference call in production. |
| 🔴 **Import crash** | `src/strategies/strategy.py:6` | `from core.order_management import OrderParams` fails at runtime — `core/order_management.py` doesn't exist on disk. Any code that imports `strategy.py` (including `ml_strategy.py`) will raise `ModuleNotFoundError` unless `core/order_management.py` is created first. |
| 🟡 **Silent bug** | `src/execution/factory_orchestrator.py:116` | `atr = sig.metadata.get("atr_abs", sig.price * 0.001)` — `MLStrategy.analyze()` never puts `atr_abs` in metadata, so `factory_orchestrator` always uses the 0.1% fallback for ATR. Bracket sizes are systematically wrong when using this orchestrator. |
| 🟡 **Broken constructor** | `grid_search_backtest.py:45`, `grid_search_backtest_q1.py:43` | `MLStrategy(model_path=..., threshold=0.50)` — old argument names. Will raise `TypeError` immediately. |
| 🟡 **Dead code** | `MLStrategy.get_order_params()` | Never called by any orchestrator. The `OrderParams` stored in `self.order_params` uses percentage multipliers incompatible with the ATR-based sizing used in production. |
| ℹ️ **Redundant override** | `MLFactoryStrategy.__init__` | Overwrites `self.pipeline` with an identical object to what `super().__init__()` just created. Harmless but unnecessary. |
| ℹ️ **Unused import** | `ml_factory_strategy.py:11` | `from core.signal import Signal` — imported but never used in this file. |
| ℹ️ **Leaked imports on ABC** | `src/strategies/strategy.py:7-10` | `talib`, `pandas`, `numpy`, `polars` imported but never used in the ABC body. |
