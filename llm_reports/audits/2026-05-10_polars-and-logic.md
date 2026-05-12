---
type: audit
date: 2026-05-10
time: 20:37 PDT
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: User-requested deep audit of Polars usage and logical robustness; intended for handoff to Gemini architect
head: 7addd18c85f38c299eaa194ea2c74862c0b006cf
scope: read-only
imported_from: CRITICAL_AUDIT_2026-05-10.md
---

# Build-A-Bot — Critical Code Audit (Polars + Logic)

- **Date:** 2026-05-10
- **Agent:** Claude Opus 4.7
- **Trigger:** User-requested deep audit of Polars usage and logical robustness across the live trading + V4 investor + V5 forex paths. Output is intended for handoff to the Gemini architect AI for a second-opinion design response.
- **HEAD commit:** `7addd18c85f38c299eaa194ea2c74862c0b006cf` (2026-05-10 01:42:23 -0700)
- **Scope:** Read-only audit. No source files modified. No tests run.
- **Branch state at audit:** `main` is 1 commit ahead of `origin/main`. Tree clean.

---

## 0. Executive Summary

Polars is the canonical dataframe library (`Pipfile`: `polars = "*"`). It's used end-to-end in providers, the live aggregator, ML feature engineering, strategies, and the orchestrator. Pandas leaks in at three places only: (a) the SimFin / yfinance edge adapters (`src/data/providers/`), (b) `scripts/investor_feature_pipeline.py` and `scripts/portfolio_orchestrator.py` (the entire V4 pipeline is pandas-native), and (c) a couple of inference hot-path pandas roundtrips that shouldn't exist (`live_orchestrator.py:1202`, `ml_strategy.py:350`).

This audit identifies **25 findings**:

| Tier | Count | Theme |
|---|---|---|
| **CRITICAL** | 7 | Bugs that affect money, correctness, or production behavior |
| **HIGH** | 8 | Race conditions, schema drift, silent failures |
| **MEDIUM (Polars)** | 5 | Polars-specific brittleness / API misuse |
| **MEDIUM (risk/exec)** | 5 | Risk-manager and execution edge cases |

The single most urgent finding is **#1** — a dead `elif` branch in `_on_trade_update` means watchdog-driven SL/TP fills never transition to COOLING. The V3.3 scalper that's currently in paper trading has been silently broken in this codepath since whenever that block was last refactored. The OANDA work just shipped in `7addd18` carries its own concerns but the live path is the bigger fire.

---

## 1. CRITICAL — Bugs affecting money or correctness

### 1.1 `_on_trade_update` dead code path — watchdog SELL fills never trigger COOLING

**File:** `src/execution/live_orchestrator.py:1545-1632`

```python
if event_type in ("fill", "partial_fill"):              # ← matches ALL fills
    async with ctx.lock:
        if ctx.state == SymbolState.PENDING:
            ctx.state = SymbolState.IN_TRADE             # BUY-fill only
            ...
elif event_type in ("canceled", "expired", "rejected"):
    ...
elif event_type == "fill":                              # ← UNREACHABLE
    order_side = str(getattr(order, "side", "")).upper()
    async with ctx.lock:
        if (ctx.state in (IN_TRADE, PENDING_EXIT)
            and order_side == "SELL"):
            await self._enter_cooling(ctx)              # ← never runs
```

The third `elif event_type == "fill"` is shadowed by the first `if`. A watchdog-driven SELL fill arrives as `event_type == "fill"`, the first branch matches, the inner `if ctx.state == PENDING` is false (state is PENDING_EXIT), nothing happens. Symbol stays PENDING_EXIT indefinitely. It never returns to FLAT, never re-enters, the cooling timer is never armed.

Secondary issue in the same function: on `canceled/expired/rejected` (lines 1602-1619), `ctx.last_client_order_id` is **not** cleared. After a manual recovery the dedup check in `_handle_signal` (line 1296) can falsely fire.

**Fix:** collapse to a single fill branch and dispatch on `order.side`:

```python
if event_type in ("fill", "partial_fill"):
    side = str(getattr(order, "side", "")).upper()
    async with ctx.lock:
        if side == "BUY" and ctx.state == SymbolState.PENDING:
            ctx.state = SymbolState.IN_TRADE
            ctx.entry_price = float(getattr(order, "filled_avg_price", 0.0) or 0.0)
            ctx.entry_qty   = float(getattr(order, "filled_qty", 0.0) or 0.0)
            ...
        elif side == "SELL" and ctx.state in (SymbolState.IN_TRADE, SymbolState.PENDING_EXIT):
            await self._enter_cooling(ctx)
```

Also clear `ctx.last_client_order_id = None` in the cancel/expire/reject branch.

---

### 1.2 OANDA tick→bar emits the wrong timestamp

**File:** `src/data/oanda_provider.py:170-178`

```python
self._tick_bars[instrument] = {
    "epoch": bar_epoch,
    "bar_start": ts,                # ← tick timestamp, NOT window floor
    "open":  mid, "high": mid, "low": mid, "close": mid,
    "volume": 0,
}
```

`bar_start` is set to the first tick's timestamp (e.g. `10:34:17.143`). When this bar is then passed to `LiveBarAggregator` downstream, the aggregator's `_window_floor` (`bar_aggregator.py:236-246`) buckets it into the **10:30** window for a 5-minute aggregation, instead of the **10:34** window the source data implies. Worse: it disagrees with the historical-bar path (`get_historical_bars`) which uses the candle's own `time` field — that one IS aligned by OANDA's API.

Result: streaming bars and historical bars are timestamped on different conventions → feature pipelines computed on the live stream produce different indicator values than the same bars from a backtest. This is a silent training/inference skew specifically for V5.

**Fix:** compute the floor explicitly when initializing the bar:

```python
bar_start = datetime.fromtimestamp(
    bar_epoch * self._stream_gran * 60, tz=timezone.utc
)
```

Side note: `bar_aggregator.py:156-158` already establishes the canonical "bars are timestamped at the window start" convention via `pl.lit(window_timestamp)`. The OANDA provider should honor it.

---

### 1.3 `LiveBarAggregator._window_floor` broken when `timeframe` doesn't evenly divide 60

**File:** `src/utils/bar_aggregator.py:236-246`

```python
def _window_floor(self, ts: datetime) -> datetime:
    floored_minute = ts.minute - (ts.minute % self.timeframe)
    return ts.replace(minute=floored_minute, second=0, microsecond=0)
```

The docstring at line 47 says "must be >= 1 and evenly divide 60 for clean clock alignment" but `__init__` only checks `>= 1` (line 54). For `timeframe=7`: windows land at minute 0, 7, 14, 21, 28, 35, 42, 49, 56 within an hour. At the top of the next hour, `ts.minute=0`, `floored_minute=0` — but the previous closed window was at 56, so the apparent gap is **4 minutes**, not 7. The forward-fill at line 186 (`gap_start = closed_window + step`) then injects a synthetic candle at `:03` of the new hour, which is in the *middle* of the new logical window. Indicators on the resulting series operate on misaligned data.

**Fix:** enforce the constraint in `__init__`:

```python
if 60 % self.timeframe != 0 and self.timeframe < 60:
    raise ValueError(
        f"timeframe={timeframe} must evenly divide 60 (or be a multiple of 60)."
    )
```

Or implement true window-floor math via epoch division (same pattern as the OANDA fix above).

---

### 1.4 `FeaturePipeline.clean_data` drops rows on ANY null, not feature subset

**File:** `src/ml/feature_pipeline.py:51-61`

```python
@staticmethod
def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    float_cols = [col for col, dtype in df.schema.items() if dtype in (pl.Float64, pl.Float32)]
    if float_cols:
        df = df.with_columns(
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c in float_cols
        )
    return df.drop_nulls()                  # ← drops on ANY column
```

`drop_nulls()` with no subset removes a row if *any* column has a null. For V4, fundamentals are sparse between quarterly reports — `Total Revenue` is non-null only on report dates and the days the join propagates forward. Many engineered ratios are NaN-by-design on the in-between days. `drop_nulls()` deletes those rows wholesale, even though LightGBM handles NaN natively (and the V4 pipeline relies on that). For inference, if the latest bar has a transient null in any metadata column (e.g. a join-side scratch column that survived), the bar is dropped silently and inference produces no signal.

**Fix:** parametrize on the feature subset:

```python
def clean_data(self, df: pl.DataFrame, *, feature_cols: list[str]) -> pl.DataFrame:
    ...
    return df.drop_nulls(subset=feature_cols)
```

The current `FeaturePipeline` only knows about its `feature_generators`; threading the feature-name list through is straightforward — every generator already declares its outputs.

---

### 1.5 `portfolio_orchestrator.execute_rebalance` liquidates positions outside the V4 universe

**File:** `scripts/portfolio_orchestrator.py:359-376`

```python
to_liquidate = [s for s in current_positions if s not in top_k]
...
for symbol in to_liquidate:
    ...
    order = trading.close_position(symbol)
```

`current_positions` is populated from `trading.get_all_positions()` (line 343) which returns **every** Alpaca position on the account. If V3.3 scalper (or any other strategy / manual position) holds, say, BTC/USD or some equity outside `UNIVERSE`, this monthly cron will liquidate it. There's no V4-ownership tag and no scope filter.

**Fix (any of):**
- Restrict liquidation to `s in (UNIVERSE - set(top_k))` rather than "all positions minus top-K"
- Use a separate Alpaca sub-account for V4
- Tag V4 entries via `client_order_id` prefix (`"V4_"`) and only liquidate matching positions

Recommendation: at minimum gate on `s in UNIVERSE` until a stronger ownership tag is in place.

---

### 1.6 OANDA historical pagination duplicates the boundary candle

**File:** `src/data/oanda_provider.py:277-280`

```python
last_ts = _parse_iso(candles[-1]["time"])
if last_ts <= chunk_start or len(candles) < _MAX_CANDLES:
    break
chunk_start = last_ts                   # ← inclusive boundary on next page
```

OANDA's `from` parameter is inclusive. Setting `chunk_start = last_ts` and re-requesting `from=last_ts, count=5000` returns the boundary candle in **both** pages. Won't show up on short ranges; training datasets pulled across multi-month windows will have duplicates every 5000 candles. A pl.DataFrame built from `rows` does not deduplicate.

**Fix:** advance by the candle period:

```python
chunk_start = last_ts + timedelta(minutes=timeframe_minutes)
```

Or post-filter with `df.unique(subset=["timestamp"], keep="first")` before returning.

---

### 1.7 OANDA `close_position` claims FIFO compliance but ignores partial-fill data

**File:** `src/execution/oanda_order_manager.py:175-189`

```python
req = v20_positions.PositionClose(
    accountID=self._account_id, instrument=oanda_symbol, data=data,
)
self._client.request(req)
...
self._net_positions[oanda_symbol] = 0      # ← assumes full close
self._avg_entry_prices[oanda_symbol] = 0.0
return True
```

OANDA `PositionClose` returns a response containing `longOrderFillTransaction` / `shortOrderFillTransaction` with the units actually filled. In thin liquidity (off-hours, illiquid pair) the response may indicate a partial close. The code zeroes local state regardless, then trusts that until the next `sync_position`. The watchdog-style closes scoped for V5 will be sensitive to this — a partial close would otherwise be invisible to the strategy.

**Fix:** parse the response transaction(s) and update `_net_positions` from `units_filled`:

```python
resp = self._client.request(req)
fill = resp.get("longOrderFillTransaction") or resp.get("shortOrderFillTransaction") or {}
units_filled = int(float(fill.get("units", 0) or 0))
self._net_positions[oanda_symbol] += units_filled  # signed: close is opposite-sign
if self._net_positions[oanda_symbol] == 0:
    self._avg_entry_prices[oanda_symbol] = 0.0
```

---

## 2. HIGH — Race conditions, schema drift, silent failures

### 2.1 Schema drift across `MarketDataProvider` implementations

The three concrete providers disagree on both the **empty-frame** shape and the **timestamp dtype**.

| Provider | `pl.DataFrame` on empty | Timestamp dtype | Volume dtype |
|---|---|---|---|
| `alpaca_provider.py:88,105` | `pl.DataFrame()` — **no schema, no columns** | nanosecond UTC (via `from_pandas`) | inherits pandas dtype (int64) |
| `polygon_provider.py:140-142` | `_BAR_SCHEMA`-shaped empty | `us` UTC | `Float64` |
| `oanda_provider.py:288,291` | `_BAR_SCHEMA`-shaped empty | `us` UTC | `Float64` |

Consequences:
- Any downstream `.select(["timestamp", ...])` on an empty Alpaca frame raises `KeyError`. Polygon/OANDA return cleanly.
- `pl.concat([alpaca_df, polygon_df])` on non-empty frames triggers a schema mismatch or a silent upcast on `timestamp` (`us` vs `ns`).
- `_BAR_SCHEMA` is **redefined** in `polygon_provider.py:25-32` and `oanda_provider.py:32-39` — the comment in the OANDA file says "mirrors `polygon_provider._BAR_SCHEMA`," which means a future edit to one is guaranteed to drift from the other.

**Fix:**
1. Hoist `_BAR_SCHEMA` into `src/data/market_provider.py` as a module constant.
2. Add a `MarketDataProvider._empty_bars()` classmethod returning `pl.DataFrame(schema=_BAR_SCHEMA)`.
3. In `alpaca_provider.get_historical_bars`, force conversion: `pl.from_pandas(df_pandas).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")), pl.col("volume").cast(pl.Float64))`.
4. Replace every `return pl.DataFrame()` / `return pl.DataFrame({col: [] for col in _BAR_SCHEMA}, schema=_BAR_SCHEMA)` with the shared `_empty_bars()` call.

---

### 2.2 `MLStrategy._check_model_updates` is racy under thread-pool inference

**File:** `src/strategies/concrete_strategies/ml_strategy.py:214-289`

Called at the start of every `generate_signals` invocation. `generate_signals` runs inside `asyncio.to_thread` from `live_orchestrator._run_inference` (line 1030+). Multiple symbols fire inference concurrently in the same thread pool. Two threads can both observe a new mtime, both enter the reload block, and both call `self.angel_trainer.load(...)`. While one load is mid-flight, `self.angel_trainer.model` is in an undefined state — the other thread that's already in `predict_proba` is operating on a partially-replaced object.

**Fix:** add a `threading.Lock` around the mtime-check + load:

```python
self._reload_lock = threading.Lock()
...
def _check_model_updates(self) -> bool:
    if not self._reload_lock.acquire(blocking=False):
        return False  # another thread is already reloading; skip
    try:
        ...existing logic...
    finally:
        self._reload_lock.release()
```

Or move reload to a single owning thread (e.g. a 5s periodic task in the orchestrator that swaps in a new strategy instance).

---

### 2.3 OandaMarketProvider drops the in-flight bar on shutdown

**File:** `src/data/oanda_provider.py:307-337`

When `stop_stream()` is called or `KeyboardInterrupt` triggers, `run_stream` exits the for-loop. The partially accumulated `self._tick_bars` is dropped — every symbol's currently-forming bar is lost. For a 1-minute scalper that's potentially up to 59 seconds of price action that the orchestrator never sees.

**Fix:** flush all in-flight bars on shutdown:

```python
finally:
    for instrument, state in list(self._tick_bars.items()):
        self._flush_bar(instrument, state)
    self._tick_bars.clear()
```

---

### 2.4 Per-bar event-loop creation in OANDA + Polygon providers

**Files:** `src/data/oanda_provider.py:199-203`, `src/data/polygon_provider.py:207-210`

```python
try:
    loop = asyncio.get_running_loop()
    loop.create_task(self._callback(bar))
except RuntimeError:
    asyncio.run(self._callback(bar))
```

Both `run_stream` methods execute in a synchronous thread/main-thread context with no running loop. `asyncio.get_running_loop()` raises `RuntimeError` every time → the fallback `asyncio.run()` builds and destroys a fresh event loop **per bar**. Bonus footgun: if a loop ever IS running on a different thread, `loop.create_task(...)` is not thread-safe — silent corruption.

**Fix:** the orchestrator holds the canonical event loop. Pass a reference in and use the thread-safe scheduling primitive:

```python
# In subscribe(...):
self._loop = asyncio.get_event_loop()  # or accept as parameter

# In _flush_bar / _dispatch:
asyncio.run_coroutine_threadsafe(self._callback(bar), self._loop)
```

Single fix applied to OANDA + Polygon; Alpaca already runs its callback inside an async handler so this pattern doesn't apply there.

---

### 2.5 `OandaOrderManager` has no thread safety for `_net_positions`

**File:** `src/execution/oanda_order_manager.py:73-74`

```python
self._net_positions: Dict[str, int] = {}
self._avg_entry_prices: Dict[str, float] = {}
```

Plain dicts mutated by `sync_position` and `close_position` (and the planned fill-stream consumer). Right now everything runs in the same thread, so this is fine. The moment the TransactionStream listener (the next planned module) starts firing updates on a separate thread, you get torn reads and dropped updates with no traceback.

**Fix:** wrap state in a `threading.RLock` *now* while the surface area is two methods:

```python
self._state_lock = threading.RLock()

def get_net_position(self, instrument: str) -> int:
    with self._state_lock:
        return self._net_positions.get(_to_oanda_symbol(instrument), 0)
```

Cheap to add today, hard to retrofit after the fill-stream consumer is wired up.

---

### 2.6 `feature_names` hardcoded in two places, no validation against trained model

**Files:**
- `src/strategies/concrete_strategies/ml_strategy.py:140-161` (the strategy's copy)
- `src/execution/live_orchestrator.py:1117-1139` (the orchestrator's copy)
- `src/ml/trainers/v3_rf_trainer.py:28` exposes `feature_names_in_` from the sklearn model

The strategy and the orchestrator each maintain their own hardcoded 18-element list of feature names. Nothing checks `set(strategy.feature_names) == set(angel_trainer.feature_names_in_)`. If training adds, removes, or renames a feature, inference will silently feed the model the wrong columns (in column order, by position) and the model will produce numbers — just wrong ones.

**Fix:** add a validation assertion at strategy load time:

```python
expected = list(self.angel_trainer.feature_names_in_ or [])
if expected and expected != self.feature_names:
    raise RuntimeError(
        f"Feature schema drift: strategy declares {self.feature_names}, "
        f"model was trained on {expected}"
    )
```

Or — preferred — save a `model_meta.json` alongside the joblib that pins the feature list, and load both together. The retrainer already writes `threshold.json` alongside the model; extend that pattern.

---

### 2.7 Pandas roundtrip + per-bar `import pandas` in inference hot path

**Files:** `src/execution/live_orchestrator.py:1202-1208`, `src/strategies/concrete_strategies/ml_strategy.py:350-355`

```python
import pandas as pd  # local import — pandas only needed here
meta_df = pd.DataFrame(feature_matrix, columns=ml_feature_names)
meta_df["angel_prob"] = angel_prob
devil_prob = float(self._strategy.devil_model.predict_proba(meta_df)[0, 1])
```

sklearn's `predict_proba` accepts ndarrays directly. Building a pandas DataFrame per signal, just to append one column, and then calling sklearn (which immediately re-extracts a numpy array) is pure overhead. The local `import pandas` triggers `sys.modules` lookup but is otherwise free — still, the dead-code comment is misleading.

**Fix:**

```python
import numpy as np  # at top
meta_features = np.hstack([feature_matrix, np.array([[angel_prob]])])
devil_prob = float(self._strategy.devil_model.predict_proba(meta_features)[0, 1])
```

One allocation, no pandas dependency, no per-bar import. Microseconds per signal, but adds up across 5+ symbols × 1m cadence × 6.5h × 252 days.

---

### 2.8 `_top_quintile_label` is functionally a top-K classifier on a 7-symbol universe

**File:** `scripts/investor_feature_pipeline.py:89-121`

```python
quintiles = pd.qcut(valid, q=5, labels=False, duplicates="drop")
top_bin = int(quintiles.max())
result[valid_mask] = (quintiles == top_bin).astype(float)
```

`UNIVERSE = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "WMT", "JNJ"]` — 7 symbols, so per-date groupby has at most 7 observations. `pd.qcut(7 values, q=5, duplicates="drop")` collapses to 3-4 bins with heavy ties; "top quintile" effectively labels the top 1-2 symbols. The documented fallback ("qcut produces < 5 bins (ties) → use the observed maximum bin so at least one symbol always receives label 1") confirms the silent re-interpretation.

Net effect: the V4 LightGBM ranker is trained on a target that's really a "top-1-or-2" classifier, not a quintile labeler. With `TOP_K = 2` in the orchestrator (`portfolio_orchestrator.py:77`), the system is internally consistent — but the framing is misleading and the training set has very few positive labels per date.

**Fix (any of):**
- Rename function and column to `_top_k_label` / `target_top_k`, set `k = TOP_K`
- Expand UNIVERSE to ~20+ symbols before calling it a quintile labeler
- Switch to a regression target (`forward_return_60d`) and let LambdaRank rank it directly without binarizing

---

## 3. MEDIUM — Polars-specific brittleness

### 3.1 `pl.concat(how="vertical_relaxed")` silently coerces dtypes

**File:** `src/ml/feature_pipeline.py:93`

```python
combined = pl.concat(processed_frames, how="vertical_relaxed")
```

`vertical_relaxed` lets Int64 + Float64 coexist by upcasting Int → Float. If `volume` drifts from int to float between providers (it does — see #2.1), downstream filters like `pl.col("volume") == 0` will operate on now-floating zeros (`0.0`) and produce surprising matches against literal int `0` in older code paths.

**Fix:** force dtypes upstream (provider-level) and use strict `vertical` concat, OR explicitly enumerate the cast in `clean_data`.

---

### 3.2 `pl.DataFrame(self.buffer)` infers schema per call

**File:** `src/utils/bar_aggregator.py:153`

```python
chunk_df = pl.DataFrame(self.buffer)        # ← schema inferred from dicts
```

`self.buffer` is a list of bar dicts arriving from a stream callback. If `volume` is int on one bar and float on the next (e.g. Alpaca int volumes mixed with OANDA float tick counts when test fixtures cross providers), the inferred schema flips between batches. The aggregate at line 155-164 then casts everything to `Float64` — so the bug doesn't surface — but the construction is fragile.

**Fix:** pass schema explicitly, as `_forward_fill_gaps` already does at line 222:

```python
chunk_df = pl.DataFrame(self.buffer, schema=self._SCHEMA)
```

---

### 3.3 `pl.lit(datetime)` tz-cast not test-covered

**File:** `src/utils/bar_aggregator.py:156-158`

```python
pl.lit(window_timestamp)
  .alias("timestamp")
  .cast(pl.Datetime(time_unit="us", time_zone="UTC")),
```

The cast is defensively correct. Polars has changed how aggressively it preserves tz-info from a Python `datetime` across recent versions (0.20 → 1.x has shifts in this area; Pipfile is now on `polars = "*"`, currently resolving to `polars==1.40.1` per commit `8fe0789`). A future Polars upgrade could silently start producing tz-naive literals if this cast is ever removed. There's no test asserting the schema.

**Fix:** add a unit test that asserts `aggregator.history_df.schema["timestamp"] == pl.Datetime(time_unit="us", time_zone="UTC")` after pushing one bar through.

---

### 3.4 `df["close"][-1]` legacy idiom

**File:** `src/utils/bar_aggregator.py:199`

```python
last_close: float = self.history_df["close"][-1]
```

Negative-index `Series.__getitem__` still works in polars 1.x but is undocumented and quietly deprecated in favor of `.last()` / `.tail(1)[0]`. The idiom is inconsistent with the rest of the codebase, which mostly uses `.tail(1)[0]` (e.g. `ml_strategy.py:327`, `live_orchestrator.py:1056`).

**Fix:** `last_close: float = self.history_df.get_column("close").last()`

---

### 3.5 Verbose empty-frame construction repeated across providers

`{col: [] for col in _BAR_SCHEMA}` appears in 3+ provider files (oanda, polygon, and would also be needed in alpaca's empty path). Polars supports `pl.DataFrame(schema=_BAR_SCHEMA)` directly — one line, correct shape, no dict-comprehension boilerplate.

**Fix:** roll into the `MarketDataProvider._empty_bars()` helper proposed in #2.1.

---

## 4. MEDIUM — Risk / execution

### 4.1 `RiskManager.calculate_bracket` rounds to 4 decimals — loses a pip on forex

**File:** `src/execution/risk_manager.py:22-36`

```python
return round(sl_dist, 4), round(tp_dist, 4)
```

EUR/USD quotes to 5 decimals (1.08234). Rounding the SL/TP distance to 4 decimals chops a full pip. For an equities scalper that's fine; for V5 forex it's a real precision bug — a 5-pip TP becomes a 4.5-pip TP after rounding noise across multiple multiplications.

**Fix:** make the precision asset-class-aware, or stop rounding here and round only at order-submission time using the broker's reported tick size.

---

### 4.2 `RiskManager.calculate_quantity` 0.0001-unit floor is meaningless on OANDA

**File:** `src/execution/risk_manager.py:76`

```python
return max(round(final_qty, 4), 0.0001)
```

OANDA's minimum trade size is **1 unit** (whole units of base currency), not fractional. 0.0001 units is unsubmittable. The current `RiskManager` is Alpaca-shaped (fractional crypto/equity); reusing it directly for the V5 forex path will hit either silent rejections or wasted floor logic.

**Fix:** asset-class-aware quantizer. Suggested shape:

```python
@dataclass
class RiskProfile:
    qty_decimals: int = 4         # 4 for Alpaca, 0 for OANDA forex
    min_qty: float = 0.0001       # 0.0001 for Alpaca, 1.0 for OANDA
```

---

### 4.3 No atomic write between miner → features → orchestrator

**File:** `scripts/portfolio_orchestrator.py:145-171`

```python
def refresh_data() -> None:
    _run_subprocess("Stage 1/4 — Data miner", [...])
    if not _RAW_DATA_PATH.exists():
        raise FileNotFoundError(...)
```

If the miner crashes mid-write of `v4_investor_data.parquet`, the next stage reads either a truncated file or, worse, an older successful copy that survived the partial overwrite. The check `if not _RAW_DATA_PATH.exists()` doesn't catch corruption.

**Fix:** write to `.tmp` and `os.replace()` for atomicity in the miner itself. Add a checksum or `_pyarrow.parquet.read_metadata()` round-trip in the orchestrator after the subprocess returns.

---

### 4.4 No concurrent-run guard on `portfolio_orchestrator`

If cron misfires (e.g. system clock jump) or the user kicks one off manually while another is mid-flight, you get duplicate Alpaca orders. There's no PID file, no `flock`, no process check.

**Fix:** wrap `main()` in a `fcntl.flock` on a sentinel file:

```python
import fcntl
lock_path = _PROJECT_ROOT / ".portfolio_orchestrator.lock"
with open(lock_path, "w") as f:
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("Another orchestrator instance is running; aborting.")
        return 1
    ...rest of main...
```

---

### 4.5 `MarketOrderRequest(TimeInForce.DAY)` at 16:30 ET cron submits after-close

**File:** `scripts/portfolio_orchestrator.py:37, 464`

```python
# Cron: 30 16 1 * *  → 16:30 ET on the 1st of every month
order_request = MarketOrderRequest(symbol=..., time_in_force=TimeInForce.DAY, ...)
```

Market closes at 16:00 ET. A DAY order submitted at 16:30 ET either gets rejected by Alpaca or queues to the **next trading session's open** — which is the most volatile window of the day. The "monthly rebalance after close" pattern intended in the docstring is incompatible with `TimeInForce.DAY`.

**Fix:** either move the cron inside RTH (e.g. `0 15 1 * *` for 15:00 ET, an hour before close), or switch to `TimeInForce.OPG` (at-the-open auction) and accept open-print pricing.

---

## 5. Cross-cutting patterns to systematize

Several findings cluster around the same structural gaps. Worth addressing as patterns rather than per-file:

1. **Single source of truth for `_BAR_SCHEMA`** — currently redefined in `oanda_provider.py:32`, `polygon_provider.py:25`, and absent from alpaca's empty path. Hoist into `src/data/market_provider.py` and import everywhere. Add a `MarketDataProvider._empty_bars()` helper.

2. **Single source of truth for symbol normalization** — `_to_oanda_symbol` is duplicated in `oanda_provider.py:63-65` and `oanda_order_manager.py:27-29` (R5 in the OANDA integration report). Move to `src/data/symbols.py` or `src/execution/oanda/_symbols.py`.

3. **Feature column list lives with the model artifact** — save `feature_names` into a `model_meta.json` alongside the joblib (extend the existing retrainer pattern that writes `threshold.json`). Strategy + orchestrator load this list, never declare it inline. Asserts catch drift at load time, not in production.

4. **One long-lived event loop, never `asyncio.run()` per callback** — every provider that bridges a sync stream to async callbacks needs the same pattern. Codify as a `BridgedAsyncCallback` helper in `src/data/_async_bridge.py` and reuse.

5. **Every error/empty path returns a schema-shaped frame** — never bare `pl.DataFrame()`. Add a linter rule or unit test that asserts every provider's `get_historical_bars` returns a frame with `set(df.columns) >= {"timestamp", "open", "high", "low", "close", "volume"}`.

---

## 6. Recommended strike order for the V5 forex push

Ranked by blast radius for the active V5 pivot specifically (per the `project_v5_forex_pivot.md` memory):

1. **Finding #1.1 — fix `_on_trade_update` dead path.** Production V3 scalper is silently broken in this codepath; V5 will inherit the same orchestrator skeleton unless this gets cleaned up first.

2. **Finding #1.2 — fix OANDA bar timestamp to window-floor.** Otherwise V5 indicators are computed on misaligned data from day one. Aggregator integration is fragile until this is corrected.

3. **Finding #2.1 — centralize `_BAR_SCHEMA` and force tz/dtype parity.** Prerequisite for V5 sharing `LiveBarAggregator` and strategy code with V3.

4. **Finding #2.5 — lock `OandaOrderManager` state.** Cheap now while the surface area is two methods; hard to retrofit once the planned TransactionStream consumer is wired.

5. **Finding #2.4 — replace per-bar `asyncio.run()` with `run_coroutine_threadsafe`.** Single helper, fixes three providers, removes a latent crash for the eventual concurrent-stream scenario.

Items 1.4, 1.5, 1.6, 4.1, 4.2 are also relevant for V5 but lower urgency. Items in §3 (Polars) are stylistic — fix as we touch the files.

---

## 7. Out of scope for this audit

Things deliberately not investigated (would extend the audit significantly):
- `src/ml/data_miner.py` — V3 data harvesting paths
- `src/day_trading/` — the 5m intermediate model the audit roadmap (2026-05-05) flagged as ghost feature
- `src/analysis/` and `src/replay_test.py`
- The 8+ root-level `backtest_*.py` files (also flagged in audit_roadmap_2026-05-05.md)
- `tests/` — coverage and quality
- `Dockerfile` / deployment concerns
- The retrainer pipeline (`src/core/retrainer.py`)

Happy to expand into any of these if Gemini's response surfaces something I missed.
