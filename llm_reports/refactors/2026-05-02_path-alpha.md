---
type: refactor
date: 2026-05-02
time: unknown
agent: Claude Sonnet 4.6
model: claude-sonnet-4-6
trigger: Path Alpha refactor — centralize bracket/sizing logic in RiskManager, fix crypto buying-power, add per-symbol lock
head: 323bf09
scope: modifies-source
related:
  - recons/2026-04-30_bracket-and-sizing.md
imported_from: PATH_ALPHA_REFACTOR_2026-05-02.md
---

# SECTION 1 — Context

The previous recon (`BRACKET_AND_SIZING_RECON_2026-04-30.md`) identified two root-cause bugs observed during the first end-to-end smoke test:

1. **SL/TP ratio inversion** — `MLStrategy` was applying its own `SL_ATR_MULTIPLIER` / `TP_ATR_MULTIPLIER` constants before passing distances to `RiskManager.calculate_bracket`, which then applied them *again*. The signal arriving at the orchestrator had already been multiplied, so the bracket was wrong by the square of the intended multiplier.

2. **Crypto buying-power miscalculation** — Alpaca reports crypto available funds in the `cash` field, not `buying_power`. The orchestrator was always passing `buying_power`, so crypto trades were being sized against a near-zero denominator and either rejected or wildly over-sized.

A third issue was surfaced during review:

3. **Race condition on concurrent signals** — `_execute_buy` and `_close_position` had no mutual exclusion per symbol. Two concurrent signals for the same symbol could both pass the `if symbol in self.active_positions` guard before either wrote to the dict.

---

# SECTION 2 — Changes

## 2.1  `src/strategies/concrete_strategies/ml_strategy.py`

**Before:** Strategy computed `sl_distance` and `tp_distance` using local constants (`SL_ATR_MULTIPLIER=0.5`, `TP_ATR_MULTIPLIER=3.0`) and applied the HF7 SL floor inline. It passed the already-multiplied distances via `Signal.raw_sl_distance` / `Signal.raw_tp_distance`.

**After:** Strategy emits raw `atr_abs` in both fields. All multiplier and floor logic lives exclusively in `RiskManager`. The local constants and the inline floor block are deleted.

```python
# REMOVED from ml_strategy.py:
SL_ATR_MULTIPLIER = 0.5
TP_ATR_MULTIPLIER = 3.0
MIN_SL_PCT = 0.0015

# REMOVED inline bracket block:
sl_distance = SL_ATR_MULTIPLIER * atr_abs
tp_distance = TP_ATR_MULTIPLIER * atr_abs
min_sl_distance = current_price * MIN_SL_PCT
if sl_distance < min_sl_distance:
    sl_distance = min_sl_distance

# NOW: raw ATR passed through
return Signal(
    direction="long",
    entry_price=current_price,
    raw_sl_distance=atr_abs,   # raw, no multiplier
    raw_tp_distance=atr_abs,   # raw, no multiplier
    ...
)
```

Also added `warnings.filterwarnings("ignore", message=".*join_asof.*")` to suppress Polars deprecation noise in logs.

---

## 2.2  `src/execution/risk_manager.py`

**`calculate_bracket` — A3 chop filter**

Return type changed from `tuple[float, float]` to `Optional[Tuple[float, float]]`.

Now returns `None` when the raw ATR stop (`raw_atr * sl_atr_multiplier`) falls below the 0.15% absolute floor. Previously it silently promoted the stop to the floor, masking low-volatility (choppy) conditions. Returning `None` lets the orchestrator skip the trade entirely.

```python
def calculate_bracket(self, entry_price: float, raw_atr: float) -> Optional[Tuple[float, float]]:
    sl_dist = raw_atr * self.profile.sl_atr_multiplier
    tp_dist = raw_atr * self.profile.tp_atr_multiplier

    # A3 chop filter — skip if ATR stop is below the absolute floor
    if sl_dist < (entry_price * self.profile.min_sl_pct):
        return None

    return round(sl_dist, 4), round(tp_dist, 4)
```

**`calculate_quantity` — crypto buying-power fix + $50 floor**

Added `cash: float = 0.0` and `is_crypto: bool = False` parameters.

For crypto, `cash * 0.95` is used as the buying-power cap instead of `buying_power * 0.95`.

Added a `$50` minimum notional guard — returns `0.0` for trades that would result in zombie fractional-share positions.

```python
def calculate_quantity(self, equity, buying_power, entry_price, sl_price,
                       cash=0.0, is_crypto=False) -> float:
    ...
    bp_source = cash if is_crypto else buying_power
    bp_qty = (bp_source * 0.95) / entry_price

    final_qty = min(risk_qty, notional_qty, bp_qty)

    # $50 minimum notional — prevents zombie fractional-share trades
    if final_qty * entry_price < 50.0:
        return 0.0

    return max(round(final_qty, 4), 0.0001)
```

---

## 2.3  `src/execution/factory_orchestrator.py`

**Per-symbol lock**

Added `self._locks: defaultdict = defaultdict(asyncio.Lock)` at init.

Both `_execute_buy` and `_close_position` now acquire `self._locks[symbol]` before touching `self.active_positions`. The double-check pattern in `_execute_buy` re-tests `symbol in self.active_positions` *after* acquiring the lock to handle the TOCTOU window:

```python
async with self._locks[symbol]:
    if symbol in self.active_positions:  # re-check inside lock
        return
    ...submit order and write active_positions...
```

**Path Alpha delegation**

`_execute_buy` now calls `self.risk_manager.calculate_bracket(entry, signal.raw_sl_distance)` first. If it returns `None`, the trade is skipped (A3 chop filter). The bracket distances from `RiskManager` are then used to compute `sl_price` and `tp_price` — `signal.raw_tp_distance` is no longer used directly.

```python
entry = signal.entry_price
bracket = self.risk_manager.calculate_bracket(entry, signal.raw_sl_distance)
if bracket is None:
    logger.info(f"[{symbol}] Volatility too low, skipping trade")
    return

sl_dist, tp_dist = bracket
sl_price = entry - sl_dist
tp_price = entry + tp_dist
```

**Crypto fields forwarded to sizing**

`account.cash` is now fetched and `is_crypto = "/" in symbol` is derived. Both are passed to `calculate_quantity`:

```python
cash = float(account.cash)
is_crypto = "/" in symbol
qty = self.risk_manager.calculate_quantity(
    equity, buying_power, entry, sl_price, cash=cash, is_crypto=is_crypto
)
```

**Zero-qty guard updated**

Changed `if qty <= 0` to `if qty == 0.0` to match the explicit sentinel `RiskManager` now returns for sub-$50 trades (a negative qty should never happen post-refactor, but `0.0` is the defined skip signal).

---

# SECTION 3 — Risk / Regression Notes

| Area | Risk | Mitigation |
|------|------|------------|
| Bracket values | Multipliers now applied once in RM instead of twice (strategy + RM) | Values confirmed in `RiskProfile` defaults: `sl=0.5`, `tp=3.0` — matches prior production constants |
| A3 filter | Trades that previously ran with a floor-promoted SL now skip entirely | Intentional — those were choppy conditions; the floor was masking bad entries |
| Crypto sizing | Uses `cash` field instead of `buying_power` | Verified Alpaca behavior: crypto margin is in `cash`; `buying_power` is equities only |
| Lock contention | Per-symbol lock serializes concurrent entries for the same symbol | Low contention in practice (one signal per symbol per bar); `defaultdict(asyncio.Lock)` is safe for async |
| `raw_tp_distance` | Strategy now passes `atr_abs` for both SL and TP fields | RM uses its own `tp_atr_multiplier` — the TP field from signal is no longer used for bracket math |

---

# SECTION 4 — Files Changed

```
M src/execution/factory_orchestrator.py
M src/execution/risk_manager.py
M src/strategies/concrete_strategies/ml_strategy.py
```

Status: uncommitted as of report date.
