# Strategy Card: RSI + Bollinger Bands

**File:** `src/strategies/concrete_strategies/rsi_bbands.py`
**Class:** `RSIBBands`
**Type:** Mean-Reversion, Two-Stage Confirmation
**Signals Generated:** BUY only (exits are handled by OrderManager SL/TP)

---

## Overview

This strategy looks for oversold conditions where price has dropped below the lower Bollinger Band while RSI confirms extreme weakness. Rather than buying immediately at the bottom (catching a falling knife), it waits for three recovery signals before entering:

1. RSI begins to recover from oversold territory.
2. Bollinger Band bandwidth is expanding (volatility increasing, suggesting a move is starting).
3. A bullish engulfing candlestick pattern confirms buying pressure on the most recent candles.

The two-stage design means the strategy arms a trigger first (Stage 1), then only fires when confirmation arrives (Stage 2). This filters out false bottoms where price continues to fall.

---

## Entry Logic: Step by Step

```
Price is falling...

STAGE 1: OVERSOLD DETECTION (Arms the trigger)
  |
  |-- Is price below the lower Bollinger Band?
  |     NO  --> Do nothing. Wait.
  |     YES --> Is RSI <= 30?
  |               NO  --> Do nothing. Wait.
  |               YES --> ARM THE TRIGGER. Mark this symbol as "Stage 1 triggered."
  |                       No order placed yet. Now watching for recovery.
  |
  v
Price begins to recover...

STAGE 2: RECOVERY CONFIRMATION (Fires the trigger)
  |
  |-- Is Stage 1 triggered for this symbol?
  |     NO  --> Skip. Stage 2 only runs after Stage 1.
  |     YES --> Check all three conditions:
  |
  |     [1] RSI RECOVERY: Is RSI between 30 and 40?
  |           (Confirms RSI is climbing out of oversold, but not yet neutral)
  |
  |     [2] VOLATILITY EXPANSION: Is Bandwidth ROC > 0.15?
  |           (Confirms Bollinger Bands are widening -- a move is underway)
  |
  |     [3] BULLISH ENGULFING: Does the last candle engulf the previous one?
  |           (Confirms buyers have taken control from sellers)
  |
  |     ALL THREE MET?
  |       YES --> GENERATE BUY SIGNAL. Reset trigger.
  |       NO  --> Hold. Keep watching.
  |
  v
RESET CONDITION:
  |
  |-- If RSI climbs above 45 without triggering Stage 2:
  |     The oversold setup is invalidated. Disarm the trigger.
  |     (The opportunity window has passed.)
```

---

## Bullish Engulfing Pattern

The engulfing check examines the last two completed candles:

```
Previous candle (must be RED):     Current candle (must be GREEN):
  Open: $100                         Open: $96    <-- below prev close
  Close: $97                         Close: $101  <-- above prev open
  (Close < Open = bearish)           (Close > Open = bullish)

The green body completely "engulfs" the red body:
  current_open < prev_close  AND  current_close > prev_open
```

This pattern signals that selling pressure on the previous candle has been overwhelmed by buying pressure on the current candle.

---

## Parameter Reference

### Indicator Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bb_period` | `int` | `20` | Bollinger Bands lookback period. The number of candles used to calculate the moving average and standard deviation. Higher values produce smoother, slower-reacting bands. |
| `bb_std_dev` | `int` | `2` | Bollinger Bands standard deviation multiplier. Controls the width of the bands. A value of 2 means the bands are placed 2 standard deviations above and below the moving average. Higher values make the bands wider (fewer touches). |
| `rsi_period` | `int` | `14` | RSI (Relative Strength Index) lookback period. The number of candles used to calculate RSI. The standard value is 14. Lower values make RSI more sensitive; higher values smooth it out. |
| `roc_period` | `int` | `9` | Rate of Change period for Bollinger Band bandwidth. Measures how fast the bandwidth (upper - lower band distance) is changing over this many candles. Used to detect volatility expansion. |

### Stage 1 Parameters (Oversold Detection)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stage1_rsi_threshold` | `int` | `30` | RSI must be at or below this value to arm Stage 1. The classic oversold level is 30. Raising this value (e.g., 40) makes Stage 1 trigger more easily. Lowering it (e.g., 20) requires a deeper oversold condition. |

**Stage 1 triggers when BOTH conditions are true:**
- `current_price < lower_bollinger_band`
- `rsi_value <= stage1_rsi_threshold`

### Stage 2 Parameters (Recovery Confirmation)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stage2_rsi_entry` | `int` | `30` | Lower bound of the RSI recovery range. RSI must be at or above this value. |
| `stage2_rsi_exit` | `int` | `40` | Upper bound of the RSI recovery range. RSI must be below this value. Together with `stage2_rsi_entry`, this defines the RSI "sweet spot" where the signal fires. |
| `stage2_min_roc` | `float` | `0.15` | Minimum bandwidth Rate of Change. The ROC must exceed this value, confirming that Bollinger Bands are expanding. Higher values require stronger volatility expansion. Setting this near zero (e.g., 0.0001) effectively disables the volatility filter. |

**Stage 2 fires when ALL conditions are true:**
- `stage2_rsi_entry <= rsi_value < stage2_rsi_exit`
- `bandwidth_roc > stage2_min_roc` (and not NaN)
- Bullish engulfing pattern present on the last two candles

**Stage 1 resets (disarms) when:**
- `rsi_value > stage2_rsi_exit + 5` (default: RSI > 45)

### Timeframe

| Parameter | Type | Default | Description |
|---|---|---|---|
| `timeframe` | `int` | `5` | The aggregation interval in minutes. The `LiveBarAggregator` converts 1-minute bars into candles of this size. The strategy analyzes these aggregated candles, not raw 1-minute data. |

### Warmup Period

Calculated automatically: `max(bb_period, rsi_period, roc_period) + 1`

With default parameters: `max(20, 14, 9) + 1 = 21 candles`

At a 5-minute timeframe, this requires 105 minutes of data. The Rapid Warmup system fetches this from historical data on startup.

---

## Order Parameters

These are returned by `get_order_params()` and control how the OrderManager sizes and manages positions opened by this strategy.

| Parameter | Value | Meaning |
|---|---|---|
| `risk_percentage` | `0.02` | Risk 2% of capital per trade. With $100,000 capital and a $100 stock, this produces a position of 20 shares ($2,000 / $100). |
| `tp_multiplier` | `1.5` | Take profit at 150% of entry price (a +50% gain). With a $100 entry, TP triggers at $150. |
| `sl_multiplier` | `0.9` | Stop loss at 90% of entry price (a -10% loss). With a $100 entry, SL triggers at $90. |
| `use_trailing_stop` | `False` | Trailing stops are not currently active. Exits are fixed at the calculated SL/TP levels. |

**Risk/Reward Ratio:** With the defaults, the strategy risks 10% to gain 50%, giving a theoretical reward-to-risk ratio of 5:1. This means the strategy only needs to be right ~17% of the time to break even (before fees and slippage).

---

## Tuning Guide

### Making the strategy more aggressive (more signals)

```python
strategy = RSIBBands(
    stage1_rsi_threshold=40,   # Arm trigger at higher RSI (easier to reach)
    stage2_rsi_entry=20,       # Widen the recovery window (lower bound)
    stage2_rsi_exit=50,        # Widen the recovery window (upper bound)
    stage2_min_roc=0.05        # Accept weaker volatility expansion
)
```

This produces more signals but with lower conviction. Useful for backtesting or high-volume screening.

### Making the strategy more conservative (fewer, higher-quality signals)

```python
strategy = RSIBBands(
    stage1_rsi_threshold=25,   # Require deeper oversold condition
    stage2_rsi_entry=25,       # Narrow the recovery window
    stage2_rsi_exit=35,        # Narrow the recovery window
    stage2_min_roc=0.25        # Require stronger volatility expansion
)
```

Fewer trades, but each one has stronger confirmation.

### Testing / Simulation mode (maximum signal generation)

```python
strategy = RSIBBands(
    stage1_rsi_threshold=70,   # Almost always triggers
    stage2_rsi_entry=10,       # Almost any RSI passes
    stage2_rsi_exit=90,        # Almost any RSI passes
    stage2_min_roc=0.0001      # Effectively no ROC filter
)
```

This is what `tests/test_live_simulation.py` uses. It is **not suitable for live trading** -- it exists to verify the order pipeline works end-to-end.

### Adjusting the timeframe

```python
strategy = RSIBBands(timeframe=15)  # Analyze on 15-minute candles
```

Longer timeframes produce smoother signals with fewer false positives, but react slower to market moves. The warmup period scales proportionally: 21 candles at 15 minutes = 315 minutes of historical data needed.

### Adjusting risk

Risk parameters are set inside the `RSIBBands.__init__` method (the `OrderParams` constructor). To change them, either:

1. Modify the defaults in `rsi_bbands.py`:
   ```python
   self.order_params = OrderParams(
       risk_percentage=0.01,    # 1% risk per trade
       tp_multiplier=1.2,       # +20% take profit
       sl_multiplier=0.95,      # -5% stop loss
   )
   ```

2. Or subclass `RSIBBands` and override `get_order_params()`.

---

## Indicators Used

| Indicator | TA-Lib Function | Purpose in Strategy |
|---|---|---|
| **Bollinger Bands** | `talib.BBANDS(close, timeperiod, nbdevup, nbdevdn, matype=0)` | Lower band defines the oversold boundary for Stage 1. Bandwidth (upper - lower) measures volatility. |
| **RSI** | `talib.RSI(close, timeperiod)` | Measures momentum. Confirms oversold in Stage 1, confirms recovery in Stage 2. |
| **Rate of Change** | `talib.ROC(bandwidth, timeperiod)` | Applied to the Bollinger Band bandwidth (not price). Measures how fast volatility is expanding. Positive ROC means bands are widening. |
| **Bullish Engulfing** | Custom implementation | Checks the last two candles for a reversal pattern. Not a TA-Lib function -- implemented directly in `is_bullish_engulfing()`. |
