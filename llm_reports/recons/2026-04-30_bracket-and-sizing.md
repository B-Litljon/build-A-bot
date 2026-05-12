---
type: recon
date: 2026-04-30
time: 2026-05-01 12:50 PDT
agent: Kimi K2.6
model: kimi-k2.6
trigger: Diagnostic recon for SL/TP ratio inversion and position sizing observed in tonight's first end-to-end smoke test
head: 323bf095b1aadad3bddce444d95ed7a796410123
scope: read-only
imported_from: BRACKET_AND_SIZING_RECON_2026-04-30.md
---

# SECTION 1 — Sanity check

1.1  `git rev-parse HEAD`
```
323bf095b1aadad3bddce444d95ed7a796410123
```

1.2  `git status --porcelain`
```
```
(No output — clean tree.)

1.3  HEAD prefix matches `323bf09`; tree is clean. Proceeding.

---

# SECTION 2 — MLStrategy.generate_signals

## 2.1  Full file: `src/strategies/concrete_strategies/ml_strategy.py`

```
1: """
2: Meta-Labeling Machine Learning Trading Strategy (Angel & Devil Architecture).
3: 
4: Implements a two-stage inference system with hot-reloading:
5: 1. The Angel (Primary Model): Learns Direction (high recall, threshold 0.40)
6: 2. The Devil (Meta Model): Learns Conviction (high precision, threshold 0.50)
7: 
8: Usage:
9:     from strategies.concrete_strategies.ml_strategy import MLStrategy
10: 
11:     strategy = MLStrategy(
12:         angel_path="models/angel_latest.pkl",
13:         devil_path="models/devil_latest.pkl",
14:         angel_threshold=0.40,
15:         devil_threshold=0.50,
16:         warmup_period=60
17:     )
18: """
19: 
20: import json
21: import logging
22: import os
23: from pathlib import Path
24: from typing import Optional
25: 
26: import numpy as np
27: import polars as pl
28: import pandas as pd
29: 
30: from strategies.base import BaseStrategy, Signal
31: from core.notification_manager import NotificationManager
32: 
33: # CRITICAL: Import FeaturePipeline to prevent training/inference skew
34: from ml.feature_pipeline import FeaturePipeline
35: from ml.features.v3_features import V3BaseFeatures, V3HTFFeatures
36: from ml.trainers.v3_rf_trainer import V3RandomForestTrainer
37: 
38: logger = logging.getLogger(__name__)
39: 
40: # Bracket computation constants (sourced from V3.4 production values in
41: # src/execution/live_orchestrator.py — copied by value to avoid depending
42: # on the broken file).
43: SL_ATR_MULTIPLIER = 0.5  # SL distance = SL_ATR_MULTIPLIER * atr_abs
44: TP_ATR_MULTIPLIER = 3.0  # TP distance = TP_ATR_MULTIPLIER * atr_abs
45: MIN_SL_PCT = 0.0015  # Floor: SL distance / entry_price >= 0.15% (HF7 hotfix)
46: 
47: 
48: class MLStrategy(BaseStrategy):
49:     """
50:     Meta-Labeling ML strategy using two-stage Angel/Devil architecture.
51: 
52:     The Angel (primary model) proposes trades with high recall.
53:     The Devil (meta model) filters false positives with high precision.
54: 
55:     Parameters
56:     ----------
57:     angel_path : str | Path
58:         Path to the Angel (primary) model joblib file.
59:     devil_path : str | Path
60:         Path to the Devil (meta model) model joblib file.
61:     angel_threshold : float
62:         Probability threshold for Angel to propose a trade (default: 0.40).
63:     devil_threshold : float
64:         Probability threshold for Devil to approve a trade (default: 0.50).
65:     warmup_period : int
66:         Minimum candles required before trading (default: 260).
67:     """
68: 
69:     def __init__(
70:         self,
71:         angel_path: str | Path = "models/angel_latest.pkl",
72:         devil_path: str | Path = "models/devil_latest.pkl",
73:         angel_threshold: float = 0.40,
74:         devil_threshold: float = 0.50,
75:         warmup_period: int = 260,  # V3.3: expanded for 5m HTF SMA-50 warm-up
76:         angel_trainer=None,
77:         devil_trainer=None,
78:         **kwargs,
79:     ):
80:         super().__init__(**kwargs)
81: 
82:         self.timeframe = 1  # 1-minute bars
83:         self.warmup = warmup_period
84:         self.angel_threshold = angel_threshold
85:         self.devil_threshold = devil_threshold
86: 
87:         # Load both models
88:         angel_file = Path(angel_path)
89:         devil_file = Path(devil_path)
90: 
91:         if not angel_file.exists():
92:             project_root = Path(__file__).resolve().parent.parent.parent.parent
93:             angel_file = project_root / angel_path
94: 
95:         if not devil_file.exists():
96:             project_root = Path(__file__).resolve().parent.parent.parent.parent
97:             devil_file = project_root / devil_path
98: 
99:         # Store model paths for hot-reloading
100:         self.angel_path = angel_file
101:         self.devil_path = devil_file
102: 
103:         # Load models and track modification times
104:         logger.info(f"Loading Angel model from {angel_file}")
105:         self.angel_trainer = (
106:             angel_trainer if angel_trainer is not None else V3RandomForestTrainer()
107:         )
108:         self.angel_trainer.load(str(angel_file))
109:         if hasattr(self.angel_trainer, "model") and hasattr(
110:             self.angel_trainer.model, "n_jobs"
111:         ):
112:             self.angel_trainer.model.n_jobs = (
113:                 1  # Prevent joblib IPC overhead on single-row inference
114:             )
115: 
116:         self.angel_mtime = os.path.getmtime(angel_file)
117:         logger.info(f"Angel model loaded via trainer (mtime: {self.angel_mtime})")
118: 
119:         logger.info(f"Loading Devil model from {devil_file}")
120:         self.devil_trainer = (
121:             devil_trainer if devil_trainer is not None else V3RandomForestTrainer()
122:         )
123:         self.devil_trainer.load(str(devil_file))
124:         if hasattr(self.devil_trainer, "model") and hasattr(
125:             self.devil_trainer.model, "n_jobs"
126:         ):
127:             self.devil_trainer.model.n_jobs = (
128:                 1  # Prevent joblib IPC overhead on single-row inference
129:             )
130: 
131:         self.devil_mtime = os.path.getmtime(devil_file)
132:         logger.info(f"Devil model loaded via trainer (mtime: {self.devil_mtime})")
133: 
134:         # Initialize notification manager for hot-reload alerts
135:         self.notification_manager = NotificationManager()
136: 
137:         # Initialize feature pipeline (imported, not duplicated!)
138:         self.pipeline = FeaturePipeline(
139:             feature_generators=[V3BaseFeatures(), V3HTFFeatures(timeframe="5m")]
140:         )
141: 
142:         # Feature columns (excluding absolute price columns to prevent leakage)
143:         # V3.4: expanded from 14 to 18 features with Phase 5 microstructure additions
144:         self.feature_names = [
145:             "rsi_14",
146:             "ppo",
147:             "natr_14",
148:             "bb_pct_b",
149:             "bb_width_pct",
150:             "price_sma50_ratio",
151:             "log_return",
152:             "hour_of_day",
153:             "dist_sma50",
154:             "vol_rel",
155:             # V3.3: HTF features
156:             "htf_rsi_14",
157:             "htf_trend_agreement",
158:             "htf_vol_rel",
159:             "htf_bb_pct_b",
160:             # Phase 5: Microstructure features
161:             "range_coil_10",
162:             "bar_body_pct",
163:             "bar_upper_wick_pct",
164:             "bar_lower_wick_pct",
165:         ]
166: 
167:         # Override devil_threshold with the value persisted by the retrainer
168:         # (models/threshold.json).  Must be called AFTER self.devil_threshold is
169:         # set above so _load_threshold() can use it as a fallback.
170:         self.devil_threshold = self._load_threshold()
171: 
172:     def _load_threshold(self) -> float:
173:         """
174:         Load the Devil model's optimal threshold from models/threshold.json.
175: 
176:         Written by retrainer.save_threshold() after a successful validation gate.
177:         Falls back to self.devil_threshold (the value passed to __init__) if the
178:         file is absent or corrupt.
179: 
180:         Returns:
181:             float: The production threshold for Devil approval decisions.
182:         """
183:         # Search relative to project root (4 levels up from this file in src/)
184:         project_root = Path(__file__).resolve().parent.parent.parent.parent
185:         threshold_path = project_root / "models" / "threshold.json"
186:         if not threshold_path.exists():
187:             logger.warning(
188:                 "_load_threshold: models/threshold.json not found — "
189:                 "using constructor default devil_threshold=%.2f",
190:                 self.devil_threshold,
191:             )
192:             return self.devil_threshold
193:         try:
194:             with open(threshold_path, "r") as fh:
195:                 data = json.load(fh)
196:             threshold = float(data["devil_threshold"])
197:             logger.info(
198:                 "_load_threshold: loaded production threshold=%.4f from %s",
199:                 threshold,
200:                 threshold_path,
201:             )
202:             return threshold
203:         except Exception as exc:
204:             logger.warning(
205:                 "_load_threshold: failed to read %s (%s) — "
206:                 "using constructor default devil_threshold=%.2f",
207:                 threshold_path,
208:                 exc,
209:                 self.devil_threshold,
210:             )
211:             return self.devil_threshold
212: 
213:     @property
214:     def warmup_period(self) -> int:
215:         """Returns minimum candles required for indicators to warm up."""
216:         return self.warmup
217: 
218:     def _check_model_updates(self) -> bool:
219:         """
220:         Check for model file updates and hot-reload if necessary.
221: 
222:         Monitors the modification times of model files and reloads
223:         models in memory if they have been updated on disk.
224: 
225:         Returns:
226:             bool: True if any model was reloaded, False otherwise.
227:         """
228:         reloaded = False
229: 
230:         try:
231:             # Check Angel model
232:             if self.angel_path.exists():
233:                 current_angel_mtime = os.path.getmtime(self.angel_path)
234:                 if current_angel_mtime > self.angel_mtime:
235:                     logger.info(
236:                         f"[HOT-RELOAD] Detected new Angel model: {self.angel_path}"
237:                     )
238:                     try:
239:                         self.angel_trainer.load(str(self.angel_path))
240:                         if hasattr(self.angel_trainer, "model") and hasattr(
241:                             self.angel_trainer.model, "n_jobs"
241:                         ):
242:                             self.angel_trainer.model.n_jobs = 1
243:                         self.angel_mtime = current_angel_mtime
244:                         logger.info(f"[HOT-RELOAD] Angel model updated successfully")
245:                         reloaded = True
246:                     except Exception as e:
247:                         logger.error(f"[HOT-RELOAD] Failed to reload Angel model: {e}")
248: 
249:             # Check Devil model
250:             if self.devil_path.exists():
251:                 current_devil_mtime = os.path.getmtime(self.devil_path)
252:                 if current_devil_mtime > self.devil_mtime:
253:                     logger.info(
254:                         f"[HOT-RELOAD] Detected new Devil model: {self.devil_path}"
255:                     )
256:                     try:
257:                         self.devil_trainer.load(str(self.devil_path))
258:                         if hasattr(self.devil_trainer, "model") and hasattr(
259:                             self.devil_trainer.model, "n_jobs"
260:                         ):
261:                             self.devil_trainer.model.n_jobs = 1
262:                         self.devil_mtime = current_devil_mtime
263:                         logger.info(f"[HOT-RELOAD] Devil model updated successfully")
264:                         reloaded = True
265:                     except Exception as e:
266:                         logger.error(f"[HOT-RELOAD] Failed to reload Devil model: {e}")
267: 
267:             # Send notification if any model was reloaded
270:             if reloaded:
271:                 # Also reload the threshold — a retrain always produces a new
272:                 # threshold.json alongside the new model weights.
273:                 old_threshold = self.devil_threshold
274:                 self.devil_threshold = self._load_threshold()
275:                 if self.devil_threshold != old_threshold:
276:                     logger.info(
277:                         "[HOT-RELOAD] Devil threshold updated: %.4f -> %.4f",
278:                         old_threshold,
279:                         self.devil_threshold,
280:                     )
281: 
282:                 alert_message = (
283:                     "🔄 [HOT-RELOAD] New model weights ingested from disk. "
284:                     f"Angel: {self.angel_path.name}, Devil: {self.devil_path.name} "
285:                     f"| devil_threshold={self.devil_threshold:.4f}"
286:                 )
287:                 logger.critical(alert_message)
288:                 self.notification_manager.send_system_message(alert_message)
289: 
290:         except Exception as e:
291:             logger.error(f"[HOT-RELOAD] Error checking for model updates: {e}")
292: 
293:         return reloaded
294: 
295:     def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
296:         """
297:         Analyze single-symbol market data using two-stage Meta-Labeling.
298: 
299:         Stage 1: Angel proposes trades (high recall, low threshold).
300:         Stage 2: Devil filters false positives (high precision).
301: 
301:         Args:
302:             df: Polars DataFrame with OHLCV data for a single symbol.
303:                 Must contain a 'symbol' column (added by callers) so the
304:                 strategy can tag emitted signals with their instrument.
305: 
306:         Returns:
307:             base.Signal on joint Angel & Devil approval, or None.
308:         """
309:         # Check for model updates at the start of each bar processing cycle
310:         self._check_model_updates()
311: 
312:         self.validate_input(df)
313: 
314:         if len(df) < self.warmup_period:
315:             logger.debug(f"Insufficient data ({len(df)} < {self.warmup_period})")
316:             return None
317: 
318:         try:
319:             # Generate features using imported FeatureEngineer
320:             features_df = self._generate_features(df)
321: 
322:             if features_df is None or len(features_df) == 0:
323:                 return None
324: 
325:             # Get latest bar's features for prediction
325:             latest_features_df = features_df[self.feature_names].tail(1)
327:             latest_features = latest_features_df.to_numpy()
328: 
329:             # Get current price for signal
330:             current_price = float(df["close"].tail(1)[0])
331: 
332:             # Resolve symbol — callers add this as a literal column before
333:             # invoking generate_signals (Option A design).
334:             symbol = str(df["symbol"].tail(1)[0]) if "symbol" in df.columns else None
335: 
336:             # ═══════════════════════════════════════════════════════════
337:             # STAGE 1: THE ANGEL (DIRECTION)
338:             # ═══════════════════════════════════════════════════════════
339:             angel_prob = self.angel_trainer.predict_proba(latest_features)[0, 1]
340: 
341:             if angel_prob < self.angel_threshold:
342:                 logger.debug(
343:                     f"[{symbol}] Angel rejected | Prob: {angel_prob:.4f} < {self.angel_threshold}"
344:                 )
345:                 return None
346: 
347:             logger.debug(f"[{symbol}] Angel proposed trade | Prob: {angel_prob:.4f}")
348: 
349:             # ═══════════════════════════════════════════════════════════
350:             # STAGE 2: THE DEVIL (CONVICTION)
351:             # ═══════════════════════════════════════════════════════════
352:             # Build meta-feature set: original features + Angel's probability
353:             meta_features = pd.DataFrame(
354:                 latest_features_df.to_numpy(), columns=self.feature_names
355:             )
356:             meta_features["angel_prob"] = angel_prob
357: 
358:             devil_prob = self.devil_trainer.predict_proba(meta_features)[0, 1]
359: 
360:             if devil_prob < self.devil_threshold:
361:                 logger.debug(
361:                     f"[{symbol}] Devil veto | Angel: {angel_prob:.2f}, Devil: {devil_prob:.2f} < {self.devil_threshold}"
362:                 )
363:                 return None
364: 
365:             # Both Angel and Devil agree — compute ATR-based brackets
366:             natr_value = float(latest_features_df["natr_14"].to_numpy()[0])
367:             # TA-Lib NATR is a percentage; convert to absolute ATR
368:             atr_abs = (natr_value / 100.0) * current_price
369: 
370:             sl_distance = SL_ATR_MULTIPLIER * atr_abs
371:             tp_distance = TP_ATR_MULTIPLIER * atr_abs
372: 
373:             # Apply HF7 SL floor
374:             min_sl_distance = current_price * MIN_SL_PCT
375:             if sl_distance < min_sl_distance:
376:                 logger.debug(
377:                     f"[{symbol}] SL floor applied | raw={sl_distance:.4f} < min={min_sl_distance:.4f}"
378:                 )
379:                 sl_distance = min_sl_distance
380: 
381:             logger.info(
382:                 f"[{symbol}] ANGEL & DEVIL AGREEMENT | "
383:                 f"Price={current_price:.2f} | "
384:                 f"Angel Prob: {angel_prob:.2f} | "
385:                 f"Devil Prob: {devil_prob:.2f} | "
386:                 f"ATR={atr_abs:.4f} | SL={sl_distance:.4f} | TP={tp_distance:.4f}"
387:             )
388: 
389:             return Signal(
390:                 direction="long",
391:                 entry_price=current_price,
392:                 raw_sl_distance=sl_distance,
393:                 raw_tp_distance=tp_distance,
394:                 metadata={
395:                     "symbol": symbol,
396:                     "angel_prob": float(angel_prob),
397:                     "devil_prob": float(devil_prob),
398:                     "atr_abs": atr_abs,
399:                     "timestamp": df["timestamp"].tail(1)[0],
400:                 },
401:             )
402: 
403:         except Exception as e:
404:             logger.error(f"[{symbol}] Error in ML analysis: {e}", exc_info=True)
405:             return None
406: 
407:     def _generate_features(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
408:         """
409:         Generate ML features using imported FeaturePipeline.
410: 
411:         This method ensures zero training/inference skew by using the exact
412:         same feature computation logic as the training pipeline.
413: 
414:         Args:
415:             df: Raw OHLCV DataFrame.
416: 
417:         Returns:
418:             DataFrame with computed features, or None if insufficient data.
419:         """
420:         try:
421:             # Use imported FeaturePipeline.run()
422:             features_df = self.pipeline.run(df)
423: 
424:             # Handle NaN values that may exist in warmup period
424:             feature_cols = [c for c in features_df.columns if c in self.feature_names]
425:             features_df = features_df.drop_nulls(subset=feature_cols)
427: 
428:             return features_df
429: 
430:         except Exception as e:
431:             logger.error(f"Feature generation failed: {e}")
432:             return None
```

## 2.2  Specifically located items with line numbers

a) `SL_ATR_MULTIPLIER` and `TP_ATR_MULTIPLIER` definitions:
```
43: SL_ATR_MULTIPLIER = 0.5  # SL distance = SL_ATR_MULTIPLIER * atr_abs
44: TP_ATR_MULTIPLIER = 3.0  # TP distance = TP_ATR_MULTIPLIER * atr_abs
```

b) `MIN_SL_PCT` definition:
```
45: MIN_SL_PCT = 0.0015  # Floor: SL distance / entry_price >= 0.15% (HF7 hotfix)
```

c) `raw_sl_distance` computation (local name `sl_distance`):
```
370:             sl_distance = SL_ATR_MULTIPLIER * atr_abs
```
Then floored at lines 374-379:
```
374:             min_sl_distance = current_price * MIN_SL_PCT
375:             if sl_distance < min_sl_distance:
376:                 logger.debug(
377:                     f"[{symbol}] SL floor applied | raw={sl_distance:.4f} < min={min_sl_distance:.4f}"
378:                 )
379:                 sl_distance = min_sl_distance
```

d) `raw_tp_distance` computation (local name `tp_distance`):
```
371:             tp_distance = TP_ATR_MULTIPLIER * atr_abs
```
(Note: `tp_distance` is **not** subject to any floor or cap.)

e) `Signal` object construction and return:
```
389:             return Signal(
390:                 direction="long",
391:                 entry_price=current_price,
392:                 raw_sl_distance=sl_distance,
393:                 raw_tp_distance=tp_distance,
394:                 metadata={
395:                     "symbol": symbol,
396:                     "angel_prob": float(angel_prob),
397:                     "devil_prob": float(devil_prob),
398:                     "atr_abs": atr_abs,
399:                     "timestamp": df["timestamp"].tail(1)[0],
400:                 },
401:             )
```

f) ATR computation / provenance:
```
366:             natr_value = float(latest_features_df["natr_14"].to_numpy()[0])
367:             # TA-Lib NATR is a percentage; convert to absolute ATR
368:             atr_abs = (natr_value / 100.0) * current_price
```
The feature column consumed is `natr_14` (produced by `V3BaseFeatures` via `talib.NATR`). `atr_abs` is **not** a feature column; it is derived at inference time from `natr_14` and `current_price`. The comment claims TA-Lib NATR is a percentage; the code divides by 100 before multiplying by price, yielding units of **price** (e.g., dollars for BTC/USD or ETH/USD).

## 2.3  Grep results

Command:
```bash
grep -n "MIN_SL_PCT\|SL_ATR_MULTIPLIER\|TP_ATR_MULTIPLIER\|raw_sl_distance\|raw_tp_distance" \
  src/strategies/concrete_strategies/ml_strategy.py \
  src/strategies/concrete_strategies/ml_factory_strategy.py
```

Output:
```
src/strategies/concrete_strategies/ml_strategy.py:43:SL_ATR_MULTIPLIER = 0.5  # SL distance = SL_ATR_MULTIPLIER * atr_abs
src/strategies/concrete_strategies/ml_strategy.py:44:TP_ATR_MULTIPLIER = 3.0  # TP distance = TP_ATR_MULTIPLIER * atr_abs
src/strategies/concrete_strategies/ml_strategy.py:45:MIN_SL_PCT = 0.0015  # Floor: SL distance / entry_price >= 0.15% (HF7 hotfix)
src/strategies/concrete_strategies/ml_strategy.py:370:            sl_distance = SL_ATR_MULTIPLIER * atr_abs
src/strategies/concrete_strategies/ml_strategy.py:371:            tp_distance = TP_ATR_MULTIPLIER * atr_abs
src/strategies/concrete_strategies/ml_strategy.py:374:            min_sl_distance = current_price * MIN_SL_PCT
src/strategies/concrete_strategies/ml_strategy.py:392:                raw_sl_distance=sl_distance,
src/strategies/concrete_strategies/ml_strategy.py:393:                raw_tp_distance=tp_distance,
```

(No matches in `ml_factory_strategy.py` — the subclass does not override bracket math.)

---

# SECTION 3 — factory_orchestrator._execute_buy

## 3.1  Full file: `src/execution/factory_orchestrator.py`

```
1: import asyncio
2: import logging
3: import signal
4: from datetime import datetime, timezone
5: from typing import List, Dict, Optional
6: 
7: import polars as pl
8: from alpaca.trading.client import TradingClient
9: from alpaca.trading.requests import MarketOrderRequest
10: from alpaca.trading.enums import OrderSide, TimeInForce
11: 
12: from data.feed import MarketDataFeed
13: from execution.risk_manager import RiskManager
14: from strategies.concrete_strategies.ml_strategy import MLStrategy
15: from utils.bar_aggregator import LiveBarAggregator
16: 
17: logger = logging.getLogger(__name__)
18: 
19: 
20: class FactoryOrchestrator:
21:     """
22:     The Router: Async event loop wiring Feed, Strategy, and Risk Manager.
23:     Includes a Universal Watchdog for software SL/TP enforcement.
24:     """
25: 
26:     def __init__(
27:         self,
28:         symbols: List[str],
29:         api_key: str,
30:         secret_key: str,
31:         strategy: MLStrategy,
32:         risk_manager: RiskManager,
33:         feed: MarketDataFeed,
34:         paper: bool = True,
35:     ):
36:         self.symbols = symbols
37:         self.feed = feed
38:         self.strategy = strategy
39:         self.risk_manager = risk_manager
40:         self.trading_client = TradingClient(api_key, secret_key, paper=paper)
41: 
42:         self.aggregators = {
43:             s: LiveBarAggregator(timeframe=1, history_size=400) for s in symbols
44:         }
45:         self.active_positions = {}  # symbol -> {sl, tp, qty}
46:         self._shutdown_event = asyncio.Event()
47: 
48:     async def run(self):
49:         """Main lifecycle entry point."""
50:         # 1. Graceful Shutdown signals
51:         loop = asyncio.get_running_loop()
52:         for s in (signal.SIGINT, signal.SIGTERM):
53:             loop.add_signal_handler(s, lambda: self._shutdown_event.set())
54: 
55:         # 2. THE WARM-UP PHASE
56:         # Solve data starvation: pull 300m history BEFORE websocket
57:         warmup_data = await self.feed.warmup_history(self.symbols, lookback_minutes=300)
58: 
59:         for symbol, df in warmup_data.items():
60:             agg = self.aggregators[symbol]
61:             logger.info(f"Injecting {len(df)} historical bars for {symbol}...")
62:             for row in df.iter_rows(named=True):
63:                 agg.add_bar(row)
64:             logger.info(
65:                 f"Aggregator for {symbol} primed. History size: {len(agg.history_df)}"
66:             )
67: 
68:         # 3. Start Data Pipe
69:         feed_task = asyncio.create_task(
70:             self.feed.subscribe(self.symbols, self._on_tick)
71:         )
72: 
73:         # 4. Universal Watchdog (1s poll)
74:         watchdog_task = asyncio.create_task(self._watchdog_loop())
75: 
76:         logger.info("FactoryOrchestrator active. Waiting for ticks...")
77: 
78:         await self._shutdown_event.wait()
79: 
80:         logger.info("Shutdown signaled. Cleaning up...")
81:         feed_task.cancel()
82:         watchdog_task.cancel()
83:         await self.feed.stop()
84: 
85:     async def _on_tick(self, tick: dict):
86:         """Routes a single bar event into the aggregator and strategy."""
87:         symbol = tick["symbol"]
88:         agg = self.aggregators[symbol]
89: 
90:         # logical clock alignment and aggregation
91:         if not agg.add_bar(tick):
92:             return  # Wait for bar to seal
93: 
94:         # Bar sealed (1m candle complete). Snapshot for inference.
95:         history = agg.history_df.clone()
96:         history = history.with_columns(pl.lit(symbol).alias("symbol"))
97: 
98:         # Offload CPU-bound ML inference to thread
99:         signal = await asyncio.to_thread(self.strategy.generate_signals, history)
100: 
101:         if signal is None:
102:             return
103: 
104:         # Execute if FLAT
105:         if symbol not in self.active_positions:
106:             await self._execute_buy(signal, symbol)
107: 
108:     async def _execute_buy(self, signal, symbol):
109:         """Calculates risk, submits order, and enters watchdog state."""
110:         # Get account for sizing
111:         account = await asyncio.to_thread(self.trading_client.get_account)
112:         equity = float(account.equity)
113:         buying_power = float(account.buying_power)
114: 
115:         # Signal now carries bracket distances directly — no ATR fallback needed
116:         entry = signal.entry_price
117:         sl_distance = signal.raw_sl_distance
118:         tp_distance = signal.raw_tp_distance
119:         sl_price = entry - sl_distance
120:         tp_price = entry + tp_distance
121: 
122:         qty = self.risk_manager.calculate_quantity(
123:             equity, buying_power, entry, sl_price
124:         )
125: 
126:         if qty <= 0:
127:             return
128: 
129:         logger.info(
130:             f"Executing BUY for {symbol} | Qty: {qty} | SL: {sl_price} | TP: {tp_price}"
131:         )
132: 
133:         req = MarketOrderRequest(
134:             symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
135:         )
136: 
137:         try:
138:             order = await asyncio.to_thread(self.trading_client.submit_order, req)
139:             self.active_positions[symbol] = {
140:                 "sl": sl_price,
141:                 "tp": tp_price,
142:                 "qty": qty,
143:                 "order_id": order.id,
144:             }
145:         except Exception as e:
146:             logger.error(f"Entry failed for {symbol}: {e}")
147: 
148:     async def _watchdog_loop(self):
149:         """Polls active symbols and enforces SL/TP targets."""
150:         while not self._shutdown_event.is_set():
151:             await asyncio.sleep(1)
152: 
153:             for symbol, pos in list(self.active_positions.items()):
154:                 # Get last known price from aggregator
155:                 last_bar = self.aggregators[symbol].history_df.tail(1)
156:                 if last_bar.is_empty():
157:                     continue
158: 
159:                 price = last_bar["close"][0]
160: 
161:                 if price <= pos["sl"] or price >= pos["tp"]:
162:                     reason = "SL" if price <= pos["sl"] else "TP"
163:                     logger.warning(f"WATCHDOG: {reason} hit for {symbol} at {price}")
164:                     await self._execute_sell(symbol, pos["qty"])
165: 
166:     async def _execute_sell(self, symbol: str, qty: float):
167:         """Market close of a position."""
168:         req = MarketOrderRequest(
169:             symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC
170:         )
171:         try:
172:             await asyncio.to_thread(self.trading_client.submit_order, req)
173:             del self.active_positions[symbol]
174:             logger.info(f"Position closed for {symbol}")
175:         except Exception as e:
176:             logger.error(f"Exit failed for {symbol}: {e}")
```

## 3.2  Verbatim body of `_execute_buy`

Lines 108-146:
```
108:     async def _execute_buy(self, signal, symbol):
109:         """Calculates risk, submits order, and enters watchdog state."""
110:         # Get account for sizing
111:         account = await asyncio.to_thread(self.trading_client.get_account)
112:         equity = float(account.equity)
113:         buying_power = float(account.buying_power)
114: 
115:         # Signal now carries bracket distances directly — no ATR fallback needed
116:         entry = signal.entry_price
117:         sl_distance = signal.raw_sl_distance
118:         tp_distance = signal.raw_tp_distance
119:         sl_price = entry - sl_distance
120:         tp_price = entry + tp_distance
121: 
122:         qty = self.risk_manager.calculate_quantity(
123:             equity, buying_power, entry, sl_price
124:         )
125: 
126:         if qty <= 0:
127:             return
128: 
129:         logger.info(
130:             f"Executing BUY for {symbol} | Qty: {qty} | SL: {sl_price} | TP: {tp_price}"
131:         )
132: 
133:         req = MarketOrderRequest(
134:             symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
135:         )
136: 
137:         try:
138:             order = await asyncio.to_thread(self.trading_client.submit_order, req)
139:             self.active_positions[symbol] = {
140:                 "sl": sl_price,
141:                 "tp": tp_price,
142:                 "qty": qty,
143:                 "order_id": order.id,
144:             }
145:         except Exception as e:
146:             logger.error(f"Entry failed for {symbol}: {e}")
```

## 3.3  Identified lines in `_execute_buy`

a) Read `raw_sl_distance` from signal:
```
117:         sl_distance = signal.raw_sl_distance
```

b) Read `raw_tp_distance` from signal:
```
118:         tp_distance = signal.raw_tp_distance
```

c) Convert raw distances into absolute price levels:
```
119:         sl_price = entry - sl_distance
120:         tp_price = entry + tp_distance
```

d) Call `RiskManager.calculate_bracket`:
**Never called.** `factory_orchestrator.py` contains **zero** references to `calculate_bracket`. The orchestrator has taken over bracket math itself by reading `raw_sl_distance` and `raw_tp_distance` directly from the Signal and computing absolute prices on lines 119-120.

e) Call `RiskManager.calculate_quantity` — exact argument list:
```
122:         qty = self.risk_manager.calculate_quantity(
123:             equity, buying_power, entry, sl_price
124:         )
```
Variables passed: `equity`, `buying_power`, `entry`, `sl_price`.

f) Submit `MarketOrderRequest` — `qty` parameter:
```
133:         req = MarketOrderRequest(
134:             symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
135:         )
```
The value passed to `qty` is the variable `qty` returned by `calculate_quantity`.

g) Persist SL/TP into `self.active_positions`:
```
139:             self.active_positions[symbol] = {
140:                 "sl": sl_price,
141:                 "tp": tp_price,
142:                 "qty": qty,
143:                 "order_id": order.id,
144:             }
```
Values stored: `sl_price`, `tp_price`, `qty`, and `order.id`.

## 3.4  CRITICAL CROSS-CHECK: Does `_execute_buy` use BOTH or ONE?

**Reality: it uses ONE, but the constants exist in TWO places.**

- `RiskManager.calculate_bracket(entry_price, atr)` is **never invoked** inside `factory_orchestrator.py`.
- The orchestrator relies entirely on the `Signal` object's `raw_sl_distance` and `raw_tp_distance`, which were computed by `MLStrategy.generate_signals` using its own copies of `SL_ATR_MULTIPLIER`, `TP_ATR_MULTIPLIER`, and `MIN_SL_PCT`.
- Therefore there is **no double-counting** (the ATR multipliers and floor are not applied twice), but there **is** duplication of constants: identical values live in both `ml_strategy.py` (module-level constants) and `risk_manager.py` (`RiskProfile` defaults).

---

# SECTION 4 — RiskManager confirmation

## 4.1  Full file: `src/execution/risk_manager.py`

```
1: import logging
2: from dataclasses import dataclass
3: from typing import Optional
4: 
5: logger = logging.getLogger(__name__)
6: 
7: @dataclass
8: class RiskProfile:
9:     sl_atr_multiplier: float = 0.5
10:     tp_atr_multiplier: float = 3.0
11:     min_sl_pct: float = 0.0015  # 0.15% absolute floor
12:     risk_per_trade: float = 0.02 # 2% of account
13:     max_notional_cap: float = 100000.0
14: 
15: class RiskManager:
16:     """
17:     The Shield: Enforces institutional-grade safety nets and dynamic sizing.
18:     """
19:     def __init__(self, profile: RiskProfile = RiskProfile()):
20:         self.profile = profile
21: 
22:     def calculate_bracket(self, entry_price: float, atr: float) -> tuple[float, float]:
23:         """
24:         Calculates SL and TP with an absolute floor for SL distance.
25:         """
26:         # Dynamic 0.5x ATR sizing (multiplier from profile)
27:         raw_sl_dist = atr * self.profile.sl_atr_multiplier
28: 
29:         # 0.15% absolute Stop Loss floor
30:         min_sl_dist = entry_price * self.profile.min_sl_pct
31: 
32:         actual_sl_dist = max(raw_sl_dist, min_sl_dist)
33: 
34:         sl_price = round(entry_price - actual_sl_dist, 4)
35:         tp_price = round(entry_price + (atr * self.profile.tp_atr_multiplier), 4)
36: 
37:         return sl_price, tp_price
38: 
39:     def calculate_quantity(self, equity: float, buying_power: float, entry_price: float, sl_price: float) -> float:
40:         """
41:         Calculates fractional position size based on risk-per-trade.
42:         """
43:         risk_dollars = equity * self.profile.risk_per_trade
44:         risk_per_share = entry_price - sl_price
45: 
46:         if risk_per_share <= 0:
47:             return 0.0
48: 
49:         risk_qty = risk_dollars / risk_per_share
50:         notional_qty = self.profile.max_notional_cap / entry_price
51:         bp_qty = (buying_power * 0.95) / entry_price
52: 
53:         final_qty = min(risk_qty, notional_qty, bp_qty)
54: 
55:         if final_qty < risk_qty:
56:             logger.warning(
57:                 f"Quantity scaled down from {risk_qty:.4f} to {final_qty:.4f} to meet notional/bp limits."
58:             )
59: 
60:         return max(round(final_qty, 4), 0.0001)
```

## 4.2  RiskProfile constant values

Lines 9-13:
```
9:     sl_atr_multiplier: float = 0.5
10:     tp_atr_multiplier: float = 3.0
11:     min_sl_pct: float = 0.0015  # 0.15% absolute floor
12:     risk_per_trade: float = 0.02 # 2% of account
13:     max_notional_cap: float = 100000.0
```

## 4.3  `calculate_bracket` and `calculate_quantity` verbatim

`calculate_bracket` (lines 22-37):
```
22:     def calculate_bracket(self, entry_price: float, atr: float) -> tuple[float, float]:
23:         """
24:         Calculates SL and TP with an absolute floor for SL distance.
25:         """
26:         # Dynamic 0.5x ATR sizing (multiplier from profile)
27:         raw_sl_dist = atr * self.profile.sl_atr_multiplier
28: 
29:         # 0.15% absolute Stop Loss floor
30:         min_sl_dist = entry_price * self.profile.min_sl_pct
31: 
32:         actual_sl_dist = max(raw_sl_dist, min_sl_dist)
33: 
34:         sl_price = round(entry_price - actual_sl_dist, 4)
35:         tp_price = round(entry_price + (atr * self.profile.tp_atr_multiplier), 4)
36: 
37:         return sl_price, tp_price
```

`calculate_quantity` (lines 39-60):
```
39:     def calculate_quantity(self, equity: float, buying_power: float, entry_price: float, sl_price: float) -> float:
40:         """
41:         Calculates fractional position size based on risk-per-trade.
42:         """
43:         risk_dollars = equity * self.profile.risk_per_trade
44:         risk_per_share = entry_price - sl_price
45: 
46:         if risk_per_share <= 0:
47:             return 0.0
48: 
49:         risk_qty = risk_dollars / risk_per_share
50:         notional_qty = self.profile.max_notional_cap / entry_price
51:         bp_qty = (buying_power * 0.95) / entry_price
52: 
53:         final_qty = min(risk_qty, notional_qty, bp_qty)
54: 
55:         if final_qty < risk_qty:
56:             logger.warning(
57:                 f"Quantity scaled down from {risk_qty:.4f} to {final_qty:.4f} to meet notional/bp limits."
58:             )
59: 
60:         return max(round(final_qty, 4), 0.0001)
```

---

# SECTION 5 — ATR feature provenance

## 5.1  Grep results

Command:
```bash
grep -rn "atr_abs\|natr_14\|ATR\b" src/ml/features/ src/strategies/concrete_strategies/ml_strategy.py | head -50
```

Output:
```
src/ml/features/v3_features.py:31:    Produces: rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct,
src/ml/features/v3_features.py:62:        natr = talib.NATR(high, low, close, timeperiod=_NATR_PERIOD)
src/ml/features/v3_features.py:71:            pl.Series("natr_14", natr),
src/strategies/concrete_strategies/ml_strategy.py:43:SL_ATR_MULTIPLIER = 0.5  # SL distance = SL_ATR_MULTIPLIER * atr_abs
src/strategies/concrete_strategies/ml_strategy.py:44:TP_ATR_MULTIPLIER = 3.0  # TP distance = TP_ATR_MULTIPLIER * atr_abs
src/strategies/concrete_strategies/ml_strategy.py:147:            "natr_14",
src/strategies/concrete_strategies/ml_strategy.py:367:            # Both Angel and Devil agree — compute ATR-based brackets
src/strategies/concrete_strategies/ml_strategy.py:368:            natr_value = float(latest_features_df["natr_14"].to_numpy()[0])
src/strategies/concrete_strategies/ml_strategy.py:369:            # TA-Lib NATR is a percentage; convert to absolute ATR
src/strategies/concrete_strategies/ml_strategy.py:370:            atr_abs = (natr_value / 100.0) * current_price
src/strategies/concrete_strategies/ml_strategy.py:372:            sl_distance = SL_ATR_MULTIPLIER * atr_abs
src/strategies/concrete_strategies/ml_strategy.py:373:            tp_distance = TP_ATR_MULTIPLIER * atr_abs
src/strategies/concrete_strategies/ml_strategy.py:388:                f"ATR={atr_abs:.4f} | SL={sl_distance:.4f} | TP={tp_distance:.4f}"
src/strategies/concrete_strategies/ml_strategy.py:400:                    "atr_abs": atr_abs,
```

## 5.2  V3BaseFeatures column production

Relevant definition lines in `src/ml/features/v3_features.py`:

Class docstring (lines 28-36):
```
28: class V3BaseFeatures(BaseFeatureGenerator):
29:     """
30:     Computes 1m technical indicators via TA-Lib and Phase 5 Microstructure features.
31:     Produces: rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct,
32:               price_sma50_ratio, log_return, hour_of_day,
33:               dist_sma50, vol_rel,
34:               range_coil_10, bar_body_pct,
35:               bar_upper_wick_pct, bar_lower_wick_pct.
36:     """
```

NATR computation and Series creation (lines 61-72):
```
61:         # Universal Volatility
62:         natr = talib.NATR(high, low, close, timeperiod=_NATR_PERIOD)
63: 
64:         df = df.with_columns(
65:             pl.Series("rsi_14", rsi),
66:             pl.Series("ppo", ppo),
67:             pl.Series("bb_upper", bb_upper),
68:             pl.Series("bb_middle", bb_middle),
69:             pl.Series("bb_lower", bb_lower),
70:             pl.Series("sma_50", sma_50),
71:             pl.Series("natr_14", natr),
72:         )
```

There is **no `atr_abs` column** produced by `V3BaseFeatures`. The only ATR-related column is `natr_14`.

There is also **no `natr_14` percentage-normalized version** separate from the raw `natr_14`. The single `natr_14` column is the direct output of `talib.NATR`.

## 5.3  ATR units assessment

The code at `ml_strategy.py:368-370` reads:
```
366:             natr_value = float(latest_features_df["natr_14"].to_numpy()[0])
367:             # TA-Lib NATR is a percentage; convert to absolute ATR
368:             atr_abs = (natr_value / 100.0) * current_price
```

For the observed ETH trade (`entry ≈ 2279.87`, reported `ATR = 0.6398`):
- If `natr_14` were in **percentage points** (e.g., `natr_14 ≈ 0.028` meaning 0.028%), then `atr_abs = (0.028 / 100) * 2279.87 ≈ 0.638`, matching the observed `0.6398`.
- If `natr_14` were in **fractional form** (e.g., `natr_14 ≈ 0.028` meaning 2.8%), then `atr_abs = 0.00028 * 2279.87 ≈ 0.638` — same numeric result because the code divides by 100 regardless.
- If `natr_14` were in **price units** directly (e.g., `natr_14 ≈ 0.6398`), then dividing by 100 would yield `atr_abs ≈ 0.0064`, which does **not** match the reported `ATR=0.6398`.

Therefore, the code **treats `natr_14` as a percentage-like value that must be divided by 100**, and the resulting `atr_abs` is in **price units** (dollars). The extraordinarily small absolute ATR (~$0.64 on a ~$2,280 asset = ~0.028%) is consistent with 1-minute bar NATR during very low volatility.

---

# SECTION 6 — Position stacking guard (Finding 4)

## 6.1  `_on_tick` and `_watchdog_loop` verbatim

`_on_tick` (lines 85-106):
```
85:     async def _on_tick(self, tick: dict):
86:         """Routes a single bar event into the aggregator and strategy."""
87:         symbol = tick["symbol"]
88:         agg = self.aggregators[symbol]
89: 
90:         # logical clock alignment and aggregation
91:         if not agg.add_bar(tick):
92:             return  # Wait for bar to seal
93: 
94:         # Bar sealed (1m candle complete). Snapshot for inference.
95:         history = agg.history_df.clone()
96:         history = history.with_columns(pl.lit(symbol).alias("symbol"))
97: 
98:         # Offload CPU-bound ML inference to thread
99:         signal = await asyncio.to_thread(self.strategy.generate_signals, history)
100: 
101:         if signal is None:
102:             return
103: 
104:         # Execute if FLAT
105:         if symbol not in self.active_positions:
106:             await self._execute_buy(signal, symbol)
```

`_watchdog_loop` (lines 148-164):
```
148:     async def _watchdog_loop(self):
149:         """Polls active symbols and enforces SL/TP targets."""
150:         while not self._shutdown_event.is_set():
151:             await asyncio.sleep(1)
152: 
153:             for symbol, pos in list(self.active_positions.items()):
154:                 # Get last known price from aggregator
155:                 last_bar = self.aggregators[symbol].history_df.tail(1)
156:                 if last_bar.is_empty():
157:                     continue
158: 
159:                 price = last_bar["close"][0]
160: 
161:                 if price <= pos["sl"] or price >= pos["tp"]:
162:                     reason = "SL" if price <= pos["sl"] else "TP"
163:                     logger.warning(f"WATCHDOG: {reason} hit for {symbol} at {price}")
164:                     await self._execute_sell(symbol, pos["qty"])
```

## 6.2  `active_positions` lifecycle trace

a) **Set** — after successful order submission in `_execute_buy`:
```
138:             order = await asyncio.to_thread(self.trading_client.submit_order, req)
139:             self.active_positions[symbol] = {
140:                 "sl": sl_price,
141:                 "tp": tp_price,
142:                 "qty": qty,
143:                 "order_id": order.id,
144:             }
```
Note: the dict is populated **only if** `submit_order` succeeds. If `submit_order` raises, the exception is caught on line 145 and `active_positions` is **not** updated.

b) **Removed** — after watchdog triggers `_execute_sell`:
```
171:         try:
172:             await asyncio.to_thread(self.trading_client.submit_order, req)
173:             del self.active_positions[symbol]
174:             logger.info(f"Position closed for {symbol}")
175:         except Exception as e:
176:             logger.error(f"Exit failed for {symbol}: {e}")
```
Note: the entry is deleted **only if** the sell order submission succeeds. If it fails, `active_positions[symbol]` remains, and the watchdog will continue to evaluate it on the next 1-second tick.

c) **Lock / atomic guard:**
There is **no lock, mutex, or asyncio.Lock** protecting `self.active_positions`. The file contains no `asyncio.Lock`, no `threading.Lock`, and no `with` block guarding dict mutations.

Concurrency scenario:
- `_on_tick` runs in the asyncio event loop (called from `feed.subscribe` callback).
- `_watchdog_loop` is a separate `asyncio.create_task` running concurrently in the same event loop.
- Both are async coroutines, so they are single-threaded with respect to each other (no true parallelism), but they can interleave at every `await` boundary.
- `_execute_buy` contains `await asyncio.to_thread(...)` calls (lines 111, 138). During these awaits, the event loop can switch to `_watchdog_loop`.
- `_watchdog_loop` contains `await asyncio.sleep(1)` (line 151) and `await self._execute_sell(...)` (line 164), allowing interleaving with `_on_tick`.

Therefore, **concurrent interleaving is possible** at `await` boundaries, but because asyncio is cooperative single-threaded, dict mutations themselves are not subject to data races in the traditional sense. However, **logical races** exist: e.g., `_on_tick` could pass the `symbol not in self.active_positions` check, then yield during `get_account` or `submit_order`, while the watchdog simultaneously detects SL/TP and calls `_execute_sell`, deleting the key. When `_execute_buy` resumes, it would overwrite `active_positions[symbol]` with a new live entry even though a sell was just submitted.

---

# SECTION 7 — join_asof warning (Finding 3)

## 7.1  Context around line 274

Viewed `src/ml/features/v3_features.py` lines 240-310 (full contents already provided in initial read; lines 240-310 are reproduced below):

```
240:             raise ValueError(
241:                 f"Invalid htf_timeframe format '{self.timeframe}'. "
242:                 "Expected format: '<N>m', '<N>h', or '<N>d' (e.g. '5m')."
243:             )
244:         value, unit = int(match.group(1)), match.group(2)
245:         td = timedelta(
246:             minutes=value if unit == "m" else 0,
247:             hours=value if unit == "h" else 0,
248:             days=value if unit == "d" else 0,
249:         )
250: 
251:         htf_bars = htf_bars.with_columns(
252:             (pl.col("timestamp") + td).alias("available_at")
253:         )
254: 
255:         # ── 5. Select join columns ───────────────────────────────────────────
256:         join_cols = [
257:             "available_at",
258:             "htf_rsi_14",
259:             "_htf_sma_50",
260:             "htf_vol_rel",
261:             "htf_bb_pct_b",
262:         ]
263:         if has_symbol:
264:             join_cols = ["symbol"] + join_cols
265: 
266:         htf_features = htf_bars.select(join_cols).sort(
267:             ["symbol", "available_at"] if has_symbol else "available_at"
268:         )
269: 
270:         # ── 6. join_asof (backward)
271:         df_sorted = df.sort(["symbol", "timestamp"] if has_symbol else "timestamp")
272: 
273:         if has_symbol:
274:             df_sorted = df_sorted.join_asof(
275:                 htf_features,
276:                 left_on="timestamp",
277:                 right_on="available_at",
278:                 by="symbol",
279:                 strategy="backward",
280:             )
281:         else:
282:             df_sorted = df_sorted.join_asof(
283:                 htf_features,
284:                 left_on="timestamp",
285:                 right_on="available_at",
286:                 strategy="backward",
287:             )
288: 
289:         # ── 7. htf_trend_agreement ───────────────────────────────────────────
290:         df_sorted = df_sorted.with_columns(
291:             pl.when(pl.col("_htf_sma_50").is_null() | pl.col("_htf_sma_50").is_nan())
292:             .then(pl.lit(0, dtype=pl.Int8))
293:             .when(pl.col("close") > pl.col("_htf_sma_50"))
294:             .then(pl.lit(1, dtype=pl.Int8))
295:             .otherwise(pl.lit(-1, dtype=pl.Int8))
296:             .alias("htf_trend_agreement")
297:         )
298: 
299:         # ── 8. Drop all intermediate columns ────────────────────────────────
300:         drop_cols = [
301:             "_htf_sma_50",
302:             "_htf_bb_upper",
303:             "_htf_bb_lower",
304:             "_htf_bb_middle",
305:             "available_at",
306:         ]
307:         existing_drops = [c for c in drop_cols if c in df_sorted.columns]
308:         df_sorted = df_sorted.drop(existing_drops)
309: 
310:         return df_sorted
```

## 7.2  Quoted items

a) Full `join_asof` call (lines 274-280):
```
274:             df_sorted = df_sorted.join_asof(
275:                 htf_features,
276:                 left_on="timestamp",
277:                 right_on="available_at",
278:                 by="symbol",
279:                 strategy="backward",
280:             )
```

b) Lines immediately before that prepare/sort the DataFrames:
```
266:         htf_features = htf_bars.select(join_cols).sort(
267:             ["symbol", "available_at"] if has_symbol else "available_at"
268:         )
269: 
270:         # ── 6. join_asof (backward)
271:         df_sorted = df.sort(["symbol", "timestamp"] if has_symbol else "timestamp")
272: 
273:         if has_symbol:
```

c) Is either input explicitly `.sort()`'d on the join key before the call?
- **`df_sorted`** — yes. Line 271 sorts it on `["symbol", "timestamp"]` (or `"timestamp"` if no symbol).
- **`htf_features`** — yes. Lines 266-268 sort it on `["symbol", "available_at"]` (or `"available_at"` if no symbol).

Both sides are explicitly sorted on their respective join keys (`timestamp` and `available_at`) **and** on the `by` key (`symbol`). Despite this, Polars emits the `UserWarning` about sortedness when `by` groups are provided.

---

# SECTION 8 — Stale Signal API check

## 8.1  Old Signal API usage grep

Command:
```bash
grep -rn "\.symbol\b\|\.price\b\|\.confidence\b\|\.signal_type\b" \
  src/execution/factory_orchestrator.py \
  src/strategies/concrete_strategies/ml_strategy.py \
  src/strategies/concrete_strategies/ml_factory_strategy.py
```

Output:
```
```
(No output — no matches in any of the three files.)

## 8.2  Old Signal module imports

Command:
```bash
grep -rn "from core.signal\|from src.core.signal" src/ *.py
```

Output:
```
src/execution/live_orchestrator.py:121:from core.signal import Signal, SignalType
src/core/notification_manager.py:18:        from core.signal import (
backtest_ml_strategy.py:14:from core.signal import Signal
backtest_ml_strategy_quick.py:13:from core.signal import Signal
backtest_quick.py:17:from core.signal import Signal
```

Files still depending on the **old** `core.signal` module:
- `src/execution/live_orchestrator.py`
- `src/core/notification_manager.py`
- `backtest_ml_strategy.py`
- `backtest_ml_strategy_quick.py`
- `backtest_quick.py`

The **factory execution path** (`factory_orchestrator.py`, `ml_strategy.py`, `ml_factory_strategy.py`) has **zero** dependencies on the old Signal API.

---

# DISCREPANCIES

1. **Memory said:** "RiskManager.calculate_bracket … it INTERNALLY applies SL_ATR_MULT, TP_ATR_MULT, and the MIN_SL_PCT floor — returning (sl_price, tp_price) as ABSOLUTE PRICES."
   **Reality on disk is:** `calculate_bracket` does indeed do this, but `factory_orchestrator._execute_buy` **never calls it**. The bracket math is performed entirely inside `MLStrategy.generate_signals`, and the orchestrator trusts the Signal's `raw_sl_distance` / `raw_tp_distance` fields. There is no double-counting, but there **is** constant duplication across two files.

2. **Memory said:** The old `core.signal.Signal` had fields `(symbol, type, price, confidence, timestamp, metadata)` and the new `strategies.base.Signal` has `(direction, entry_price, raw_sl_distance, raw_tp_distance, metadata)`. The prompt asked to verify that MLStrategy "fully migrated."
   **Reality on disk is:** The factory execution path (`factory_orchestrator.py`, `ml_strategy.py`, `ml_factory_strategy.py`) is fully migrated — zero references to `.symbol`, `.price`, `.confidence`, or `.signal_type` on Signal objects. However, `live_orchestrator.py`, `notification_manager.py`, and several backtest scripts still import from `core.signal`.

3. **Memory said:** For Finding 4, "Need to verify _on_tick's active_positions guard is working as intended."
   **Reality on disk is:** The guard (`if symbol not in self.active_positions`) exists on line 105, but there is **no lock** around the subsequent `_execute_buy` body or around the watchdog's `_execute_sell` / `del self.active_positions[symbol]`. Because `_execute_buy` contains multiple `await` points (`asyncio.to_thread` calls), a logical race is possible where the watchdog removes the key after the guard passes but before the buy order completes, leading to an overwrite of `active_positions` with a new live position after a sell was already issued.

4. **Memory said:** The join_asof warning might be due to unsorted inputs.
   **Reality on disk is:** Both input DataFrames (`df_sorted` and `htf_features`) **are** explicitly `.sort()`'d on their join keys and on the `by` column before `join_asof` is called. The warning persists despite explicit sorting, which is a Polars behavior when `by` is provided.
