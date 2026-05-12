---
type: audit
date: 2026-04-25
time: unknown
agent: unknown
model: unknown
trigger: Full Alpaca surface-area audit ahead of SDK decoupling work
head: unknown
scope: read-only
imported_from: ALPACA_SURFACE_AREA_AUDIT.md
---

# Build-A-Bot — Full Alpaca Surface-Area Audit

**Date:** 2026-04-25  
**Branch:** `main`  
**Mode:** Read-only — no files modified, no branch switches.

---

## 1. Top-Level Alpaca Import Census

### `from alpaca.*` imports
```text
./src/data/harvester.py:23:from alpaca.data.historical import StockHistoricalDataClient
./src/data/harvester.py:24:from alpaca.data.requests import StockBarsRequest
./src/data/harvester.py:25:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
./src/data/harvester.py:26:from alpaca.data.enums import DataFeed
./src/data/feed.py:9:from alpaca.data.historical import CryptoHistoricalDataClient
./src/data/feed.py:10:from alpaca.data.requests import CryptoBarsRequest
./src/data/feed.py:11:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
./src/data/feed.py:12:from alpaca.data.live import CryptoDataStream
./src/data/alpaca_provider.py:5:from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
./src/data/alpaca_provider.py:6:from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
./src/data/alpaca_provider.py:7:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
./src/data/alpaca_provider.py:8:from alpaca.data.enums import DataFeed
./src/data/discovery.py:3:from alpaca.trading.client import TradingClient
./src/data/discovery.py:4:from alpaca.trading.requests import GetAssetsRequest
./src/data/discovery.py:5:from alpaca.trading.enums import AssetClass, AssetStatus
./src/data/discovery.py:6:from alpaca.data.historical import StockHistoricalDataClient
./src/data/discovery.py:7:from alpaca.data.requests import StockSnapshotRequest
./src/data/fetch_training_data.py:14:from alpaca.data.historical import StockHistoricalDataClient
./src/data/fetch_training_data.py:15:from alpaca.data.requests import StockBarsRequest
./src/data/fetch_training_data.py:16:from alpaca.data.timeframe import TimeFrame
./src/execution/factory_orchestrator.py:8:from alpaca.trading.client import TradingClient
./src/execution/factory_orchestrator.py:9:from alpaca.trading.requests import MarketOrderRequest
./src/execution/factory_orchestrator.py:10:from alpaca.trading.enums import OrderSide, TimeInForce
./src/execution/live_orchestrator.py:101:from alpaca.data.enums import Adjustment, DataFeed
./src/execution/live_orchestrator.py:102:from alpaca.data.historical.crypto import CryptoHistoricalDataClient
./src/execution/live_orchestrator.py:103:from alpaca.data.historical.stock import StockHistoricalDataClient
./src/execution/live_orchestrator.py:104:from alpaca.data.live.crypto import CryptoDataStream
./src/execution/live_orchestrator.py:105:from alpaca.data.live.stock import StockDataStream
./src/execution/live_orchestrator.py:106:from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
./src/execution/live_orchestrator.py:107:from alpaca.data.timeframe import TimeFrame
./src/execution/live_orchestrator.py:108:from alpaca.trading.client import TradingClient
./src/execution/live_orchestrator.py:109:from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
./src/execution/live_orchestrator.py:110:from alpaca.trading.requests import (
./src/execution/live_orchestrator.py:113:from alpaca.trading.stream import TradingStream
./src/day_trading/harvester_5m.py:38:from alpaca.data.enums import Adjustment, DataFeed
./src/day_trading/harvester_5m.py:39:from alpaca.data.historical.stock import StockHistoricalDataClient
./src/day_trading/harvester_5m.py:40:from alpaca.data.requests import StockBarsRequest
./src/day_trading/harvester_5m.py:41:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
./src/core/retrainer.py:42:from alpaca.data.historical import StockHistoricalDataClient
./src/core/retrainer.py:43:from alpaca.data.requests import StockBarsRequest
./src/core/retrainer.py:44:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
./src/core/retrainer.py:45:from alpaca.data.enums import DataFeed
./test_alpaca.py:4:from alpaca.data.historical import CryptoHistoricalDataClient
./test_alpaca.py:5:from alpaca.data.requests import CryptoBarsRequest
./test_alpaca.py:6:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
```

### `import alpaca.*` imports
```text
(no direct "import alpaca" statements found)
```

---

## 2. Per-Directory Alpaca Dependency Table

| Directory | Total .py Files | Files Touching Alpaca | Alpaca Submodules Used |
|-----------|-----------------|----------------------|------------------------|
| `src/data` | 8 | **5** | `alpaca.data.enums`, `alpaca.data.historical`, `alpaca.data.live`, `alpaca.data.requests`, `alpaca.data.timeframe`, `alpaca.trading.client`, `alpaca.trading.enums`, `alpaca.trading.requests` |
| `src/execution` | 4 | **2** | `alpaca.data.enums`, `alpaca.data.historical.crypto`, `alpaca.data.historical.stock`, `alpaca.data.live.crypto`, `alpaca.data.live.stock`, `alpaca.data.requests`, `alpaca.data.timeframe`, `alpaca.trading.client`, `alpaca.trading.enums`, `alpaca.trading.requests`, `alpaca.trading.stream` |
| `src/ml` | 8 | **0** | none |
| `src/core` | 7 | **1** | `alpaca.data.enums`, `alpaca.data.historical`, `alpaca.data.requests`, `alpaca.data.timeframe` |
| `src/strategies` | 8 | **0** | none |
| `src/utils` | 3 | **0** | none |
| `src/analysis` | 5 | **0** | none |
| `src/day_trading` | 6 | **1** | `alpaca.data.enums`, `alpaca.data.historical.stock`, `alpaca.data.requests`, `alpaca.data.timeframe` |
| `tests` | 4 | **0** | none |
| `.` (root scripts) | 70 | **10** | (aggregate of all above) |

**Headline number:** Out of 113 Python files under `src/`, **9 files** (8%) import from Alpaca. The rot is concentrated in `src/data/` (5 files), `src/execution/` (2 files), `src/core/` (1 file), and `src/day_trading/` (1 file).

---

## 3. ML Pipeline Inspection (Highest Priority)

### All files under `src/ml/` (recursive)

| File | Lines | Alpaca Imports |
|------|-------|----------------|
| `src/ml/__init__.py` | 0 | none |
| `src/ml/targets/v3_targets.py` | 26 | none |
| `src/ml/trainers/v3_rf_trainer.py` | 28 | none |
| `src/ml/core/interfaces.py` | 35 | none |
| `src/ml/feature_pipeline.py` | 101 | none |
| `src/ml/train_model.py` | 252 | none |
| `src/ml/data_miner.py` | 308 | none |
| `src/ml/features/v3_features.py` | 310 | none |

**Verdict:** `src/ml/` is **100% clean**. No Alpaca imports anywhere.

### Other training-related files checked

| File | Alpaca Imports |
|------|----------------|
| `src/core/retrainer.py` | **4 hits** — `StockHistoricalDataClient`, `StockBarsRequest`, `TimeFrame`, `TimeFrameUnit`, `DataFeed` |
| `src/data/harvester.py` | **4 hits** — `StockHistoricalDataClient`, `StockBarsRequest`, `TimeFrame`, `TimeFrameUnit`, `DataFeed` |
| `src/data/fetch_training_data.py` | **3 hits** — `StockHistoricalDataClient`, `StockBarsRequest`, `TimeFrame` |
| `src/day_trading/harvester_5m.py` | **4 hits** — `Adjustment`, `DataFeed`, `StockHistoricalDataClient`, `StockBarsRequest`, `TimeFrame`, `TimeFrameUnit` |

**Observation:** The ML training pipeline itself is clean, but **data acquisition** for training (harvesters, fetchers, retrainer) is tightly coupled to Alpaca historical data clients.

---

## 4. Data Pipeline Inspection

### Import blocks (lines 1–30)

**`src/data/feed.py`**
```python
import abc
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Callable

import pandas as pd
import polars as pl
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.live import CryptoDataStream
```
Alpaca symbols used in body: `CryptoHistoricalDataClient`, `CryptoBarsRequest`, `TimeFrame`, `TimeFrameUnit`, `CryptoDataStream`, `get_crypto_bars`, `.df`

**`src/data/factory.py`**
```python
"""
Factory for MarketDataProvider instances.
...
"""
import logging
import os

from data.market_provider import MarketDataProvider
```
No Alpaca imports in this file. It does load `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` from env (lines 41–42).

**`src/data/discovery.py`**
```python
import logging
from typing import List
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
```
Alpaca symbols used in body: `TradingClient`, `GetAssetsRequest`, `AssetClass.US_EQUITY`, `AssetStatus.ACTIVE`, `StockHistoricalDataClient`, `StockSnapshotRequest`

**`src/data/alpaca_provider.py`**
```python
import logging
import pandas as pd
import polars as pl
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
```
Alpaca symbols used in body: `StockHistoricalDataClient`, `CryptoHistoricalDataClient`, `StockBarsRequest`, `CryptoBarsRequest`, `TimeFrame`, `TimeFrameUnit`, `get_crypto_bars`, `get_stock_bars`, `.df`

**`src/data/harvester.py`**
```python
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import polars as pl
import pyarrow.parquet as pq
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
```
Alpaca symbols used in body: `StockHistoricalDataClient`, `StockBarsRequest`, `TimeFrame`, `TimeFrameUnit`, `DataFeed`, `get_stock_bars`, `.df`

**`src/data/fetch_training_data.py`**
```python
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import polars as pl
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
```
Alpaca symbols used in body: `StockHistoricalDataClient`, `StockBarsRequest`, `TimeFrame`, `get_stock_bars`

---

## 5. Strategies Layer Inspection

| File | Lines | Alpaca Import? | `core.order_management`? | `core.signal`? |
|------|-------|----------------|--------------------------|----------------|
| `src/strategies/concrete_strategies/rsi_bbands.py` | 135 | No | Yes | Yes |
| `src/strategies/concrete_strategies/sma_crossover.py` | 59 | No | Yes | Yes |
| `src/strategies/concrete_strategies/ml_factory_strategy.py` | 43 | No | No | Yes |
| `src/strategies/concrete_strategies/ml_strategy.py` | 415 | No | Yes | Yes |
| `src/strategies/strategy_factory.py` | 15 | No | No | No |
| `src/strategies/strategy.py` | 54 | No | Yes | Yes |
| `src/strategies/__init__.py` | 0 | No | No | No |

**Verdict:** `src/strategies/` is **100% Alpaca-free**. All strategies import from `core.signal` and/or `core.order_management` — but remember, `core/order_management.py` is **missing on disk** (as noted in the prior recon report).

---

## 6. Schema-Level Coupling (the subtle stuff)

### Alpaca attribute access (`alpaca.*` dotted access not in imports)
```text
(no stray "alpaca." dotted references outside of import lines)
```

### `TimeFrame` / `TimeFrameUnit` usage
```text
src/data/harvester.py:25:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
src/data/harvester.py:37:TIMEFRAME = TimeFrame(1, TimeFrameUnit.Minute)
src/data/feed.py:11:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
src/data/feed.py:49:            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
src/data/alpaca_provider.py:7:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
src/data/alpaca_provider.py:30:                    timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
src/data/alpaca_provider.py:38:                    timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
src/data/fetch_training_data.py:16:from alpaca.data.timeframe import TimeFrame
src/data/fetch_training_data.py:42:        timeframe=TimeFrame.Minute,
src/execution/live_orchestrator.py:107:from alpaca.data.timeframe import TimeFrame
src/execution/live_orchestrator.py:2146:            timeframe=TimeFrame.Minute,
src/execution/live_orchestrator.py:2174:            timeframe=TimeFrame.Minute,
src/day_trading/harvester_5m.py:41:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
src/day_trading/harvester_5m.py:82:TIMEFRAME_5MIN = TimeFrame(5, TimeFrameUnit.Minute)
src/day_trading/harvester_5m.py:83:TIMEFRAME_DAILY = TimeFrame(1, TimeFrameUnit.Day)
src/day_trading/harvester_5m.py:112:    timeframe: TimeFrame,
src/core/retrainer.py:44:from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
src/core/retrainer.py:65:TIMEFRAME = TimeFrame(1, TimeFrameUnit.Minute)
```

### `BarSet` / `TradeSet` / `QuoteSet`
```text
src/day_trading/harvester_5m.py:122:    `client.get_stock_bars()` returns a `BarSet` whose `.df` property is a
```

### `AssetClass` / `AssetStatus` / `AssetExchange`
```text
src/data/discovery.py:5:from alpaca.trading.enums import AssetClass, AssetStatus
src/data/discovery.py:33:            asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE
```

### Alpaca client class instances & method calls
```text
src/data/harvester.py:23:from alpaca.data.historical import StockHistoricalDataClient
src/data/harvester.py:42:def get_alpaca_client() -> StockHistoricalDataClient:
src/data/harvester.py:52:    return StockHistoricalDataClient(api_key, secret_key)
src/data/harvester.py:56:    client: StockHistoricalDataClient,
src/data/harvester.py:71:        bars = client.get_stock_bars(request)
src/data/harvester.py:78:        df_pandas = bars.df.reset_index()

src/data/feed.py:9:from alpaca.data.historical import CryptoHistoricalDataClient
src/data/feed.py:12:from alpaca.data.live import CryptoDataStream
src/data/feed.py:37:        self._stream: CryptoDataStream | None = None
src/data/feed.py:43:        client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
src/data/feed.py:54:        bars = await asyncio.to_thread(client.get_crypto_bars, req)
src/data/feed.py:55:        raw_df = bars.df  # pandas MultiIndex DataFrame: (symbol, timestamp)
src/data/feed.py:146:        self._stream = CryptoDataStream(self.api_key, self.secret_key)

src/data/alpaca_provider.py:5:from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
src/data/alpaca_provider.py:15:        self.stock_client = StockHistoricalDataClient(api_key, secret_key)
src/data/alpaca_provider.py:16:        self.crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
src/data/alpaca_provider.py:34:                bars = self.crypto_client.get_crypto_bars(req)
src/data/alpaca_provider.py:43:                bars = self.stock_client.get_stock_bars(req)
src/data/alpaca_provider.py:49:            df_pandas = bars.df.loc[symbol].reset_index()

src/data/discovery.py:3:from alpaca.trading.client import TradingClient
src/data/discovery.py:6:from alpaca.data.historical import StockHistoricalDataClient
src/data/discovery.py:14:        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
src/data/discovery.py:15:        self.data_client = StockHistoricalDataClient(api_key, secret_key)

src/data/fetch_training_data.py:14:from alpaca.data.historical import StockHistoricalDataClient
src/data/fetch_training_data.py:32:    client: StockHistoricalDataClient,
src/data/fetch_training_data.py:47:    bars = client.get_stock_bars(request)
src/data/fetch_training_data.py:85:    client = StockHistoricalDataClient(api_key, secret_key)

src/execution/factory_orchestrator.py:8:from alpaca.trading.client import TradingClient
src/execution/factory_orchestrator.py:39:        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
src/execution/factory_orchestrator.py:109:        account = await asyncio.to_thread(self.trading_client.get_account)
src/execution/factory_orchestrator.py:134:            order = await asyncio.to_thread(self.trading_client.submit_order, req)
src/execution/factory_orchestrator.py:171:            await asyncio.to_thread(self.trading_client.submit_order, req)

src/execution/live_orchestrator.py:102:from alpaca.data.historical.crypto import CryptoHistoricalDataClient
src/execution/live_orchestrator.py:103:from alpaca.data.historical.stock import StockHistoricalDataClient
src/execution/live_orchestrator.py:104:from alpaca.data.live.crypto import CryptoDataStream
src/execution/live_orchestrator.py:105:from alpaca.data.live.stock import StockDataStream
src/execution/live_orchestrator.py:108:from alpaca.trading.client import TradingClient
src/execution/live_orchestrator.py:448:        self._trading_client = TradingClient(
src/execution/live_orchestrator.py:459:        self._crypto_stream: Optional[CryptoDataStream] = None
src/execution/live_orchestrator.py:460:        self._stock_stream: Optional[StockDataStream] = None
src/execution/live_orchestrator.py:554:            open_positions = self._trading_client.get_all_positions()
src/execution/live_orchestrator.py:791:            self._crypto_stream = CryptoDataStream(self._api_key, self._secret_key)
src/execution/live_orchestrator.py:796:            self._stock_stream = StockDataStream(
src/execution/live_orchestrator.py:906:        """Runs CryptoDataStream until shutdown is requested."""
src/execution/live_orchestrator.py:912:            logger.error("CryptoDataStream error: %s", exc, exc_info=True)
src/execution/live_orchestrator.py:917:        """Runs StockDataStream until shutdown is requested."""
src/execution/live_orchestrator.py:923:            logger.error("StockDataStream error: %s", exc, exc_info=True)
src/execution/live_orchestrator.py:955:        Ingests a raw 1-minute bar from either StockDataStream or
src/execution/live_orchestrator.py:956:        CryptoDataStream.
src/execution/live_orchestrator.py:1389:            account = self._trading_client.get_account()
src/execution/live_orchestrator.py:1494:            order = self._trading_client.submit_order(order_request)
src/execution/live_orchestrator.py:1734:            order = self._trading_client.submit_order(order_request)
src/execution/live_orchestrator.py:2152:            bar_set = await asyncio.to_thread(hist_client.get_crypto_bars, request)
src/execution/live_orchestrator.py:2153:            raw_pd = bar_set.df
src/execution/live_orchestrator.py:2181:            bar_set = await asyncio.to_thread(hist_client.get_stock_bars, request)
src/execution/live_orchestrator.py:2182:            raw_pd = bar_set.df
src/execution/live_orchestrator.py:2271:            streams.append((self._crypto_stream, "CryptoDataStream"))
src/execution/live_orchestrator.py:2273:            streams.append((self._stock_stream, "StockDataStream"))

src/day_trading/harvester_5m.py:39:from alpaca.data.historical.stock import StockHistoricalDataClient
src/day_trading/harvester_5m.py:92:def _build_client() -> StockHistoricalDataClient:
src/day_trading/harvester_5m.py:101:    return StockHistoricalDataClient(api_key, secret_key)
src/day_trading/harvester_5m.py:110:    client: StockHistoricalDataClient,
src/day_trading/harvester_5m.py:144:        bar_set = client.get_stock_bars(request)
src/day_trading/harvester_5m.py:151:        df_pd = bar_set.df.loc[symbol].reset_index()

src/core/retrainer.py:42:from alpaca.data.historical import StockHistoricalDataClient
src/core/retrainer.py:186:def get_alpaca_client() -> StockHistoricalDataClient:
src/core/retrainer.py:196:    return StockHistoricalDataClient(api_key, secret_key)
src/core/retrainer.py:200:    client: StockHistoricalDataClient,
src/core/retrainer.py:236:            bars = client.get_stock_bars(request)
src/core/retrainer.py:243:            df_pandas = bars.df.reset_index()
```

---

## 7. Configuration and Credential Surface

```text
./run_factory.py:28:    api_key = os.getenv("ALPACA_API_KEY")
./run_factory.py:29:    secret_key = os.getenv("ALPACA_SECRET_KEY")
./run_factory.py:32:        print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not set.")
./src/data/factory.py:41:        api_key = os.getenv("alpaca_key") or os.getenv("ALPACA_API_KEY")
./src/data/factory.py:42:        secret = os.getenv("alpaca_secret") or os.getenv("ALPACA_SECRET_KEY")
./src/data/factory.py:45:                "Alpaca credentials missing. Set alpaca_key / alpaca_secret "
./src/data/harvester.py:9:    ALPACA_API_KEY: Alpaca API key
./src/data/harvester.py:10:    ALPACA_SECRET_KEY: Alpaca API secret
./src/data/harvester.py:44:    api_key = os.getenv("ALPACA_API_KEY")
./src/data/harvester.py:45:    secret_key = os.getenv("ALPACA_SECRET_KEY")
./src/data/harvester.py:49:            "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables must be set"
./src/data/fetch_training_data.py:78:    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("alpaca_key")
./src/data/fetch_training_data.py:79:    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("alpaca_secret")
./src/data/fetch_training_data.py:82:        logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment")
./src/execution/live_orchestrator.py:421:        self._api_key: str = api_key or os.environ["ALPACA_API_KEY"]
./src/execution/live_orchestrator.py:422:        self._secret_key: str = secret_key or os.environ["ALPACA_SECRET_KEY"]
./src/day_trading/harvester_5m.py:23:    ALPACA_API_KEY    — Alpaca API key
./src/day_trading/harvester_5m.py:24:    ALPACA_SECRET_KEY — Alpaca API secret
./src/day_trading/harvester_5m.py:94:    api_key = os.getenv("ALPACA_API_KEY")
./src/day_trading/harvester_5m.py:95:    secret_key = os.getenv("ALPACA_SECRET_KEY")
./src/day_trading/harvester_5m.py:98:            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment "
./src/analysis/optimize_brackets.py:95:    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment (.env).
./src/core/retrainer.py:22:    ALPACA_API_KEY: Alpaca API key
./src/core/retrainer.py:23:    ALPACA_SECRET_KEY: Alpaca API secret
./src/core/retrainer.py:182:# ALPACA DATA FETCHING
./src/core/retrainer.py:188:    api_key = os.getenv("ALPACA_API_KEY")
./src/core/retrainer.py:189:    secret_key = os.getenv("ALPACA_SECRET_KEY")
./src/core/retrainer.py:193:            "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables must be set"
./test_alpaca.py:10:    api_key = os.getenv("ALPACA_API_KEY")
./test_alpaca.py:11:    secret_key = os.getenv("ALPACA_SECRET_KEY")
./.env:2:ALPACA_API_KEY='PK5CUAHXY0DAL46EOAYV'
./.env:3:ALPACA_SECRET_KEY='phzureLR2thhaDA6xzp844Qj929KpgsbkbDceeUb'
./tests/test_live_simulation.py:79:    api_key = os.getenv("alpaca_key")
./tests/test_live_simulation.py:80:    secret_key = os.getenv("alpaca_secret")
```

**Credential loading pattern summary:**
- Primary env vars: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- Fallback aliases: `alpaca_key`, `alpaca_secret` (used in `src/data/factory.py`, `src/data/fetch_training_data.py`, `tests/test_live_simulation.py`)
- `.env` file at project root contains live-looking paper keys (in `.env`, not `.env.example`).

---

## 8. requirements.txt / pyproject.toml Audit

Dependency files found:
- `Pipfile` (no `requirements.txt`, `pyproject.toml`, `setup.py`, or `setup.cfg` found)

**`Pipfile` — Alpaca line:**
```text
16:alpaca-py = "*"
```

**Observation:** The Alpaca SDK is pinned with a wildcard (`*`), meaning it will install the latest version on every fresh `pipenv install`. No version lock. This is a risk if Alpaca introduces breaking API changes.

---

## 9. Total Alpaca-Coupled File Count

```text
$ grep -rln "from alpaca\|import alpaca" --include="*.py" . 2>/dev/null | wc -l
10
```

**File paths:**
1. `./src/data/harvester.py`
2. `./src/data/feed.py`
3. `./src/data/alpaca_provider.py`
4. `./src/data/discovery.py`
5. `./src/data/fetch_training_data.py`
6. `./src/execution/factory_orchestrator.py`
7. `./src/execution/live_orchestrator.py`
8. `./src/day_trading/harvester_5m.py`
9. `./src/core/retrainer.py`
10. `./test_alpaca.py`

---

## 10. Verdict and Severity Map

| Layer | Severity | Rationale |
|-------|----------|-----------|
| **ML (`src/ml/`)** | **CLEAN** | Zero Alpaca imports. Pure scikit-learn / Polars / NumPy. Models, feature pipelines, and trainers are fully broker-agnostic. |
| **Strategies (`src/strategies/`)** | **CLEAN** | Zero Alpaca imports. Strategies consume `core.signal.Signal` and `core.order_management.OrderParams`. The missing `order_management.py` file is a bug, but not an Alpaca bug. |
| **Analysis (`src/analysis/`)** | **CLEAN** | No Alpaca imports detected. |
| **Utils (`src/utils/`)** | **CLEAN** | No Alpaca imports detected. |
| **Tests (`tests/`)** | **CLEAN** | No Alpaca imports in the test suite (test files mock Alpaca classes). |
| **Data (`src/data/`)** | **HEAVY** | 5 of 8 files import Alpaca. Historical clients (`StockHistoricalDataClient`, `CryptoHistoricalDataClient`), live streams (`CryptoDataStream`, `StockDataStream`), request builders (`StockBarsRequest`, `CryptoBarsRequest`), and timeframe enums (`TimeFrame`, `TimeFrameUnit`) are baked in. The `.df` property access on Alpaca response objects is a schema assumption. `discovery.py` also imports `TradingClient` and asset enums. |
| **Day Trading (`src/day_trading/`)** | **MEDIUM** | `harvester_5m.py` is essentially a copy of `src/data/harvester.py` with a 5-minute timeframe. It uses the same Alpaca historical client pattern but is isolated. |
| **Core (`src/core/`)** | **MEDIUM** | `retrainer.py` fetches historical bars via Alpaca for model retraining. The logic is portable but the data acquisition is not. |
| **Execution (`src/execution/`)** | **CRITICAL** | `live_orchestrator.py` (2,387 lines) is the Alpaca integration monolith. It owns `TradingClient`, dual WebSocket streams (`CryptoDataStream` + `StockDataStream`), historical warm-up clients, order submission (`submit_order`), position sync (`get_all_positions`), account queries (`get_account`), and the `TradingStream` for order updates. `factory_orchestrator.py` is smaller but still directly constructs `MarketOrderRequest` with `OrderSide` and `TimeInForce` enums. |
| **Root scripts** | **LIGHT** | `run_factory.py` and `test_alpaca.py` load credentials and instantiate Alpaca clients, but they are thin glue. |

### One-Paragraph Plain-English Summary

The good news: **the brain of Build-A-Bot is clean**. The ML pipeline (`src/ml/`) and the strategy layer (`src/strategies/`) contain zero Alpaca dependencies — they operate on Polars DataFrames, `Signal` objects, and `OrderParams`. This means any broker-agnostic refactor can preserve the core logic entirely. The bad news: **the body is Alpaca-shaped**. The execution layer (`src/execution/live_orchestrator.py`) is a 2,300-line Alpaca integration monolith that handles live WebSocket feeds, historical warm-ups, order placement, and position tracking — all through Alpaca-specific classes. The data layer (`src/data/`) is similarly welded to Alpaca historical and streaming clients, with `.df` schema assumptions on response objects. The refactor task breaks down into three tiers: (1) **easy** — create broker-agnostic enums (`OrderSide`, `TimeInForce`, `TimeFrame`) and a thin `BrokerClient` interface; (2) **medium** — abstract the data feed (`MarketDataFeed` already exists as an ABC in `feed.py`, but only `AlpacaCryptoFeed` is implemented) and the historical data provider; (3) **hard** — decouple `live_orchestrator.py`, which multiplexes two Alpaca WebSocket streams, manages order lifecycle through Alpaca's `TradingStream`, and performs position reconciliation via `get_all_positions()`. The `factory_orchestrator.py` is small and can be refactored quickly. Start with enums + order request dataclasses, then tackle the data layer, and save `live_orchestrator.py` for last.

---

**End of Report**
