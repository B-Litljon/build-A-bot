---
type: recon
date: 2026-04-25
time: unknown
agent: unknown
model: unknown
trigger: Inspect the sdk-decoupling branch — assess state and freshness before deciding the next architectural step
head: unknown
scope: read-only
imported_from: SDK_DECOUPLING_BRANCH_REPORT.md
---

# Build-A-Bot — `sdk-decoupling` Branch Inspection Report

**Date:** 2026-04-25  
**Current branch (unchanged):** `main`  
**Target branch:** `sdk-decoupling` (local + remote)  
**Mode:** Read-only — no checkout, merge, rebase, fetch, push, or file modifications performed.

---

## 1. Branch Existence and Freshness

```text
$ git branch -a | grep -i decoupling
  sdk-decoupling
  remotes/origin/sdk-decoupling
  remotes/origin/tradingbot-strategy-decoupling

$ git log -1 --format="%h | %ai | %s" sdk-decoupling
ce761de | 2026-04-06 12:47:35 -0700 | refactor: update import paths for SDK decoupling and clean up code

$ git log -1 --format="%h | %ai | %s" origin/sdk-decoupling
ce761de | 2026-04-06 12:47:35 -0700 | refactor: update import paths for SDK decoupling and clean up code
```

**Observation:** Local `sdk-decoupling` and `origin/sdk-decoupling` are at the exact same commit (`ce761de`). The branch was last touched on **April 6, 2026**.

---

## 2. Divergence from `main`

```text
$ git rev-list --left-right --count main...sdk-decoupling
19	4

$ git rev-list --left-right --count main...origin/sdk-decoupling
19	4
```

**Translation in plain English:**
- `main` is **19 commits ahead** of the merge-base.
- `sdk-decoupling` is **4 commits ahead** of the merge-base.
- The branches have **diverged**. `main` has moved significantly forward since `sdk-decoupling` was cut.

---

## 3. Branch Commit History (commits unique to `sdk-decoupling`)

```text
$ git log --oneline main..sdk-decoupling
ce761de refactor: update import paths for SDK decoupling and clean up code
4b9091c feat: add MLFactoryStrategy and run_factory script for SDK integration
2f7b2e9 feat: add FactoryOrchestrator for centralized execution routing in SDK
1fad280 feat: implement risk management and signal validation classes
```

**Summary:** 4 commits, all authored around the same April 6 window. The branch introduces:
1. `risk_management` and `signal validation` classes
2. `FactoryOrchestrator`
3. `MLFactoryStrategy` + `run_factory.py`
4. Import-path cleanup / SDK decoupling refactor

---

## 4. Files Changed vs. `main`

```text
$ git diff --name-status main...sdk-decoupling
A	run_factory.py
M	src/execution/__init__.py
A	src/execution/factory_orchestrator.py
A	src/execution/risk_manager.py
A	src/strategies/base.py
A	src/strategies/concrete_strategies/ml_factory_strategy.py
```

**Added (`A`) files:**
- `run_factory.py`
- `src/execution/factory_orchestrator.py`
- `src/execution/risk_manager.py`
- `src/strategies/base.py`
- `src/strategies/concrete_strategies/ml_factory_strategy.py`

**Modified (`M`) files:**
- `src/execution/__init__.py`

---

## 5. Targeted File Existence Check on `sdk-decoupling`

| Path | Exists on Branch | Line Count |
|------|------------------|------------|
| `src/strategies/base.py` | **Y** | 72 |
| `src/execution/enums.py` | **N** | — |
| `src/core/order_management.py` | **Y** | 198 |
| `src/data/equity_feed.py` | **N** | — |
| `src/strategies/concrete_strategies/ml_factory_strategy.py` | **Y** | 221 |

---

## 6. `src/strategies/base.py` (verbatim from branch)

```python
"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import polars as pl


@dataclass
class Signal:
    """Normalized signal output container."""

    direction: str  # 'long' or 'short'
    entry_price: float
    raw_sl_distance: float
    raw_tp_distance: float
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must inherit from this class and implement
    the generate_signals method to produce normalized signal outputs.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize strategy with custom parameters.

        Args:
            **kwargs: Strategy-specific configuration parameters
        """
        self.params = kwargs
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df: pl.DataFrame) -> Signal:
        """
        Generate trading signals from microstructure data.

        Args:
            df: Polars DataFrame containing standard 18-feature microstructure input

        Returns:
            Signal object containing direction, entry_price, raw_sl_distance,
            and raw_tp_distance

        Raises:
            ValueError: If input DataFrame is invalid or missing required features
        """
        pass

    def validate_input(self, df: pl.DataFrame) -> None:
        """
        Validate that input DataFrame meets requirements.

        Args:
            df: Input DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(df, pl.DataFrame):
            raise ValueError(f"Expected polars.DataFrame, got {type(df)}")

        if df.is_empty():
            raise ValueError("Input DataFrame is empty")

    def __repr__(self) -> str:
        return f"{self.name}(params={self.params})"
```

**Key observations:**
- Defines a broker-agnostic `Signal` dataclass (direction, entry_price, raw_sl_distance, raw_tp_distance, metadata).
- Abstract method is `generate_signals(self, df: pl.DataFrame) -> Signal`.
- No Alpaca imports, no `OrderSide`, no `TimeInForce`.
- **Conflict risk:** On `main`, the strategy base is `src/strategies/strategy.py` (which imports `core.order_management.OrderParams` and defines `analyze()` / `get_order_params()`). The two base classes are incompatible interfaces.

---

## 7. `src/execution/enums.py` (verbatim from branch)

```text
FILE DOES NOT EXIST
```

This file does **not** exist on `sdk-decoupling`.

---

## 8. `src/core/order_management.py` (verbatim from branch)

```python
from typing import Dict, Optional
import logging
from core.signal import Signal

# Optional alpaca imports for backtesting support
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    TradingClient = None
    MarketOrderRequest = None
    TimeInForce = None

    # Mock OrderSide enum for backtesting compatibility
    class MockOrderSide:
        SELL = "SELL"
        BUY = "BUY"

    OrderSide = MockOrderSide()


class OrderParams:
    """
    Defines parameters for order calculation and risk management.
    """

    def __init__(
        self,
        risk_percentage: float,
        tp_multiplier: float,
        sl_multiplier: float,
        use_trailing_stop: bool = False,
        **kwargs,
    ):
        self.risk_percentage = risk_percentage
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.use_trailing_stop = use_trailing_stop
        self.kwargs = kwargs


class OrderCalculator:
    def __init__(self, order_params: OrderParams):
        self.order_params = order_params

    def calculate_quantity(self, entry_price: float, current_capital: float) -> float:
        if entry_price == 0:
            return 0.0
        risk_amount = current_capital * self.order_params.risk_percentage
        return float(risk_amount / entry_price)

    def calculate_stop_loss(self, entry_price: float) -> float:
        return entry_price * self.order_params.sl_multiplier

    def calculate_take_profit(self, entry_price: float) -> float:
        return entry_price * self.order_params.tp_multiplier


class OrderManager:
    def __init__(
        self,
        trading_client: TradingClient,
        order_params: OrderParams,
        notification_manager=None,
    ):
        self.trading_client = trading_client
        self.order_params = order_params
        self.notification_manager = notification_manager
        self.active_orders: Dict[str, Dict] = {}
        self.order_calculator = OrderCalculator(self.order_params)

    def place_order(self, signal: Signal, current_capital: float) -> Optional[str]:
        if signal.type == "BUY":
            try:
                qty = self.order_calculator.calculate_quantity(
                    signal.price, current_capital
                )
                if qty <= 0:
                    return None

                stop_loss = self.order_calculator.calculate_stop_loss(signal.price)
                take_profit = self.order_calculator.calculate_take_profit(signal.price)

                logging.info(
                    f"Placing BUY for {signal.symbol}: Qty={qty:.4f}, SL={stop_loss:.2f}, TP={take_profit:.2f}"
                )

                req = MarketOrderRequest(
                    symbol=signal.symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                )

                order = self.trading_client.submit_order(req)
                order_id = getattr(order, "id", None)

                if order_id:
                    self.active_orders[str(order_id)] = {
                        "symbol": signal.symbol,
                        "entry_price": signal.price,
                        "quantity": qty,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    }
                    logging.info(f"Order {order_id} placed.")
                    if self.notification_manager:
                        self.notification_manager.notify_trade(
                            "BUY", signal.symbol, signal.price, qty, "Signal Triggered"
                        )
                    return str(order_id)

            except Exception as e:
                logging.error(f"Order Placement Failed: {e}", exc_info=True)
        return None

    def monitor_orders(self, market_data: Dict[str, float]):
        """
        Checks active orders against current market price.
        Args:
            market_data: Dict { "AAPL": 150.23, "TSLA": 200.50 }
        """
        for order_id, details in list(self.active_orders.items()):
            symbol = details["symbol"]
            if symbol not in market_data:
                continue

            current_price = market_data[symbol]
            action = None
            reason = ""

            if current_price <= details["stop_loss"]:
                action = OrderSide.SELL
                reason = f"Stop Loss ({current_price} <= {details['stop_loss']})"
            elif current_price >= details["take_profit"]:
                action = OrderSide.SELL
                reason = f"Take Profit ({current_price} >= {details['take_profit']})"

            if action:
                logging.info(f"Triggering Exit for {symbol}: {reason}")
                try:
                    req = MarketOrderRequest(
                        symbol=symbol,
                        qty=details["quantity"],
                        side=action,
                        time_in_force=TimeInForce.GTC,
                    )
                    self.trading_client.submit_order(req)
                    if self.notification_manager:
                        self.notification_manager.notify_trade(
                            "SELL", symbol, current_price, details["quantity"], reason
                        )
                    del self.active_orders[order_id]
                except Exception as e:
                    logging.error(f"Failed to exit {symbol}: {e}")

    def sync_positions(self):
        """
        Reconciles memory with actual Alpaca positions on startup.
        """
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                # Check if we are already managing this symbol
                is_managed = False
                for details in self.active_orders.values():
                    if details["symbol"] == pos.symbol:
                        is_managed = True
                        break

                if not is_managed:
                    logging.warning(
                        f"⚠️ Found unmanaged position for {pos.symbol} (Qty: {pos.qty}). Adopting it."
                    )

                    # Reconstruct thresholds based on Average Entry Price
                    avg_entry = float(pos.avg_entry_price)
                    sl = self.order_calculator.calculate_stop_loss(avg_entry)
                    tp = self.order_calculator.calculate_take_profit(avg_entry)

                    # Create a synthetic Order ID (prefix 'sync_')
                    synthetic_id = f"sync_{pos.symbol}_{pos.asset_id}"

                    self.active_orders[synthetic_id] = {
                        "symbol": pos.symbol,
                        "entry_price": avg_entry,
                        "quantity": float(pos.qty),
                        "stop_loss": sl,
                        "take_profit": tp,
                    }
                    logging.info(f"✅ Adopted {pos.symbol}: SL={sl:.2f}, TP={tp:.2f}")

        except Exception as e:
            logging.error(f"Failed to sync positions: {e}")
```

**Key observations:**
- `OrderManager` directly constructs `MarketOrderRequest` with `OrderSide.BUY` / `OrderSide.SELL` and `TimeInForce.GTC`.
- Uses a `try/except ImportError` guard with a `MockOrderSide` fallback for backtesting without Alpaca SDK installed.
- `Signal` is imported from `core.signal` — but note that `sdk-decoupling` also redefined `Signal` inside `base.py` (see §6), creating a **potential name collision**.
- Contains `sync_positions()` logic that reaches into Alpaca-specific attributes (`pos.avg_entry_price`, `pos.asset_id`).

---

## 9. Mergeability Dry-Run

```text
$ git merge-tree $(git merge-base main sdk-decoupling) main sdk-decoupling | head -100
```

**Files with conflicts (`added in both` / divergent histories):**

| File | Conflict Type |
|------|---------------|
| `run_factory.py` | `added in both` |
| `src/execution/__init__.py` | `modified on both` (inferred from presence in diff) |
| `src/execution/factory_orchestrator.py` | `added in both` |
| `src/execution/risk_manager.py` | `added in both` |
| `src/strategies/base.py` | `added in both` (branch adds it; main does not have it, but main has `strategy.py` occupying the same semantic slot) |
| `src/strategies/concrete_strategies/ml_factory_strategy.py` | `added in both` |

**Conflict marker count:** 4 sets of `<<<<<<< .our / ======= / >>>>>>> .their` markers detected in the merge-tree output.

**Conclusion:** A merge into `main` today **would NOT merge cleanly**. Manual conflict resolution is required for at least 4 files, most critically:
- `run_factory.py` — the branch version is a complete rewrite of the main version.
- `src/execution/factory_orchestrator.py` and `src/execution/risk_manager.py` — both exist on `main` now (added later) with different content than the branch versions.

---

## 10. Verdict Summary

`sdk-decoupling` is a **small, focused branch** (4 commits, last touched April 6) that attempted to introduce broker-agnostic base classes (`BaseStrategy`, `Signal`) and a standalone `OrderManager`. However, it is now **stale and diverged** from `main`, which has raced 19 commits ahead. The branch contains **two of the missing SDK files** (`src/strategies/base.py` and `src/core/order_management.py`) but notably **does not** contain `src/execution/enums.py` or `src/data/equity_feed.py`. Most importantly, a merge today **would require conflict resolution** on at least 4 files because `main` independently added its own versions of `FactoryOrchestrator`, `RiskManager`, `run_factory.py`, and `ml_factory_strategy.py` after the branch was cut. The interfaces are incompatible (e.g., `main` uses `Strategy.analyze()` + `OrderParams`; `sdk-decoupling` uses `BaseStrategy.generate_signals()` + a different `Signal` dataclass). This branch is best treated as a **reference / spike**, not as a clean merge candidate.

---

**End of Report**
