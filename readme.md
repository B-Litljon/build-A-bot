# Build-A-Bot: Your Modular Trading Bot Toolkit  INCOMPLETE

**Build-A-Bot** empowers you to design, implement, and deploy your own trading strategies with ease. Whether you're a day trader, a scalper, an intraday trader, or a long-term investor, Build-A-Bot provides the building blocks to create a customized trading bot tailored to your needs.

## Project Goals

*   **Modularity:** Build-A-Bot is designed with a modular architecture, allowing you to easily swap different components (strategies, order management logic, data sources) in and out.
*   **Extensibility:** The framework is built to be extended. You can easily create and integrate new strategies, indicators, and data sources.
*   **User-Friendliness:**  Build-A-Bot aims to simplify the process of creating trading bots, making it accessible even if you're not a seasoned developer.
*   **Flexibility:** The bot supports a wide range of trading styles, from high-frequency scalping to long-term investing.
*   **Transparency:** The code is open-source, well-documented, and easy to understand, promoting trust and allowing for community contributions.

## Core Components

This section dives into the essential parts of the Build-A-Bot framework.

### 1. OrderParams: Defining Your Trading Style

The `OrderParams` class is the foundation of your strategy's risk management and order execution settings. It's like the DNA of your trading approach, dictating how your bot will behave in the market.

**File:** `utils/order_params.py`

**Class:** `OrderParams`

**Purpose:** Stores parameters that define how orders are calculated and managed.

**Constructor:**

```python
def __init__(self, risk_percentage=0.02, tp_multiplier=1.10, sl_multiplier=0.95, 
             sma_short_period=None, sma_long_period=None, sma_crossover_type=None, **kwargs):
```

**Attributes:**

*   `risk_percentage` (float): **The cornerstone of your risk management.** This parameter determines what percentage of your total trading capital you are willing to risk on a single trade. For instance, a value of `0.02` means you'll risk 2% of your capital on each trade.
*   `tp_multiplier` (float): **Your profit-taking ambition.** This multiplier, when applied to the entry price of an order, calculates the take-profit level. A `tp_multiplier` of `1.10` sets your take-profit target at 10% above the entry price.
*   `sl_multiplier` (float): **Your safety net.** This multiplier determines your stop-loss level when applied to the entry price. A `sl_multiplier` of `0.95` places your stop-loss at 5% below the entry price.
*   `sma_short_period` (int, optional): **The short-term perspective for trailing stops.** This attribute (if used) defines the period for calculating a short-term Simple Moving Average (SMA). It's used in conjunction with `sma_long_period` and `sma_crossover_type` to implement a dynamic trailing stop-loss mechanism.
*   `sma_long_period` (int, optional): **The long-term perspective for trailing stops.** Similar to `sma_short_period`, but this defines the period for a long-term SMA, also used for trailing stop-loss calculations.
*   `sma_crossover_type` (str, optional): **The trailing stop behavior.** This parameter dictates how the trailing stop-loss will be adjusted. It can be either:
    *   `"long"`: The stop-loss trails *below* the price, commonly used for long (buy) positions.
    *   `"short"`: The stop-loss trails *above* the price, commonly used for short-sell positions.
*   `**kwargs`: **The flexibility factor.** This allows you to add any other custom parameters to your `OrderParams` object, making your strategy even more adaptable. These extra parameters can be accessed as attributes of the `OrderParams` instance.

**Example:**

```python
from order_params import OrderParams

# Conservative parameters
conservative_params = OrderParams(risk_percentage=0.01, tp_multiplier=1.05, sl_multiplier=0.98)

# Aggressive parameters with trailing stop
aggressive_params = OrderParams(risk_percentage=0.03, tp_multiplier=1.2, sl_multiplier=0.9, sma_short_period=14, sma_long_period=28, sma_crossover_type="long")

# Using a custom parameter
custom_params = OrderParams(risk_percentage=0.02, my_custom_parameter="some_value") 
print(custom_params.my_custom_parameter)  # Output: some_value
```

### 2. Strategy: The Brain of Your Bot

The `Strategy` class (and its concrete implementations like `EMACrossoverStrategy`) is where your trading logic resides. It defines how the bot analyzes market data, generates trading signals, and what parameters should be used for order placement.

**File:** `strategies/strategy.py` (and specific strategy files like `strategies/ema_crossover.py`)

**Abstract Class:** `Strategy`

**Purpose:** Defines the interface for all trading strategies.

**Methods:**

*   `analyze(self, data)`: **The core of your strategy.** This *abstract method* (meaning it must be implemented by concrete strategy classes) takes market data as input and returns a list of `Signal` objects, indicating buy or sell actions.
*   `get_order_params(self)`: **The strategy's order settings.** This *abstract method* returns an `OrderParams` object containing the default parameters that should be used with this particular strategy.

**Concrete Implementation:** `EMACrossoverStrategy`

**Purpose:** A specific strategy that uses the crossover of Exponential Moving Averages (EMAs) to generate trading signals.

**Constructor:**

```python
def __init__(self, short_ema_period=50, long_ema_period=200, **kwargs):
```

**Attributes:**

*   `short_ema_period` (int): The period for the short-term EMA.
*   `long_ema_period` (int): The period for the long-term EMA.

**Methods:**

*   `analyze(self, data)`: Calculates the short and long EMAs from the input `data`. Generates "BUY" signals when the short EMA crosses above the long EMA and "SELL" signals when the short EMA crosses below the long EMA. Returns these signals as a list of `Signal` objects.
*   `get_order_params(self)`: Returns a default `OrderParams` object tailored for the EMA crossover strategy.

### 3. Signal: Communicating Trading Actions

The `Signal` class represents a trading signal generated by a `Strategy`.

**File:** `core/signal.py`

**Class:** `Signal`

**Purpose:** Encapsulates a trading action (buy or sell) along with relevant information.

**Constructor:**

```python
def __init__(self, type, symbol, entry_price, quantity=None):
```

**Attributes:**

*   `type` (str): The type of signal, either "BUY" or "SELL".
*   `symbol` (str): The symbol of the asset to trade (e.g., "AAPL", "BTCUSD").
*   `entry_price` (float): The suggested entry price for the trade.
*   `quantity` (float, optional): The suggested quantity to trade. If not provided, the `OrderManager` or `TradingBot` might calculate it based on risk parameters.

### 4. OrderManager: The Order Executor

The `OrderManager` class handles the crucial task of interacting with the trading API (e.g., Alpaca) to place, monitor, and manage orders.

**File:** `core/order_manager.py`

**Class:** `OrderManager`

**Purpose:** Executes orders based on signals and manages active orders according to the defined `OrderParams`.

**Constructor:**

```python
def __init__(self, trading_client, order_params):
```

**Attributes:**

*   `trading_client`: An instance of a `TradingClient` (e.g., `AlpacaTradingClient`) that provides an interface to the brokerage or exchange.
*   `order_params`: An `OrderParams` object that specifies the order settings (risk percentage, stop-loss, take-profit, etc.).
*   `active_orders`: A data structure (e.g., a dictionary) to store and track currently active orders.

**Methods:**

*   `place_order(self, symbol, side, quantity, entry_price)`: Places a buy or sell order through the `trading_client`. Calculates stop-loss and take-profit levels based on `order_params`. Stores the order details in `active_orders`.
*   `monitor_orders(self, market_data)`:  Tracks active orders. Implements stop-loss, take-profit, and trailing stop-loss logic based on current market data and the `order_params`. If a stop-loss or take-profit is hit, it places an order to close the position and updates the `active_orders` accordingly.

### 5. TradingClient: The API Interface

The `TradingClient` is an abstract class (or an interface) that defines how the bot interacts with a specific trading API. Concrete implementations (like `AlpacaTradingClient`) handle the specifics of each API.

**File:** `trading_clients/trading_client.py` (and specific client files like `trading_clients/alpaca_trading_client.py`)

**Abstract Class/Interface:** `TradingClient`

**Purpose:** Provides a standardized way to interact with different trading APIs.

**Methods (examples):**

*   `submit_order(...)`: Submits an order to the exchange.
*   `get_account_info(...)`: Retrieves account information.
*   `get_positions(...)`: Retrieves currently open positions.
*   `...`: Other methods as needed for interacting with the API.

**Concrete Implementation:** `AlpacaTradingClient` (or `MockTradingClient` for testing)

**Purpose:** Handles the specific communication with the Alpaca trading API (or simulates API interaction for testing purposes).

### 6. TradingBot: The Orchestrator

The `TradingBot` is the central class that brings all the other components together. It runs the main trading loop, fetches data, gets signals from the strategy, and uses the `OrderManager` to execute trades.

**File:** `trading_bot.py`

**Class:** `TradingBot`

**Purpose:**  The main engine of the trading bot.

**Constructor:**

```python
def __init__(self, strategy, capital, trading_client):
```

**Attributes:**

*   `strategy`: An instance of a `Strategy` class (e.g., `EMACrossoverStrategy`).
*   `capital`: The initial trading capital.
*   `trading_client`: An instance of a `TradingClient`.
*   `order_manager`: An instance of `OrderManager`.

**Methods:**

*   `run(self)`: The main trading loop.
    1.  Fetches market data (using a `DataHandler` - to be implemented).
    2.  Passes the data to the `strategy` to get trading signals.
    3.  Processes signals:
        *   For "BUY" signals, calculates the quantity based on risk management and calls `order_manager.place_order()`.
        *   For "SELL" signals, handles them appropriately (e.g., close positions).
    4.  Calls `order_manager.monitor_orders()` to manage active orders.
    5.  Implements logging of actions, performance tracking, and other essential bot logic.

## Getting Started

1.  **Installation:**
    *   Clone the repository: `git clone [repository URL]`
    *   Install dependencies: `pip install -r requirements.txt`
2.  **Configuration:**
    *   Set up your API keys (if using a real trading client like Alpaca).
    *   Create an instance of `OrderParams` with your desired settings.
    *   Choose a strategy (e.g., `EMACrossoverStrategy`) and configure its parameters.
3.  **Running the Bot:**

```python
from alpaca.trading.client import TradingClient  # Or your chosen trading client
from strategies.ema_crossover import EMACrossoverStrategy
from utils.order_params import OrderParams
from core.order_manager import OrderManager
from trading_bot import TradingBot

# 1. Configure OrderParams
order_params = OrderParams(risk_percentage=0.01, tp_multiplier=1.1, sl_multiplier=0.98)

# 2. Initialize Trading Client (Alpaca example)
trading_client = TradingClient("YOUR_ALPACA_API_KEY", "YOUR_ALPACA_API_SECRET", paper=True)  # paper=True for paper trading

# 3. Instantiate OrderManager
order_manager = OrderManager(trading_client, order_params)

# 4. Select and configure your Strategy
strategy = EMACrossoverStrategy(short_ema_period=20, long_ema_period=50)

# 5. Create the TradingBot
bot = TradingBot(strategy, capital=10000, trading_client=trading_client)

# 6. Start the bot
bot.run()


## Future Development

*   **Data Handling:** Implement a robust `DataHandler` class to fetch, store, and manage market data from various sources.
*   **Backtesting:** Create a `Backtester` component to evaluate strategies on historical data before deploying them live.
*   **More Strategies:** Develop and add more trading strategies (e.g., RSI divergence, Bollinger Bands, etc.).
*   **Advanced Order Types:** Implement support for different order types (e.g., limit orders, bracket orders).
*   **Portfolio Management:** Add features for managing multiple positions and assets.
*   **User Interface:** Consider building a graphical user interface (GUI) or a web interface to make the bot more user-friendly.

## Contributing

Contributions are welcome! If you'd like to contribute to Build-A-Bot, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and write tests.
4.  Submit a pull request.
```
