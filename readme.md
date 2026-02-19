# Build-A-Bot

An algorithmic trading engine for Alpaca, built in Python. It selects active stocks on startup, aggregates live market data into higher-timeframe candles, runs technical analysis through a pluggable strategy system, and executes orders with built-in stop-loss and take-profit management.

**Current Strategy:** RSI + Bollinger Bands (mean-reversion, two-stage confirmation with engulfing pattern detection).

**Core Capabilities:**
- **Rapid Warmup** -- Fetches historical data on startup so the strategy can generate signals immediately, no waiting for hours of live bars.
- **State Reconciliation** -- Syncs with Alpaca positions on every restart. If the bot crashes and restarts, it adopts any open positions and resumes monitoring them.
- **Zero-Latency Aggregation** -- Converts streaming 1-minute bars into 5-minute candles in memory using Polars DataFrames.
- **Tick-Level Exit Monitoring** -- Stop-loss and take-profit are checked on every incoming 1-minute bar, not just when aggregated candles complete.

> **Status:** V1.0 -- Production-ready for Paper Trading.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Data Engine** | [Polars](https://pola.rs) | High-performance DataFrames for candle aggregation and history |
| **Broker SDK** | [alpaca-py](https://github.com/alpacahq/alpaca-py) | REST API + WebSocket streaming for market data and order execution |
| **Technical Analysis** | [TA-Lib](https://ta-lib.org) | RSI, Bollinger Bands, Rate of Change, SMA indicators |
| **Async Runtime** | asyncio | Async warmup and state sync before the blocking stream starts |
| **Validation** | Pydantic (via alpaca-py) | Request/response validation for API calls |
| **Runtime** | Python 3.12 | |

---

## Installation

### Prerequisites

- Python 3.12+
- [TA-Lib C library](https://ta-lib.org) installed on your system
- An [Alpaca](https://alpaca.markets) account (free paper trading account works)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd build-A-bot
   ```

2. **Install dependencies with Pipenv:**
   ```bash
   pip install pipenv
   pipenv install
   pipenv shell
   ```

   Or with pip directly:
   ```bash
   pip install polars numpy ta-lib python-dotenv pretty-errors alpaca-py
   ```

3. **Create your `.env` file** in the project root:
   ```
   alpaca_key=YOUR_ALPACA_API_KEY
   alpaca_secret=YOUR_ALPACA_SECRET_KEY
   ```

   Get your keys from the [Alpaca Dashboard](https://app.alpaca.markets). Use **paper trading** keys while testing.

---

## Usage

### Live / Paper Trading

```bash
python main.py
```

**What happens on startup:**
1. Fetches the top 3 most active stocks by volume from Alpaca's screener.
2. **Warmup** -- Pulls historical 1-minute bars and pre-fills the bar aggregators so the strategy has enough data to analyze immediately.
3. **State Sync** -- Queries Alpaca for any existing open positions and adopts them into the internal order tracker.
4. **Stream** -- Subscribes to the live WebSocket feed and begins processing bars in real-time.

The bot runs until you stop it with `Ctrl+C`.

### Configuration

Strategy parameters are set when instantiating `RSIBBands()` in `main.py`.

**Strict Mode (Default)** -- Conservative entry rules. Suitable for live/paper trading:
```python
strategy = RSIBBands()
# Defaults: RSI must drop to 30, bandwidth ROC > 0.15, bullish engulfing required
```

**Loose Mode** -- Relaxed thresholds that trigger signals more frequently. Useful for testing that the pipeline works end-to-end:
```python
strategy = RSIBBands(
    stage1_rsi_threshold=70,
    stage2_rsi_entry=10,
    stage2_rsi_exit=90,
    stage2_min_roc=0.0001
)
```

**Capital and Symbol Count** are also configured in `main.py`:
```python
# Change capital allocation
bot = TradingBot(strategy=strategy, capital=50000.0, ...)

# Change number of symbols (default is top 3)
symbols = active_stocks_df["ticker"].head(5).to_list()
```

See [docs/STRATEGY_RSI_BBANDS.md](docs/STRATEGY_RSI_BBANDS.md) for a full parameter reference.

### Backtesting / Simulation

Run the replay simulation against real historical data:

```bash
python tests/test_live_simulation.py
```

This fetches 2 days of 1-minute SPY data from Alpaca and replays it through the bot at maximum speed using mock trading clients (no real orders are placed). It uses intentionally loose strategy parameters to verify order generation.

### Strategy Unit Tests

Test the RSI+BBands logic with synthetic data:

```bash
python tests/test_strategy_logic.py
```

Runs two test cases:
- **Positive path** -- Verifies a BUY signal is generated when all conditions are met (oversold RSI, bandwidth expansion, bullish engulfing pattern).
- **Negative path** -- Verifies no signal is generated when the engulfing pattern is absent.

---

## Project Structure

```
build-A-bot/
  main.py                              # Entry point (hybrid async/sync)
  .env                                 # API credentials (not committed)
  Pipfile                              # Dependencies
  src/
    core/
      trading_bot.py                   # TradingBot orchestrator
      order_management.py              # OrderParams, OrderCalculator, OrderManager
      signal.py                        # Signal data class
      ws_stream_simulator.py           # Historical data replay for testing
    strategies/
      strategy.py                      # Abstract Strategy base class
      strategy_factory.py              # Factory for creating strategies by name
      concrete_strategies/
        rsi_bbands.py                  # RSI + Bollinger Bands strategy
        sma_crossover.py               # SMA Crossover strategy
    utils/
      bar_aggregator.py                # LiveBarAggregator (1m -> Nm conversion)
    data/
      api_requests.py                  # AlpacaClient (REST API wrapper)
  tests/
    test_live_simulation.py            # End-to-end replay test
    test_strategy_logic.py             # Strategy unit test
  docs/
    ARCHITECTURE.md                    # System design deep-dive
    STRATEGY_RSI_BBANDS.md             # Strategy parameter reference
```

---

## Troubleshooting

### "subscription does not permit querying recent SIP data"

**Cause:** Alpaca Free Tier accounts cannot fetch market data from the last 15 minutes.

**Fix:** Already handled. The warmup method shifts its data window back 16 minutes. If you see this error, confirm you are on the latest version of `src/core/trading_bot.py`.

### "No active stocks found. Exiting."

**Cause:** The Alpaca screener returned no results. This happens on weekends and outside market hours.

**Fix:** Wait for market hours (Mon-Fri, 9:30 AM - 4:00 PM ET), or hardcode symbols in `main.py` for testing:
```python
symbols = ["SPY", "AAPL", "TSLA"]
```

### "Warming up... X/21 candles"

**Cause:** The bar aggregator hasn't accumulated enough candles for the strategy to analyze. With a 5-minute timeframe and 21-candle warmup period, this takes about 105 minutes of live data if warmup has no historical bars.

**Fix:** This is normal if historical data was sparse. On weekends, the warmup may find zero bars -- the bot will still run and accumulate candles from live data once the market opens.

### "Found unmanaged position for XXXX. Adopting it."

**Cause:** The bot found an open position in your Alpaca account that it wasn't tracking internally.

**Fix:** This is expected behavior after a restart. The bot is adopting the position and will manage its stop-loss and take-profit going forward. No action required.

### Pydantic "expected an instance of..." warnings

**Cause:** Passing a raw string where an Enum is expected in Alpaca API requests.

**Fix:** Already handled. The `MostActivesRequest` in `api_requests.py` now passes `MostActivesBy.VOLUME` explicitly instead of a string.

---

## Further Reading

- [Architecture Guide](docs/ARCHITECTURE.md) -- How the hybrid engine, data pipeline, and safety systems work.
- [RSI+BBands Strategy Card](docs/STRATEGY_RSI_BBANDS.md) -- Full parameter reference and logic explanation.
