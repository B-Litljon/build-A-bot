from typing import Dict, List, TYPE_CHECKING
from core.signal import Signal
from strategies.strategy import Strategy
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.models.bars import Bar
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from core.order_management import OrderManager
from utils.bar_aggregator import LiveBarAggregator as lba
import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from data.api_requests import AlpacaClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TradingBot:
    """
    The main trading bot class.
    ...
    """
    def __init__(self,
                 strategy: Strategy,
                 capital: float,
                 trading_client: TradingClient,
                 live_stock_data: StockDataStream,
                 symbols: List[str],  # Changed from symbol: str
                 target_intervals: List[int] = [5, 15],
                 notification_manager=None):
        self.strategy = strategy
        self.capital = capital
        self.trading_client = trading_client
        self.notification_manager = notification_manager
        self.order_manager = OrderManager(trading_client, strategy.get_order_params(), notification_manager=self.notification_manager)
        self.live_stock_data = live_stock_data
        self.symbols = symbols
        self.target_intervals = target_intervals
        # Use a dictionary to store a bar aggregator for each symbol
        self.lba_dict = {
            symbol: lba(timeframe=strategy.timeframe, history_size=240)
            for symbol in self.symbols
        }
        logging.info(f"TradingBot initialized for symbols: {self.symbols}")

    async def warmup(self, data_client: 'AlpacaClient'):
        """
        Fetches historical data to pre-fill the bar aggregators.
        This allows the strategy to run immediately without waiting hours for live data.
        """
        logging.info("Starting Rapid Warmup...")

        required_candles = self.strategy.warmup_period
        agg_timeframe = int(list(self.lba_dict.values())[0].timeframe)
        lookback_minutes = int(required_candles * agg_timeframe * 1.5)

        logging.info(
            f"Warmup Requirement: {required_candles} candles ({agg_timeframe}m timeframe). Fetching last {lookback_minutes} minutes."
        )

        # Shift back 16 minutes to avoid "subscription does not permit querying recent SIP data" error
        end_time = datetime.now(ZoneInfo("America/New_York")) - timedelta(minutes=16)
        start_time = end_time - timedelta(minutes=lookback_minutes)
        start_utc = start_time.astimezone(ZoneInfo("UTC"))
        end_utc = end_time.astimezone(ZoneInfo("UTC"))

        for symbol in self.symbols:
            try:
                logging.info(f"Fetching warmup data for {symbol}...")
                bars_df = data_client.get_historical_ohlcv(
                    symbol=symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start_date=start_time.isoformat(),
                    end_date=end_time.isoformat(),
                )

                if bars_df.is_empty():
                    logging.warning(f"No warmup data found for {symbol}.")
                    continue

                count = 0
                for row in bars_df.iter_rows(named=True):
                    ts = row["timestamp"]
                    if hasattr(ts, "to_pydatetime"):
                        ts = ts.to_pydatetime()
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=ZoneInfo("UTC"))

                    if ts < start_utc or ts > end_utc:
                        continue

                    bar_data = {
                        "timestamp": ts,
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                    }
                    self.lba_dict[symbol].add_bar(bar_data)
                    count += 1

                logging.info(f"Warmed up {symbol} with {count} 1m bars.")
            except Exception as e:
                logging.error(f"Failed to warmup {symbol}: {e}")

        logging.info("Warmup Complete. Bot is ready to trade.")

    async def handle_bar_update(self, raw_bar: Bar):
        """
        Async handler for incoming raw bar updates from the data stream.
        """
        symbol = raw_bar.symbol
        current_price = raw_bar.close

        # --- 1. EXIT LOGIC (Check every tick) ---
        # Create a mini market_data dict for the monitor
        self.order_manager.monitor_orders({symbol: current_price})

        # --- 2. EXISTING AGGREGATION LOGIC ---
        logging.info(
            f"Received raw bar for {symbol}:\n"
            f"  Open: {raw_bar.open}\n"
            f"  High: {raw_bar.high}\n"
            f"  Low: {raw_bar.low}\n"
            f"  Close: {raw_bar.close}\n"
            f"  Volume: {raw_bar.volume}"
        )
        try:
            formatted_bar = {
                "timestamp": raw_bar.timestamp,
                "open": raw_bar.open,
                "high": raw_bar.high,
                "low": raw_bar.low,
                "close": raw_bar.close,
                "volume": raw_bar.volume,
            }

            # Get the correct bar aggregator for the symbol
            symbol_lba = self.lba_dict.get(symbol)
            if not symbol_lba:
                logging.warning(f"No bar aggregator found for symbol: {symbol}")
                return

            is_new_agg_bar = symbol_lba.add_bar(formatted_bar)

            if is_new_agg_bar:
                logging.info(f"New aggregated bar created for {symbol}")
                candles = symbol_lba.history_df
                if len(candles) >= self.strategy.warmup_period:
                    signals = self.strategy.analyze({symbol: candles})
                    self.place_orders(signals)
                else:
                    logging.info(f"Warming up... {len(candles)}/{self.strategy.warmup_period} candles")

        except Exception as e:
            logging.error(f"Error handling bar update for {symbol}: {e}", exc_info=True)

    def place_orders(self, signals: List[Signal]):
        """Places orders based on the received signals."""
        for signal in signals:
            if signal.type == "BUY":
                self.order_manager.place_order(signal, self.capital)

    async def run(self):
        """
        Starts the trading bot: syncs state and subscribes.
        """
        logging.info("Syncing internal state with Alpaca positions...")
        self.order_manager.sync_positions()

        logging.info(f"Subscribing to bar updates for symbols: {self.symbols}")
        self.live_stock_data.subscribe_bars(self.handle_bar_update, *self.symbols)
        logging.info("Bot initialized and waiting for stream start.")

    async def log_status_periodically(self, interval: int = 30):
        """Logs the bot's status at regular intervals."""
        while True:
            self.log_status()
            await asyncio.sleep(interval)

    def log_status(self):
        """
        Logs the current status of the bot.
        """
        logging.info("--- Bot Status ---")
        logging.info(f"Current Capital: {self.capital}")
        logging.info(f"Active Orders: {self.order_manager.active_orders}")
        logging.info("------------------")
