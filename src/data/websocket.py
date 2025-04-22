from alpaca.data.live import StockDataStream
from alpaca.trading.stream import TradingStream
import asyncio

class WebsocketHandler:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.trade_stream = TradingStream(api_key, api_secret, paper=paper)
        self.data_stream = StockDataStream(api_key, api_secret, paper=paper)

    async def trade_updates_handler(self, data):
        print("Trade Update:", data)

    async def bar_updates_handler(self, data):
        print("Bar Update:", data)

    async def quote_updates_handler(self, data):
        print("Quote Update:", data)

    def subscribe_trade_updates(self):
        self.trade_stream.subscribe_trade_updates(self.trade_updates_handler)

    def subscribe_bar_updates(self, symbol: str):
        self.data_stream.subscribe_bars(self.bar_updates_handler, symbol)

    def subscribe_quote_updates(self, symbol: str):
        self.data_stream.subscribe_quotes(self.quote_updates_handler, symbol)

    def run(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.data_stream.run())
        loop.run_until_complete(self.trade_stream.run())