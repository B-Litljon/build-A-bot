import asyncio
import os
from datetime import datetime, timezone, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

async def main():
    client = CryptoHistoricalDataClient(api_key, secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=300)
    req = CryptoBarsRequest(
        symbol_or_symbols=["BTC/USD"],
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start,
        end=end,
    )
    bars = client.get_crypto_bars(req)
    raw_df = bars.df
    print(raw_df.index.names)
    print(raw_df.loc["BTC/USD"].reset_index().columns)

asyncio.run(main())
