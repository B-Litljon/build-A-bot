from src.data.api import AlpacaClient
import os
import pretty_errors
from dotenv import load_dotenv

load_dotenv()
alpaca_key = os.getenv("alpaca_key")
alpaca_secret = os.getenv("alpaca_secret")


alpaca_client = AlpacaClient(alpaca_key, alpaca_secret)

def __main__():
    # most_active_stocks = alpaca_client.get_most_active_stocks()
    # print(most_active_stocks)
    stock_bars = alpaca_client.get_stock_bar_data(alpaca_client.stock_client, "1D", "2021-01-01", "2024-01-10")
    stock_bars_df = stock_bars.df
    print(stock_bars_df)