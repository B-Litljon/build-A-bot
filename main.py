from src.data.api import AlpacaClient
import os
import pretty_errors
from dotenv import load_dotenv

load_dotenv()
alpaca_key = os.getenv("alpaca_key")
alpaca_secret = os.getenv("alpaca_secret")


alpaca_client = AlpacaClient(alpaca_key, alpaca_secret)

def __main__():
    # define the timeframe and lookback window
     # Calculate the timeframe, start_date, and end_date using the method
    timeframe, start_date, end_date = alpaca_client.time_and_lookback_window(
        "Day", 33
    )

    # get the most active stocks
    most_active_stocks = alpaca_client.get_most_active_stocks()

    # get the historical data for the most active stocks
    stock_bar_data = alpaca_client.get_stock_bar_data(alpaca_client.stock_client, timeframe, start_date, end_date)

    # Now you can print or process the stock_bar_data as needed
    print(stock_bar_data)

if __name__ == "__main__":  # Add this block
    __main__()