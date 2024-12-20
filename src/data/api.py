import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import polars as pl  # Use polars with the correct alias
import numpy as np
import pretty_errors

from alpaca.data import (
    StockHistoricalDataClient,
    StockBarsRequest,
    OptionChainRequest,
    TimeFrame,
    TimeFrameUnit
)
from alpaca.data.timeframe import TimeFrameUnit
from alpaca.data.requests import MostActivesRequest
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
import alpaca.common.exceptions

load_dotenv()
alpaca_key = os.getenv("alpaca_key")
alpaca_secret = os.getenv("alpaca_secret")

class AlpacaClient:
    def __init__(self, alpaca_key, alpaca_secret):
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret
        self.stock_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        self.option_client = OptionHistoricalDataClient(alpaca_key, alpaca_secret)
        self.screener_client = ScreenerClient(alpaca_key, alpaca_secret)
        self.ticker_df = pl.DataFrame({"ticker": []})

    def time_and_lookback_window(self, timeframe, days_back, end_date=None):
        """
        Calculates the start and end dates for a given timeframe and lookback window.

        Args:
            timeframe (str): The timeframe unit (e.g., 'Day', 'Hour', 'Minute').
            days_back (int): The number of days to look back from the end date.
            end_date (str, optional): The end date in 'YYYY-MM-DD' format. 
                                       Defaults to today if not provided.

        Returns:
            tuple: A tuple containing the TimeFrame object, start date, and end date.
        """
        timeframe_unit = TimeFrameUnit(timeframe)
        timeframe = TimeFrame(1, timeframe_unit)

        if end_date is None:
            end_date = datetime.now(ZoneInfo("America/New_York"))
        else:
            # Assuming end_date is provided as a string in 'YYYY-MM-DD' format
            end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("America/New_York"))  

        start_date = end_date - timedelta(days=days_back)

        return timeframe, start_date, end_date

    def get_most_active_stocks(self):
        """
        Retrieves the most active stocks based on volume.

        Returns:
            list: A list of ticker symbols for the most active stocks.
        """
        most_actives_request = MostActivesRequest()
        most_actives_response = self.screener_client.get_most_actives(most_actives_request)
        #return most_actives_response
        
        # Convert list of dictionaries to Polars DataFrame
        watchlist = pl.DataFrame(most_actives_response.most_actives)
        
        # Extract ticker symbols from the 'symbol' column
        ticker_symbols = watchlist['symbol'].str.slice(1).to_list() 
        
        # Create a new DataFrame with the extracted tickers
        self.ticker_df = pl.DataFrame({'ticker': ticker_symbols}) 
        return self.ticker_df

    def get_stock_bar_data(self, stock_client, timeframe, start_date, end_date):
        ticker_symbols = self.ticker_df["ticker"].to_list()

        stock_bars_request = StockBarsRequest(
            symbol_or_symbols=ticker_symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            feed='iex'  # Use the IEX data feed
        )

        print("Stock bars request:", stock_bars_request)  # Print the request

        try:
            stock_bars = stock_client.get_stock_bars(stock_bars_request)
            print("API response:", stock_bars)  # Print the response
            print("Stock bars data:", stock_bars.df)  # Print the DataFrame
            return stock_bars
        except alpaca.common.exceptions.AlpacaAPIError as e:
            print(f"Alpaca API Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None

    def get_option_chain_data(self, option_client):  # Removed ticker_symbols argument
        option_chain_data = {}
        
        # Iterate over tickers in the Polars DataFrame
        for ticker in self.ticker_df['ticker']:  
            option_chain_request = OptionChainRequest(
                underlying_symbol=ticker
            )
            try:
                option_chain = option_client.get_option_chain(option_chain_request)
                option_chain_data[ticker] = option_chain
            except Exception as e:
                print(f"Error fetching option chain data for {ticker}: {e}")
        print('Options chain data (raw):\n\n', option_chain_data)
        return option_chain_data
