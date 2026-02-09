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
from alpaca.data.enums import MostActivesBy
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

        This method is used to determine the appropriate start and end dates for fetching historical
        stock data. It takes a timeframe unit (e.g., 'Day', 'Hour', 'Minute'), a number of days to
        look back, and an optional end date. If no end date is provided, it defaults to the current
        date and time in the America/New_York time zone.

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

    def get_historical_ohlcv(self, symbol: str, timeframe: TimeFrame, start_date: str, end_date: str) -> pl.DataFrame:
        """
        Fetches historical OHLCV data and returns it as a Polars DataFrame.
        """
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            bars = self.stock_client.get_stock_bars(request_params)
            # Reset index to preserve 'symbol' and 'timestamp' as columns
            pandas_df = bars.df.reset_index()
            try:
                df = pl.from_pandas(pandas_df)
            except ImportError:
                df = pl.DataFrame(pandas_df.to_dict(orient="list"))
            return df
        except alpaca.common.exceptions.APIError as e:
            print(f"Alpaca API Error: {e}")
            return pl.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return pl.DataFrame()

    def get_most_active_stocks(self):
        """
        Retrieves the most active stocks based on volume.

        This method fetches the most active stocks from the Alpaca API using the `MostActivesRequest`.
        It processes the API response to extract the ticker symbols and stores them in a Polars DataFrame
        (`self.ticker_df`) for later use in other methods.

        Returns:
            pl.DataFrame: A Polars DataFrame containing a single column 'ticker' with the list of
                         ticker symbols for the most active stocks.
        """
        # Explicitly use the MostActivesBy Enum to avoid Pydantic warnings
        most_actives_request = MostActivesRequest(by=MostActivesBy.VOLUME, top=10)
        most_actives_response = self.screener_client.get_most_actives(most_actives_request)
        
        # Convert list of dictionaries to Polars DataFrame
        watchlist = pl.DataFrame(most_actives_response.most_actives)
        
        # Extract ticker symbols from the 'symbol' column
        ticker_symbols = watchlist['symbol'].str.slice(1).to_list() 
    
        # Create a new DataFrame with the extracted tickers
        self.ticker_df = pl.DataFrame({'ticker': ticker_symbols}) 
        return self.ticker_df

    def get_stock_bar_data(self, stock_client, timeframe, start_date, end_date):
        """
        Fetches historical stock bar data for the tickers stored in `self.ticker_df`.

        This method uses the `StockBarsRequest` to retrieve historical bar data for the most active
        stocks, which are stored in the `self.ticker_df` DataFrame. It retrieves data from the IEX
        data feed for the specified timeframe and date range.

        Args:
            stock_client (StockHistoricalDataClient): An instance of the Alpaca StockHistoricalDataClient.
            timeframe (TimeFrame): The timeframe for the bars (e.g., TimeFrame(1, TimeFrameUnit.Day)).
            start_date (datetime): The starting date for the historical data.
            end_date (datetime): The ending date for the historical data.

        Returns:
            StockBars: A StockBars object containing the historical bar data, or None if an error occurs.
        """
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
        except alpaca.common.exceptions.APIError as e:
            print(f"Alpaca API Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None

    def get_option_chain_data(self, option_client):  # Removed ticker_symbols argument
        """
        Retrieves option chain data for the tickers stored in `self.ticker_df`.

        This method iterates through the ticker symbols in the `self.ticker_df` DataFrame and fetches
        the option chain data for each ticker using the `OptionChainRequest`. It stores the data in a
        dictionary where the keys are the ticker symbols and the values are the corresponding
        option chain objects.

        Args:
            option_client (OptionHistoricalDataClient): An instance of the Alpaca OptionHistoricalDataClient.

        Returns:
            dict: A dictionary containing option chain data for each ticker symbol.
        """
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
