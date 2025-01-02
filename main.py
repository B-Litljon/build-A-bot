from src.data.api_requests import AlpacaClient
import os
import pretty_errors
from dotenv import load_dotenv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import polars as pl  

load_dotenv()
alpaca_key = os.getenv("alpaca_key")
alpaca_secret = os.getenv("alpaca_secret")


alpaca_client = AlpacaClient(alpaca_key, alpaca_secret)
"""
dev log: 
 12-20-2024: started working on the data processor; but now that I think about it I should probably just focus on creating the logic 
 to actually get live data and, calculate indicators for ONE single stock and then write the trigger logic for the trading bot, as well as 
 the logic for calculating the order size and placing the order. similar to how we built mk0 before. you got a bit ahead of yourself and thats alright
 just remember the goal is to get a working trading bot, that wins 51% of the time. that is mvp. 
 after we achieve that we can add in all the other bells and whistles like machine learning 

 refer back to the source code for mk0 and look for areas you can improve. for example:
    1. we calculate the order size, we calculate the exit and entry points. 
        perhaps we can add some machine learning there to determine if the trade is worth it or not

    things like that. make what works, work better.
"""
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
    stock_bar_df = stock_bar_data.df
    # Now you can print or process the stock_bar_data as needed
    print(stock_bar_df)
    # now that we have the data in a dataframe ready to be processed, we can sort the data by ticker symbol into individual polars dataframes for each stock to simplify processing later
    # iterate through each ticker symbol in the index

    # get the data for each ticker and store it in its own dataframe named after the ticker

if __name__ == "__main__":  # Add this block
    __main__()