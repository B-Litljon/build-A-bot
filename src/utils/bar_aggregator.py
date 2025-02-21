import polars as pl

# so basically what we have is a function that accepts ohclv data as input. (alpaca only has 1m and 1d)
# we pass an argument for the timeframe we want to aggregate to. (eg: 5m, 15m, 1h, 4h)
# so this function will keep receiving the candle data and also create a list.
# when the list reaches the desired length, it will aggregate the data and return it.
# the function will then clear the list and start over.
# this function will actually handle the wss connection, the data stream, and the aggregation.
# the trading bot will only need to call this function and pass the desired timeframe.