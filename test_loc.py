import pandas as pd
import numpy as np

arrays = [
    ["BTC/USD", "BTC/USD", "ETH/USD", "ETH/USD"],
    [1, 2, 1, 2],
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=["symbol", "timestamp"])
df = pd.DataFrame({"close": [10, 20, 30, 40]}, index=index)

print(df.loc["BTC/USD"].reset_index().columns)
