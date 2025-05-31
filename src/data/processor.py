import pandas as pd
import polars as pl

class DataProcessor:
    def __init__(self, foreign_df):
        """
        Initializes the DataProcessor with a Pandas DataFrame that may need formatting.

        Args:
            foreign_df (pd.DataFrame): The DataFrame from an external source.
        """
        self.foreign_df = foreign_df  # Store the original DataFrame
        self.master_df = self.format_dataframe()  # Format and store the master DataFrame

    def format_dataframe(self):
        """
        Formats the DataFrame returned from the Alpaca API.

        Returns:
            pd.DataFrame: The formatted master DataFrame with a MultiIndex.
        """
        df = self.foreign_df.copy()

        # 1. Reset index to make 'symbol' and 'timestamp' regular columns
        df.reset_index(inplace=True)

        # 2. Convert 'timestamp' to datetime (if it's not already)
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 3. Set the MultiIndex ('symbol', 'timestamp')
        df.set_index(['symbol', 'timestamp'], inplace=True)
        df.sort_index(inplace=True)

        return df

        return df
    def index_nav(self, level_name, level_value):
        """
        Navigates the multi-index DataFrame to extract data at a specified level.

        Args:
            level_name (str): The name of the level in the MultiIndex.
            level_value (str): The value to select at the specified level.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the extracted data.
        """
        try:
            # Use Pandas .xs to extract data at the specified level
            extracted_data = self.master_df.xs(level_value, level=level_name)  
        except KeyError:
            print(f"Value '{level_value}' not found at level '{level_name}' in the DataFrame.")
            return None

        # Convert to Polars DataFrame
        polars_df = pl.from_pandas(extracted_data)  
        return polars_df


    def calculate_indicator(self, polars_df, indicator_func, **kwargs):
        """
        Calculates a technical indicator using TA-Lib.

        Args:
            polars_df (pl.DataFrame): The Polars DataFrame containing OHLCV data.
            indicator_func (function): The TA-Lib indicator function to apply.
            **kwargs: Keyword arguments to pass to the indicator function.

        Returns:
            pl.DataFrame: The Polars DataFrame with the indicator added as a new column.
        """
        # Assuming your OHLCV columns are named 'open', 'high', 'low', 'close', 'volume'
        # Adjust column names if necessary
        close_prices = polars_df["close"].to_numpy()  # TA-Lib works with NumPy arrays

        # Calculate the indicator using TA-Lib
        indicator_values = indicator_func(close_prices, **kwargs)  

        # Add the indicator as a new column to the Polars DataFrame
        polars_df = polars_df.with_columns(
            pl.Series(name="indicator", values=indicator_values)
        )
        return polars_df