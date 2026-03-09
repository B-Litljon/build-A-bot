#!/usr/bin/env python3
import sys, os, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.abspath("src"))

import polars as pl
from datetime import datetime, timezone
from strategies.concrete_strategies.ml_strategy import MLStrategy
from utils.bar_aggregator import LiveBarAggregator
import time

# Load data
df = pl.read_parquet("data/raw/SPY_1min.parquet")
df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end = datetime(2024, 1, 3, tzinfo=timezone.utc)  # Just 2 days
test_df = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))

print(f"Testing with {len(test_df)} bars (2 days)")

strategy = MLStrategy(model_path="src/ml/models/rf_model.joblib", threshold=0.50)
lba = LiveBarAggregator(timeframe=1, history_size=400)
symbol = "SPY"

total_analysis_time = 0
analysis_count = 0

start_time = time.time()

for i, row in enumerate(test_df.iter_rows(named=True)):
    is_new = lba.add_bar({
        "timestamp": row["timestamp"],
        "open": row["open"],
        "high": row["high"],
        "low": row["low"],
        "close": row["close"],
        "volume": row["volume"],
    })
    
    if is_new:
        hist = lba.history_df
        if len(hist) >= strategy.warmup_period:
            t0 = time.time()
            sigs = strategy.analyze({symbol: hist})
            t1 = time.time()
            total_analysis_time += (t1 - t0)
            analysis_count += 1

elapsed = time.time() - start_time
print(f"Total time: {elapsed:.1f}s")
print(f"Analysis calls: {analysis_count}")
print(f"Avg analysis time: {total_analysis_time/analysis_count*1000:.1f}ms" if analysis_count > 0 else "No analysis calls")
print(f"Bars/sec: {len(test_df)/elapsed:.0f}")
