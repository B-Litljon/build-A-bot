import os
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import core.retrainer as R
from data.factory import get_market_provider
from scripts.run_catboost_ab import _fetch_cached

def prefetch():
    provider = get_market_provider()
    symbols = R.get_asset_config("oanda")["tickers"]
    days_back = int(os.environ.get("CB_DAYS_BACK", "730"))
    
    print(f"=== Prefetching H1 (60m) bars for {symbols} ({days_back} days) ===")
    _fetch_cached(provider, symbols, days_back, 60)
    print("=== H1 prefetch complete! ===")
    
    print(f"=== Prefetching H4 (240m) bars for {symbols} ({days_back} days) ===")
    _fetch_cached(provider, symbols, days_back, 240)
    print("=== H4 prefetch complete! ===")

if __name__ == "__main__":
    prefetch()
