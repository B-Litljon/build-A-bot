import abc
from typing import List, Dict, Callable
import polars as pl

class MarketDataFeed(abc.ABC):
    @abc.abstractmethod
    async def warmup_history(self, symbols: List[str], lookback_minutes: int) -> Dict[str, pl.DataFrame]:
        pass

    @abc.abstractmethod
    async def subscribe(self, symbols: List[str], on_tick: Callable):
        pass

    @abc.abstractmethod
    async def stop(self):
        pass

class AlpacaCryptoFeed(MarketDataFeed):
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key

    async def warmup_history(self, symbols: List[str], lookback_minutes: int) -> Dict[str, pl.DataFrame]:
        # Minimal dummy implementation since it's missing in the working tree
        return {s: pl.DataFrame() for s in symbols}

    async def subscribe(self, symbols: List[str], on_tick: Callable):
        pass

    async def stop(self):
        pass
