import sys
import os
from abc import ABC, abstractmethod
from typing import List, Dict
from core.signal import Signal
from core.order_management import OrderParams
import talib
import pandas as pd
import numpy as np
import polars as pl 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Methods:
        analyze(self, data): Analyzes market data and returns a list of Signals.
        get_order_params(self): Returns default OrderParams for the strategy.
    """
    @abstractmethod
    def analyze(self, data) -> List[Signal]:
        """
        Analyzes market data and returns a list of trading signals.

        Args:
            data: Market data (format depends on your DataHandler).

        Returns:
            List[Signal]: A list of trading signals.
        """
        pass

    @abstractmethod
    def get_order_params(self) -> OrderParams:
        """
        Returns the default OrderParams for this strategy.

        Returns:
            OrderParams: An OrderParams object.
        """
        pass
