"""
Abstract base class for fundamental data providers.

Concrete implementations live in ``src/data/providers/`` and must not
be imported from this module.  This file must not import any vendor SDKs.

Concrete implementations:
  - YFinanceFundamentalProvider  (src/data/providers/yf_fundamentals.py)
"""

from __future__ import annotations

import abc

import pandas as pd


class FundamentalProvider(abc.ABC):
    """Unified contract for sourcing company fundamental data."""

    @abc.abstractmethod
    def get_company_info(self, symbol: str) -> dict:
        """
        Return static company metadata for *symbol*.

        Keys include (but are not limited to): sector, industry,
        longName, country, marketCap, currency, exchange.

        Returns an empty dict on failure — never raises.
        """
        ...

    @abc.abstractmethod
    def get_valuation_metrics(self, symbol: str) -> dict:
        """
        Return point-in-time valuation and quality ratios for *symbol*.

        Keys include (but are not limited to): trailingPE, forwardPE,
        priceToBook, enterpriseToEbitda, pegRatio,
        priceToSalesTrailingTwelveMonths, returnOnEquity, grossMargins.

        All values are floats or None.  Returns an empty dict on failure.
        """
        ...

    @abc.abstractmethod
    def get_quarterly_financials(self, symbol: str) -> pd.DataFrame:
        """
        Return quarterly income-statement data for *symbol*.

        Returns
        -------
        pandas.DataFrame
            Index  : DatetimeIndex, name="period_end" (descending — most
                     recent quarter first).
            Columns: accounting line items
                     (e.g. "Total Revenue", "Gross Profit", "Net Income").
            Returns an empty DataFrame on failure — never raises.
        """
        ...
