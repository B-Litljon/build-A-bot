"""
FundamentalProvider backed by Yahoo Finance (yfinance).

**For Proof-of-Concept / Research Only.**

Yahoo Finance data is unofficial, subject to rate limits, and may differ
from authoritative sources (SEC EDGAR, SimFin, Bloomberg).  Use this
adapter for feature prototyping and backtesting only.  Do not rely on it
for production capital allocation decisions.

No API key is required.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from data.fundamentals import FundamentalProvider

logger = logging.getLogger(__name__)

# Fields extracted by get_company_info
_COMPANY_INFO_KEYS: list[str] = [
    "longName",
    "sector",
    "industry",
    "country",
    "fullTimeEmployees",
    "website",
    "marketCap",
    "currency",
    "exchange",
    "quoteType",
]

# Fields extracted by get_valuation_metrics
# Covers Value, Quality, and Growth factor families.
_VALUATION_KEYS: list[str] = [
    # Value
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "enterpriseToEbitda",
    "pegRatio",
    "priceToSalesTrailingTwelveMonths",
    "marketCap",
    "enterpriseValue",
    # Quality
    "returnOnEquity",
    "returnOnAssets",
    "grossMargins",
    "operatingMargins",
    "profitMargins",
    "debtToEquity",
    "currentRatio",
    # Growth
    "revenueGrowth",
    "earningsGrowth",
    "trailingEps",
    "forwardEps",
]


class YFinanceFundamentalProvider(FundamentalProvider):
    """
    Yahoo Finance adapter for company fundamentals.

    .. warning::
        **For Proof-of-Concept / Research Only.**

        All data is sourced from Yahoo Finance's unofficial API via
        yfinance.  Treat values as indicative only and validate against
        SEC EDGAR before use in production.

    Parameters
    ----------
    None — no credentials required.
    """

    # ── FundamentalProvider interface ─────────────────────────────────

    def get_company_info(self, symbol: str) -> dict:
        """
        Return static company metadata for *symbol*.

        Uses .get() on every key — never raises on missing fields.
        """
        try:
            info = yf.Ticker(symbol).info
            return {k: info.get(k) for k in _COMPANY_INFO_KEYS}
        except Exception as exc:
            logger.warning("get_company_info failed for %s: %s", symbol, exc)
            return {}

    def get_valuation_metrics(self, symbol: str) -> dict:
        """
        Return valuation, quality, and growth ratios for *symbol*.

        All numeric values are floats or None (never KeyError).
        """
        try:
            info = yf.Ticker(symbol).info
            return {k: info.get(k) for k in _VALUATION_KEYS}
        except Exception as exc:
            logger.warning("get_valuation_metrics failed for %s: %s", symbol, exc)
            return {}

    def get_quarterly_financials(self, symbol: str) -> pd.DataFrame:
        """
        Return quarterly income-statement data for *symbol*.

        Transposes yfinance's column-major layout (line items as rows,
        dates as columns) so that dates become the index and line items
        become columns.  Index is sorted descending — most recent first.

        Returns an empty DataFrame on any failure.
        """
        try:
            raw = yf.Ticker(symbol).quarterly_financials
            if raw is None or raw.empty:
                logger.warning("No quarterly financials returned for %s.", symbol)
                return pd.DataFrame()

            # raw shape: rows = line items, columns = period-end dates
            # After .T: rows = period-end dates, columns = line items
            df = raw.T.copy()
            df.index = pd.DatetimeIndex(df.index)
            df.index.name = "period_end"
            df = df.sort_index(ascending=False)
            return df

        except Exception as exc:
            logger.warning(
                "get_quarterly_financials failed for %s: %s", symbol, exc
            )
            return pd.DataFrame()
