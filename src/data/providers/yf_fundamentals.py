"""
FundamentalProvider backed by Yahoo Finance (yfinance).

**For Proof-of-Concept / Research Only.**

Yahoo Finance data is unofficial, subject to rate limits, and may differ
from authoritative sources (SEC EDGAR, SimFin, Bloomberg).  Use this
adapter for feature prototyping and backtesting only.  Do not rely on it
for production capital allocation decisions.

No API key is required.

Glossary:
    YFinanceFundamentalProvider -- pulls company data from Yahoo. Returns
        empty on any failure, which the composite provider treats as "try the
        next source".
    _COMPANY_INFO_KEYS -- the subset of Yahoo's large, unstable info blob this
        adapter actually reads (name, sector, industry, country, ...). Pinning
        the list means an upstream schema change drops fields instead of
        crashing.
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
        Return quarterly income-statement + balance-sheet data for *symbol*,
        normalized to the SimFin-style column names the V4 feature pipeline
        expects.

        Pulls income statement (``quarterly_financials``) and balance sheet
        (``quarterly_balance_sheet``) and merges them on the period-end index.
        If the balance sheet fetch fails, the income statement is returned
        alone (the never-raise contract is preserved).

        Column mapping (yfinance native → contract name):

        ====================================  =======================
        yfinance name                         contract name
        ====================================  =======================
        Total Revenue                         Total Revenue  (unchanged)
        Gross Profit                          Gross Profit   (unchanged)
        Net Income                            Net Income     (unchanged)
        Diluted Average Shares                Shares (Diluted)
        Total Assets                          Total Assets   (unchanged)
        Stockholders Equity                   Total Equity
        Total Liabilities Net Minority Int.   Total Liabilities
        ====================================  =======================

        Returns an empty DataFrame on any failure.
        """
        # yfinance native → pipeline contract
        _RENAME: dict[str, str] = {
            "Diluted Average Shares": "Shares (Diluted)",
            "Stockholders Equity": "Total Equity",
            "Total Liabilities Net Minority Interest": "Total Liabilities",
        }

        try:
            ticker = yf.Ticker(symbol)

            raw_inc = ticker.quarterly_financials
            if raw_inc is None or raw_inc.empty:
                logger.warning("No quarterly financials returned for %s.", symbol)
                return pd.DataFrame()

            inc = raw_inc.T.copy()
            inc.index = pd.DatetimeIndex(inc.index)
            inc.index.name = "period_end"

            # Pull balance sheet and merge; degrade gracefully on failure
            try:
                raw_bs = ticker.quarterly_balance_sheet
                if raw_bs is not None and not raw_bs.empty:
                    bs = raw_bs.T.copy()
                    bs.index = pd.DatetimeIndex(bs.index)
                    bs.index.name = "period_end"
                    # Keep only balance-sheet columns not already in income stmt
                    bs_new_cols = [c for c in bs.columns if c not in inc.columns]
                    inc = inc.join(bs[bs_new_cols], how="left")
            except Exception as exc:
                logger.warning(
                    "get_quarterly_financials balance sheet failed for %s (income only): %s",
                    symbol, exc,
                )

            inc = inc.rename(columns=_RENAME)
            inc = inc.sort_index(ascending=False)
            return inc

        except Exception as exc:
            logger.warning(
                "get_quarterly_financials failed for %s: %s", symbol, exc
            )
            return pd.DataFrame()
