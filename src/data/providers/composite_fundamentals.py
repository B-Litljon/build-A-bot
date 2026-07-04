"""
Composite fundamental provider — chains multiple FundamentalProvider instances
and returns the first non-empty result from each method.

v1 is first-source-wins per symbol per call.  A future v2 could merge
field-level results across sources (e.g. income statement from SimFin,
balance sheet from Yahoo when one source is missing fields), but that
requires field-level awareness that would leak provider internals into
this layer.
"""

from __future__ import annotations

import logging

import pandas as pd

from data.fundamentals import FundamentalProvider

logger = logging.getLogger(__name__)


class CompositeFundamentalProvider(FundamentalProvider):
    """
    Chains an ordered list of FundamentalProvider instances and returns the
    first non-empty result.

    Parameters
    ----------
    providers:
        Ordered list of providers.  Earlier entries take precedence.
        ``CompositeFundamentalProvider([])`` returns empty for all methods —
        the "no fundamentals" case that the feature pipeline handles gracefully.

    Contract
    --------
    Never raises.  Returns empty dict / empty DataFrame on miss.
    """

    def __init__(self, providers: list[FundamentalProvider]) -> None:
        self._providers = providers

    def get_company_info(self, symbol: str) -> dict:
        for p in self._providers:
            try:
                result = p.get_company_info(symbol)
            except Exception as exc:
                logger.debug("get_company_info %s from %s raised: %s", symbol, type(p).__name__, exc)
                result = {}
            if result:
                logger.debug("get_company_info %s answered by %s", symbol, type(p).__name__)
                return result
        return {}

    def get_valuation_metrics(self, symbol: str) -> dict:
        for p in self._providers:
            try:
                result = p.get_valuation_metrics(symbol)
            except Exception as exc:
                logger.debug("get_valuation_metrics %s from %s raised: %s", symbol, type(p).__name__, exc)
                result = {}
            if result:
                logger.debug("get_valuation_metrics %s answered by %s", symbol, type(p).__name__)
                return result
        return {}

    def get_quarterly_financials(self, symbol: str) -> pd.DataFrame:
        for p in self._providers:
            try:
                result = p.get_quarterly_financials(symbol)
            except Exception as exc:
                logger.debug("get_quarterly_financials %s from %s raised: %s", symbol, type(p).__name__, exc)
                result = pd.DataFrame()
            if not result.empty:
                logger.debug("get_quarterly_financials %s answered by %s", symbol, type(p).__name__)
                return result
        return pd.DataFrame()
