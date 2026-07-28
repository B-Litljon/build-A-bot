"""
Abstract base class for macroeconomic data providers.

Concrete implementations live in ``src/data/providers/`` and must not
be imported from this module.  This file must not import any vendor SDKs.

Concrete implementations:
  - YFinanceMacroProvider  (src/data/providers/yf_macro.py)

For production use, replace the yfinance PoC adapter with a FRED-backed
implementation (fredapi) for authoritative macro series such as the Fed
Funds Rate (FEDFUNDS), CPI (CPIAUCSL), and 10Y Treasury (DGS10).

Glossary:
    MacroProvider -- the contract for economy-wide time series (interest
        rates, volatility indices) as opposed to per-company data.
    get_macro_series -- returns one daily series for a named indicator from a
        start date onward. Unknown names return an empty Series rather than
        raising, so adding an indicator cannot break an existing caller.
    indicator_name -- a human-readable key such as "VIX" or "10Y_YIELD". Each
        adapter documents the names it supports; there is no shared registry.
    start_date -- inclusive ISO-8601 "YYYY-MM-DD" string.
"""

from __future__ import annotations

import abc

import pandas as pd


class MacroProvider(abc.ABC):
    """Unified contract for sourcing macroeconomic time-series data."""

    @abc.abstractmethod
    def get_macro_series(
        self,
        indicator_name: str,
        start_date: str,
    ) -> pd.Series:
        """
        Return a daily time series for the named macro indicator.

        Parameters
        ----------
        indicator_name : str
            Human-readable key for the indicator.  Each concrete adapter
            documents its supported names (e.g. "VIX", "10Y_YIELD").
        start_date : str
            Inclusive start date in ISO-8601 format ("YYYY-MM-DD").

        Returns
        -------
        pandas.Series
            UTC-aware DatetimeIndex, daily frequency.
            Name attribute set to *indicator_name*.
            Returns an empty Series on unknown indicator or failure —
            never raises.
        """
        ...
