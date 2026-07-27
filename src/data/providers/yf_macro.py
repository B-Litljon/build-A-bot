"""
MacroProvider backed by Yahoo Finance (yfinance).

**For Proof-of-Concept / Research Only.**

Yahoo Finance carries a limited set of macro proxies as publicly-traded
instruments or indices.  For authoritative macroeconomic series, replace
this adapter with a FRED-backed implementation (fredapi) — see the note
in src/data/macro.py.

No API key is required.

Glossary:
    YFinanceMacroProvider -- maps friendly indicator names onto Yahoo tickers
        and returns the resulting daily series.
    _INDICATOR_MAP -- the name-to-ticker table: VIX, 10Y_YIELD, 2Y_YIELD,
        SP500, NASDAQ, DJI, GOLD, OIL, DXY. Extend it to add an indicator.
    ^TNX (10Y_YIELD) -- CAUTION: Yahoo reports this as the yield times ten, so
        a 4.2% yield arrives as 42.0. Nothing in this module rescales it.
    ^IRX (2Y_YIELD) -- MISNAMED: this is the 13-week Treasury bill rate, not
        the 2-year. It is the closest free Yahoo proxy; use FRED's DGS2 if the
        real 2-year matters.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from data.macro import MacroProvider

logger = logging.getLogger(__name__)

# Mapping of human-readable indicator names to Yahoo Finance tickers.
# Extend this dict as new macro proxies are needed.
#
# NOTE on yield indices:
#   ^TNX reports the 10Y Treasury yield multiplied by 10 (Yahoo convention).
#   ^IRX reports the 13-week T-Bill discount rate, not the 2Y — it is the
#   closest free proxy available via Yahoo Finance.
#   For true 2Y yield, use FRED series DGS2.
_INDICATOR_MAP: dict[str, str] = {
    "VIX": "^VIX",
    "10Y_YIELD": "^TNX",
    "2Y_YIELD": "^IRX",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DJI": "^DJI",
    "GOLD": "GC=F",
    "OIL": "CL=F",
    "DXY": "DX-Y.NYB",
}


class YFinanceMacroProvider(MacroProvider):
    """
    Yahoo Finance adapter for macroeconomic time-series indicators.

    .. warning::
        **For Proof-of-Concept / Research Only.**

        Yield proxies sourced from CBOE index tickers (^TNX, ^IRX) are
        indicative only and require scaling for percentage conversion.
        Replace with FRED for production use.

    Supported indicator names
    -------------------------
    VIX, 10Y_YIELD, 2Y_YIELD, SP500, NASDAQ, DJI, GOLD, OIL, DXY

    Parameters
    ----------
    None — no credentials required.
    """

    # ── MacroProvider interface ───────────────────────────────────────

    def get_macro_series(
        self,
        indicator_name: str,
        start_date: str,
    ) -> pd.Series:
        """
        Download a daily close-price time series for *indicator_name*.

        Parameters
        ----------
        indicator_name : str
            One of the supported names listed in the class docstring.
            Case-sensitive.
        start_date : str
            Inclusive start in ISO-8601 format ("YYYY-MM-DD").

        Returns
        -------
        pd.Series
            UTC-aware DatetimeIndex, daily frequency, name=indicator_name.
            Empty Series on unknown indicator or download failure.
        """
        ticker_symbol = _INDICATOR_MAP.get(indicator_name)
        if ticker_symbol is None:
            known = ", ".join(sorted(_INDICATOR_MAP))
            logger.warning(
                "Unknown indicator '%s'. Supported names: %s",
                indicator_name,
                known,
            )
            return pd.Series(name=indicator_name, dtype=float)

        try:
            raw = yf.download(
                tickers=ticker_symbol,
                start=start_date,
                progress=False,
                auto_adjust=True,
            )

            if raw is None or raw.empty:
                logger.warning(
                    "yf.download returned no data for %s (%s).",
                    indicator_name,
                    ticker_symbol,
                )
                return pd.Series(name=indicator_name, dtype=float)

            # yf.download may return MultiIndex columns — flatten to get
            # a simple 'Close' key regardless of yfinance version.
            if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
                raw.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in raw.columns
                ]

            if "Close" not in raw.columns:
                logger.warning(
                    "No 'Close' column in yf.download result for %s (%s).",
                    indicator_name,
                    ticker_symbol,
                )
                return pd.Series(name=indicator_name, dtype=float)

            series = raw["Close"].copy()
            series.name = indicator_name

            # Ensure UTC-aware index
            if series.index.tz is None:
                series.index = series.index.tz_localize("UTC")
            else:
                series.index = series.index.tz_convert("UTC")

            return series.sort_index()

        except Exception as exc:
            logger.warning(
                "get_macro_series failed for %s (%s): %s",
                indicator_name,
                ticker_symbol,
                exc,
            )
            return pd.Series(name=indicator_name, dtype=float)
