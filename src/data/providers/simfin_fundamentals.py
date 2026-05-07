"""
FundamentalProvider backed by SimFin (institutional-grade SEC fundamentals).

SimFin uses a *bulk-download* data model: full datasets (income, balance,
derived ratios, companies, industries) are downloaded once per refresh
window and cached on local disk as CSVs.  Per-symbol calls then slice the
in-memory frames — no per-call network round-trips.

Three sector variants are swept for income/balance/derived data:
``general`` (most companies), ``banks``, and ``insurance``.  Banks and
insurance carriers report different line items, so SimFin partitions them
into separate datasets.  ``_find_in_variants`` resolves a ticker to its
correct partition transparently.

Column renames preserve downstream compatibility with the existing
yfinance-shaped contract (``Total Revenue``, ``Operating Income``) so the
V4 feature pipeline (`scripts/investor_feature_pipeline.py`) requires no
changes.

Requires ``SIMFIN_API_KEY`` in the process environment (loaded via
``python-dotenv`` at the orchestrator).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import simfin as sf

from data.fundamentals import FundamentalProvider

logger = logging.getLogger(__name__)

# Project-local cache (keeps SimFin CSVs alongside other V4 raw data)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / "raw" / "simfin_cache"

# Re-download bulk CSVs only if older than this many days (SimFin default)
_REFRESH_DAYS_DEFAULT = 30

# Sector variants — JPM (banks), AIG/MET (insurance), most others (general)
_VARIANTS: tuple[str, ...] = ("general", "banks", "insurance")

# ── Loader registries ────────────────────────────────────────────────
# SimFin exposes one loader function per (statement, variant) pair.

_INCOME_LOADERS: dict[str, Callable] = {
    "general": sf.load_income,
    "banks": sf.load_income_banks,
    "insurance": sf.load_income_insurance,
}

_BALANCE_LOADERS: dict[str, Callable] = {
    "general": sf.load_balance,
    "banks": sf.load_balance_banks,
    "insurance": sf.load_balance_insurance,
}

_DERIVED_LOADERS: dict[str, Callable] = {
    "general": sf.load_derived,
    "banks": sf.load_derived_banks,
    "insurance": sf.load_derived_insurance,
}

# ── Column rename maps ───────────────────────────────────────────────
# Map SimFin's native column names to the yfinance-style names that the
# V4 feature pipeline expects.  Only renames known collisions; SimFin
# native names (e.g. "Cost of Revenue", "Total Assets") pass through.

_INCOME_RENAME: dict[str, str] = {
    "Revenue": "Total Revenue",
    "Operating Income (Loss)": "Operating Income",
    "Pretax Income (Loss)": "Pretax Income",
}

# Derived dataset → valuation/quality/growth ratios.  SimFin column names
# may include qualifiers (e.g. "Price to Earnings Ratio (quarterly)") so
# we accept either form via best-effort fallback in _coerce_metric.
_VALUATION_FIELDS: dict[str, tuple[str, ...]] = {
    # Value
    "trailingPE": ("Price to Earnings Ratio (quarterly)", "Price to Earnings Ratio"),
    "priceToBook": ("Price to Book Value",),
    "enterpriseToEbitda": ("EV/EBITDA",),
    "priceToSalesTrailingTwelveMonths": (
        "Price to Sales Ratio (quarterly)",
        "Price to Sales Ratio",
    ),
    "marketCap": ("Market-Cap", "Market Cap"),
    "enterpriseValue": ("Enterprise Value",),
    # Quality
    "returnOnEquity": ("Return on Equity",),
    "returnOnAssets": ("Return on Assets",),
    "grossMargins": ("Gross Profit Margin",),
    "operatingMargins": ("Operating Margin",),
    "profitMargins": ("Net Profit Margin",),
    "debtToEquity": ("Liabilities to Equity Ratio",),
    "currentRatio": ("Current Ratio",),
    # Growth (proxy)
    "trailingEps": ("Earnings Per Share, Basic",),
}

# ABC-contract fields with no SimFin analogue — return None for these
_VALUATION_NULL_FIELDS: tuple[str, ...] = (
    "forwardPE",
    "pegRatio",
    "revenueGrowth",
    "earningsGrowth",
    "forwardEps",
)


def _coerce_metric(row: pd.Series, candidates: tuple[str, ...]) -> float | None:
    """Return the first non-null value among *candidates*, cast to float."""
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
    return None


class SimFinFundamentalProvider(FundamentalProvider):
    """
    SimFin adapter for company fundamentals.

    Parameters
    ----------
    api_key : str | None
        SimFin API key.  Defaults to ``os.environ["SIMFIN_API_KEY"]``.
    data_dir : str | os.PathLike | None
        Local cache directory for bulk-download CSVs.  Defaults to
        ``<project>/data/raw/simfin_cache/``.
    market : str
        SimFin market code.  Default ``"us"``.
    refresh_days : int
        Re-download cached datasets older than this many days.
    """

    def __init__(
        self,
        api_key: str | None = None,
        data_dir: str | os.PathLike | None = None,
        market: str = "us",
        refresh_days: int = _REFRESH_DAYS_DEFAULT,
    ) -> None:
        api_key = api_key or os.environ.get("SIMFIN_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SIMFIN_API_KEY missing — set it in .env or pass api_key=..."
            )

        sf.set_api_key(api_key)

        cache = Path(data_dir) if data_dir is not None else _DEFAULT_CACHE_DIR
        cache.mkdir(parents=True, exist_ok=True)
        sf.set_data_dir(str(cache))

        self._market = market
        self._refresh_days = refresh_days
        # In-memory frame cache — populated lazily on first access
        self._frames: dict[str, pd.DataFrame] = {}

    # ── lazy bulk loaders ────────────────────────────────────────────

    def _load(self, key: str, fn: Callable, **kwargs) -> pd.DataFrame:
        """Wrap a SimFin loader with cache + exception-safe fallback."""
        if key in self._frames:
            return self._frames[key]
        try:
            df = fn(refresh_days=self._refresh_days, **kwargs)
        except Exception as exc:
            logger.warning("SimFin load(%s) failed: %s", key, exc)
            df = pd.DataFrame()
        self._frames[key] = df
        return df

    def _companies(self) -> pd.DataFrame:
        # load_companies has market='us' and index='Ticker' baked in
        return self._load("companies", sf.load_companies)

    def _industries(self) -> pd.DataFrame:
        # load_industries has index='IndustryId' baked in
        return self._load("industries", sf.load_industries)

    def _statement(
        self, kind: str, variant: str
    ) -> pd.DataFrame:
        """Lazy-load (income | balance | derived) × (variant) and cache."""
        registry = {
            "income": _INCOME_LOADERS,
            "balance": _BALANCE_LOADERS,
            "derived": _DERIVED_LOADERS,
        }[kind]
        return self._load(
            f"{kind}_{variant}",
            registry[variant],
            variant="quarterly",
            market=self._market,
            index=["Ticker", "Report Date"],
        )

    # ── per-symbol slicing ───────────────────────────────────────────

    @staticmethod
    def _slice_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Slice (Ticker, Report Date)-indexed frame down to *symbol* only."""
        if df.empty:
            return pd.DataFrame()
        try:
            sub = df.xs(symbol, level="Ticker", drop_level=True)
        except KeyError:
            return pd.DataFrame()
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        return sub.sort_index(ascending=False)

    def _find_in_variants(self, symbol: str, kind: str) -> pd.DataFrame:
        """
        Try general → banks → insurance and return the first variant that
        contains *symbol*.  Empty DataFrame if absent everywhere.
        """
        for variant in _VARIANTS:
            sub = self._slice_symbol(self._statement(kind, variant), symbol)
            if not sub.empty:
                return sub
        return pd.DataFrame()

    # ── FundamentalProvider interface ────────────────────────────────

    def get_company_info(self, symbol: str) -> dict:
        """Return static company metadata for *symbol*."""
        try:
            companies = self._companies()
            if companies.empty or symbol not in companies.index:
                logger.warning("SimFin has no company record for %s.", symbol)
                return {}

            row = companies.loc[symbol]

            info: dict = {
                "longName": row.get("Company Name"),
                "fullTimeEmployees": row.get("Number Employees"),
                "exchange": row.get("Market"),
                "currency": row.get("Main Currency"),
                "country": "United States" if self._market == "us" else None,
                "quoteType": "EQUITY",
                "website": None,  # not present in SimFin companies dataset
            }

            # Resolve sector / industry via the industries lookup table
            industries = self._industries()
            industry_id = row.get("IndustryId")
            if (
                not industries.empty
                and industry_id is not None
                and industry_id in industries.index
            ):
                ind = industries.loc[industry_id]
                info["sector"] = ind.get("Sector")
                info["industry"] = ind.get("Industry")
            else:
                info["sector"] = None
                info["industry"] = None

            # Backfill marketCap from the most recent derived snapshot
            der = self._find_in_variants(symbol, "derived")
            if not der.empty:
                info["marketCap"] = _coerce_metric(
                    der.iloc[0], ("Market-Cap", "Market Cap")
                )
            else:
                info["marketCap"] = None

            return info

        except Exception as exc:
            logger.warning("get_company_info failed for %s: %s", symbol, exc)
            return {}

    def get_valuation_metrics(self, symbol: str) -> dict:
        """Return point-in-time valuation/quality/growth ratios for *symbol*."""
        try:
            der = self._find_in_variants(symbol, "derived")
            if der.empty:
                logger.warning("SimFin derived ratios empty for %s.", symbol)
                return {}

            latest = der.iloc[0]  # most recent quarter (descending sort)

            metrics: dict[str, float | None] = {
                friendly: _coerce_metric(latest, candidates)
                for friendly, candidates in _VALUATION_FIELDS.items()
            }
            for null_field in _VALUATION_NULL_FIELDS:
                metrics[null_field] = None
            return metrics

        except Exception as exc:
            logger.warning(
                "get_valuation_metrics failed for %s: %s", symbol, exc
            )
            return {}

    def get_quarterly_financials(self, symbol: str) -> pd.DataFrame:
        """
        Return quarterly income-statement and balance-sheet data for *symbol*.

        Index is ``DatetimeIndex`` named ``"period_end"`` (descending — most
        recent quarter first).  Column names follow yfinance conventions
        (``Total Revenue``, ``Operating Income``) so the V4 feature pipeline
        works unchanged.
        """
        try:
            income = self._find_in_variants(symbol, "income")
            balance = self._find_in_variants(symbol, "balance")

            if income.empty and balance.empty:
                logger.warning("No SimFin financials for %s.", symbol)
                return pd.DataFrame()

            # Drop metadata columns that don't belong in a feature frame
            _META_COLS = (
                "SimFinId",
                "Currency",
                "Fiscal Year",
                "Fiscal Period",
                "Publish Date",
                "Restated Date",
                "Shares (Basic)",
                "Shares (Diluted)",
            )
            for df in (income, balance):
                drop = [c for c in _META_COLS if c in df.columns]
                if drop:
                    df.drop(columns=drop, inplace=True)

            if income.empty:
                merged = balance
            elif balance.empty:
                merged = income.rename(columns=_INCOME_RENAME)
            else:
                merged = income.rename(columns=_INCOME_RENAME).join(
                    balance, how="outer", rsuffix="_balance"
                )

            merged.index = pd.DatetimeIndex(merged.index)
            merged.index.name = "period_end"
            return merged.sort_index(ascending=False)

        except Exception as exc:
            logger.warning(
                "get_quarterly_financials failed for %s: %s", symbol, exc
            )
            return pd.DataFrame()
