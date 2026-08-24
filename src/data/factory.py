"""
Factory for MarketDataProvider and FundamentalProvider instances.

Market provider — reads ``DATA_SOURCE`` env:
    - ``alpaca``  (default) — Alpaca REST + WebSocket
    - ``polygon`` — Polygon.io REST + WebSocket
    - ``yahoo``   — Yahoo Finance (polling, **paper trading only**)
    - ``oanda``   — OANDA v20 REST + Streaming (forex, V5 forex bot)

Fundamental provider — reads ``FUNDAMENTAL_SOURCES`` env (comma-separated):
    - ``simfin``           (default) — SimFin institutional fundamentals
    - ``yfinance``/``yahoo`` — Yahoo Finance fundamentals
    - ``none``             — no fundamentals (empty responses)
    Multiple sources are chained: first non-empty result wins.

This is the single switch that decides which market the bot is trading. The
live forex bot runs with DATA_SOURCE=oanda; the equities work uses alpaca.
Vendor SDKs are imported lazily inside each branch so that, for example,
running OANDA never requires the Polygon package to be installed.

Glossary:
    get_market_provider -- reads DATA_SOURCE and returns a ready-to-use
        provider. Raises ValueError on an unrecognised value rather than
        silently defaulting, so a typo cannot quietly trade the wrong market.
    DATA_SOURCE -- "alpaca" (default), "polygon", "yahoo", or "oanda".
    PAPER_MODE -- Alpaca only; "True" (the default) routes to the paper
        account. Note the default is paper, so real money requires an explicit
        opt-out.
    OANDA_ENV -- "practice" (default) or "live", the OANDA equivalent.
    OANDA_STREAM_GRANULARITY_MIN -- bar size the live tick stream aggregates
        into, in minutes (default 1).
    YAHOO_POLL_INTERVAL -- seconds between Yahoo polls (default 60); Yahoo has
        no push feed.
    get_fundamental_provider -- builds the company-financials source used by
        the equities investor, not the forex bot.
    FUNDAMENTAL_SOURCES -- ordered comma-separated list ("simfin" default,
        also "yfinance"/"yahoo", or "none"). Sources are chained and the first
        non-empty answer wins, so a primary can fall back to a secondary.
    _REGISTRY -- maps those source names to fully-qualified class paths, which
        are imported by string so an unused vendor's package is never loaded.
"""

import logging
import os

from src.data.market_provider import MarketDataProvider

logger = logging.getLogger(__name__)


def get_market_provider() -> MarketDataProvider:
    """
    Instantiate and return the MarketDataProvider indicated by the
    ``DATA_SOURCE`` environment variable.

    Returns
    -------
    MarketDataProvider
        A fully-initialised provider ready for ``get_historical_bars``,
        ``subscribe``, and ``run_stream``.

    Raises
    ------
    ValueError
        If ``DATA_SOURCE`` is set to an unrecognised value.
    """
    source = os.getenv("DATA_SOURCE", "alpaca").strip().lower()

    if source == "alpaca":
        from src.data.alpaca_provider import AlpacaProvider

        api_key = os.getenv("alpaca_key") or os.getenv("ALPACA_API_KEY")
        secret = os.getenv("alpaca_secret") or os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret:
            raise ValueError(
                "Alpaca credentials missing. Set alpaca_key / alpaca_secret "
                "in your .env file."
            )
        is_paper = os.getenv("PAPER_MODE", "True").lower() == "true"
        logger.info("Data source: Alpaca (%s)", "paper" if is_paper else "live")
        return AlpacaProvider(api_key, secret, paper=is_paper)

    if source == "polygon":
        from src.data.polygon_provider import PolygonDataProvider

        api_key = os.getenv("poly_keys") or os.getenv("POLYGON_API_KEY")
        logger.info("Data source: Polygon")
        return PolygonDataProvider(api_key=api_key)

    if source == "yahoo":
        from src.data.yahoo_provider import YahooDataProvider

        poll = int(os.getenv("YAHOO_POLL_INTERVAL", "60"))
        logger.info("Data source: Yahoo Finance (poll every %ds)", poll)
        return YahooDataProvider(poll_interval=poll)

    if source == "oanda":
        from src.data.oanda_provider import OandaMarketProvider

        environment = os.getenv("OANDA_ENV", "practice").strip().lower()
        api_key = os.getenv("OANDA_API_KEY")
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        stream_gran = int(os.getenv("OANDA_STREAM_GRANULARITY_MIN", "1"))
        logger.info("Data source: OANDA (%s)", environment)
        return OandaMarketProvider(
            environment=environment,
            api_key=api_key,
            account_id=account_id,
            stream_granularity_minutes=stream_gran,
        )

    raise ValueError(
        f"Unknown DATA_SOURCE={source!r}. "
        f"Expected one of: alpaca, polygon, yahoo, oanda."
    )


def get_fundamental_provider():
    """
    Build a FundamentalProvider from the ``FUNDAMENTAL_SOURCES`` env var.

    ``FUNDAMENTAL_SOURCES`` is a comma-separated ordered list of source names.
    The providers are chained: the first one to return a non-empty result wins.

    Supported names
    ---------------
    simfin            SimFin institutional fundamentals (requires SIMFIN_API_KEY)
    yfinance / yahoo  Yahoo Finance (no API key; unofficial)
    none              No fundamentals — returns empty for every call

    Defaults to ``simfin`` when the variable is unset, preserving today's
    verified pipeline behaviour.

    Raises
    ------
    ValueError
        If any token in the list is not recognised.
    """
    from src.data.fundamentals import FundamentalProvider
    from src.data.providers.composite_fundamentals import CompositeFundamentalProvider

    raw = os.getenv("FUNDAMENTAL_SOURCES", "simfin").strip()

    if not raw or raw.lower() == "none":
        logger.info("Fundamental sources: none (empty responses)")
        return CompositeFundamentalProvider([])

    tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]

    _REGISTRY: dict[str, str] = {
        "simfin": "src.data.providers.simfin_fundamentals.SimFinFundamentalProvider",
        "yfinance": "src.data.providers.yf_fundamentals.YFinanceFundamentalProvider",
        "yahoo": "src.data.providers.yf_fundamentals.YFinanceFundamentalProvider",
    }

    providers: list[FundamentalProvider] = []
    for token in tokens:
        if token == "none":
            continue
        if token not in _REGISTRY:
            raise ValueError(
                f"Unknown FUNDAMENTAL_SOURCES token {token!r}. "
                f"Expected one of: {', '.join(sorted(_REGISTRY))}."
            )
        module_path, _, cls_name = _REGISTRY[token].rpartition(".")
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        providers.append(cls())

    logger.info(
        "Fundamental sources: %s",
        ", ".join(type(p).__name__ for p in providers) if providers else "none",
    )
    return CompositeFundamentalProvider(providers)
