"""
Factory for MarketDataProvider instances.

Reads the ``DATA_SOURCE`` environment variable and returns the
appropriate provider.  Supported values:

    - ``alpaca``  (default) — Alpaca REST + WebSocket
    - ``polygon`` — Polygon.io REST + WebSocket
    - ``yahoo``   — Yahoo Finance (polling, **paper trading only**)
    - ``oanda``   — OANDA v20 REST + Streaming (forex, V5 scalper)
"""

import logging
import os

from data.market_provider import MarketDataProvider

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
        from data.alpaca_provider import AlpacaProvider

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
        from data.polygon_provider import PolygonDataProvider

        api_key = os.getenv("poly_keys") or os.getenv("POLYGON_API_KEY")
        logger.info("Data source: Polygon")
        return PolygonDataProvider(api_key=api_key)

    if source == "yahoo":
        from data.yahoo_provider import YahooDataProvider

        poll = int(os.getenv("YAHOO_POLL_INTERVAL", "60"))
        logger.info("Data source: Yahoo Finance (poll every %ds)", poll)
        return YahooDataProvider(poll_interval=poll)

    if source == "oanda":
        from data.oanda_provider import OandaMarketProvider

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
