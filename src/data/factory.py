"""
Factory for MarketDataProvider instances.

Reads the ``DATA_SOURCE`` environment variable and returns the
appropriate provider.  Supported values:

    - ``alpaca``  (default) — Alpaca REST + WebSocket
    - ``polygon`` — Polygon.io REST + WebSocket
    - ``yahoo``   — Yahoo Finance (polling, **paper trading only**)
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

    raise ValueError(
        f"Unknown DATA_SOURCE={source!r}. Expected one of: alpaca, polygon, yahoo."
    )
