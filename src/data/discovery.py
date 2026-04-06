import logging
from typing import List
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest

logger = logging.getLogger(__name__)


class DiscoveryService:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

    def get_in_play_tickers(
        self,
        min_price: float = 10.0,
        max_price: float = 200.0,
        min_gap_pct: float = 2.0,
        top_n: int = 50,
    ) -> List[str]:
        """
        Scans market for tickers matching price and gap criteria.
        WARNING: Alpaca Free Tier (IEX) volume data is fragmented. RVOL/Gap metrics are approximated.
        """
        logger.warning(
            "DiscoveryService initialized with IEX routing. Volume metrics are approximations."
        )

        req = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE
        )
        assets = self.trading_client.get_all_assets(req)
        symbols = [
            a.symbol for a in assets if a.tradable and a.marginable and a.fractionable
        ]

        in_play = []
        chunk_size = 1000

        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            try:
                snap_req = StockSnapshotRequest(symbol_or_symbols=chunk)
                snapshots = self.data_client.get_stock_snapshot(snap_req)

                for symbol, snapshot in snapshots.items():
                    if (
                        not snapshot
                        or not snapshot.daily_bar
                        or not snapshot.previous_daily_bar
                    ):
                        continue

                    close_price = snapshot.daily_bar.close
                    prev_close = snapshot.previous_daily_bar.close

                    if close_price == 0 or prev_close == 0:
                        continue

                    gap_pct = ((close_price - prev_close) / prev_close) * 100

                    if min_price <= close_price <= max_price and gap_pct >= min_gap_pct:
                        in_play.append(
                            {
                                "symbol": symbol,
                                "price": close_price,
                                "gap_pct": gap_pct,
                                "volume": snapshot.daily_bar.volume,
                            }
                        )
            except Exception as e:
                logger.error(f"[DiscoveryService] Error fetching chunk {i}: {e}")

        in_play.sort(key=lambda x: x["volume"], reverse=True)
        return [x["symbol"] for x in in_play[:top_n]]
