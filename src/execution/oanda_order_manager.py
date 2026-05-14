"""
FIFO-compliant order/position state manager for OANDA v20.

Foundation class for the V5 forex scalper execution path. Tracks the
*net* position per instrument (signed units + broker-reported average
entry price) — never per-trade lots — to comply with U.S. NFA FIFO and
no-hedging rules enforced by OANDA.

Required environment variables:
    OANDA_API_KEY       - Bearer token from hub.oanda.com
    OANDA_ACCOUNT_ID    - Account ID (numeric string)

Scope of this module: state + close. Entry methods, fill-stream
consumers, and watchdog wiring live in separate modules.
"""

import logging
import os
import threading
from typing import Dict, Optional

import oandapyV20
import oandapyV20.endpoints.positions as v20_positions

logger = logging.getLogger(__name__)


def _to_oanda_symbol(symbol: str) -> str:
    """Normalise 'EUR/USD', 'EURUSD', or 'EUR_USD' → 'EUR_USD'."""
    return symbol.replace("/", "_").upper()


class OandaOrderManager:
    """
    OANDA v20 net-position manager.

    Holds at most one signed net position per instrument. Designed for
    NFA FIFO compliance — no per-trade lot tracking, no hedged
    long+short on the same instrument.

    Parameters
    ----------
    environment : str
        ``'practice'`` for paper trading, ``'live'`` for real money.
    api_key : str, optional
        Falls back to the ``OANDA_API_KEY`` environment variable.
    account_id : str, optional
        Falls back to the ``OANDA_ACCOUNT_ID`` environment variable.
    """

    def __init__(
        self,
        environment: str = "practice",
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
    ):
        self._api_key = api_key or os.getenv("OANDA_API_KEY")
        self._account_id = account_id or os.getenv("OANDA_ACCOUNT_ID")
        if not self._api_key:
            raise ValueError(
                "OANDA API key required. Set OANDA_API_KEY or pass api_key=."
            )
        if not self._account_id:
            raise ValueError(
                "OANDA account ID required. Set OANDA_ACCOUNT_ID or pass account_id=."
            )

        self._environment = environment
        self._client = oandapyV20.API(
            access_token=self._api_key,
            environment=environment,
        )

        self._state_lock = threading.RLock()
        self._net_positions: Dict[str, int] = {}
        self._avg_entry_prices: Dict[str, float] = {}

        logger.info(
            "OandaOrderManager initialized (environment=%s).", environment
        )

    # ── state accessors ───────────────────────────────────────────────

    def get_net_position(self, instrument: str) -> int:
        """Signed net units (positive=long, negative=short, 0=flat)."""
        with self._state_lock:
            return self._net_positions.get(_to_oanda_symbol(instrument), 0)

    def get_average_entry_price(self, instrument: str) -> float:
        """Broker-reported average entry price; 0.0 when flat."""
        with self._state_lock:
            return self._avg_entry_prices.get(_to_oanda_symbol(instrument), 0.0)

    # ── broker sync ───────────────────────────────────────────────────

    def sync_position(self, instrument: str) -> None:
        """
        Pull authoritative net-position state from OANDA's
        ``/v3/accounts/{id}/positions/{instrument}`` endpoint and
        refresh internal state.

        Under FIFO/no-hedging, exactly one of ``long`` or ``short`` will
        carry non-zero units for any given instrument; OANDA returns
        short units as a negative numeric string.
        """
        oanda_symbol = _to_oanda_symbol(instrument)
        try:
            req = v20_positions.PositionDetails(
                accountID=self._account_id, instrument=oanda_symbol
            )
            self._client.request(req)
            position = req.response.get("position", {})

            long_side = position.get("long", {}) or {}
            short_side = position.get("short", {}) or {}
            long_units = int(float(long_side.get("units", "0") or "0"))
            short_units = int(float(short_side.get("units", "0") or "0"))

            with self._state_lock:
                if long_units > 0:
                    self._net_positions[oanda_symbol] = long_units
                    self._avg_entry_prices[oanda_symbol] = float(
                        long_side.get("averagePrice", "0") or "0"
                    )
                elif short_units < 0:
                    self._net_positions[oanda_symbol] = short_units
                    self._avg_entry_prices[oanda_symbol] = float(
                        short_side.get("averagePrice", "0") or "0"
                    )
                else:
                    self._net_positions[oanda_symbol] = 0
                    self._avg_entry_prices[oanda_symbol] = 0.0

                logger.info(
                    "[%s] OandaOrderManager sync | net=%d | avg=%.5f",
                    oanda_symbol,
                    self._net_positions[oanda_symbol],
                    self._avg_entry_prices[oanda_symbol],
                )
        except Exception as e:
            logger.error(
                "[%s] OandaOrderManager.sync_position failed: %s",
                oanda_symbol,
                e,
                exc_info=True,
            )

    # ── FIFO close ────────────────────────────────────────────────────

    def close_position(self, instrument: str) -> bool:
        """
        Flatten the net position for *instrument* via OANDA's
        ``/positions/{instrument}/close`` endpoint.

        Uses ``"ALL"`` semantics so the broker liquidates whatever is
        actually open, even if local state has drifted. Returns True if
        a close request was submitted, False if already flat or on
        error (state untouched on error so a retry can be attempted).

        Note: ``oandapyV20.contrib.requests.PositionCloseRequest`` is
        bypassed here because its ``Units("ALL")`` validator raises
        ``ValueError: incorrect units: ALL``. The underlying REST
        endpoint accepts the string fine.
        """
        oanda_symbol = _to_oanda_symbol(instrument)
        with self._state_lock:
            net = self._net_positions.get(oanda_symbol, 0)

        if net == 0:
            logger.info(
                "[%s] OandaOrderManager.close_position: already flat — no-op.",
                oanda_symbol,
            )
            return False

        if net > 0:
            data = {"longUnits": "ALL"}
        else:
            data = {"shortUnits": "ALL"}

        try:
            req = v20_positions.PositionClose(
                accountID=self._account_id,
                instrument=oanda_symbol,
                data=data,
            )
            self._client.request(req)
            resp = req.response

            # ── Fix 1.7: Parse actual filled units ────────────────────
            # OANDA returns 'longOrderFillTransaction' or 'shortOrderFillTransaction'
            # containing 'units' as a signed string (e.g. "-100" for a sell).
            fill_l = resp.get("longOrderFillTransaction", {}) or {}
            fill_s = resp.get("shortOrderFillTransaction", {}) or {}

            units_l = int(fill_l.get("units", "0"))
            units_s = int(fill_s.get("units", "0"))
            total_filled = units_l + units_s

            with self._state_lock:
                prev_net = self._net_positions.get(oanda_symbol, 0)
                self._net_positions[oanda_symbol] = prev_net + total_filled

                if self._net_positions[oanda_symbol] == 0:
                    self._avg_entry_prices[oanda_symbol] = 0.0

                logger.info(
                    "[%s] OandaOrderManager close | fill=%d | net: %d -> %d",
                    oanda_symbol,
                    total_filled,
                    prev_net,
                    self._net_positions[oanda_symbol],
                )

            return True

        except Exception as e:
            logger.error(
                "[%s] OandaOrderManager.close_position failed (net was %d): %s",
                oanda_symbol,
                net,
                e,
                exc_info=True,
            )
            return False
