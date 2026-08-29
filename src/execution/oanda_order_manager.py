"""
FIFO-compliant order/position state manager for OANDA v20.

Foundation class for the V5 forex bot execution path. Tracks the
*net* position per instrument (signed units + broker-reported average
entry price) — never per-trade lots — to comply with U.S. NFA FIFO and
no-hedging rules enforced by OANDA.

Required environment variables:
    OANDA_API_KEY       - Bearer token from hub.oanda.com
    OANDA_ACCOUNT_ID    - Account ID (numeric string)

Scope of this module: state + close. Entry methods, fill-stream
consumers, and watchdog wiring live in separate modules.

Why NET position and not individual trades: US regulation (NFA) requires FIFO
closing and forbids holding opposing positions in the same pair. Tracking one
signed number per instrument makes those rules impossible to violate by
construction, rather than something the code has to remember to check.

Glossary:
    OandaOrderManager -- owns the truth about what is currently held.
    net position -- one signed unit count per instrument: positive is long,
        negative is short, zero is flat. There is no concept of separate lots.
    get_net_position / get_average_entry_price -- cached reads of that state;
        the average entry price is the broker's number, not a locally computed
        one.
    sync_position -- refreshes local state from the broker. The broker is
        always authoritative; local state is a cache.
    close_position -- flattens an instrument. Raises OrderCloseError on broker
        failure and LEAVES LOCAL STATE UNTOUCHED, so a failed close never makes
        the bot believe it is flat when it is not. That asymmetry is
        deliberate: believing you are flat while holding a position is far more
        dangerous than the reverse.
    submit_target_position -- expresses an order as "I want to end at N units"
        rather than "buy N units", and retries a failed send. Each attempt
        sends the REMAINING delta, so a lost fill cannot be sent twice.
    _place_order -- submits one market order and parses the fill; the retry
        loop in submit_target_position wraps it.
    _result_if_at_target -- after an order was sent, reports a fill when the
        broker is now at target. Consulted both between retries AND after the
        attempts run out, because the last attempt can fail ambiguously and
        still have filled.
    unverified (result key) -- True only when an order was sent, failed
        ambiguously, and the broker could NOT be re-read. The position may or
        may not exist, so the caller must park and reconcile rather than treat
        it as a clean miss. False on every other path, including clean misses.
    _error_body -- tolerant JSON decode of a V20Error payload; {} when the
        body is absent or undecodable.
    _order_never_filled -- the SAFETY question. True for any 4xx, which is
        rejected before execution, so the position cannot have moved. False
        for 5xx/timeouts, which are ambiguous and force a re-sync.
    _is_permanent_reject -- the RETRY question. True only when the body
        carries a reject reason (INSTRUMENT_NOT_TRADEABLE, INSUFFICIENT_MARGIN,
        ...), which will still hold on a retry. A bare 401/403/429 has no
        reason and IS retried: it never filled, and it is transient.
    _request_timeout -- HTTP timeout (seconds) applied to every REST call via
        this client; prevents a half-open socket from hanging the order path.
    _entry_max_attempts / _entry_retry_delay -- entry retry count and its
        linear backoff base (seconds), both env-tunable.
    _to_oanda_symbol -- normalises 'EUR/USD' / 'EURUSD' / 'EUR_USD' to the
        underscore form.
    _state_lock -- guards the position cache; this class is touched from both
        the event loop and worker threads.
    OrderCloseError -- raised when the broker rejects a close.
"""

import json
import logging
import os
import threading
import time
from typing import Dict, Optional

import oandapyV20
import oandapyV20.endpoints.orders as v20_orders
import oandapyV20.endpoints.positions as v20_positions
from oandapyV20.exceptions import V20Error

logger = logging.getLogger(__name__)


def _to_oanda_symbol(symbol: str) -> str:
    """Normalise 'EUR/USD', 'EURUSD', or 'EUR_USD' → 'EUR_USD'."""
    return symbol.replace("/", "_").upper()


def _error_body(exc: Exception) -> dict:
    """
    Best-effort decode of a ``V20Error`` payload. ``V20Error.msg`` is the raw
    response body, so a business reject arrives as JSON here. Never raises —
    an undecodable body simply yields ``{}``, which callers read as "no
    reject reason present".
    """
    if not isinstance(exc, V20Error):
        return {}
    try:
        body = json.loads(exc.msg)
    except (ValueError, TypeError):
        return {}
    return body if isinstance(body, dict) else {}


def _order_never_filled(exc: Exception) -> bool:
    """
    True when *exc* proves no order reached the matching engine, so the
    position cannot have moved.

    Any 4xx is rejected before execution: a business reject (HTTP 400 with an
    ``orderRejectTransaction``), an auth failure (401/403), or a rate limit
    (429). 5xx, connection resets and timeouts are *ambiguous* — the order may
    have filled and the response been lost.

    This answers the SAFETY question only ("could the position have moved?").
    Whether a retry is worth making is a separate question — see
    ``_is_permanent_reject``.
    """
    return isinstance(exc, V20Error) and exc.code < 500


def _is_permanent_reject(exc: Exception) -> bool:
    """
    True when the broker refused the order for a reason that will still hold
    on a retry — ``INSTRUMENT_NOT_TRADEABLE``, ``INSUFFICIENT_MARGIN``,
    ``MARKET_HALTED``, ``FIFO_VIOLATION_SAFEGUARD_VIOLATION``. Detected by the
    presence of a reject reason in the body, not by HTTP status.

    A bare 401/403/429 carries NO reject reason and is *transient*: verified
    on 2026-08-28, an identical 401 on this account failed at 04:30:07 and the
    same request succeeded at 04:30:10 with the same token. Those are safe to
    retry precisely because ``_order_never_filled`` proves nothing filled, and
    worth retrying because the bot sees only a few tradeable signals a week.
    """
    if not _order_never_filled(exc):
        return False
    body = _error_body(exc)
    reject = body.get("orderRejectTransaction") or {}
    return bool(reject.get("rejectReason") or body.get("errorCode"))


class OrderCloseError(Exception):
    """A position close request failed at the broker (state untouched)."""


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

        # HTTP timeout for every REST call through this client. Without it a
        # half-open socket hangs the order/close path forever — and stops and
        # targets are enforced in software by the caller. The data client sets
        # the same timeout for the same reason (src/data/oanda_provider.py).
        self._request_timeout = float(os.getenv("OANDA_REQUEST_TIMEOUT", "30"))

        # Entry retry policy. A retry after an *ambiguous* failure (timeout,
        # 5xx, reset) must re-read the broker position first, or a lost fill
        # would be resubmitted and double the position. Definitive rejects are
        # never retried.
        self._entry_max_attempts = int(os.getenv("OANDA_ENTRY_MAX_ATTEMPTS", "3"))
        self._entry_retry_delay = float(os.getenv("OANDA_ENTRY_RETRY_DELAY", "2.0"))

        self._client = oandapyV20.API(
            access_token=self._api_key,
            environment=environment,
            request_params={"timeout": self._request_timeout},
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

    def sync_position(self, instrument: str) -> bool:
        """
        Pull authoritative net-position state from OANDA's
        ``/v3/accounts/{id}/positions/{instrument}`` endpoint and
        refresh internal state.

        Under FIFO/no-hedging, exactly one of ``long`` or ``short`` will
        carry non-zero units for any given instrument; OANDA returns
        short units as a negative numeric string.

        Returns True if state was refreshed from the broker, False if the
        request failed (local state untouched — caller must not assume
        the position is flat).
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
            return True
        except V20Error as e:
            # A 404 / NO_SUCH_POSITION is OANDA's way of saying the
            # instrument has never held a position this account-lifetime —
            # i.e. it is flat. That is a *successful* sync, not a failure;
            # treating it as one would make boot reconciliation refuse to
            # start on a clean account.
            if e.code == 404 and "NO_SUCH_POSITION" in str(e):
                with self._state_lock:
                    self._net_positions[oanda_symbol] = 0
                    self._avg_entry_prices[oanda_symbol] = 0.0
                logger.info(
                    "[%s] OandaOrderManager sync | net=0 (no position on broker)",
                    oanda_symbol,
                )
                return True
            logger.error(
                "[%s] OandaOrderManager.sync_position failed: %s",
                oanda_symbol,
                e,
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.error(
                "[%s] OandaOrderManager.sync_position failed: %s",
                oanda_symbol,
                e,
                exc_info=True,
            )
            return False

    # ── FIFO close ────────────────────────────────────────────────────

    def close_position(self, instrument: str) -> bool:
        """
        Flatten the net position for *instrument* via OANDA's
        ``/positions/{instrument}/close`` endpoint.

        Uses ``"ALL"`` semantics so the broker liquidates whatever is
        actually open, even if local state has drifted. Returns True if
        a close request was submitted, False if already flat.

        Raises
        ------
        OrderCloseError
            If the broker request fails. Local state is untouched so a
            retry can be attempted; callers MUST treat the position as
            still open.

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
            raise OrderCloseError(
                f"close_position({oanda_symbol}) failed with net={net}: {e}"
            ) from e

    # ── target-position entry / reversal ───────────────────────────────

    def submit_target_position(self, symbol: str, target_units: int) -> dict:
        """
        Move the net position for *symbol* to *target_units* via OANDA v20
        market orders, retrying safely after an ambiguous failure.

        Each attempt expresses the *remaining* delta (``target_units`` minus
        the current net position) as a single FOK market order, so the call
        is idempotent in effect. That idempotency is only real because of the
        re-sync: when an attempt fails *ambiguously* (timeout, 5xx, reset)
        the order may already have filled, so this method re-reads the broker
        position before computing the next delta.

        Failures are classified on two separate questions. *Could the position
        have moved?* — any 4xx is rejected before execution, so no. *Will a
        retry help?* — only if the body carries no reject reason. A business
        reject (INSTRUMENT_NOT_TRADEABLE, INSUFFICIENT_MARGIN) is permanent
        and returns immediately; a bare 401/403/429 is a transient blip and is
        retried without a re-sync, since nothing filled.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``'EUR/USD'``).
        target_units : int
            Signed net target (>0 long, <0 short, 0 flat).

        Returns
        -------
        dict
            ``{'filled': int, 'avg_price': float, 'closed_units': int,
            'opened_units': int, 'position_units': int,
            'position_avg_price': float}``

            ``filled``/``avg_price`` describe the whole market order (on a
            reversal that includes the closing leg). ``position_units`` and
            ``position_avg_price`` are the authoritative resulting net
            position — use these for position records. ``filled == 0`` means
            no position was taken (or could not be verified); callers must
            not record one.
        """
        oanda_symbol = _to_oanda_symbol(symbol)

        with self._state_lock:
            current_net = self._net_positions.get(oanda_symbol, 0)
            current_avg = self._avg_entry_prices.get(oanda_symbol, 0.0)

        # The delta of the order most recently sent (0 until the first
        # submit). When a re-sync finds us already at target, this is how we
        # report "that order actually filled" instead of a spurious no-op.
        sent_delta = 0

        for attempt in range(1, self._entry_max_attempts + 1):
            delta = target_units - current_net

            if delta == 0:
                reached = self._result_if_at_target(
                    oanda_symbol, target_units, current_net, current_avg,
                    sent_delta,
                )
                if reached is not None:
                    return reached
                return {
                    "filled": 0,
                    "avg_price": 0.0,
                    "closed_units": 0,
                    "opened_units": 0,
                    "position_units": current_net,
                    "position_avg_price": current_avg,
                    "unverified": False,
                }

            sent_delta = delta
            try:
                return self._place_order(
                    oanda_symbol, delta, current_net, current_avg, target_units
                )
            except Exception as e:
                if _is_permanent_reject(e):
                    logger.error(
                        "[%s] submit_target_position: permanent reject "
                        "(target=%d delta=%d), not retrying: %s",
                        oanda_symbol, target_units, delta, e,
                    )
                    return {
                        "filled": 0,
                        "avg_price": 0.0,
                        "closed_units": 0,
                        "opened_units": 0,
                        "position_units": current_net,
                        "position_avg_price": current_avg,
                        # The broker refused it outright; nothing to reconcile.
                        "unverified": False,
                    }

                if _order_never_filled(e):
                    # Transient 4xx: an auth blip (401/403) or a rate limit
                    # (429), carrying no reject reason. Nothing reached the
                    # matching engine, so the delta is still correct and no
                    # re-sync is needed — and a re-sync would probably hit the
                    # same blip. This is the case that lost a live trade on
                    # 2026-08-26; retrying it is the whole point.
                    logger.warning(
                        "[%s] submit_target_position attempt %d/%d hit a "
                        "transient reject (target=%d delta=%d); nothing "
                        "filled, retrying: %s",
                        oanda_symbol, attempt, self._entry_max_attempts,
                        target_units, delta, e,
                    )
                    if attempt < self._entry_max_attempts:
                        time.sleep(self._entry_retry_delay * attempt)
                    continue

                # Ambiguous: the order may have filled and the response been
                # lost. Re-read the broker before deciding — retrying with a
                # stale delta is how positions double.
                logger.warning(
                    "[%s] submit_target_position attempt %d/%d failed "
                    "ambiguously (target=%d delta=%d); re-syncing broker "
                    "state before retry: %s",
                    oanda_symbol, attempt, self._entry_max_attempts,
                    target_units, delta, e,
                )
                if not self.sync_position(oanda_symbol):
                    logger.critical(
                        "[%s] submit_target_position: cannot verify broker "
                        "state after ambiguous order failure; refusing to "
                        "retry (manual check required): %s",
                        oanda_symbol, e,
                    )
                    return {
                        "filled": 0,
                        "avg_price": 0.0,
                        "closed_units": 0,
                        "opened_units": 0,
                        "position_units": current_net,
                        "position_avg_price": current_avg,
                        # The order may have filled. The caller MUST NOT treat
                        # this as a clean miss: park it and reconcile.
                        "unverified": True,
                    }

                with self._state_lock:
                    current_net = self._net_positions.get(oanda_symbol, 0)
                    current_avg = self._avg_entry_prices.get(oanda_symbol, 0.0)

                if attempt < self._entry_max_attempts:
                    time.sleep(self._entry_retry_delay * attempt)

        # The final attempt is not special-cased anywhere above: it can fail
        # ambiguously *and* have filled, with its own re-sync proving it.
        # Reporting a zero fill here would strand that position untracked and
        # unstopped, which is the exact hazard the retry exists to avoid.
        reached = self._result_if_at_target(
            oanda_symbol, target_units, current_net, current_avg, sent_delta
        )
        if reached is not None:
            return reached

        logger.error(
            "[%s] submit_target_position failed after %d attempts "
            "(target=%d); see prior warnings",
            oanda_symbol, self._entry_max_attempts, target_units,
        )
        return {
            "filled": 0,
            "avg_price": 0.0,
            "closed_units": 0,
            "opened_units": 0,
            "position_units": current_net,
            "position_avg_price": current_avg,
            "unverified": False,
        }

    def _result_if_at_target(
        self,
        oanda_symbol: str,
        target_units: int,
        current_net: int,
        current_avg: float,
        sent_delta: int,
    ) -> Optional[dict]:
        """
        A fill result when an order was sent and the broker is now AT
        *target_units*, else ``None``.

        Called from two places, and both matter: between retries, and after
        the attempts are exhausted. An attempt can fail ambiguously and still
        have filled — the re-sync is what proves it. Reporting a zero fill in
        that case leaves a live position that the caller never records, so
        the software stop monitor never watches it.

        ``sent_delta == 0`` means no order ever left, so being at target is
        just the caller asking for a position it already holds — a genuine
        no-op, not a fill.
        """
        if sent_delta == 0 or current_net != target_units:
            return None

        logger.info(
            "[%s] submit_target_position: re-sync confirms position at "
            "target (net=%d) after an ambiguous attempt — reporting fill",
            oanda_symbol, current_net,
        )
        return {
            "filled": abs(sent_delta),
            "avg_price": current_avg,
            "closed_units": 0,
            "opened_units": abs(sent_delta),
            "position_units": current_net,
            "position_avg_price": current_avg,
            "unverified": False,
        }

    def _place_order(
        self,
        oanda_symbol: str,
        delta: int,
        current_net: int,
        current_avg: float,
        target_units: int,
    ) -> dict:
        """
        Submit ONE market order for *delta* units and parse the fill.

        Raises on any HTTP/network failure — the caller decides whether the
        failure never filled or is ambiguous, and whether to retry. Returns the
        success dict on a fill, or a ``filled == 0`` dict when the 200
        response carries no ``orderFillTransaction`` (the order was accepted
        but did not fill, e.g. FOK cancelled; the broker position is
        unchanged).
        """
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": oanda_symbol,
                "units": str(delta),
            }
        }

        req = v20_orders.OrderCreate(
            accountID=self._account_id,
            data=order_data,
        )
        self._client.request(req)
        resp = req.response

        fill_tx = resp.get("orderFillTransaction", {}) or {}
        if not fill_tx:
            logger.error(
                "[%s] submit_target_position: no orderFillTransaction in response",
                oanda_symbol,
            )
            return {
                "filled": 0,
                "avg_price": 0.0,
                "closed_units": 0,
                "opened_units": 0,
                "position_units": current_net,
                "position_avg_price": current_avg,
                # A 201 carrying no fill transaction is the broker saying the
                # order was killed, not that its outcome is unknown.
                "unverified": False,
            }

        fill_units = int(fill_tx.get("units", "0"))
        fill_price = float(fill_tx.get("price", "0") or "0")

        trades_closed = fill_tx.get("tradesClosed", []) or []
        trade_opened = fill_tx.get("tradeOpened") or {}

        closed_units = sum(
            abs(int(t.get("units", "0"))) for t in trades_closed
        )
        opened_units = (
            abs(int(trade_opened.get("units", "0")))
            if trade_opened
            else 0
        )
        total_filled = abs(fill_units)

        with self._state_lock:
            old_net = self._net_positions.get(oanda_symbol, 0)
            old_avg = self._avg_entry_prices.get(oanda_symbol, 0.0)
            new_net = old_net + fill_units

            self._net_positions[oanda_symbol] = new_net

            if new_net == 0:
                self._avg_entry_prices[oanda_symbol] = 0.0
            elif old_net != 0 and (old_net > 0) == (new_net > 0):
                # Same direction — add or reduce.
                if abs(new_net) > abs(old_net):
                    # Adding: blend old avg with opened-leg price.
                    opened_price = float(
                        trade_opened.get("price", fill_price)
                        or fill_price
                    ) if trade_opened else fill_price
                    added = abs(new_net) - abs(old_net)
                    self._avg_entry_prices[oanda_symbol] = (
                        abs(old_net) * old_avg + added * opened_price
                    ) / abs(new_net)
                # else: reduction — keep old avg.
            else:
                # Fresh open or reversal — opened leg is the whole position.
                opened_price = float(
                    trade_opened.get("price", fill_price)
                    or fill_price
                ) if trade_opened else fill_price
                self._avg_entry_prices[oanda_symbol] = opened_price

            position_avg_price = self._avg_entry_prices[oanda_symbol]

        logger.info(
            "[%s] submit_target_position | target=%d delta=%d "
            "fill_price=%.5f resulting_net=%d",
            oanda_symbol,
            target_units,
            delta,
            fill_price,
            new_net,
        )

        return {
            "filled": total_filled,
            "avg_price": fill_price,
            "closed_units": closed_units,
            "opened_units": opened_units,
            "position_units": new_net,
            "position_avg_price": position_avg_price,
            "unverified": False,
        }
