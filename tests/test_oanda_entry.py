import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

from oandapyV20.exceptions import V20Error

"""
Tests for OandaOrderManager entry logic -- net-position arithmetic.

Positions are tracked as ONE signed number per instrument (US rules forbid
holding both directions at once), so every entry is really "move from current
units to target units". These tests pin that arithmetic.

Glossary:
    test_flat_to_long -- the simple case: no position to a positive one.
    test_long_to_short_reversal -- crossing through zero in a single order,
        which is one instruction rather than a close followed by an open.
    test_delta_zero_noop -- already at target sends NO order. Important because
        the interface is "end at N units", so a repeat call must be harmless;
        combined with the re-sync-before-retry in submit_target_position, this
        is what makes retrying after an ambiguous network failure safe.
    test_last_attempt_ambiguous_fill_is_reported -- the final attempt can
        fail ambiguously and still have filled; that fill must be reported or
        the position is never recorded and never stopped.
    test_sync_failure_flags_result_unverified -- an unreadable broker after a
        sent order sets unverified, which is the caller's cue to park.
    test_clean_miss_is_not_flagged_unverified -- a business reject never
        filled, so it must NOT be parked.
    test_auth_401_is_retried -- a bare 401 has no reject reason and is
        transient, so the retry loop must re-send rather than give up.
    test_transient_401_then_success_captures_the_trade -- regression for the
        2026-08-26 GBP_AUD miss: the retry takes the trade, and the delta is
        sent once, never doubled.
    test_business_reject_beats_transient_classification -- a 400 with a
        rejectReason stays permanent; the safety and retry questions must not
        collapse into one.
    test_api_error_leaves_state -- a rejected order must leave local state
        untouched, so the bot never believes a failed order succeeded.
    test_add_to_existing_position_weighted_avg -- adding to a position
        recomputes the average entry price by size-weighting, since that price
        is what the stop and target are measured from.
"""

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.execution.oanda_order_manager import OandaOrderManager


class TestOandaEntry(unittest.TestCase):
    """Mocked unit tests for OandaOrderManager.submit_target_position()."""

    @patch("oandapyV20.API")
    def test_flat_to_long(self, mock_api_cls):
        """(a) Flat -> long: tradeOpened parsed, state updated."""
        manager = OandaOrderManager(api_key="fake", account_id="123")

        mock_req = MagicMock()
        mock_req.response = {
            "orderFillTransaction": {
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "units": "100",
                "price": "1.08500",
                "tradeOpened": {
                    "tradeID": "100",
                    "units": "100",
                    "price": "1.08500",
                },
            }
        }

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=mock_req
        ):
            result = manager.submit_target_position("EUR/USD", 100)

        self.assertEqual(result["filled"], 100)
        self.assertAlmostEqual(result["avg_price"], 1.08500)
        self.assertEqual(result["closed_units"], 0)
        self.assertEqual(result["opened_units"], 100)
        self.assertEqual(manager.get_net_position("EUR_USD"), 100)
        self.assertAlmostEqual(
            manager.get_average_entry_price("EUR_USD"), 1.08500
        )

    @patch("oandapyV20.API")
    def test_long_to_short_reversal(self, mock_api_cls):
        """(b) Long -> larger short reversal: tradesClosed + tradeOpened parsed."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        manager._net_positions["EUR_USD"] = 100
        manager._avg_entry_prices["EUR_USD"] = 1.08000

        mock_req = MagicMock()
        mock_req.response = {
            "orderFillTransaction": {
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "units": "-150",
                "price": "1.09000",
                "tradesClosed": [
                    {
                        "tradeID": "100",
                        "units": "-100",
                        "price": "1.08000",
                    }
                ],
                "tradeOpened": {
                    "tradeID": "101",
                    "units": "-50",
                    "price": "1.09000",
                },
            }
        }

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=mock_req
        ):
            result = manager.submit_target_position("EUR/USD", -50)

        self.assertEqual(result["filled"], 150)
        self.assertAlmostEqual(result["avg_price"], 1.09000)
        self.assertEqual(result["closed_units"], 100)
        self.assertEqual(result["opened_units"], 50)
        self.assertEqual(manager.get_net_position("EUR_USD"), -50)
        self.assertAlmostEqual(
            manager.get_average_entry_price("EUR_USD"), 1.09000
        )

    @patch("oandapyV20.API")
    def test_delta_zero_noop(self, mock_api_cls):
        """(c) delta == 0: no API call, state untouched."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        manager._net_positions["EUR_USD"] = 50

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate"
        ) as mock_order:
            result = manager.submit_target_position("EUR/USD", 50)
            mock_order.assert_not_called()

        self.assertEqual(result["filled"], 0)
        self.assertAlmostEqual(result["avg_price"], 0.0)
        self.assertEqual(result["closed_units"], 0)
        self.assertEqual(result["opened_units"], 0)
        self.assertEqual(manager.get_net_position("EUR_USD"), 50)

    @patch("oandapyV20.API")
    def test_api_error_leaves_state(self, mock_api_cls):
        """(d) API error: log ERROR, DO NOT mutate state."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        manager._net_positions["EUR_USD"] = 50
        manager._avg_entry_prices["EUR_USD"] = 1.08000

        mock_client = mock_api_cls.return_value
        mock_client.request.side_effect = Exception("Connection refused")

        result = manager.submit_target_position("EUR/USD", 100)

        self.assertEqual(result["filled"], 0)
        self.assertAlmostEqual(result["avg_price"], 0.0)
        self.assertEqual(result["closed_units"], 0)
        self.assertEqual(result["opened_units"], 0)
        self.assertEqual(manager.get_net_position("EUR_USD"), 50)
        self.assertAlmostEqual(
            manager.get_average_entry_price("EUR_USD"), 1.08000
        )

    @patch("oandapyV20.API")
    def test_add_to_existing_position_weighted_avg(self, mock_api_cls):
        """Add to long position: weighted average entry price computed."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        manager._net_positions["EUR_USD"] = 100
        manager._avg_entry_prices["EUR_USD"] = 1.08000

        mock_req = MagicMock()
        mock_req.response = {
            "orderFillTransaction": {
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "units": "100",
                "price": "1.09000",
                "tradeOpened": {
                    "tradeID": "101",
                    "units": "100",
                    "price": "1.09000",
                },
            }
        }

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=mock_req
        ):
            result = manager.submit_target_position("EUR/USD", 200)

        self.assertEqual(result["filled"], 100)
        self.assertAlmostEqual(result["avg_price"], 1.09000)
        self.assertEqual(result["closed_units"], 0)
        self.assertEqual(result["opened_units"], 100)
        self.assertEqual(manager.get_net_position("EUR_USD"), 200)
        # Weighted average: (100*1.08 + 100*1.09) / 200 = 1.085
        self.assertAlmostEqual(
            manager.get_average_entry_price("EUR_USD"), 1.08500
        )

    # ── retry-safety (the double-fill hazard) ─────────────────────────

    @patch("oandapyV20.API")
    def test_ambiguous_failure_resyncs_before_retry_no_double_fill(
        self, mock_api_cls
    ):
        """Timeout on attempt 1 + broker still flat + retry = ONE position.

        The regression that matters: if the retry reused the stale delta, a
        lost fill would double the position. The re-sync must show the broker
        flat before the same delta is sent again.
        """
        manager = OandaOrderManager(api_key="fake", account_id="123")
        mock_client = mock_api_cls.return_value

        order_req = MagicMock()
        order_req.response = {
            "orderFillTransaction": {
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "units": "1000",
                "price": "1.08500",
                "tradeOpened": {
                    "tradeID": "1", "units": "1000", "price": "1.08500",
                },
            }
        }
        position_req = MagicMock()
        position_req.response = {}  # flat: no "position" key

        # attempt 1 submit -> ambiguous 5xx; sync -> flat; attempt 2 -> fill.
        mock_client.request.side_effect = [
            V20Error(500, "gateway timeout"),
            None,
            None,
        ]

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=order_req
        ) as mock_order_create, patch(
            "oandapyV20.endpoints.positions.PositionDetails",
            return_value=position_req,
        ), patch("time.sleep"):
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertEqual(result["filled"], 1000)
        self.assertEqual(manager.get_net_position("EUR_USD"), 1000)
        units_sent = [
            c.kwargs["data"]["order"]["units"]
            for c in mock_order_create.call_args_list
        ]
        # Both orders were for the SAME 1000 delta — never a doubled 2000.
        self.assertEqual(units_sent, ["1000", "1000"])

    @patch("oandapyV20.API")
    def test_ambiguous_failure_that_filled_is_reported(self, mock_api_cls):
        """Attempt 1 fills but the response is lost; the re-sync reports it.

        The result must carry a non-zero fill so the caller records the
        position (and its stop/target). A no-op return here would strand an
        untracked, unwatched fill.
        """
        manager = OandaOrderManager(api_key="fake", account_id="123")
        mock_client = mock_api_cls.return_value

        order_req = MagicMock()
        position_req = MagicMock()
        position_req.response = {
            "position": {
                "long": {"units": "1000", "averagePrice": "1.08500"},
                "short": {"units": "0", "averagePrice": "0"},
            }
        }

        # attempt 1 submit -> ambiguous; sync -> broker holds 1000 long.
        mock_client.request.side_effect = [
            V20Error(503, "service unavailable"),
            None,
        ]

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=order_req
        ) as mock_order_create, patch(
            "oandapyV20.endpoints.positions.PositionDetails",
            return_value=position_req,
        ), patch("time.sleep"):
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertNotEqual(result["filled"], 0)
        self.assertEqual(result["position_units"], 1000)
        self.assertEqual(manager.get_net_position("EUR_USD"), 1000)
        # Re-sync confirmed the fill — no second order was sent.
        self.assertEqual(mock_order_create.call_count, 1)

    @patch("oandapyV20.API")
    def test_business_reject_not_retried(self, mock_api_cls):
        """A 400 business reject never filled; retrying would only re-reject."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        mock_client = mock_api_cls.return_value
        order_req = MagicMock()

        mock_client.request.side_effect = V20Error(
            400, '{"errorCode":"INSTRUMENT_NOT_TRADEABLE"}'
        )

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=order_req
        ) as mock_order_create, patch("time.sleep") as mock_sleep:
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertEqual(result["filled"], 0)
        self.assertEqual(mock_order_create.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertEqual(manager.get_net_position("EUR_USD"), 0)

    @patch("oandapyV20.API")
    def test_auth_401_is_retried(self, mock_api_cls):
        """
        A bare 401 carries no reject reason and is transient, so it IS retried.

        Verified against the live account on 2026-08-28: the identical 401
        failed at 04:30:07 and the same request succeeded at 04:30:10 on the
        same token. Retrying is safe because a 4xx never reached the matching
        engine, so no fill can be duplicated.
        """
        manager = OandaOrderManager(api_key="fake", account_id="123")
        mock_client = mock_api_cls.return_value
        order_req = MagicMock()

        mock_client.request.side_effect = V20Error(
            401, '{"errorMessage":"Insufficient authorization to perform request."}'
        )

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=order_req
        ) as mock_order_create, patch("time.sleep"):
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertEqual(result["filled"], 0)
        self.assertEqual(mock_order_create.call_count, 3)
        # Nothing filled, so the cache must still read flat.
        self.assertEqual(manager.get_net_position("EUR_USD"), 0)

    @patch("oandapyV20.API")
    def test_transient_401_then_success_captures_the_trade(self, mock_api_cls):
        """
        Regression for the 2026-08-26 GBP_AUD miss: a signal cleared every
        gate, the order hit a one-off 401, and the trade was lost. The retry
        must now take that trade — and send the delta exactly once.
        """
        manager = OandaOrderManager(api_key="fake", account_id="123")
        mock_client = mock_api_cls.return_value
        order_req = MagicMock()
        order_req.response = {
            "orderFillTransaction": {
                "units": "1000",
                "price": "1.89000",
                "tradeOpened": {
                    "tradeID": "1", "units": "1000", "price": "1.89000",
                },
            }
        }

        mock_client.request.side_effect = [
            V20Error(401, '{"errorMessage":"Insufficient authorization to perform request."}'),
            None,
        ]

        units_sent = []

        def capture(accountID, data):
            units_sent.append(data["order"]["units"])
            return order_req

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", side_effect=capture
        ), patch("time.sleep"):
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertEqual(result["filled"], 1000)
        self.assertEqual(result["position_units"], 1000)
        self.assertEqual(manager.get_net_position("EUR_USD"), 1000)
        # The same delta twice, never doubled: 1000 then 1000, ending at 1000.
        self.assertEqual(units_sent, ["1000", "1000"])

    @patch("oandapyV20.API")
    def test_business_reject_beats_transient_classification(self, mock_api_cls):
        """
        A 400 carrying a rejectReason is permanent even though it is a 4xx —
        the two questions ("did it fill?" / "will a retry help?") must not be
        collapsed back into one.
        """
        manager = OandaOrderManager(api_key="fake", account_id="123")
        mock_client = mock_api_cls.return_value
        order_req = MagicMock()

        mock_client.request.side_effect = V20Error(
            400,
            '{"orderRejectTransaction":{"type":"MARKET_ORDER_REJECT",'
            '"rejectReason":"INSUFFICIENT_MARGIN"},'
            '"errorCode":"INSUFFICIENT_MARGIN"}',
        )

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=order_req
        ) as mock_order_create, patch("time.sleep") as mock_sleep:
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertEqual(result["filled"], 0)
        self.assertEqual(mock_order_create.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("oandapyV20.API")
    def test_sync_failure_after_ambiguous_refuses_retry(self, mock_api_cls):
        """If the broker cannot be re-read, do NOT retry blindly.

        Retrying with an unverifiable state is how a lost fill gets doubled.
        The failure dict (filled=0) is the conservative answer.
        """
        manager = OandaOrderManager(api_key="fake", account_id="123")
        mock_client = mock_api_cls.return_value
        order_req = MagicMock()
        position_req = MagicMock()

        # Both the submit AND the re-sync fail (connection reset).
        mock_client.request.side_effect = Exception("connection reset")

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=order_req
        ) as mock_order_create, patch(
            "oandapyV20.endpoints.positions.PositionDetails",
            return_value=position_req,
        ), patch("time.sleep") as mock_sleep:
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertEqual(result["filled"], 0)
        self.assertEqual(mock_order_create.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertEqual(manager.get_net_position("EUR_USD"), 0)

    @patch("oandapyV20.API")
    def test_last_attempt_ambiguous_fill_is_reported(self, mock_api_cls):
        """
        The FINAL attempt can fail ambiguously and still have filled, with its
        own re-sync proving it. Reporting a zero fill there would strand a
        live position that the caller never records — so nothing would watch
        its stop.
        """
        manager = OandaOrderManager(api_key="fake", account_id="123")
        broker = {"units": 0}
        calls = {"n": 0}

        mock_client = mock_api_cls.return_value

        def request(req):
            calls["n"] += 1
            # Every attempt fails ambiguously; the last one fills first.
            if calls["n"] == manager._entry_max_attempts:
                broker["units"] += 1000
            raise V20Error(503, "service unavailable")

        mock_client.request.side_effect = request

        def fake_sync(instrument):
            with manager._state_lock:
                manager._net_positions["EUR_USD"] = broker["units"]
                manager._avg_entry_prices["EUR_USD"] = (
                    1.085 if broker["units"] else 0.0
                )
            return True

        manager.sync_position = fake_sync

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=MagicMock()
        ), patch("time.sleep"):
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertEqual(broker["units"], 1000)
        # The caller records only when filled != 0.
        self.assertNotEqual(result["filled"], 0)
        self.assertEqual(result["position_units"], 1000)
        self.assertFalse(result["unverified"])

    @patch("oandapyV20.API")
    def test_sync_failure_flags_result_unverified(self, mock_api_cls):
        """
        Order sent, ambiguous failure, broker unreadable: the outcome is
        genuinely unknown. The result must say so, so the caller parks and
        reconciles instead of discarding a possibly-live position.
        """
        manager = OandaOrderManager(api_key="fake", account_id="123")
        mock_client = mock_api_cls.return_value
        mock_client.request.side_effect = V20Error(503, "service unavailable")
        manager.sync_position = lambda instrument: False

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=MagicMock()
        ), patch("time.sleep"):
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertEqual(result["filled"], 0)
        self.assertTrue(result["unverified"])

    @patch("oandapyV20.API")
    def test_clean_miss_is_not_flagged_unverified(self, mock_api_cls):
        """
        A business reject never filled, so it is a clean miss — flagging it
        unverified would park a position that does not exist and block the
        symbol for nothing.
        """
        manager = OandaOrderManager(api_key="fake", account_id="123")
        mock_client = mock_api_cls.return_value
        mock_client.request.side_effect = V20Error(
            400, '{"errorCode":"INSTRUMENT_NOT_TRADEABLE"}'
        )

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=MagicMock()
        ), patch("time.sleep"):
            result = manager.submit_target_position("EUR/USD", 1000)

        self.assertEqual(result["filled"], 0)
        self.assertFalse(result["unverified"])

    @patch("oandapyV20.API")
    def test_client_uses_http_timeout(self, mock_api_cls):
        """The order client must carry an HTTP timeout (no infinite hang)."""
        OandaOrderManager(api_key="fake", account_id="123")
        kwargs = mock_api_cls.call_args.kwargs
        self.assertIn("request_params", kwargs)
        self.assertEqual(kwargs["request_params"].get("timeout"), 30.0)


if __name__ == "__main__":
    unittest.main()
