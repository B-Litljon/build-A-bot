"""Tests for the trading_mcp control + retrain-status additions.

Covers the retrain-log parser and the two-step confirm-token flow for the
start/stop control tools — all without spawning or killing a real process
(the side-effecting helpers are monkeypatched).
"""
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

# trading_mcp.py lives at the repo root.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import trading_mcp as tm  # noqa: E402


_REJECTED_LOG = """\
2026-06-19 19:42:53  INFO      Symbols: XAU_USD, XAG_USD
2026-06-19 19:42:53  INFO      Date range: 2025-06-20 to 2026-06-20
2026-06-19 19:42:53  INFO      VALIDATION GATE SUMMARY
2026-06-19 19:42:53  INFO      Mean Brier Score : 0.3153 (threshold <= 0.3)
2026-06-19 19:42:53  INFO      Mean EV          : 0.397871 (threshold >= 0.0005)
2026-06-19 19:42:53  INFO      Profit Factor    : 1.0455 (threshold >= 1.2, Fold 3 OOS)
2026-06-19 19:42:53  INFO      Pooled OOS Trades: 224 across 3 folds (dynamic floor >= 209)
2026-06-19 19:42:53  INFO      Gate Result      : FAILED
2026-06-19 19:42:53  WARNING   MODELS REJECTED — Production weights retained
2026-06-19 19:42:53  WARNING     Rejection: Brier 0.3153 > 0.3 threshold
2026-06-19 19:42:53  WARNING     Rejection: Profit Factor 1.0455 < 1.2 threshold
"""


class TestRetrainStatusParsing(unittest.TestCase):
    def test_parses_rejected_run(self):
        with mock.patch.object(tm, "_latest_retrain_log") as latest, \
             mock.patch.object(tm.Path, "read_text", return_value=_REJECTED_LOG):
            latest.return_value = "/repo/logs/retrain_metals_x.log"
            info = tm._retrain_status()
        self.assertEqual(info["verdict"], "REJECTED")
        self.assertEqual(info["gate_result"], "FAILED")
        self.assertEqual(info["profit_factor"], 1.0455)
        self.assertEqual(info["mean_brier"], 0.3153)
        self.assertEqual(info["pooled_oos_trades"], 224)
        self.assertTrue(info["complete"])
        self.assertEqual(len(info["rejection_reasons"]), 2)

    def test_no_log(self):
        with mock.patch.object(tm, "_latest_retrain_log", return_value=None):
            info = tm._retrain_status()
        self.assertIn("note", info)


class TestConfirmTokenFlow(unittest.TestCase):
    def setUp(self):
        tm._pending.clear()
        self.addCleanup(tm._pending.clear)

    def test_start_requires_then_consumes_token(self):
        with mock.patch.object(tm, "_soak_pid", return_value=None), \
             mock.patch.object(tm, "_spawn_soak", return_value={"pid": 4242, "log": "logs/x.log"}) as spawn:
            first = tm._start_soak(symbols="XAU_USD,XAG_USD")
            self.assertEqual(first["status"], "confirmation_required")
            token = first["confirm_token"]
            spawn.assert_not_called()  # nothing launched yet

            # wrong token is rejected
            bad = tm._start_soak(symbols="XAU_USD,XAG_USD", confirm_token="deadbe")
            self.assertEqual(bad["status"], "rejected")
            spawn.assert_not_called()

            # need a fresh token (the bad attempt didn't consume it) — reissue
            first = tm._start_soak(symbols="XAU_USD,XAG_USD")
            token = first["confirm_token"]
            ok = tm._start_soak(symbols="XAU_USD,XAG_USD", confirm_token=token)
            self.assertEqual(ok["status"], "started")
            self.assertEqual(ok["pid"], 4242)
            spawn.assert_called_once_with("XAU_USD,XAG_USD")

    def test_token_is_single_use(self):
        with mock.patch.object(tm, "_soak_pid", return_value=None), \
             mock.patch.object(tm, "_spawn_soak", return_value={"pid": 1, "log": "l"}):
            token = tm._start_soak(symbols="XAU_USD")["confirm_token"]
            tm._start_soak(symbols="XAU_USD", confirm_token=token)
            replay = tm._start_soak(symbols="XAU_USD", confirm_token=token)
        self.assertEqual(replay["status"], "rejected")

    def test_token_bound_to_args(self):
        with mock.patch.object(tm, "_soak_pid", return_value=None), \
             mock.patch.object(tm, "_spawn_soak", return_value={"pid": 1, "log": "l"}):
            token = tm._start_soak(symbols="XAU_USD")["confirm_token"]
            mismatch = tm._start_soak(symbols="XAG_USD", confirm_token=token)
        self.assertEqual(mismatch["status"], "rejected")

    def test_expired_token_rejected(self):
        with mock.patch.object(tm, "_soak_pid", return_value=None), \
             mock.patch.object(tm, "_spawn_soak", return_value={"pid": 1, "log": "l"}):
            token = tm._start_soak(symbols="XAU_USD")["confirm_token"]
            tm._pending["start_soak"]["expires"] = time.time() - 1  # force-expire
            out = tm._start_soak(symbols="XAU_USD", confirm_token=token)
        self.assertEqual(out["status"], "rejected")
        self.assertIn("expired", out["reason"])

    def test_start_refuses_when_already_running(self):
        with mock.patch.object(tm, "_soak_pid", return_value=999), \
             mock.patch.object(tm, "_pid_alive", return_value=True):
            out = tm._start_soak(symbols="XAU_USD")
        self.assertEqual(out["status"], "refused")

    def test_stop_noop_when_nothing_running(self):
        with mock.patch.object(tm, "_soak_pid", return_value=None):
            out = tm._stop_soak()
        self.assertEqual(out["status"], "noop")

    def test_stop_confirm_then_sigterm(self):
        with mock.patch.object(tm, "_soak_pid", return_value=777), \
             mock.patch.object(tm, "_pid_alive", return_value=True), \
             mock.patch.object(tm, "_ps_field", return_value="01:23"), \
             mock.patch.object(tm.os, "kill") as kill:
            first = tm._stop_soak()
            self.assertEqual(first["status"], "confirmation_required")
            kill.assert_not_called()
            out = tm._stop_soak(confirm_token=first["confirm_token"])
            self.assertEqual(out["status"], "stopping")
            kill.assert_called_once_with(777, tm.signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
