import inspect
import json
import os
import queue
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock

"""
Tests for the structured event sink -- the telemetry feeding the dashboard.

This module runs INSIDE the live trading process, so the tests here are mostly
about what it must never do: raise, block, or touch the tick path. A dashboard
going dark is an inconvenience; a telemetry bug reaching the order path is a
money-losing one.

Glossary:
    _drain -- waits for the writer thread to flush, so assertions read a
        settled file rather than racing the queue.
    test_emit_roundtrip -- an emitted event lands as one parseable JSON line
        with an automatic timestamp.
    test_daily_rotation -- events file is named for the event's own UTC date,
        so a line never lands in yesterday's file after midnight.
    test_status_is_atomic -- a reader polling status.json must never observe a
        partial write; the temp file must not be the target path.
    test_disabled_writes_nothing -- EVENTS_ENABLED=0 is a real off switch.
    test_emit_never_raises -- a broken sink is silent, not fatal.
    test_full_queue_drops_instead_of_blocking -- back-pressure is resolved by
        losing telemetry, never by stalling the caller.
    test_writer_survives_bad_payload -- one unserialisable event must not kill
        the writer for every later event.
    test_entry_still_records_when_telemetry_raises -- the integration that
        matters: with the sink failing under every call, a position is still
        opened and tracked.
    test_emit_args_are_cheap -- call-site arguments must not compute anything
        that can throw, since those expressions are evaluated in the trading
        path BEFORE emit()'s own safety net applies.
    test_tick_path_has_no_emit -- source-level guard on the <50 µs constraint.
"""

warnings.filterwarnings("ignore", category=RuntimeWarning)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.core import events  # noqa: E402


def _drain(timeout=2.0):
    """Block until the writer thread has emptied the queue."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if events._queue is None or events._queue.empty():
            time.sleep(0.05)  # let the in-flight item finish writing
            return
        time.sleep(0.01)


class TestEvents(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        # Reset module state between tests: the writer thread and queue are
        # process-global.
        events.shutdown()
        events._queue = None
        events._thread = None
        events._dropped = 0
        events._errors = 0
        events.configure(log_dir=self.dir, enabled=True)

    def tearDown(self):
        events.shutdown()
        events._queue = None
        events._thread = None
        self._tmp.cleanup()

    # ── basics ──────────────────────────────────────────────────────────

    def test_emit_roundtrip(self):
        """An event lands as one JSON line, with a timestamp added for free."""
        events.emit("entry", sym="NZD_JPY", units=1000, entry=94.182)
        _drain()

        lines = events.events_path().read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["ev"], "entry")
        self.assertEqual(rec["sym"], "NZD_JPY")
        self.assertEqual(rec["units"], 1000)
        self.assertIn("ts", rec)

    def test_daily_rotation(self):
        """The filename comes from the event's own UTC date."""
        events.emit("bar", sym="GBP_JPY")
        _drain()
        expected = events.events_path()
        self.assertTrue(expected.exists())
        self.assertRegex(expected.name, r"^events-\d{4}-\d{2}-\d{2}\.jsonl$")

    def test_status_is_atomic(self):
        """status.json is replaced, never partially overwritten in place."""
        events.write_status({"pid": 123, "positions": {}})
        _drain()

        path = events.status_path()
        payload = json.loads(path.read_text())
        self.assertEqual(payload["pid"], 123)
        self.assertIn("ts", payload)
        # The temp file must be gone: its lingering presence would mean the
        # rename never happened and a reader could hit a half-written file.
        self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_status_overwrite_keeps_file_valid(self):
        """Repeated snapshots always leave a parseable file behind."""
        for i in range(20):
            events.write_status({"seq": i})
        _drain()
        payload = json.loads(events.status_path().read_text())
        self.assertEqual(payload["seq"], 19)

    def test_disabled_writes_nothing(self):
        """EVENTS_ENABLED=0 is a real off switch."""
        events.shutdown()
        events._queue = None
        events._thread = None
        events.configure(log_dir=self.dir, enabled=False)

        events.emit("entry", sym="GBP_JPY")
        events.write_status({"pid": 1})
        _drain()

        self.assertEqual(list(self.dir.glob("events-*.jsonl")), [])
        self.assertFalse(events.status_path().exists())

    # ── failure behaviour ───────────────────────────────────────────────

    def test_emit_never_raises(self):
        """A broken sink is silent, not fatal."""
        original = events._put
        events._put = MagicMock(side_effect=RuntimeError("disk on fire"))
        try:
            events.emit("entry", sym="GBP_JPY")       # must not raise
            events.write_status({"pid": 1})           # must not raise
        finally:
            events._put = original

    def test_full_queue_drops_instead_of_blocking(self):
        """Back-pressure loses telemetry rather than stalling the caller."""
        events._queue = queue.Queue(maxsize=1)
        events._queue.put(("event", {"ts": "x", "ev": "filler"}))
        events._thread = MagicMock()  # nothing is draining
        events._thread.is_alive.return_value = True

        started = time.monotonic()
        for _ in range(50):
            events.emit("bar", sym="GBP_JPY")
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)        # never blocked
        self.assertGreaterEqual(events._dropped, 49)
        self.assertGreater(events.stats()["dropped"], 0)

    def test_writer_survives_bad_payload(self):
        """One unserialisable event must not kill the writer thread."""
        class Unserialisable:
            def __repr__(self):
                raise ValueError("nope")

        events.emit("bad", obj=Unserialisable())
        events.emit("good", sym="GBP_JPY")
        _drain()

        text = events.events_path().read_text()
        self.assertIn('"ev": "good"', text)

    # ── integration with the trading path ───────────────────────────────

    def test_entry_still_records_when_telemetry_raises(self):
        """With emit() raising on every call, a position is still opened."""
        import asyncio
        from src.execution.oanda_scalper_orchestrator import (
            OandaScalperOrchestrator,
        )
        from src.execution import oanda_scalper_orchestrator as orch_mod

        provider = MagicMock()
        provider._stream_gran = 15
        strategy = MagicMock()
        strategy.warmup_period = 3
        order_manager = MagicMock()
        risk_manager = MagicMock()

        orch = OandaScalperOrchestrator(
            symbols=["GBP_JPY"],
            provider=provider,
            strategy=strategy,
            order_manager=order_manager,
            risk_manager=risk_manager,
            units_per_trade=1000,
            warmup_period=3,
            flatten_on_exit=False,
            notifier=MagicMock(),
        )

        signal = MagicMock()
        signal.direction = "long"
        signal.entry_price = 215.46
        signal.raw_sl_distance = 0.39
        signal.metadata = {}
        strategy.generate_signals.return_value = signal
        risk_manager.calculate_bracket.return_value = (0.39, 0.73)
        order_manager.submit_target_position.return_value = {
            "filled": 1000, "avg_price": 215.462, "closed_units": 0,
            "opened_units": 1000, "position_units": 1000,
            "position_avg_price": 215.462,
        }
        orch._bar_buffers["GBP_JPY"] = [
            {"symbol": "GBP_JPY", "timestamp": i, "open": 215.0, "high": 215.5,
             "low": 214.9, "close": 215.4, "volume": 1.0}
            for i in range(3)
        ]

        # Inject the failure where it can realistically happen: the sink
        # underneath emit() -- a full disk, a permission error, a dead writer
        # thread. emit() itself is contractually silent, so replacing IT would
        # test the mock rather than the guarantee.
        real_put = orch_mod.events._put
        orch_mod.events._put = MagicMock(side_effect=OSError("no space left"))
        try:
            asyncio.run(orch._evaluate_and_trade("GBP_JPY"))
        finally:
            orch_mod.events._put = real_put

        order_manager.submit_target_position.assert_called_once_with(
            "GBP_JPY", 1000
        )
        self.assertEqual(orch._positions["GBP_JPY"]["units"], 1000)
        self.assertEqual(orch._positions["GBP_JPY"]["state"], "OPEN")

    def test_emit_args_are_cheap(self):
        """
        Call-site arguments must not compute anything that can throw.

        emit() swallows its own exceptions, but its ARGUMENTS are evaluated by
        the caller first — in the trading path, outside that safety net. So
        `angel=probs[0]` or `pct=hits/total` would be a live-path crash dressed
        up as telemetry. Arithmetic and indexing are therefore banned inside
        emit calls; compute into a local first, where the surrounding
        try/except owns it.
        """
        import ast

        targets = [
            project_root / "src/execution/oanda_scalper_orchestrator.py",
            project_root / "src/strategies/concrete_strategies/ml_strategy.py",
        ]
        checked = 0
        for path in targets:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if not (isinstance(fn, ast.Attribute)
                        and fn.attr in ("emit", "write_status")
                        and isinstance(fn.value, ast.Name)
                        and fn.value.id == "events"):
                    continue
                checked += 1
                for kw in node.keywords:
                    for sub in ast.walk(kw.value):
                        self.assertNotIsInstance(
                            sub, ast.BinOp,
                            f"{path.name}:{node.lineno} arithmetic inside an "
                            f"events call ({kw.arg}=) — compute it into a "
                            f"local first",
                        )
                        self.assertNotIsInstance(
                            sub, ast.Subscript,
                            f"{path.name}:{node.lineno} indexing inside an "
                            f"events call ({kw.arg}=) — compute it into a "
                            f"local first",
                        )
        self.assertGreater(checked, 5, "expected to find the emit call sites")

    def test_tick_path_has_no_emit(self):
        """
        The tick callback must stay under 50 µs with no I/O.

        A source-level check, because the cost of an emit here would not show
        up as a test failure anywhere else — it would show up as a stalled
        price feed in production.
        """
        from src.execution.oanda_scalper_orchestrator import (
            OandaScalperOrchestrator,
        )

        for name in ("_on_tick", "_get_spread", "_sample_spread_calibration"):
            src = inspect.getsource(getattr(OandaScalperOrchestrator, name))
            self.assertNotIn(
                "events.emit", src,
                f"{name} runs on (or near) the tick path and must not emit",
            )
            self.assertNotIn("events.write_status", src, f"{name} must not write status")


if __name__ == "__main__":
    unittest.main()
