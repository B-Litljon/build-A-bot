"""
Tests for the serving-side pipette quantization of live bar prices
(2026-10-01, gated by OANDA_ROUND_MID_TO_PIPETTE — DEFAULT OFF).

The training data (OANDA REST candles) is pipette-quantized: JPY crosses to
3 decimals, 5-decimal pairs to 5. The streamed mid = (bid + ask) / 2 of two
adjacent-pipette quotes carries a half-pipette digit the candles can never
print, which is why ~79% of GBP_JPY live closes diverged from the REST candle
at the same timestamp (audit, 2026-09). When enabled, the prices _flush_bar
EMITS are quantized to the instrument's pipette; the raw tick callback, the
internal bar state, and the volume field are never touched.

The headline invariant of this file: with the flag UNSET the emitted bar is
bit-identical to the pre-fix behaviour — the soak reads this tree, so live
price handling may only change by explicit env choice.

Glossary:
    OANDA_ROUND_MID_TO_PIPETTE -- the env flag; unset/0 = inert.
    _round_mids -- the provider's subscribe-time snapshot of the flag (a
        restart is the switch point; a mid-stream env flip must not change
        price handling).
    PIPETTE_DECIMALS -- JPY crosses 3; everything else 5
        (OANDA_PIPETTE_DECIMALS_DEFAULT).
    ROUND_HALF_UP -- the tie convention; exact .5 ties are common on adjacent-
        pipette quotes and builtin round() would decide them by the binary
        float representation (and half-to-even).
"""

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.data.oanda_provider import (  # noqa: E402
    OANDA_PIPETTE_DECIMALS_DEFAULT,
    OandaMarketProvider,
    PIPETTE_DECIMALS,
    _mid_rounding_enabled,
    _round_mid,
    pipette_decimals,
)

_JPY_RAW = (150.123 + 150.125) / 2.0  # 150.124 float; one digit past 3dp absent


def _flush(provider, instrument, prices):
    """Drive _flush_bar and return the emitted bar dict."""
    store = []

    async def cb(bar):
        store.append(bar)

    provider._loop = None  # forces the asyncio.run(self._callback(bar)) path
    provider._callback = cb
    open_, high, low, close = prices
    state = {
        "epoch": 0,
        "bar_start": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 7,
    }
    provider._flush_bar(instrument, state)
    return store[0], state


class TestPipetteTable(unittest.TestCase):
    def test_jpy_pairs_are_three_decimals(self):
        for sym in ("GBP_JPY", "EUR_JPY", "AUD_JPY", "NZD_JPY", "USD_JPY"):
            self.assertEqual(pipette_decimals(sym), 3, sym)
            self.assertIn(sym, PIPETTE_DECIMALS)

    def test_non_jpy_default_is_five(self):
        for sym in ("GBP_AUD", "GBP_NZD", "EUR_USD", "XAU_USD", "US2000_USD"):
            self.assertEqual(pipette_decimals(sym), 5, sym)
        self.assertEqual(OANDA_PIPETTE_DECIMALS_DEFAULT, 5)

    def test_slash_and_bare_spellings_normalise(self):
        self.assertEqual(pipette_decimals("GBP/JPY"), 3)


class TestRoundMid(unittest.TestCase):
    def setUp(self):
        self._had = os.environ.pop("OANDA_ROUND_MID_TO_PIPETTE", None)

    def tearDown(self):
        if self._had is not None:
            os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = self._had
        else:
            os.environ.pop("OANDA_ROUND_MID_TO_PIPETTE", None)

    def test_default_off_returns_mid_unchanged(self):
        self.assertFalse(_mid_rounding_enabled())
        self.assertEqual(_round_mid("GBP_JPY", _JPY_RAW), _JPY_RAW)
        self.assertEqual(_round_mid("GBP_JPY", 150.1235), 150.1235)  # tie NOT rounded

    def test_jpy_rounds_to_three(self):
        os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = "1"
        self.assertEqual(_round_mid("GBP_JPY", _JPY_RAW), 150.124)
        # Real live-divergence shape: adjacent-pipette quotes, half-pipette rem.
        self.assertEqual(_round_mid("GBP_JPY", (150.1231 + 150.1232) / 2.0), 150.123)

    def test_five_dp_pair_rounds_to_five(self):
        os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = "1"
        self.assertEqual(_round_mid("GBP_AUD", (0.65012 + 0.65013) / 2.0), 0.65013)
        self.assertEqual(_round_mid("EUR_USD", (1.08000 + 1.08005) / 2.0), 1.08003)

    def test_exact_tie_rounds_half_up(self):
        """150.1235 -> 150.124 (builtin round() gives 150.123 here — half-to-
        even decided by the binary float)."""
        os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = "1"
        self.assertEqual(_round_mid("GBP_JPY", 150.1235), 150.124)

    def test_falsey_values_disable(self):
        for v in ("0", "false", "no", "off", ""):
            os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = v
            self.assertFalse(_mid_rounding_enabled(), repr(v))


class TestFlushBarInertByDefault(unittest.TestCase):
    """THE invariant: with the flag unset, emitted bars are bit-identical to
    the pre-fix behaviour, and internal tick state is never rounded."""

    def setUp(self):
        self._had = os.environ.pop("OANDA_ROUND_MID_TO_PIPETTE", None)
        self.provider = OandaMarketProvider(
            environment="practice", api_key="fake", account_id="123"
        )

    def tearDown(self):
        if self._had is not None:
            os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = self._had
        else:
            os.environ.pop("OANDA_ROUND_MID_TO_PIPETTE", None)

    def test_default_off_emits_unrounded_and_state_untouched(self):
        self.assertFalse(self.provider._round_mids)
        bar, state = _flush(self.provider, "GBP_JPY", (_JPY_RAW,) * 4)
        self.assertEqual(bar["close"], _JPY_RAW)
        self.assertEqual(bar["open"], _JPY_RAW)
        self.assertEqual(state["close"], _JPY_RAW)
        self.assertEqual(bar["volume"], 7.0)

    def test_flag_on_emits_rounded_but_state_untouched(self):
        os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = "1"
        provider = OandaMarketProvider(
            environment="practice", api_key="fake", account_id="123"
        )
        bar, state = _flush(provider, "GBP_JPY", (_JPY_RAW,) * 4)
        self.assertEqual(bar["close"], 150.124)
        self.assertEqual(state["close"], _JPY_RAW, "internal tick state stays raw")
        self.assertEqual(bar["volume"], 7.0, "volume is never rounded")

    def test_flag_on_gbp_aud_five_decimals(self):
        os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = "1"
        provider = OandaMarketProvider(
            environment="practice", api_key="fake", account_id="123"
        )
        aud = (0.65012 + 0.65013) / 2.0
        bar, _ = _flush(provider, "GBP_AUD", (aud,) * 4)
        self.assertEqual(bar["close"], 0.65013)

    def test_snapshot_at_init_flag_flip_midstream_ignored(self):
        """The provider reads the flag ONCE (at construction); a mid-stream env
        flip must not change price handling — a restart is the switch point."""
        provider = OandaMarketProvider(
            environment="practice", api_key="fake", account_id="123"
        )
        self.assertFalse(provider._round_mids)
        os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = "1"
        bar, _ = _flush(provider, "GBP_JPY", (_JPY_RAW,) * 4)
        self.assertEqual(bar["close"], _JPY_RAW, "mid-stream flip must be ignored")

    def test_flag_on_before_init_rounds(self):
        os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = "1"
        provider = OandaMarketProvider(
            environment="practice", api_key="fake", account_id="123"
        )
        self.assertTrue(provider._round_mids)
        bar, _ = _flush(provider, "GBP_JPY", (_JPY_RAW,) * 4)
        self.assertEqual(bar["close"], 150.124)


class TestEndToEndTickPipeline(unittest.TestCase):
    """Full _handle_tick -> _flush_bar with the flag on: a JPY bar's emitted
    prices carry at most 3 decimals though individual mids were finer."""

    def _msg(self, bid, ask, time="2024-01-01T00:00:00.000000000Z"):
        return {
            "type": "PRICE",
            "instrument": "GBP_JPY",
            "time": time,
            "bids": [{"price": bid}],
            "asks": [{"price": ask}],
        }

    def setUp(self):
        self._had = os.environ.pop("OANDA_ROUND_MID_TO_PIPETTE", None)

    def tearDown(self):
        if self._had is not None:
            os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = self._had
        else:
            os.environ.pop("OANDA_ROUND_MID_TO_PIPETTE", None)

    def _setup_provider(self):
        self.provider = OandaMarketProvider(
            environment="practice", api_key="fake", account_id="123"
        )
        self.bars = []

        async def cb(bar):
            self.bars.append(bar)

        self.provider._callback = cb
        self.provider._loop = None
        self.provider.subscribe(["GBP/JPY"], cb)

    def _run(self, ticks):
        for m in ticks:
            self.provider._handle_tick(m)
        return self.bars[0]

    def test_bar_close_quantized_when_enabled(self):
        os.environ["OANDA_ROUND_MID_TO_PIPETTE"] = "1"
        self._setup_provider()
        # Bar 1: mids 150.12385, 150.12455; flush via the third tick's rollover.
        self._run(
            [
                self._msg("150.1238", "150.1239"),
                self._msg("150.1245", "150.1246"),
                self._msg(
                    "150.1300",
                    "150.1301",
                    time="2024-01-01T00:02:00.000000000Z",
                ),
            ]
        )
        emitted = self.bars[0]
        self.assertEqual(emitted["close"], 150.125)
        self.assertEqual(emitted["high"], 150.125)
        for v in (emitted["open"], emitted["high"], emitted["low"], emitted["close"]):
            self.assertEqual(round(v, 3), v)
        # The now-current bar's internal state keeps the raw mid until flush.
        self.assertEqual(
            self.provider._tick_bars["GBP_JPY"]["close"],
            (150.1300 + 150.1301) / 2.0,
        )

    def test_bar_close_unrounded_when_disabled(self):
        self._setup_provider()
        self._run(
            [
                self._msg("150.1238", "150.1239"),
                self._msg(
                    "150.1300",
                    "150.1301",
                    time="2024-01-01T00:02:00.000000000Z",
                ),
            ]
        )
        # The flush fires on the ROLLOVER tick, so the emitted bar is the old
        # one — its close is the LAST mid of the 00:00 bar, still carrying the
        # raw half-pipette digit (150.12385) untouched.
        self.assertEqual(self.bars[0]["close"], 150.12385)


if __name__ == "__main__":
    unittest.main()