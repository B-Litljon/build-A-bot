"""
Deterministic tests for ``scripts/option_capability_probe.py``.

Every test mocks both SDK clients; no credential or network is involved.
The three mandatory cases from the dispatch brief (section 8):

- account level 2 -> verdict false, exit 2
- account level 3 + populated chain -> verdict true, exit 0
- empty chain -> verdict false on the data leg
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.option_capability_probe import main, run_probe


def _account(level):
    """Fake `Account` carrying only the approval attribute."""
    return SimpleNamespace(options_approved_level=level)


def _snap(iv=0.18, delta=0.25):
    return SimpleNamespace(
        symbol="SPY241101P00500000",
        implied_volatility=iv,
        greeks=SimpleNamespace(delta=delta, gamma=0.02, theta=-0.03, vega=0.11, rho=0.0),
    )


class _Trading:
    def __init__(self, account):
        self._account = account

    def get_account(self):
        return self._account


class _Option:
    def __init__(self, snaps):
        self._snaps = snaps

    def get_option_chain(self, _req):
        return self._snaps


# ── mandatory case 1: level 2 -------------------------------------------------

def test_level2_account_fails(capsys):
    report = run_probe(
        "SPY",
        trading_client=_Trading(_account(2)),
        option_client=_Option({"SPY241101P00500000": _snap()}),
    )
    assert report["verdict"] is False
    assert report["legs"]["account"]["verdict"] is False
    assert report["legs"]["account"]["attribute_present"] == "options_approved_level"
    assert report["legs"]["chain"]["verdict"] is True  # chain itself is fine


# ── mandatory case 2: level 3 + chain rows ------------------------------------

def test_level3_with_chain_passes():
    report = run_probe(
        "SPY",
        trading_client=_Trading(_account(3)),
        option_client=_Option(
            {
                "SPY241101P00500000": _snap(),
                "SPY241101P00510000": _snap(iv=0.19, delta=0.30),
            }
        ),
    )
    assert report["verdict"] is True
    assert report["legs"]["account"]["verdict"] is True
    assert report["legs"]["chain"]["verdict"] is True
    assert report["legs"]["chain"]["rows"] == 2
    assert report["legs"]["chain"]["iv_present"] is True
    assert report["legs"]["chain"]["delta_present"] is True


# ── mandatory case 3: empty chain ----------------------------------------------

def test_empty_chain_fails():
    report = run_probe(
        "SPY",
        trading_client=_Trading(_account(3)),
        option_client=_Option({}),
    )
    assert report["verdict"] is False
    assert report["legs"]["chain"]["verdict"] is False
    assert report["legs"]["chain"]["rows"] == 0


# ── CLI-level behaviour --------------------------------------------------------

def test_cli_exit_codes(monkeypatch, capsys):
    # Patch credential presence + client construction on the module so main()
    # runs end-to-end against mocks.
    import scripts.option_capability_probe as probe

    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.trading.client import TradingClient

    monkeypatch.setattr(TradingClient, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(OptionHistoricalDataClient, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(
        probe, "run_probe", lambda sym, *, trading_client, option_client, verbose=True: run_probe(
            sym,
            trading_client=_Trading(_account(2)),
            option_client=_Option({"X": _snap()}),
        )
    )
    assert main(["--symbol", "SPY"]) == 2

    monkeypatch.setattr(
        probe, "run_probe", lambda sym, *, trading_client, option_client, verbose=True: run_probe(
            sym,
            trading_client=_Trading(_account(3)),
            option_client=_Option({"X": _snap()}),
        )
    )
    assert main(["--symbol", "SPY"]) == 0


def test_missing_credentials_exit_2(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert main(["--symbol", "SPY"]) == 2
