"""
Lane 4 Stage 0: Alpaca options capability probe (READ-ONLY).

Answers one question before any research runs: can this account source and
trade multi-leg index options? Prints a single JSON verdict and exits
``0`` only if both mandatory legs hold:

    1. ``options_approved_level >= 3`` on the account (spreads/multi-leg).
    2. Historical option chain data returns rows carrying
       ``implied_volatility`` and ``delta``.

Greeks (leg 3) are reported but advisory only. A missing account permission
or empty chain exits ``2`` with a one-line human-readable blocker; the lane's
stop report is written from this output. No order of any kind is submitted —
``TradingClient`` is used for ``get_account()`` only.

Run from the repo worktree root:

    PYTHONPATH=src:. python scripts/option_capability_probe.py --symbol SPY

The account attribute carrying the approval level is NOT assumed — Alpaca has
exposed it variously as ``options_approved_level`` and
``options_trading_level``; the probe introspects ``dir(account)`` and reports
both spellings, preferring ``options_approved_level``.

Glossary:
    options_approved_level -- Alpaca account field: 0=disabled, 1=covered
        calls/cash-secured puts, 2=long calls/puts, 3=spreads/straddles
        (the level this lane needs for short credit spreads / iron condors).
    OptionChainRequest -- the SDK request model for
        ``OptionHistoricalDataClient.get_option_chain``; carries
        ``underlying_symbol`` plus expiry/strike-range filters.
    leg -- one verdict in the JSON report (``account``/``chain``/``greeks``).

Deterministic tests mock the two SDK clients, so no credential is needed for
``pytest``; only a manual/live probe reads ``.env``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The probe runs from worktrees of the build-A-bot checkout; credentials live
# in the shared main checkout's .env, so prefer it over a worktree-local file
# (which may not exist) without ever printing a value.
for _env_path in (_PROJECT_ROOT / ".env", Path("/mnt/storage/mystuf/development/build-A-bot/.env")):
    if _env_path.is_file():
        load_dotenv(_env_path)
        break

LEVEL_ATTR_CANDIDATES = (
    "options_approved_level",
    "options_trading_level",
    "options_buying_level",
)
MIN_APPROVAL_LEVEL = 3
_MIN_CHAIN_ROWS = 1


def _extract_approval_level(account: object) -> tuple[str | None, int | None]:
    """
    Return (attribute_name, level) from a live or mock account.

    Introspection order follows Alpaca's own docs; ``dir()`` catches mock
    objects too. An attribute present but ``None`` means the account was never
    approved and reads as level 0 — that is a real answer, not "unknown".
    """
    for name in LEVEL_ATTR_CANDIDATES:
        if name in dir(account):
            value = getattr(account, name)
            return name, int(value) if value is not None else 0
    return None, None


def run_probe(symbol: str, *, trading_client, option_client, verbose: bool = True) -> dict:
    """
    Execute both mandatory legs against already-constructed clients.

    Split from ``main`` so tests can pass mocks without mocking the SDK's
    import machinery. Never calls an order endpoint.
    """
    report: dict = {
        "symbol": symbol,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "legs": {},
    }

    # ── Leg 1: account approval ──────────────────────────────────────────
    account = trading_client.get_account()
    attr_name, level = _extract_approval_level(account)
    account_leg = {
        "attribute_present": attr_name,
        "options_approved_level": level,
        "verdict": (level is not None and level >= MIN_APPROVAL_LEVEL)
        if attr_name is not None
        else "unknown",
    }
    report["legs"]["account"] = account_leg

    # ── Leg 2: historical option chain ───────────────────────────────────
    from alpaca.data.requests import OptionChainRequest

    # "Narrow window": contract expirations within the next ~3 weeks.
    # Strike range is NOT set here — Alpaca returns every listed strike of a
    # SPY/QQQ/IWM chain (thousands of rows before expiry filtering); the
    # probe's job is data availability, not strike selection.
    today = _dt.date.today()
    req = OptionChainRequest(
        underlying_symbol=symbol,
        expiration_date_gte=today,
        expiration_date_lte=today + _dt.timedelta(days=21),
    )
    chain_leg: dict = {"verdict": False, "rows": 0}
    try:
        snapshots = option_client.get_option_chain(req)
    except Exception as exc:  # network, entitlement, or schema failure
        chain_leg.update({"error": f"{type(exc).__name__}: {exc}", "rows": 0})
    else:
        snapshots = snapshots or {}
        rows = len(snapshots)
        exps, deltas_seen = [], False
        iv_seen = False
        strikes = []
        for sym, snap in snapshots.items():
            # Contract symbol encodes expiry: SPY  241101P00500000 — but the
            # OCC string is not parseable without a helper, so take the
            # expiry from the snapshot's symbol attribute if present.
            exp = getattr(snap, "symbol", sym)
            exps.append(exp[3:9] if isinstance(exp, str) else None)
            if getattr(snap, "implied_volatility", None) is not None:
                iv_seen = True
            greeks = getattr(snap, "greeks", None)
            if greeks is not None and getattr(greeks, "delta", None) is not None:
                deltas_seen = True
        chain_leg.update(
            {
                "rows": rows,
                "iv_present": iv_seen,
                "delta_present": deltas_seen,
                "sample_symbols": sorted(snapshots.keys())[:3],
                "verdict": rows >= _MIN_CHAIN_ROWS and iv_seen and deltas_seen,
            }
        )
    report["legs"]["chain"] = chain_leg

    # ── Leg 3: greeks advisory ───────────────────────────────────────────
    greeks_ok = chain_leg.get("delta_present", False)
    report["legs"]["greeks"] = {
        "verdict": greeks_ok,
        "note": "greeks ride the option snapshot (OptionsGreeks: delta/gamma/theta/vega/rho); "
        "not mandatory at Stage 0 because the harvester needs only iv + delta for its filters",
    }

    both_mandatory = (
        report["legs"]["account"]["verdict"] is True
        and report["legs"]["chain"]["verdict"] is True
    )
    report["verdict"] = both_mandatory
    if not both_mandatory:
        missing = [
            name
            for name in ("account", "chain")
            if report["legs"][name]["verdict"] is not True
        ]
        report["blocker"] = (
            "option capability BLOCKED: missing "
            + ", ".join(missing)
            + (" (account: level %s) " % level if "account" in missing else "")
        )
    if verbose:
        print(json.dumps(report, indent=2, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="SPY", choices=["SPY", "QQQ", "IWM"])
    args = parser.parse_args(argv)

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("option capability BLOCKED: ALPACA_API_KEY / ALPACA_SECRET_KEY not set", file=sys.stderr)
        return 2

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.trading.client import TradingClient

    trading_client = TradingClient(api_key, secret_key, paper=True)
    option_client = OptionHistoricalDataClient(api_key, secret_key)
    report = run_probe(args.symbol, trading_client=trading_client, option_client=option_client)

    if not report["verdict"]:
        print(report.get("blocker", "option capability BLOCKED"), file=sys.stderr)
        return 2
    print(
        "option capability OK: %s level=%s, chain rows=%s"
        % (
            args.symbol,
            report["legs"]["account"]["options_approved_level"],
            report["legs"]["chain"]["rows"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
