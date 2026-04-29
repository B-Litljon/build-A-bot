# STOP AND REPORT: Contract Inconsistency Detected

During Task 1, I inspected the three concrete provider classes (`AlpacaProvider`, `PolygonDataProvider`, and `YahooDataProvider`). I discovered that they **do not all implement the same contract**.

Specifically:
- `PolygonDataProvider` and `YahooDataProvider` implement four methods: `get_active_symbols`, `get_historical_bars`, `subscribe`, and `run_stream`.
- `AlpacaProvider` implements **only one** method: `get_historical_bars`. It is completely missing `get_active_symbols`, `subscribe`, and `run_stream`. (It also does not currently inherit from any ABC).

Furthermore, while the docstring in `src/data/factory.py` claims that the returned `MarketDataProvider` will be "ready for ``get_historical_bars``, ``subscribe``, and ``run_stream``", the default Alpaca provider lacks the streaming methods entirely (they seem to be implemented separately in `src/data/feed.py` as `AlpacaCryptoFeed`).

Per the explicit instruction: *"If you discover during work that the three concrete providers have inconsistent method signatures (i.e. they don't all implement the same contract), stop and report — do not attempt to reconcile silently"*, I have halted the operation and made no modifications to any files.

Please advise on how you would like to proceed (e.g., should I write an ABC matching only the Polygon/Yahoo contract, or a minimal one with just `get_historical_bars`?).
