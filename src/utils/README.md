# `src/utils`

Small, dependency-light helpers with no knowledge of brokers, models, or
strategies. Only one module here is live. It sits at the very front of the
runtime pipeline: a broker's 1-minute bar stream goes in, sealed
higher-timeframe candles come out, and everything downstream (indicators,
feature engineering, the model gate) reads the DataFrame it maintains. Nothing
in this folder touches disk or the network.

See the root [GLOSSARY.md](../../GLOSSARY.md) for domain terms (sealed bar,
OHLCV, HTF).

## Files

### `bar_aggregator.py`
`LiveBarAggregator` — converts a stream of 1-minute bars into wall-clock-aligned
candles of any timeframe that divides 60. Instantiated once per (symbol,
timeframe) pair. Its two jobs are window alignment (a 12:34 bar belongs to the
12:30 window, regardless of how many bars arrived) and gap repair (missing
intervals are forward-filled with synthetic flat candles so TA-Lib always sees
an evenly-spaced series). `add_bar()` returning `True` is the signal that a
candle sealed and a strategy should run.

- **Imports from repo:** none — standard library plus Polars only.
- **Imported by:** `src/execution/live_orchestrator.py` (Alpaca scalper),
  `src/execution/factory_orchestrator.py`, `backtest_60.py`.
- **Data artifacts:** none. State is in-memory only; `history_df` is trimmed to
  `history_size` and never persisted.

### `risk_management.py`
**Empty file (0 bytes).** Nothing imports it. The live position-sizing and
bracket logic actually lives in [`src/execution/risk_manager.py`](../execution/)
(`class RiskManager`). Flagged as an abandoned stub — see the glossary-pass
report.

- **Imports from repo:** none. **Imported by:** nothing. **Data artifacts:** none.

### `__init__.py`
Empty package marker. Modules here are imported by path
(`from utils.bar_aggregator import LiveBarAggregator`, which relies on `src`
being on `PYTHONPATH`) rather than re-exported.
