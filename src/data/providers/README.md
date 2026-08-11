# `src/data/providers`

Adapters for **company fundamentals** and **macroeconomic series** — the slow,
descriptive data used by the monthly equities investor to rank stocks. The
forex bot never reads any of this.

Everything here implements a contract defined one level up (`FundamentalProvider`
in [`../fundamentals.py`](../fundamentals.py), `MacroProvider` in
[`../macro.py`](../macro.py)) and obeys the same rule as the market adapters:
**return empty on failure, never raise.** That is what lets the composite
provider treat a failure as "try the next source" instead of a crash.

`FUNDAMENTAL_SOURCES` (read by [`../factory.py`](../factory.py)) picks which of
these get chained, in order.

See the root [GLOSSARY.md](../../../GLOSSARY.md) for domain terms.

> No `__init__.py` — implicit namespace package, same as the parent folder.

## Files

### `composite_fundamentals.py`
`CompositeFundamentalProvider` — wraps an ordered list of providers and returns
the first non-empty answer. It is *itself* a `FundamentalProvider`, so callers
can't tell whether they hold one source or five. An empty list is legal and
answers empty to everything (the supported "no fundamentals" mode).

First-non-empty is resolved **per method call**, so one symbol's company info
and its financials can legitimately come from different vendors. Exceptions
from any source are logged at debug level and treated as a miss.

- **Imports from repo:** `data.fundamentals`.
- **Imported by:** `src/data/factory.py`, `tests/test_composite_fundamentals.py`.
- **Data artifacts:** none of its own.

### `simfin_fundamentals.py`
`SimFinFundamentalProvider` — the default source. Uses a **bulk-download**
model rather than per-symbol API calls: it downloads whole datasets once, caches
them as CSVs, and then answers every symbol from memory.

The subtlety worth knowing: SimFin partitions companies into `general`, `banks`
and `insurance` datasets, because banks and insurers report different line
items. `_find_in_variants` resolves a ticker to its partition transparently, so
callers never need to know a company's sector. Column names are then renamed to
the Yahoo-shaped ones (`Total Revenue`, `Operating Income`) the investor's
feature pipeline already expects, which is what keeps the sources swappable.

- **Requires:** `SIMFIN_API_KEY` in the environment.
- **Imports from repo:** `data.fundamentals`.
- **Imported by:** `src/data/factory.py` — by *string* path, so the SimFin
  package is only loaded if actually selected.
- **Reads/writes:** `data/raw/simfin_cache/` (bulk CSVs, re-downloaded when
  older than 30 days).

### `yf_fundamentals.py`
`YFinanceFundamentalProvider` — **research/PoC only.** Yahoo data is unofficial,
rate-limited, and may disagree with authoritative sources. `_COMPANY_INFO_KEYS`
pins the subset of Yahoo's large, unstable info blob that's actually read, so an
upstream schema change drops fields instead of crashing.

- **Imports from repo:** `data.fundamentals`.
- **Imported by:** `src/data/factory.py` (by string path, as `yfinance`/`yahoo`).
- **Data artifacts:** none (live HTTP).

### `yf_macro.py`
`YFinanceMacroProvider` — maps friendly indicator names onto Yahoo tickers via
`_INDICATOR_MAP` (VIX, 10Y_YIELD, 2Y_YIELD, SP500, NASDAQ, DJI, GOLD, OIL, DXY).

Two documented traps, both inherited from Yahoo and **not corrected in code**:

| Key | Ticker | Trap |
|---|---|---|
| `10Y_YIELD` | `^TNX` | Reported as **yield × 10** — a 4.2% yield arrives as `42.0`. |
| `2Y_YIELD` | `^IRX` | **Misnamed.** `^IRX` is the 13-week T-bill rate, not the 2-year. |

For anything authoritative, replace this with a FRED-backed adapter (`DGS10`,
`DGS2`, `FEDFUNDS`, `CPIAUCSL`) as noted in `../macro.py`.

- **Imports from repo:** `data.macro`.
- **Imported by:** `scripts/investor_data_miner.py`.
- **Data artifacts:** none (live HTTP).
