# `src/day_trading` — "Universal Scalper V4.0"

A **self-contained, dormant experiment**: a separate Angel/Devil model that
trades 5-minute bars and closes everything by the end of the day. It is *not*
the live scalper and *not* the monthly equities investor — a third, distinct
thing that happens to share the two-stage idea.

Nothing outside this folder imports it. Every artifact carries a **`dt_`
prefix** (`data/raw/dt_*.parquet`, `models/dt_angel_latest.pkl`), which is the
mechanism that lets it coexist with the live system without any chance of
overwriting a production model or dataset. Listed under "The Boneyard" in
[`table-o-content.md`](../../table-o-content.md).

## What makes it different from the main scalper

Two ideas drive nearly every design choice here:

1. **Two time scales at once.** Decisions are made on 5-minute bars, but
   "how big is a normal move for this stock" comes from *daily* bars. Every
   bracket and label is expressed as a fraction of the daily range, which is
   what lets one threshold work across a \$20 stock and a \$500 one.
2. **The trading day has a shape.** A session has an open, a middle and a
   close, so the model gets features for *where in the day it is* — and,
   because everything is closed at the bell, entries are only labelled tradeable
   in the **first 90 minutes**. A late entry has too little time left to work.

## Pipeline

```
harvester_5m.py  →  build_dataset.py  →  train_model.py
   (fetch)            (features+labels)     (fit + validate)
```

See the root [GLOSSARY.md](../../GLOSSARY.md) for shared terms (angel/devil,
NATR, bracket, walk-forward, OOF).

## Files

### `harvester_5m.py`
Fetches **two** datasets per symbol: 5-minute bars (the decision timeframe) and
daily bars (the context). `DAILY_WARMUP_DAYS = 30` pulls extra daily history
beyond the training window so the 14-period daily volatility measure is already
warm on the very first 5-minute training bar — without it the earliest rows
would carry null daily context and be dropped.

- **Imports from repo:** `data.enums`, `data.timeframe`.
- **Writes:** `data/raw/dt_<SYMBOL>_5min.parquet`,
  `data/raw/dt_<SYMBOL>_daily.parquet`.

### `features.py`
Three generators chained through the shared `FeaturePipeline`:
`DayTradeBaseFeatures` (TA-Lib on 5-minute bars, same periods as the scalper),
`DayTradeDailyJoin` (yesterday's finished daily numbers — **only the previous
session's close, never today's still-forming daily bar**), and
`DayTradeIntradayFeatures` (session dynamics).

`DAY_TRADE_FEATURE_COLS` is the 22-column model input in four groups. The
session-aware ones are the interesting additions: `session_progress` (0.0 at
the open → 1.0 at the close), `vwap_dist` (above or below what the average
participant paid today), `gap_pct` (overnight move), `first_30m_vol_rel` (was
the open busy?), and `range_exhaustion` (how much of a typical day's range is
already used — near 1.0 means little movement may be left).

- **Imports from repo:** `ml.core.interfaces`, `ml.feature_pipeline`.
- **Data artifacts:** none — pure DataFrame transformation.

### `targets.py`
The two labels, both scaled by the **daily** range:

- **Angel** — was the best price reached before the close (the "maximum
  favorable excursion") at least `0.6 ×` the daily range? *Was there a real
  move to catch?*
- **Devil** — did price avoid falling `0.4 ×` the daily range against you
  first? *Direction is worthless if you'd have been stopped out on the way.*

`ENTRY_WINDOW_MAX_PROGRESS ≈ 0.2308` (90 of 390 minutes) restricts positive
labels to the first 90 minutes. Later bars are forced to **0, not null** — a
deliberate choice so they remain in the dataset as negative examples and the
indicator warm-up is preserved.

- **Imports from repo:** `ml.core.interfaces`. **Data artifacts:** none.

### `build_dataset.py`
Runs features + targets per symbol, then concatenates. Per-symbol processing is
what stops one symbol's history leaking into another's indicators.

- **Imports from repo:** `day_trading.features`, `day_trading.targets`,
  `ml.feature_pipeline`.
- **Reads:** `data/raw/dt_*.parquet`.
  **Writes:** `data/processed/dt_training_data.parquet`.

### `train_model.py`
The two-stage fit with its own walk-forward validation (2 expanding date-based
folds, ~30% unseen each).

The detail worth carrying over to any other two-stage training you write:
`ANGEL_OOF_SPLITS = 5` generates the Angel probabilities **out of fold**, so the
rows the Devil trains on were selected by models that never saw them. Without
that, the Devil trains on the Angel's overconfident in-sample opinions and
inherits its blind spots. Likewise the Devil *trains* on the survival question
but is *evaluated* on the full outcome, so the reported profit factor reflects
real trades rather than the proxy.

- **Imports from repo:** `day_trading.features` (for `DAY_TRADE_FEATURE_COLS`).
- **Reads:** `data/processed/dt_training_data.parquet`.
- **Writes:** `models/dt_angel_latest.pkl`, `models/dt_devil_latest.pkl`,
  `models/dt_threshold.json`.

### `__init__.py`
Package marker noting the isolation rule.
