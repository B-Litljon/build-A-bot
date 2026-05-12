---
type: recon
date: 2026-04-29
time: 12:55:10 PDT
agent: Kimi K2.6
model: kimi-k2.6
trigger: Two grid-search backtest files import core.order_management.OrderParams — classify before Act 1 / Act 2 decisions
head: unknown
scope: read-only
imported_from: GRID_SEARCH_RECON_2026-04-29_1255.md
---

**Date:** 2026-04-29
**Time:** 2026-04-29 12:55:10 PDT
**Agent:** Kimi K2.6
**Trigger:** Two grid-search backtest files import `core.order_management.OrderParams`. Need to classify before Act 1 / Act 2 decisions.

---

## Pre-flight git state

```
$ git status --short
(empty — clean working tree)

$ git rev-parse --abbrev-ref HEAD
main
```

---

## File metadata table

| File | Lines | Size | Last modified | Last commit | Total commits |
|------|-------|------|---------------|-------------|---------------|
| `grid_search_backtest.py` | 233 | 8,116 bytes | 2026-03-08 22:27:05 PDT | `1068ac1` (2026-03-08 21:34:45) | 2 |
| `grid_search_backtest_q1.py` | 221 | 8,341 bytes | 2026-03-08 22:27:05 PDT | `1068ac1` (2026-03-08 21:34:45) | 2 |

**Assessment:** Both files were last touched on 2026-03-08 in the same commit (`1068ac1`: "feat: Expand history size for LiveBarAggregator to support multi-timeframe features"). They have only two commits in their history, suggesting they were created together and have not been independently maintained since.

---

## Imports + intent

### `grid_search_backtest.py` (lines 1–40)

```python
#!/usr/bin/env python3
"""Grid Search Backtest - Threshold 0.50 with 3 Risk Profiles (A/B/C)"""

import sys, os, logging

logging.disable(logging.CRITICAL)
for name in logging.Logger.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).propagate = False
sys.path.insert(0, os.path.abspath("src"))

import polars as pl
from datetime import datetime, timezone
from strategies.concrete_strategies.ml_strategy import MLStrategy
from utils.bar_aggregator import LiveBarAggregator
from core.order_management import OrderParams

# Load data - Test Set (2024 onwards)
df = pl.read_parquet("data/raw/SPY_1min.parquet")
df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
test_df = df.filter(pl.col("timestamp") >= start)

print(f"Grid Search: {len(test_df)} bars (2024 Full Year - Test Set)")
print(f"Threshold: 0.50 (Model Native Optimization Point)")
print("=" * 70)

# Risk Profiles to test (as specified by user)
risk_profiles = [
    ("Config A (Scalper)", 0.998, 1.005),  # SL 0.2%, TP 0.5%
    ("Config B (Balanced)", 0.995, 1.005),  # SL 0.5%, TP 0.5%
    ("Config C (Swinger)", 0.995, 1.010),  # SL 0.5%, TP 1.0%
]

results = []
TIMEOUT_BARS = 15
```

### `grid_search_backtest_q1.py` (lines 1–40)

```python
#!/usr/bin/env python3
"""Grid Search Backtest - Threshold 0.50 with 3 Risk Profiles (Q1 2024)"""

import sys, os, logging

logging.disable(logging.CRITICAL)
for name in logging.Logger.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).propagate = False
sys.path.insert(0, os.path.abspath("src"))

import polars as pl
from datetime import datetime, timezone
from strategies.concrete_strategies.ml_strategy import MLStrategy
from utils.bar_aggregator import LiveBarAggregator
from core.order_management import OrderParams

# Load data - Q1 2024 Test Set
df = pl.read_parquet("data/raw/SPY_1min.parquet")
df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end = datetime(2024, 4, 1, tzinfo=timezone.utc)
test_df = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))

print(f"Grid Search: {len(test_df)} bars (Q1 2024 - Test Set)")
print(f"Threshold: 0.50 (Model Native Optimization Point)")
print("=" * 80)

# Risk Profiles to test (as specified by user)
risk_profiles = [
    ("Config A (Scalper)", 0.998, 1.005),   # SL 0.2%, TP 0.5%
    ("Config B (Balanced)", 0.995, 1.005),  # SL 0.5%, TP 0.5%
    ("Config C (Swinger)", 0.995, 1.010),   # SL 0.5%, TP 1.0%
]

results = []
TIMEOUT_BARS = 15
```

**No `if __name__ == "__main__"` block in either file.** Both scripts run at module level when executed directly.

---

## Strategy references

### `grid_search_backtest.py`

```
14:from strategies.concrete_strategies.ml_strategy import MLStrategy
45:    strategy = MLStrategy(model_path="src/ml/models/rf_model.joblib", threshold=0.50)
```

### `grid_search_backtest_q1.py`

```
14:from strategies.concrete_strategies.ml_strategy import MLStrategy
43:    strategy = MLStrategy(model_path="src/ml/models/rf_model.joblib", threshold=0.50)
```

**Assessment:** Both files import **only** `MLStrategy` (the live ML strategy). They do **not** import any V1 strategies (`RSIBBands`, `SMACrossover`). The constructor call uses `model_path` and `threshold` — an **older signature** than the current `MLStrategy` which expects `angel_path`/`devil_path`/`angel_threshold`/`devil_threshold`.

---

## Grid-search patterns

### `grid_search_backtest.py`

| Pattern | Lines | Details |
|---------|-------|---------|
| Parameter sweep | 28–33 | `risk_profiles` list of 3 tuples: `(name, sl_mult, tp_mult)` |
| Results collection | 35, 152–164 | `results = []` + `results.append({...})` with profile metrics |
| Disk output | **None** | Results are printed to stdout only; no CSV / Parquet / JSON write |

### `grid_search_backtest_q1.py`

| Pattern | Lines | Details |
|---------|-------|---------|
| Parameter sweep | 29–34 | Identical `risk_profiles` list |
| Results collection | 36, 144–155 | `results = []` + `results.append({...})` |
| Disk output | **None** | Results are printed to stdout only |

**Assessment:** Neither file writes results to disk. The "grid search" is a hard-coded 3-config sweep (A/B/C risk profiles) with no `itertools.product`, no `param_grid`, and no hyperparameter exploration beyond SL/TP multipliers.

---

## Cross-reference results

```
./table-o-content.md:587:| `grid_search_backtest.py` | Full 2024 SPY | Bar-by-bar simulation with `MLStrategy.analyze()` per candle. | Same A/B/C configs, threshold 0.50, 15-bar timeout. |
./table-o-content.md:588:| `grid_search_backtest_q1.py` | Q1 2024 SPY | Same as above, restricted to Jan-Mar 2024 for faster iteration. | Same A/B/C configs. |
./table-o-content.md:1047:| `grid_search_backtest.py` | Bar-by-bar grid search (full year) |
./table-o-content.md:1048:| `grid_search_backtest_q1.py` | Bar-by-bar grid search (Q1 only) |
```

**Assessment:** Neither file is invoked by `run_*.py`, CI configs, Makefiles, or shell scripts. They are referenced only in `table-o-content.md` (project documentation). They appear to be standalone manual-run scripts, not integrated into any automated pipeline.

---

## Verdict

| File | Classification | Rationale |
|------|----------------|-----------|
| `grid_search_backtest.py` | **LIVE ML — STALE** | Imports `MLStrategy` (live ML strategy), not V1. Uses an outdated `MLStrategy` constructor signature (`model_path`, `threshold` vs current `angel_path`/`devil_path`). Imports `OrderParams` from missing `core.order_management`. No disk output. Only referenced in docs. |
| `grid_search_backtest_q1.py` | **LIVE ML — STALE** | Identical to above, just restricted to Q1 2024 for faster iteration. Same classification. |

---

## Summary and recommendation

These two files are **near-identical quarterly variants** of the same backtest script. `grid_search_backtest.py` runs the full 2024 SPY dataset; `grid_search_backtest_q1.py` runs only Jan–Mar 2024 for faster iteration. Both are **live ML backtest tooling**, not dead V1 code, but they are **significantly stale**:

1. **Constructor mismatch:** They call `MLStrategy(model_path=..., threshold=0.50)`, but the current `MLStrategy.__init__` expects `angel_path`, `devil_path`, `angel_threshold`, `devil_threshold`, and `warmup_period`. This script would fail immediately if run against the current codebase.

2. **Missing module dependency:** Both import `OrderParams` from `core.order_management`, which no longer exists. Whoever migrates these files will need to point them at the new OrderParams location (created in Act 2).

3. **No result persistence:** The scripts print a summary table to stdout but do not write CSV, Parquet, or JSON. Any results produced are ephemeral. If these are meant to be part of a reproducible research workflow, they need disk output.

4. **Manual execution only:** They are not referenced by any runner, CI, or automation. They are standalone scripts for manual experimentation.

**Recommendation:** Do **not** delete these in Act 1. Flag them for **Act 2 (or a follow-up Act 3)** to:
- Update the `MLStrategy` constructor call to match the current Angel/Devil signature
- Replace the `core.order_management` import with the new OrderParams location
- Optionally merge the two files into a single script with a `--date-range` CLI argument, since they differ only in `start`/`end` filtering
- Optionally add disk output (CSV/Parquet) if these results are meant to be archived

---

*End of report*
