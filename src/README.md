# `src/`

All the library code. Two loose .py files live here; everything else is a
package with its own README.

**Import convention:** entry-point scripts prepend `src/` to `sys.path`, which
is why modules import each other as `data.factory` rather than
`src.data.factory`. Both spellings appear in the tree (the offline pipeline
tends to use the `src.` prefix, live code tends not to) — that inconsistency is
real, harmless, and worth knowing before you assume a module is missing.

## The packages, by when they run

| Package | Role | Runs during |
|---|---|---|
| [`data/`](data/) | Vendor adapters behind one interface. `DATA_SOURCE` picks one. | training + live |
| [`ml/`](ml/) | The feature factory — bars → model inputs. | training + live |
| [`strategies/`](strategies/) | The decision. Bars in, `Signal` or `None` out. | live |
| [`execution/`](execution/) | Brokers, orders, position state, software stops. | **live** |
| [`core/`](core/) | Shared types, Discord notifier, **and the training pipeline**. | mixed |
| [`utils/`](utils/) | Bar aggregation. | live |
| [`analysis/`](analysis/) | Offline diagnostics, run by hand. | never live |
| [`lab/`](lab/) | The feature lab: candidate features scored by the retrainer's own gate (built 2026-09-21; v2 frame-hash/ablate/A-B 2026-09-23). | never live |

The one that surprises people is `core/` — it holds both shared domain types
*and* `retrainer/` (a package since 2026-09-16), the entire training and promotion
pipeline. See
[`core/README.md`](core/).

A `day_trading/` package (a dormant 5-minute experiment) lived here until
2026-09-16; it and the whole Alpaca scalper lane were deleted in the
downsizing pass — git history has them.

## Data flow

```
provider → LiveBarAggregator → FeaturePipeline → MLStrategy (angel→devil)
                                                      ↓ Signal
                                          RiskManager (sizing + chop veto)
                                                      ↓
                                              Orchestrator → broker
```

Training runs the same middle section against saved history, which is the point
— see the symmetry note in [`ml/README.md`](ml/).

## Loose files

### `replay_test.py`
Phase 2 of `run_pipeline.sh`. Replays saved bars past the models as if arriving
live and records every signal. **Cross-sectional**: it advances all symbols one
timestamp at a time rather than finishing one symbol before starting the next,
which is what a real feed does. `MockAlpacaProvider` stands in for the feed.

Its docstring says "CSV export" but it writes **parquet** — stale wording.

- **Imports from repo:** `ml.feature_pipeline`, `ml.features.v3_features`,
  `ml.train_model`, `ml.trainers.v3_rf_trainer`.
- **Reads:** `data/oos_bars.parquet`, legacy root-level model pickles.
  **Writes:** `data/signal_ledger.parquet`.

### `evaluate_performance.py`
Grades those recorded signals: win rate, net profit, max drawdown, profit
factor, using volatility-scaled brackets (0.5×/3.0× ATR, 45-bar hold).

This is what `run_pipeline.sh` Phase 3 **actually runs** — not
`src/core/resolver.py`, despite that module's usage line claiming otherwise.

Note its regime-aware threshold: `BASE_THRESHOLD = 0.50` normally, raised to
`HIGH_VOLATILITY_THRESHOLD = 0.75` in high-volatility conditions, because
that's where drift analysis found calibration breaking down.

- **Imports from repo:** none.
- **Reads:** `data/oos_bars.parquet`, `data/signal_ledger.parquet`,
  `data/drift_report.json`. **Writes:** `data/evaluation_results.parquet`.

> ⚠️ **Ledger format split:** `replay_test.py` writes and
> `evaluate_performance.py` / `reinforcement_voter.py` read
> `signal_ledger.**parquet**`, while `core/resolver.py` reads
> `signal_ledger.**csv**`. Same base name, different formats — further evidence
> `resolver.py` is orphaned.
