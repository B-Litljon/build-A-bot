# `scripts/`

Hand-run tools. Nothing here is imported by the trading code; each file is a
standalone program you invoke yourself (or, for the investor, from cron).

Three unrelated groups share this folder:

1. **The V4 Investor** (5 files) — a complete second product: a monthly stock
   ranker on Alpaca. Holds positions for weeks, never watches a tick.
2. **Diagnostics & calibration** (3 files) — tools for interrogating a frozen
   model or turning live measurements into config.
3. **Factory launchers** (2 files) — paper-trading runners for the Alpaca
   Factory path.

See the root [GLOSSARY.md](../GLOSSARY.md) for domain terms.

---

## 1. The V4 Investor — monthly stock ranker

A separate product from the forex scalper, sharing almost nothing but the repo.
Pipeline:

```
investor_data_miner → investor_feature_pipeline → investor_train_model
                                ↓
                      portfolio_orchestrator  (monthly cron, places orders)
```

### `investor_universe.py`
`UNIVERSE` (96 tickers) and `SECTORS` (ticker → sector). Single source of truth,
imported by the other three, so they can never disagree about which companies
exist. Every name was chosen to have a **full 5-year history** — no recent
listings — so walk-forward folds aren't silently unbalanced by companies that
didn't yet exist.
- **Imported by:** `investor_data_miner.py`, `investor_feature_pipeline.py`,
  `portfolio_orchestrator.py`.

### `investor_data_miner.py`
Merges three layers that update at wildly different rates — daily prices,
quarterly company financials, daily macro (VIX, 10Y yield) — into one daily
table.

> **`_FUNDAMENTAL_LAG_DAYS = 45` is the most important number here.** Company
> results aren't public the instant a quarter ends; filing takes weeks. Every
> fundamental is shifted forward 45 days before joining, so each row only sees
> numbers genuinely published by then. Without it the model ranks stocks using
> earnings nobody had seen — brilliant in testing, useless live.

- **Imports from repo:** `data.factory`, `data.providers.yf_macro`.
- **Writes:** `data/raw/v4_investor_data.parquet`.

### `investor_feature_pipeline.py`
Builds momentum (63/126/252-day trailing returns), smoothed macro trends, and
margin ratios, plus the target: **`target_top_quintile`** — did this stock land
in the best-performing fifth of the universe over the next 60 trading days.
That's a *relative* question, which is what makes this a ranker rather than a
return predictor.

`--inference` switches the output file and **retains** the newest rows that
training deliberately drops (their 60-day future hasn't happened, so they have
no label) — those rows are exactly what prediction needs.

- **Reads:** `data/raw/v4_investor_data.parquet`.
- **Writes:** `data/processed/v4_training_features.parquet` or
  `v4_inference_features.parquet`.

### `investor_train_model.py`
LambdaRank (learning-to-rank — it optimises the *order* of the list, not each
score, matching the job) with expanding walk-forward folds and its own
promotion gate.

Two subtleties:
- **`EMBARGO_DAYS = 60`** — the target looks 60 days ahead, so a training
  window's tail overlaps the test window's future. A 60-day gap between them
  prevents scoring the model on outcomes it partly saw.
- **Gates are on *lift*, not raw precision.** The target is a top *quintile*, so
  random guessing scores 0.20; `GATE_P1_MIN_LIFT = 1.3` means 30% better than
  chance. `GATE_P8_MIN_LIFT = 1.1` is the one that matters in practice — the
  orchestrator deploys `TOP_K=8`, so it gates at the depth actually traded.
  `INVESTOR_GATE_FORCE=1` is a deliberately awkward escape hatch.
- **The benchmark gate (added 2026-08-03) is the one that bites.** Every
  threshold above measures lift over *random*, which cannot tell "better than
  guessing" from "better than doing nothing" — on 2026-07-03 a model passed
  all of them while losing to an equal-weighted basket of the same universe.
  The benchmark gate simulates the deployed basket (`TOP_K=8`, `SECTOR_CAP=2`)
  against equal-weighting the whole universe, and requires the monthly excess
  to clear `INVESTOR_GATE_BENCH_BPS` (default 25) at **every** fold alignment
  in `INVESTOR_GATE_BENCH_ALIGNMENTS` (default `0,7,14,21,28`). Both the
  non-zero floor and the multi-alignment requirement are deliberate: a single
  measurement of this quantity moves ~50 bps between reasonable
  implementations, so anything less gates on noise. If prices cannot be
  loaded, the gate **fails closed**.

- **Reads:** `data/processed/v4_training_features.parquet`, plus
  `data/raw/v4_investor_data.parquet` (closing prices — the training frame
  carries none, since prices leak the forward return).
- **Writes:** `models/v4_investor_lgbm.txt` and its `.metadata.json` sidecar
  (now including a `benchmark` block with the per-alignment spread).

### `portfolio_orchestrator.py`
The only part that places real orders. Monthly cron
(`30 16 1 * *`): refresh → features → rank → move the account to the target
basket.

- **`TOP_K = 8`** at **equal 12.5% weight** — a deliberate refusal to bet more
  on the top pick, since the ranker's confidence ordering hasn't proven
  reliable enough to size on.
- **`SECTOR_CAP = 2`** — at most two holdings per sector, so the basket can't
  quietly become an all-in bet on one industry.
- **`REBALANCE_DEADBAND = 0.005`** — leave a holding alone if it's within 0.5%
  of target, rather than paying costs monthly to correct trivial drift.
- **`EQUITY_BUFFER = 0.99`** — 1% headroom so rounding and price drift between
  sizing and filling can't overdraw the account.
- Steps 1–2 run as **subprocesses**, so a crash in data refresh can't leave
  this process half-updated mid-rebalance.

- **Reads:** the inference parquet + `models/v4_investor_lgbm.txt`.
  **Writes:** broker orders.

---

## 2. Diagnostics & calibration

### `probe_model.py` — "the bot hasn't traded in days, is it broken?"
Answers **without retraining or touching any weights**, and distinguishes the
two possible causes: **DRIFT** (inputs have moved somewhere the model was never
trained) vs **HONEST** (inputs look normal; the setups genuinely aren't there).
DRIFT suggests retraining may help; HONEST means retraining would chase noise.

Two diagnostics: **PSI** per feature per instrument, and **TreeSHAP**, which
turns "the model is quiet" into "the model is quiet *because* momentum features
are suppressing it".

> ⚠️ Don't read raw PSI against the textbook 0.10/0.25 cutoffs. Those assume
> independent samples; market bars are autocorrelated, so ordinary quiet data
> scores high. The verdict compares against the **null calibration** stored in
> `feature_stats.json`. See [`src/ml/feature_stats.py`](../src/ml/).

- **Imports from repo:** `ml.feature_stats`, `ml.feature_pipeline`,
  `ml.features.v3_features`, `data.oanda_provider`.
- **Reads:** a model dir (pickles + `feature_stats.json` + metadata) and live
  history via OANDA REST. **Writes:** nothing.

### `bake_spread_alphas.py`
Turns observation into configuration: parses `SPREAD_CALIB` lines out of a soak
log and writes a per-instrument trading-cost table, replacing the single
placeholder assumption (0.15) with measured reality.

`alpha_emp` is dimensionless — 0.07 means the toll is 7% of a typical move
(cheap), 0.93 means 93% (effectively untradeable).

> ⚠️ `--denomination-minutes` is load-bearing: a typical move depends on bar
> size, so alphas measured on 15-minute bars are **not valid** for 1-minute
> bars. It's recorded in the output and the retrainer warns loudly on mismatch.

- **Imports from repo:** `execution.oanda_scalper_orchestrator`.
- **Reads:** a soak log. **Writes:** a spread-table JSON (e.g.
  `config/spread_alphas_m15.json`), consumed via `RETRAIN_SPREAD_TABLE`.

### `generate_feature_stats.py`
Backfills the `feature_stats.json` sidecar for models trained before
2026-07-07, so older models can still be probed for drift.

Everything depends on **faithfully reconstructing the original training frame**,
which is why it reuses the retrainer's own fetch, pipeline, veto and cleaning
functions. The **end-date argument must be the model's training date** (from its
`metadata.json`), not today — otherwise the reference describes a window the
model never saw, and manufactures drift that was never there.

- **Imports from repo:** `core.retrainer`, `ml.feature_stats`, `data.factory`,
  `execution.risk_manager`.
- **Writes:** `feature_stats.json` into the given model dir.

---

## 3. Factory launchers

### `run_paper_live.py`
Paper-money launcher for `FactoryOrchestrator`. Near-duplicate of
`run_factory.py` at the repo root; the paper-only guarantee is **procedural**
(use paper keys in `.env`), not enforced in code.
- **Imports from repo:** `execution.factory_orchestrator`,
  `execution.risk_manager`, `data.feed`, `strategies.concrete_strategies`.

### `smoke_test.py`
Three hand-run sanity checks: construction against paper keys, the chop
filter's refusal path, and the $50 minimum-notional floor.

Its **"DO NOT COMMIT" header is stale** — the file is committed, and equivalent
assertions now live in `tests/test_risk_manager.py`.
- **Imports from repo:** `execution.factory_orchestrator`,
  `execution.risk_manager`, `data.feed`.
