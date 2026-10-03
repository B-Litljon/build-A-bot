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
4. **Experiment harnesses** — pinned, single-variable retrain runs you can
   compare one against another (`retrain_arm.sh`).

See the root [GLOSSARY.md](../GLOSSARY.md) for domain terms.

---

## 1. The V4 Investor — monthly stock ranker

A separate product from the forex bot, sharing almost nothing but the repo.
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
  against equal-weighting the whole universe. **`INVESTOR_GATE_BENCH_T`
  (default 2.0) is the constraint that binds** — the mean monthly excess has a
  standard error of roughly 60 bps on ~30 months, so a point estimate on its
  own means little. The current model illustrates it: +47.3 bps looks
  substantial and is t = 0.75. `INVESTOR_GATE_BENCH_BPS` (default 25) is a
  secondary sanity floor, checked at every alignment in
  `INVESTOR_GATE_BENCH_ALIGNMENTS` (default `0,7,14,21,28`) to catch results
  that hang on one lucky set of fold boundaries — but note the alignments'
  excess series correlate ~0.71, so five are worth about **2.2 independent
  samples**, not five. If prices cannot be loaded, the gate **fails closed**.
  If nothing ever clears this bar, that is a finding, not a broken gate: the
  fallback is equal-weighting.

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

### `angel_bar_frontier.py` — "could this model pass its own gate at ANY bar?"

`validate_candidate` says *why* a run failed; it does not say whether any setting
could have succeeded. Those are different questions, and the second one decides
between "keep tuning" and "this model has no certifiable edge".

The gate wants two things at once — enough pooled trades (a chop-scaled backstop,
23 on the reference basket) **and** a pooled PF 95% lower bound ≥ 1.2 — and the
retrainer's bar is chosen to MAXIMISE EV over a quantile grid of the OOF scores,
which parks it at the thin top of the distribution. This script replaces that one
choice with a fixed population quantile ("approve the top X%"), which dissolves the
trade-count problem, and then measures what happens to the evidence.

Reference run (cached M15 basket, 226,992 engineered rows): **0 of 6 points satisfy
both criteria**; win rate never exceeds 0.273 against a 0.333 break-even, the PF
lower bound never exceeds **0.7271** against 1.2, and the binding rejection becomes
`EV < 0.0005` at every point. VERDICT: **unreachable** — and that is a statement
about the model's evidence, not its configuration. The tool prints that sentence
itself, because the tempting response to an unreachable gate is to loosen a
threshold, which manufactures a pass without an edge.

Exit 0 when some bar satisfies both criteria, 2 when none does (the retrainer's
"trained but rejected" convention). Cache-only on purpose — deterministic, no
network — so it needs `src/analysis/build_strategy_matrix.py` to have populated
`analysis_cache/strategy_matrix/`.

> ⚠️ It prints the HTF pairing it used, and you should read that line. `M<gran>`
> alone does not determine it: `get_asset_config`'s default assumes **M1** (`"5m"`),
> so an M15 caller that reuses `cfg` for feature engineering silently trains on the
> wrong higher-timeframe features. The mapping here mirrors
> `run_oanda.py`'s `_GRANULARITY_PROFILES`.

- **Imports from repo:** `core.retrainer`, `execution.risk_manager`.
- **Reads:** `analysis_cache/strategy_matrix/<SYM>_M<gran>.parquet`.
- **Writes:** nothing (it monkeypatches one function in-process and restores it).

### `reprice_band_geometry.py` — "the near-top band on a wide bracket?"

Prices the one combination the 2026-09-14 decision view left standing: the
model's score has a small edge in the top decile and none at the extreme top
(where the live bar sits), and a wide bracket cuts the spread toll ~6x. This
re-walks every row of `logs/graded_decisions.parquet` under four geometries
(live 2.0/4.0/45, wide 10.25/2.74/45, wide with a longer hold, and the
60-sweep's best cell) and reports win rate, gross R, spread toll and net R per
score band.

Reference run (18,625 fiat decisions, 2026-07-31 → 09-18): **no band,
quintile, top-decile or certified population is positive at any geometry.**
Widening improves the certified population ~5x (−0.52 → −0.11R per trade) and
still loses, and the top decile is *worse* than the average row at the wide
geometry — the near-top-band edge does not transfer. There is no reason left
to serve wide static brackets or to expect a wide-label retrain to pass.
Report: [`llm_reports/recons/2026-09-20_reprice-wide-geometry-band-analysis.md`](../llm_reports/recons/2026-09-20_reprice-wide-geometry-band-analysis.md).

> ⚠️ It validates itself: the static arm must reproduce the ledger's own `won`
> column (99.94% on the reference run) or the script exits 2 without printing
> numbers. If you change the walk convention, fix the validation first.

- **Imports from repo:** `data.oanda_provider`.
- **Reads:** `logs/graded_decisions.parquet`, OANDA M15 bars (cached under
  `analysis_cache/2026-09-20_reprice_band_geometry/`).
- **Writes:** `reprice_trades.parquet` in the same cache dir; stdout tables.

### `bake_spread_alphas.py`
Turns observation into configuration: parses `SPREAD_CALIB` lines out of a soak
log and writes a per-instrument trading-cost table, replacing the single
placeholder assumption (0.15) with measured reality.

`alpha_emp` is dimensionless — 0.07 means the toll is 7% of a typical move
(cheap), 0.93 means 93% (effectively untradeable).

> ⚠️ `--denomination-minutes` is load-bearing: a typical move depends on bar
> size, so alphas measured on 15-minute bars are **not valid** for 1-minute
> bars. It's recorded in the output and the retrainer warns loudly on mismatch.

- **Imports from repo:** `execution.oanda_forex_orchestrator`.
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

### `run_catboost_ab.py`
Stage-2 A/B: retrains the Angel/Devil pair twice — once per `MODEL_FAMILY`
(`lightgbm` incumbent, `catboost` candidate) — on an identical holdout-carved,
per-symbol-labelled frame, so the estimator family is the only variable that
changes. Nothing is written under `models/`; a promoted result would need a
separate, deliberate step. Exit 0 = CatBoost strictly beat the incumbent on
every promotion-bar metric; exit 2 = incumbent stands.

- **Imports from repo:** `core.retrainer` (`make_classifier`, `MODEL_FAMILY`,
  `validate_candidate`), `data.factory`.
- **Reads:** bars via the provider (cache-backed). **Writes:** two CSVs
  (`logs/ab_result_<family>.txt`) and two OOS ledger parquets
  (`logs/ab_ledger_<family>.parquet`).

---

## 3. Factory launchers — DELETED 2026-09-16

`run_paper_live.py` and `smoke_test.py` went away with the dormant
Alpaca/Factory lane (`factory_orchestrator.py`, `run_factory.py`,
`run_live.py`); git history has them.

### `diagnose_5yr_holdout.py`
Diagnostic, not a gate: trains the would-be 5-year artifact on the
post-holdout remainder and scores it on the untouched holdout so a rejected
config's overfitting gap can be quantified. Saves nothing, promotes nothing.
Scores through `_score_artifact_holdout` (tail purge included) and prints the
stable-gate verdict (confidence-bound PF) alongside the point estimates.
Written for the 2026-08-24 artifact-holdout report
(`llm_reports/refactors/2026-08-24_artifact-holdout-gate.md`).

- **Imports from repo:** `core.retrainer`, `data.factory`.
- **Reads:** bars via OANDA REST (pinned window env vars). **Writes:** stdout
  only.

### `run_stability_batch.sh`
The holdout-stability demonstration batch: six full retrainer runs — the
2-year and 5-year configs, each pinned to `RETRAIN_END_DATE` 2026-08-09,
2026-08-16, and 2026-08-23 — sequentially, one run at a time, each solo on
the machine at LightGBM default threads. Sequential is deliberate: this box
has 6 physical cores, and any two-way overlap collapses fit speed ~100× once
both jobs are mid-refit (measured 2026-08-24), so one-at-a-time is both
faster end to end and keeps every pin measured under identical conditions.
Verdicts land in `logs/stability_<cfg>_<pin>.log`; nothing touches
`models/forex_m15_wide`. The script exists so the m2m brief's "demonstrate
across three or more pinned endpoints" requirement is reproducible, not a
one-off.

- **Imports from repo:** none (shell).
- **Reads:** `.env`, OANDA REST. **Writes:** `logs/stability_*.log`, side
  model dirs `models/forex_m15_stability_*` (only on promotion).

### `retrain_arm.sh`
Runs ONE retrain experiment arm with every drifting setting pinned, so arms
differ by exactly one merged lane and nothing else. `baseline` is untouched
`main`; every other arm is `main` plus one lane branch, run in that tree. Each
arm writes its own log (`logs/arm-<arm>.log`) plus its own summary and
provenance (`models/candidates/<arm>/`), and no arm can touch
`models/forex_m15_wide`. Exit 0 = promoted **into the candidate dir**, 2 =
rejected, which is a *result* rather than a failure — and note a rejected arm
writes no model at all, so the numbers in the summary are the evidence.

The pin list is load-bearing, not ceremony: 730 days,
`RETRAIN_END_DATE=2026-10-01` (without a pinned end date the retrain reads "up
to now", so two arms run an hour apart see different data and stop being
comparable — same reasoning as `run_stability_batch.sh` above), 18% holdout,
`DATA_SOURCE=oanda`, and **`RETRAIN_TIMEFRAME_MINUTES=15` +
`RETRAIN_HTF_TIMEFRAME=1h`**. `--end=YYYY-MM-DD` and `--days=NNNN` re-pin a single
run, which is how an arm gets checked across several windows rather than trusted
from one — and a longer window is the only way to buy more trades at the gate,
which is where the binding constraint now sits:
the same unchanged code scored a pooled fold bound of 0.61 and 1.04 on two
different 730-day windows (2026-09-09 vs 2026-10-02), so a single window is not
a measurement of an arm. That last pair is the trap: `get_asset_config()`
defaults `timeframe = 1` for *both* asset classes, so a bare `python -m
src.core.retrainer` silently retrains a one-minute model — 15× the fetch, and
nothing like the served M15 artifact (measured 2026-10-02, after a six-minute
aborted run; the comment above `_HTF_FOR_TIMEFRAME` names the same trap for the
HTF pairing). A preflight resolves the real config and refuses to fetch unless
it comes out forex / 15min / 1h / 730 days / 0.18.

It also refuses a `--spread-table` arm in a tree whose
`core/retrainer/_labels.py` has no `alpha_table` (the flag would then move
`cost_ratio` and the veto but not the labels), refuses a candidate dir that
already exists (`save_models` merges into an existing `metadata.json`, so a
reused dir would carry stale keys from an earlier arm), and unsets
`DISCORD_WEBHOOK_URL` so N arms do not post N embed reports.

**Run arms one at a time, never two at once** — the same 6-core reason as
`run_stability_batch.sh`: overlapping retrainer runs collapse fit speed.

- **Imports from repo:** none (shell); invokes `python -m src.core.retrainer`.
- **Reads:** `.env`, OANDA REST. **Writes:** `logs/arm-<arm>.log`,
  `models/candidates/<arm>/` (provenance + summary; a model only on promotion).
