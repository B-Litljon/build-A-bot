# `src/lab` — the feature lab

An **offline-only** package for trying candidate features against the model's
own ruler. It composes three existing pieces — the `ml` feature generators,
`core.retrainer`'s labels and promotion gate, and
`analysis.strategy_backtester` — and adds no trading code. Nothing in
`src/execution/` or `run_oanda.py` imports it, so a broken lab cannot reach the
live bot.

The question the lab answers is not "does this feature set look good on a
backtest?" — it is **"would this feature set have promoted?"** The verdict
metric is `validate_candidate`, the exact function a production retrain calls,
with `edge_over_random` as the primary number and a live-gated backtest as the
cost-side check. Lab numbers are directly comparable to the served model's.

Built 2026-09-21 against `llm_reports/handoffs/2026-09-21_feature-lab-plan.md`;
v2 (frame-hash covers generator state, gate-list threading, `ablate`, the
estimator A/B) built 2026-09-22/23 against
`llm_reports/handoffs/2026-09-22_feature-lab-v2-build-tasks.md` on branch
`lab/w1-w4`.

## The loop

```bash
# from the repo root
PYTHONPATH=src:. python -m lab.cli list
PYTHONPATH=src:. python -m lab.cli run --name v3_base_control
PYTHONPATH=src:. python -m lab.cli run --spec specs/my_feature.py
PYTHONPATH=src:. python -m lab.cli replay --name v3_base_control
PYTHONPATH=src:. python -m lab.cli ablate --name v3_base_control
```

`run` loads bars (cached parquet basket), builds or loads the frame, runs the
fold gate, replays the models through the live-gated backtester, prints a JSON
summary, and writes a recon report to `llm_reports/recons/`. Exit codes mirror
the retrainer: **0** gate passed, **2** gate rejected, **1** error.

`replay` asks the other baseline question — *what would the SERVED artifact have
done on this frame?* It loads `--model-dir` (default: `OANDA_MODEL_DIR`, else
`models/forex_m15_wide`), pins the artifact's own `threshold.json` bars, prepares
the same content-hashed frame, and writes a report with the raw pre-gate
population, a split at the artifact's recorded holdout window, and the live-gated
replay. No gate, no retrain, no PASS/FAIL; exit 0 on a completed replay.

`ablate` asks the interaction question — *what does each registered family add
in situ?* It expands the spec into N+1 arms (the full cocktail plus one per
family with that family removed), runs the real gate on each with identical
labels/veto/geometry/data/cost-table (only `feature_sets` varies), and reports
per family the delta `edge(full) − edge(full − X)` WITH a Clopper-Pearson
interval — never a bare point. Needs ≥ 2 registered families (a single-family
spec has no cocktail to subtract against; use `run`). Writes
`llm_reports/recons/<date>_lab-ablate-<spec>.md`.

Estimator A/B (the architecture seam): `MODEL_FAMILY=catboost python -m lab.cli
run --name <seed>` runs the seed spec under CatBoost. The env-selected family is
the run's arm (precedence: explicit `--model-family` flag > env > the spec's
declared family — measured and closed 2026-09-23, see
`llm_reports/recons/2026-09-23_lab-model-family-ab.md`). The frame hash excludes
the estimator, so both arms share one cached frame by construction.

Adding a candidate feature is one class plus a registration — **which now
REQUIRES a version** (folded into the frame-cache key; bump it whenever you edit
anything inside the generator that changes the frame):

```python
# src/lab/features.py or a spec file
from lab.registry import register_feature
from ml.core.interfaces import BaseFeatureGenerator

class MyFeatures(BaseFeatureGenerator):
    feature_cols = ("my_feature",)

    def generate(self, df):                       # per symbol, causal
        return df.with_columns(...)               # .over("symbol")

register_feature("my_family", version=1, description="...")(MyFeatures)
```

then `FeatureSpec(feature_sets=("v3_base", "my_family"))`. No pipeline,
retrainer, or live-code edit is needed.

## Files

### `spec.py`
`FeatureSpec` — one frozen, hashable object describing the whole experiment:
data window, feature families, bracket geometry, label knobs, cost-table switch,
estimator family. `content_hash()` is the frame-cache key; it includes the
spread-table bytes, the frame-affecting environment (veto switches, risk
profile values, behavior veto), every registered family's `(name, version)`
pair, and the resolved state of any extra generator (constructor args
included; non-serializable state raises rather than degrading to a
class-name-only hash). It deliberately EXCLUDES `gate.model_family`/`n_folds` —
run provenance, not frame content — which is what lets the estimator A/B reuse
one cached frame.

- **Imports from repo:** `ml.core.interfaces`; the registry (lazily, for
  family versions).
- **Imported by:** `registry`, `frames`, `data`, `experiments`, `specs`, tests.
- **Reads/writes:** reads the spread table's bytes when hashing; writes nothing.

### `registry.py`
The feature-family registry. `register_family` / `register_feature` bind a name
to generators + the model-facing columns + a REQUIRED `version: int` (>= 1)
that feeds the frame-cache key — bump it when the family's generators change
frame contents. `family_version(name)` resolves one family's version (raising
on unknown names, the same loud contract as the expanders); `get_generators`
and `feature_columns` expand a spec. Built-in `v3_base` is the production V3
stack (Base + HTF + Session + Cost) and its columns are exactly the retrainer's
`BASE_FEATURE_COLS` (+ `cost_ratio` when the table is active). Unknown family
names raise — never a silent skip.

- **Imports from repo:** `ml.core.interfaces`; the built-in builders import
  `execution.risk_manager` and `ml.features.v3_features` lazily.
- **Imported by:** `frames`, `cli`, `specs`, `__init__`, tests.
- **Reads/writes:** nothing.

### `features.py`
The seed candidate family, `LabMicrostructureFeatures` (`ms_close_pos_20`,
`ms_ret_z_10`, `ms_updown_vol_10`, `ms_autocorr_20`), and the template for a new
one. Needs `log_return` from `v3_base`, so it must be ordered after it.

- **Imports from repo:** `ml.core.interfaces`, `lab.registry`.
- **Imported by:** `specs` (registration), user spec files.
- **Reads/writes:** nothing.

### `data.py`
`load_bars` wraps `analysis.build_strategy_matrix.load_basket` (cached parquet
first, OANDA fallback, common-window trim). `resolve_alpha_table` returns the
spec's cost table. `DEFAULT_CACHE_DIR` is `analysis_cache/strategy_matrix/`.

- **Imports from repo:** `analysis.build_strategy_matrix` (lazily).
- **Imported by:** `frames`, `experiments`, `__init__`.
- **Reads/writes:** reads and writes the bar cache; `refresh=True` deletes the
  spec's cached files first.

### `frames.py`
`build_frame` — generators, then `core.retrainer`'s `apply_labels_and_veto` (the
2026-09-21 extraction), then the Phase-3a unresolvable-tail purge. Returns
`FrameResult` (frame, feature columns, chop-veto rate, purge count, resolved
alpha table, content hash). The parity contract lives here: for
`feature_sets=("v3_base",)` the result equals `engineer_features_and_labels`
plus the production tail purge, row for row.

- **Imports from repo:** `core.retrainer._features`, `core.retrainer._gate`
  (purge helpers), `execution.risk_manager` — all lazily.
- **Imported by:** `experiments`, `__init__`, tests.
- **Reads/writes:** nothing.

### `gate.py`
`run_gate` calls `validate_candidate` with the spec's geometry and the frame's
veto rate, scoping `RETRAIN_DEVIL_LABEL` to the spec's label kind for the
duration. `require_model_family` refuses a spec whose estimator differs from
the loaded `MODEL_FAMILY` (read at retrainer import time) — except on the CLI
run/ablate path, where an env-selected family IS the run's arm
(`allow_env_override=True`, the W4 estimator A/B; still loud, and the bare
spec-contract form still refuses). `GateResult` carries the report, models,
feature lists, and frozen thresholds.

- **Imports from repo:** `core.retrainer._gate` / `_common` / `_types`.
- **Imported by:** `experiments`, `ablate`, `__init__`, tests.
- **Reads/writes:** nothing.

### `backtest.py`
`LabModelStrategy` replays the gate-trained models as a `BaseStrategy` and
`run_model_backtest` walks it symbol by symbol through `run_backtest` with a
live `RiskManager`, so Gates A/B/C veto entries and the per-instrument toll
applies. It threads the gate's OWN feature lists (`GateResult.angel_features`
/ `devil_features`) into the replay — never a re-derivation from the frame's
columns, which silently diverge once HMM features append their columns to the
trained schema (`core/retrainer/_gate.py:606`).
`run_artifact_backtest` is the same replay for a served artifact at its own
pinned bars, and both share one `_replay_models` body so they cannot drift.
`raw_sl_distance` is raw ATR (`close * natr_14 / 100`) — the RiskManager owns
the multipliers.

- **Imports from repo:** `analysis.strategy_backtester`,
  `analysis.behavior_matrix`, `strategies.base`.
- **Imported by:** `experiments`, `artifact`, `__init__`, tests.
- **Reads/writes:** nothing.

### `experiments.py`
`ExperimentRunner` — the loop plus the frame cache at
`analysis_cache/lab_frames/<content-hash>.parquet` + a JSON sidecar. A hit
requires both files and a matching hash; writes are atomic. `prepare_frame` is
the frame half on its own, shared with the artifact replay. `ExperimentResult`
and its `summary()` are what the CLI prints.

- **Imports from repo:** `lab.*`.
- **Imported by:** `cli`, `__init__`.
- **Reads/writes:** reads/writes the frame cache.

### `artifact.py`
`load_served_artifact` reads the served model directory (`angel_latest.pkl`,
`devil_latest.pkl`, `threshold.json`, `metadata.json`) and `ServedArtifact`
carries the pair, the pinned bars, their source, and the fit-time feature order
(`feature_names_in_`, CatBoost's `feature_names_`). `replay_served_artifact`
prepares the spec's frame, validates the artifact's schema against it,
`predict_probabilities` scores the whole frame, and `run_artifact_backtest`
(lab.backtest) replays it live-gated. `ArtifactReplayResult.summary()` adds the
raw population and a split at the recorded holdout window.

- **Imports from repo:** `core.thresholds` (lazily), `lab.backtest`,
  `lab.experiments`.
- **Imported by:** `cli`, `__init__`.
- **Reads/writes:** reads the model dir and the frame cache; writes nothing.

### `report.py`
Renders a run into `llm_reports/recons/<date>_lab-<slug>.md` (frontmatter per
`llm_reports/README.md`) with a Caveats section that flags a failed gate, thin
trade counts, and a disabled cost table. `render_artifact_report` /
`write_artifact_report` are the replay variants (slug defaults to
`served-artifact-<spec>` so a replay can never overwrite a gate report), with an
honesty box that states the replay is not a promotion gate and is largely
in-sample for the served artifact. `render_ablation_report` /
`write_ablation_report` (slug `ablate-<spec>`) render the ablation table — one
row per variant with the delta AND its Clopper-Pearson interval — plus the
thin-trades caveat and the reading rule: a delta consistent with zero on thin
trades is not a drop decision.

- **Imports from repo:** none.
- **Imported by:** `cli`, `__init__`.
- **Reads/writes:** writes the report file.

### `ablate.py`
`ablate` expands a spec into N+1 arms (the full cocktail plus one per family
with that family removed — labels, veto, geometry, data and cost table carried
over UNCHANGED, so the delta reads as a family effect) and runs the real gate
on each, sharing one bar load and the frame cache. `delta_with_ci` computes
edge(full) − edge(full − X) with a Clopper-Pearson interval on the underlying
win-rate difference (clipped to the feasible win-rate range); `THIN_TRADES`
arms the caveat that fires whenever an arm's pooled trades fall below it —
a delta consistent with zero on thin trades is never a drop decision.
`AblationResult` and its `summary()` are what the CLI prints.

- **Imports from repo:** `lab.gate` (lazily), `core.retrainer._common` via
  scipy lazily for the beta quantile.
- **Imported by:** `cli`, `report`, tests.
- **Reads/writes:** writes nothing (the report writer does).

### `specs.py`
The three seed experiments: `v3_base_control` (calibration), 
`spread_table_control` (the configured-but-unused cost asymmetry), and
`microstructure` (the candidate family, cost table off so the feature set is the
only changed variable).

### `stats.py`
Shared multiple-comparison statistics for the quant research lanes (Lane 5,
2026-09-24): `deflated_sharpe_ratio` (Bailey & López de Prado 2014, a
probability), `cscv_pbo` (Bailey et al. 2014, S=8 blocks / 70 combos),
`hlz_haircut_sharpe` (Harvey–Liu–Zhu 2016, adjusted Sharpe), and
`clopper_pearson_lower` (one-sided CP binomial bound). Built against the
signatures pinned in the lane briefs; before this file there was NO
DSR/CSCV/HLZ code anywhere in the repo.

- **Imports from repo:** nothing (numpy, scipy only).
- **Imported by:** `altcoin_topquint`, `vwap_reversion`; the other 2026-09-24
  lanes were told to build the same file if absent.
- **Reads/writes:** nothing.

### `altcoin_topquint.py`
Audit A — weekly LightGBM `lambdarank` top-quintile cross-sectional momentum
over a liquidity-floored Alpaca altcoin basket, with the dual equal-weight +
BTC-hold benchmark, a 31.6 bps/side friction, and the DSR/PBO/CP gate.
Research-only; the realized run is falsified at the data layer (the $250k
floor admits < 5 assets on Alpaca's free-tier crypto volume).

- **Imports from repo:** `lab.stats`. Data from Alpaca's
  `CryptoHistoricalDataClient` (fetched by the run script, not imported here).
- **Imported by:** `tests/test_lab_altcoin.py`.
- **Reads/writes:** nothing; the loader/walk-forward engine consumes
  caller-supplied daily-close arrays.

### `fix_audit.py`
Audit B — DST-correct London WM/R (16:00 Europe/London) and Tokyo Nakane
(9:55 JST) fix identification (zoneinfo), the Gotobi-day calendar, and the
coarse M15 fix-bar study across the six fiat M15 caches. Carries the
first-class data-resolution limitation: no tick/bid-ask data exists anywhere,
so a fix bar indistinguishable at M15 is NOT a falsification of tick-level
widening.

- **Imports from repo:** nothing (stdlib zoneinfo + numpy).
- **Imported by:** `fix_collector`'s window helper, `tests/test_lab_fix_audit.py`.
- **Reads/writes:** nothing; consumes caller-supplied M15 frames.

### `fix_collector.py`
The atomic tick-parquet writer for Audit B's collector —
`FixTickBuffer` + `write_ticks_atomic` (temp file + fsync + `os.replace`,
the same convention as `core/events.py` and `lab/frames.py`). The six-column
schema (`timestamp_utc, symbol, bid, ask, mid, source_latency_ms`) is the
contract `scripts/fix_tick_collector.py` writes.

- **Imports from repo:** nothing (pyarrow only at write time).
- **Imported by:** `scripts/fix_tick_collector.py`, `tests/test_lab_fix_audit.py`.
- **Reads/writes:** writes `data/ticks/fix_ticks_YYYY-MM-DD.parquet`
  atomically; nothing reads it yet (the collector is uninstalled and unrun).

### `vwap_reversion.py`
Audit C — session-scale VWAP reversion at the 08:00–09:00 UTC London open,
k·σ fade entry, inventory-imbalance / re-extension-stop / hard-session-close
exits, 1% NAV risk sizing, k swept over {1.5, 2.0, 2.5, 3.0} with the
positive-expectancy-independent-of-k gate. Research-only; measured net EV is
−0.047R to −0.140R per trade — falsified independent of k.

- **Imports from repo:** `lab.stats`.
- **Imported by:** `tests/test_lab_vwap_reversion.py`.
- **Reads/writes:** nothing; consumes caller-supplied M15 frames.

### `cli.py`
`list`, `run`, `replay` and `ablate`; loads user spec files and sets
`MODEL_FAMILY` from the flag, the environment, or the spec — in that precedence
— before the retrainer is imported (the replay path never touches the
retrainer). The env beating the spec is the W4 estimator-A/B seam; an explicit
`--model-family` flag must still agree with the spec.

- **Imports from repo:** none at module scope.
- **Reads/writes:** writes reports by default.

## What v1 did NOT do (v2 status, built 2026-09-22/23 on `lab/w1-w4`)

- **Frame-cache hash now covers generator state (W2 fix in the review's
  numbering, shipped as W1).** Editing a lookback inside a registered
  generator requires bumping its registration version; extra generators hash
  their resolved `__dict__`. The prior hole — a class-name-only hash — is
  pinned shut by tests (`tests/test_lab_spec.py`).
- **The backtest consumes the gate's OWN feature lists (W2).** No
  re-derivation from `frame.feature_cols`, which would silently diverge under
  HMM features (`core/retrainer/_gate.py:606`).
- **`lab ablate` (W3) is built** — the interaction question, with CIs.
- **The estimator A/B (W4) ran and closed**: CatBoost scored worse than random
  on the same frame (`llm_reports/recons/2026-09-23_lab-model-family-ab.md`);
  the architecture axis stays closed at the estimator-swap level.
- **Still not done (deliberately):** no candidate artifact holdout —
  `_score_artifact_holdout` engineers its slice with the production feature
  list, so it cannot score a *candidate* feature set without generalizing that
  function. The fold gate's `edge_over_random` is the verdict for candidates;
  fold 3's validation window is the recent-regime check. (The SERVED artifact
  *is* replayed — `lab.artifact` — because its schema is known; that is a
  baseline, not a promotion test.)
- **No angel-bar frontier** (`gate.run_frontier` in the plan). The standalone
  `scripts/angel_bar_frontier.py` already answers that question on the cached
  basket; porting it needs per-fold OOF probabilities, which
  `validate_candidate` does not expose.
- **No new-label axis.** Targets are fixed; `label.kind` selects between the two
  existing ones. The plan's recommendation was to keep targets fixed in v1
  because the label vocabulary is where silent leakage lives.

## Caveats that apply to every run

- A gate **FAIL** means the replayed models are the Fold-3 placeholders, not a
  promoted artifact; the backtest is indicative.
- A served-artifact **replay** has no verdict at all, and its frame is largely
  in-sample for the artifact — only the recorded-holdout rows in its window
  table are unseen. It is a baseline, not evidence of skill.
- `edge_over_random` is in **win-rate units**, not R. At the EV-maximising Angel
  bar the gate approves very few trades (the control: 30 pooled), so a large
  positive edge there is small-sample noise — the same +0.179-on-30-trades
  artifact the 2026-09-14 work withdrew. Read it with
  `pooled_oos_trades` beside it.
- Frame parquet is ~25–35 MB per cached spec; `analysis_cache/` is gitignored
  and files can be deleted freely (they rebuild).

See the root [GLOSSARY.md](../../GLOSSARY.md) for domain terms (angel/devil,
bracket, NATR, chop veto, OOS, edge over random).
