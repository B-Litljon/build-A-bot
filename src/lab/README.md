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

Built 2026-09-21 against `llm_reports/handoffs/2026-09-21_feature-lab-plan.md`.

## The loop

```bash
# from the repo root
PYTHONPATH=src:. python -m lab.cli list
PYTHONPATH=src:. python -m lab.cli run --name v3_base_control
PYTHONPATH=src:. python -m lab.cli run --spec specs/my_feature.py
PYTHONPATH=src:. python -m lab.cli replay --name v3_base_control
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

Adding a candidate feature is one class plus a registration:

```python
# src/lab/features.py or a spec file
from lab.registry import register_feature
from ml.core.interfaces import BaseFeatureGenerator

class MyFeatures(BaseFeatureGenerator):
    feature_cols = ("my_feature",)

    def generate(self, df):                       # per symbol, causal
        return df.with_columns(...)               # .over("symbol")

register_feature("my_family", description="...")(MyFeatures)
```

then `FeatureSpec(feature_sets=("v3_base", "my_family"))`. No pipeline,
retrainer, or live-code edit is needed.

## Files

### `spec.py`
`FeatureSpec` — one frozen, hashable object describing the whole experiment:
data window, feature families, bracket geometry, label knobs, cost-table switch,
estimator family. `content_hash()` is the frame-cache key; it includes the
spread-table bytes and the frame-affecting environment (veto switches, risk
profile values, behavior veto).

- **Imports from repo:** `ml.core.interfaces`.
- **Imported by:** `registry`, `frames`, `data`, `experiments`, `specs`, tests.
- **Reads/writes:** reads the spread table's bytes when hashing; writes nothing.

### `registry.py`
The feature-family registry. `register_family` / `register_feature` bind a name
to generators + the model-facing columns; `get_generators` and
`feature_columns` expand a spec. Built-in `v3_base` is the production V3 stack
(Base + HTF + Session + Cost) and its columns are exactly the retrainer's
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
duration. `require_model_family` refuses a spec whose estimator differs from the
loaded `MODEL_FAMILY` (it is read at retrainer import time — launch with the env
var set). `GateResult` carries the report, models, feature lists, and frozen
thresholds.

- **Imports from repo:** `core.retrainer._gate` / `_common` / `_types`.
- **Imported by:** `experiments`, `__init__`, tests.
- **Reads/writes:** nothing.

### `backtest.py`
`LabModelStrategy` replays the gate-trained models as a `BaseStrategy` and
`run_model_backtest` walks it symbol by symbol through `run_backtest` with a
live `RiskManager`, so Gates A/B/C veto entries and the per-instrument toll
applies. `run_artifact_backtest` is the same replay for a served artifact at its
own pinned bars, and both share one `_replay_models` body so they cannot drift.
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
in-sample for the served artifact.

- **Imports from repo:** none.
- **Imported by:** `cli`, `__init__`.
- **Reads/writes:** writes the report file.

### `specs.py`
The three seed experiments: `v3_base_control` (calibration), 
`spread_table_control` (the configured-but-unused cost asymmetry), and
`microstructure` (the candidate family, cost table off so the feature set is the
only changed variable).

### `cli.py`
`list`, `run` and `replay`; loads user spec files and sets `MODEL_FAMILY` from
the spec before the retrainer is imported (the replay path never touches the
retrainer).

- **Imports from repo:** none at module scope.
- **Reads/writes:** writes reports by default.

## What v1 does NOT do (deliberately)

- **No candidate artifact holdout.** `_score_artifact_holdout` engineers its
  slice with the production feature list, so it cannot score a *candidate*
  feature set without generalizing that function. The fold gate's
  `edge_over_random` is the verdict for candidates; fold 3's validation window is
  the recent-regime check. (The SERVED artifact *is* replayed — `lab.artifact` —
  because its schema is known; that is a baseline, not a promotion test.)
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
