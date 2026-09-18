---
type: refactor
date: 2026-09-16
time: 20:05 PDT
agent: DeepSeek Harness (dsh, web)
model: deepseek-v4-pro (GLM-5.3-flash session)
trigger: "User request: the checkout had grown to ~1.5 GB; shrink it and identify what doesn't need to be there"
head: 1f63781
scope: deletes-data (no source-code behavior changes)
files_touched:
  - run_soak.sh
  - GLOSSARY.md
  - llm_reports/m2m-prompts/README.md
  - llm_reports/m2m-prompts/2026-06-22_data-source-mixing.md (moved from m2m_prompts/)
  - llm_reports/m2m-prompts/2026-06-22_investor-edge-phase-a.md (moved)
  - llm_reports/m2m-prompts/2026-06-22_investor-edge-phase-b.md (moved)
  - llm_reports/m2m-prompts/2026-06-22_investor-gate.md (moved)
  - llm_reports/m2m-prompts/2026-06-26_london-breakout-results-to-gemini.md (moved)
  - llm_reports/m2m-prompts/2026-07-03_investor-live-rebalance-and-cron.md (moved)
  - llm_reports/m2m-prompts/2026-08-10_algorithm-and-strategy-research.md (moved)
  - llm_reports/m2m-prompts/2026-08-17_score-compression-and-calibration.md (moved)
---

# Repo downsizing pass — 1.5 GB → ~400 MB

User-approved four-tier cleanup. Everything deleted was untracked, regenerable,
or moved; no source edits except the two documented below. Tests: **550 passed,
6 subtests passed** after the pass (venv python 3.12, `PYTHONPATH=src:.`).

## What was removed

- `data/_retired_2026-09-09/` (755 MB) — retired training parquet + old forex
  1m cache. Pipeline-regenerable; nothing referenced it outside the
  2026-09-09 refactor report.
- `src/ml/models/*.joblib` (42 MB) — V4-era RandomForest artifacts
  (`rf_model`, `angel_rf_model`, `universal_rf_model`, `devil_rf_model`,
  gitignored). Referenced only by `src/analysis/` offline diagnostics.
- `data/raw/*_1min.parquet` (~57 MB) — the old 1-minute equities bars. The
  V4 investor lane consumes `v4_investor_data.parquet` and the SimFin cache
  (both kept); the M15 soak fetches from OANDA. The dormant day-trading
  `dt_*` parquets were also kept — they are its only inputs.
- 24 experiment dirs under `models/` (~37 MB): the four `forex_m15_stability_*`,
  `forex_m15_{5yr,5yr_v3,2yr_v3,2yr_check,holdout_2yr}`, `forex_m30_5yr`,
  `forex_m45_5yr`, `forex_swing`, all six `newgate_*`, `sweep_mp{150,600}`,
  `_ab_5yr`, `_legacy_equities_2026-05`, `_phantom_lowered_floor_20260829`,
  `test_barrier_sidecar`. Each was grepped for live references first — the
  only two hits were string fixtures in `tests/test_retraining_notification.py`
  (formats the name, never reads disk) and a self-creating temp dir in
  `scripts/test_sidecar_retrain.py`.
  **Kept:** `forex_m15_wide/` (served by the live soak per `soak.service`),
  `forex_m15_wide_backup_20260829/` (rollback), `forex_h4_catboost/`
  (gated-out candidate, kept for audit), root `dt_*` and `v4_investor_lgbm.*`.
- June-vintage soak/paper logs (deleted outright); July–Aug-16 soak logs
  gzipped. `soak_2026-08-16_1405.log` alone was 427 MB → 311 MB gz (dense
  float-text compresses poorly; the run is M1 vintage with no diagnostic
  value left — deleting it outright is a reasonable next step if 311 MB
  matters again).
  The live log `soak_2026-09-15_1624.log` was not touched (verified still
  being written after the pass).
- `m2m_prompts/` top-level folder folded into `llm_reports/m2m-prompts/`
  (`git mv`, history preserved). Its ledger convention (frontmatter +
  `result_commit` two-way link) is preserved as a section in the latter's
  README. `GLOSSARY.md` top-level table updated for both this and the
  `models/` prune.
- Residue: `catboost_info/`, `.ruff_cache/`, `.pytest_cache/`, all
  `__pycache__/` (including the stale `src/autopilot/` and `src/research/`
  trees GLOSSARY.md already flagged as sourceless), `backtest_output.log`,
  an empty `viz/2026-05-23/`, and a stray uv-created `.venv/` pointing at
  system python 3.14 — the interpreter that fails test collection per
  CLAUDE.md.

## What was deliberately kept

- **`analysis_cache/` (9.1 MB)** — looks like residue but is not:
  `scripts/angel_bar_frontier.py:79`, `scripts/evaluate_barriers.py:104`,
  `scripts/run_decision_report.py:77` and `build_strategy_matrix.py:319`
  all read `analysis_cache/strategy_matrix` as their **default input**.
  Deleting it would break the documented re-runnable tooling.
- `data/raw/simfin_cache/` (28 MB), `v4_investor_data.parquet`,
  `data/processed/` — live inputs to the monthly V4 investor cron lane.
- `logs/status.json`, recent `events-*.jsonl` — read live by the dashboard.

## Behavior change (one)

`run_soak.sh` now prunes logs at every launch, before creating this run's
live log: gzips `logs/soak_*.log` older than 30 days, deletes gzips older
than 180 days. The 427 MB single-file blowup (one soak run, one log, no
rotation) cannot recur. The prune runs pre-launch, so the file the new run
is about to write is never eligible. `bash -n` clean; the already-running
soak is unaffected until its next (re)start.

## Soak state after the pass

PID 897548 serving `models/forex_m15_wide`, writing
`logs/soak_2026-09-15_1624.log` — confirmed live (log mtime advancing,
SEAM_BACKFILL lines at 20:00 PDT).
