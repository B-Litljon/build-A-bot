---
type: refactor
date: 2026-09-16
time: 20:45 PDT
agent: DeepSeek Harness (dsh, web)
model: deepseek-v4-pro (GLM-5.3-flash session)
trigger: "User: the repo has several files that are thousands of lines long — can the code itself be downsized without deleting documentation?"
head: 1f63781
scope: deletes-source + splits-source (behavior-preserving)
files_touched:
  - DELETED: src/day_trading/ (whole package)
  - DELETED: src/execution/live_orchestrator.py (2,663 lines)
  - DELETED: src/execution/factory_orchestrator.py
  - DELETED: run_live.py, run_factory.py, chop_ab_test.py, run_chop_ab.sh
  - DELETED: scripts/run_paper_live.py, scripts/smoke_test.py
  - DELETED: tests/execution/test_live_orchestrator.py, tests/verify_warmup.py
  - DELETED: src/core/signal.py (Alpaca-path Signal; last consumer was send_trade_alert)
  - DELETED: src/data/fetch_training_data.py (dead module, same-named live function survives elsewhere)
  - REMOVED: NotificationManager.send_trade_alert (dead after lane deletion)
  - SPLIT: src/core/retrainer.py (4,366 lines) -> src/core/retrainer/ package
  - src/execution/__init__.py, src/core/notification_manager.py, src/strategies/base.py
  - tests: holdout_gate, gate_tradeable_scoring, dynamic_thresholds, devil_label_switch imports, ml_strategy_guards, oos_ledger_capture
  - scripts/test_sidecar_retrain.py, scripts/investor_train_model.py, src/analysis/optimize_brackets.py
  - CLAUDE.md, GLOSSARY.md, src/README.md, src/core/README.md, src/execution/README.md,
    src/data/README.md, src/ml/README.md, src/ml/core/README.md, src/strategies/README.md,
    scripts/README.md, tests/README.md, tests/__init__.py
---

# Code downsizing — dead lanes out, retrainer monolith split

Second half of the 2026-09-16 downsizing session (part one, the data/log
prune, is `refactors/2026-09-16_repo-downsizing-pass.md`). Suite: **538
passed, 6 subtests passed** (was 550 + 6; the −12 = the deleted
live_orchestrator thread-ownership tests). `compileall` clean. The live soak
(PID 897548, `forex_m15_wide`) ran through all of it untouched; it imports
none of this.

## Lane deletions (~6,000 lines)

Both lanes were documented dormant (GLOSSARY: "never live" / "not live"),
nothing in the soak, investor cron, or retrainer imported them, and git
history preserves them:

- **`src/day_trading/`** — 2,670 lines, zero external imports (verified with
  `rg -ln '\bday_trading\b'` outside the package: only READMEs/GLOSSARY).
  The `dt_*` model artifacts at `models/` root and raw `dt_*` parquets were
  kept (tiny, and the lane's only outputs).
- **Alpaca scalper lane** — `live_orchestrator.py` (2,663 lines),
  `factory_orchestrator.py`, `run_live.py`, `run_factory.py`,
  `chop_ab_test.py` + `run_chop_ab.sh`, `scripts/run_paper_live.py`,
  `scripts/smoke_test.py`, `tests/execution/test_live_orchestrator.py`,
  `tests/verify_warmup.py`. Follow-through: `core/signal.py` (its only
  remaining importers were this lane) and `NotificationManager.send_trade_alert`
  (its only caller) are gone. The monthly V4 investor lane is unaffected —
  it runs `scripts/portfolio_orchestrator.py` + `v4_investor_lgbm.txt`.
- **Deliberately kept** from the pre-split review: `soak`-referenced
  `src/replay_test.py` and `src/evaluate_performance.py` ARE still invoked by
  `run_pipeline.sh` phases — they looked dead from `rg` because they're run as
  `python -m src.replay_test`. Check shell entry points before trusting import
  maps, lesson recorded.

## retrainer.py → src/core/retrainer/ package

4,366 lines, ~40 definitions. New layout (no behavior change; every body is a
verbatim slice except the call-site fixups below):

| module | lines | owns |
|---|---|---|
| `_common.py` | 588 | imports, dual-path bootstrap, all env constants, classifier factory, FEATURE_COLS, spread table |
| `_types.py` | 108 | FoldMetrics / HoldoutMetrics / ValidationReport |
| `_data.py` | 165 | fetch_training_data, _split_holdout |
| `_labels.py` | 303 | Devil targets (ATR bracket sim + survival), chop-veto mask |
| `_features.py` | 342 | engineer_features_and_labels, time-decay weights |
| `_train.py` | 344 | refit_models, devil min-child floor |
| `_thresholds.py` | 177 | optimal threshold / angel-bar search |
| `_gate.py` | 1,308 | validate_candidate + holdout eval + tradeable mask + base-rate telemetry |
| `_persist.py` | 563 | promote_or_reject, barrier sidecars, atomic save_* |
| `_pipeline.py` | 396 | main() |
| `__init__.py` | 445 | facade: original docstring/glossary verbatim + 107-name re-export |
| `__main__.py` | 13 | keeps `python -m src.core.retrainer` working (run_pipeline.sh:186) |

`_gate.py` is still 1.3k lines because `validate_candidate` (823 lines) is one
indivisible scoring procedure; further splitting would break its readability,
not improve it. That and `oanda_forex_orchestrator.py` (2,503, live) are the
only remaining >1k files — the latter was explicitly deferred.

### The mechanical traps, and how they were handled

1. **Monkeypatch targets move.** Tests patch `R.HOLDOUT_FRAC` /
   `R.fetch_training_data` / `R.refit_models` / `R.promote_or_reject` /
   `R.UNTRADEABLE_SYMBOLS`. After the split the readers live in submodules, so
   those names are read *through their owning module* at call time
   (`common.HOLDOUT_FRAC`, `_data.fetch_training_data(...)`, etc.) and the 7
   patch sites (5 in tests, 1 in scripts, 1 multi-block) were retargeted to
   `core.retrainer._common` / `._data` / `._train` / `._persist`. Facade
   re-export alone would have silently made those patches no-ops.
2. **Two tests read the module's source code** as a policy pin
   (`test_gate_tradeable_scoring` checks `save_models` writes
   `"behavior_veto": sorted(BEHAVIOR_VETO_LABELS)`; `test_oos_ledger_capture`
   demands the ledger capture sit under `if oos_ledger is not None:`).
   Retargeted to `_persist.py` / `_gate.py` respectively.
3. **One env-var test reloaded `core.retrainer` to re-read
   `RETRAIN_BEHAVIOR_VETO`** — reloading the facade only re-imports, doesn't
   recompute. The test now reloads `core.retrainer._common`, where the env
   read actually happens.
4. **Tuple-target constants** (`ANGEL_PARAMS, DEVIL_PARAMS = get_hyperparameters(...)`)
   weren't caught by the name-mapping pass; added to the facade explicitly.
5. **`sliding_window_view` was local-imported inside a function** — survived
   the split untouched since fixups only touched module-level call sites.

All cross-module edges were computed with an AST pass first: the dependency
graph is a DAG (`_common` -> everything; `_features` -> `_labels`; `_gate` ->
`_train`, `_thresholds`, `_types`; `_pipeline` -> nearly all). No cycles.

### Facade compatibility proof

- `import core.retrainer as R` and `import src.core.retrainer` — both verified
  to expose the full historical surface.
- `python -m src.core.retrainer` smoke-tested (fails fast and correctly on a
  bogus DATA_SOURCE — exit 1 with the expected "Unknown DATA_SOURCE" error).
- 13 distinct import spellings across tests/scripts (`from core.retrainer
  import X`, `import core.retrainer as R`, attribute access `R.validate_candidate`)
  all exercised by the suite.
