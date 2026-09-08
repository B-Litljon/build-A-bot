# `llm_reports/m2m-prompts`

Numbered agent briefs for the quant-architecture roadmap (2026-09-08).
One stage per file; point an agent at the file and it has the full contract.

| Stage | File | State |
|---|---|---|
| 0 | — (measurement, no brief needed; done 2026-09-08, see `logs/decision_report_2026-09-08.txt`) | COMPLETE |
| 1 | — (barriers, built directly: `src/ml/barriers/`, `scripts/evaluate_barriers.py`) | COMPLETE (offline pass; live wiring is its own change) |
| 2 | `2026-09-08_stage-02-catboost-swap.md` | pending |
| 3 | `2026-09-08_stage-03-calibration-ivap.md` | pending (depends on 2) |
| 4 | `2026-09-08_stage-04-directional-decoupling.md` | pending (separate worktree) |
| 5 | — (H1/H4 sweep, done 2026-09-08: both timeframes negative; `logs/strategy_matrix_H{1,4}.csv`) | COMPLETE |
| 6 | `2026-09-08_stage-06-crypto-trend.md` | pending (independent) |

Every brief carries the standing rails inline; the stage-2 file holds the
canonical copy.
