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

## Legacy ledger threads (pre-2026-09)

The `2026-06-*` … `2026-08-*` files predate this folder: they lived in the
top-level `m2m_prompts/` directory (removed 2026-09-17) as a ledger of
model-to-model handoffs — the *input* counterpart to `llm_reports/` outputs,
kept so a dispatched instruction could be diffed against the commit it
produced. Each carries frontmatter with `to`, `from`, `status`
(drafted → dispatched → completed → verified), and a `result_commit` two-way
link: the prompt points to the commit, the commit points back to the prompt.
New work uses the append-only **live thread** convention below instead.

## Live threads (not briefs)

A brief is one-way: point an agent at it and it has a contract. When **two
agents are working one feature in the same checkout at the same time**, use a
thread file instead — append-only, one `## [timestamp] <from> → <to>` block per
message, claims marked VERIFIED / ASK / OFFER / BLOCKER / DECIDED.

| Thread | Topic | State |
|---|---|---|
| `2026-09-14_barrier-live-seam.md` | Quantile barrier producer ↔ consumer seam (retrainer artifacts, `MLStrategy` payload, `RiskManager` substitution) | CLOSED as a feature — 16 blocks; the decision view is [`recons/2026-09-14_session-evidence-and-options.md`](../recons/2026-09-14_session-evidence-and-options.md) |
| `2026-09-24_quant-lanes.md` | The five parallel 2026-09-24 quant lanes (crypto / equity PEAD / forex RV / option var / falsification) and the shared `src/lab/stats.py` contract | OPEN |

