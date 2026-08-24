---
type: recon
date: 2026-08-22
time: 20:25 PDT
agent: opencode (DeepSeek V4 Pro)
model: deepseek-v4-pro
trigger: "Brandon asked to plan a tool that identifies market behavior and picks the algorithm best suited to it; scope settled as an offline diagnostic + recommender ranking existing model variants."
head: b76f375eafe9d69e920c617fdcd559a41958b293
scope: read-only
related:
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
  - recons/2026-08-17_5yr-candidate-and-the-untradeable-basket.md
  - m2m_prompts/2026-08-10_algorithm-and-strategy-research.md
---

# Market-behavior classifier + algorithm recommender: plan

## Context

Brandon asked to plan a tool that (1) identifies market behavior and (2) decides
which kind of algorithm would perform best in that behavior. Two scoping
questions settled the shape: the tool is an **offline diagnostic + recommender**
(no live code, no risk to the running soak), and the "algorithms" it ranks are
**variants of the current model** (model dir, threshold pair, bracket widths,
chop-gate config) rather than new strategy types.

The question this recon answers: what should this tool be, given what the repo
already has — and what exists today that it should reuse rather than rebuild.

## Investigation

Read `GLOSSARY.md` end to end, the 2026-08-10 research brief
(`m2m_prompts/2026-08-10_algorithm-and-strategy-research.md`), and the most
recent recon (`llm_reports/recons/2026-08-17_5yr-candidate-and-the-untradeable-basket.md`)
for measurement discipline. Dispatched a codebase sweep (explore agent) for
regime detection, strategy architecture, backtesting, training, meta-switching
logic, and feature engineering. Key findings from that sweep, all confirmed in
the glossary or the brief:

1. **Regime detection already exists in three unrelated forms** — the glossary
   itself warns of the name collision (`GLOSSARY.md:409-412`):
   - Volatility-band gate: `RiskManager._evaluate_dynamic_gates()`
     (`src/execution/risk_manager.py:386-465`) — percentile rank of current NATR
     in its trailing 260-bar window; live Gate B.
   - HMM hidden states: `src/ml/regimes/hmm_regime.py`, 3-state GaussianHMM on
     `[log_return, natr_14]`, currently OFF in production (no `hmm_latest.pkl`
     in `models/forex/`).
   - Offline ATR-band bucketing: `calculate_atr_regimes()` at
     `src/analysis/reinforcement_voter.py:117` — Low/Normal/High by p33/p67
     quantiles, the only existing per-regime performance breakdown
     (`data/drift_report.json`).

2. **Only one strategy is registered.**
   `STRATEGIES` in `src/strategies/concrete_strategies/__init__.py:13` has a
   single entry (`ml_strategy`); runtime selection is hardcoded in
   `run_oanda.py:234`, not config-driven. There is no per-regime algorithm
   selection or ensemble weighting anywhere; the closest thing is the chop
   filter gates (`risk_manager.py`) and the blunt `ATR_KILL_SWITCH_THRESHOLD`
   in `src/execution/live_orchestrator.py:240`.

3. **The measurement harness is hardcoded in exactly the two places the tool
   needs to generalize.** `src/replay_test.py` reads legacy root model paths
   (lines 77-78); `src/evaluate_performance.py` hardcodes SL/TP at 0.5×/3.0×
   ATR with a 45-bar hold (lines 71-73) and already consumes `drift_report.json`
   to shift the Devil threshold in the high-ATR band — a miniature precedent for
   regime-conditional behavior.

4. **The research brief lists "regime-conditional strategy switching" as an open
   lead** (Part 2), and its Part 3 demands cheapest-falsification designs. Its
   §3 central constraint (transaction cost dominates predictive skill) and §5
   dead ends (gross-scored promotion gate, thin live fill rates of 1-1.5/week)
   shape what the recommender may honestly claim.

## Findings

**Finding 1 (high) — the repo already carries ~80% of this tool as parts; the
work is assembly, not invention.** A deterministic volatility-band tagger exists
(reinforcement_voter), a statistical one exists (HMM, off), a replay + grading
harness exists, and per-regime reporting has one working example.

**Finding 2 (high) — the plan, four phases, offline and read-only toward the
soak:**

- **Phase 1 — Behavior tagger.** New `src/ml/regimes/behavior_tagger.py`:
  deterministic, sealed-bar-only labels (no lookahead, no black-box classifier —
  it must be reproducible live later per the symmetry contract). Two dimensions
  from features already built: volatility band (Low/Normal/High, reuse the
  p33/p67 convention of `reinforcement_voter.py:117`) and trend-vs-range (from
  `ppo` / `htf_trend_agreement` / BB width). Composite labels like `trend_high`,
  `range_low`, `normal`. Optional later: HMM states from `hmm_regime.py` as a
  third, compared against the rule-based one. Tests: reproducibility, no
  future-peek, label transition matrix.

- **Phase 2 — Candidate set.** Each candidate is an evaluation config, not new
  strategy code: model dir (`models/forex/`, `forex_m15_wide/`, `forex_m15_5yr/`,
  …), threshold pair (per-dir `threshold.json`), bracket widths (current 2×/4×
  vs the retired 1×/2×), chop-gate config. Requires parameterizing the two
  hardcoded harness points from Investigation item 3. Keep the grid curated
  (~6-10 candidates), not a full cross-product — the 2026-08-17 recon's power
  analysis shows cells thin out fast.

- **Phase 3 — Matrix + recommender.** New `src/analysis/behavior_matrix.py`:
  tag every bar, run each candidate through replay + grade, emit
  `data/behavior_matrix.parquet` (regime × candidate × {n, win rate, EV, PF,
  net-of-cost}). Re-score net of measured spread (the gross-PF trap the brief
  calls out in §6); suppress cells below a trade floor; bootstrap CIs so "best
  in trend_high" carries honest uncertainty. Output: markdown report in
  `llm_reports/` + machine-readable JSON.

- **Phase 4 — Housekeeping.** README entries for touched folders, `Glossary:`
  headers in new modules, GLOSSARY.md terms (`behavior tag`, `candidate`,
  `matrix`), tests, `python -m compileall -q src/`.

**Finding 3 (medium) — the tool's honest deliverable is a hypothesis generator,
not proof.** With 1-1.5 live fills/week and thin per-cell sample sizes, the
matrix ranks candidates per regime with wide intervals; its job is to direct
later experiments (and eventually a live switching layer), not to certify one.

## Verification

Read-only session; nothing was executed and no files were modified. Citations
from the explore sweep were spot-checked against `GLOSSARY.md` and the research
brief, which I read directly; the live-soak safety rules (CLAUDE.md) were
observed throughout. Line-level citations from the sweep itself (risk_manager,
replay_test, evaluate_performance, hmm_regime) were not re-verified by eye and
are marked as such here; Phase 1 should re-confirm them before editing.

## Risk & follow-ups

1. **Thin cells.** The matrix will have regimes × candidates × instruments;
   treat any cell with n < ~30 trades as uninformative.
2. **Symmetry contract.** The tagger must be computable identically at training
   time and (eventually) live; a tagger that leaks the future makes the matrix
   fiction.
3. **Do not touch `models/forex/`** — it is hot-reloaded by the live soak; the
   tool only reads artifacts.
4. **Recommended next step:** build Phase 1 (the tagger) first — it is small,
   testable in isolation, and unblocks everything downstream. The Phase 2
   generalization of `replay_test.py`/`evaluate_performance.py` is the highest
   leverage follow-up; the 2026-08-17 recon's recommendation (gate should score
   only broker-tradeable instruments) overlaps it and should be decided
   together.
5. No `m2m_prompts/` brief was requested for this work (explicitly declined);
   the plan lives here until a work order is cut.

## Files touched

Read only:

- `GLOSSARY.md` (full)
- `m2m_prompts/2026-08-10_algorithm-and-strategy-research.md` (full)
- `llm_reports/README.md`, `llm_reports/_TEMPLATE.md`
- `llm_reports/recons/2026-08-17_5yr-candidate-and-the-untradeable-basket.md`
- `CLAUDE.md` (working-tree copy)

Read via explore-agent sweep (line citations not independently re-verified):

- `src/execution/risk_manager.py` (241-465)
- `src/execution/oanda_forex_orchestrator.py` (371-572, 1124-1130)
- `src/core/retrainer.py` (393, 430-460, 747-808, 1303-1523, 1953-2023)
- `src/strategies/base.py`, `src/strategies/concrete_strategies/__init__.py`,
  `src/strategies/concrete_strategies/ml_strategy.py`
- `src/replay_test.py`, `src/evaluate_performance.py`
- `src/analysis/reinforcement_voter.py`, `src/core/feedback_loop.py`
- `src/ml/regimes/hmm_regime.py`, `src/ml/features/v3_features.py`,
  `src/ml/feature_pipeline.py`, `src/ml/core/interfaces.py`,
  `src/ml/feature_stats.py`
