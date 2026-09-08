# STAGE 4 — Long/short pipeline decoupling (FX)

You are implementing Stage 4 of the quant-architecture roadmap. Read the
stage-1..3 recon notes in `llm_reports/recons/` first to know what is
promoted. Work in a SEPARATE git worktree if stages 1-3 are still in flight —
you touch the same files (`src/core/retrainer.py`, model-dir layout).

## Hard rails

Same standing rails as every stage (see stage-2 file, "Hard rails").
Additional stage-specific rules:

- A direction side that cannot clear the holdout gate is WITHHELD, never
  pooled back in to dilute the verdict. Same gate, same
  `HOLDOUT_PF_CONFIDENCE = 0.95`, no relaxation for a side model.
- The live bot's net-position model (FIFO, one signed number per instrument,
  see `src/execution/oanda_order_manager.py`) makes same-bar both-direction
  entries illegal by construction. Your arbitration must respect that.

## What to build

1. Split Angel/Devil per sign: `models/forex_m15_long/`,
   `models/forex_m15_short/`, each with the full artifact set (pkl pair,
   `threshold.json`, `spread_alphas.json`, `metadata.json`, plus
   `barriers_*`/`calibration.json` if stages 1/3 promoted).
2. Sign-mirror the features where direction matters: `log_return`,
   `dist_sma50`, `ppo` flip sign for the short model; symmetric features
   (natr_14, session flags, vol_rel) do not. Document the mirror list in the
   model dir metadata so drift probing knows.
3. Retrainer trains both sides on the same fold boundaries; each side gets
   its own holdout verdict. Pooled comparison metric: the decoupled PAIR's
   CP-PF lower bound >= the incumbent pooled model's on identical data.
4. Arbitration at inference (MLStrategy): if both sides fire on one bar, the
   higher CALIBRATED score wins; tie -> stand down. Verify against
   `oanda_order_manager.submit_target_position` semantics before assuming.
5. The soak currently serves `models/forex_m15_wide` (see
   `systemctl --user cat soak.service`). You are NOT promoting anything —
   the split dirs are produced and validated offline; promotion is a human
   step that changes OANDA_MODEL_DIR.

## Promotion bar

- Each side individually: CP-PF lower bound >= 1.0 on the holdout at 95% or
  the side is withheld.
- The pair, where both sides are live, must beat the pooled incumbent's
  bound on identical data.
- Full test suite green.

## Deliverable

Recon note in `llm_reports/recons/`: per-side numbers, fold-by-fold, raw
outputs quoted. A side failing its gate is a result, not a failure of your
work — report which side has an edge and which does not.