---
type: refactor
date: 2026-07-27
time: "22:50 PDT"
agent: "Claude Fable 5"
model: claude-fable-5
trigger: "Green-lit post-soak plan: fix the reconnect seam that ate ~half the soak's signals, and hoist the eight-way-hardcoded ANGEL_THRESHOLD before any sweep"
head: faff4c32b2ee5a1784c9e27f15f9e206206dd28b
scope: modifies-source
related:
  - recons/2026-07-27_threshold-ev-study.md
files_touched:
  - src/execution/oanda_scalper_orchestrator.py
  - src/core/thresholds.py
  - src/core/retrainer.py
  - src/execution/live_orchestrator.py
  - src/strategies/concrete_strategies/ml_strategy.py
  - src/ml/train_model.py
  - src/day_trading/train_model.py
  - src/analysis/optimize_brackets.py
  - src/analysis/failure_modes.py
  - src/replay_test.py
  - tests/test_oanda_scalper.py
  - tests/test_ml_strategy_guards.py
  - tests/test_retrainer_output_dir.py
  - src/execution/README.md
  - src/core/README.md
  - src/analysis/README.md
  - tests/README.md
  - GLOSSARY.md
  - CLAUDE.md
  - .gitignore
---

# Seam catch-up + ANGEL_THRESHOLD hoist (commits d66f6ec, 7b946c8)

## Context

The Jul 13–27 soak lost ~6 of 15 would-be signals to reconnect gaps: every
disconnect wipes the bar buffers and re-primes from REST, but primed bars
were never evaluated, so anything that sealed while the stream was down was
silently skipped. Separately, the Angel proposal bar (0.40) was hardcoded in
eight files while the Devil's training population and the bracket fit are
both conditioned on it — a sweep would have meant editing all eight in
lockstep or silently desynchronising the stages. Both fixes are prerequisites
for any threshold work and useful regardless of its outcome (the companion
recon concluded the threshold should NOT move).

## Findings / Changes

**Commit `d66f6ec` — seam catch-up** (`oanda_scalper_orchestrator.py`):

- Extracted the decision tail of `_on_bar` (inference → guards → bracket →
  order) into `_evaluate_and_trade`, shared by both paths. Behavior-neutral
  for the stream path.
- New `_catch_up_missed_bars`: after every prime (boot and reconnect), score
  the newest sealed bar per symbol iff fresh (≤ one bar period by default;
  `SEAM_CATCHUP_MAX_AGE_SECONDS` tunes, `0` disables) and not already
  scored. Runs before the stream (re)starts, so it cannot race `_on_bar`.
- New `_last_scored_ts` dedup: shared by stream path and catch-up,
  deliberately NOT reset on reconnect — repeated re-primes inside one bar
  period cannot double-score (= double-order) the same bar. Marked BEFORE
  evaluation: a failed evaluation must not retry into a duplicate order.
- Evaluation errors are contained per symbol so they cannot kill the
  reconnect loop (which would leave the bot permanently disconnected).
- Boot semantics documented: a fresh signal bar sealed just before a crash
  relaunch is now scored — this can deliberately re-enter a position that
  `_reconcile_on_boot` just flattened, with fresh brackets.

**Commit `7b946c8` — threshold hoist + artifact pinning**:

- New `src/core/thresholds.py`: single source of truth;
  `ANGEL_THRESHOLD` env var overrides at process start (train/analysis-time
  knob).
- All eight hardcode sites now import it: `retrainer.py` (Devil population
  filter + validation gate), `ml_strategy.py` (constructor default),
  `live_orchestrator.py`, `ml/train_model.py`, `day_trading/train_model.py`,
  `optimize_brackets.py`, `failure_modes.py`, `replay_test.py`.
- `retrainer.save_threshold()` now pins `angel_threshold` into
  `threshold.json` (and `metadata.json`) so a model pair records the
  population it was trained at.
- `MLStrategy._load_threshold` → `_load_thresholds`: reads BOTH bars from
  `threshold.json`; the artifact value wins over the constructor default;
  hot-reload picks up Angel changes; pre-2026-07 artifacts (devil-only file)
  keep working via fallback.
- Flagged pre-existing rot while there: `src/analysis/optimize_brackets.py`
  has been **import-dead since 59a1125 (2026-05-22)** (references removed
  `get_alpaca_client`) — its bracket numbers predate the M15 era; the live
  M15 brackets actually come from `RiskProfile.for_asset_class("forex")`.
  README flagged, not fixed (no current consumer).

## Verification

- Full suite: 118 → **128 tests, all passing** (7 seam tests + 3 threshold
  tests); `python -m compileall -q src/` clean.
- New seam tests pin: fresh missed bar scored and traded exactly once;
  stale bar (weekend gap) skipped; repeat reconnects dedup before inference;
  warmup respected; `max_age=0` disables; an exception inside evaluation
  does not escape; a stream-scored bar is ineligible for catch-up.
- New threshold tests pin: pinned `angel_threshold` overrides the default;
  legacy devil-only `threshold.json` keeps the default; missing file keeps
  both constructor values; `save_threshold` writes both keys.
- Env-override smoke: `ANGEL_THRESHOLD=0.325` propagates identically through
  `core.thresholds`, `src.core.thresholds`, retrainer, failure_modes, and
  replay_test in one process.

## Risk & follow-ups

- The catch-up path can enter a trade up to ~one bar period after the signal
  bar sealed; entry price may have drifted within that window (same order of
  drift as normal end-of-bar execution, but worth remembering when reading
  fills that follow a reconnect — grep `SEAM_CATCHUP` in the soak log).
- Legacy Alpaca-path trainers now share the env knob; a persistent
  `ANGEL_THRESHOLD` in `.env` would retune their next training run too. The
  artifact pinning defuses the live-side risk (deployed pairs carry their
  own bar).
- Docs updated per CLAUDE.md three-layer rule (module glossaries, folder
  READMEs, GLOSSARY.md, test counts 114 → 128 in CLAUDE.md/tests README).

## Files touched

See frontmatter. Diffs: `git show d66f6ec` (seam, +290/−4 over 4 files),
`git show 7b946c8` (hoist, +364/−60 over 18 files incl. docs).
