---
type: refactor
date: 2026-05-25
time: 19:50 UTC
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: User asked to make A3 chop floor runtime-tunable while the training/inference asymmetry stays open
head: 82dbdba (main tip at branch creation)
scope: modifies-source
related:
  - refactors/2026-05-23_lightgbm-hmm-pilot.md
  - audits/2026-05-24_lightgbm-pilot-soak-readiness.md
files_touched:
  - src/execution/risk_manager.py
---

# Configurable A3 chop floor + kill switch

## Context

The A3 chop filter at `src/execution/risk_manager.py:54` vetoes any
signal whose ATR-derived SL distance falls below a hardcoded floor.
Recent work made the floor asset-class-aware (2.0 pips for forex via
`RiskProfile.for_asset_class("forex")`, 0.15% for equities), but the
floor values were still compile-time constants.

The deeper issue is a **training/inference asymmetry**: the retrainer
does not simulate this filter during bracket-target generation, so the
Devil's PF 2.22 was learned on the full Angel-approved population
including bars the runtime filter would have vetoed. Yesterday's commit
`82dbdba` added a counter on `OandaScalperOrchestrator` to measure how
often this happens in practice; the soak's chop-veto rate over the
first week will decide whether the asymmetry is academic (< 1%) or
load-bearing (> 5%).

This change is the **operational counterpart** to that telemetry work:
expose the floor and a kill switch as env vars so the operator can
tune the filter in production without code edits, and so the soak can
be re-run with the filter disabled if needed to A/B test live PF
against the unfiltered training distribution.

**This does NOT close the asymmetry.** Closing it requires changing
the retrainer's training-set selection logic, which is a larger change
with a full retrain + gate evaluation downstream. That work is
deferred until the soak's chop-counter data tells us if the asymmetry
is actually biting hard enough to justify it.

## Findings / Changes

### Change 1 — Env vars on `RiskProfile.for_asset_class`

`src/execution/risk_manager.py` exports three new env var names at
module scope so they're discoverable:

```python
ENV_FOREX_MIN_SL_PIPS    = "RISK_FOREX_MIN_SL_PIPS"
ENV_EQUITIES_MIN_SL_PCT  = "RISK_EQUITIES_MIN_SL_PCT"
ENV_CHOP_FILTER_ENABLED  = "RISK_CHOP_FILTER_ENABLED"
```

`for_asset_class` reads the appropriate env var per branch with the
prior hardcoded value as the default fallback. Behavior with no env
vars set is identical to before this change.

### Change 2 — Kill switch in `calculate_bracket`

When `_chop_filter_enabled()` returns False (env var in
`0|false|no|off`), the floor comparison still computes but the
rejection is suppressed — the bracket is returned and the trade
proceeds. The would-have-vetoed condition is logged so operators can
still see how often the filter *would* have triggered while it's
disabled.

When the filter is enabled (default), the rejection log line now
includes the actual `sl_dist`, the active `floor`, and the shortfall —
useful diagnostic granularity beyond the orchestrator-side counter.

## Verification

```
default forex min_sl_pips=2.0                  ✓ matches prior hardcoded
default equities min_sl_pct=0.0015             ✓ matches prior hardcoded
RISK_FOREX_MIN_SL_PIPS=1.5    → min_sl_pips=1.5     ✓ override works
RISK_EQUITIES_MIN_SL_PCT=0.001 → min_sl_pct=0.001    ✓ override works
RISK_CHOP_FILTER_ENABLED unset, tiny sl_dist  → None (rejected)
RISK_CHOP_FILTER_ENABLED=0,    tiny sl_dist   → (sl, tp) (passed)
```

All truthy variants resolve correctly:
- Falsy: `0`, `false`, `False`, `no`, `off`
- Truthy: `1`, `true`, `yes`, `on`, `""` (unset), or any other value

## Risk & follow-ups

- **Training/inference asymmetry remains open.** This change exposes
  the runtime knob; it does not align the retrainer with the runtime
  filter. The Devil still learns on a slightly different population
  than it will face live.
- **Soak data informs the next step.** If commit `82dbdba`'s counter
  shows the chop filter biting < 1% of approvals through Friday's
  check-in, the asymmetry is academic and no further work is needed.
  If > 5%, the retrainer-side fix becomes mandatory before any
  real-money promotion.
- **No `.env.example` updated.** This repo doesn't have one (`.env`
  is in `.gitignore` and there's no template). New operators should
  consult this report or `src/execution/risk_manager.py` for the env
  var names.

## Files touched

- `src/execution/risk_manager.py` — added env var imports +
  `_chop_filter_enabled()` helper + plumbing in
  `RiskProfile.for_asset_class` and `RiskManager.calculate_bracket`.
