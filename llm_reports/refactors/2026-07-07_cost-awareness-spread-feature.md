---
type: refactor
date: 2026-07-07
time: 19:54 PDT
agent: Claude Fable 5
model: claude-fable-5
trigger: Brandon directed a retrain with a spread-cost feature after the M15 soak showed conviction landing exclusively on cost-untradeable GBP pairs
head: 9ef50fbd501dabb9f0621f5a5ff8f605ee15dd5a
scope: modifies-source
related:
  - handoffs/2026-06-19_gate-c-and-m1-tradeability.md
files_touched:
  - scripts/bake_spread_alphas.py
  - config/spread_alphas_m15.json
  - src/ml/features/v3_features.py
  - src/core/retrainer.py
  - src/strategies/concrete_strategies/ml_strategy.py
  - src/execution/risk_manager.py
  - run_oanda.py
  - tests/test_cost_feature.py
---

## Context

The M15 soak (2026-07-02 →, pid 933537) produced zero trades in its first
(holiday-shortened) week. Log analysis showed the model's only angel_prob≥0.40
bars landed on GBP_AUD/GBP_NZD — the two most expensive instruments (measured
spread cost 0.58 / 0.90 of a typical move) — while cheap instruments (XAU 0.07,
XAG 0.15) never lit up. Root problem: the model is symbol-blind AND cost-blind,
and worse, the training chop veto priced ALL instruments at a flat
`spread_atr_alpha=0.15`, so training kept GBP_NZD setups that live Gate A
always vetoes → part of the promoted model's validated edge was fictional.

Fix: one per-instrument cost table, measured empirically by the running soak's
SPREAD_CALIB logging, used in BOTH places — a `cost_ratio` feature the model
sees, and per-instrument alphas in the training veto. Retrain into a side dir.

## Investigation

- `_compute_chop_veto_mask` (src/core/retrainer.py) deletes rows via
  `df.filter(~mask)` AFTER target generation; alpha was the scalar
  `profile.spread_atr_alpha` (risk_manager.py, env RISK_SPREAD_ATR_ALPHA,
  default 0.15). chop_veto_rate relaxes the pooled OOS trade floor
  (`300*(1-rate)`).
- Training and live share the FeaturePipeline generators
  (V3Base/V3HTF/V3Session) — a new generator is symmetric by construction.
- Live SPREAD_CALIB (oanda_scalper_orchestrator) computes per-instrument
  `alpha_emp = median(spread_pct)/median(baseline_natr)` but was log-only.
- MLStrategy sources its schema from the model's `feature_names_in_` and
  selects columns strictly by name → an extra pipeline column is provably
  harmless to old 22-feature models. `threshold.json` load/save was the
  pattern to copy for the new artifact.
- Two symmetry traps found during design review: (1) a strict rolling median
  would null cost_ratio across the whole live warmup buffer → clean_data drops
  every row → strategy silently returns None forever; `min_samples=1` is
  mandatory and matches the veto's expanding median. (2) TA-Lib NATR emits
  float NaN which polars rolling aggregates PROPAGATE (nulls are skipped) →
  `fill_nan(None)` inside the generator.

## Findings / Changes

- **scripts/bake_spread_alphas.py (new):** parses last SPREAD_CALIB line per
  symbol from a soak log → JSON table with `denomination_minutes`,
  `default_alpha`, `samples`, `alphas`. Refuses thin samples (`--min-n`).
- **config/spread_alphas_m15.json (new):** baked from
  soak_2026-07-02_0215.log (n≈330/symbol, M15-denominated): XAU 0.0718,
  XAG 0.1462, EUR_JPY 0.3538, AUD_JPY 0.3624, GBP_JPY 0.3787, GBP_AUD 0.5834,
  NZD_JPY 0.6615, GBP_NZD 0.9032. Holiday-week-pessimistic (safe direction).
- **v3_features.py:** new `V3CostFeatures` — `cost_ratio = alpha_sym ×
  baseline_natr / natr_14` (live Gate A inequality rearranged; time-varying).
  No-op when `alpha_table is None`; mutable table attr for hot-reload.
- **retrainer.py:** `RETRAIN_SPREAD_TABLE` env gate (unset → bit-identical to
  before); `cost_ratio` appended to FEATURE_COLS only when set; per-symbol
  alpha lookup + per-symbol veto-count logging in `_compute_chop_veto_mask`;
  generator wired into `engineer_features_and_labels`; denomination-mismatch
  warning; `save_spread_table()` copies table → `model_dir/spread_alphas.json`
  on gate pass.
- **ml_strategy.py:** `_load_spread_table()` from `angel_path.parent`;
  boot RuntimeError if schema needs cost_ratio but table missing; hot-reload
  re-stats the table and swaps `_cost_gen.alpha_table` under the reload lock;
  heartbeat line now appends `cost_ratio=%.3f`; new `regime_window` kwarg.
- **risk_manager.py:** `RiskManager(alpha_overrides=...)` — per-instrument
  alpha in Gate A's stale-spread PROXY branch only (fresh tick spread wins).
- **run_oanda.py:** loads `spread_alphas.json` from `_MODEL_DIR`, wires
  `alpha_overrides` + `regime_window=risk_profile.regime_window`.

**Retrain result (the headline): gate FAILED — informatively.**
- Cost run (730d M15 ending 2026-07-07, logs/retrain_m15_cost_2026-07-07_1649.log):
  Brier 0.3399 ❌ (>0.30), PF 1.44 ✅, pooled OOS 360 ✅. Nothing saved;
  `models/forex_m15_cost` never created.
- Veto asymmetry worked as designed: GBP_NZD 91.2% vetoed, NZD_JPY 49.9%,
  GBP_AUD 28.8%, XAU/XAG 0% cost-vetoed. Overall 35.7% (vs 22.6% flat).
- **Control ablation** (same window, NO table,
  logs/retrain_m15_control_2026-07-07_1651.log): Brier 0.3150 → ALSO FAILED.
  The 5-day-newer window (holiday chop now in-window) is the PRIMARY cause;
  honest cost handling adds a secondary ~0.025 Brier via the reshaped
  (thinner) training population. Confirms the 07-02 promoted edge was partly
  propped up by flat-0.15 pricing of GBP_NZD.
- **Feature importance probe** (full-data Angel refit): cost_ratio 8th/23 at
  5.1% gain (top: vol_rel 15.5%, hour_of_day 12.9%). Correctly wired, used
  meaningfully, NOT a back-door instrument label.

## Verification

- 9 new unit tests (tests/test_cost_feature.py): veto-baseline parity,
  no-op bit-identity, alpha scaling + default fallback, veto asymmetry +
  flat-alpha equivalence, clean_data row parity, pooled-vs-single-symbol
  golden parity to 1e-12, exactly-warmup finite newest row. Full suite:
  100 passed.
- Old-model regression: booted new code against models/forex_m15 (no table) —
  22-feature schema, no cost_ratio column emitted, generate_signals clean.
- Env-gate check: FEATURE_COLS is 22 without RETRAIN_SPREAD_TABLE, 23 with.
- Soak pid 933537 verified alive before and after; prod models/ untouched.

## Risk & follow-ups

1. **Re-bake after a clean (non-holiday) week** and re-run — removes the
   window confound AND likely pulls NZD_JPY/GBP_AUD out of the veto-heavy
   zone (holiday alphas are pessimistic). The soak keeps collecting.
2. Optional feature-only ablation (feature + flat veto) to split
   feature-effect vs veto-population-effect. Importance probe says it's a
   valid test.
3. Do NOT loosen the Brier gate — it correctly caught both runs.
4. Live Gate A fresh-tick path unchanged by design; only the stale proxy
   branch uses table alphas.
5. Known accepted divergence: training-veto numpy median NaN-poisons
   leading-NaN windows (Gate A skips via isfinite) while the polars feature
   skips them — confined to ~first regime_window+13 bars per symbol.
6. Gate C retrainer symmetry TODO remains open (deliberately out of scope).

## Files touched

- `scripts/bake_spread_alphas.py` (new, ~110 lines)
- `config/spread_alphas_m15.json` (new artifact, checked in)
- `src/ml/features/v3_features.py` (+~95 lines: V3CostFeatures)
- `src/core/retrainer.py` (env gate ~l.241-270; FEATURE_COLS ~l.311;
  veto mask l.593-700; generator wiring l.756-775; clean/return l.866-885;
  save_spread_table l.1966-1995; main() phase 2.5 l.2052-2075)
- `src/strategies/concrete_strategies/ml_strategy.py` (table load, boot
  guard, hot-reload block, heartbeat, regime_window kwarg)
- `src/execution/risk_manager.py` (alpha_overrides; proxy branch)
- `run_oanda.py` (table load + wiring)
- `tests/test_cost_feature.py` (new, 9 tests)
