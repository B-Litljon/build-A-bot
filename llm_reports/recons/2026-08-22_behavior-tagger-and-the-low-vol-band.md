---
type: recon
date: 2026-08-22
time: 21:40 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "Brandon asked me to lead the market-behavior classifier + algorithm recommender build, after reviewing DeepSeek's plan."
head: b76f375eafe9d69e920c617fdcd559a41958b293
scope: modifies-source
related:
  - recons/2026-08-22_market-behavior-classifier-and-algorithm-recommender.md
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
  - recons/2026-08-17_5yr-candidate-and-the-untradeable-basket.md
files_touched:
  - src/ml/regimes/behavior_tagger.py
  - tests/test_behavior_tagger.py
  - src/ml/regimes/README.md
  - GLOSSARY.md
---

# Behavior tagger (Phase 1), and what it says about the low-volatility band

## Context

DeepSeek's 2026-08-22 plan proposed a four-phase offline tool: tag each bar with
a market behavior, then score model-configuration candidates per behavior.
Brandon asked me to lead the build. This report covers **Phase 1 only** — the
tagger — plus the first result obtained by running it on real bars.

Two corrections to the plan preceded any code; both are load-bearing.

## Investigation

**Correction 1 — the plan's proposed estimator leaks the future.** It says to
reuse "the p33/p67 convention of `reinforcement_voter.py:117`". That function
computes its cut points over the *entire* frame:

```python
atr_stats = df.select([
    pl.col(atr_col).quantile(0.33).alias("p33"),
    pl.col(atr_col).quantile(0.67).alias("p67"),
])
```

A bar's label therefore depends on bars that had not happened yet. This
directly violates the plan's own Risk item 2 (symmetry contract) and would make
every downstream per-cell number fiction.

The **live** Gate B (`RiskManager._evaluate_dynamic_gates`,
`risk_manager.py:413-427`) is already causal:

```python
arr = np.asarray(regime_series, dtype=float)
arr = arr[np.isfinite(arr)]
current = float(arr[-1])
warm = n >= p.regime_min_samples
pctile_rank = float(np.mean(arr <= current)) if warm else 0.5
```

The tagger adopts *this* estimator — trailing window, same length (260 bars),
same finite-filtering order, same cold-start rule (60 bars) — and keeps thirds
only as the bucketing convention.

**Correction 2 — the plan's safety note guards the wrong directory.** Risk item
3 says do not touch `models/forex/` because the live soak hot-reloads it. It
does not: that directory was last written **2026-06-14**. The live soak loads
`models/forex_m15_wide`, confirmed in the boot log of today's service test.

**Citation spot-check.** The plan flagged its own line citations as unverified.
I checked them and they are accurate: `_evaluate_dynamic_gates` at
`risk_manager.py:386`, `calculate_atr_regimes` at `reinforcement_voter.py:117`
with p33/p67, single-entry `STRATEGIES`, strategy hardcoded at
`run_oanda.py:234`, HMM on `(log_return, natr_14)` with no `hmm_latest.pkl`
anywhere on disk, `evaluate_performance.py` hardcoding 0.5x/3.0x and a 45-bar
hold, `ATR_KILL_SWITCH_THRESHOLD`, and `data/drift_report.json`.

## Findings / Changes

**Finding 1 (high) — the low-volatility band is where the money dies, and Gate
B's cut may be one third too lenient.**

Tagged 53,563 real M15 bars (8 instruments,
`analysis_cache/2026-07-27_m15_threshold_sweep/study_scored.parquet`), of which
3,803 carry simulated trade outcomes. **All figures GROSS of spread.**

| behavior | n | win% | mean pnl% | PF | 95% CI on mean |
|---|---:|---:|---:|---:|---|
| trend_high | 662 | 51.7 | +0.0489 | 1.40 | [+0.0155, +0.0828] * |
| trend_normal | 304 | 57.9 | +0.0446 | 1.75 | [+0.0159, +0.0744] * |
| mixed_high | 344 | 45.9 | +0.0411 | 1.31 | [-0.0045, +0.0885] |
| range_high | 274 | 47.8 | +0.0410 | 1.36 | [-0.0052, +0.0921] |
| range_normal | 429 | 50.8 | +0.0323 | 1.38 | [+0.0040, +0.0628] * |
| mixed_normal | 423 | 49.6 | +0.0203 | 1.24 | [-0.0062, +0.0490] |
| trend_low | 191 | 48.2 | -0.0046 | 0.94 | [-0.0368, +0.0302] |
| mixed_low | 480 | 42.9 | -0.0109 | 0.84 | [-0.0277, +0.0063] |
| range_low | 696 | 36.5 | -0.0325 | 0.63 | [-0.0477, -0.0161] * |

`*` = 95% CI excludes zero. Volatility orders the table perfectly: every
`_high` cell is profitable, every `_low` cell is not, `_normal` sits between.

Splitting on Gate B's actual cut point sharpens it:

| vol_rank band | n | win% | mean pnl% | PF |
|---|---:|---:|---:|---:|
| 0.00-0.20 (Gate B **vetoes**) | 856 | 34.6 | -0.0380 | 0.56 |
| 0.20-0.33 (Gate B **allows**) | 519 | 49.5 | +0.0052 | **1.08** |
| 0.33-0.67 | 1163 | 52.0 | +0.0301 | 1.38 |
| 0.67-1.00 | 1289 | 49.2 | +0.0446 | 1.36 |

Gate B is aimed correctly — what it vetoes is catastrophic (PF 0.56). But the
slice it *admits* just above the cut is gross-break-even at **PF 1.08**. Per
the 2026-08-08 spread-toll work, gross-to-net costs roughly 0.35-0.40 PF points
on this system (gross 1.373 → net 1.004). A 1.08 gross band is therefore
**net-negative**: 519 trades of pure cost. The decile view agrees — deciles 0-1
and 1-2 are 0.62 and 0.51, decile 2-3 is 1.03, and everything from 0.3 up is
≥1.16.

**Hypothesis (not a shipping decision): move Gate B's `regime_pctile` from 20
to ~33.** This is one in-sample window; it must clear the walk-forward gate
before it goes near the live tree.

**Finding 2 (medium) — the two axes are related but not redundant.**
`corr(vol_rank, trend_rank) = +0.355`. High volatility does tend to accompany
strong momentum, but the grid does not collapse to one dimension. All nine
cells populate healthily (6.4%-17.1% of warm bars; smallest cell n=191 trades),
and only 0.9% of bars are `cold`.

**Finding 3 (medium) — independent validation of the estimator.** The tagger's
`vol_rank` reproduces the July sweep's separately-computed `regime_rank` at
**corr 1.0000**, mean absolute difference 0.0040 over 53,091 bars. Two
independent implementations agreeing this closely is good evidence the rank is
the live gate's rank.

**Change — `src/ml/regimes/behavior_tagger.py`** (new, numpy-only, no repo
imports). `tag_bar` labels one bar from trailing windows (the live-shaped
entry point, designed to take the deques the orchestrator already maintains);
`tag_series` labels a history and is *defined* as repeated `tag_bar`.
`trend_strength_from_ppo` discards momentum sign, since a hard downtrend and a
hard uptrend are the same behavior and splitting them would halve every cell.

## Verification

`PYTHONPATH=src:. python -m pytest -q` → **235 passed** (was 200 at session
start; +10 log filters, +25 here). `python -m compileall -q src/` clean.

The two tests that carry the weight:

- `test_no_lookahead_prefix_invariance` — tagging the first *k* bars must give
  byte-identical results to the first *k* tags of the full history, checked at
  k ∈ {61, 100, 259, 260, 261, 500, 899}. If any tag consulted a later bar,
  removing the future would change it.
- `test_rank_agrees_with_live_gate_b_decision` — constructs a real
  `RiskManager` from the forex profile and drives `_evaluate_dynamic_gates`
  over 300 randomized windows, asserting the tagger's rank predicts Gate B's
  veto verdict every time. This is the symmetry contract, pinned against
  production code rather than a copy of it.

Also pinned: constants match `RiskProfile.for_asset_class("forex")` (drift
guard), window slides rather than expands forever, appending a bar never
rewrites history, trailing-NaN handling matches Gate B's filter-then-take-last
order, cold-start is labelled `cold` rather than guessed, and the label
vocabulary is closed at nine composites plus `cold`.

Two test-fixture bugs were found and fixed during the run (windows built with
the current bar at the wrong end); the module was correct both times.

## Risk & follow-ups

1. **Everything above is GROSS.** The ranking is probably robust; the levels are
   not. Re-scoring net of measured spread is Phase 3 work and will move every
   PF down.
2. **Single window, and not walk-forward.** The July sweep's population also
   includes thresholds below the live 0.40 bar, so it takes far more trades
   than production would. Direction should hold; magnitudes will not.
3. **The Gate B finding touches live money.** It is a hypothesis for the retrain
   gate, not a config edit. Do not change `regime_pctile` in the working tree —
   the soak launches from it.
4. **Open decision, needed before Phase 2:** the plan proposes generalizing
   `replay_test.py` / `evaluate_performance.py`. Those are Alpaca-era and
   hardcode the *equities* configuration (root `models/angel_latest.pkl`,
   0.5x/3.0x brackets, threshold 0.50), and `table-o-content.md:346` calls that
   harness dormant. The maintained forex scorer is the retrainer's walk-forward
   path. My recommendation is to extend that instead, which also merges with
   the 2026-08-17 recommendation to score only broker-tradeable instruments.
5. **`models/forex/` is not live** — correct DeepSeek's Risk item 3 before
   anyone acts on it. `models/forex_m15_wide` is the hot directory.
