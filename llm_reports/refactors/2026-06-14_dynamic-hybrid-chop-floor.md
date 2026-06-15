---
type: refactor
date: 2026-06-14
time: 18:16 PDT
agent: Claude Opus 4.8
model: claude-opus-4-8
trigger: Architect (Gemini) + Claude M2M — replace the static A3 chop floor with a symmetric, dual-mode, coupled hybrid filter and close the training/inference asymmetry
head: cb09170c96dddf0e3faca84ae3a6687a07890365
scope: modifies-source
related:
  - refactors/2026-05-25_configurable-chop-floor.md
  - audits/2026-06-09_full-repo-audit.md
files_touched:
  - src/execution/risk_manager.py
  - src/execution/oanda_scalper_orchestrator.py
  - src/core/retrainer.py
  - tests/test_risk_manager.py
---

# Dynamic Hybrid Chop Floor — coupled cost + regime gates, symmetric live↔training

**Authored 2026-06-14 18:16 PDT · Claude Opus 4.8 (`claude-opus-4-8`).**

## Context

The static 2.0-pip A3 floor in `RiskManager.calculate_bracket` vetoed **2/2**
Devil-approved signals in the 49h V5 soak (both GBP_JPY, sl_dist 1.07/0.99 pips
< a ~2-pip spread). Diagnosis: the floor is sound in principle — a stop tighter
than the spread is a guaranteed loser — but it was (a) misnamed (a min-volatility
/ spread floor, not a "chop" detector), (b) miscalibrated (2.0 pips never
validated; only 0.2 was), and (c) **asymmetric**: the retrainer generated targets
at `sl_mult·ATR` with no floor, so PF 2.22 was measured on a population the live
filter forbids.

This refactor replaces it with the agreed **Option 4** — a coupled hybrid of a
transaction-cost gate and a low-volatility regime gate — simulated **identically**
in live execution and the training pipeline so the model trains only on the
tradeable population. The directionality of the volatility coupling was itself a
peer disagreement (Claude "tighten" vs Gemini "loosen"); rather than pick one,
**both modes ship behind a runtime toggle** for soak/backtest A/B selection.

## Summary of the M2M prompt

Implement a symmetric coupled chop filter with dual-mode tracking:
`RISK_COUPLING_MODE ∈ {tighten, loosen}`; stateful incremental NATR (deque,
maxlen 260) to avoid vector recompute / Wilder seed drift; vectorized Polars
masking in the retrainer with an era-scaled (not static) spread proxy; coupling
`scale = max(0,(rank−0.5)/0.5)`, `k_eff = base·(1±coupling·scale)` clipped to
`≥ 1.0`; split telemetry; tests for both modes incl. clipping and edge cases;
green pytest; this report.

## Architecture

### Shared kernel (single source of truth)
`coupled_keff(base, coupling, mode, pctile_rank)` in `risk_manager.py` is imported
by **both** the live gate and the retrainer mask, so the math cannot drift. It is
scalar- and array-vectorized.

```
scale  = max(0, (pctile_rank − 0.5) / 0.5)        # couples only above the median
tighten: k_eff = base · (1 + coupling · scale)    # Claude: more cost discipline as vol expands
loosen:  k_eff = base · (1 − coupling · scale)    # Gemini: relax during high-momentum runs
k_eff  = max(1.0, k_eff)                           # clip: spread can never exceed sl_dist
```

Gate params live on `RiskProfile` (`spread_k_base`, `spread_k_coupling`,
`spread_k_coupling_mode`, `regime_pctile`, `regime_window`, `regime_min_samples`,
`spread_atr_alpha`) and are env-overridable in `for_asset_class("forex")`. Because
`retrainer.py` already sources `sl_mult/tp_mult` from the same `RiskProfile`
(`:107-108`), the gate parameters inherit the identical symmetry path — symmetry
by construction, not duplication.

### Two gates (live: `RiskManager._evaluate_dynamic_gates`)
- **Gate B — regime:** veto if `pctile_rank < regime_pctile/100` (current NATR in
  the bottom P% of its trailing window). Active only once the window is *warm*
  (`n ≥ regime_min_samples`); below that, `pctile_rank` is held neutral (0.5).
- **Gate A — cost:** veto if `sl_dist < k_eff · spread_proxy`. `spread_proxy` is
  the live bid-ask spread when fresh, else a **volatility-scaled proxy**
  `alpha · baseline_ATR` (baseline = median of the window). Runs at any window
  size (cost protection needs no warm-up).
- Veto = A **or** B; Gate B is evaluated first. `RiskManager.last_veto_gate`
  records which fired, read by the orchestrator for split telemetry.

### Stateful, drift-free NATR (orchestrator)
`_seed_regime` runs one `talib.NATR` over the priming frame at boot to seed the
per-symbol `deque(maxlen=regime_window)` and the running Wilder ATR. `_update_regime`
then advances the Wilder recursion **O(1) per closed bar** (`atr = (atr·13 + TR)/14`)
— no per-bar vector recompute, no reseed drift. The priming tail was widened to
`max(warmup, regime_window) + NATR_PERIOD + 5` bars so the deque seeds full. All
windows are **bar-count**, never calendar.

`_on_tick` captures the live spread with two GIL-atomic dict writes (no lock,
sub-µs) — the <50µs tick budget is preserved.

### Symmetric training mask (`retrainer._compute_chop_veto_mask`)
Per symbol, vectorized: `sliding_window_view` for the full-window region and an
expanding loop (~259 rows) for the cold-start region, computing the rank, the
neutral-when-cold coupling rank, and the median baseline. Gate A/B applied with
the **same `coupled_keff`**. The era-scaled proxy (`alpha · baseline`) replaces
static historical spread constants. Vetoed rows are dropped **after** target
generation (see deviation #2). Verified bit-identical to the live gate: **0
mismatches / 400 rows across tighten+loosen, coupling 0.0–1.0, multiple seeds.**

### Telemetry
`_a3_chop_rejections` split into `_spread_gate_rejections` / `_regime_gate_rejections`
(combined total retained); per-gate veto ratios logged so the soak reveals which
gate binds.

## Deviations from the prompt (with justification)

1. **Report location.** Prompt asked for `llm_reports/architecture/…` (and earlier
   a root `YYYY-MM-DD_HHMM.md`). The repo `llm_reports/README.md` defines a fixed
   taxonomy with no `architecture/` folder; shipped code → `refactors/`. Followed
   the repo convention; topic kebab-cased, date-prefixed, no time suffix.
2. **Veto applied AFTER target generation, not before.** The prompt said "purge
   vetoed rows before calculation of target parameters." Doing so would corrupt
   the bracket forward-walk: the SL/TP path between entry and exit must include
   *every* bar, even ones we won't enter on. The veto governs whether we **enter**
   on bar `i`, so targets are computed on the full contiguous series and vetoed
   *entry* rows are dropped afterward. This is the correct order and still yields
   "training population = live-tradeable population."
3. **Gate-enable flags as `RiskProfile` properties** (not dataclass fields) so they
   read the env live and never need threading through constructors.

## Tracking keys (for the next agent)
- Env: `RISK_COUPLING_MODE` (tighten|loosen), `RISK_SPREAD_K`,
  `RISK_SPREAD_K_COUPLING`, `RISK_REGIME_PCTILE`, `RISK_REGIME_WINDOW`,
  `RISK_REGIME_MIN_SAMPLES`, `RISK_SPREAD_ATR_ALPHA`, `RISK_SPREAD_GATE_ENABLED`,
  `RISK_REGIME_GATE_ENABLED`, `RISK_SPREAD_STALE_SECONDS`, `RISK_CHOP_FILTER_ENABLED`
  (master kill).
- Defaults: `spread_k_base=1.5`, `coupling=0.0` (decoupled/safe), `mode=tighten`,
  `regime_pctile=20`, `regime_window=260`, `regime_min_samples=60`,
  `spread_atr_alpha=0.15`, `spread_stale=5s`.
- Telemetry attrs: `_spread_gate_rejections`, `_regime_gate_rejections`,
  `_a3_chop_rejections` (combined), `_devil_approved_total`.
- Veto identifiers: `GATE_SPREAD`, `GATE_REGIME`, `GATE_STATIC`, `GATE_NONE`.

## Verification
- `pytest tests/` → **61 passed** (16 in `test_risk_manager.py`, incl. both modes,
  clipping floor, regime/cost vetoes, proxy fallback, cold-start bypass, rollover).
- Live↔training symmetry cross-check: **0/400 mismatches** across modes/coupling/seeds.

## Unfinished / next steps
1. **Calibrate `spread_atr_alpha`** from measured `median(live_spread)/median(baseline_ATR)`
   per instrument during the next soak — the default 0.15 is a placeholder. Until
   then, *live-fresh* Gate A uses real spread (correct) and only the *training* /
   *stale-fallback* proxy depends on alpha (residual, documented asymmetry).
2. **Retrain the 8-instrument forex basket** and re-gate; expect a justified PF
   shift now that PF is measured on the tradeable population. Per
   [[feedback_verify_retrain_handoffs]]: re-run, do not trust reported metrics.
3. **Soak with split telemetry** through a US-session vol window; pick `tighten`
   vs `loosen` from which gate binds and the realized fill quality. Non-zero
   qualified fills is the success signal (the prior 0-trade soak was a dead-tape +
   sub-spread artifact).
4. Coupling default is 0.0 (decoupled) — dial up only after the soak shows the
   spread distribution justifies it.
