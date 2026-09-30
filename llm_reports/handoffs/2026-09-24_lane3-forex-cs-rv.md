---
type: handoff
date: 2026-09-24
time: 15:47 PDT
agent: dsh (DeepSeek V4.1 Flash via Ollama Cloud)
model: ollama-cloud/deepseek-v4.1-flash
trigger: Lane 3 dispatch — cross-sectional relative-value forex to salvage M15 OANDA lane
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
files_touched:
  - src/lab/forex_cs_rv.py
  - tests/test_lab_forex_cs_rv.py
  - llm_reports/recons/2026-09-24_lab-forex-cs-rv.md
  - llm_reports/stops/2026-09-24_forex-cs-rv-decommission.md (conditional, only if gate fails)
related:
  - llm_reports/recons/2026-09-14_session-evidence-and-options.md (single-instrument bracket closure)
  - llm_reports/recons/2026-09-23_lab-model-family-ab.md (CatBoost A/B verdict)
  - llm_reports/recons/2026-09-21_lab-served-artifact-baseline.md (served artifact 0-fills evidence)
  - llm_reports/audits/2026-09-08_high-benefit-fixes-ranked.md (five fixes all done)
---

## 1. System Persona & Scope

You are a **quantitative researcher** and **data-pipeline engineer** producing a **research-only falsification artifact**. You are not building execution, not submitting orders, and not modifying the live M15 soak. Your deliverable is a **cross-sectional relative-value (RV) basket model** trained and gated through the existing `src/lab` harness, plus a decommissioning report if the gate fails.

**Scope lock — you may only touch:**
- `src/lab/forex_cs_rv.py` (new)
- `tests/test_lab_forex_cs_rv.py` (new)
- One recon report in `llm_reports/recons/`
- If the gate fails: one decommissioning report in `llm_reports/stops/`
- `src/lab/README.md` entry and `GLOSSARY.md` additions.

You must **not** edit `src/core/retrainer/` modules, `src/execution/`, existing lab modules other than the ones above, any existing test, or any production config. You may **read** every module in `src/lab/` and `src/core/retrainer/` freely — the contract signatures you must use are pinned below and were verified 2026-09-24 against the actual files.

**Soak guard:** identical to Lanes 1–2; the live soak is in the sibling checkout.

## 2. Context & Problem Statement

Single-instrument directional bracket trading on M15 spot forex is measurably closed (2026-09-14 session + 2026-09-20 reprice + 2026-09-21 artifact replay + 2026-09-23 CatBoost A/B). Every one of the five high-benefit fixes in `audits/2026-09-08_high-benefit-fixes-ranked.md` is now in the code; the market is still a driftless martingale at the served 2.0×/4.0× geometry, and the wide-bracket reprice (2026-09-20) showed even the certified top cohort of the live bar nets **−0.516R/trade**. The last untested cost-side hypothesis (spread-table asymmetry) is closed by the 2026-09-21 lab run.

What remains is the **model structure**: the production target is a binary survival label with ~25% base rate, and the output probability distribution compresses toward that base rate, producing 0.3833-bar approval rates of 0.028% and zero fills. This lane replaces the single-instrument survival bet with a **cross-sectional relative-value bet** — "will currency X outperform the basket median over horizon τ" — which by construction is a 50/50 label and therefore cannot compress.

If this gate fails too, the honest conclusion is that the M15 spot forex axis is structurally unprofitable for this account, and the deliverable becomes a formal decommissioning report.

## 3. Execution & Platform Constraints

- **Venue:** OANDA forex (research only, no orders).
- **Existing bar inventory (verified 2026-09-24):**
  - Tradeable fiat basket (6 pairs): `AUD_JPY, EUR_JPY, GBP_JPY, NZD_JPY, GBP_AUD, GBP_NZD` — matches `DEFAULT_TRADEABLE_6` in `src/lab/spec.py:68` and the keys of `config/spread_alphas_m15.json`.
  - Full training basket (8 incl. metals): `XAU_USD, XAG_USD` added; metals are in-training but untradeable (`_common.py:184-188`, `UNTRADEABLE_SYMBOLS`).
  - Canonical M15 cache: `analysis_cache/strategy_matrix/{sym}_M15.parquet`, ~49,600 rows per symbol, 2024-09-08 → 2026-09-08, 6 columns `[timestamp, open, high, low, close, volume]`, no `symbol` column (added at load).
  - A second, shorter cache: `data/cache/ab_catboost/{sym}_M15_60d_20260908.parquet` (8 symbols incl. metals, 60 days only).
- **USD/JPY funding-index mandate — cache check first.** The 6-pair basket contains **no USD pairs**. Isolate-the-USD/JPY-leg requires either `*_USD` crosses (EUR/USD, GBP/USD, AUD/USD, NZD/USD, USD/JPY, USD/CHF, USD/CAD) or the synthetic USD index implied by the pairs. Two options, in order:
  1. **Try the full 7-cross USD basket.** `fetch_training_data` in `src/core/retrainer/_data.py:28` fetches live from the provider; you may use `OandaMarketProvider` (the same one `scripts/reprice_band_geometry.py:102` uses on cache miss) to fetch the 7 USD crosses at M15 for the same 730-day window and cache them to `analysis_cache/strategy_matrix/USD_CROSS_M15.parquet`. Atomic write. Check account permission first — the practice account may or may not expose every cross; if a cross 404s, drop it (record which) and fall to option 2 with the survivors.
  2. **Fallback:** construct synthetic currency indexes from the existing 6-pair cache only (AUD, EUR, GBP, JPY, NZD). Document clearly that USD cannot be isolated.
- **Cost model:** the toll floor is the measured 0.25R at the served 2×/4× geometry. Report edge-over-random **net** of it (see §6).

## 4. Mathematical & Algorithmic Formulation

### 4.1 Synthetic geometric-mean currency index
Define for each currency `c ∈ {AUD, EUR, GBP, JPY, NZD [, USD]}` a basket of `K_c` pairs where currency `c` appears (e.g. `GBP_JPY`, `GBP_AUD`, `GBP_NZD` for GBP; `AUD_JPY`/`EUR_JPY`/`NZD_JPY` for JPY):

`I_{c,t} = (∏_{j=1..K_c} S_{pair_j,t})^{1/K_c}`, where `S` is the price of a pair that has currency `c` on exactly one side. If a pair has `c` on the quote side, use `1/S`. Log transform for RV math: `L_{c,t} = ln I_{c,t}`.

Consistency requirement: the pair universe must be **closed** — every currency in the pair list must appear at least twice (so each currency's index has ≥2 pairs). The 6-pair fiat basket satisfies this (AUD/EUR/GBP/JPY/NZD each appear in ≥2 pairs); adding the 7 USD crosses keeps it closed. **Pin the pair-universe closure by test.**

### 4.2 RV forward-return target (the deliberate 50% balance)
Forward log change over horizon τ bars:
`ΔL_{c,t→t+τ} = L_{c,t+τ} − L_{c,t}`
At each timestamp `t`, cross-sectional median over currencies: `m_t = Median({ΔL_{k,t→t+τ}})`.
Label: `y_{c,t} = I( ΔL_{c,t→t+τ} > m_t )`.
By construction each timestamp's cross-section has **exactly ⌈K/2⌉ positives** (up to ties — break ties deterministically by currency code). **Pin the 50% rate by test.**

### 4.3 Features
Use the existing V3 feature stack as the base: register a new family `"cs_rv"` in `src/lab/registry.py` (decorator `register_feature(name="cs_rv", columns=..., version=1)`) whose generator computes, per currency (not per pair): RV features on the indexes (momentum, vol, RSI on `I_{c,t}` and `L`, cross-sectional ranks, days since index regime shift) **plus** carries along the pair-level V3BaseFeatures for the pair that most directly expresses the currency's RV signal (e.g. GBP index uses `GBP_JPY` pair features as carrier). Document the carrier mapping in code.

You must also define a `FeatureSpec` (frozen dataclass, `src/lab/spec.py:135-233`) named `forex_cs_rv`:
- `symbols = DEFAULT_TRADEABLE_6` (or expanded with USD crosses if §3 fetched them)
- `granularity = 15`, `days_back = 730`
- `feature_sets = ("v3_base", "cs_rv")`
- `geometry = GeometrySpec(sl_mult=2.0, tp_mult=4.0, max_hold=45)` — keep the production geometry for comparability with every prior M15 report
- `label = LabelSpec(kind="survival")` (the label is already balanced; do not try the macro variant)
- `gate = GateConfig(model_family="lightgbm", n_folds=3)`

### 4.4 Gate path (exact call)
Use `lab.frames.build_frame(spec, bars)` → `lab.gate.run_gate(frame, spec, n_folds=spec.gate.n_folds)` → `lab.backtest.run_model_backtest(frame, gate, spec)`. These signatures are pinned by `tests/test_lab_*` and are the supported path — do not re-implement a gate.

## 5. Data Ingestion & Feature Engineering Spec

- **Bars:** load `analysis_cache/strategy_matrix/{sym}_M15.parquet` for the 6 fiat pairs (and `USD_CROSS_M15.parquet` if option 1 of §3 succeeded). Concatenate with `symbol` column.
- **Index construction:** pivot closes to a `timestamp × currency` matrix using the formulas of §4.1; handle missing bars by forward-fill with a 1-bar limit then drop the row.
- **Leakage guards:**
  - Every rolling feature and the index must be computed on bars up to and including `t` only.
  - The forward return at `t` uses bars `> t` — assert this by test.
  - The cross-sectional median at `t` is computed across currencies **within the same `t`** only (never across time).
  - `apply_labels_and_veto` (from `core/retrainer/_features.py:139-150`) must still be called for the chop veto and tail purge so the frame is comparable to every prior lab run. The survival target it emits is *not* your training label; you will train a classifier against `y_{c,t}` instead. Keep both columns in the frame: production column for the veto path, RV column for training.
- **Cache key:** with W1 merged, your `cs_rv` family must have `version=1` and any internal constant changes bump the version — follow `_require_version` semantics exactly. `FeatureSpec.content_hash` will then change when the family changes; rely on it, do not bypass it.

## 6. Mandatory Statistical Falsification Suite

### 6.1 Primary gate (decisive)
- **Edge-over-random > +0.10R net of the 0.25R toll**, measured on the validation holdout and on `run_model_backtest`'s pooled result.
- **Unit warning — read carefully.** `lab` reports `ValidationReport.edge_over_random` in **win-rate (percentage-point) units**, not R (`_types.py:103-108`, `ablate.py:150-152`). To express it in R for comparison against the 0.25R toll you must convert. At the served 2×/4× geometry: `EV_R ≈ p × 4.0 − (1−p) × 2.0 − toll_R` per trade where `p` is the win rate. A base rate of `p0 = 0.2540` (measured, M15 fiat, `recons/2026-09-14_session-evidence-and-options.md`) nets `−0.09R`. Define `edge_R = (p_model − p0) × (4.0 − toll_R_units) − 1×toll_R` — at 2:1, `1 pp ≈ 0.03R`. State the conversion explicitly in the report. The gate fires only if `edge_R > +0.10R` **after** subtracting 0.25R.
- Use the existing `lab.ablate.delta_with_ci` for the family-level CI (Clopper-Pearson, quadrature, `[−1,1]` clip, `lab/ablate.py:141-197`) — do not re-implement.

### 6.2 Universal stats contract
Same `src/lab/stats.py` contract as Lanes 1–2 (DSR, CSCV PBO, HLZ). If Lane 1 or 2 has already landed it in your worktree, import it; otherwise build it identically. Count trials = every distinct (feature set, geometry, horizon τ) evaluated.

### 6.3 The decommissioning branch (mandatory if gate fails)
If the primary gate fails, write `llm_reports/stops/2026-09-24_forex-cs-rv-decommission.md` following the `stops/` convention (`llm_reports/README.md:15-29` — "blocked by a dependency/risk"; body: `## Context`, `## What was tried`, `## Why it failed`, `## What's needed to unblock`, `## Recommendation`). Frame it honestly: this lane's failure, together with the bracket closure, the wide-bracket reprice, the served-artifact replay, and the CatBoost A/B, is the fourth independent falsification and the recommendation should be to **freeze the M15 forex lane** (keep the soak running as a feed monitor, never serve a new M15 bracket model) rather than to try a fifth geometry.

### 6.4 Report skeleton (recon path)
`llm_reports/recons/2026-09-24_lab-forex-cs-rv.md`, recons template, sections: Context → Data (pair universe incl. USD cross availability) → Method (index math, 50% label, feature family, geometry) → Results (per-fold + pooled win rate, edge_over_random in both units, edge_R after toll, DSR/HLZ/PBO, trial count, CI) → Falsification verdict → Files touched.

### 6.5 Deterministic test cases (mandatory)
1. **Index consistency:** with a synthetic 3-currency closed pair universe (e.g. A/B, A/C, B/C with fabricated prices), the three `I_c` indexes reproduce exactly when recomputed from scratch; the cross-product identity `I_A/I_B = S_{A/B}` holds to 1e-12.
2. **50% label balance:** over 1000 synthetic timestamps with 5 currencies, the positive rate per timestamp is exactly 0.5 up to ties.
3. **Forward-return leak guard:** computing `ΔL` at `t` must not change if you truncate the frame after `t`.
4. **Closure guard:** a pair universe that leaves a currency with only one pair raises.
5. **Tie-breaking determinism:** same input median ties always resolve identically.
6. **Stats module:** if `src/lab/stats.py` was built in your worktree, same contract tests as Lane 1 §6.4 #6.

### 6.6 Abort criteria
- USD-cross fetch fails with permission/404 for > half the requested crosses → document the reduced universe and proceed on the 6-pair baseline.
- The chop veto + tail purge path (`build_frame`) errors on the new family — do not patch around it; the failure itself is a falsification signal, report it.
- Any leak-guard test fails after fixes.

## 7. Docs rule
Update `src/lab/README.md` with the `forex_cs_rv.py` entry (imports/imports-by/reads-writes) and add `GLOSSARY.md` entries for "cross-sectional relative value", "currency index", "funding leg", "50%-balanced label".

## 8. What "done" looks like
- Branch `lane/forex-cs-rv` holds only your files.
- Tests green, compileall green.
- Recon file exists with the honest gate verdict; if failed, the decommissioning stop-file also exists.
- Commit `feat(lab): lane3 forex cross-sectional relative-value basket [falsification gate]`.
