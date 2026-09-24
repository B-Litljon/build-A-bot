---
type: recon
date: 2026-09-24
time: 17:10 PDT
agent: dsh
model: ollama-cloud/deepseek-v4.1-flash
trigger: Lane 1 dispatch — daily crypto trend momentum + volatility targeting, falsification gate
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
related:
  - llm_reports/recons/2026-09-14_session-evidence-and-options.md
  - llm_reports/handoffs/2026-09-24_lane1-crypto-trend-mom-tv.md
files_touched:
  - src/lab/momentum_crypto.py
  - src/lab/stats.py
  - src/lab/__init__.py
  - tests/test_lab_momentum_crypto.py
  - tests/test_lab_stats.py
  - src/lab/README.md
  - GLOSSARY.md
---

# Lane 1: crypto trend momentum + vol targeting (2021-01-01 → 2026-09-24)

## Context

The forex M15 axis measured out as a driftless martingale whose bracket toll
(~0.25R at the served 2.0×/4.0×) destroys all gross expectancy
(`recons/2026-09-14_session-evidence-and-options.md`). The same session closed
the crypto *bracket* axis at random entries (0/27 D1 and 0/27 H4 geometries
positive) but measured a D1 toll of only **0.0497R** — an order of magnitude
cheaper than forex — against BTC buy-and-hold of +283% over that study window.
Lane 1 therefore tests the one strategy class that does not depend on bracket
geometry at all: **time-series momentum with volatility targeting** on daily
BTC/USD + ETH/USD spot, holding for weeks.

Question being answered: does a median-of-four momentum sign ensemble with
inverse-vol sizing and an honest cost model survive a full multiple-testing
falsification battery (DSR, CSCV PBO, HLZ) on Alpaca daily crypto?

**Verdict up front: GATE FAIL — PBO 0.93 ≥ 0.50.** The strategy is strongly
profitable (net CAGR 48.8% vs BTC B&H 20.1%, MaxDD 14.3% vs 76.7%), but the
parameter selection behind that number does not transfer out-of-sample: in
65 of the 70 CSCV splits, the in-sample best arm ranked at or below the
test-half median. Everything else passed; this one bar is decisive.

## Investigation

Built per the dispatch brief
(`handoffs/2026-09-24_lane1-crypto-trend-mom-tv.md`). The 2021 probe
succeeded on the first try — Alpaca's free-tier crypto feed serves 2021-01-01
onward for both symbols, so **no window degradation was needed** and the gate
is evaluated over the full aimed window.

One credential wrinkle, recorded so the next agent does not re-derive it: the
shell env-var interpolation path (`ALPACA_API_KEY=$(grep ... .env)`) produced
nginx 401s, while `set -a; . ./.env` against the same file works. Values were
never printed. No orders are submitted anywhere in this lane.

New code: `src/lab/momentum_crypto.py` (the harness) and `src/lab/stats.py`
(the shared falsification statistics — first DSR/PBO/HLZ implementation in
the repo; Lanes 2–5 import these signatures, which is why they are pinned
verbatim from the brief and covered by contract tests).

## Findings / Changes

### Data & window

- **Realized window: 2021-01-01 → 2026-09-24, 2,093 common trading days**
  (2,093 bars per symbol after strict same-day inner join; 4,186 cached rows).
- Source: Alpaca spot crypto via `AlpacaProvider.get_historical_bars(symbol,
  1440, …)` only. No direct REST calls.
- Cache: `analysis_cache/lab_frames/crypto_daily_bars.parquet` —
  7 declared bar columns (`timestamp, symbol, open, high, low, close, volume`)
  in order plus a `fetched_at_utc` stamp, written atomically (temp +
  `os.replace`). `--refresh-bars` refetches; otherwise re-runs are cache-only.
- Degradation from the 2021 target: **none** — the probe floor is the target
  start itself.

### Method (all formulas per brief §4; two open choices pinned)

- Momentum: `r_{k,i,t} = ln(P_t / P_{t-k})`, k ∈ {21, 63, 126, 252} days.
- Vote: **median of the four sign values** (the brief's primary option),
  snapped to {-1, 0, +1} with |median| < 0.5 → 0. A 2-2 tie is exactly 0 =
  no position. Long-only: votes are clipped at `max(0, S)`.
- Sizing: `w* = (σ_target / σ̂) · max(0, S)`, σ̂ = std(daily log returns,
  last N_σ days, ddof=1) × √365; clip [0,1]; normalize if Σw* > 1; cash earns
  0%. **Default N_σ = 60**, with 20 swept (both reported).
- Rebalance buffer δ = 0.05 on the target weight, strict `>` — fires when the
  weight move exceeds 5 percentage points.
- Execution: signal at close *t*, fill at **open t+1**; daily portfolio return
  uses open-to-open asset returns. Day 0 is flat. This is the pinned leak
  guard; the shift test asserts day-199 signals are identical with and
  without the future appended.
- Costs: `cost_t = Σ|Δw| × (33 + 25) / 10_000` — 33 bps amortized half-spread
  + 25 bps taker per unit of one-way turnover, applied on every weight change.
- σ_target ∈ {0.20, 0.25, 0.30}, N_σ ∈ {20, 60} → **6 arms = 6 trials**
  counted for DSR/HLZ (momentum lookbacks are fixed, not trialed).

### Results

Best arm by net Calmar: **N60–T0.20**. Best arm by *net CAGR* is N60–T0.30
(0.6287); the Calmar-best and CAGR-best coincide only loosely, which is
itself part of the PBO story below.

| metric | Lane 1 net | Lane 1 gross | BTC B&H | ETH B&H |
|---|---|---|---|---|
| Total return | **+875.1%** | +1440.6% | +186.3% | +267.3% |
| CAGR | +48.79% | +61.15% | +20.14% | +25.48% |
| Max drawdown | **14.30%** | 14.21% | 76.68% | 79.31% |
| Calmar | **3.413** | 4.302 | 0.263 | 0.321 |
| Sharpe (√365) | 1.773 | 2.091 | 0.320 | 0.293 |

Net NAV path (N60–T0.20): EOY2021 1.631 → EOY2022 1.679 → EOY2023 3.130 →
EOY2024 5.918 → EOY2025 8.618 → 2026-09-24 9.751. Gross–net drag is 5.66 NAV
units on 79.4 units of cumulative turnover across 227 rebalance days (of
2,093). Average gross exposure 0.327; in-market 51.9% of days.

Per-arm net results:

| arm | total ret | CAGR | MaxDD | Calmar | Sharpe |
|---|---|---|---|---|---|
| N20–T0.20 | +856.5% | 48.29% | 15.02% | 3.215 | 1.753 |
| N20–T0.25 | +1273.2% | 57.95% | 19.01% | 3.047 | 1.734 |
| N20–T0.30 | +1567.1% | 63.38% | 22.30% | 2.843 | 1.694 |
| **N60–T0.20** | **+875.1%** | **48.79%** | **14.30%** | **3.413** | **1.773** |
| N60–T0.25 | +1314.0% | 58.75% | 18.12% | 3.243 | 1.764 |
| N60–T0.30 | +1537.7% | 62.87% | 20.05% | 3.136 | 1.700 |

### Falsification battery (n_trials = 6, n_obs = 2,092)

- **DSR = 1.0000** — with Sharpe 1.77 on 2,092 daily observations (skew
  +1.04, kurt 10.80), far above the E[max] of 6 null trials. Pass (> 0.95).
- **CSCV PBO = 0.9286** — in 65 of 70 eight-block train/test splits, the
  arm that won in-sample ranked at or below the test-half median. **Fail
  (< 0.50 required).** With N = 6 highly correlated arms this is close to
  the statistic's ceiling behavior — the arms share one signal and differ
  only in sizing knobs, so any regime rotation that flips which knob value
  wins in-sample flips the test rank. The bar the brief sets does not care
  about the excuse, and neither does this section.
- **HLZ haircut Sharpe = 0.390 (from 1.773), HLZ t = 17.81** — the
  unit-information haircut at 6 trials. Pass (> 3.0).
- Primary gate: Calmar 3.413 > BTC B&H 0.263 (pass); MaxDD 14.3% < 30%
  (pass).

### Falsification verdict

**GATE FAIL — PBO 0.93 ≥ 0.50.** The strategy is genuinely profitable over
the full 2021→2026 window and beats buy-and-hold on every risk-adjusted
measure, but the arm-selection procedure does not survive combinatorial
cross-validation: which (N_σ, σ_target) pair leads is regime-dependent, so
the headline number is partly a selection artifact. Per the brief: not
softened, and the pass on the other four bars does not change the verdict.

## Verification

```
PYTHONPATH=src:. python -m pytest tests/test_lab_stats.py tests/test_lab_momentum_crypto.py -q
# 34 passed in 1.08s
```

Full suite: **646 passed, 17 subtests** (baseline before this lane: 612
passed / 17 subtests — the delta is exactly the 34 new tests).
`python -m compileall -q src/` clean. All reported numbers recompute from the
cached parquet via `python -m lab.momentum_crypto --json` (cache hit; the
rerun in this session reproduced every figure above to the last digit).

Deterministic cases from the brief §6.4 are all in
`tests/test_lab_momentum_crypto.py`: signal causality at day 199 (§6.4.1),
buffer hold/fire regions (§6.4.2 — see the test's note: with buffer 0.05 and
w_prev 0.50, the literal boundary fires because 0.55−0.50 ≈ 0.0500000000007
in double precision; the test pins the strict-inside/strict-outside contract
and documents the boundary resolution), exact cost `0.5 × 58/10_000` on a
forced weight change (§6.4.3), byte-identical re-save of the cache (§6.4.4),
and the 7-column schema order (§6.4.5).

## Risk & follow-ups

- **The gate fails, and the failure is about selection, not the market.** A
  fixed parametrization chosen *a priori* (e.g. N60–T0.25 without looking at
  results) would carry no PBO problem by construction — but that decision
  belongs to a fresh lane with its own gate, not retro-fitted here.
- PBO with N = 6 near-identical arms is a coarse instrument; if a later lane
  runs a genuinely parametrized grid (k-lookbacks varied), this same
  `lab.stats.cscv_pbo` will separate better.
- The harness is long-only/cash; the 2021 and 2024 drawdown windows were
  survived by sitting flat (51.9% in-market). Extending the basket would
  exercise the timestamp-join guard (`_pivot_daily` inner-joins per asset)
  — do not relax it to a forward-fill without a leak review.
- `lab/stats.py` HLZ form: the two-argument contract pins the
  unit-information benchmark (haircut = z × 1). If a later lane wants the
  sampling-variance-weighted form, add a keyword with a default rather than
  changing the signature.
- The cost model treats the 33/25 bps as exhaustive; no slippage-vs-open
  modeling. Fill-at-open is the simplification — state it when citing.

## Files touched

- `src/lab/momentum_crypto.py` (new, ~620 lines) — harness.
- `src/lab/stats.py` (new, ~250 lines) — DSR / CSCV PBO / HLZ + `hlz_t_stat`.
- `src/lab/__init__.py` — lazy facade entries for the three stats functions.
- `tests/test_lab_momentum_crypto.py` (new) — 17 tests.
- `tests/test_lab_stats.py` (new) — 17 tests.
- `src/lab/README.md` — file entries for both new modules.
- `GLOSSARY.md` — "crypto trend mom tv" + stats terms under The feature lab.
- Read: `llm_reports/recons/2026-09-14_session-evidence-and-options.md`,
  `src/data/alpaca_provider.py`, the dispatch handoff, `llm_reports/README.md`.
