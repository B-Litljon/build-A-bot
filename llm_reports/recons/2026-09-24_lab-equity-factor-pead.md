---
type: recon
date: 2026-09-24
time: 16:10 PDT
agent: opencode (kimi-k3)
model: ollama-cloud/kimi-k3
trigger: Lane 2 dispatch — weekly equity factor rebase + PEAD falsification gate
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
files_touched:
  - src/lab/stats.py
  - src/lab/__init__.py
  - scripts/investor_feature_pipeline_weekly.py
  - scripts/investor_train_model_weekly.py
  - scripts/portfolio_orchestrator_weekly.py
  - tests/test_lab_stats.py
  - tests/test_investor_weekly.py
  - scripts/README.md
  - GLOSSARY.md
  - llm_reports/m2m-prompts/2026-09-24_quant-lanes.md
related:
  - llm_reports/handoffs/2026-09-24_lane2-equity-factor-pead.md
  - llm_reports/recons/2026-09-14_session-evidence-and-options.md
---

# Lane 2 — weekly equity factor rebase + PEAD

## Context

The production V4 investor lane is a **monthly** LightGBM LambdaRank ranker over a
96-ticker universe with a 60-day forward target, a 60-day embargo, and a
benchmark gate that compares the deployed equal-weight top-8 basket against
equal-weighting the whole universe. This lane asks the falsification question the
brief poses: **does the signal hold at a 5-trading-day (weekly) horizon, and does
adding Post-Earnings Announcement Drift (PEAD) improve it enough to clear
stricter multiple-testing hurdles (DSR > 0.95, HLZ t > 3.0, CSCV-PBO < 0.50,
excess ≥ +80 bps/month)?**

This sits against the standing finding of
[`2026-09-14_session-evidence-and-options.md`](2026-09-14_session-evidence-and-options.md):
the forex lanes measure an edge of ≈ +0.045R against a ≈0.09R requirement — a
third of what is needed, with every cost-side lever closed. The equity lane is a
*different* market and a *different* label, so the question was legitimately open
going in.

Two up-front facts that shaped the work:

1. **The PEAD leg is NOT data-blocked.** The SimFin quarterly caches ship a fully
   populated `Publish Date` column (0 nulls in 45,936 income rows, verified
   2026-09-24: `Publish Date | nulls: 0 ; sample 2020-12-18`; publish lags vs
   fiscal-quarter `Report Date` 1–63 days, median 33, no negative lags). Per the
   brief §3 this is source #2 made available, so **no announcement-date
   reconstruction was attempted** and the §6.5 abort is **not** triggered.
2. **`src/lab/stats.py` did not exist** in this worktree (Lane 1 had not landed
   it — verified `find` returned nothing across all five sibling lane
   worktrees). Per the brief §6.2 I implemented it from Lane 1's exact pinned
   contract and recorded both the build and my consumption of it in the
   `m2m-prompts/2026-09-24_quant-lanes.md` thread.

## Data & window

- **Prices:** `data/raw/v4_investor_data.parquet` — 120,384 rows, 96 symbols,
  2021-09-01 → 2026-08-31, 1,254 unique trading days (read-only; symlinked into
  this worktree's git-ignored `data/` from the live checkout — the live soak's
  branch, service, and files were never modified).
- **Fundamentals / earnings:** SimFin `us-income-quarterly.csv` +
  `us-balance-quarterly.csv`. Only **72 of the 96** universe tickers have income
  rows (1,271 filings); 66 have ≥9 quarters, enough for the 8-quarter SUE window.
  EPS = `Net Income (Common) / Shares (Diluted)`. This sparsity is a real
  limitation: SUE is NaN for the 24 uncovered names (LightGBM-native).
- **Materialized artifact:** `data/processed/earnings_calendar.parquet`
  (atomic), 1,271 rows × 5 cols `[symbol, earnings_date, eps_reported,
  publish_date, source]`, publish range 2020-11-17 → 2025-10-08.

## Method

**Features (15 present).** The 13 cross-sectional factors the production frame
actually yields — momentum 3m/6m/12m/12-1m, reversal 1m, vol 60d/120d, ROA, D/E,
gross profitability, gross/operating/net margins — each rank-normalized per ISO
week (`ebitda_margin` is absent: the raw parquet carries no `EBITDA` column, and
production likewise trained without it). **Plus the two PEAD additions, kept on
their raw scale so the ranker can read surprise magnitude:** `sue` (seasonal
random-walk standardized unexpected earnings, 8-quarter trailing window, min 4)
and `days_since_earnings` (trading days since the most recent `publish_date`).
`days_until_earnings` is a **prefilter only** (drop rows with `0 ≤ d ≤ 2`), never
a model input.

**Leak guards, all asserted and tested:**
- **Point-in-time:** every fundamental consumed at date `t` has `publish_date ≤
  t` — the asof join enforces it and a future-dated match raises. SimFin's
  `Restated Date` and all TTM/derived columns are excluded from PEAD (not
  point-in-time provable). Tested with a fake `publish_date = t+5`.
- **Target:** `fwd_log_ret_5d = ln(close_{t+5}/close_t)`, verified per-row to
  equal `ln` of the close 5 trading days ahead.
- **Pre-announcement:** rows in the `0 ≤ days_until_earnings ≤ 2` band flagged
  and dropped before grouping (3,201 rows in this window); a row with
  `days_since_earnings < 0` would be counted and dropped (none occurred).
- **Target embargo:** the final 5 trading days per symbol carry NaN labels and
  are dropped in training mode (480 rows).

**CV:** purged group time-series CV over ISO-week query groups (`lambdarank`
`group` = rows/week). Expanding train; **atomic 10-CALENDAR-day embargo** between
train and validate, asserted per fold as `train.max(date) + 10d < val.min(date)`
(no rows in the gap). `n_folds = 5`. Model: LightGBM `lambdarank`, `ndcg_at=[5,10]`,
max_depth 6, num_leaves 31, lr 0.05, feature_fraction 0.9, `deterministic=True`,
`random_state=42`.

**Trial count:** `_N_TRIALS = 1` — exactly one (feature set, embargo length,
hyperparameter tuple) was fitted for the gate. Multiple-testing burden reported
honestly; I did not grid-search DSR/HLZ into significance.

## Results

Per-fold (rank-normalized 13 base + PEAD, 5-day label):

| fold | train ≤ | val ≥ | train wks | val wks | Spearman IC | val rows |
|---|---|---|---|---|---|---|
| 1 | 2022-05-20 | 2022-05-31 | 38 | 36 | +0.0056 | 16,050 |
| 2 | 2023-02-03 | 2023-02-21 | 75 | 35 | +0.0201 | 15,714 |
| 3 | 2023-10-20 | 2023-11-06 | 112 | 35 | +0.0256 | 15,486 |
| 4 | 2024-07-05 | 2024-07-22 | 149 | 35 | +0.0289 | 15,556 |
| 5 | 2025-03-21 | 2025-04-07 | 186 | 73 | +0.0579 | 32,899 |

Gate metrics (equal-weight top-8 basket, sector-cap 2, weekly rebalance vs the
equal-weight universe; excess annualized √21, in bps/month):

| metric | value | bar | verdict |
|---|---|---|---|
| excess / month | **+190.0 bps** | ≥ +80 | PASS |
| DSR | **1.0000** | > 0.95 | PASS |
| HLZ t | **+19.09** | > 3.0 | PASS (adj SR 3.677, unadj t 3.39) |
| PBO (CSCV) | **0.300** | < 0.50 | PASS |

Distribution of the 214 weekly excess observations: mean +45.5 bps, median +19.7
bps, 56% of weeks positive, std 187 bps. The mean is **not** an outlier artifact:
a 5%-trimmed mean is +40.7 bps/week with t = 4.03 (robustness *rises* under
trimming, since the big weekly swings are symmetric).

**The PEAD ablation is the informative part:**

| configuration | excess (bps/mo) | mean IC | t |
|---|---|---|---|
| base 13 only | +184.1 | 0.0371 | 3.10 |
| **base + PEAD** | **+208.3** | **0.0270** | **3.56** |
| **PEAD only** | **+18.6** | **0.0036** | **0.49** |

## Falsification verdict

**GATE PASS on the aggregate config — but the edge is the base factors re-timed
to a 5-day hold, NOT the PEAD signal, and it must not be read as ranking skill.**
Per-fold Spearman IC is 0.006–0.058 (mean 0.028) — per-name ordering is near
noise. PEAD/SUE is **inert here**: PEAD-only excess is +18.6 bps/month at t=0.49
(statistically zero) and on only 52.9% of rows; adding it to the base lifts
excess +184 → +208 bps/month and t 3.10 → 3.56 while *lowering* IC 0.037 →
0.027. The +80 bps/month bar clears because a diversified top-8 momentum/quality
basket rebalanced weekly happens to beat the equal-weight universe in this
2021–2026 window, robust to trimming — a finding worth recording, but it is a
low-skill, high-variance basket effect, not a promotable ranking model, and PEAD
(the anomaly this lane was built to test) does not carry it. Recommendation: **do
not promote a live weekly lane on this evidence**; the verdict that matters for
the research question is "PEAD does not rescue a weekly rebase."

## Files touched

- `src/lab/stats.py` (new — DSR / CSCV-PBO / HLZ, shared with Lanes 1–5) and a
  lazy-export block in `src/lab/__init__.py`.
- `scripts/investor_feature_pipeline_weekly.py` (new — features + earnings
  calendar + PIT guard + prefilter).
- `scripts/investor_train_model_weekly.py` (new — purged group CV + falsification
  gate; writes `metrics.json` always, model artifact on PASS).
- `scripts/portfolio_orchestrator_weekly.py` (new — research-only weekly ranker;
  dry-run by default, refuses without a passing-gate model).
- `tests/test_lab_stats.py` (16 tests), `tests/test_investor_weekly.py` (10 —
  all six mandatory leak/determinism cases).
- `scripts/README.md`, `GLOSSARY.md`, `llm_reports/m2m-prompts/2026-09-24_quant-lanes.md`.

Verification (from this worktree): `pytest -q` → **642 passed / 17 subtests /
0 errors** (the 2 `test_ml_strategy_guards.py` setup errors were pre-existing —
they load the live, git-ignored `models/forex_m15_wide/*.pkl`; resolved in this
worktree by read-only symlinks to the sibling checkout, which was otherwise
untouched). `python -m compileall -q src/ scripts/` → clean.
