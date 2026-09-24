---
type: handoff
date: 2026-09-24
time: 15:46 PDT
agent: dsh (DeepSeek V4.1 Flash via Ollama Cloud)
model: ollama-cloud/deepseek-v4.1-flash
trigger: Lane 2 dispatch — refactor equity ranker to weekly cadence with PEAD
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
files_touched:
  - scripts/portfolio_orchestrator_weekly.py
  - scripts/investor_feature_pipeline_weekly.py
  - scripts/investor_train_model_weekly.py
  - src/lab/stats.py
  - tests/test_lab_stats.py
  - tests/test_investor_weekly.py
  - llm_reports/recons/2026-09-24_lab-equity-factor-pead.md
related:
  - scripts/portfolio_orchestrator.py (existing monthly lane, 96-ticker universe, 14-feature LambdaRank model)
  - scripts/investor_train_model.py (existing walk-forward, TRAIN_DAYS=504, EMBARGO_DAYS=60)
  - llm_reports/recons/2026-07-03_1941.md (installed crontab evidence)
---

## 1. System Persona & Scope

You are a **quantitative researcher** and **execution engineer** building a **research artifact** for a weekly equity factor strategy. Your scope is **offline backtest + falsification only**. The existing monthly V4 lane (`scripts/portfolio_orchestrator.py`, `run_investor_rebalance.sh`, the live crontab at `30 16 1 * *`) is **production and untouched**. You will create *parallel* weekly variants, never modify the working monthly path.

**Scope lock — you may only touch:**
- `scripts/portfolio_orchestrator_weekly.py` (new)
- `scripts/investor_feature_pipeline_weekly.py` (new)
- `scripts/investor_train_model_weekly.py` (new)
- `src/lab/stats.py` (new — the DSR/CSCV/HLZ module, shared with Lane 1; implement only if not already present from Lane 1, otherwise import it)
- `tests/test_investor_weekly.py`, `tests/test_lab_stats.py` (new)
- One recon report `llm_reports/recons/2026-09-24_lab-equity-factor-pead.md`
- `scripts/README.md` entry if a README exists in `scripts/`, `GLOSSARY.md` additions.

You must **not** edit `scripts/portfolio_orchestrator.py`, `scripts/investor_feature_pipeline.py`, `scripts/investor_train_model.py`, `run_investor_rebalance.sh`, any crontab, `src/execution/`, or any file inside `src/core/retrainer/`. Treating the existing production lane as read-only is non-negotiable.

**Soak guard:** same as Lane 1 — soak is live on `main` in the sibling checkout; do not switch its branch or restart its service. All work happens in your branch/worktree at `/mnt/storage/mystuf/development/build-a-bot-lanes/equity-factor-pead`.

**Test commands:**
```bash
cd /mnt/storage/mystuf/development/build-a-bot-lanes/equity-factor-pead
PYTHONPATH=src:. /home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python -m pytest -q
/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python -m compileall -q src/
```

## 2. Context & Problem Statement

The existing V4 investor lane is a **monthly** cross-sectional LightGBM LambdaRank ranker over a 96-ticker universe (11 GICS sectors). It selects top-K, executes on the 1st of the month, and uses fundamentals loaded from SimFin quarterly income/balance caches (`data/raw/simfin_cache/`). Its walk-forward (`investor_train_model.py:139-141`) has `TRAIN_DAYS=504`, `EMBARGO_DAYS=60`, `TEST_DAYS=60`, `n_folds=10`. The model artifact on disk (`models/v4_investor_lgbm.txt`, 100 trees, 14 features) was trained 2026-07-04 and carries `gate_passed: true`, but the 2026-08-03 sidecar update lacked a `benchmark` section — meaning its promotability evidence is stale.

This lane asks whether the signal holds at a **5-to-10-day horizon** and whether adding **Post-Earnings Announcement Drift (PEAD)** — a known, mechanically exploitable anomaly — improves it enough to clear stricter multiple-testing hurdles than the original.

**The operational bottleneck:** the current pipeline's fundamentals are quarterly point-in-time (revenue, margins, ROA, D/E). There is **no estimates dataset, no announcement-date column, no surprise computation, and no publish-date discipline** anywhere in the repo. To do PEAD honestly you must (a) source or derive an announcement calendar with release timestamps and reported EPS, (b) avoid look-ahead from SimFin's own `TTM` and derived ratios (which are not point-in-time safe without filtering), and (c) re-validate cross-validation with a true **temporal embargo**, not the current day-count embargo.

## 3. Execution & Platform Constraints

- **Venue:** Alpaca equities, paper, long-only, no margin, no shorting (existing `.env` `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`; do not print).
- **Data path for prices:** the existing miner uses `yfinance` (`investor_data_miner.py:17,102`) — **not the Alpaca provider**. Keep that convention for the weekly refactor. Do not inject Alpaca market data; Alpaca is for orders only in the parent lane.
- **Fundamentals source:** SimFin via `SimFinFundamentalProvider` (`src/data/providers/simfin_fundamentals.py:149`) — quarterly income + balance caches **already populated** at `data/raw/simfin_cache/us-income-quarterly.csv` (45,936 rows), `us-balance-quarterly.csv` (45,936 rows), plus banks/insurance variants. Read-only access; do not re-download unless `--refresh-fundamentals` is passed.
- **PEAD/SUE source — the critical piece:**
  - SimFin does **not** ship consensus estimates. You have three acceptable sources, in order of preference:
    1. **Derive SUE from actuals:** `SUE_q = (EPS_q − E[EPS_q]) / std(EPS_q − E[EPS_q])` where the expectation is a simple time-series forecast on the trailing 8 quarters (e.g. seasonal random walk or 4-qtr moving average). This is **point-in-time safe** if you only use quarters whose publish date ≤ the signal date. This is the recommended fallback because it requires nothing external.
    2. **SimFin 'announcement' metadata:** inspect `data/raw/simfin_cache/*.csv` headers — SimFin bulk files sometimes include a `publishDate` or `announcementDate` column. If present, use it as `t_earnings`; if absent, reconstruct it as the **first trading day after quarter-end** (e.g. fiscal quarter end + 45 days, the standard 10-Q filing lag). Document whichever you chose.
    3. **External earnings calendar (yfinance `get_earnings_dates` or similar):** acceptable for prototyping but note `yfinance` earnings dates are sparse and revisionist; flag as such.
  - **Hard rule:** for every feature row at date `t`, every fundamental input must have a publish/announcement date `≤ t`. Add a unit test that asserts this. SimFin's own TTM/derived columns are **inadmissible** for PEAD unless you can prove they are point-in-time (you likely cannot — exclude them from PEAD features but you may keep the existing 14-factor cross-sectional set for the ranker if you can reproduce its point-in-time safety; if not, restrict to the raw quarterly fields).

## 4. Mathematical & Algorithmic Formulation

### 4.1 Feature set (weekly rebase)
Base cross-sectional features (keep the existing 14: momentum 3m/6m/12m/12-1m, reversal 1m, vol 60d/120d, ROA, D/E, gross profitability, gross/operating/net/EBITDA margins). **Add exactly:**
- `sue` — standardized unexpected earnings (per §3).
- `days_since_earnings = t − t_earnings` (in trading days, integer).
- `days_until_earnings = t_earnings_next − t` — used **only as a filter** (see §4.3), never as a target input.

### 4.2 Target
Weekly forward return, `r_fwd = ln(P_{t+5d}/P_t)` (or the 5-trading-day forward close-to-close return). Cross-sectional ranks per week: label = `floor(4 × rank_pct)` quintiles.

### 4.3 Query group & filter
- Group by ISO week `Q_t` (the `lambdarank` `group` arg).
- **Prefilter:** before ranking, drop any row where `0 ≤ days_until_earnings ≤ 2` (avoid holding into an imminent announcement) and any row with `days_since_earnings < 0` (impossible = bad publish-date bookkeeping; count and log these, do not silently fix).

### 4.4 Model
- LightGBM `lambdarank`, `ndcg_at=[5, 10]`. Same `gbdt` boosting as the current model. Max depth 6, `num_leaves=31`, `learning_rate=0.05`, `feature_fraction=0.9`. Deterministic seed `random_state=42`, `deterministic=True`.

### 4.5 Purged Group Time-Series CV with atomic 10-day embargo
- Walk forward: train `[t0, t1]`, embargo `(t1, t1+10d)` — **no rows in this gap in either set**, validate `(t1+10d, t2]`, then expand. Embargo length = 10 calendar days (covers the 5-day forward label window plus margin).
- Use `GroupKFold`-style splitting where the group is the week, then apply the purge and embargo by date, not by row count.
- `n_folds = 5` minimum.

## 5. Data Ingestion & Feature Engineering Spec

- **Price data:** `data/raw/v4_investor_data.parquet` (120,384 rows × 55 cols, 2021-09-01 → 2026-08-31, 96 symbols) is the canonical input — it already contains OHLCV + VIX + 10Y. If you need longer history for walk-forward stability, use the yfinance miner to extend backward (read-only, cached), but do not shorten the forward window.
- **Fundamentals:** read the SimFin caches directly; do not re-implement the CSV parsing in SimFinFundamentalProvider — reuse it where possible.
- **Announcement dates:** whichever source from §3 you chose must be materialized as a new parquet `data/processed/earnings_calendar.parquet` with columns `[symbol, earnings_date, eps_reported, publish_date, source]` and `publish_date` must be `≤` every feature date that consumes that row. Atomic write.
- **Leakage guards (mandatory):**
  - Fundamentals publish-date guard: every fundamental row used at date `t` has `publish_date ≤ t`.
  - Embargo guard: assert `train.max(date) + 10d < val.min(date)` for every fold.
  - Target guard: the target at row `t` uses only closes `> t`.
  - All rolling features use closed-right windows, shifted by 1 before the signal date.

## 6. Mandatory Statistical Falsification Suite

### 6.1 Primary gates
- **Excess ≥ +80 bps/month** vs the equal-weight universe (average daily excess × √21 monthly annualization).
- **DSR > 0.95** (Bailey-López de Prado formula from `src/lab/stats.deflated_sharpe_ratio`; count trials as every distinct (feature set, embargo length, hyperparameter tuple) you fitted).
- **Harvey-Liu-Zhu t-stat > 3.0** across 5-fold CV (`src/lab/stats.hlz_haircut_sharpe` — interpret the adjusted Sharpe t-statistic; record the unadjusted t too).
- **PBO < 0.50** from CSCV (`src/lab/stats.cscv_pbo`).

### 6.2 Shared stats module
**Lane 1 is building `src/lab/stats.py` with the exact contract** (DSR, CSCV PBO, HLZ haircut Sharpe). Check whether it landed in your worktree (it may not — parallel execution). If absent, implement it yourself with the identical signatures Lane 1 will use (copy them from Lane 1's brief §6.2 if you have access, or use the formulas below verbatim). If present, import it — do not duplicate. Coordinate via the `llm_reports/m2m-prompts/2026-09-24_quant-lanes.md` thread (append-only; add a `### VERIFIED — <claim>` block when you add the file and another when your caller consumes it).

Formula copy (canonical):
```
DSR = Φ( (SR_hat − SR_0)·√(n−1) / √(1 − skew·SR_hat + ((kurt−1)/4)·SR_hat²) )
SR_0 = E[max of n_trials N(0, 1/n)] ≈ √(2·ln(n_trials)) − (γ+ln(ln(n_trials)))/√(2·ln(n_trials)) for large n_trials; use scipy exact order-statistic expectation where feasible.
CSCV PBO as described in Lane 1 §6.2.
HLZ haircut Sharpe = SR_hat − z_{1−1/(2N)}·SE(SR_hat), with SE using skew/kurt correction as above.
```

### 6.3 Report skeleton
`llm_reports/recons/2026-09-24_lab-equity-factor-pead.md`, frontmatter per `llm_reports/README.md`, sections: Context → Data & window → Method (features, target, embargo) → Results (per-fold IC, NDCG@5/@10, weekly returns, excess vs equal-weight, monthly annualized excess, DSR, HLZ t, PBO, trial count) → Falsification verdict → Files touched.

### 6.4 Deterministic test cases (mandatory)
1. **Point-in-time guard:** construct a fake SimFin row with `publish_date = t+5`; assert it is excluded from features at `t`.
2. **Embargo guard:** assert `train.max(date) + 10d < val.min(date)` for every fold of your CV splitter.
3. **Buffer/announcement filter:** a row with `days_until_earnings = 2` must be dropped from the ranking group.
4. **Target alignment:** the label for row at date `t` equals `ln(close_{t+5} / close_t)`.
5. **Deterministic training:** two identical runs produce identical model predictions on a fixed validation slice (seed pinned).
6. **Stats module:** `deflated_sharpe_ratio(0, 10, 100, 0, 3)` returns a defined probability in `[0,1]`; `cscv_pbo` on a T×N zeros matrix returns `0.5` (no ordering signal = coin flip); `hlz_haircut_sharpe(1.0, 10) < 1.0`.

### 6.5 Abort criteria
- If the SimFin caches lack a publish-date column **and** you cannot reconstruct one within ±2 trading days of a plausible announcement date, stop and write the report stating the PEAD leg is **data-blocked**; deliver only the weekly-rebase leg (no PEAD) and mark the gate verdict accordingly.
- If any leakage guard test fails after fixes, stop and report.
- If PBO > 0.50 on the final configuration, the verdict is "overfit — no promotion" regardless of raw Sharpe.

## 7. Docs rule
Update or create `scripts/README.md` with an entry for the three new weekly scripts (purpose, inputs, outputs, invocation) following the Layer-2 README table format. Add `GLOSSARY.md` entries for "SUE", "PEAD", "point-in-time (PIT) filter", "embargo".

## 8. What "done" looks like
- Only your files modified on `lane/equity-factor-pead`.
- Tests green, compileall green.
- Recon report exists, gates evaluated honestly, artifacts (calendar parquet, stats module if built by you) land atomically.
- Commit message `feat(investor): lane2 weekly factor rebase + PEAD [falsification gate]`.
