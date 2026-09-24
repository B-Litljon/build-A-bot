---
type: handoff
date: 2026-09-24
time: 15:49 PDT
agent: dsh (DeepSeek V4.1 Flash via Ollama Cloud)
model: ollama-cloud/deepseek-v4.1-flash
trigger: Lane 5 dispatch — three background falsification audits
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
files_touched:
  - src/lab/altcoin_topquint.py
  - src/lab/fix_audit.py
  - src/lab/fix_collector.py
  - scripts/fix_tick_collector.py
  - systemd/fix-tick-collector.service
  - src/lab/vwap_reversion.py
  - src/lab/stats.py
  - tests/test_lab_altcoin.py
  - tests/test_lab_fix_audit.py
  - tests/test_lab_vwap_reversion.py
  - llm_reports/recons/2026-09-24_lab-altcoin-cross-sectional.md
  - llm_reports/recons/2026-09-24_lab-fix-flow-audit.md
  - llm_reports/recons/2026-09-24_lab-vwap-reversion.md
related:
  - llm_reports/recons/2026-09-14_session-evidence-and-options.md (crypto axis verdict, toll figures)
  - src/data/alpaca_provider.py (crypto data path)
  - src/data/oanda_provider.py (OANDA quote stream, tick hook precedent)
---

## 1. System Persona & Scope

You are a **quantitative researcher** running three **independent background falsification audits**, each producing a research report with a falsification verdict. None of these are live trading systems; only the Fix Tick Collector produces a deployable artifact (the collector service file, which ships **uninstalled**).

**Scope lock — you may only touch:**
- `src/lab/altcoin_topquint.py` (new) — Audit A
- `src/lab/fix_audit.py` (new) + `src/lab/fix_collector.py` (new) — Audit B
- `scripts/fix_tick_collector.py` (new) + `systemd/fix-tick-collector.service` (new, uninstalled unit template) — Audit B's collector
- `src/lab/vwap_reversion.py` (new) — Audit C
- `src/lab/stats.py` (shared; build if absent, reuse if Lanes 1–4 landed it)
- Three test files, three reports (paths above)
- `src/lab/README.md` entry, `GLOSSARY.md` additions.

You must **not** modify any existing lab module, retrainer module, execution module, or production scheduler.

**Soak guard:** identical to Lanes 1–4.

**Test commands:** same venv + `PYTHONPATH=src:.; pytest -q` + `compileall -q src/` as Lanes 1–4.

## 2. Context & Problem Statement

Three falsification audits, each testing whether one of three candidate edges survives honest measurement:

**Audit A — Altcoin cross-sectional momentum.** The 2026-09-14 crypto study closed D1/H4 forex-style bracket geometries (0 of 27 positive at random) but did not test **cross-sectional momentum** across a basket of liquid altcoins — a documented anomaly in crypto markets. The question: does a weekly LambdaRank top-quintile harvester beat both an equal-weight basket and BTC buy-and-hold?

**Audit B — Calendar flow liquidity (London WM/R fix & Tokyo Nakane fix).** Two well-known intraday liquidity events: the 4 PM London WM/Reuters fix and the 9:55 AM JST Tokyo Nakane fix (Gotobi days = 5/10/15/20/25/month-end). Market-maker spread-widening ahead of the fix is the standard institutional defense; the question is whether the widening exceeds the expected drift, in which case the "fade the fix" route should be eliminated. **Critical data constraint (verified 2026-09-24): no tick or bid/ask data exists anywhere in this repo.** All cache is mid-price M15 OHLCV. The highest-fidelity spread artifact is `config/spread_alphas_m15.json` — six per-instrument median-spread scalars, not a time series. A tick collector is therefore a deliverable, not a precondition.

**Audit C — Session-scale VWAP reversion.** Fade `k·σ` deviations from a session VWAP anchored at London open (08:00–09:00 UTC), with exits anchored on inventory structure rather than arbitrary targets. Known FX intraday mean-reversion hypothesis.

## 3. Execution & Platform Constraints

- **Venue:** Alpaca crypto (Audit A) and OANDA forex M15 (Audits B and C). All research-only, no orders.
- **Audit B data floor:** M15 mid bars from `analysis_cache/strategy_matrix/{sym}_M15.parquet` for the 6 fiat pairs (2024-09-08 → 2026-09-08). No tick data anywhere; do not pretend otherwise.
- **Audit B collector:** uses `OandaMarketProvider` (`src/data/oanda_provider.py`) streaming path. The collector **must** use the repo's atomic-write convention (temp file + `os.replace`, fsync before replace) — this is the same pattern as `src/core/events.py:180-195` and `src/lab/experiments.py:151`; reuse it.
- **Audit C data floor:** same M15 cache.

## 4. Audit A — Altcoin cross-sectional momentum

### 4.1 Math & algorithm
- Universe: 15–20 most-liquid Alpaca altcoins ≥ some liquidity floor. Use: `BTC/USD, ETH/USD, SOL/USD, AVAX/USD, LINK/USD, DOT/USD, LTC/USD, BCH/USD, UNI/USD, AAVE/USD, MKR/USD, CRV/USD, SNX/USD, COMP/USD, YFI/USD, SUSHI/USD, UMA/USD, ZRX/USD, BAT/USD, GRT/USD`. Filter by ≥30 days of non-zero volume in the sampled window and median daily volume above a $250k proxy (volume × close) — Alpaca crypto volume is unreliable for some alts; drop a symbol from the universe if its volume data is sparse and document the drop.
- Feature: 4 momentum lookbacks `{21, 63, 126, 252}`-day log returns on daily closes.
- Model: LightGBM `lambdarank`, `ndcg_at=[5, 10]`, weekly query groups (`Q_t` = ISO week). Target: next-week return quintile (0–4).
- Execution proxy: weekly top-quintile basket, 1/N within the quintile, friction 6.6 bps spread + 25 bps taker as in Lane 1.

### 4.2 Dual benchmark
- Equal-weight basket of the same universe for the same weeks.
- BTC buy-and-hold.
- The strategy must beat **both** by a statistically significant margin (Clopper-Pearson lower bound on weekly outperformance rate > 0, or equivalently the HLZ t-stat on the excess series > 3.0).

### 4.3 Gate
- Dual benchmark passes.
- DSR > 0.95, PBO < 0.50.
- If the top quintile cannot beat BTC, the report must say so in one line and stop.

### 4.4 Leakage guards
- Weekly query groups computed on closed daily bars only; the week's target uses only data after the week's last signal bar.
- Universe membership at week `t` uses only liquidity data ≤ `t` (no survivorship from future volume).

### 4.5 Tests
- Synthetic 10-asset universe with monotonically ranked returns: ranker must identify the top group.
- Universe guard: an asset with volume < floor at entry week must be excluded.
- Friction test: forced turnover of 50% of the basket must produce a NAV cost of 0.5 × (33+25)/10000 on that week.
- Determinism: two runs with same inputs produce same outputs.

### 4.6 Report
`llm_reports/recons/2026-09-24_lab-altcoin-cross-sectional.md`, recons template.

## 5. Audit B — Calendar flow liquidity sweep

### 5.1 Coarse study (from M15 bars — the only data available)
Compute, per pair, per timestamp:
- The session return `r_t = ln(C_t / C_{t-1})` for the M15 bar that **contains** the fix instant (16:00 London WM/R → UTC varies with DST: 16:00 BST = 15:00 UTC summer; use zoneinfo `Europe/London`). For Tokyo 09:55 JST: 00:55 UTC fixed.
- A reference return over the same bar on non-fix days (e.g. all days, or non-Gotobi days for Tokyo).
- The 99th-percentile |return| on fix days vs non-fix days; the same for the next bar (post-fix drift).
- A "high-low range" per M15 bar (as % of mid) as a coarse proxy for spread/vol activity. **State explicitly: this is not a bid/ask spread measure.**

### 5.2 Verdict rule
- If the fix-bar |return| 99th percentile exceeds the non-fix 99th percentile by > 2× and the post-fix drift is directionally consistent with reversion, the route is **not obviously eliminated** — but with M15 data this is at most a "needs tick data" verdict, never a "falsified" verdict. **State that limitation prominently.**
- If the fix-bar and non-fix-bar ranges are indistinguishable, the post-2015 spread-widening concern is **not supported at M15 resolution**. This is *not* the same as "no widening exists"; say precisely what was measured.

### 5.3 Tick collector (build but do not install)
- `scripts/fix_tick_collector.py`: streams quotes from `OandaMarketProvider` for the 6 fiat pairs, filters ticks to a window ±5 min around each fix, writes daily parquet `data/ticks/fix_ticks_YYYY-MM-DD.parquet` with columns `[timestamp_utc, symbol, bid, ask, mid, source_latency_ms]`. Atomic daily write via temp+rename. Retries on disconnect with exponential backoff. Runs continuously, daemon mode.
- `systemd/fix-tick-collector.service`: **template only, uninstalled.** Modeled on `soak.service` (same user-unit conventions: `Linger=yes` note, `KillMode=mixed`, `SIGTERM`, `TimeoutStopSec=90`, `Restart=on-failure`). Include a comment block at top saying "install via `systemctl --user link <this file>` only after the coarse study is reviewed."
- **Do not install or enable the service.** Do not run the collector. Ship it only.

### 5.4 Tests
- Fix-window identification: DST-correct UTC conversion for London fix across the 2026 DST boundary; Tokyo JST constant.
- Collector file format: written parquet has the 6 columns; atomic rename leaves no `.tmp` after completion.
- Coarse-study determinism: same bar cache → same verdict numbers.

### 5.5 Report
`llm_reports/recons/2026-09-24_lab-fix-flow-audit.md`, recons template; make the data-resolution limitation a first-class section.

## 6. Audit C — Session-scale VWAP reversion

### 6.1 Math & algorithm
- Session: 08:00–09:00 UTC (London open hour), daily.
- Session VWAP: `VWAP_t = (Σ P·V over session bars up to t) / (Σ V)`.
- Session σ: realized σ of returns within the session.
- Entry: fade when `|P_t − VWAP_t| > k·σ_session`, `k ∈ {1.5, 2.0, 2.5, 3.0}` — pick one and report all four in the sweep, with N=4 as the trial count for stats.
- Exit: anchored on **inventory imbalance** of the session's bars, not a fixed target. Define imbalance as the running signed volume imbalance `Σ sign(r) · V` within the session; exit when imbalance crosses zero (net inventory neutralized) or at session end 09:00 UTC whichever first.
- Sizing: fixed 1% NAV risk per trade, stop at `k·σ` re-extension (i.e. exit when the deviation moves further from VWAP by another `k·σ`), so R is well-defined.

### 6.2 Gate
- Net EV > +0.25R per trade (after the 0.25R toll — i.e. **gross EV > 0.50R** to clear it, or state your toll convention and compute net consistently).
- DSR > 0.95, PBO < 0.50, HLZ t > 3.0.
- The exit-imbalance rule must show positive expectancy **independent of k choice** within the sweep (if only one k clears, that is a red flag for overfit and must be flagged in the verdict).

### 6.3 Leakage guards
- `σ_session` at time `t` uses only bars ≤ `t`.
- Exit decision at time `t` uses imbalance ≤ `t`.
- VWAP is cumulative from session open only — no lookback into prior sessions.

### 6.4 Tests
- VWAP arithmetic on a 5-bar synthetic session.
- Imbalance sign flip triggers exit on the exact bar where the sign flips.
- Session boundary: a trade that has not exited by 09:00 UTC is force-closed at 09:00.
- k-sweep determinism.

### 6.5 Report
`llm_reports/recons/2026-09-24_lab-vwap-reversion.md`, recons template.

## 7. Shared stats contract

All three audits use `src/lab/stats.py` (DSR, CSCV PBO, HLZ) — same contract as Lanes 1–4. Build it if absent (Lane 1's brief §6.2 carries the canonical formulas — use them verbatim, `scipy.stats.norm` for the Normal CDF/quantile and `scipy.stats.beta` for the CP bound). Trial counts:
- Audit A: number of distinct (lookback tuple, K, rebalance cadence) evaluated.
- Audit B: number of distinct fix events × pair tested — these are not independent trials in the HLZ sense; use `n_trials = 2` (two fixes) for the HLZ, the honest conservative value.
- Audit C: 4 (the k sweep).

## 8. Abort criteria

- Any audit whose leakage guard tests fail after fixes → abort that audit only; reports for the others still ship.
- Audit B collector cannot import `OandaMarketProvider` stream path without exceptions → write the collector with a `# TODO: confirm stream callback signature at runtime` block but do not run it; mark the collector portion of the deliverable as **unverified** in the report.
- Altcoin volume floor leaves < 5 assets → report the tiny universe as the finding.

## 9. Docs rule
`src/lab/README.md`: three new file entries. `GLOSSARY.md`: entries for "cross-sectional momentum", "WM/R fix", "Tokyo Nakane fix", "Gotobi day", "VWAP reversion", "inventory imbalance exit".

## 10. What "done" looks like
- Branch `lane/falsification-audits` holds your files only.
- Three reports exist with honest falsification verdicts per audit.
- Collector ships as an uninstalled template; no service is running.
- Tests green; compileall green.
- Single commit `feat(research): lane5 falsification audits (altcoin xs-mom, fix flow, vwap reversion)`.
