---
type: handoff
date: 2026-09-24
time: 15:45 PDT
agent: dsh (DeepSeek V4.1 Flash via Ollama Cloud)
model: ollama-cloud/deepseek-v4.1-flash
trigger: Lane 1 dispatch — daily crypto trend + volatility targeting
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
files_touched:
  - src/lab/momentum_crypto.py
  - tests/test_lab_momentum_crypto.py
  - src/lab/README.md
  - GLOSSARY.md
related:
  - llm_reports/recons/2026-09-14_session-evidence-and-options.md (crypto axis verdict: 0/27 D1 + 0/27 H4 geometries positive at random; D1 toll 0.0497R; BTC B&H +283%)
---

## 1. System Persona & Scope

You are a **quantitative researcher** and **execution engineer**. Your deliverable is an **offline, deterministic, self-contained backtest harness** for a daily spot crypto trend-following strategy. You build nothing live, no scheduler, no Alpaca **order submission** — this is a **research artifact only**: data ingestion + signal + sizing + cost model + falsification statistics. Your only write targets are the worktree branch `lane/crypto-trend-mom-tv` and new files under `src/lab/`, `tests/`, `analysis_cache/lab_frames/` (frame cache), and `llm_reports/recons/` (the recon report).

**Scope lock — you may only touch:** `src/lab/momentum_crypto.py` (new), `tests/test_lab_momentum_crypto.py` (new), one recon report in `llm_reports/recons/`, `src/lab/README.md` (add your file entry), `GLOSSARY.md` (add your terms). You must **not** edit `src/execution/`, `run_oanda.py`, `src/core/retrainer/`, the lab's existing modules (`spec.py`, `registry.py`, `backtest.py`, `gate.py`, `frames.py`, `experiments.py`, `ablate.py`, `report.py`, `cli.py`, `artifact.py`, `features.py`, `data.py`, `specs.py`), or any test outside your own.

**Soak guard:** the M15 forex soak runs from `/mnt/storage/mystuf/development/build-A-bot` as `soak.service` (systemd --user). A cron watchdog (`soak_watchdog.sh`, every 5 min, PAM-locked crontab) relaunches from the **working tree on the checked-out branch**. You must **never** `git checkout`/`git switch`/`git merge` inside that checkout and never run `systemctl --user stop/restart soak.service`. All your work happens in the branch/worktree at `/mnt/storage/mystuf/development/build-a-bot-lanes/crypto-trend-mom-tv` only. Verify with `ps aux | grep '[r]un_oanda'` and `systemctl --user is-active soak.service` before assuming the soak is down.

**Test/verification commands (run from the worktree root, not the main checkout):**
```bash
cd /mnt/storage/mystuf/development/build-a-bot-lanes/crypto-trend-mom-tv
PYTHONPATH=src:. /home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python -m pytest -q
/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python -m compileall -q src/
```
System python 3.14 fails test collection; the pipenv venv above is mandatory.

## 2. Context & Problem Statement

The pipeline is abandoning single-instrument directional bracket trading on 15-minute spot forex. Measured evidence (2026-09-14): the forex M15 axis is a **driftless martingale** where the realized round-trip toll (~0.25R at the served 2.0×/4.0× bracket) destroys all gross expectancy; the live model's proposals sit below its approval bar because GBDTs trained on 25%-base-rate survival labels output compressed probabilities. The served artifact proposes on 0.028% of bars and **0 in the 2,691 bars after its training data ends** — the bar saying no is the correct behavior.

Lane 1 redeploys that research effort onto a market and horizon where the **friction budget is an order of magnitude smaller relative to the risk premium**: daily BTC/USD and ETH/USD spot on Alpaca, with holding periods measured in weeks and a cost model of 6.6 bps round-trip spread + 0.25% taker fee. The 2026-09-14 crypto study already measured D1 toll at **0.0497R** (vs forex's 0.25R) and buy-and-hold at BTC +283% / SOL +460% over the study window — but it also measured **0 of 27 random-entry geometries positive at D1**, so this lane is **not** a "bracket" bet. It is **time-series momentum with volatility targeting**, the canonical asset-allocation overlay that does not depend on directional brackets.

**Your bottleneck:** there is **no deterministic, leak-free, cost-aware daily crypto backtest harness** in the repo. The 2026-09-14 crypto numbers were produced by scratch scripts that "did not survive the session" (`llm_reports/recons/2026-09-14_session-evidence-and-options.md:319-320`). No crypto bars are cached on disk. You must build the harness and produce a falsifiable verdict.

## 3. Execution & Platform Constraints

- **Venue:** Alpaca spot crypto, **paper** lane (`ALPACA_API_KEY`/`ALPACA_SECRET_KEY` env vars; read from `.env` in the worktree root via `dotenv` if you need them — do **not** print values, do **not** submit orders).
- **Constraints:** long-only, `0 ≤ w_{i,t} ≤ 1.0`, no margin, no short. The portfolio is 2-asset (BTC/USD, ETH/USD) plus a cash bucket earning 0%.
- **Data path:** use the existing `AlpacaProvider` (`src/data/alpaca_provider.py:92`): `AlpacaProvider(api_key=..., secret_key=..., paper=True)` → `get_historical_bars(symbol, timeframe_minutes=1440, start, end)`. The 2026-09-14 fix (`_timeframe_for`, tests `tests/test_alpaca_timeframe.py`) makes daily crypto fetchable. Do **not** call the Alpaca REST API directly; go through the provider.
- **Cost model (mandatory, applied to every weight change):** spread 6.6 bps round-trip (use 33 bps/half-turn if you amortize) + taker fee 0.25% = 25 bps one-way. Implement as `cost_t = Σ_i |Δw_{i,t}| × (spread_bp/2 + fee_bp) / 10_000`. Record both gross and net NAV.
- **Caching:** persist fetched bars to `analysis_cache/lab_frames/crypto_daily_bars.parquet` (columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `fetched_at_utc`). Atomic write (temp + `os.replace`). Re-runs must use the cache unless `--refresh-bars`.

## 4. Mathematical & Algorithmic Formulation

### 4.1 Signal
For each asset `i ∈ {BTC/USD, ETH/USD}` and lookback `k ∈ {21, 63, 126, 252}` trading days:
- Momentum: `r_{k,i,t} = ln(P_{i,t} / P_{i,t-k})`.
- Sign: `s_{k,i,t} = sign(r_{k,i,t})` (with `s=0` if `|r|` below a deadband of `1e-6` to avoid noise on flat windows).
- Ensemble vote: `S_{i,t} = median({s_{21}, s_{63}, s_{126}, s_{252}})` — median of four signs lands in `{−1, 0, +1}`; `S=0` means tie (no position). Alternatively implement additive `S = Σ s_k / |Σ s_k|` guarded against zero; **document your choice** and pin it by test.

### 4.2 Raw sizing (inverse-vol target, long-only)
- Realized vol estimate: `σ̂_{i,t} = std(daily ln returns over last N_σ days) × √365` (crypto trades 365). `N_σ ∈ {20, 60}` — implement both, choose one default, report both.
- Target vol: `σ_target ∈ [0.20, 0.30]`, default 0.25.
- Raw weight: `w*_{i,t} = (σ_target / σ̂_{i,t}) × max(0, S_{i,t})`. Clip to `[0, 1.0]`. Portfolio-level: if `Σ_i w*_{i,t} > 1`, scale each by `1/Σ`. Cash gets the remainder.
- **Lookback alignment:** `σ̂` and all momentum returns at day `t` must use closes up to and including `t`. Order executes **next open** `t+1`. This shift is the non-negotiable leak guard — pin it by test.

### 4.3 Rebalance buffer
- Trade only if `|w*_{i,t} − w_{i,t-1}| > δ`, `δ = 0.05`. Otherwise hold previous weight.
- The buffer applies to the **target** weight before the cash sweep. After buffer, residual cash is the 1−Σw.

### 4.4 Backtest loop contract
- Daily bars, signal computed on close `t`, fill assumed at open `t+1` (use `open` price, not close) — model the overnight gap honestly.
- Returns: daily portfolio return = `Σ_i w_{i,t} × r_{i,t+1}` where `r` is open-to-open. NAV starts at 1.
- Drawdown series tracked continuously; annualize with √365 (daily crypto).

## 5. Data Ingestion & Feature Engineering Spec

- **Query:** `AlpacaProvider.get_historical_bars("BTC/USD", 1440, start=..., end=...)` and same for `ETH/USD`. Aim window `2021-01-01 → 2026-09-24` (today).
- **Fallback rule (mandatory):** Alpaca's free-tier crypto history may not cover 2021. Probe first: fetch a narrow `2021-01-01 → 2021-01-15` window. If empty or errored, **degrade the window** to the earliest date the feed returns in a bisect walk (report the discovered floor), and **state the actual tested range in the report title** (e.g. `2022-XX-XX → 2026-09-24`). Do **not** silently truncate: the gate (`§6`) must be evaluated over the *realized* window and the report must say so.
- **Leakage guards:**
  - All rolling features are computed on **closed bars only** (no intra-day peek). Use `pl.DataFrame.rolling` or `pandas.rolling` with `closed='right'` semantics, then shift by 1 before the execution date.
  - The momentum lookback at `t` must be computable from data up to `t` only — no forward joins.
  - Cross-asset contamination is not present (each asset's features use only that asset's closes), but code must be structured so a future basket extension does not accidentally cross-join timestamps across assets.
- **No feature engineering beyond §4.** This lane is deliberately minimal; anything extra is scope creep and out of scope.

## 6. Mandatory Statistical Falsification Suite

Every number below is **computed in the harness** and printed in the report. No narrative, no "it looks good."

### 6.1 Primary gate
- **Calmar ratio** = `CAGR / |MaxDD|` over the realized window, annualized with √365. Target `> BTC buy-and-hold Calmar over the same window`.
- **Max drawdown < 30%** peak-to-trough on the net NAV.
- If either fails, the report's verdict is **"does not clear gate"** — do not soften it. Record the actual numbers.

### 6.2 Universal stats contract (shared with all lanes)
There is **no DSR, CSCV, PBO, or Harvey-Liu-Zhu code anywhere in the repo right now** (verified 2026-09-24: zero matches in `src/`; `statsmodels` is not imported). You will be the first to add it. Implement it in `src/lab/stats.py` (new file) with these exact signatures, because Lanes 2–5 will import it:

```python
def deflated_sharpe_ratio(
    sr_hat: float, n_trials: int, n_obs: int,
    skew: float, kurt: float, *, var_sr: float | None = None,
) -> float: ...

def cscv_pbo(logret_matrix: "np.ndarray") -> float:
    """Input: T×N daily log-return matrix, T = observations, N = strategies.
    Returns the PBO scalar."""

def hlz_haircut_sharpe(sr_hat: float, n_trials: int) -> float: ...
```

- **DSR formula (Bailey & López de Prado 2014):**
  `DSR = Φ( (SR_hat − SR_0) · √(n_obs−1) / √(1 − skew·SR_hat + ((kurt−1)/4)·SR_hat²) )`
  where `SR_0 = E[max(SR)]` under the null, the expected max of `n_trials` draws from a Normal with variance `var_sr` (default `var_sr = 1/n_obs`). `Φ` is the Normal CDF. Use `scipy.stats.norm`. Return the **probability** in `[0, 1]`. For the "how many trials" question: count every distinct (k-lookback, N_σ, σ_target) triple you *evaluated*, not the grid you *thought about*.
- **CSCV PBO:** implement exactly Bailey, Borwein, López de Prado, Zhu (2014). Take the T×N daily log-return matrix; split rows into S=8 contiguous blocks; enumerate all `C(8,4)=70` combinations; for each combo `C`, form train (the four blocks in `C`) and test (the complement); per strategy compute train SR and test SR; rank strategies by train SR; let `n*` be the argmax; compute `ω = rank_test(n*) / (N+1)` (average-rank if ties); `λ = logit(ω)` clipped to `±10`; `PBO = (#{λ < 0})/70`. Return the scalar. **Note:** with your lane's ~2 assets × small param grid, N is small; the statistic is still defined and is what a reviewer expects.
- **HLZ haircut Sharpe (Harvey, Liu, Zhu 2016, eq. approx):** `HLZ = SR_hat − z_{1−1/(2·n_trials)} · sqrt( (1 − skew·SR_hat + (kurt−1)/4 · SR_hat²) / (n_obs−1) )` where `z` is the Normal quantile. Use the same `scipy.stats.norm`. This is the *expected maximum SR* adjustment, reported as an adjusted SR number, not a probability.

### 6.3 Report skeleton (mandatory, file `llm_reports/recons/2026-09-24_lab-crypto-trend-mom-tv.md`)
Use the `recons` frontmatter template (`llm_reports/README.md`) with `type: recon`, `agent: dsh`, `model: ollama-cloud/deepseek-v4.1-flash`, `head: <your HEAD>`, `scope: read-only` (research) or `modifies-source`. Body sections:
- **Context** — cite the forex martingale finding and the 0.0497R D1 toll from `recons/2026-09-14_session-evidence-and-options.md`.
- **Data & window** — actual realized window, source, cache path, any degradation from the 2021 target.
- **Method** — formulas §4, leak guards §5.
- **Results** — NAV curve (ASCII or tabular), Calmar, MaxDD, gross vs net, benchmark Calmars, DSR, PBO, HLZ, trial count, per-fold OOS metrics from validation.
- **Falsification verdict** — one line: gate pass / fail and which bar failed.
- **Files touched** — list.

### 6.4 Deterministic test cases (mandatory, in `tests/test_lab_momentum_crypto.py`)
1. **Shift test:** with a synthetic 300-day price series that trends up for the first 200 days and down for the last 100, the signal at day 199 must not depend on days 200+. Compute it twice, once truncated and once full, assert equality.
2. **Buffer test:** with `w*_t` oscillating within ±0.04 of `w_{t-1}`, no trade is issued. At exactly 0.05 the trade fires.
3. **Cost test:** a synthetic 10-day run with one forced weight change of 0.5 must reduce NAV by exactly `0.5 × (33+25)/10000` on the trade day (net − gross).
4. **Cache test:** running the fetch twice with the same `start/end` must produce byte-identical cached parquet (atomic rename).
5. **Schema test:** the cached bars file has the 7 declared columns in order.

### 6.5 Abort criteria
Stop and write the report immediately if:
- The Alpaca crypto feed at `1440` returns nothing for the discovered floor window — record the actual floor and proceed with the degraded window; the gate still applies to what you measured.
- The realized window is < 365 days — report the measured stats with a "**short sample**" banner and do not evaluate the gate.
- Any leakage guard test (§6.4 #1) fails.

## 7. Docs rule
Update `src/lab/README.md` with a `### momentum_crypto.py` entry (one paragraph: imports from repo, imported by, reads/writes) and add a `GLOSSARY.md` entry for `crypto trend mom tv` under the appropriate thematic section. Format follows the existing Layer-2/Layer-3 conventions.

## 8. What "done" looks like
- `git -C /mnt/storage/mystuf/development/build-a-bot-lanes/crypto-trend-mom-tv status` shows only your files modified/added.
- `PYTHONPATH=src:. ... pytest -q` exit 0 with your new tests passing.
- `compileall -q src/` exit 0.
- `llm_reports/recons/2026-09-24_lab-crypto-trend-mom-tv.md` exists, carries the gate verdict, and every number in it is reproducible from a `--no-refresh` re-run of your harness.
- `git commit` with a message `feat(lab): lane1 crypto trend momentum + vol targeting [falsification gate]` (do not push; Brandon reviews).
