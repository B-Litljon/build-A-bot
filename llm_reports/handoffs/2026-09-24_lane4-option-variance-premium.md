---
type: handoff
date: 2026-09-24
time: 15:48 PDT
agent: dsh (DeepSeek V4.1 Flash via Ollama Cloud)
model: ollama-cloud/deepseek-v4.1-flash
trigger: Lane 4 dispatch — index option variance-premium harvester (conditional on capability)
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
files_touched:
  - scripts/option_capability_probe.py
  - scripts/option_variance_harvest.py
  - src/lab/stats.py
  - tests/test_option_probe.py
  - tests/test_option_harvest.py
  - llm_reports/stops/2026-09-24_option-capability-blocked.md (if Stage 0 stops)
  - llm_reports/recons/2026-09-24_lab-option-variance-premium.md (if research completes)
related:
  - src/data/alpaca_provider.py (Alpaca SDK 0.43.2, OptionHistoricalDataClient present)
  - llm_reports/audits/2026-06-09_full-repo-audit.md (no options code precedent)
---

## 1. System Persona & Scope

You are an **algorithmic execution engineer** building a **capability-gated research stack**. The deliverable is **conditional**: a two-stage pipeline where **Stage 0** is a programmatic verification that the account and SDK actually support Level-3 multi-leg option trading and historical options data, and **Stage 1** (the research) runs only if Stage 0 passes. If Stage 0 fails, the deliverable is a **stop report**, not a backtest.

**Scope lock — you may only touch:**
- `scripts/option_capability_probe.py` (new)
- `scripts/option_variance_harvest.py` (new)
- `src/lab/stats.py` (shared; build if absent, reuse if Lane 1/2 landed it)
- `tests/test_option_probe.py`, `tests/test_option_harvest.py`
- One report (either `stops/` or `recons/`, depending on Stage 0)
- `scripts/README.md` entry, `GLOSSARY.md` additions.

You must **not** edit `src/execution/`, `run_oanda.py`, investor scripts, the lab package, or any existing test. You must **not** submit a live order as part of the probe — the probe reads capability metadata only.

**Soak guard:** same as Lanes 1–3.

## 2. Context & Problem Statement

The forex martingale conclusion and the +0.045R vs 0.09R edge budget push the pipeline toward markets where the **risk premium itself exceeds the friction by an order of magnitude**. Equity index **variance premium** — the empirical tendency of implied volatility to exceed realized volatility — is the canonical such edge, with a wide literature (e.g. Bakshi-Kapadia, Israelov-Nielsen). Unlike bracket trading, a short-volatility defined-risk strategy profits from a structural risk-transfer premium, not from predicting direction.

**The operational bottleneck:** the repo contains **zero options infrastructure** (verified by grep 2026-09-24 — no `implied volatility`, no `IVR`, no spread, no greeks code anywhere). The Alpaca SDK installed (`alpaca-py==0.43.2`) exposes `OptionHistoricalDataClient` (`alpaca.data.historical.option`) and request models for option legs (`trading/requests.py:169` `OptionLegRequest`, `:581` `GetOptionContractsRequest`, `MarketOrderRequest.legs/order_class/position_intent`), but **no verified end-to-end multi-leg submission path exists in this codebase**. Submitting a live multi-leg options order as a capability test is a money-losing bug if anything is wrong. So the first job is a **read-only Stage 0 probe** that asks the API what the account can do, and refuses to proceed if the answer is not "Level 3 + data available".

## 3. Execution & Platform Constraints

- **Venue:** Alpaca paper equities/options, `.env` has `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` (used by existing scripts; do not print).
- **Symbols:** SPY, QQQ, IWM (and no others).
- **Account constraint:** the paper account's option approval level is unknown. You will discover it via `TradingClient.get_account()` (attribute is commonly `options_approved_level` or nested config on `Account` — use `dir(account)` introspection at runtime; do not assume the attribute name).
- **Data constraint:** `alpaca.data.historical.option.OptionHistoricalDataClient` and `alpaca.data.live.option` are importable in this SDK version, but data availability and history depth are unverified. Stage 0 must attempt a tiny historical chain fetch for SPY and report what it actually returns.

## 4. Stage 0 — the capability probe (MANDATORY FIRST STEP)

Implement `scripts/option_capability_probe.py` with this exact CLI:

```
PYTHONPATH=src:. python scripts/option_capability_probe.py --symbol SPY
```

It must print a JSON report with three verdicts and exit `0` only if both are affirmative:

1. **`options_approved_level >= 3`** (or equivalent multi-leg flag on the account) — verdict `true|false|unknown`. If `unknown`, treat as `false`. Do not submit any order.
2. **Historical option chain fetch works:** call `OptionHistoricalDataClient.get_option_chain(...)` for SPY with a narrow window (e.g. next week expiry, ±5 strikes) and report row count, earliest/latest expiry, and whether `implied_volatility` and `delta` columns are present. Verdict `true|false`.
3. **Greeks available from data or computable from OHLC:** boolean.

If either 1 or 2 is `false`, exit `2` and print a human-readable line describing which capability is missing. **Stop conditions (write the stop report and exit):**
- Account level < 3 → `stops/2026-09-24_option-capability-blocked.md` says the account cannot trade multi-leg strategies; recommend upgrading or abandoning the lane.
- Historical option data returns zero rows or lacks `implied_volatility`/`delta` → the report says the SDK/account cannot source the inputs needed for entry triggers; recommend a data-source change (e.g. paid Polygon/CBOE feed) before any further work.

## 5. Stage 1 — the research (runs only if Stage 0 exits 0)

### 5.1 Mathematical & algorithmic formulation

**Universe:** SPY, QQQ, IWM. Entry window: 30–45 DTE.

**Candidate selection (per day):**
- Pull the chain for each underlying; filter expiries into `[30d, 45d]`.
- Structure: **put credit spread** (sell Δ∈[0.20, 0.30] put, buy put 5 strikes lower) or **iron condor** (both wings Δ∈[0.20, 0.30], 5-strike wide). Choose one per underlying once and document the choice; do not cherry-pick per trade.
- **Entry trigger (repurposed veto logic as positive filter):**
  - Compute IVR: `rank of current ATM IV over the last 252 trading days` (percentile, 0–100).
  - Gate A (spread) repurposed: skip entry if the option-chain bid/ask width as % of mid > threshold.
  - Gate B (volatility band) repurposed: enter only when the underlying's realized vol is in the second-highest quintile of its 1-year distribution — the vol regime where variance premium empirically concentrates.
  - **Chop-veto inverted:** the existing `_compute_chop_veto_mask` (`src/core/retrainer/_labels.py:163-168`) discards 23.7% of consolidation rows for directional trading. For short-vol you **want** these — a rangebound underlying is exactly where a short condor wins. Implement `chop_friendly = chop_veto_mask` as a *positive* entry requirement when IVR ∈ [25, 30]: enter **only** on rows the directional strategy would veto. (This is the repo-integration mandate — reuse `_compute_chop_veto_mask` directly, do not re-implement it.)
  - **HMM regime filter (optional but preferred):** if `USE_HMM_FEATURES` path is importable, condition entry on the HMM's calm/mean-reverting state; otherwise document its absence.

**Sizing:** fixed 1% NAV risk per trade (max loss = spread width − credit). Number of contracts = `floor(1% NAV / max_loss_per_spread)`. No leverage.

**Exit rules (mechanical, non-negotiable):**
- Exit at 50% credit harvested OR at 21 DTE remaining, whichever first — both are hard.
- Additional hard stop at 2× credit loss (defined-risk fallback).

### 5.2 Backtest contract

- Use the option bars themselves (close, IV, delta) to reconstruct positions day-by-day; do not simulate option prices from the underlying — use what the feed gives.
- P&L per day = mark-to-market of the open spreads at historical option mid prices (bid+ask)/2.
- Costs: Alpaca options commission $0.65/contract each way plus half-spread slippage.
- Output: daily portfolio returns, per-underlying and pooled; equity curve; Sharpe (annualized √252); per-trade R distribution; max drawdown.

### 5.3 Falsification mandatory gates
- **Realized Sharpe > SPY buy-and-hold Sharpe** over the identical window (not the option data window).
- **DSR > 0.95** with **skew/kurt adjustments** (the formula already carries them; a short-vol strategy is negatively skewed, so DSR will be stricter than a Gaussian assumption — this is by design).
- **PBO < 0.50**.
- **HLZ t-stat > 3.0**.

### 5.4 Shared stats module
`src/lab/stats.py` as in Lanes 1–3. Same contract.

## 6. Data Ingestion & Feature Engineering Spec

- **Option data:** `OptionHistoricalDataClient.get_option_chain(underlying_symbol='SPY', ...)` — pull daily snapshots for each underlying for the test window (aim 2022-01-01 → 2026-09-24, degrade if feed shallower). Cache atomic parquet to `analysis_cache/lab_frames/option_chain_{symbol}.parquet` with columns `[timestamp, expiry, strike, right, bid, ask, mid, iv, delta, underlying_close]`.
- **Underlying bars:** yfinance daily for SPY/QQQ/IWM for the buy-and-hold benchmark and the HMM/chop/gate computations — these feed the repo's existing filters, which are equity-bar-based.
- **IVR computation:** ATM IV = IV of the option with |delta| closest to 0.50 at 30–45 DTE, per day. IVR = rolling 252d percentile.
- **Leakage guards:** every filtering decision at date `t` uses data at `t` only; the option entry uses the chain at `t` open, exit at first trigger day after.

## 7. Report skeletons

**Stop report (if Stage 0 fails):** `llm_reports/stops/2026-09-24_option-capability-blocked.md` — frontmatter `type: stop`, `files_touched` listing the probe script only. Body sections per `llm_reports/README.md` stop convention: Context → What was probed (JSON probe output verbatim) → Blocker → What's needed to unblock (account level upgrade, or paid data feed, or both) → Recommendation (hold/abandon the lane).

**Recon report (if Stage 1 runs):** `llm_reports/recons/2026-09-24_lab-option-variance-premium.md`, recons template, sections: Context → Data & window → Method (entry triggers incl. inverted chop veto, sizing, exits) → Results (daily returns, Sharpe vs SPY, per-trade R histogram, DSR/HLZ/PBO, trial count) → Falsification verdict → Files touched.

## 8. Deterministic test cases (mandatory)

`tests/test_option_probe.py`:
- Mock `TradingClient.get_account` returning `options_approved_level=2` → probe exits with verdict `false` and code `2`.
- Mock returning `=3` plus a chain with rows → verdict `true`, exit `0`.
- Mock chain empty → verdict `false` on data leg.

`tests/test_option_harvest.py`:
- **21-DTE exit:** a position opened at 40 DTE must be exited no later than the bar where DTE hits 21.
- **50% profit exit:** on a price path where the spread mid halves within 5 days, exit fires at the halving day, not later.
- **Inverted chop entry:** entry fires only on bars where `_compute_chop_veto_mask` is `True` (given IVR in window); a bar with veto=False and IVR=50 produces no entry.
- **IVR calculation:** synthetic 300-day IV series rising linearly → IVR at day 299 = 100th percentile.

## 9. Abort criteria
- Stage 0 probe verdict level < 3 → stop, write stop report, do not attempt Stage 1.
- Option chain history < 252 days → cannot compute IVR; report with "insufficient history" banner and no gate evaluation.
- Any leakage guard test fails after fixes.

## 10. Docs rule
Update `scripts/README.md` with entries for the two new scripts. Add `GLOSSARY.md` entries for "IVR", "variance premium", "iron condor", "credit spread", "defined-risk", "inverted chop veto".

## 11. What "done" looks like
- Branch `lane/option-variance-premium` holds only your files.
- Probe script exists and would exit deterministically on a mock account.
- If Stage 0 ran on the real account: report exists with the verdict reflected honestly.
- Tests green; compileall green.
- Commit `feat(research): lane4 option variance premium capability probe + harvest [conditional]`.
