# STAGE 6 — Daily-bar crypto trend engine (long/flat, BTC/ETH)

You are implementing Stage 6 of the quant-architecture roadmap. Independent
of stages 1-5 (separate orchestrator, separate data path). NOTE: stage 5
(H1/H4 strategy-library re-sweep) is DONE — 2026-09-08, both timeframes
negative (H1 gross EV -0.000R across 45 cells, H4 -0.025R; only positive
cell was n=3). Do not re-run it; read `logs/strategy_matrix_H1.csv` and
`logs/strategy_matrix_H4.csv` for the evidence.

## Hard rails

Same standing rails as every stage (see stage-2 file, "Hard rails").
Stage-specific:

- Benchmark = BUY-AND-HOLD BTC, not zero. A prior equities effort in this
  repo passed every "beats random" gate while losing to the trivial
  benchmark. Do not repeat it. Gate on a t-statistic of the excess.
- Fees are the cost, not spread: 0.15-0.25% round trip (Alpaca crypto). Bake
  a fee table as the analog of `config/spread_alphas_m15.json`.
- Paper only. No capital. The soak is live and untouched by you.

## What to build

1. `src/execution/crypto_trend_orchestrator.py` — daily bars via
   `src/data/alpaca_provider.py` (already exists), long/flat:
   hold BTC/USD, ETH/USD when close > regime filter (reuse
   `donchian_breakout.py` / `sma_crossover.py` through `BaseStrategy` at
   daily granularity), flat otherwise. State machine mirrors the forex
   orchestrator's FLAT -> PENDING -> IN_TRADE -> ... pattern but simpler.
2. Inverse-vol sizing: w_t = min(1, sigma_target / sigma_hat_t), sigma_hat =
   30-day EWMA stdev of daily log returns, sigma_target default 40%
   annualized. Recompute daily on the sealed bar. Floor at min-notional,
   cap at broker max.
3. Offline evaluation first: run the rule over ~5 years of daily data
   (Alpaca can serve it), realised accounting (fees on every turn),
   benchmarked vs buy-and-hold with a t-stat on the excess, n>=30 daily
   decisions.
4. Shadow mode: the orchestrator logs decisions to
   `logs/crypto_trend_decisions.jsonl` without placing orders, until a human
   flips a flag. >=60 daily decisions logged before any live discussion.

## Promotion bar (offline, before shadow even starts)

- Net (after fees) t-stat of daily excess vs buy-and-hold >= 2.0 on the
  full window AND positive in >= 2 of 3 non-overlapping 18-month sub-windows
  (sub-windows are not independent — say so when reporting; the point is
  regime robustness, not statistical power).
- Max drawdown (with inverse-vol sizing) < buy-and-hold's max drawdown.
- Full test suite green (`PYTHONPATH=src:. venv-python -m pytest -q`).

## Deliverable

Recon note in `llm_reports/recons/` with the equity curves' summary stats
and raw outputs. Beating buy-and-hold on 4 years of BTC with a simple
trend rule is genuinely hard — a null result is a legitimate deliverable,
report it honestly if that is what the data says.