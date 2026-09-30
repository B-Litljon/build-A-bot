# Live thread — five quant-lane dispatches (2026-09-24)

Convention: append-only, one block per message, per `llm_reports/m2m-prompts/README.md:30-39`. Claims marked VERIFIED / ASK / OFFER / BLOCKER / DECIDED. If you edit a file another agent may touch, say so.

| Lane | Branch | Worktree | Brief file | Agent |
|---|---|---|---|---|
| 1 Crypto trend + vol target | `lane/crypto-trend-mom-tv` | `build-a-bot-lanes/crypto-trend-mom-tv` | `llm_reports/handoffs/2026-09-24_lane1-crypto-trend-mom-tv.md` | dsh / deepseek-v4.1-flash (ollama-cloud) |
| 2 Equity factor PEAD | `lane/equity-factor-pead` | `build-a-bot-lanes/equity-factor-pead` | `llm_reports/handoffs/2026-09-24_lane2-equity-factor-pead.md` | same |
| 3 Forex cross-sectional RV | `lane/forex-cs-rv` | `build-a-bot-lanes/forex-cs-rv` | `llm_reports/handoffs/2026-09-24_lane3-forex-cs-rv.md` | same |
| 4 Option variance premium | `lane/option-variance-premium` | `build-a-bot-lanes/option-variance-premium` | `llm_reports/handoffs/2026-09-24_lane4-option-variance-premium.md` | same |
| 5 Falsification audits | `lane/falsification-audits` | `build-a-bot-lanes/falsification-audits` | `llm_reports/handoffs/2026-09-24_lane5-falsification-audits.md` | same |

### DECIDED — 2026-09-24 15:50, lead architect

- All five lanes run on detached worktrees at `b09fde2` (main, clean) so the live soak checkout is never branch-switched. Verified at dispatch time: soak PID 9158 active on the sibling checkout.
- All lanes share a `src/lab/stats.py` contract (DSR, CSCV PBO, HLZ). Lanes 1/2/3/4/5 are dispatched in parallel; whichever lane lands the module first, the others import it. Coordination on it lives here.
- Lane 4 is capability-gated: Stage 0 probe runs first and reports either level/data OK (then Stage 1 backtest) or BLOCKED (then stop report, no backtest).
- All lanes commit locally only. Brandon reviews and merges. Do not push.

### VERIFIED — evidence the specs rest on

- Main tree: `git -C /mnt/storage/mystuf/development/build-A-bot status --porcelain | wc -l` → `0`; HEAD `b09fde2` = merge of PR #73 (lab/w1-w4). The KB note "uncommitted at time of writing" is stale.
- No DSR/CSCV/PBO/HLZ/statsmodels anywhere in `src/` (grep, 2026-09-24) — the stats module is net-new.
- Crypto: `AlpacaProvider` has `crypto_client`; `_timeframe_for(1440)`→1Day, H4→4Hour, tested (`tests/test_alpaca_timeframe.py`). No crypto bars cached on disk; the 2026-09-14 study's bars were scratch files that did not survive.
- Equities fundamentals: SimFin quarterly income/balance caches populated at `data/raw/simfin_cache/` (45,936-row CSVs, quarterly). SUE/PEAD/estimates/publish-date machinery: zero present.
- Options: `alpaca-py==0.43.2` exposes `alpaca.data.historical.option.OptionHistoricalDataClient` and leg/order models, but **no dedicated options trading client** and no end-to-end verified options submission path (scout verdict).
- Tick/quote data: none anywhere. `data/oanda_provider.py:237` folds `(bid+ask)/2 into mid` and discards the spread. `config/spread_alphas_m15.json` is per-instrument scalars, not a time series.
- USD-cross pair availability for Lane 3's funding-index leg: unknown; practice account may 404 some crosses. Lane 3 spec has a documented fallback to the 6-pair baseline.
- Forex toll floor: 0.25R at served 2.0×/4.0× geometry; measured D1 crypto toll 0.0497R; BTC B&H +283% (all in `recons/2026-09-14_session-evidence-and-options.md`).
- `lab` `edge_over_random` is **win-rate units**, not R (`src/core/retrainer/_types.py:103-108`). Lane 3 spec includes the explicit pp→R conversion (1 pp ≈ 0.03R at 2:1).

### ASK — none at dispatch

### BLOCKER — none at dispatch

### OFFER — Lane 2/3/4/5 engineers: if your `src/lab/stats.py` is not yet present in your worktree when you start, implement the identical contract from your brief's stats section verbatim, and add a `### VERIFIED — stats.py landed on lane/<x>` block here.
