---
to: agentic-research-model (assignee TBD)
from: claude-opus-5
date: 2026-08-10
status: drafted
branch: feat/wider-brackets-and-rename
topic: Research candidate ML algorithms and trading strategies that could clear this project's cost floor; inventory what our existing harness can already test
result_commit:
related_memory: project_stop_width_spread_toll, project_v4_investor_no_edge_finding, project_m15_soak_2026-07-13
related_report: llm_reports/recons/2026-08-08_stop-width-and-the-spread-toll.md
---

# Research brief: what should we try next?

## 0. Your task, in one paragraph

We run two automated trading products. Neither has demonstrated an edge that
survives transaction costs. We have solid infrastructure and honest
measurement, and we are not short of engineering capacity — we are short of
**ideas that could plausibly clear the cost floor described in §3.** Research
and propose candidate approaches: machine-learning algorithms, target
formulations, and trading strategies. Rank them. For each, say what it would
take to falsify cheaply using the tools in §4. **Do not write production
code.** The deliverable is a research report (§7).

**Read §3 before proposing anything.** The single most common way to waste our
time is to propose something that predicts direction better without addressing
why our costs eat the prediction.

## 1. What the two products are

### Product A — "V5 OANDA forex bot" (the active one)

An intraday long-only bot on OANDA's practice (paper) account, 15-minute bars,
8-instrument basket. Architecture is **two-stage meta-labeling**:

- **Angel** (stage 1, LightGBM binary classifier) — direction. Label:
  `close[t+3] > close[t] + 1.0 x ATR(14)`, i.e. "does price run one typical
  bar-range within 45 minutes."
- **Devil** (stage 2, LightGBM, trained ONLY on Angel-approved rows, and given
  `angel_prob` as an extra input) — conviction. Label: "did the position
  survive 5 bars without the stop being hit."
- A trade fires only when `angel_prob >= 0.40` AND `devil_prob >= 0.66`.
- Exit is a fixed volatility bracket: stop at `2.0 x ATR`, target at `4.0 x ATR`
  (2:1), max hold 45 bars. Stops are enforced **in software** by the bot
  process, not by broker orders.
- Three pre-trade veto gates: (A) cost — reject unless `stop >= 3 x spread`;
  (B) regime — reject the bottom 20% of the instrument's own rolling
  volatility distribution; (C) time — reject a daily rollover blackout window
  when spreads blow out ~10x.

**Live record: 13 fills lifetime, 2 wins / 11 losses.** Small enough that it is
NOT statistically distinguishable from chance (P(>=11 stops | fair 2:1
geometry) = 0.14). Do not over-fit your thinking to those 13 trades.

### Product B — "V4 equities investor" (live on paper, monthly)

A monthly cross-sectional stock ranker on Alpaca paper. LightGBM **LambdaRank**
over a 96-name universe spanning 11 sectors (verified: `scripts/investor_universe.py`). Ranks by "will this land in the
top quintile of 60-trading-day forward returns," then holds an equal-weight
top-8 basket with a max of 2 names per sector, rebalanced monthly by cron.
Walk-forward validation: 504-day expanding train window, 60-day embargo,
60-day test folds.

## 2. Where each one actually stands (measured, not assumed)

**Forex bot — break-even after costs, at best.**

We measured bracket behaviour over 40,650 hypothetical entries (every bar,
6 tradeable crosses, ~3 months). Findings:

- A 1x-ATR stop / 2x-ATR target hits the stop **66.6%** of the time against
  **66.7%** for fair 2:1 geometry on driftless price. The bracket is fair at
  every width from 0.75x to 4x. Random entries return ~0.00 R gross, as they
  must — this doubles as the simulator's unbiasedness check.
- **The spread ate a median 40.2% of the stop distance** at the old 1x setting.
  A 2:1 bracket breaking even at 33.3% wins for free needed **46.7%**.
- We widened the stop to 2x on 2026-08-08 (toll -> 21.8%, break-even -> 40.6%)
  and retrained to match. The new model passed our gate at Brier 0.2182,
  PF 1.3735, 486 pooled out-of-sample trades.
- **But our gate scores profit factor GROSS.** Re-scored after the measured
  toll: the new model is **1.373 -> 1.004** and the previously-shipped model
  was **1.540 -> 0.878**. So the change moved us from losing to break-even,
  not to profitable. Macro win rate 40.7% sits almost exactly on the 40.6%
  break-even the bracket requires.

**Investor — no measured edge over a trivial benchmark.**

Walk-forward over 2.4 years out-of-sample: strategy 24.9% CAGR / Sharpe 1.52
versus **equal-weighting all 96 names** at 24.9% CAGR / Sharpe 2.18.
P(strategy beats benchmark) = 52%, i.e. a coin flip. Against 1,000 random
sector-capped 8-name baskets, the shipped model sits **below the median of
random**. Our retrain gate never tested lift-over-benchmark, only
lift-over-random — that is now fixed but the model still has no proven edge.

## 3. THE CENTRAL CONSTRAINT — read this before proposing anything

**On the forex side, transaction cost dominates predictive skill.**

Costs are paid per round trip and are roughly fixed in price terms; the
predicted move scales with the horizon. So the toll as a fraction of the
trade shrinks as the horizon lengthens:

| Stop width | Median spread as % of risk | Win rate needed at 2:1 |
|---|---|---|
| 1.0x ATR | 40.2% | 46.7% |
| 2.0x ATR | 21.8% | 40.6% |
| 3.0x ATR | 14.5% | 38.2% |

Per instrument at the old 1x setting the spread ate: AUD_JPY 31.8%, GBP_JPY
33.9%, EUR_JPY 35.4%, GBP_AUD 51.6%, NZD_JPY 53.8%, **GBP_NZD 60.9%**.

**Implications you should reason from:**

1. A proposal that improves directional accuracy by a few points but keeps the
   same holding period may still lose money. Quantify the cost impact.
2. Longer holding periods structurally help — but push us out of intraday
   trading into swing/position trading, which is a different product with
   different risks (overnight gaps, carry/swap charges, far fewer independent
   observations per year).
3. Trading *less often but better* is as valid a direction as trading more.
4. Anything that lowers cost per unit of predicted move is interesting:
   longer horizons, limit/passive entries instead of market orders, trading
   only the cheapest instruments, trading only the cheapest hours.

**A second hard constraint: statistical power.** At current thresholds we get
roughly 1–1.5 fills per week. Validating anything on live trades alone would
take years. Proposals must be falsifiable **offline**, on historical bars.

## 4. Tools genuinely at your disposal

Everything below exists, is tested, and runs today. 189 passing tests.

**Environment (verified 2026-08-10, not assumed):** Python 3.12.13 on Linux.
Installed: Polars 1.40, pandas 3.0, NumPy 1.26, LightGBM 4.6, scikit-learn 1.8,
TA-Lib 0.6.8, hmmlearn 0.3.3, SciPy 1.17. **Not installed but installable:**
PyTorch, TensorFlow, statsmodels.

Hardware: 12 CPU cores, 27 GB RAM, and an **NVIDIA RTX 3050 Ti Laptop GPU with
4 GB VRAM** (driver 610.43, no CUDA toolkit installed — runtime only). Treat
4 GB as the real ceiling: small sequence models over bar windows are feasible,
anything needing large batches or a big transformer is not. Our datasets are
~50k bars per instrument (~400k rows pooled), which is small by deep-learning
standards and large enough to overfit spectacularly — factor that into any
neural proposal.

**Data**
- **OANDA v20** (practice) — REST history + live streaming, forex and metals,
  any granularity from seconds to daily. We routinely pull 730 days of M15
  (~50k bars per instrument). ~68 instruments on the account.
- **Alpaca** (paper) — US equities/crypto, daily and intraday.
- **Fundamentals/macro** — yfinance + SimFin adapters behind a composite
  provider (`src/data/providers/`). The investor trains on 14 features, split
  evenly: 7 price-derived (`mom_3m`, `mom_6m`, `mom_12m`, `mom_12_1`,
  `reversal_1m`, `vol_60d`, `vol_120d`) and 7 fundamental (`roa`,
  `debt_to_equity`, `gross_profitability`, `gross_margin`, `operating_margin`,
  `net_margin`, `ebitda_margin`). `ebitda_margin` is currently DROPPED — it was
  populated in 3.3% of training rows and 0% at inference, a live train/serve
  skew — leaving 13 in use.
- Live spread telemetry: the bot records real bid/ask spreads per instrument
  per bar (`SPREAD_CALIB`), so cost modelling uses measured, not assumed, costs.

**Feature engineering**
- `src/ml/feature_pipeline.py` — a single pipeline shared by training AND live
  inference, which is what guarantees zero train/serve skew. 22 features today:
  10 single-bar indicators (RSI, PPO, NATR, Bollinger %B/width, SMA50 ratio and
  distance, log return, hour-of-day, relative volume), 4 higher-timeframe views,
  4 candle-shape microstructure measures, 4 one-hot session flags.
- `src/ml/regimes/hmm_regime.py` — Gaussian HMM hidden-state regime detection,
  wired but currently OFF in production.
- `src/ml/feature_stats.py` + `scripts/probe_model.py` — drift/attribution probe
  using PSI **with null calibration** (textbook PSI thresholds false-alarm badly
  on autocorrelated market bars — our nulls reach 8.3) plus TreeSHAP.

**Training + validation**
- `src/core/retrainer.py` — the full pipeline: fetch, label, chop-veto filter,
  3-fold expanding walk-forward validation, an automated promotion gate
  (Brier <= 0.30, EV >= 0.0005, profit factor >= 1.2 on the final fold, and a
  dynamic pooled-trade floor that scales with the veto rate), atomic model
  promotion, Discord reporting. `RETRAIN_MODEL_DIR` isolates candidates from
  production.
- Backtest/replay harness in `analysis_cache/` — bar-level bracket simulation
  charged real measured spreads, reproducing the live gates. Two worked
  examples committed: a threshold EV study and the stop-geometry study.

**Execution + ops**
- `src/execution/oanda_forex_orchestrator.py` — async live bot: streaming,
  reconnect with jittered backoff, in-flight bar recovery on reconnect,
  software stop/target monitor, position reconciliation at boot, entry guards
  (post-exit cooldown, max 2 positions per currency leg).
- `src/execution/risk_manager.py` — bracket sizing + the three veto gates,
  mirrored symmetrically in the retrainer so the model only trains on bars the
  live bot would actually have traded.
- JSONL telemetry (`logs/events-*.jsonl`) — every bar evaluation with its
  probabilities and outcome, every gate veto, every entry/exit.
- Cron watchdog for uptime; Discord notifications.

**What we do NOT have:** order-book/tick depth data, news or sentiment feeds,
options data, alternative data, or a live-money account. (We DO get raw tick
bid/ask from OANDA's stream, so tick-level spread and micro-timing are
available — but not depth-of-book.)

## 5. Known dead ends — do not re-propose without new evidence

Each was tested and rejected with numbers:

1. **Lowering the entry threshold below 0.40** to get more trades. Out-of-sample
   after costs: the 0.35–0.40 band ran PF 0.12 / 7% win rate; 0.325–0.35 was
   break-even; below 0.325 bled. Only >= 0.40 was profitable.
2. **A metals-only model** — failed the gate (PF 1.05, weak separation). Also
   moot: our account cannot trade XAU/XAG (`INSTRUMENT_NOT_TRADEABLE`).
3. **1-minute bars** — only metals cleared the cost gate at M1, and metals are
   broker-dead for us. This is what drove the M15 migration.
4. **A calm G7-majors basket** — failed the integrity gate with no signal
   separation. The current basket is deliberately volatility-first.
5. **Shrinking the basket to 6 of 8** — rejected; the full basket is
   load-bearing even though we can only trade 6 of them.
6. **Adding an explicit cost feature (`cost_ratio`)** — failed the gate on
   calibration; a no-cost control on the same window also failed, so the window
   was the primary cause, but it did not help.
7. **Risk-adjusted or graded labels for the investor** (Sharpe-like targets,
   trailing-risk targets, 5-bucket graded labels) — no effect or worse.
8. **Position guardrails / stop-losses on the investor** — unsupported; the
   parameter sweep was non-monotonic and the apparent winners were
   buy-and-hold artifacts.

## 6. Open leads we already suspect (extend or kill these, and go beyond them)

- **Prediction horizon is the investor's real lever.** Retraining the ranker on
  a 10-day forward label instead of 60-day moved monthly excess return from
  −17bps to +110bps, with a smooth monotone gradient across 5/60 days and
  consistent sign across all 5 fold alignments. **Not yet significant**
  (t = 2.05, n = 32 months, and it was best-of-13 variants). The honest cure is
  more history (mine back to ~2015), which we have not done.
- **Probability calibration.** The forex Angel's probabilities are compressed —
  90th percentile around 0.24 against a 0.40 threshold — so the threshold sits
  far out in the tail and volume is starved. Recalibration (Platt/isotonic, or
  a loss/label change that widens the distribution) may matter more than a
  better classifier.
- **Our gate scores gross, not net.** Charging the measured spread inside
  `validate_candidate` would stop us promoting models that lose money after
  costs. This is a known fix, not yet done.
- **The Devil may be close to a rubber stamp** on the newest model: separation
  gaps of +0.042 / +0.007 / +0.021 across folds, approving 89–97% of what stage
  one proposes. Either fix stage two or replace the two-stage design.

## 7. What we want back

A research report. Prioritise **depth and honesty over breadth** — five
well-reasoned candidates beat twenty listed ones.

**Part 1 — ML algorithms / model formulations.** What should we consider beyond
LightGBM two-stage meta-labeling? Consider at minimum: sequence models on bar
windows; probability calibration methods; conformal prediction for
abstention-style thresholds; quantile/distributional regression to predict a
return *distribution* rather than a binary; direct optimisation of a trading
objective rather than classification accuracy; changing what we predict
(holding time, path-dependent outcomes, cost-adjusted returns). For each: does
it survive ~50k bars per instrument on 4 GB of VRAM, and what is its
overfitting risk at that sample size?

**Part 2 — Trading strategies / structures.** What structurally different
approaches suit a cost floor like ours? Consider at minimum: longer horizons
(H1/H4/daily swing); market-making or passive limit entry instead of paying the
spread; cross-sectional relative-value or pairs/statistical arbitrage across our
instrument set; carry; volatility-targeted position sizing rather than fixed
brackets; regime-conditional strategy switching; portfolio construction across
the 6 tradeable crosses instead of independent per-instrument trades. For each:
estimate the cost per unit of predicted move and compare against §3.

**Part 3 — Cheapest falsification path.** For every candidate, state what data
and how much work it needs, and design the smallest experiment that could kill
it. Explicitly mark which candidates can be tested with §4 as-is versus which
need new infrastructure, and say what that infrastructure is.

**Part 4 — Ranked shortlist.** Rank by (expected edge net of cost) x
(probability it survives honest validation) / (implementation cost). Recommend
the ONE you would do first, and say what would change your mind.

**Ground rules.**

- Cite real prior art with enough detail for us to find it. Distinguish
  peer-reviewed/replicated results from blog claims, and say which is which.
- Be explicit about what is speculative. We would rather have a well-labelled
  guess than a confident-sounding one.
- Assume an adversarial reader. We have already killed several of our own ideas
  with the numbers in §2 and §5, and we will do the same to yours.
- **Do not propose leverage increases, martingale/averaging-down schemes, or
  anything whose expected value depends on avoiding a stop.** Our stops are
  software-enforced and non-negotiable.
- If your honest conclusion is "the edge is not there and the cost floor is
  the whole story," say so. That is a publishable result for us, not a failure.
