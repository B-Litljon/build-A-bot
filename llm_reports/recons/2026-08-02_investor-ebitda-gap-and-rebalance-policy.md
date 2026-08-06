---
type: recon
date: 2026-08-02
time: 18:50 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "Investigate the ebitda gap; weigh monthly vs quarterly rebalance; add guardrails so the rebalance stops selling winners and cuts losers"
head: 6d0a74d4a709634253418c6f4f116046a9592ea3
scope: read-only
related:
  - 2026-07-03_1941.md
---

# V4 Investor: the EBITDA gap, rebalance frequency, and whether guardrails help

## Context

The 2026-08-01 monthly rebalance sold NUE (+15.2%) and CAT (−17.3%) alongside
five others. Brandon asked three things:

1. Why does the run log warn `'EBITDA' not found — skipping ebitda_margin`?
2. Monthly vs quarterly rebalancing — which is right?
3. Add guardrails so the rebalance keeps winners and cuts sinking ships.

Question 3 turned out to depend on a prior question nobody had asked: **does
the selection model beat a naive equal-weight basket at all?** It does not, in
the only window where it can be measured honestly. That reframes the guardrail
work and is the most important finding here.

## Investigation

### The EBITDA gap

`scripts/investor_feature_pipeline.py:101-107` builds margin ratios by dividing
named raw columns by revenue:

```python
_NUMERATOR_COLS: dict[str, str] = {
    "gross_margin":     "Gross Profit",
    "operating_margin": "Operating Income",
    "net_margin":       "Net Income",
    "ebitda_margin":    "EBITDA",        # <-- no such column
}
```

The raw frame (`data/raw/v4_investor_data.parquet`, 55 columns) has no column
named `EBITDA`. It has the two ingredients — `Operating Income` and
`Depreciation & Amortization` — but SimFin never exposes the combined line.

Coverage measured directly:

| Column | Non-null rows | Symbols |
|---|---|---|
| `Operating Income` | 92,602 / 120,480 (76.9%) | 80 / 96 |
| `Total Revenue` | 92,602 / 120,480 (76.9%) | 80 / 96 |
| `Depreciation & Amortization` | 42,654 / 120,480 (35.4%) | 42 / 96 |
| computable EBITDA (OpInc + D&A) | 42,654 / 120,480 (35.4%) | 42 / 96 |

The subtlety: `ebitda_margin` was **not** absent during training. It was
3,830 non-null of 114,624 rows — **3.3% coverage, 17 of 96 symbols**
(AMT, APD, CCI, COP, EOG, EXC, GIS, GOOGL, HON, JNJ, MO, O, PSA, SO, T, WELL,
XOM). Those 3.3% are enough for LightGBM to have built splits on the column —
it carries 1.96% of total gain in the deployed booster — but at inference the
column is 0% populated, so every one of those splits now takes the missing
branch.

This is train/serve skew, not merely a missing feature.

### Rebalance policy

The first backtest scored history with the deployed booster and produced
130% CAGR at −8% max drawdown. That is look-ahead, not skill: the deployed
model (`models/v4_investor_lgbm.txt`, trained 2026-07-03) was fit on the whole
window. Discarded.

Rebuilt walk-forward, mirroring `scripts/investor_train_model.py:94-96,286-300`
exactly — expanding train window, 60-day embargo (= the forward-return
horizon), 60-day test folds. 10 folds, 57,600 out-of-sample rows,
**2023-10-02 → 2026-02-23 (2.4 years)**. Every score used below came from a
booster that had never seen the row's date, nor anything within 60 trading
days before it.

Policies then replayed the live selection rule
(`scripts/portfolio_orchestrator.py:294-312` — walk the ranking top-down,
skip any name whose sector is already at `SECTOR_CAP=2`), equal-weighted,
held to the next rebalance.

## Findings

### 1. `ebitda_margin` is unfixable at useful coverage — drop it. (severity: low-moderate)

Computing `EBITDA = Operating Income + D&A` raises coverage from 3.3% to
35.4%, still leaving 54 of 96 symbols blank, because SimFin reports D&A on the
income statement for only 42 names. The provider
(`src/data/providers/simfin_fundamentals.py:69-99`) loads income, balance and
derived datasets — no cash-flow statement, which is where the remaining
companies report D&A.

Recommend removing `ebitda_margin` from `_NUMERATOR_COLS` and from
`FACTOR_COLS` (`scripts/investor_train_model.py:120-124`), then retraining.
**Either choice requires a retrain** — the model currently encodes "this
column is almost always missing," so populating it to 35% without retraining
would be a second, opposite skew.

Impact is small in isolation (1.96% of gain), but see finding 3: the model has
no measurable edge, so data-quality defects are no longer safely ignorable.

### 2. Monthly vs quarterly is undetermined — keep monthly. (severity: informational)

| Policy | Rebalances | Total | CAGR | Sharpe | Max DD | Turnover |
|---|---|---|---|---|---|---|
| monthly (live rule) | 28 | 67.2% | 24.9% | 1.52 | −8.2% | 53% |
| quarterly (live rule) | 9 | 40.9% | 17.3% | 0.92 | −7.1% | 75% |

Monthly looks decisively better. It is not. Bootstrap (10,000 resamples) on
the annualised gap:

```
monthly - quarterly: +8.1%   95% CI [-28.9%, +43.4%]
P(monthly better) = 68%
```

The confidence interval spans zero by a wide margin. Nine quarterly
observations cannot settle this. Keep monthly on the grounds that there is no
evidence to justify a change, and that monthly yields 3× the observations for
future evaluation — not because it is demonstrated superior.

### 3. The model does not beat equal-weighting the universe. (severity: HIGH)

| | Total | CAGR | Sharpe | Max DD | Hit rate |
|---|---|---|---|---|---|
| Strategy (monthly, live rule) | 67.2% | 24.9% | 1.52 | −8.2% | 64% |
| **Equal-weight all 96** | **67.4%** | **24.9%** | **2.18** | **−5.9%** | **79%** |

Identical return, materially worse risk, and 53% monthly turnover to achieve
it. Paired bootstrap on monthly excess return:

```
mean monthly excess: +0.04%   95% CI [-1.27%, +1.44%]
P(strategy beats benchmark) = 52%
```

A coin flip. Over this window the eight-name selection bought nothing that
owning all 96 equally would not have delivered, with less risk and no trading.

Caveat, stated plainly: 2.4 years, 28 monthly observations, a single strong
bull market (benchmark +67%), and a regime where breadth was wide — conditions
under which concentrated stock-picking has a structurally hard time adding
value. This is not proof the model is worthless. It is proof that its edge is
currently unmeasured and, at this sample size, indistinguishable from zero.

### 4. No guardrail configuration shows a real benefit. (severity: moderate)

Keep-incumbent-while-still-ranked-inside-N, swept:

| N | CAGR | Sharpe | Max DD | Turnover |
|---|---|---|---|---|
| 10 | 25.5% | 1.49 | −10.6% | 43% |
| 15 | 17.3% | 1.01 | −13.8% | 28% |
| 20 | 24.4% | 1.39 | −12.1% | 22% |
| 30 | 29.0% | 1.46 | −13.0% | 14% |
| 40 | 24.9% | 1.43 | −10.4% | 11% |

**The sweep is non-monotonic** — 25.5 → 17.3 → 24.4 → 29.0 → 24.9. A genuine
effect varies smoothly with the parameter. This does not. It is noise, and
picking N=30 because it printed 29.0% would be fitting the backtest.

The rows that looked spectacular are artifacts:

| Policy | CAGR | Turnover | Distinct baskets in 28 months |
|---|---|---|---|
| 15% stop only | 45.2% | 4% | **3** |
| 25% stop only | 37.6% | 4% | 3 |
| sinking-ship only (1m < −10%) | 32.2% | 8% | **7** |

Three distinct baskets over 28 months is buy-and-hold of the October 2023
basket (`AMD, BKNG, CAT, DE, GM, NFLX, NUE, ORCL`) through a bull market.
n=1 draw of eight stocks. Not evidence about stops.

Tested honestly — alongside rank-keeping, so the basket still turns over — the
sinking-ship rule **hurts**:

| Policy | CAGR | Sharpe |
|---|---|---|
| monthly + keep≤20 | 24.4% | 1.39 |
| monthly + keep≤20 + 15% stop | 23.7% | 1.35 |
| monthly + keep≤20 + sinking-ship | **20.3%** | 1.17 |

There is a mechanical reason. The model's single strongest preference is
**high volatility**: `vol_120d` is 27.6% of total gain and correlates +0.80
with the score cross-sectionally; `vol_60d` +0.79. A deliberately
high-volatility basket has frequent −10% months that recover. A rule that
sells on a bad month systematically sells the rebound it was selected for.

### 5. NUE and CAT were sold by the sector cap, not by the model.

Confirmed in `logs/investor_rebalance_2026-08-01_1630.log:887-888`:

```
skipping CAT    (sector industrials already at cap 2)
skipping NUE    (sector materials already at cap 2)
```

Re-scoring the 2026-07-31 inference frame: **CAT ranked #4 of 96, NUE #7** —
both comfortably inside the top 8. UPS and DE outranked CAT within
industrials; FCX and NEM outranked NUE within materials. The premise behind
"stop selling winners" is real, but the mechanism is the diversification cap,
not a model judgement about those names.

The model does mildly penalise recent gains (`reversal_1m` correlates −0.14
with score; `mom_12_1` excludes the most recent month by construction). NUE's
`reversal_1m` sat at the 94th percentile. That is a nudge, not the cause.

## Verification

- Walk-forward score generation mirrors the shipped trainer's constants and
  fold boundaries (`TRAIN_DAYS=504`, `EMBARGO_DAYS=60`, `TEST_DAYS=60`); fold
  boundaries printed and inspected — no test window overlaps its training
  window, and the 60-day embargo covers the forward-return horizon.
- The in-sample run was retained deliberately as a control: 130% CAGR /
  −8% max DD versus 24.9% / −8.2% out-of-sample quantifies how badly
  in-sample scoring flatters this pipeline.
- Feature coverage counted directly from the raw and training parquets, not
  inferred from the warning text.
- Rank claims for CAT/NUE reproduced by re-scoring
  `data/processed/v4_inference_features.parquet` at its latest date
  (2026-07-31) with the deployed booster and cross-checked against the
  orchestrator's own skip lines in the run log.
- Turnover artifacts confirmed by counting distinct baskets per policy rather
  than trusting the turnover percentage alone.

**Not verified / limits:** no transaction costs or slippage modelled (this
favours the high-turnover live rule, so the turnover reduction in finding 4 is
worth somewhat more than the table shows). Single window, single regime. The
universe is survivorship-clean by construction (fixed 96 large-caps chosen in
2026) — a real bias that inflates every number here, benchmark included.

## Risk & follow-ups

Recommended order — data first, then evaluation, then behaviour:

1. **Drop `ebitda_margin`, retrain.** Removes the skew. Small, safe, unblocks
   any honest measurement that follows.
2. **Add an equal-weight-universe benchmark to the retrain gate.** The gate
   (`scripts/investor_train_model.py`, mirroring `src/core/retrainer.py:214-219`)
   currently tests lift over *random* (P@K vs the 0.20 base rate). It passed
   on 2026-07-03 while the model was not beating equal-weighting. A gate that
   cannot distinguish "better than random" from "better than doing nothing"
   is the defect that let finding 3 go unnoticed.
3. **Defer the guardrails.** Nothing in the data supports adding them, and
   tuning exit rules on a selection stage with no measured edge is polishing
   the wrong surface.
4. **One guardrail is defensible on non-backtest grounds:** keep an incumbent
   while it still ranks inside the top 20. It cut turnover 53% → 22% for a
   statistically identical return (24.4% vs 24.9%, well inside the noise
   band). Justify it as cost and churn reduction, which the untested-costs
   caveat makes conservative — not as a return improvement.
5. **Longer window.** 2.4 years of out-of-sample is the binding constraint on
   every question here. The universe was widened to 96 names on 2026-07-03
   (commit aebcf06); history before that is reconstructable but was never
   traded.

## Files touched

Read-only. Files read:

- `scripts/portfolio_orchestrator.py` (1-60, 90-115, 294-312, 380-450)
- `scripts/investor_feature_pipeline.py` (1-60, 101-155, 274-305)
- `scripts/investor_train_model.py` (88-160, 280-410)
- `scripts/investor_universe.py` (1-35)
- `src/data/providers/simfin_fundamentals.py` (60-135)
- `logs/investor_rebalance_2026-08-01_1630.log`
- `data/raw/v4_investor_data.parquet`,
  `data/processed/v4_{training,inference}_features.parquet`,
  `models/v4_investor_lgbm.txt`

Analysis scripts (scratchpad, not committed): `walkforward_scores.py`,
`rebal_backtest.py`, `rebal_sweep.py`.
