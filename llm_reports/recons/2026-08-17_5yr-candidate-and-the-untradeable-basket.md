---
type: recon
date: 2026-08-17
time: 14:05 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "Brandon green-lit task 1 of the 2026-08-17 work order: decide whether to promote models/forex_m15_5yr over the live models/forex_m15_wide."
head: b76f375eafe9d69e920c617fdcd559a41958b293
scope: read-only
related:
  - recons/2026-07-27_threshold-ev-study.md
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
  - m2m_prompts/2026-08-17_score-compression-and-calibration.md
---

## Context

The 2026-08-11 timeframe study left `models/forex_m15_5yr` on disk as an
unpromoted candidate, recorded as scoring **profit factor 1.72 against the
shipped model's 1.37**. The question put to this session: re-run the numbers and
decide whether to promote it.

Two things came out of the re-run that matter more than the promotion verdict:
the original comparison was not measuring what it appeared to measure, and the
promotion gate itself is scoring the wrong basket of instruments.

## Investigation

### 1. The reported comparison was never apples-to-apples

Recovered the original gate logs (`logs/retrain_m15_wide_v2_2026-08-08_2255.log`,
`logs/retrain_m15_5yr_1538.log`). Both PF numbers are real and reproduce from the
logs. But `validate_candidate()` splits the **training window** into thirds
(`src/core/retrainer.py:1465-1470`), so the final out-of-sample fold scales with
the lookback:

| Model | Training span | Fold-3 OOS window | Fold-3 PF |
|---|---|---|---|
| `forex_m15_wide` (live) | 2024-08-09 → 2026-08-09 (2y) | ~2026-04-09 → 2026-08-09 (4 mo) | 1.3735 |
| `forex_m15_5yr` | 2021-08-12 → 2026-08-11 (5y) | ~2025-10-09 → 2026-08-11 (10 mo) | 1.7196 |

Different periods, different lengths. A longer lookback automatically buys a
longer, older test window. The two numbers were never comparable, so the
"1.72 beats 1.37" conclusion did not follow from them.

### 2. Matched-cutoff re-run

Re-ran both configurations with an identical end date (`RETRAIN_END_DATE=2026-05-01`),
identical everything else, only the lookback differing:

| Lookback | Fold-3 PF | Pooled OOS trades | Gate |
|---|---|---|---|
| 730d (live config) | **1.1538** | 334 | **FAILED** (< 1.20) |
| 1825d (candidate) | **1.8148** | 312 | PASSED |

The live configuration fails its own promotion gate once the test window moves
back three months. The 1.37 it shipped on was partly a property of the period.

### 3. A common-window head-to-head, and a result that looked wrong

Built a scorer that trains both configurations to a shared cutoff and grades
them on the *same* forward window, reusing the retrainer's own
`engineer_features_and_labels` / `refit_models` / `_find_optimal_threshold` so the
measurement matches the gate's. Devil threshold frozen on a pre-cutoff
calibration slice, never on the test window. First result: win rates of 66-72%
where the gate has never measured above 48%.

That was treated as a probable bug in the new harness, not a discovery.
Ruled out in order:

- **Test period unusual?** No. Base rate of a winning bracket is 25.1% in the
  test window against 25.0% over the full five years, and 23.9-26.0% in every
  individual year.
- **Look-ahead in the higher-timeframe features?** No. `V3HTFFeatures`
  (`src/ml/features/v3_features.py:463-478`) stamps each HTF bar with
  `available_at = bar_start + timeframe` and joins backward via `join_asof`.
  Correctly guarded.
- **Score distribution drifted from live?** No — it matches closely. Harness
  median 0.160 / p99 0.336 / 0.30% above the 0.40 bar, against live
  (n=2,424 bars, 08-11 → 08-17) median 0.160 / p99 0.326 / 0.25% above.

The actual cause: **the gate pools all 8 trained instruments; the head-to-head
scored only the 6 the broker will trade.** XAU/XAG have been untradeable since
2026-07-14 and the live bot drops them at boot (`untradeable_dropped` event).

### 4. Replication across three cutoffs

One window is one draw, so the head-to-head was repeated at three cutoffs, each
graded on the following 108 days, both configurations trained only on data
before the cutoff.

## Findings

**Finding 1 (high) — the gate's score is dominated by instruments the bot
cannot trade.** Across the three windows, metals were **38-60% of every model's
selected trades**, and their results are decoupled from the crosses:

| Test window | 6 crosses (net PF) | Metals (net PF) |
|---|---|---|
| Sep 2025 +108d | 0.56 / 1.32 | 2.20 / 7.07 |
| Jan 2026 +108d | 2.07 / 2.05 | 1.11 / 0.61 |
| May 2026 +108d | 2.70 / 3.46 | 0.29 / 0.59 |

(2-year / 5-year.) Note the sign flips: in the May window metals dragged the
pooled score **down** hard, in the September window they dragged it **up**. The
defensible claim is not "metals are bad" — it is that roughly half of what the
gate measures is untradeable and uncorrelated with what is, so the pooled PF is
a poor proxy for live performance in either direction. In the May window the
live config scores 1.20 pooled (90% CI 0.89-1.59, i.e. possibly losing) versus
2.70 on tradeable instruments alone.

**Finding 2 (medium) — the 5-year lookback is better or tied in all three
windows, but not separably so.** Six tradeable crosses, pooled across all three
windows:

| Config | Trades | W/L | Win rate | PF gross |
|---|---|---|---|---|
| 2-year (live) | 131 | 79/52 | 60.3% | 3.038 |
| 5-year (candidate) | 101 | 66/35 | 65.3% | 3.771 |

Per-window net PF: 0.56/2.07/2.70 (2-year) against 1.32/2.05/3.46 (5-year). The
candidate wins the weakest window outright, ties the middle, and wins the last.
Every per-window 90% confidence interval overlaps its counterpart, so this is a
consistent direction, not a demonstrated difference. The candidate is also
markedly more selective — 101 trades against 131, and far fewer metals picks.

**Finding 3 (medium) — the Devil threshold is an unstable parameter.** The
per-run frozen threshold came out 0.66, 0.10, 0.22, 0.60, 0.10, 0.66 across the
six fits, swept from only 12-92 calibration proposals. It also approves
96-100% of Angel proposals in every run, in the gate's own logs as well as here
(fold-3: 172→164, 107→103, 200→199). The Devil is close to a rubber stamp on
this data, and the threshold it ships with is close to noise.

**Finding 4 (low, positive) — the Angel's score is monotone.** Win rate by score
bucket on the May test window, 6 crosses: 19.0% / 25.0% / 26.6% / 27.2% / 28.0%
/ 23.7% / 48.0% / 61.3% / 72.7%. Rising, with the lift concentrated above 0.35.
The edge is not an artifact of where the threshold happens to sit.

**Finding 5 — the live 2W/9L record does not contradict the above.** Nine of
those eleven fills ran under the *old* bracket geometry (1.0x stop / 2.0x
target), replaced on 2026-08-08. Under current geometry the live record is 1W/1L.
There is almost no live evidence yet either way.

## Verification

- Both original gate logs re-read; the 1.72 and 1.37 figures reproduce exactly
  from `Profit Factor (macro) = 368.00 / 214.00` and `= 228.00 / 166.00`.
- Matched-cutoff retrains run through the unmodified `src.core.retrainer` entry
  point with only env knobs set; both wrote to side dirs
  (`RETRAIN_MODEL_DIR=models/_ab_2yr|_ab_5yr`). The 2-year run was **rejected** by
  the gate and therefore saved nothing.
- The head-to-head PF formula matches the gate's exactly
  (`retrainer.py:1820-1826`): gross profit = wins x 4.0, gross loss = losses x 2.0.
- Net-of-spread charges each trade `alpha x ATR`, with per-instrument alpha the
  pooled median of 75 live `SPREAD_CALIB` observations per cross from August soak
  logs (GBP_JPY 0.339 … GBP_NZD 0.851).
- 90% intervals are 4,000-sample bootstraps over trades, seeded.
- Base rates, look-ahead guard, and score-distribution agreement with live all
  checked independently (see Investigation).
- **Live soak untouched throughout**: pid 374138 held ~23h uptime across all
  fetching and retraining, zero reconnects, zero rate-limit errors, positions
  empty, bars scoring normally.

## Risk & follow-ups

1. **Sample sizes are small.** 101-131 trades pooled, 10-76 per window. Every
   interval is wide. Nothing here is a precise estimate.
2. **Simulated entries skip live-only gates.** Gate C (time blackout, still an
   open TODO at `retrainer.py:770-775`), the post-stop cooldown, and the
   per-currency exposure cap are not modelled. Live will take fewer trades than
   the counts above.
3. **The measured configuration is not the artifact on disk.**
   `models/forex_m15_5yr` was trained through 2026-08-11; this recon validates
   the 5-year *configuration* at three earlier cutoffs. Promoting the existing
   artifact infers from one to the other.
4. **Recommended next step, ahead of any promotion:** make the gate score only
   broker-tradeable instruments. This is a change to what is *measured*, not
   what is *trained* — note that shrinking the training basket to 6/8 was tried
   on 2026-07-02 and rejected, so the two are not the same experiment.
5. The Devil's instability (Finding 3) deserves its own investigation; it may
   belong with the score-compression research already dispatched.

## Files touched

Read only:

- `src/core/retrainer.py` (195-270, 359-400, 858-1050, 1083-1135, 1296-1360,
  1368-1500, 1540-1830, 2165-2324)
- `src/ml/features/v3_features.py` (380-530)
- `logs/retrain_m15_wide_v2_2026-08-08_2255.log`, `logs/retrain_m15_5yr_1538.log`,
  `logs/retrain_m30_5yr_1604.log`, `logs/retrain_m45_5yr_1607.log`
- `logs/soak_2026-08-*.log`, `logs/events-2026-08-1[1-7].jsonl`, `logs/status.json`
- `models/forex_m15{,_wide,_5yr,_m30_5yr,_m45_5yr}/metadata.json`, `threshold.json`

Written (scratchpad only, nothing in the repo): matched-cutoff driver,
head-to-head scorer, three diagnostics.
