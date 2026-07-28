---
type: recon
date: 2026-07-27
time: "22:45 PDT"
agent: "Claude Fable 5"
model: claude-fable-5
trigger: "Post-soak threshold sweep proposed dropping the Angel bar 0.40 → 0.325–0.35 for volume; question = which threshold makes money AFTER the spread toll, and is the Devil really a rubber stamp?"
head: faff4c32b2ee5a1784c9e27f15f9e206206dd28b
scope: modifies-data
related:
  - refactors/2026-07-27_seam-catchup-and-threshold-hoist.md
files_touched:
  - models/forex_m15/threshold.json
  - analysis_cache/2026-07-27_m15_threshold_sweep/threshold_ev_study.py
  - analysis_cache/2026-07-27_m15_threshold_sweep/study_scored.parquet
  - analysis_cache/2026-07-27_m15_threshold_sweep/study_output.txt
---

# Threshold EV study: the proposed cut to 0.325–0.35 would have lost money

## Context

The Jul 13–27 soak ended with a threshold sweep (separate session) showing
0.40 sits at the 99.75th percentile of `angel_prob` and proposing a drop to
0.325–0.35 to reach ~2–3 fills/week. That sweep counted **signals**, never
outcomes. This recon answers the money question before any threshold surgery:
for each candidate Angel bar, what would the added trades have **earned after
the spread toll**, out-of-sample? Secondary question: is the Devil a rubber
stamp (its pass rate is 75–88% everywhere, and the 2026-07-21 currencies-only
retrain printed "NO SIGNAL"), or a real filter?

## Investigation

Method (`threshold_ev_study.py`, extends the sweep session's scaffolding):

- Re-scored **2026-04-20 → 2026-07-28**, all 8 basket instruments, through the
  frozen production `models/forex_m15` pair and the exact live
  `FeaturePipeline`. Scoring validated bit-identical against the independent
  sweep session on all 8,087 overlapping bars (max |Δprob| = 0.0e0).
- **Out-of-sample = bars after 2026-07-03 00:00 UTC** (the pair finished
  training 2026-07-02 08:03 UTC). Everything in the EV tables is OOS; the
  in-sample side is used only for the Devil diagnosis and shown as such.
- Simulated the live bracket long on every candidate bar (`angel_prob ≥
  0.25`): entry at bar close, SL = 1.0×ATR, TP = 2.0×ATR
  (`RiskProfile.for_asset_class("forex")`), SL-first on same-bar collision
  (conservative), 192-bar cap (timeout rate: 0.0%).
- **Cost**: each trade charged its instrument's median spread from the soak's
  shutdown SPREAD_CALIB (n≈830–870/instrument, clean non-holiday window):
  EUR_JPY 0.0145% … GBP_NZD 0.0405% of price.
- **Live gates reproduced**: Gate A cost (1.0×ATR ≥ 1.5×median spread —
  median approximation of the live tick spread; min-SL floors ignored, they
  bind at ~1/5 of a typical cross ATR), Gate B regime (NATR rank ≥ P20 in
  trailing 260, cold-start bypass < 60), Gate C rollover blackout
  (16:55–17:30 America/New_York).
- EV pooled over the **six broker-tradeable crosses only** — metals score but
  cannot trade on this account (`INSTRUMENT_NOT_TRADEABLE`, 2026-07-14).

## Findings / Changes

**1. The money lives only in the extreme tail. Keep 0.40.** (decision-grade)

OOS, tradeable six, marginal bands with the live funnel applied:

| band (angel_prob) | n | WR | avg %/trade | PF |
|---|---|---|---|---|
| ≥ 0.400 (today) | 4 | 75% | +0.023 | 2.96 |
| 0.350–0.400 | 14 | 7% | −0.047 | 0.12 |
| 0.325–0.350 | 11 | 45% | +0.000 | 1.01 |
| 0.300–0.325 | 30 | 47% | +0.002 | 1.06 |
| 0.250–0.300 | 312 | 35% | −0.018 | 0.59 |

The sweep's proposed band (0.325–0.35) is breakeven; the band just above it
(0.35–0.40) is sharply negative; everything below 0.325 is a wall of bleed
(at 0.25 the full funnel loses −0.017%/trade × 370 trades ≈ −6.4% in 3.6
weeks). Full-threshold (not banded) numbers tell the same story: angel-only
OOS PF crosses 1.0 only at ≥ 0.40 (PF 1.41, n=20; 0.45 → PF 1.28, n=11).
Dropping the threshold buys volume made of losing trades. "2–3 fills/week"
was a fine outcome and a bad target.

**2. The Devil is NOT a rubber stamp — it's an unproven light-touch filter.**
(reverses this morning's working verdict)

On its actual training population (angel_prob ≥ 0.40):

| window | n | pass rate | passed avg / WR | vetoed avg / WR | spearman(devil_prob, pnl) |
|---|---|---|---|---|---|
| in-sample | 133 | 86% | +0.108% / 87% | −0.047% / 33% | **+0.614** |
| out-of-sample | 20 | 90% | +0.004% / 61% | +0.076% / 100% (n=2) | −0.242 |

In-sample it genuinely separates winners from losers — the ~14% it vetoes are
bad trades. Out-of-sample the sample (20 rows, 2 vetoes) cannot support any
conclusion. The "100.0% approval / NO SIGNAL" evidence came from the
**rejected** currencies-only candidate, not this pair. Verdict: **keep the
Devil**; do not delete on current evidence; if the Angel bar ever moves, the
Devil must be retrained on the new population in the same change
(`retrainer.py` Phase 5.5 filter); re-run this pass/veto diagnostic once a
few dozen live fills exist.

**3. The volume bottleneck is the model's compressed probability ceiling,
not the threshold.** p90 of angel_prob ≈ 0.24 in both IS and OOS; ≥0.40 is
0.25% of bars. The distribution is stable across the boundary (p50 0.156 IS
vs 0.159 OOS — no drift story). More trades at the same quality requires a
model/calibration change (the LightGBM narrow-distribution thread), not a
lower bar. Meanwhile the shipped seam catch-up (~2× evaluated signals at any
threshold) and the watchdog are the honest volume recovery.

**4. Per-instrument, only AUD_JPY / EUR_JPY / GBP_JPY produce gate-passing
trades at 0.40 OOS.** GBP_AUD, GBP_NZD, NZD_JPY die at the cost gate (their
soak vetoes were all deep misses, not near-misses). With metals broker-dead,
the effective live basket is three JPY crosses — worth remembering when
setting expectations (~1–1.5 evaluated fills/week at current uptime).

**5. Change applied: pinned `angel_threshold: 0.40` into
`models/forex_m15/threshold.json`** (atomic temp+rename; `devil_threshold`
0.48 untouched). With the hoist (see related refactor report), the deployed
pair now carries its own Angel bar and `MLStrategy` prefers it over the env
default — an accidental `ANGEL_THRESHOLD` env var can no longer move the
live gate.

## Verification

- Scoring cross-validated bit-identical against the independent sweep
  session's `scored.parquet` (8,087 overlapping bars, max |Δ| = 0.0e0).
- Funnel counts reconcile with the live soak log: study OOS finds 20
  proposals ≥0.40 / 18 Devil-passed on the tradeable six vs the log's 22
  heartbeat proposals / 9 evaluated live (seam losses) / 2 gate survivors —
  consistent given the outage gaps the seam fix now closes.
- A NaN leak (final bars with no forward data to simulate) was found in the
  first run's low-threshold rows and fixed; corrected numbers are in this
  report and `study_output.txt` documents the raw run.
- Timeout rate 0.0% — every simulated trade resolved within the 192-bar cap,
  so the missing live time-exit doesn't distort outcomes.

## Risk & follow-ups

- **Small OOS n.** The decision survives it asymmetrically: the proposal
  needed the marginal band to be clearly positive, and it is breakeven-to-
  sharply-negative. But the ≥0.40 PF (1.41 on n=20) is an estimate, not a
  promise; the open profitability question stays open until live fills
  accumulate.
- Gate A uses the median spread, not per-bar live spread; min-SL floors and
  intra-bar path (which of SL/TP was touched first) are approximated
  conservatively.
- Follow-ups, in order: (1) relaunch the soak with the seam fix (branch
  choice is deliberate — the watchdog launches whatever is checked out);
  (2) accumulate ≥1 month of live fills at 0.40; (3) re-run the Devil
  pass/veto diagnostic on live outcomes; (4) consider a probability
  recalibration experiment (isotonic on OOF probs) as the volume play;
  (5) bracket geometry sweep only after there is a sample to fit on —
  `optimize_brackets.py` is import-dead (see refactor report) and its
  numbers predate the M15 era anyway.

## Files touched

Modified: `models/forex_m15/threshold.json` (pinned angel_threshold).
Written (gitignored analysis artifacts):
`analysis_cache/2026-07-27_m15_threshold_sweep/{threshold_ev_study.py,
study_scored.parquet, study_output.txt}`.
Read: `src/execution/risk_manager.py`, `src/strategies/concrete_strategies/
ml_strategy.py`, `src/core/retrainer.py`, `logs/soak_2026-07-13_1815.log`,
`logs/retrain_m15_currencies_only_2026-07-21_1554.log`, sweep session's
`scored.parquet`/`threshold_sweep.py`.
