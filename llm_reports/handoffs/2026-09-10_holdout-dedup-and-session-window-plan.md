---
type: handoff
date: 2026-09-10
time: 18:20 PDT
agent: Kimi K3 (architect sub-agent)
model: ollama-cloud/kimi-k3
trigger: "Brandon asked for a course-of-action plan, as an agent-to-agent handoff, to act on the two structural gaps surfaced in recons/2026-09-10_holdout-loss-cluster-and-session-timing.md (correlated same-bar JPY-cross clustering + session-timing skew). Neither gap has a fix yet; that report was diagnostic only."
head: 535d95054c5097e414cbe5ff4d1f64de1b57258b
scope: read-only
related:
  - recons/2026-09-10_holdout-loss-cluster-and-session-timing.md
  - audits/2026-09-08_high-benefit-fixes-ranked.md
  - refactors/2026-09-09_safety-and-training-integrity-pass.md
  - handoffs/2026-09-06_architecture-brief-for-strategy-research.md
files_touched: []
---

# Plan — holdout cluster dedup + session window (for the implementing agent)

**Authored 2026-09-10 18:20 PDT · Kimi K3 (architect sub-agent). Read-only; no
code changed by this document.**

This is a plan and a set of decision points, not shipped work. It exists to be
handed to whichever agent (or Brandon) actually implements a fix. **No
implementation has begun.** Read the source recon report first; this document
assumes it.

## ⚠ FIRST: a correction to the source report (verify before building)

Per the Primary Evidence Rule, I read the code the recon report cites before
planning on it. **Finding #2 of that report is overstated and must be
corrected before anyone builds on it.**

The report claims (line 100–105, 144–148):

> "There is no correlation cap anywhere in the pipeline (confirmed by grep —
> no per-timestamp or per-direction dedup in `_evaluate_holdout` or the fold
> scoring path)... exists anywhere in the training-side scoring **OR the live
> orchestrator**."

The live-orchestrator half of that sentence is **wrong.** The orchestrator has
had a correlated-exposure cap since 2026-07-30:

- `src/execution/oanda_forex_orchestrator.py:355` — `self._max_per_currency =
  int(os.getenv("OANDA_MAX_PER_CURRENCY", "2"))`.
- `oanda_forex_orchestrator.py:767` — `_exposure_conflict()`, returns a block
  reason if opening would push a signed currency leg past the cap.
- `oanda_forex_orchestrator.py:750` — `_currency_legs()`, decomposes a
  symbol+units into signed legs (long GBP_JPY = `+GBP, -JPY`).
- `oanda_forex_orchestrator.py:1466` — the cap is checked and reserved in
  `_pending_entries` *under one lock* so two same-bar entries can't both pass.
- Documented in `src/execution/README.md:100–109` ("correlated-exposure cap
  (`OANDA_MAX_PER_CURRENCY`, default 2)").

So the true gap is **narrower and different** from what the report states:

1. **The live cap is a size limit, not a same-bar/same-direction dedup.** With
   the default `OANDA_MAX_PER_CURRENCY=2`, *two* of a 4-symbol same-direction
   JPY-cross cluster still pass. The cap limits *how much* correlated exposure
   accumulates; it does not treat 4 simultaneous same-leg fires as *one bet*.
   Whether that's the desired behavior is a design decision (see Decision D1),
   not a missing feature.
2. **The training-side scoring genuinely has NO mirror of any of this.** I
   grepped `retrainer.py` — `_evaluate_holdout` (2153), the fold gate
   (`_holdout_pf_lower_bound` calls at 3129/3132), and the PF computation
   (2272–2277) all treat every approved row as an independent Bernoulli draw.
   There is no currency-leg accounting, no same-timestamp dedup, nothing.
   **This is the real, unambiguous gap**: the *metric* over-counts correlated
   cluster events, so the gate's Clopper-Pearson lower bound is computed on a
   sample whose trials are not independent — understating the true variance.

The report's Finding #3 (Gate C window too narrow) I verified as accurate —
`_DEFAULT_BLACKOUT_ET = "16:55-17:30"` (`risk_manager.py:170`), DST-correct via
`America/New_York` (`risk_manager.py:467–482`, mirrored at
`retrainer.py:1204–1229`), and it does not cover the observed 21:00–01:00 UTC
loss cluster. That finding stands as written.

**Why this correction matters for the plan:** the report's framing invites
"build a correlation cap, none exists." Building that would duplicate the live
cap and still miss the actual defect (the training-side metric). The work below
is scoped to the *verified* gaps, not the report's broader claim.

## Context

The 2026-09-09 retrain's artifact holdout scored WR 14.6% / PF 0.34 (41 trades),
and a rerun scored WR 12.0% / PF 0.27 (25 trades). The recon showed two
structural patterns in the per-trade list: (a) 8/25 trades were two calendar
moments where 4 JPY-cross symbols fired on the same bar in the same direction
(both clusters lost), and (b) essentially all remaining approvals sat in the
21:00–01:00 UTC NY-close→Asia-open chop band that Gate C doesn't cover.

The decision Brandon faces, and that this plan supports: **whether to
prototype (a) a training-side correlation-dedup so cluster events score as one
trade, (b) a broader session stand-down than Gate C, both, or neither.** Any
prototype is a training-side experiment in a side model dir first — no live
effect — matching how every candidate change in this repo has been staged.

## The invariant this plan protects

The single most important invariant in this codebase (per
`handoffs/2026-09-06_architecture-brief-for-strategy-research.md` §2) is
**train/live symmetry**: the training pipeline and the live bot must evaluate
the *same* population of trades, or the model learns from a distribution it
never trades on. Both gaps below are symmetry breaks of the same kind — live
has a guard the training metric lacks. Every fix here must be applied
**identically on both sides or not at all**, or it becomes the next
quietly-wrong evaluation.

## Investigation (what I read to build this)

- `recons/2026-09-10_holdout-loss-cluster-and-session-timing.md` — the source.
- `src/core/retrainer.py` — `_evaluate_holdout` (2153), PF from macro wins
  (2272–2277), `_holdout_pf_lower_bound` (2304), fold-gate aggregate
  (3112–3134), `_compute_chop_veto_mask` Gate C mirror (1204–1229),
  `_tail_cutoff_by_symbol` (glossary 159). Confirmed: **no** correlation/dedup
  logic anywhere in scoring.
- `src/execution/oanda_forex_orchestrator.py` — `_max_per_currency` (355),
  `_currency_legs` (750), `_exposure_conflict` (767), reservation under lock
  (1466). Confirmed the live cap exists and is a *size* cap.
- `src/execution/risk_manager.py` — `_DEFAULT_BLACKOUT_ET` (170),
  `ENV_BLACKOUT_ET` (154), `_parse_blackout_et` (177), `blackout_start/end`
  fields (254–255), `for_asset_class("forex")` populates them (260–277),
  `time_gate_enabled` (292), `_in_blackout` (467). Confirmed window is
  env-overridable already via `RISK_BLACKOUT_ET`.
- `src/execution/README.md:100–109` — documented intent of the live cap.

## Decision points (Brandon must resolve these BEFORE implementation)

These change what gets built. Do not let the implementing agent guess them.

**D1 — What does "one bet" mean for correlated clusters?**
The live cap is a size cap (default 2 per signed leg). The training metric
needs a *dedup* rule to count a cluster as one trial. Options:
- **(a) Same-bar, same-direction, same-quote-currency** collapse: group approved
  rows by `(timestamp, direction, quote_ccy)` and score each group as ONE trade
  (pooled outcome: e.g. the macro_win they all share, since same-direction
  JPY-crosses resolve near-identically — verify this empirically before
  assuming). This mirrors the economic reality that they're one directional
  yen bet. **Recommended starting point** — it matches the observed cluster
  (all four `*_JPY` long on the same bar).
- (b) Collapse by signed *leg* (matches `_currency_legs` exactly) rather than
  quote currency. Broader — would also collapse e.g. long EUR_JPY + short
  EUR_USD (shared `+EUR`). More faithful to the live cap but harder to reason
  about for a *metric*.
- (c) Don't dedup the metric; instead tighten the **live** cap to
  `OANDA_MAX_PER_CURRENCY=1` and accept the smaller live sample. This changes
  live behavior, so it's a bigger decision and out of scope for a training-side
  experiment.

**D2 — Does dedup *help* or *hurt* the apparent metric, and do we care which?**
Collapsing the two 4-trade losing clusters into single losing trials *raises*
WR/PF (removes ~6 extra losses) but *reduces* n (smaller sample → wider
Clopper-Pearson bound → lower lower-bound). The gate uses the lower bound, so
dedup could paradoxically make promotion *harder*. Decide whether the goal is
a *more honest* metric (dedup, accept whichever way the bound moves) or a
*more permissive* one. **This plan assumes the goal is honesty** — dedup is
about not lying to the gate, not about passing it.

**D3 — Session window: widen Gate C, or add a separate Gate D?**
- **(a) Widen the existing Gate C window** via `RISK_BLACKOUT_ET` (already a
  live env knob — `risk_manager.py:154,260`). Cheap, no code. But the rollover
  spread-spike (what Gate C is *for*) and the post-rollover thin-liquidity chop
  (what the recon observed) are *different mechanisms* with different windows;
  conflating them under one env var muddies both.
- **(b) Add a separate session stand-down** (call it Gate D) covering the
  NY-close→Asia-open band, with its own window and its own training mirror in
  `_compute_chop_veto_mask`. More honest about the mechanism; more code; must be
  mirrored exactly on both sides. **Recommended** if the goal is a persistent
  behavioral fix rather than a one-off widening.
- Either way the window must be **defined in `America/New_York`** and mirrored
  in `retrainer.py`, or train and live diverge on the DST boundary (the rollover
  window is a different UTC hour in summer vs winter — `_in_blackout` handles
  this via `_NY_TZ`; any new gate must too).

**D4 — Is this worth doing at all right now?**
The standing evidence (architecture brief §5) is that transaction cost, not
entry logic, is what's killing M15 forex, and the 2026-09-06 ledger shows
net-negative EV at every threshold. Deduping the metric and widening a blackout
*improve the honesty of the evaluation* but do not create edge. Brandon should
decide whether these are gated behind "yes we still believe M15 forex is
salvageable" or are worth doing purely as metric hygiene. **This plan takes no
position** — it scopes the work; it does not argue for the strategy's
viability.

## Proposed work plan (only after D1–D4 are answered)

Ordered, each stage independently verifiable, each with a stop condition.

**Stage 0 — Empirical confirm (read-only, ~30 min). Do this first.**
Before changing any scoring, verify the two clusters actually resolved
identically. The recon *assumed* "same-direction JPY-cross on the same bar =
one bet 4x." Confirm: pull the 4 Aug-3 and 4 Sep-7 trades and check their
`macro_win` outcomes and realised moves are (near-)identical, and measure the
actual return correlation of the four `*_JPY` pairs on those bars. If they did
NOT move together, the clustering premise weakens and D1 changes. Also confirm
Gate C really doesn't already cover part of 21:00–01:00 in the *winter* (EST)
window — the blackout is NY-local, so in EST it's 21:55–22:30 UTC, which *does*
touch the bottom of the cluster band. The recon noted this; quantify it on the
actual trade timestamps before designing a window.

**Stage 1 — Training-side dedup prototype (side model dir, the metric only).**
Implement D1(a)/(b) as a post-processing step *inside* the holdout and fold
scoring paths in `retrainer.py`:
- Where `approved_mask`/macro wins are computed (`_evaluate_holdout` around
  2272–2277; fold path around 3032/3091–3096), group the approved rows by the
  chosen cluster key and collapse each group to one (wins, trades) contribution
  *before* `_holdout_pf_lower_bound` is called (3129/3132, 3861).
- Guard the new path behind an env flag (e.g. `HOLDOUT_CLUSTER_DEDUP`) so the
  committed behavior is unchanged unless opted in — same pattern as the
  debug-dump instrumentation the recon used and reverted.
- **Do not touch the live orchestrator in this stage** (unless D1(c) chosen).
- Run the full retrain recipe into a *fresh* side dir (never
  `models/forex_m15_wide`), with and without the flag, and report both the
  point PF/WR and the Clopper-Pearson lower bound each way. Success = the
  metric moves the way D2 predicts and the change is confined to scoring.

**Stage 2 — Session-window prototype (training mirror first).**
Whichever of D3(a)/(b) is chosen:
- Add the window in NY-local time, mirrored in `_compute_chop_veto_mask`
  (retrainer.py:1204–1229 lives next to the existing Gate C mirror) and, if a
  new gate, in `RiskManager` next to `_in_blackout` (risk_manager.py:467).
- The veto-mask OR-assignment at `retrainer.py:1288` (`veto[idx] |= ...`) is the
  pattern to follow — a new gate must OR in, not assign, to avoid clobbering the
  blackout mask (that bug shape is called out in the comment at 1286–1287).
- Report the per-symbol `gate_c` (or new `gate_d`) veto counts from the
  chop-veto log line (1292) so the band's aggressiveness is visible per pair.
- Re-score the *existing* 25-trade and 41-trade holdout dumps
  (`logs/holdout_debug_dump.csv`) under the new band, offline, before any
  retrain — how many of the losing trades would the new window have vetoed?
  This is a cheap dry-run against retained evidence.

**Stage 3 — Live-side mirroring (only if a prototype graduates).**
If (and only if) a side-dir retrain with the changes clears the gate and
Brandon approves, mirror the change live in the *same* commit: live cap change
in the orchestrator, session window in `RiskManager`. Train/live must change
together or the invariant breaks. Update `Glossary:` blocks, the folder
`README.md`, and `GLOSSARY.md` for any new identifiers per CLAUDE.md.

## Verification (how the implementing agent proves it worked)

- **Determinism check:** the dedup grouping is a pure function of
  `(timestamp, direction, symbol)` — assert no RNG, no dependence on row order
  beyond the existing `symbol,timestamp` sort. Add a unit test with a synthetic
  4-symbol same-bar cluster and assert it collapses to one trial.
- **Symmetry check:** for any session window, assert `RiskManager._in_blackout`
  (or the new gate) and the `_compute_chop_veto_mask` mirror agree on a grid of
  timestamps spanning a DST boundary (both a summer and a winter date).
- **Metric-sanity check:** recompute the Clopper-Pearson lower bound by hand on
  the deduped (wins, trades) for the 25-trade dump and confirm it matches the
  code's `_holdout_pf_lower_bound` output.
- **Full suite:** `PYTHONPATH=src:. <venv python> -m pytest -q` must stay green
  (baseline 411 passed / 5 skipped as of 2026-09-09) and
  `python -m compileall -q src/` clean before any commit.
- **Soak safety:** the live soak may be running (`ps aux | grep run_oanda`;
  `systemctl --user status soak.service`). Training-side-only stages must be
  provably read-only w.r.t. the live process and `models/forex_m15_wide`. Do
  not let a prototype retrain write anywhere the hot-reload seam
  (`ml_strategy._check_model_updates`) can pick up.

## Risk & follow-ups

- **Small-sample trap.** The whole effect rests on 25–41 trades, two clusters,
  and one session band. Dedup shrinks n further; a new gate shrinks the
  approved set. Any conclusion is fragile — this is metric *hygiene*, not a
  discovered edge. Weight the Clopper-Pearson bound, not point PF, and expect
  ±1 day of data to move it (the recon's Finding #1: retrains aren't exactly
  reproducible day-to-day).
- **Don't tune to one snapshot.** Re-validate any window against both the
  41-trade and 25-trade samples (the recon's own follow-up advice), or refetch
  fresh — a window tuned to the Aug-3 / Sep-7 clusters specifically is overfit
  to two events.
- **The correction above changes Finding #2's scope.** If Brandon reads only
  the recon report, he'll believe the live bot has no correlation protection,
  which is false (it caps at 2). Whatever is decided, the record should be
  corrected so the next agent doesn't plan against the wrong fact — consider a
  one-line erratum appended to the recon, or a reference back to this handoff.
- **Out of scope but adjacent (from audits/2026-09-08):** Gate A runs flat
  spread alpha 0.15 while soak calibrations measure 0.34–0.89; the
  `retrainer._compute_chop_veto_mask` Gate C mirror is a standing TODO.
  Interaction between those and any session-window change should be checked so
  fixes don't double-count or contradict.
- **Scratch artifacts** `logs/holdout_debug_dump.csv`,
  `logs/retrain_20260909_debug.log`, `logs/retrain_20260909.log` are retained
  evidence for Stage 0/2 dry-runs; safe to delete once reviewed.

## Files touched

_Read-only planning. Files read/grepped (already examined, for the next agent):_
- `llm_reports/recons/2026-09-10_holdout-loss-cluster-and-session-timing.md` (source)
- `src/core/retrainer.py` — `_evaluate_holdout` (2153), PF computation (2272–2277),
  `_holdout_pf_lower_bound` (2304), fold gate aggregate (3109–3134),
  `_compute_chop_veto_mask` + Gate C mirror (1159–1299), `_split_holdout` (976),
  `_tail_cutoff_by_symbol` (glossary 159)
- `src/execution/oanda_forex_orchestrator.py` — `_max_per_currency` (355),
  `_currency_legs` (750), `_exposure_conflict` (767), cap reservation (1466),
  `_pending_entries` (360)
- `src/execution/risk_manager.py` — `_DEFAULT_BLACKOUT_ET` (170),
  `ENV_BLACKOUT_ET` (154), `_parse_blackout_et` (177), `RiskProfile` blackout
  fields (254–277), `time_gate_enabled` (292), `_in_blackout` (467)
- `src/execution/README.md` (entry-guard documentation, 100–109)
