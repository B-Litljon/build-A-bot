---
type: recon
date: 2026-09-14
time: 14:10 PDT
agent: DeepSeek Harness (dsh, web)
model: deepseek-v4-flash
trigger: "Rounds of barrier measurement kept returning edge-less populations. Tracing why led to the served artifact's provenance: the live bot runs a model whose calibration and gate bugs were fixed nine days ago, and no artifact in the tree postdates those fixes."
head: f0d65089756646b6e389796b96d1a057939bc67f
scope: read-only
related:
  - audits/2026-09-08_high-benefit-fixes-ranked.md
  - recons/2026-09-14_bracket-population-and-live-trade-rate.md
  - recons/2026-08-29_trim-100x15-promotion.md
  - m2m-prompts/2026-09-14_barrier-live-seam.md
files_touched: []
---

# The served model is a pre-fix artifact — the retrain, not the bracket, is the unlock

## Context

Seven rounds of barrier work concluded that the bracket is not the problem: the
learned geometry adds nothing over a constant wide bracket (+0.0037R), bracket
*width* is worth ~0.08R net, and every population, τ pair and arm is net negative
because gross expectancy is ~0. That left one question — if not the bracket, what?
Tracing it pointed at this repo's own audit trail, and then at the **provenance of
the artifact the soak is running**.

## Investigation

Read `audits/2026-09-08_high-benefit-fixes-ranked.md` items 19–22, then verified
each claim against the current code rather than the comments, then surveyed every
`models/*/metadata.json` for what was actually trained and when.

## Findings

1. **Both pre-retrain defects the audit named are FIXED in the code** (2026-09-09,
   the day after the audit):

   - *Item 19 — OOF leakage into the calibration.* `TimeSeriesSplit` ran over row
     index on a symbol-blocked frame, so each validation fold was one or two whole
     symbol blocks and the "out-of-fold" Angel probabilities the production bar is
     calibrated from came from models that had seen other symbols' future dates.
     Now: the loop permutes by timestamp and writes probabilities back to original
     indices (`retrainer.py:1766-1786`), head filled only from the earliest
     window, and `generate_time_decay_weights` ranks by timestamp rather than
     basket position (`:1629-1636`).
   - *Item 20 — near-vacuous gate.* EV came from the **5-bar survival** win rate
     mapped through the **45-bar** R:R. Now: EV from macro targets
     (`ev = win_rate × (tp_mult/sl_mult) − (1 − win_rate)`, `:2020-2030`) with the
     threshold frozen on the penultimate fold for a strict-OOS final fold
     (`:2946-2975`).

2. **No artifact in the tree postdates those fixes.** Newest by `trained_at`:
   `sweep_mp150` / `sweep_mp600` (2026-08-31), the served `forex_m15_wide`
   (2026-08-30), and the five `newgate_*` candidates (2026-08-30). The only
   exception, `forex_h4_catboost` (2026-09-14), records `holdout.used = false` and
   is a side directory.

3. **Every one of them records a gate result the bracket cannot produce.** Recorded
   holdout EV: 1.17, 1.34, 1.40, 1.48, 1.50, 1.56, 1.583, 1.60, 1.69, 1.70, 1.71,
   1.81, 1.897, **2.00**. At a 2:1 bracket the maximum possible EV is +2.0R, which
   requires a 100% win rate. The served artifact is internally inconsistent:
   `expected_value = 1.583333` alongside `win_rate = 0.6944`, which implies
   +1.08R. That is the survival/macro mix-up, recorded.

4. **Therefore the served bar (0.3833) was calibrated on leaked probabilities and
   its promotion gate pass was vacuous.** This is the mechanism behind an entire
   cluster of observations that were previously read as separate symptoms: a bar
   sitting above the live distribution's maximum (per-symbol live maxima
   0.206–0.249), 3 proposals in 15 days, 0 fills, the 2026-09-08 graded ledger's
   *inverted* top Angel band (0.40+ → 11.8% win, n=34), and the "no measured edge"
   conclusion in the audit's own item 22.

5. **Audit item 21's fork is now decided — and it is not the barrier.** That item
   asked where the next evidence should come from: "retrained calibrated Angel vs
   barrier-driven bracket redesign". Rounds 4–7 measured the second half and
   answered it: the learned geometry adds nothing (0.0037R over a constant wide
   bracket), and the width it converges on is reproducible from `RiskProfile`
   alone. The evidence has to come from a **retrained, honestly-calibrated Angel**.

## Verification

- Code read at the cited line ranges (not the comments) for items 19 and 20, in
  `src/core/retrainer.py`.
- `models/*/metadata.json` surveyed programmatically; the table above is its
  output, including `trained_at`, `angel_threshold`, `holdout.expected_value`.
- Cross-checks from the companion recon
  (`recons/2026-09-14_bracket-population-and-live-trade-rate.md`): live heartbeats
  (`proposed=0/30 … vs threshold=0.38`), 3 agreements / 3 Gate B vetoes /
  0 positions since the promotion, and 14 of 56,728 bars passing both stages.
- Read-only: nothing was trained, promoted, or touched. The soak is untouched
  (same PID, five files in the served dir).

## Risk & follow-ups

- ⚠️ **The next retrain writes barrier artifacts into the served directory by
  default.** `retrainer.py:4173` calls `fit_and_save_barriers` on `if promoted:`
  with `RETRAIN_LEARN_BARRIERS` defaulting to 1. They would be ignored while
  `BARRIER_GEOMETRY_ENABLED` is off, but they arrive looking deployed and with no
  promotion verdict recorded. Set `RETRAIN_LEARN_BARRIERS=0` for the calibration
  retrain (or wire `RETRAIN_BARRIER_VERDICT` so they carry their own FAIL).
- **The unlock is a retrain**, and it is the human's call, not an agent's: it
  replaces the artifact the soak trades. It would produce the first artifact
  calibrated on honest OOF probabilities and the first judged by a gate that can
  fail — which also means it may legitimately **fail**, where its predecessor
  could not.
- **Expect the bar to move down and the proposal rate up.** At a bar inside the
  live distribution the bot will trade more often; the economics measured so far
  (gross EV ~0 at every population) say that more trades is not the same as better
  trades, so the graded ledger should be regenerated after the retrain and read
  before any interpretation of the new rate.
- **This does not resurrect the barrier feature.** Its status is unchanged:
  correct machinery, no demonstrated edge, default off.

---

## Addendum (same session, 14:20) — pre-flighted, and the headline recommendation is WRONG

I ran the retrainer's own validation path on the cached basket with the **fixed**
code (served hyperparameters 100×15, 17 features, nothing promoted) before anyone
spent a retrain on the recommendation above. Two results, both against it:

```
gate_passed : False
rejections  : ['Fold 3 PF point estimate 1.4545 on 19 trades, but 95% lower bound
               0.5965 < 1.2 — the most recent regime cannot prove it beats break-even',
               'Pooled fold PF 95% lower bound 0.7727 < 1.2 (13 wins / 30 trades)']
mean_ev +0.3655 (bar 0.0005) | folds: 2 / 9 / 19 trades, win 1.0 / 1.0 / 0.789

ANGEL BAR the fixed calibration would pin : 0.3564   (served today: 0.3833)
```

1. **The retrain is not "the unlock".** The leakage fix moves the calibrated bar
   by 0.027 (0.3833 → 0.3564) — same part of the distribution's tail, marginally
   more reachable. Finding 4 above stands (the served bar *was* calibrated on
   leaked probabilities), but "fixing it restores the bot" does not.
2. **A retrain today would be REJECTED**, and that is the gate working: EV-from-
   macro passes, the Clopper-Pearson PF lower bounds do not, on **19 and 30
   validation trades**. The served artifact would stay exactly where it is.
3. **The real constraint is evidence per unit of time.** Thirty pooled trades
   cannot clear a CI lower bound, and no bracket, τ pair or calibration change
   manufactures trades. The cheapest lever is the counterfactual one: the
   graded-decision ledger (`logs/graded_decisions.parquet`, ~15k decisions,
   regenerable) grades decisions the bot declined to take, growing the effective
   sample without pretending that relaxing the bar makes extra trades equally
   informative.

Caveat on the pre-flight itself: cached basket, `alpha_table=None` (no
`cost_ratio`), no holdout — a proxy for the real run, not the run. Its verdict and
its bar estimate are strong priors, not results of the shipped pipeline.

The one recommendation that survives unchanged: **keep `RETRAIN_LEARN_BARRIERS=0`**
for any retrain, so a rejected run cannot leave unverdicted barrier artifacts in
the served directory.

---

## Addendum 2 (same session, 15:05) — the Devil stage is trained on the wrong question

Addendum 1 found the served artifact pre-dates the calibration and gate fixes. This
extends the same class of finding one level down, into the second stage.

**The Devil predicts its own label and not the outcome that is traded** (honest OOF
probabilities, fixed algorithm, chronological permutation):

```
AUC(Devil score -> its own label, 5-bar survival) : 0.6609
AUC(Devil score -> MACRO 45-bar bracket outcome)  : 0.4722   (below chance)
AUC(Angel score -> MACRO 45-bar bracket outcome)  : 0.5378
```

0.4722 means its conviction is mildly *anti*-informative about the bracket the live
path places. That is not a bar problem — the stage answers a different question
(5-bar survival of a 2.0×ATR stop) than the system bets on (TP-before-SL over 45
bars at 2.0×/4.0×). It is audit item 20's defect one level down: item 20 fixed the
*gate* pricing survival through macro R:R; the Devil is still trained that way. The
repair is to train it on the macro bracket label, in the same retrain batch.

**And its pinned bar is decorative** (companion finding, round 11): pass rate
**1.000** at 0.44 on honest OOF probabilities, ~0.93 on live bars. So the live
system's only real selection is the Angel.

**A Devil cut looked spectacular out of window and should not be trusted:**
`angel>=0.25` plus the top-50% Devil score gave net **+0.817R ± 0.460 on 28
held-out trades**. Three reasons to discount it: it is one of eight cells searched
(~34% chance of at least one spurious positive); the direction contradicts the
score's own AUC of 0.472, so a proxy is more likely than a signal; and the cuts
are unstable across windows (the train-window top-10% cut selects zero holdout
bars). It is a candidate for one confirming run, not an action.

**Cheap decisive follow-up:** retrain the Devil on `devil_target_macro`, recompute
its honest OOF AUC against the macro outcome, re-run the cut sweep. Above ~0.55 and
the stage is salvageable with a proper bar; near or below 0.5 on this data and it
should be removed rather than tuned, which would also collapse `ml_strategy` to a
single-stage decision.

---

## Addendum 3 (same session, 15:20) — the Devil label fix is VALIDATED, and still not an unlock

Addendum 2 proposed retraining the Devil on `devil_target_macro` and said the
follow-up would decide "salvageable vs empty". It is salvageable — the fix is real
and window-stable — but it does not produce positive expectancy anywhere.

**The label change works** (same features, same chronological 5-fold OOF scheme):

| score | AUC all | AUC train 75% | AUC holdout 25% |
|---|---:|---:|---:|
| Devil — survival label (**ships today**) | 0.4722 | 0.4768 | **0.4564** |
| Devil — macro label (**proposal**) | 0.5839 | 0.5857 | **0.5806** |
| Angel (reference) | 0.5378 | 0.5380 | 0.5379 |

The shipping Devil is anti-informative and *worse* out of window; the macro-trained
one gains ~0.11 AUC and holds it. The two scores correlate at **−0.166**, which is
the mechanism: the second stage currently scores nearly the opposite of what
predicts the bracket. Same defect class as audit item 20, now confirmed at the model
level. **Train the Devil on `devil_target_macro` in the calibration retrain batch.**

**But expectancy does not follow.** Held-out final 25%, 6 cells pre-registered with
a Bonferroni z threshold of 2.64 (the discipline earlier rounds lacked):

| arm | keep | trades | win | gross | net | z |
|---|---:|---:|---:|---:|---:|---:|
| static | 100% | 583 | 0.280 | +0.056 | −0.162 | −2.85 |
| static | 50% | 538 | 0.294 | +0.089 | −0.133 | −2.23 |
| static | 25% | 380 | 0.316 | +0.138 | −0.093 | −1.31 |
| wide | 100% | 508 | 0.557 | −0.010 | −0.054 | −2.99 |
| wide | 50% | 476 | 0.567 | −0.009 | −0.053 | −2.86 |
| wide | 25% | 352 | 0.577 | −0.016 | −0.061 | −2.76 |

The filter moves in the right direction (win rate and net both improve as the cut
tightens, and the population survives — 1309 to 1698 bars, versus the 26-row tails a
survival-trained Devil produced) but **no cell is positive**, the best is
significantly negative, and everything sits below the 0.3333 break-even.

**Net:** one validated model-quality fix to add to the retrain batch; still no
configuration with positive net expectancy out of window. Four candidate cells have
now been produced by three rounds of searching and all four died under a control.

---

## Addendum 4 (same session, 15:55) — the served model is NOT damaged, and the calibration evidence cuts both ways

Regenerating the live graded ledger (it ended 2026-09-08) produced a reading that
contradicted the companion recon's honest-OOF curve: on the served model's live
probabilities the top Angel band **inverts** (0.40+ → 13.2% win, n=38, sweep
negative at every bar), where the cached-basket OOF curve had it at 39.0% (n=213).

Holding the rows fixed resolves it. Joining the ledger's own bars and outcomes to
honest OOF probabilities for the same bars (10,544 matched) and recomputing both:

| | correlation | AUC vs macro win |
|---|---:|---:|
| served live probabilities | — | 0.5210 |
| honest OOF probabilities | **0.9071** | 0.5220 |

Same rows, same answer key, same band edges — and **both** sources show the top
bands inverted, on 17 and 5 rows respectively. Two conclusions:

1. **The served model is not damaged.** Its live probabilities track a freshly
   retrained model at r=0.907 with an indistinguishable AUC, so the second-stage
   label defect (Addendum 3) is the only validated model-level fault.
2. **The top-band disagreement is a window effect**, not a model effect: 213 rows
   over two years versus 17–22 rows over the recent 45 days, and neither sample can
   overrule the other.

**Retraction.** Addendum 1's sibling claim — that the 2026-09-09 leakage fix
"un-inverted the top band" — compared *live decisions* (n=34) against the *cached
basket* (n=213), i.e. different rows. It was never a leaked-vs-honest comparison on
identical data, and the tie-breaker above shows both probability sources invert on
the same live rows. The fix remains justified by construction; the calibration
improvement attributed to it is unsupported and should not be relied on.

**What still stands:** the Devil label fix (Addendum 3), the retrain being rejected
for want of trades, and the evidence-per-unit-of-time wall — now measured from three
directions (validation folds 30 trades/2yr; ledger top band 17–38 rows/45 days;
holdout replays 11–109 trades/cell).
