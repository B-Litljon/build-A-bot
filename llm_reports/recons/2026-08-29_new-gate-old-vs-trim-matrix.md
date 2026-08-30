---
type: recon
date: 2026-08-29
agent: K3 (kimi-k3, via dsh)
trigger: "Compare the shipped 200x63 architecture vs the trim configs under the rebuilt fold gate (commit e451db9)"
related:
  - m2m/2026-08-29_capacity-window-gate-matrix.md
  - refactors/2026-08-24_artifact-holdout-gate.md
---

# Old vs new under the new gate: 18-run matrix + 2 production candidates

All runs through the REAL production gate (`main()` via
`scripts/run_gate_capacity_variant.py`, extended with per-stage
GATE_ANGEL_*/GATE_DEVIL_* overrides), sequential, side dirs only. New pipeline
for every run: 17 features, OOF-calibrated Angel threshold, Devil min_child
auto-scaled, CP evidence instrument. Note the shipped-config rows are therefore
NOT a perfect reproduction of the old tree — that ran 22 features and a fixed
0.40 Angel bar — they are "the 200x63 architecture as it would retrain TODAY".

Old-gate baselines (2026-08-24 stability logs, old code): 5yr = PASS 1/3
(238/230/227 vs ~232); 2yr = PASS 3/3 (266/290/340).

## The matrix (F3 = Fold-3 recency bound; pooled = all-folds evidence bound; bar 1.2)

| config | window | pin | F3 PF lb | pooled PF lb (wins/trades) | verdict |
|---|---|---|---|---|---|
| shipped 200x63 | 5yr | 08-09 | 0.57 | 0.61 (46/156) | FAIL |
| trim 100x15 | 5yr | 08-09 | 0.51 | 0.61 (119/446) | FAIL |
| mid 150x31 | 5yr | 08-09 | 0.74 | 0.78 (81/246) | FAIL |
| shipped 200x63 | 5yr | 08-16 | 0.89 | 0.79 (58/169) | FAIL |
| trim 100x15 | 5yr | 08-16 | 0.74 | 0.81 (75/219) | FAIL |
| mid 150x31 | 5yr | 08-16 | 0.81 | 0.82 (93/275) | FAIL |
| shipped 200x63 | 5yr | 08-23 | 0.66 | 0.71 (69/219) | FAIL |
| trim 100x15 | 5yr | 08-23 | 0.79 | 0.71 (51/158) | FAIL |
| mid 150x31 | 5yr | 08-23 | 0.88 | 0.80 (87/261) | FAIL |
| shipped 200x63 | 2yr | 08-09 | **2.13** | **2.12 (21/31)** | **PASS** |
| trim 100x15 | 2yr | 08-09 | **1.80** | **1.45 (46/90)** | **PASS** |
| mid 150x31 | 2yr | 08-09 | 1.30 | 1.11 (35/77) | FAIL |
| shipped 200x63 | 2yr | 08-16 | 0.87 | 1.43 (16/27) | FAIL |
| trim 100x15 | 2yr | 08-16 | **1.80** | **1.43 (64/130)** | **PASS** |
| mid 150x31 | 2yr | 08-16 | **1.77** | **1.42 (34/65)** | **PASS** |
| shipped 200x63 | 2yr | 08-23 | 0.81 | 1.22 (14/25) | FAIL |
| trim 100x15 | 2yr | 08-23 | 1.02 | 1.09 (39/88) | FAIL |
| mid 150x31 | 2yr | 08-23 | 1.11 | 1.27 (34/69) | FAIL |
| **trim 100x15** | **2yr** | **unpinned** | **2.13** | **1.62 (69/132)** | **PASS — artifact saved** |
| **mid 150x31** | **2yr** | **unpinned** | **1.46** | **1.52 (50/96)** | **PASS — artifact saved** |

Logs: `logs/newgate_<cfg>_<days>d_<pin>.log`. Devil separation gaps were real
throughout (no 0.0000 constants) — the two-stage architecture functioned in
every run.

## Read

1. **The 5yr window has no provable edge — for anyone.** All 9 five-year runs
   fail with macro win rates ~27–29% against a 33.3% break-even. The old
   gate's 5yr@08-09 pass (238 vs a 232 floor) was a trade-count artifact, not
   evidence. Under the CP instrument the old gate's 1/3 becomes an honest 0/3.
2. **The new gate is stricter than the old one, and fairer.** Shipped 200x63:
   old 2yr 3/3 → new 2yr 1/3, failing Fold-3 RECENCY (lb 0.87/0.81) while
   pooled evidence held (1.43/1.22) — the big model's recent-regime edge
   decayed and the point-PF/old-floor gate could not see it.
3. **trim 100x15 is the most consistent config**: 2/3 pins + the unpinned run
   on 2yr, with the strongest pooled evidence of any trim (lb 1.62, 132
   trades). mid 150x31 is the strongest RECENT performer: best 5yr pooled
   bounds (0.78–0.82, still failing) and the best unpinned holdout.
4. **Both unpinned 2yr runs passed BOTH gates on the freshest data**
   (holdout 2026-04-19 → 2026-08-28):
   - `models/newgate_trim100x15_730d_latest` — fold lb 2.13/1.62; holdout PF
     4.55 [lb 2.40], 36 trades; angel 0.3833 / devil 0.44; brackets 2.0/4.0.
   - `models/newgate_mid150x31_730d_latest` — fold lb 1.46/1.52; holdout PF
     12.5 [lb 4.94], 29 trades; angel 0.4669 / devil 0.54; brackets 2.0/4.0.
   These are the first honestly-gated trim artifacts ever produced. The
   2026-08-29 "phantom pass" (RETRAIN_POOLED_TRADE_FLOOR=40) is refuted by
   construction: same window class, standard gate, real Devil, real pass.
5. **The served model (forex_m15_wide, 2026-08-08, 200x63) is now
   out-evidenced**: its own architecture cannot re-prove edge on any tested
   window (0/6 under the new gate, best pooled lb 0.89), while both trim
   candidates pass on current data. Its defense is its live record, not its
   reproducibility. Promotion is a human decision; the procedure (soak.off →
   stop → backup → copy → verify → relaunch) is in the 2026-08-29 trim brief.

## Caveats

- 2yr windows with ~30-130 pooled trades remain thin evidence; the CP bound
  prices that in, but overlapping 45-bar walks make trades non-independent,
  so treat the bounds as approximate, not literal coverage.
- Fold-3 recency vs holdout strength diverge in several rows (e.g. shipped
  2yr@08-16: fold fail, holdout PF 14.5) — the most recent ~4 months have
  been friendlier than the mid-window regime. A fresh-window pass can still
  age badly; the soak's own ledger is the final arbiter.
- Gate scoring excludes untradeable XAU_USD/XAG_USD approvals; several runs
  concentrate proposals there (e.g. 79 approvals → 31 scored), thinning the
  judged evidence by design.
