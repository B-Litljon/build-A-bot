---
type: recon
date: 2026-08-29
agent: K3 (kimi-k3, via dsh)
trigger: "User decision: promote the trim 100x15 unpinned candidate to the live soak; retain the other artifacts for future testing"
related:
  - recons/2026-08-29_new-gate-old-vs-trim-matrix.md
  - m2m/2026-08-29_trim-100x15-retrain-and-promotion.md
---

# Promotion: trim 100x15 is now the served model (staged for the Sunday open)

## What was promoted and why

`models/newgate_trim100x15_730d_latest` → `models/forex_m15_wide/` (the dir
`soak.service` serves). The first honestly-gated trim artifact: fold gate
Fold-3 PF lb 2.13 / pooled lb 1.62 (69 wins / 132 trades), artifact holdout
2026-04-19→2026-08-28 PF 4.55 [95% lb 2.40] on 36 trades, Brier 0.1375.
Trained on the 2-year window through 2026-08-28, 17 features, brackets
2.0x/4.0x (matches the tree), behavior veto off. Thresholds pinned in
threshold.json: angel 0.3833 (OOF-calibrated), devil 0.44. Artifact shapes
verified before the swap: Angel 100 trees/15 leaves/17 features, Devil
18 features (17 + angel_prob).

Chose 100x15 over 150x31 for consistency: 2/3 pins + the unpinned pass, more
pooled trades (132 vs 96). 150x31 had the flashier unpinned holdout (lb 4.94)
on fewer trades (29) and only 1/3 pins.

## Promotion steps (executed 2026-08-29 ~19:20 PT, market closed)

1. No open positions in the current soak log; candidate artifacts verified.
2. `touch soak.off` FIRST (watchdog can't relaunch mid-swap), then
   `systemctl --user stop soak.service` — inactive within seconds.
3. Old model backed up to `models/forex_m15_wide_backup_20260829/`.
4. Five artifacts copied in (angel/devil pkls, threshold.json, metadata.json,
   feature_stats.json); served-dir metadata re-verified in place.
5. `rm soak.off`. The soak did NOT immediately relaunch: the watchdog's
   weekend blackout (soak_watchdog.sh, no cold starts Fri 14:00 → Sun 14:05
   PT) is holding it by design. **Relaunch is staged for Sun 2026-08-30
   ~14:05 PT (21:05 UTC)**, just after the forex open, booting cold into the
   new artifacts. The served dir was never touched while the bot ran.

## Sunday verification checklist (after ~14:10 PT)

- `systemctl --user is-active soak.service` → active; `logs/watchdog.log`
  shows a fresh `launch OK`.
- Newest `logs/soak_*.log`: clean model load, NO `SCHEMA MISMATCH`, no
  hot-reload CRITICALs; MLStrategy should report 17 features from
  `feature_names_in_` and the pinned angel bar 0.3833.
- First decisions at the new model's tempo: it proposes far more selectively
  than the old 200x63 (by design — the trim trades less, with a live Devil).

## Retained artifacts (held for future testing)

| directory | what it is | why kept |
|---|---|---|
| `models/forex_m15_wide_backup_20260829/` | the replaced 200x63 served model (2026-08-08, wide brackets) | rollback path + future A/B |
| `models/newgate_mid150x31_730d_latest/` | gate-passed 150x31 candidate (fold lb 1.46/1.52, holdout lb 4.94) | alternative candidate for future promotion/testing |
| `models/newgate_trim100x15_730d_202608{09,16}/` | 100x15 pin-pass artifacts | window-robustness reference |
| `models/newgate_shipped200x63_730d_20260809/` | the shipped config's only new-gate pass | comparison reference |
| `models/newgate_trim100x15_730d_latest/` | the promoted candidate itself (source of the served copy) | provenance |
| `models/_phantom_lowered_floor_20260829/` | the lowered-floor "phantom pass" + the copy that destroyed `models/forex_m15` | forensic evidence only — never promote |

## Rollback

If the new model misbehaves: `touch soak.off` → `systemctl --user stop
soak.service` → copy the five files from
`models/forex_m15_wide_backup_20260829/` back into `models/forex_m15_wide/` →
`rm soak.off`. The backup is byte-identical to what served the soak since
2026-08-08.

## Caveats

- The evidence behind this promotion is ~4.5 months of holdout plus a 2yr
  walk-forward on 132 pooled trades — real but young. The trim's live trade
  frequency will be lower than the old model's; judge it on the soak ledger
  (per-trade PF against the 2:1 bracket), not on daily P&L noise.
- The served model it replaces could not re-prove edge on any tested window
  under the new gate (0/6); staying on it was the weaker-evidence choice.
  This promotion swaps "old, unreproducible, but live-soaked" for "new,
  gate-proven on fresh data, but unsoaked" — the soak from Sunday onward IS
  the confirmation experiment.
