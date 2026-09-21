---
type: recon
date: 2026-09-20
time: 21:50 PDT
agent: opencode
model: deepseek-flash
trigger: "Brandon asked to test the one combination the 2026-09-14 decision view left standing: the near-top Angel band served on a wide bracket. Re-price the live decision ledger under alternative geometry, band by band."
head: b6dabf3464ce16073eda53b9990d7f8f3f0793fb
scope: "one new script + cached bars; no production weights, no live config touched"
related:
  - recons/2026-09-14_session-evidence-and-options.md
  - m2m-prompts/2026-09-14_barrier-live-seam.md
  - audits/2026-09-08_high-benefit-fixes-ranked.md
files_touched:
  - scripts/reprice_band_geometry.py
  - scripts/README.md
  - analysis_cache/2026-09-20_reprice_band_geometry/ (bars + reprice_trades.parquet)
---

# The wide-bracket re-price, decided — the near-top band does not transfer

## What was asked, and what was built

The 2026-09-14 session closed every lever on the cost side except one and left a
specific hypothesis: the model's score has a small real edge in the top decile
(+2.2pp over base, p=0.03) and none at the extreme top where the live bar
(0.3833) sits; a wide bracket cuts the spread toll ~6x. Serve the near-top band
on the wide bracket and the two effects might meet above zero.

`scripts/reprice_band_geometry.py` prices exactly that with **no retraining and
no live change**: it re-walks every row of `logs/graded_decisions.parquet`
against fresh OANDA M15 bars under four geometries and reports per-band win
rate, gross R, spread toll and net R.

Conventions mirror the surviving 2026-08-08 studies (long-only, entry at bar
close, SL-first on a same-bar collision, spread charged once from the soak
SPREAD_CALIB medians, R normalised by stop distance). The static arm must
reproduce the ledger's own `won` column — **it does, on 99.94% of 18,625 fiat
decisions** (2026-07-31 → 2026-09-18), so the bars, ATR and walk convention
match the report's answer key.

## The result

Net R per trade, by arm. `certified` = rows where the Angel and Devil both
approved, or the Devil vetoed an Angel approval (`verdict ∈ {agreement,
devil_veto}`, n=38) — the live bar's certificate.

| arm (stop × target × hold) | pooled n | pooled win | pooled net R | top-decile net R | certified win | certified net R |
|---|---:|---:|---:|---:|---:|---:|
| static 2.0/4.0/45 *(live)* | 18,625 | 0.249 | **−0.288** | −0.268 | 0.158 | **−0.516** |
| wide 10.25/2.74/45 | 18,625 | 0.498 | **−0.081** | −0.102 | 0.474 | −0.108 |
| wide 10.25/2.74/192 | 18,625 | 0.719 | −0.078 | −0.098 | 0.684 | −0.076 |
| best 8.0/1.0/90 *(60-sweep best)* | 18,625 | 0.848 | −0.085 | −0.100 | 0.816 | −0.045 |

**No arm, band, quintile, top-decile or certified population is positive at any
geometry.** The best measured cell in the whole run is the wide arm's
within-symbol 4th quintile at −0.0516R; the near-top bands the hypothesis
depended on are −0.060 (0.15–0.20), −0.061 (0.20–0.25), −0.097 (0.25–0.30).

## The reading

1. **The toll budget reproduces.** Static toll per trade averages ~0.27R and
   wide ~0.052R — the same ~5x dilution the 2026-09-14 recon measured
   (0.246R → 0.041R), on an independent window and population. The cost model
   was right.
2. **Widening cuts the bleeding but cannot stop it.** The certified population
   goes from −0.516R/trade (win 15.8%) to −0.108R/trade (win 47.4%) — roughly
   5x less bad, and still losing. Every population improves; none crosses.
3. **The near-top band does not transfer to the wide geometry — it inverts.**
   At wide, the top decile (−0.102) and the certified rows (−0.108) are *worse*
   than the average row (−0.081). The score carries no positive relation to the
   wide-geometry outcome. This is the hypothesis's kill shot: it was reasonable
   to expect the static-label edge to be geometry-specific, and it is — but the
   measured transfer is *negative*, not merely absent.
4. **The live certificate is the worst population in the file.** At the served
   geometry the rows the bar approves net −0.516R/trade. The 0 fills are not
   unlucky; they are the only thing separating the soak from its worst cohort.
   This is the same inversion the 2026-09-20 decision report flagged, now
   priced in R.
5. The retrained-on-wide-labels variant is **formally still untested** — these
   are static-model scores. But the re-price removes the reason to expect it to
   pass: wide random-entry gross ≈ 0, so a wide-trained model must supply
   ~+0.05R of genuine selectivity, and the current model's demonstrated
   selectivity (+0.045R at its home geometry) does not survive the change of
   target at all.

## What this changes

- **Do not change `RiskProfile.for_asset_class("forex")` to the wide numbers.**
  It is a real improvement in the loss rate (−0.29 → −0.08R/trade) and still a
  losing configuration, it requires extending `RISK_SIZING_ENABLED` beyond
  learned geometry to avoid ~5x dollar risk at fixed units
  (`oanda_forex_orchestrator.py:1486` guards on `GEOMETRY_BARRIER`), and the
  gate would reject it — correctly.
- **Do not run the wide-label retrain as an edge play.** If it is run for
  information, pre-register the readout: top-decile net R at the wide geometry
  must be positive out of window; if it inverts like the static model's, stop.
- **Options 3 and 5 of the 2026-09-14 decision view are now jointly closed**
  for this model family and basket. The remaining edge-side work is unchanged:
  metals (a human decision about the account), a genuinely different
  feature/target design, or a different market.

## Caveats, recorded so nobody re-derives them

- Scores are the served static-trained artifact's. A wide-label retrain would
  produce different scores; this script does not and cannot price that.
- Window is 49 days of live decisions; the certified subsets are n=38.
- Tolls are fixed per-instrument medians from the 2026-07-27 soak; live spreads
  vary around them. Timeouts exit at market close; the live bot has no time
  exit. Hold caps are 45/90/192 bars by arm.
- The `devil_veto` subset (n=8) is positive at wide (+0.11R/trade). Consistent
  with the 2026-09-14 note that the Devil is anti-correlated with outcome, but
  n=8 and explicitly **not evidence**.

## Reproducing

```
PYTHONPATH=src:. <venv>/bin/python scripts/reprice_band_geometry.py
```

Bars are cached under `analysis_cache/2026-09-20_reprice_band_geometry/`; pass
`--refresh` to re-fetch. The script exits 2 if the static validation ever fails,
so a bars/convention regression cannot silently produce numbers.
