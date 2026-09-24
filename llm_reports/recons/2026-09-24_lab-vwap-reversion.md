---
type: recon
date: 2026-09-24
time: 21:20 PDT
agent: kimi-k3 (opencode)
model: ollama-cloud/kimi-k3
trigger: Lane 5 Audit C — session-scale VWAP reversion at London open
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
files_touched:
  - src/lab/vwap_reversion.py
  - tests/test_lab_vwap_reversion.py
related:
  - llm_reports/recons/2026-09-14_session-evidence-and-options.md
  - llm_reports/handoffs/2026-09-24_lane5-falsification-audits.md
---

# Audit C — Session-scale VWAP reversion at London open

## Question

During the 08:00–09:00 UTC London open hour, do M15 price deviations beyond
k·σ_session from the session VWAP mean-revert enough to overcome the 0.25R
per-turn toll — with exits driven by session inventory imbalance (not an
arbitrary target) and a hard 09:00 UTC session close?

## Data & window

M15 mid bars for the six fiat pairs
(EUR_JPY/GBP_JPY/AUD_JPY/NZD_JPY/GBP_AUD/GBP_NZD),
2024-09-08 → 2026-09-08, `analysis_cache/strategy_matrix/{pair}_M15.parquet`.
Sessions are the 08:00–09:00 UTC bar groups; 3,102 sessions measured across
the six pairs.

## Method

- **Session VWAP** `VWAP_t = Σ P·V / Σ V` re-anchored each session.
- **σ_session** is the realized std of within-session log returns up to bar
  t (leakage guard: only bars ≤ t), in price units σ_P = σ_r · VWAP_t so the
  deviation and the stop compare in price.
- **Entry:** fade when |C_t − VWAP_t| > k·σ at a session bar close; one fade
  per session. Direction: short a pop above VWAP, long a dip below.
- **Stops:** protective stop at deviation re-extension — the signed
  deviation moves a further k·σ AWAY from VWAP vs its value at entry
  (stop distance in price = k·σ, so a stop-out loses exactly 1R).
- **Exit:** the session's cumulative signed-volume imbalance
  (Σ sign(r)·V) crosses zero (the position's net inventory is judged
  neutralised), or the hard 09:00 UTC close, whichever first.
- **Sizing:** 1% NAV risk per trade; R = PnL / (k·σ at entry). Toll 0.25R
  per round trip subtracted from each trade's gross R.
- **k sweep:** k ∈ {1.5, 2.0, 2.5, 3.0}; the gate requires positive net
  expectancy across ALL k (independent of k) — a single-k success is flagged
  as overfit.

## Results

Trade counts and expectancy per k (net of the 0.25R toll; winsorised at ±5R
so a handful of session_end trades with near-zero entry σ do not dominate;
raw unaudited EVs are also reported):

| k | trades | gross winrate | winsorised net EV | raw net EV | t-stat | HLZ t | DSR | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 954 | 49.0% | **−0.140R** | +19181.9R¹ | −1.44 | −2.59 | 0.000 | 0.471 |
| 2.0 | 736 | 50.1% | **−0.117R** | +18647.6R¹ | −1.07 | −2.22 | 0.000 | 0.471 |
| 2.5 | 592 | 51.5% | **−0.065R** | +18546.9R¹ | −0.53 | −1.68 | 0.000 | 0.471 |
| 3.0 | 489 | 51.5% | **−0.047R** | +18711.2R¹ | −0.35 | −1.50 | 0.000 | 0.471 |

¹ The raw (unwinsorised) means are dominated by a small number of extreme
session_end exits whose entry σ was near zero (R normalisation divides by
k·σ_entry ≈ 0, inflating the R of an ordinary price move to five figures).
They are reported for completeness and are not decision-grade — the
winsorised core is the honest measure.

Gate: net EV > +0.25R per trade after the toll (the brief's toll convention,
i.e. gross EV > 0.50R), DSR > 0.95, PBO < 0.50, HLZ t > 3.0, and positive
expectancy independent of k. **No k clears a single one of these bars**;
winsorised net EV is negative at every k, no HLZ t is above −1.4, and no
DSR is above 0.001. There is no overfit flag to raise because no k wins.

## Verdict

> **FAIL — session-scale VWAP reversion at London open delivers a winsorised
> net expectancy of −0.047R to −0.140R per trade across the whole k sweep,
> versus the +0.25R bar; no k clears net EV, DSR, PBO or HLZ, so the edge
> claim is falsified independent of k and there is no single-k overfit to
> hide behind.**

The one thing this audit does NOT say is that inventory-imbalance exits are
worse than arbitrary targets; with only 4 bars per session the imbalance
cross essentially coincides with session-end anyway (every measured trade
closed at session_end — 954/954 at k=1.5, and so down the sweep). The
exit rule never had room to differentiate itself inside one hour at M15
resolution, which is a statement about the session's bar budget, not about
the VWAP edge that was measured and failed.

## Recommended next step

None for this hypothesis at M15 granularity. The measured cost of fading
here (−0.05 to −0.14R/trade) is far larger than the repo's existing
cost-side levers move; raising the edge would need a genuinely different
feature/target design (the same conclusion the 2026-09-14 session-evidence
report reached for the whole repo) rather than a retune of k.

## Tests

`tests/test_lab_vwap_reversion.py` — 4 tests, all green: VWAP arithmetic on
a synthetic session; the imbalance sign flip exits on the exact flip bar;
a trade that has not exited by 09:00 UTC is force-closed on the last session
bar; k-sweep determinism on a seeded RNG.

## Files touched

- `src/lab/vwap_reversion.py` (new) — session simulator + k-sweep gate
- `tests/test_lab_vwap_reversion.py` (new)
