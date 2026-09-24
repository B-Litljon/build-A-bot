---
type: recon
date: 2026-09-24
time: 21:20 PDT
agent: kimi-k3 (opencode)
model: ollama-cloud/kimi-k3
trigger: Lane 5 Audit A — altcoin cross-sectional momentum falsification
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
files_touched:
  - src/lab/altcoin_topquint.py
  - src/lab/stats.py
  - tests/test_lab_altcoin.py
related:
  - llm_reports/recons/2026-09-14_session-evidence-and-options.md
  - llm_reports/handoffs/2026-09-24_lane5-falsification-audits.md
---

# Audit A — Altcoin cross-sectional momentum

## Question

Does a weekly LightGBM `lambdarank` top-quintile harvester over a basket of
liquid Alpaca crypto pairs beat **both** an equal-weight basket and BTC
buy-and-hold, after a realistic 31.6 bps/side (6.6 spread + 25 taker)
rebalancing toll, with multiple-testing adjustments that don't lie about
having tried many configurations?

The 2026-09-14 crypto axis verdict (`session-evidence-and-options`) closed
D1/H4 forex-style bracket geometries (0 of 27 positive at random entries on
Alpaca daily bars; best D1 cell −0.021R). It did **not** test the
cross-sectional momentum anomaly — whether the winners among alts keep
winning into the next week relative to the rest of the basket. This audit
fills that gap.

## Data & window

**Source.** Alpaca crypto daily bars (the venue the brief nominates; NOT
Binance — binance.com is geo-blocked from this host, HTTP 451, verified in
`analysis_cache/crypto_data_probe/probe_results.json`) via
`alpaca.data.historical.CryptoHistoricalDataClient`, fetched 2026-09-24 and
cached at `/tmp/opencode/lane5_crypto/`.

**Realized window.** Per-symbol, 2021-01-01 → 2026-09-20 for the longest
series. The 20-pair universe the brief nominates is **not fully tradeable**:
SNX/USD, COMP/USD, UMA/USD, ZRX/USD return 0 rows from Alpaca crypto (checked
2026-09-24), and MKR/USD's series ends 2025-09-05 (delisted on the venue).
16 of 20 pairs return data at all.

**The liquidity floor that decides the audit.** The brief's admission rule
is ≥30 non-zero-volume days in the sampled window AND median daily dollar
volume (volume × close) ≥ $250k. Measured median dollar volume over each
symbol's full series:

| symbol | median $vol | symbol | median $vol |
|---|---:|---|---:|
| BTC/USD | **383,022** | AVAX/USD | 9,868 |
| SOL/USD | **985,501** | AAVE/USD | 8,015 |
| ETH/USD | 206,880 | GRT/USD | 7,956 |
| LINK/USD | 24,996 | SUSHI/USD | 5,336 |
| LTC/USD | 16,189 | DOT/USD | 3,209 |
| BCH/USD | 11,767 | BAT/USD | 2,260 |
| UNI/USD | 12,460 | CRV/USD | 773 |
| MKR/USD | 16,534 | YFI/USD | 161 |
| (4 pairs return no data) | | | |

Only **two** symbols (BTC, SOL) clear $250k. At a $50k floor: 3 (BTC, ETH,
SOL). At $20k: 4. On the trailing-60-day window the engine actually applies
per week (the no-look-ahead rule), the picture is far thinner — every
trailing-60-day median dollar volume, BTC included, sits **below** $250k
(BTC 107k, ETH 48k, SOL 19k, next-highest UNI at 7k).

## Method

Daily closes, 4 momentum lookbacks {21, 63, 126, 252}, weekly ISO query
groups, next-week return quintile target, weekly top-quintile basket 1/N,
friction 31.6 bps per dollar of one-way turnover. Dual benchmark =
equal-weight basket + BTC hold; gate = DSR > 0.95, CSCV PBO < 0.50, and a
Clopper-Pearson 95% one-sided lower bound on the weekly beat-both rate > 0.
Leakage guards per the brief (universe membership uses only data ≤ the
signal week; target is the next-week window strictly after the signal bar).

The audit harness is `src/lab/altcoin_topquint.py`; all four synthetic
fixture tests in `tests/test_lab_altcoin.py` (ranker identifies top group,
universe guard excludes sub-floor assets and is not rescued by future
volume, friction cost on a forced 0.5 turnover is exactly 0.5 × 31.6/10000
NAV, run determinism, and the no-look-ahead feature test) pass.

## Results

**No audit could be run.** The universe guard admits 0 symbols per week
across the whole realized window, so zero rebalancing weeks are measured,
the strategy NAV series is empty, and the gate is vacuously not evaluable.

This is the brief's own abort criterion ("Altcoin volume floor leaves < 5
assets → report the tiny universe as the finding"), triggered not by a
borderline universe but by a zero-tradeable universe at the venue whose data
the audit was scoped to.

## Falsification verdict

> **FALSIFIED AT THE DATA LAYER — the $250k liquidity floor leaves < 5
> tradeable assets on Alpaca crypto (0 at the trailing-60-day admission the
> leakage guard enforces; at best 2, BTC and SOL, on full-history medians),
> so the cross-sectional-momentum strategy is unevaluable and the audit
> closed without testing the anomaly.**

Subordinate finding worth recording: the volume floor itself is the binding
constraint. Alpaca's free-tier crypto volume is the IEX slice of a fragmented
market; "liquid enough to matter" names like ETH and LINK sit one to two
orders of magnitude below their consolidated volume, so ANY liquidity-gated
cross-sectional crypto strategy scoped to this feed inherits the same floor
failure. That is a property of the data feed, not of the strategy — the right
conclusion is "cannot test on Alpaca free-tier volume", not "the anomaly is
dead".

## Recommended next step

If the anomaly is to be falsified for real, re-run against a consolidated
crypto volume source — Binance (geo-blocked here), Kraken (open, 1432
pairs, per the same probe), or CoinGecko's volume (open) — none of which is
plumbed into this repo. The harness, fixtures and gate here are
volume-source-agnostic: rerunning against a consolidated feed is a data
adapter, not new audit code. Until then the audit stands closed at the data
layer with the harness ready.

## Files touched

- `src/lab/altcoin_topquint.py` (new) — audit harness
- `src/lab/stats.py` (new, shared with Lane 5's other audits) — DSR / CSCV
  PBO / HLZ / CP bound, canonical formulas from the Lane 1 brief §6.2
- `tests/test_lab_altcoin.py` (new) — the four mandated fixture tests plus
  two leakage-guard tests
