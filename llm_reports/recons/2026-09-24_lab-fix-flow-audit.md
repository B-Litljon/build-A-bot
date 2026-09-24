---
type: recon
date: 2026-09-24
time: 21:20 PDT
agent: kimi-k3 (opencode)
model: ollama-cloud/kimi-k3
trigger: Lane 5 Audit B — calendar-flow liquidity around the London WM/R and Tokyo Nakane fixes
head: b09fde2f1c99cac893e38d87b2ae8678ae17e862
scope: modifies-source
files_touched:
  - src/lab/fix_audit.py
  - src/lab/fix_collector.py
  - scripts/fix_tick_collector.py
  - systemd/fix-tick-collector.service
  - tests/test_lab_fix_audit.py
related:
  - llm_reports/handoffs/2026-09-24_lane5-falsification-audits.md
---

# Audit B — Fix-flow liquidity: the WM/R 16:00 London and Tokyo Nakane 9:55 fixes

## Question

Should the "fade the fix" route be eliminated for this basket because
market-maker spread-widening into the fix exceeds the drift it defends
against? The two events: the WM/Reuters 4 PM London benchmark and the 9:55
AM JST Tokyo Nakane fix (Gotobi days — 5/10/15/20/25/month-end — carry the
heaviest corporate flow).

## Data-resolution limitation — first-class

**There is no tick, quote, or bid/ask spread time series anywhere in this
repo** (verified 2026-09-24). Every cached forex artifact is M15 mid-price
OHLCV in `analysis_cache/strategy_matrix/{pair}_M15.parquet`; the only
spread artefact is `config/spread_alphas_m15.json` — six per-instrument
median scalars, not a time series.

**What this audit measures, precisely:** on M15 mid bars, the log return of
the bar CONTAINING each fix instant, its high/low range as a % of mid, the
next bar's (post-fix) return, and the same clock-time bars on non-fix days.

**What it can NOT measure: bid/ask widening.** High-low range on a mid-price
bar is a coarse proxy for intraday activity, NOT a spread measure; a fix bar
indistinguishable from a non-fix bar at M15 resolution does NOT falsify
tick-level widening. That is why `scripts/fix_tick_collector.py` and
`systemd/fix-tick-collector.service` exist — they capture the bid/ask ticks
the coarse study cannot see, and they ship **uninstalled and unrun**.

## Data & window

M15 mid bars, six fiat pairs, 2024-09-08 → 2026-09-08 (49,645 bars/pair;
EUR_JPY/GBP_JPY/AUD_JPY/NZD_JPY/GBP_AUD/GBP_NZD). 517 weekday London fixes
per pair; 101 Gotobi Tokyo fixes per pair.

Fix instants are DST-correct via `zoneinfo`: the London fix is 16:00
Europe/London (15:00 UTC in BST, 16:00 UTC in GMT) and the Tokyo fix is 9:55
Asia/Tokyo (00:55 UTC always). The DST-boundary test
(`test_london_fix_dst_boundary`) walks 2026-03-27→30 and 2026-10-23→26 and
verifies the UTC offset flips on the correct days.

## Results

### WM/R London fix (16:00 Europe/London) — 517 fix days/pair, no non-fix reference

| pair | p99 \|r\| fix-bar | post-fix sign-flip |
|---|---:|---:|
| EUR_JPY | 0.00168 | 0.499 |
| GBP_JPY | 0.00194 | 0.503 |
| AUD_JPY | 0.00271 | 0.516 |
| NZD_JPY | 0.00259 | 0.518 |
| GBP_AUD | 0.00174 | 0.472 |
| GBP_NZD | 0.00140 | 0.460 |

Because every weekday is a London fix day, there is **no non-fix weekday
reference at the same clock time**; the fix cohort is reported against itself.
Sign-flip rates hover at chance (≈ 0.5).

### Tokyo Nakane fix (00:55 UTC, Gotobi days) — 101 fix vs 417 non-fix days/pair

| pair | p99 \|r\| fix | p99 \|r\| non-fix | \|r\| ratio | range ratio | sign-flip |
|---|---:|---:|---:|---:|---:|
| EUR_JPY | 0.00146 | 0.00216 | 0.67× | 0.840 | 0.535 |
| GBP_JPY | 0.00145 | 0.00207 | 0.70× | 0.919 | 0.535 |
| AUD_JPY | 0.00194 | 0.00300 | 0.65× | 0.651 | 0.465 |
| NZD_JPY | 0.00188 | 0.00258 | 0.73× | 0.665 | 0.455 |
| GBP_AUD | 0.00089 | 0.00137 | 0.65× | 0.548 | 0.505 |
| GBP_NZD | 0.00095 | 0.00137 | 0.70× | 0.548 | 0.436 |

Fix-day bars are, if anything, **quieter** than non-fix bars at the same
clock time: the fix-day 99th-percentile |return| runs 0.65–0.73× the non-fix
(reference: brief's falsification trigger is > **2×**), and the fix-day range
runs 0.55–0.92× non-fix. Post-fix sign flips sit at chance.

## Verdict

> **FIX AND NON-FIX BARS ARE INDISTINGUISHABLE AT M15 RESOLUTION (|r| p99
> ratio 0.69×, range ratio 0.66×). The post-2015 spread-widening concern is
> NOT SUPPORTED AT M15 RESOLUTION — this is not evidence that no tick-level
> widening exists; at this resolution the route is never "falsified", only
> "needs tick data", which the shipped-uninstalled collector now exists to
> gather.**

Two cautions, both load-bearing. (1) The London fix has no non-fix weekday
reference at its own clock time, so this study says nothing directional
about WM/R — only that Tokyo Gotobi days are quiet. (2) Nothing here measures
spreads; a wider bid/ask into the fix is exactly the thing the coarse bars
cannot see.

## The tick collector (shipped uninstalled and unrun)

`scripts/fix_tick_collector.py` streams the six pairs via
`OandaMarketProvider.subscribe(..., tick_callback=...)`, keeps only ticks
inside ±5 min of each fix, and writes `data/ticks/fix_ticks_YYYY-MM-DD.parquet`
(columns `timestamp_utc, symbol, bid, ask, mid, source_latency_ms`) with the
repo's own temp-file + fsync + `os.replace` atomic-write convention
(`lab/fix_collector.py:write_ticks_atomic`), reconnecting with exponential
backoff (1 s → 60 s full jitter) and flushing on SIGTERM. It is supervised by
`systemd/fix-tick-collector.service` — an **uninstalled template** whose
header says "install only after the coarse study is reviewed".

It was **not installed, not enabled, and not run** during this lane. The
runtime stream wrap (`Collector.run` stamping `_last_tick_ts` for latency)
carries a `# TODO: confirm stream callback signature at runtime` note and
is the one piece of this deliverable that is unverified against the live
provider — by design, per the brief's abort criterion for the collector.

## Tests

`tests/test_lab_fix_audit.py` — 7 tests, all green: DST-correct UTC
conversion across the 2026 London DST boundaries and constant 00:55 UTC for
Tokyo; Gotobi calendar (5/10/15/20/25/month-end); `bar_containing` finds the
right 15-min bar and rejects off-grid instants; the parquet schema carries
exactly the six named columns and the atomic rename leaves no `.tmp`; the
coarse study is byte-for-byte deterministic on a fixed cache.

## Files touched

- `src/lab/fix_audit.py` (new) — DST/calendar helpers + the coarse M15 study
- `src/lab/fix_collector.py` (new) — tick buffer + atomic parquet writer
- `scripts/fix_tick_collector.py` (new) — the OANDA streaming collector
- `systemd/fix-tick-collector.service` (new, **uninstalled** template)
- `tests/test_lab_fix_audit.py` (new)
