---
type: handoff
date: 2026-09-15
time: 17:10 PDT
agent: dsh
model: deepseek-v4-flash
trigger: "Brandon asked for a portable brief on the live stream/bar pipeline for another harness to investigate"
head: f0d6508
scope: read-only
related:
  - recons/2026-09-15_feed-pauses-and-the-liveness-watchdog.md
  - m2m-prompts/2026-09-14_barrier-live-seam.md
files_touched:
  - llm_reports/handoffs/2026-09-15_stream-bar-pipeline-brief.md
---

# Brief: is the live bar pipeline building the same bars the models were trained on?

## Context

Hand this to another agent. The subject is the **live M15 soak's bar construction**, not
the strategy research (that is settled separately — see the `build-a-bot-edge-budget`
topic in `~/.agent-knowledge/kb/` and `recons/2026-09-14_session-evidence-and-options.md`;
do not re-derive it).

What is already known and measured, so you do not repeat it:

- Training data comes from **OANDA REST candles** (`core/retrainer.fetch_training_data` →
  `data/oanda_provider.get_historical_bars`, `/v3/instruments/<sym>/candles`, mid price).
- Live bars are aggregated **locally from websocket ticks** in
  `data/oanda_provider.py` (`_handle_tick`, `mid = (bid+ask)/2`, ~line 210-270).
- Warm-up history at boot/reconnect is REST and is joined to the live stream at a "seam".
  The bar straddling it is dropped and re-fetched from REST because the local copy is
  incomplete (`_backfill_seam_bar`, `oanda_forex_orchestrator.py:1243`). Measured
  2026-09-15: every seam event attributes to a prime round, exactly one per symbol per
  crossing, 34-776s after the prime — it is a reconnect cost, not a routine path.
- A bar that seals while the stream is down is recovered by `_catch_up_missed_bars`
  if it is within one bar period (900s, `SEAM_CATCHUP_MAX_AGE_SECONDS`); older missed
  bars are abandoned by design.
- The liveness watchdog was fixed on 2026-09-15 (`risk_manager.scheduled_market_pause`):
  it no longer treats the daily 5pm-ET rollover or the weekend closure as a dead feed.

**Both paths use mid price, so bid/ask skew is already ruled out.** Do not re-check it.

## The question

**Do locally aggregated stream bars match OANDA's REST candles for the same timestamp,
and if not, does the difference reach the model?**

The mechanism to test: REST candles are computed server-side from *all* ticks. The local
aggregation sees only the ticks delivered over the websocket, and the tick callback is
explicitly allowed to drop work while the read thread is busy ("may do nothing if the
thread is blocked mid-read"). So a live bar's **high/low can be narrower, and its volume
different, from the true candle** — and `natr_14` is computed from exactly those
highs/lows.

If they differ, the live model is being fed features computed on a slightly different
series than the one it was trained on. That is a train/serve skew, and it is invisible in
every log the bot currently writes.

## How to answer it

1. Pick N bars (say 100-200) across a couple of days and, for each symbol, compare the
   locally aggregated bar against the REST candle for the same timestamp: open, high,
   low, close, volume, and the resulting `natr_14`.
2. The instrumentation does not exist yet — the seam path deliberately throws the local
   copy away, so you will need to add a temporary, opt-in comparison (env-gated, default
   off, writing to a file under `logs/`) rather than changing what gets scored. Do not
   alter the scored bar while measuring.
3. Report the distribution of differences, not just the mean: a systematic one-tick skew
   and occasional large gaps have different causes. Note whether gaps cluster after
   reconnects or busy periods.
4. If there IS a difference, size the consequence: recompute `natr_14` both ways and see
   whether the live features would move a decision (e.g. would the Angel bar or a Gate B
   percentile have flipped on the affected bars?).

## Secondary questions in the same area, all cheap

- **Reconnect rate.** 11 reconnects in the 42 hours to 2026-09-15, most at the 21:00 UTC
  rollover. Now that the pause gate stops the watchdog force-disconnecting during those
  windows, does the rate fall? Compare `grep -c "Stream reconnect in" logs/soak_*.log`
  before and after 2026-09-15 16:24 PDT. If the watchdog was *causing* those reconnects,
  that is worth knowing.
- **Missed-signal accounting.** The catch-up rule abandons bars older than one period.
  Enumerate outages longer than 900s and estimate whether any would-be signal was lost —
  the 2026-07 figure on file is "~6 of 15 would-be signals died in these gaps".
- **Duplicate-order risk.** `_last_scored_ts` is the dedup that prevents scoring one bar
  twice. Check that *every* path which calls `_evaluate_and_trade` sets it first (stream,
  seam backfill, catch-up, reconcile) and that nothing resets it on reconnect.

## Ground rules

- **Do not restart or stop the soak without asking.** If you must stop it, `touch soak.off`
  first or `soak_watchdog.sh` relaunches it within 5 minutes. Restarting is a deliberate
  act: `systemctl --user restart soak.service`.
- **Never touch `models/forex_m15_wide`**, and do not enable `BARRIER_GEOMETRY_ENABLED`
  or `RISK_SIZING_ENABLED`.
- Measurement code must be env-gated and default-off; the soak runs from this working
  tree, so anything you leave in `src/` goes live on the next restart.
- Use the venv python: `PYTHONPATH=src:. /home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python`.
- Read `CLAUDE.md` first — three-layer docs must be updated in the same change, and
  `llm_reports/README.md` has the frontmatter + six-section convention.

## Where to write it up

A recon at `llm_reports/recons/2026-09-XX_<topic>.md`, appended to the live thread
`llm_reports/m2m-prompts/2026-09-14_barrier-live-seam.md`, and — because it is a
diagnosed mechanism rather than a code fact — a paragraph in the `build-a-bot-soak`
topic in `~/.agent-knowledge/kb/` followed by `agent-kb sync`.

## Findings / Changes

_n/a — this is a brief, not a report._

## Verification

The measurements quoted above were taken 2026-09-15 from `logs/soak_2026-09-13_2125.log`
and `logs/soak_2026-09-15_1624.log` by grepping `SEAM_BACKFILL` / `SEAM_CATCHUP` /
`Stream reconnect in` and attributing each seam event to the most recent prime round:
60 events, 0 unattributed, 6 per round, 34-776s after the prime. The pause-gate fix that
preceded this brief is in `src/execution/risk_manager.py` (`scheduled_market_pause`) with
14 tests; suite 550 passed.

## Risk & follow-ups

If the bars differ materially, the fix is a design decision, not a patch — the options
are to build live bars from REST candles (as the retrainer does, at the cost of a poll
per bar), to reconcile the local bar against REST at seal time, or to accept the skew and
document it. Bring the measurement before choosing.

## Files touched

`llm_reports/handoffs/2026-09-15_stream-bar-pipeline-brief.md` (this brief).

Read, for the investigation: `src/data/oanda_provider.py` (`_handle_tick`, `get_historical_bars`,
`_granularity`), `src/execution/oanda_forex_orchestrator.py` (46-90 design notes, 1185-1300
seam paths, 1990-2070 catch-up), `logs/soak_*.log`.
