---
type: recon
date: 2026-09-15
time: 16:40 PDT
agent: dsh
model: deepseek-v4-flash
trigger: "Brandon: 'the bot is sending me discord messages stating that the feed has been dead, investigate'"
head: f0d6508
scope: modifies-source
related:
  - recons/2026-08-08_stop-width-and-the-spread-toll.md
  - m2m-prompts/2026-09-14_barrier-live-seam.md
files_touched:
  - src/execution/risk_manager.py
  - src/execution/oanda_forex_orchestrator.py
  - src/execution/README.md
  - tests/test_risk_manager.py
  - tests/test_stream_liveness.py
---

# The feed is not dead — the market is closed

## Context

Brandon reported Discord messages from the soak saying the feed had died. The alert is
a CRITICAL from the stream-liveness watchdog, and the question was whether the feed was
genuinely down — in which case an open position is an unwatched position, because stops
here are software-enforced.

**Answer: the feed was never dead. The watchdog was reporting scheduled market pauses as
outages.** At the time of investigation nothing was wrong: PID 9674, stream connected, 6
symbols primed, no positions, `status.json` fresh.

## Investigation

The alert is `_check_stream_liveness` (`src/execution/oanda_forex_orchestrator.py:2125`),
which fires when prices are silent past `_stream_stale_seconds` (60s,
`OANDA_STREAM_STALE_SECONDS`):

```
[CRITICAL] No PRICE for 215s (threshold 60s) — stream alive but silent —
           no positions held; forcing reconnect
```

Counting the trigger lines per run and per hour produced the pattern:

| run | trigger lines | incidents | reconnects | max price age |
|---|---:|---:|---:|---:|
| `soak_2026-09-11_0015.log` | **12,674** | 14 | 12 | **51,072s (14.2h)** |
| `soak_2026-09-13_2125.log` | 64 | 4 | 11 | 346s |

Both clusters in the current run start within two seconds of **21:00 UTC** — 09-14 at
14:00:11 PDT, 09-15 at 14:00:09 PDT. A schedule, not an outage. 21:00 UTC in September
is 17:00 ET.

`grep -rn "rollover" src/` found the window already modelled:
`src/execution/risk_manager.py:177` defines **Gate C**, a 16:55–17:30
**America/New_York** blackout added for exactly this ("the daily rollover, when spreads
briefly blow out roughly tenfold"). It blacks out *entries*
(`risk_manager.py:574`). The liveness probe never consulted it — that is the bug.

The weekend run explains the other 12,674: it spans Friday 21:00 UTC (17:00 ET, the
forex weekly close) to Sunday 02:18. Every line in it says `no positions held`.

Why it is not merely noise — the same method:

```python
if has_positions:
    await self._flatten_all()
self._provider.force_disconnect("liveness watchdog: stream stale")
```

Positions are legitimately held *across* the rollover (Gate C blocks only new entries),
so an open trade at 16:55 ET would have been closed by the watchdog during the daily
rollover, at the widest spreads of the day, on the belief that the feed had died.

## Findings / Changes

1. **HIGH — the liveness watchdog had no market-hours awareness.** Every daily rollover
   and every weekend produced CRITICAL lines, Discord alerts, futile reconnects, and
   would have flattened held positions. Fixed.
2. **MEDIUM — a duplicated notion of "closed".** `soak_watchdog.sh:76-84` carries its own
   Pacific-time weekend test (Fri ≥14:00, Sat, Sun <14:05). It is correct by accident of
   PT and ET shifting together, and covers only the weekend. Left in place; noted so the
   next agent does not assume one definition exists.
3. **LOW — the alert text is unfalsifiable from the phone.** `No PRICE for Ns` reads as a
   dead feed when it can equally be a closed market. The pause path now logs
   `Feed quiet during the scheduled <pause>` at INFO, which distinguishes them in the
   log.

**Fix:** `scheduled_market_pause(when=None, spec=None)` in `risk_manager.py`, beside the
blackout logic it reuses — returns `PAUSE_WEEKEND` (Fri 17:00 ET → Sun 17:00 ET),
`PAUSE_DAILY_ROLLOVER` (inside Gate C's window) or `None`, anchored to
`America/New_York` so it tracks DST. `_check_stream_liveness` returns early during a
pause: one INFO line per pause (new `_liveness_pause_logged`), no CRITICAL, no Discord,
no flatten, no reconnect — and it re-arms `_liveness_alert_fired` so the first genuine
outage after the reopen still alerts. Nothing is skipped permanently.

Fail-safe direction is deliberate: no zoneinfo database ⇒ `None` ⇒ the watchdog stays
fully armed, because suppressing by mistake costs an unwatched position while not
suppressing costs visible, recoverable noise.

## Verification

Timestamps replayed through the helper before restarting the live process:

```
09-15 rollover (real incident start) -> daily rollover      Tue 10:00 ET          -> None
09-14 rollover (real incident start) -> daily rollover      winter 22:05 UTC (EST)-> daily rollover
Fri 17:00 ET (weekly close)          -> weekend closure     winter 21:05 UTC      -> None
Sat / Sun 12:00 ET                   -> weekend closure     Sun 17:30 ET (reopened)-> None
```

14 new tests: 8 pinning the helper (boundaries, DST, naive-as-UTC, spec override, a
malformed spec disabling only the rollover, the two real incident moments) and 6 the
behaviour (no flatten / no reconnect / no Discord during a pause, guard re-armed, one
log line per pause, and the incident path unchanged outside one). The pre-existing
liveness tests now pin the clock outside any pause in `setUp` — otherwise the suite would
pass or fail depending on whether it ran over a weekend. **Suite: 550 passed, 6 subtests
passed.** `compileall` clean.

Live restart at 16:24:29 PDT, after confirming `positions: {}`: new PID **897548**, log
`logs/soak_2026-09-15_1624.log`, 6 symbols primed, 23:00 UTC bar caught up, stream
connected, `status.json` fresh. **Zero `No PRICE` triggers in the ~60 probes (~10
minutes) since** — the direct measurement that prices are flowing. The four boot-time
404 `NO_SUCH_POSITION` lines are pre-existing (the previous run's boot has the identical
four). Both the watchdog's start and restart paths go through `soak.service`, so a
restart cannot leave a duplicate bot.

Two test expectations I wrote were wrong and the code was right: Friday 16:59 ET is
genuinely inside the 16:55–17:30 rollover window, and Sunday 17:00 ET (the weekly
reopen) lands inside it too — the two pause kinds meet at the 5pm-ET boundary, so the
pause continues as a rollover until 17:30 ET. That is correct (the reopen is when spreads
are widest) and is now pinned.

## Risk & follow-ups

- **Holidays are not covered.** Irregular per-year dates, and a wrong calendar is worse
  than none: a holiday pause will still alert. A holiday table is the follow-up if those
  alerts become annoying.
- **A genuine outage inside the rollover window is suppressed** for those 35 minutes
  (entry is already blocked and the market is known-toxic). Accepted as the cheaper
  error.
- **The first rollover after this change is 2026-09-16 13:55 PDT.** Expect one INFO line
  and no Discord message; if a CRITICAL still appears there, the gate is not being
  reached and the fix has failed.
- `soak_watchdog.sh`'s duplicate weekend test could be pointed at the same definition,
  which would need a small CLI wrapper. Not urgent.
- **CORRECTION to this report's first draft**, which claimed "`SEAM_BACKFILL` runs for
  every sealed bar (~54 events in 2 days), so bars are routinely rebuilt from REST".
  Both halves were wrong. Measured: the 54 events in the 09-13 run are **9 seam
  crossings × 6 symbols**, and all 60 events across both runs attribute to a prime round
  with zero orphans (34-776s after the prime, i.e. when the first post-seam bar seals).
  A seam crossing happens once per boot/reconnect, not once per bar — it is the cost of
  the *reconnect*, and its frequency measures stream instability, nothing else. See the
  `_last_scored_ts` / `_seam_crossed` design notes in `oanda_forex_orchestrator.py:46-85`.

## Files touched

**Modified:**
- `src/execution/risk_manager.py` — `WEEKLY_CLOSE_ET`/`WEEKLY_OPEN_ET`,
  `PAUSE_WEEKEND`/`PAUSE_DAILY_ROLLOVER`, `scheduled_market_pause` (~line 213), Layer-3
  glossary entries.
- `src/execution/oanda_forex_orchestrator.py` — imports the helper; `_liveness_pause_logged`
  init (~line 500); early return in `_check_stream_liveness` (~line 2125); liveness
  docstring.
- `src/execution/README.md` — the liveness-watchdog bullet now documents the gate.
- `tests/test_risk_manager.py` — `TestScheduledMarketPause` (8 tests).
- `tests/test_stream_liveness.py` — `setUp` clock pin + `TestScheduledPauseNoAction` (6 tests).

**Read (so a follow-up knows what is already examined):** `src/execution/oanda_forex_orchestrator.py`
(2125-2205 liveness, 490-502 policy, 2248-2260 watchdog loop),
`src/execution/risk_manager.py` (177-235 Gate C + pause helper, 566-600 gate order,
635-650 `_in_blackout`), `src/core/notification_manager.py`, `soak_watchdog.sh`,
`logs/soak_2026-09-11_0015.log`, `logs/soak_2026-09-13_2125.log`, `logs/status.json`,
`tests/test_stream_liveness.py`, `llm_reports/README.md`.
