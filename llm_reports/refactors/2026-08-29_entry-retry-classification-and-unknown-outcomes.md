---
type: refactor
date: 2026-08-29
time: 09:40 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "finish the entry-retry work: the shipped classifier still dropped the 401 that started this, and the unknown-outcome path could strand a live position"
head: b89cfc2e8f6332f81d8ee04c3cb4b8e02b0785cb
scope: modifies-source
files_touched:
  - src/execution/oanda_order_manager.py
  - src/execution/oanda_forex_orchestrator.py
  - src/execution/README.md
  - tests/test_oanda_entry.py
  - tests/test_unverified_entry.py
related:
  - refactors/2026-08-28_order-retry-safety.md
---

# The retry that still dropped the trade, and two ways to lose a position

## Context

`2026-08-28_order-retry-safety.md` (DeepSeek V4 Pro) added the retry loop and
closed the double-fill trapdoor by re-syncing the broker before recomputing
the delta. That core is correct and is kept unchanged here.

But it classified failures on a single question — *is this a definitive
reject?* — answered as "any 4xx", and used that answer to decide **whether to
retry**. The 2026-08-26 GBP_AUD incident that motivated the whole change is a
401. So the shipped code would have dropped that trade exactly as before.

Two questions were collapsed into one, and they have different answers:

| | Could the position have moved? | Will a retry help? |
|---|---|---|
| 400 + `rejectReason` | no | **no** — same rejection |
| bare 401 / 403 / 429 | no | **yes** — transient |
| 5xx / timeout / reset | **unknown** | only after a re-sync |

The 4xx proof that *nothing filled* is precisely what makes retrying a 401
**safe**. The old code used that proof to justify giving up instead.

## Investigation

**The 401 is transient, not a credential problem.** Same token, same account,
in `logs/soak_2026-08-23_1405.log`:

```
04:30:07  .../instruments/NZD_JPY/candles failed [401,{"errorMessage":"Insufficient authorization to perform request."}]
04:30:09  performing request .../instruments/NZD_JPY/candles
04:30:10  SEAM_BACKFILL [NZD_JPY] recovered the in-flight bar ... (attempt 2)
```

Three seconds apart, no re-auth in between. The data path already retries and
recovered; only the order path gave up.

**The discriminator is the body, not the status.** Verified against every
order rejection in the soak logs (4× 400, 1× 401): a business reject carries
`orderRejectTransaction.rejectReason` *and* a top-level `errorCode`; the 401
carries only `errorMessage`. So "has a reject reason" separates permanent from
transient, while HTTP status separates filled-impossible from ambiguous.

**Two ways to strand a live position, both proven by execution** rather than
by reading, using a fake broker that fills but drops the response:

1. *Ambiguous failure + failed re-sync.* Known and documented as a residual in
   the prior report. The result reported `filled=0`; the caller read that as a
   clean miss and dropped it. Broker held 1000 units; the bot tracked none.

2. *Ambiguous failure on the FINAL attempt.* **Introduced by the retry loop and
   not previously identified.** The last attempt fills, its own re-sync proves
   it, and then the loop exits into a post-loop `return` that hardcodes
   `"filled": 0` while reporting `position_units: 1000`. Measured:

   ```
   attempts: 3   BROKER actually holds: 1000
   reported filled: 0   reported position: 1000
   caller records position? False
   >> BUG: live position of 1000 units, caller drops it -> UNTRACKED, NO STOP
   ```

   Stops here are software, so an unrecorded position is an unwatched one.

## Findings / Changes

**1. The classifier splits into two questions** (`oanda_order_manager.py`).
`_is_definitive_order_reject` is replaced by `_order_never_filled` (the safety
question: any 4xx) and `_is_permanent_reject` (the retry question: does the
body carry a reject reason?), with `_error_body` as a tolerant JSON decode.
A permanent reject returns immediately; a bare 401/403/429 is retried **without
a re-sync**, since nothing filled and the re-sync would likely hit the same
blip; 5xx and transport errors keep the existing re-sync-then-retry path.

**2. `_result_if_at_target`** is extracted and consulted in *both* places that
need it — between retries, and after the attempts are exhausted. That closes
hazard #2 above. The final attempt is not special: it can fail ambiguously and
still have filled.

**3. An explicit `unverified` key on every result.** True only when an order
was sent, failed ambiguously, and the broker could **not** be re-read. False
everywhere else, including clean misses — a definitive reject must not park a
position that does not exist.

**4. Unknown outcomes are parked, not discarded** (`oanda_forex_orchestrator.py`).
`_park_unverified_entry` records the entry as `ENTRY_UNRECONCILED`: counted by
the exposure caps so the bot stays conservative, but ignored by `_on_tick`,
because running a software stop against a position that may not exist could
close something the account doesn't hold. It logs CRITICAL, emits telemetry,
and raises a Discord alert.

**5. `_reconcile_unverified_entries`** settles parked records on the existing
liveness loop (every 10s) rather than a fire-and-forget task, which could be
GC'd or swallow an exception unnoticed. Flat → drop the record and start the
cooldown. Open → promote to `OPEN` with the bracket rebuilt from the stored
`sl_dist`/`tp_dist` **around the price actually filled**, since the signal
price may be stale by however long the failure took and a stop measured from
it could sit on the wrong side of the market. The approved *distances* are
preserved exactly, so the cost gate is not reopened. A failed sync leaves the
record parked for the next pass — never resolved by assumption.

## Verification

- `PYTHONPATH=src:. python -m pytest -q` → **343 passed** (325 before this
  work, 331 after DeepSeek's change, 343 now).
- `python -m compileall -q src/` clean.
- Behaviour measured directly against the real classes with a fake broker,
  before and after:

  | Scenario | Before | After |
  |---|---|---|
  | Transient 401 (the incident) | **trade lost** | **trade captured** |
  | Ambiguous failure that filled | no double fill | no double fill |
  | Ambiguous fill on the LAST attempt | **untracked position** | reported and recorded |
  | Business reject | not retried | not retried |
  | Ambiguous + broker unreadable | **silently dropped** | parked + alerted |

- New tests: 5 in `tests/test_oanda_entry.py` (401 retried, incident
  regression, business-reject-still-permanent, last-attempt fill reported,
  `unverified` set/not-set) and 7 in `tests/test_unverified_entry.py` (park,
  don't-park-a-clean-miss, parked record not stop-monitored, reconcile to
  flat / to open / short orientation, sync failure keeps it parked).
- The live soak (`soak.service`, pid 420097, 6 days up) was not restarted or
  touched. The running process holds the old module in memory; these changes
  take effect on its next start.

## Risk & follow-ups

- **Brackets are still anchored to the bar close on the normal entry path.**
  `sl_price`/`tp_price` are computed from `signal.entry_price` while the
  recorded `entry` is the real fill — a mixed anchor that predates this work.
  The retry widens the gap (worst case ~90s: 3 attempts × a 30s HTTP timeout
  plus backoff). Re-anchoring is ~6 lines and the distances are what the cost
  gate approved, so it does not reopen that gate — but it changes bracket
  placement on *every* trade, and this branch is a bracket-sizing experiment.
  **Deliberately left for Brandon's call**, since shipping it mid-soak would
  confound the results. Note the reconcile path already anchors on the fill,
  because for a recovered position there is no honest alternative.
- **`close_position` still no-ops on a cached net of 0** and blends
  `prev_net + total_filled` against a possibly-drifted cache. Unchanged here;
  it is on the close path and deserves its own report. It matters more now
  only in the sense that a parked entry may need force-closing.
- **No same-symbol in-flight guard.** `_exposure_conflict` excludes the symbol
  from both dicts, so nothing structurally prevents two concurrent entries on
  one instrument; the parked record now holds the slot for the unknown case,
  which narrows but does not close it.
- Worst-case entry latency is ~90s. The default executor has 16 workers for 6
  instruments, so this does not starve the stop-close path, but
  `OANDA_ENTRY_MAX_ATTEMPTS` / `OANDA_REQUEST_TIMEOUT` are the knobs if that
  tail proves too long in practice.

## Files touched

- `src/execution/oanda_order_manager.py` — `_error_body`,
  `_order_never_filled`, `_is_permanent_reject` (replacing
  `_is_definitive_order_reject`), `_result_if_at_target`, `unverified` on
  every return path, Glossary.
- `src/execution/oanda_forex_orchestrator.py` — `_park_unverified_entry`,
  `_reconcile_unverified_entries`, the unverified branch on the zero-fill
  path, the reconciler wired into `_liveness_watchdog`, Glossary.
- `src/execution/README.md` — corrected the classification description; added
  the unknown-outcome section.
- `tests/test_oanda_entry.py` — +5 tests, Glossary.
- `tests/test_unverified_entry.py` — new, 7 tests.
