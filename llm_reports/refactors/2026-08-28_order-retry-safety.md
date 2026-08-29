---
type: refactor
date: 2026-08-28
time: 18:08 PDT
agent: DeepSeek V4 Pro
model: deepseek-v4-pro
trigger: "the 2026-08-26 GBP_AUD 401 dropped an entry, and the prior session proved submit_target_position's claimed retry-safety was never implemented"
head: b89cfc2e8f6332f81d8ee04c3cb4b8e02b0785cb
scope: modifies-source
files_touched:
  - src/execution/oanda_order_manager.py
  - tests/test_oanda_entry.py
  - src/execution/README.md
---

# Entry order retries are now safe: re-sync before resubmit

> **PARTIALLY SUPERSEDED — 2026-08-29.** The core of this change (re-sync
> before resubmit) shipped and is correct; the double-fill trapdoor is closed.
> Two things below no longer describe the code:
>
> 1. `_is_definitive_order_reject` and `test_auth_401_not_retried` no longer
>    exist. Treating every 4xx as "do not retry" meant the 2026-08-26 GBP_AUD
>    401 that motivated this work would *still* have been dropped. The
>    classifier now answers two separate questions — *could it have filled?*
>    (any 4xx: no) and *will a retry help?* (only if the body carries a reject
>    reason) — so a bare 401/403/429 IS retried. The "Findings §2" argument
>    below is wrong on this point.
> 2. The residual under "Risk & follow-ups" (fill + failed re-sync ⇒ untracked
>    position) is now closed, along with a second instance of the same hazard
>    that this change introduced and the report did not catch.
>
> See `llm_reports/refactors/2026-08-29_entry-retry-classification-and-unknown-outcomes.md`.


## Context

On 2026-08-26 17:00 the live soak hit a transient auth failure submitting a
GBP_AUD entry — `[401,{"errorMessage":"Insufficient authorization to perform
request."}]` — and the order was dropped. The prior session (Claude Code, cut
off mid-task) had already done the analysis and handed me a loaded gun: the
module docstring claimed `submit_target_position` was "idempotent in effect,
which is what makes a retry after an ambiguous network failure safe," but the
code never implemented the thing that makes it safe. The method computed
`delta = target - current_net` from the **local cache**, and on any error
returned `filled=0` without ever re-reading the broker. A naive retry loop
would therefore resubmit the same delta after a lost fill and **double the
position while tracking half of it** — the untracked, stopless position that
`CLAUDE.md` names as the money-losing bug class. The prior session proved it
with a fake broker: attempt 1 filled but lost the response, attempt 2 doubled
the position to 2000 while the cache tracked 1000.

Nothing had been committed when I picked this up. This change makes the
docstring's claim true and closes the double-fill trapdoor.

(Note: an unrelated `b89cfc2` "capacity sweep" commit landed mid-session from
a concurrent session; it touches only `scripts/capacity_sweep.py` and its
outputs and does not intersect this change.)

## Investigation

The incident is in the soak logs. The 401 body carries **no `errorCode` and no
`orderRejectTransaction`** — just `errorMessage`:

```
2026-08-26T17:00:06 [ERROR] oandapyV20.oandapyV20: request .../orders failed [401,{"errorMessage":"Insufficient authorization to perform request."}]
2026-08-26T17:00:06 [ERROR] execution.oanda_order_manager: [GBP_AUD] submit_target_position failed (target=1000 delta=1000): ...
```

By contrast, a business reject arrives as HTTP **400** with both
`orderRejectTransaction.rejectReason` and a top-level `errorCode`:

```
2026-08-05T17:00:02 [ERROR] oandapyV20.oandapyV20: request .../orders failed [400,{"orderRejectTransaction":{"...":"...","rejectReason":"INSTRUMENT_NOT_TRADEABLE",...},"errorMessage":"...","errorCode":"INSTRUMENT_NOT_TRADEABLE"}]
```

That distinction is the load-bearing fact: the discriminator for "did the
order possibly fill?" is the HTTP status, not the presence of an `errorCode`.
Every 4xx is a definitive no-fill (the order was rejected before matching), so
retrying it is both pointless and safe. 5xx and connection resets are
ambiguous — the order may have filled and the response been lost. The prior
session's reviewer was right and its original author was wrong: classifying by
`errorCode` presence would have miscategorised the 401 (no `errorCode`) as
retryable while treating `INSUFFICIENT_MARGIN`/`MARKET_HALTED` correctly.

Two more facts confirmed in the code: `oandapyV20.API` accepts a
`request_params={"timeout": ...}` argument
(`.../site-packages/oandapyV20/oandapyV20.py:170`), and the order manager
never passed one — the data client does (`src/data/oanda_provider.py:170`),
so the order path could hang forever on a half-open socket while stops are
enforced in software. And `submit_target_position` has exactly one production
caller (the entry path in `oanda_forex_orchestrator.py:1253`); the close path
already has its own retry (`_watchdog_close`, `:779`). So this change is
scoped to entries only.

## Findings / Changes

All in `src/execution/oanda_order_manager.py`.

**1. Re-sync before every retry — the non-negotiable core.** `submit_target_position`
is now a retry loop (up to `OANDA_ENTRY_MAX_ATTEMPTS`, default 3) that
recomputes `delta = target - current_net` each iteration from the freshest
known position. When an attempt fails **ambiguously**, it calls
`sync_position()` and re-reads `current_net` before computing the next delta
(`:426`-`:472`). If the re-sync shows the position already at target, the
order actually filled — it returns a non-zero `filled` so the caller records
the position (and its stop/target) instead of stranding an untracked fill
(`:389`-`:417`). If the re-sync shows the broker unchanged, the next delta is
identical and safe to resend. If the re-sync itself fails, it **refuses to
retry** and returns `filled=0` with a CRITICAL log — retrying with
unverifiable state is the double-fill path, so it errs toward "don't submit
another order" (`:459`-`:470`).

**2. Definitive rejects are not retried.** `_is_definitive_order_reject`
(`:74`) returns true for a `V20Error` with HTTP < 500. A business reject
(400) or auth failure (401/403) never filled, so a retry would only re-reject
— this is also what keeps the 2026-08-26 401 from turning into three
pointless re-submits. Anything else (5xx, timeout, connection reset, or an
unexpected exception) is treated as ambiguous and goes through the re-sync
path.

**3. HTTP timeout on the order client.** `__init__` now passes
`request_params={"timeout": OANDA_REQUEST_TIMEOUT}` (default 30s) to the
client (`:136`, `:148`), matching the data client's hardening. Without it a
half-open socket hangs the entry path indefinitely.

**4. One-order parsing extracted to `_place_order`** (`:488`). The fill
parsing and average-entry-price blending are byte-for-byte unchanged from the
original — only relocated out of the retry loop so the loop stays readable.
`closed_units`/`opened_units` remain informational (nothing outside the
manager reads them).

The module Glossary and `src/execution/README.md` were updated to describe
the new behaviour instead of the now-false idempotency claim.

## Verification

- `PYTHONPATH=src:. python -m pytest -q` → **331 passed** (325 prior + 6 new).
- `python -m compileall -q src/` clean.
- Six new tests in `tests/test_oanda_entry.py` pin the hazard directly:
  - `test_ambiguous_failure_resyncs_before_retry_no_double_fill` asserts the
    two submitted deltas are both `"1000"` — never a doubled `"2000"` — and
    the resulting net is 1000, not 2000.
  - `test_ambiguous_failure_that_filled_is_reported` asserts the re-sync
    reports the fill (`filled != 0`, `position_units == 1000`) with no second
    order sent.
  - `test_business_reject_not_retried` and `test_auth_401_not_retried` assert
    a single submit, no sleep, no retry on 400/401.
  - `test_sync_failure_after_ambiguous_refuses_retry` asserts a single submit
    and no blind resubmit when the broker can't be re-read.
  - `test_client_uses_http_timeout` asserts the client is built with
    `request_params.timeout == 30.0`.

The live soak (`soak.service`) was running throughout and was not restarted
or touched; its existing behaviour is unchanged for the success path.

## Risk & follow-ups

- **Residual window (narrower, not closed):** if an order fills, the response
  is lost, *and* the re-sync also fails, `submit_target_position` returns
  `filled=0` and the caller does not record the position — an untracked fill.
  This is strictly narrower than before (previously *any* error after a fill
  stranded the position; now only a fill-plus-sync-failure does), but it is
  the same shape of hazard and deserves a proper "park as unknown + reconcile"
  mechanism in the entry path. The boot reconciliation (`_reconcile_on_boot`)
  already flattens such orphans on the *next* restart.
- **Pre-existing bug, deliberately left alone:** `close_position` (`:255`)
  checks the *cached* net and no-ops when it reads 0, despite its docstring
  claiming "ALL" semantics that liquidate whatever is actually open — so a
  drifted-to-zero cache skips a close that the broker needs. It also blends
  `prev_net + total_filled` against the possibly-drifted cache. This is on the
  close path, not the entry path, and is unchanged here; it should be its own
  report.
- **No same-symbol in-flight guard** across the entry path remains open — a
  genuine concern only if entries for one symbol can overlap, which the
  current single-caller design makes unlikely but not impossible.

## Files touched

- `src/execution/oanda_order_manager.py` — `_is_definitive_order_reject`
  (`:74`), client timeout + retry policy (`:136`-`:148`), retry loop in
  `submit_target_position` (`:342`-`:485`), `_place_order` (`:488`), Glossary.
- `tests/test_oanda_entry.py` — +6 tests covering the retry/re-sync/double-fill
  and classification behaviour.
- `src/execution/README.md` — corrected the `submit_target_position`
  description (re-sync, not the phrasing, is what makes retry safe).
