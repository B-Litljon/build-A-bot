# `dashboard/`

A read-only web view of the soak, so checking on the bot doesn't mean SSH and
`grep`. Two pieces:

| Path | What | Who writes it |
|---|---|---|
| `api/` | Rust (axum) HTTP API — reads the bot's telemetry, merges broker truth | scaffold by Claude, extended by Brandon |
| `web/` | TypeScript front-end (Vite) | Brandon |

**Nothing here can affect trading.** It only reads files and calls read-only
broker endpoints; no route mutates anything, and the OANDA client (once
written) should expose no order or close methods at all — the safest guarantee
is a capability that doesn't exist. The control tools that *can* start and stop
the soak already live in `trading_mcp.py`, behind a two-step confirm token.

## Where the data comes from

Three sources, and it matters which answers what:

| Question | Source | Why |
|---|---|---|
| Is the bot alive? What does it hold? | `logs/status.json` | Written atomically each bar by `src/core/events.py`; carries the bot's own view |
| What has it been doing? | `logs/events-YYYY-MM-DD.jsonl` | Append-only fact stream — per-bar probabilities, entries, exits, vetoes, guard blocks |
| How much money did it make? | **OANDA REST** | The only authority on fills and realized P&L. The bot's log knows what it *asked* for, not what it *got* |

That last row is why `oanda.rs` exists at all. On 2026-07-30 the bot's log
showed four orders and four closes; only the broker knew they netted −$2.97.

## Event stream

One JSON object per line. `ts` (UTC ISO-8601) and `ev` on every event.

| `ev` | Key fields | Notes |
|---|---|---|
| `boot` | `pid`, `symbols`, `granularity`, `units`, `cooldown_s`, `max_per_ccy`, `angel_thr`, `devil_thr` | Once per process start, after boot reconciliation |
| `bar` | `sym`, `bar_ts`, `close`, `angel`, `devil`, `outcome`, `proposed` | **Every evaluation.** `outcome` ∈ `angel_reject` \| `devil_veto` \| `agreement`; `devil` is null when the Angel never cleared its bar |
| `entry` | `sym`, `dir`, `units`, `entry`, `sl`, `tp`, `angel`, `devil` | After the fill is recorded |
| `exit` | `sym`, `units`, `dir`, `entry`, `sl`, `tp`, `reason`, `attempt` | `reason` ∈ `watchdog` \| `flatten`. **No exit price** — get it from OANDA |
| `gate_veto` | `sym`, `gate`, `spread`, `regime`, `time`, `devil_approved` | Cost/regime/time chop gates; counters are cumulative totals |
| `guard_block` | `sym`, `guard`, `detail`, `remaining_s`, `total` | `guard` ∈ `cooldown` \| `exposure` |
| `calib` | `sym`, `n`, `alpha_emp`, `med_spread_pct`, `med_baseline_natr` | Periodic spread calibration |
| `stream` | `kind`, `sym?`, `delay_s?`, `age_s?`, `attempt?` | `kind` ∈ `disconnect` \| `seam_catchup` \| `seam_backfill` |
| `heartbeat` | `sym`, `median`, `p75`, `max`, `proposed`, `n_bars`, `threshold` | 30-bar probability summary |

Volume is low — roughly 800 `bar` events a day across eight instruments, a few
dozen of everything else.

## Running it

```bash
cd dashboard/api
cargo run                      # http://127.0.0.1:8787
tailscale serve --bg 8787      # then reachable from the phone, tailnet TLS
```

| Env | Default | Meaning |
|---|---|---|
| `SOAK_REPO_DIR` | the repo path | Root for every other path |
| `EVENTS_DIR` | `<repo>/logs` | Where telemetry is read from |
| `DASHBOARD_BIND` | `127.0.0.1:8787` | **Keep it loopback**; use `tailscale serve` for remote |
| `DASHBOARD_WEB_DIR` | `dashboard/web/dist` | Built front-end, served at `/` when present |
| `DASHBOARD_TAIL_MS` | `1000` | JSONL poll interval |
| `DASHBOARD_RING` | `20000` | Events held in memory |

## Status: what's built, what's next

**Working now** — `config.rs`, `events.rs` (incremental tailer + ring),
`status.rs` (snapshot + freshness), and three routes:

- `GET /api/health` — API uptime, buffered events, **and whether the BOT looks
  alive** (snapshot age against two bar periods). An API that answers 200 while
  the bot is dead is worse than no dashboard.
- `GET /api/status` — the snapshot, tagged `ok` / `missing` / `unreadable`.
- `GET /api/events?ev=&limit=` — newest-first, optionally filtered by kind.

`cargo test` covers the parsing edges that actually happen: a malformed line is
skipped not fatal, a half-written tail line is re-read once complete,
incremental reads never duplicate, a truncated file resets cleanly, and a
missing/garbage snapshot reports honestly instead of looking healthy.

**To build** — in the order they earn their keep:

1. `oanda.rs` — read-only broker client. `GET /v3/accounts/{id}/summary`,
   `/openTrades`, and `/transactions/idrange?from=&to=` walked forward from the
   last seen ID. Cache with a ≥5 s TTL so a refreshing browser can't hammer the
   broker. Token from `OANDA_API_KEY`, server-side only, never in a response.
   Practice host is `api-fxpractice.oanda.com`.
2. `derive.rs` — closed trades (pair `ORDER_FILL`s by instrument, `pl` field is
   realized P&L in account currency), equity curve, win rate, the funnel.
3. Remaining routes — `/api/summary` (the phone view), `/api/positions`,
   `/api/trades`, `/api/signals`, `/api/gates`, `/api/calibration`.
4. `sse.rs` — `tokio::sync::broadcast` fed by the tailer, streamed at
   `/api/events/stream`. The front-end should still poll `/api/summary` every
   10 s as the fallback that survives a dropped stream.
5. `web/` — Vite + TS. Suggested panels: status bar; **angel probability per
   bar against the 0.40 line** (the "why isn't it trading?" view, and the whole
   reason `bar` events exist); trade log with P&L; the veto/guard funnel;
   per-instrument cost table. `ts-rs` on the Rust structs exports types into
   `web/src/types/` so the contract lives in one place.

Two details that will bite otherwise:

- **`GET /positions/{instrument}` 404s for never-traded instruments.** That's
  normal, not an error — the boot reconciler hits it every start.
- **Cross-check any P&L view against the known-good day**: 2026-07-30 was four
  fills netting −$2.97, ending balance 99,994.33.
