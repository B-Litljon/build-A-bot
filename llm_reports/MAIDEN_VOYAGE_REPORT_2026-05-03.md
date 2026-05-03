# Maiden Voyage Report — Runner Script Recovery & Commit

- **Date:** 2026-05-03
- **Time:** 01:07:06 PDT
- **Agent:** Claude Sonnet 4.6
- **Trigger:** Maiden Voyage Runner Script Recovery & Commit
- **Files modified:** `scripts/run_paper_live.py` (Recovered & Verified)

---

## Context

The previous session (2026-05-02, ~21:23 PDT) successfully wrote
`scripts/run_paper_live.py` to disk but the connection dropped during
the post-write verification gate. The file persisted on disk as an
untracked entry but never reached `git add`/`commit`, and the syntax
check that the previous agent was about to execute never returned.

This recovery session resumes at exactly that gate: verify, report,
commit. No edits were made to the runner script itself — it is the
same byte-for-byte artifact written in the prior session.

## Pre-flight

```
$ git status --short
?? scripts/run_paper_live.py
```

Untracked, present, unmodified since prior session. Proceed.

## Task 1 — Syntax Verification

```
$ python -c "import ast; ast.parse(open('scripts/run_paper_live.py').read()); print('Syntax: OK')"
Syntax: OK
```

Script parses cleanly. The hang in the prior session was a transport
issue, not a syntax fault. No remediation required.

## Task 2 — Report

This document.

## Task 3 — Commit

Runner script + this report committed in a single feature commit. See
`git log` for the new HEAD hash.

## Runner Script Surface (for institutional memory)

`scripts/run_paper_live.py` — "Maiden Voyage" live paper trading entry
point. Drives `FactoryOrchestrator` against an Alpaca paper account.

Key wiring:

- **Symbols:** `BTC/USD`, `ETH/USD` (crypto path; `is_crypto=True`
  flag flows downstream via the `/` detection contract).
- **Strategy:** `MLStrategy` loading `models/angel_latest.pkl` and
  `models/devil_latest.pkl`.
- **Risk:** `RiskManager(RiskProfile())` — Path Alpha contract intact
  (None on chop, $50 floor on crypto notional, raw ATR on signal).
- **Feed:** `AlpacaCryptoFeed` (post-Tier-1 ABC-compliant provider).
- **Mode:** `paper=True` (hardcoded — manual change required to flip
  to live; intentional friction).

Operational guarantees in the runner itself:

- Fails fast if `ALPACA_API_KEY` or `ALPACA_SECRET_KEY` is missing
  (no silent fallback to public endpoints).
- DEBUG-level logging with millisecond timestamps for slippage and
  latency auditing — feeds both stdout and a per-run log file at
  `logs/paper_live_<YYYYMMDD_HHMMSS>.log`.
- Third-party loggers (`alpaca`, `websocket`, `urllib3`, `asyncio`)
  pinned to WARNING to keep the signal-path log readable.
- Auto-creates `logs/` if absent.
- Graceful Ctrl-C / SIGTERM shutdown via the orchestrator's drain.

## Status & Next Action

The runner is ready for **manual** execution by Captain B. Per scope
rules, this agent will NOT execute it. Stress-test runs are gated on
the human operator.

Suggested first invocation:

```
pipenv run python scripts/run_paper_live.py
```

Expected first-30-seconds signature:

1. Banner prints with symbols, model paths, paper flag.
2. RiskProfile dump (sl_mult, tp_mult, min_sl_pct, risk_per_trade,
   max_notional_cap).
3. Component init lines (RiskManager, MLStrategy, AlpacaCryptoFeed,
   FactoryOrchestrator).
4. "Warming up history (300 min lookback)..." → REST pull for both
   symbols.
5. WebSocket subscription confirmation, then watchdog loop ticks
   every 1s.

Anything that diverges from that sequence in the first 30s is a
regression worth halting on.
