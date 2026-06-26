---
type: handoff
date: 2026-06-26
time: 13:24 PDT
agent: Claude Opus 4.8
model: claude-opus-4-8
trigger: Brandon is recentering the whole project onto ONE simple, hardcoded forex strategy and clearing context to ship. This hands a fresh Claude instance everything needed to implement Gemini's "London Open breakout on GBP/JPY" backtest — the first concrete step of the pivot.
head: 6f8e58290f3467b32a11ad77f5bf35fac919869f
branch: feature/forex-swing-h1 (START A NEW BRANCH for this — see Guardrails)
scope: new isolated module + backtest (no ML)
related:
  - m2m_prompts/2026-06-23_forex-swing-h1-phase-a.md
files_in_scope:
  - src/strategies/concrete_strategies/london_breakout.py  (CREATE)
  - a new standalone backtest script (e.g. backtest_london.py)  (CREATE)
---

# PIVOT → simple London-Open breakout on GBP/JPY — session handoff

**Authored 2026-06-26 13:24 PDT · Claude Opus 4.8.** Read this top to bottom before doing anything. Brandon cleared context to focus on shipping; you are the fresh instance.

## TL;DR — your first action
Implement a **dead-simple, hardcoded "London Open volatility breakout" on GBP/JPY** and **backtest it (vectorized, no ML)**. This is a deliberate strip-down after 18 months of over-built ML. The decision to drop ML is FINAL (Brandon + his architect LLM Gemini agreed) — **do not relitigate it.** Your job is to prove (or disprove) a small, real edge *on paper, after costs* — not to make a pretty backtest.

## Why this pivot (the real goal)
Brandon recentered on the only goal that matters: **make actual money.** Constraints are strict and real:
- **Capital: < $400** (very limited, cannot afford to lose it).
- **Target: ≥ $20/week.** ⚠️ Be honest with him: on $400 that's ~5%/week — extreme. The *first* milestone is NOT $20/week; it's **prove a small positive edge that survives costs, and don't blow up the $400.** Chasing $20/week with leverage is how small accounts go to zero.
- **Simplify** ("Palm Pilot / constraints breed focus"). Back to basics like his original hand-coded algorithm — one pair, one rule he fully understands.
- **Forex** chosen (fits <$400 via tiny "unit" sizing, 24/5, no US Pattern-Day-Trader rule, spot FX avoids CFD/PDT restrictions). GBP/JPY chosen for its explosive session-overlap volatility ("the dragon").

## Honest state of the project (so you don't rebuild what exists or trust what doesn't)
**Nothing here reliably makes money yet — that's the whole reason for the pivot.**
- **1-min ML scalper:** a 4-day live PAPER soak on gold/silver is RUNNING right now (PID in `/tmp/soak.pid`, log `logs/soak_2026-06-22_0120.log`). Infra is flawless but it made **ZERO trades** — the model's confidence never reaches its 0.40 action threshold on metals (maxes ~0.27). DO NOT disturb this soak (see Guardrails).
- **Hourly swing model:** just tested (branch `feature/forex-swing-h1`, uncommitted). Cost-gate thesis proven (the 6 cross pairs become tradeable at H1) but the model FAILS the validation gate (borderline: profit factor 0.69→0.97 after a data-starvation fix, still <1.2). Parked.
- **Stock-picker (investor):** on branch `feature/investor-edge`; barely beats random. Parked.
- Lots of built infra (autopilot rails, data-source mixing, etc.) — all on other branches, all secondary now. The m2m_prompts/ ledger + memory have the details. **Don't get pulled back into any of it.**

## The plan (Gemini's spec — implement this)
**Strategy: London Open volatility breakout, GBP/JPY, deterministic, no ML.**
- **Setup:** compute the High and Low of the **Asian consolidation range** (≈ 00:00–07:00 GMT).
- **Trigger:** during the **London session** (≈ 07:00–15:00 GMT), go **long (+1)** if price breaks above the Asian High, **short (−1)** if it breaks below the Asian Low. One trade per side per day (first break); no re-entry after a stop (decide & document).
- **Risk:** hardcoded **1:2 risk/reward**. Stop at the opposite side of the Asian range (or the range midpoint to keep risk tighter — this is an OPEN decision, test both). Target = 2× the stop distance.
- **Vectorized** NumPy/pandas. **No** LightGBM / sklearn / any ML lib.
- Keep it **isolated** from `src/ml/` and `src/day_trading/`.
- **Output metrics:** Total Trades, Win Rate, Profit Factor, Max Drawdown — **computed WITH realistic costs** (see caveats).

## Repo facts I verified for you (saves you the recon)
- **Where the strategy file goes:** `src/strategies/concrete_strategies/` exists (holds `ml_strategy.py`). Put `london_breakout.py` there.
- **Live strategy interface** (for LATER live wiring, not the backtest): `src/strategies/base.py` → `BaseStrategy.generate_signals(df: pl.DataFrame) -> Signal(direction, entry_price, raw_sl_distance, raw_tp_distance, metadata)`. NOTE: per project ruling, live bracket sizing is owned by the RiskManager's ATR multipliers, and `raw_tp_distance` is intentionally discarded live — so the 1:2 RR will need mapping to RiskManager multipliers when/if you go live. For the BACKTEST, keep it standalone and self-contained (the strategy controls its own SL/TP).
- **Data:** fetch GBP/JPY via `src/data/oanda_provider.py` → `get_historical_bars(symbol="GBP_JPY", timeframe_minutes=..., start=..., end=...)`. `.env` has OANDA practice creds (`set -a; . ./.env; set +a` before running; the provider needs `OANDA_API_KEY`). Granularity is configurable (M1…D supported).
- **Backtest harness to crib (don't reuse directly):** `backtest_quick.py` shows the data-load pattern but is ML/SPY-specific. Gemini said "edit backtest_quick.py"; I recommend instead **creating a clean `backtest_london.py`** to honor the "keep it isolated" instruction. Minor deviation — document it.
- **Runtime:** venv python `/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python`; run with `PYTHONPATH=src:.` from repo root.

## Honest caveats & discipline (READ — this is where it goes wrong)
1. **London breakout is a famous, heavily-arbitraged retail strategy.** Naive backtests flatter it badly. Treat a rosy result with suspicion.
2. **Model costs or the backtest is a lie.** GBP/JPY spread is ~2–3 pips normally and **blows out at the volatile open** (the exact moment you're trading) + slippage + stop-hunting. Bake in a realistic per-trade cost (start ~3 pips, and stress-test at 5). A strategy that only wins at 0 cost is dead on arrival.
3. **Position sizing on <$400 is brutal with a wide stop.** GBP/JPY breakout stops can be 30–60 pips. Risking 1–2% ($4–8) means tiny unit sizes — confirm the math is even viable and that one stop-out doesn't break the account.
4. **Don't over-optimize** the session windows / stop variant on one dataset — that's the data-snooping trap (try 50 settings, keep the lucky winner, it fails live). Pick sensible defaults, test out-of-sample, prefer robust over spectacular.
5. **The arc is backtest → paper → tiny-live, in that order.** This task is the backtest. If it shows a real post-cost edge, NEXT is forward paper-trading on the OANDA practice account (infra exists), THEN tiny real size. Do not skip steps.

## Open decisions to confirm with Brandon (don't silently guess)
- **Timeframe:** the Asian-range / London-breakout pattern is intraday — it needs sub-hourly bars (M5 or M15), NOT H1. Gemini didn't specify. Recommend M15 to start; confirm.
- **Session times & DST:** 00:00–07:00 / 07:00–15:00 GMT — OANDA timestamps are UTC. London shifts with DST (BST in summer); decide whether to anchor to UTC or local London and document. This materially affects the range.
- **Stop placement:** opposite Asian range edge vs midpoint (risk/frequency tradeoff). Test both, report both.

## Guardrails (hard)
- **DO NOT touch the running soak**, `src/execution/`, `run_soak.sh`, or anything the soak loads. Verify the soak is still alive (`ps -p $(cat /tmp/soak.pid)`) but leave it be.
- **DO NOT relitigate dropping ML.** Brandon + Gemini decided. Build the deterministic rule.
- **DO NOT water down metrics to force a "pass."** We've twice caught watered-down gates this project; report the honest result, good or bad. If it has no edge after costs, SAY SO — that's a valid, valuable outcome.
- **Start a clean branch** (e.g. `feature/london-breakout`) off `main` or current — keep it isolated from the parked ML branches.

## Working with Gemini (active co-developer — not a one-off)
Gemini is Brandon's architect LLM with 18 months of project history, and it is an **ongoing part of this build**, not a one-time consult. The development loop is **back-and-forth M2M between you (Claude) and Gemini**:
- **Gemini architects / proposes** (the London-breakout spec came from it). It is repo-aware — treat it as a **peer**, cross-verify its claims, don't brief it as if it's blind (see `feedback_architect_validation`).
- **You (Claude) implement + VERIFY by running + report honest results** (trades, win rate, profit factor, drawdown at real costs).
- **Gemini reviews and refines** the rules (session times, stop variant, filters); you re-run. Repeat until the edge is proven or clearly absent.
- **Log every exchange in both directions** as dated entries in `m2m_prompts/` per `reference_m2m_prompts_ledger` (frontmatter with `to:`/`from:`/`status:`/`result_commit:`). Brandon ferries the prompts between the two of you (copy-paste via his GUIs). So: write your hand-back-to-Gemini as a clean m2m doc (what you built, the real numbers, your open questions for it), and expect Gemini's next instruction as an m2m in return. Keep the ledger as the shared source of truth so neither side loses the thread.

## Pointers
- **Memory** (`MEMORY.md` index): read `feedback_minimize_finance_jargon` (Brandon wants ALL jargon kept minimal, plain language, short answers), `feedback_no_confabulated_monitoring` (verify the soak is alive before claiming it), `feedback_verify_retrain_handoffs` (re-run before trusting any "it works" claim), `reference_m2m_prompts_ledger` (how handoffs are logged), `project_architect_workflow` (Gemini is the repo-aware architect Brandon consults; peer-verify, don't treat as blind).
- **Reporting:** Gemini asked the implementing model to write a markdown report in `llm_reports/` when done — filename `YYYY-MM-DD_HHMM.md`, with timestamp, model, summary of the prompt, what you did, deviations, and what's left. Follow the project convention in `llm_reports/README.md` + `_TEMPLATE.md`.

## Suggested first steps
1. Verify soak alive (read-only), then start branch `feature/london-breakout`.
2. Confirm timeframe (M15?) + session/DST handling with Brandon.
3. Fetch ~2–3 yrs GBP/JPY M15 from OANDA; build `london_breakout.py` (pure vectorized rules) + `backtest_london.py`.
4. Report Trades / Win Rate / Profit Factor / Max Drawdown **at 0, 3, and 5 pips cost.** Be honest about whether an edge survives.
