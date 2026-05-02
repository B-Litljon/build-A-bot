TO:   Head Orchestrator (Incoming Agent)
FROM: Claude Sonnet 4.6 / Gemini Lead Architect pipeline
DATE: 2026-05-02
RE:   Build-A-Bot — Post Act 3 Handoff

================================================================
REPO STATE
================================================================

Branch:      main
HEAD:        0aeb058
Clean tree:  YES (no uncommitted changes)

Recent log:
  0aeb058  fix(execution): Act 3 - Path Alpha risk migration and lock implementation
  323bf09  feat(sdk): Act 2 — migrate MLStrategy to BaseStrategy, fix ATR fallback
  8ad654e  chore(strategies): remove V1 strategies, clean registries and tests

================================================================
WHAT WAS JUST COMPLETED (Act 3)
================================================================

Five files were committed in 0aeb058:

1. src/execution/__init__.py
   - SEVERED the eager LiveOrchestrator import that was crashing all
     factory-path imports. Now exports only FactoryOrchestrator and
     RiskManager. LiveOrchestrator remains quarantined on disk.

2. src/execution/risk_manager.py  (Path Alpha)
   - calculate_bracket() now returns Optional[Tuple]. Returns None
     (A3 chop filter) when 0.5x ATR < 0.15% absolute SL floor, instead
     of silently promoting the stop. Caller must handle None = skip trade.
   - calculate_quantity() accepts cash and is_crypto params. Uses
     cash * 0.95 as buying-power cap for crypto (Alpaca reports crypto
     funds in account.cash, not account.buying_power).
   - $50 minimum notional guard: returns 0.0 for zombie fractional trades.

3. src/execution/factory_orchestrator.py
   - Per-symbol asyncio.Lock via defaultdict(asyncio.Lock). Both
     _execute_buy and _close_position acquire the lock before touching
     active_positions. Double-check pattern inside lock prevents TOCTOU.
   - Delegates bracket computation to RiskManager (Path Alpha). Skips
     trade if calculate_bracket() returns None.
   - Passes account.cash and is_crypto flag to calculate_quantity().

4. src/strategies/concrete_strategies/ml_strategy.py
   - Removed local bracket constants (SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
     MIN_SL_PCT) and the inline HF7 floor block. RiskManager owns all of
     this now.
   - Signal now carries raw atr_abs in both raw_sl_distance and
     raw_tp_distance. RiskManager applies multipliers downstream.
   - Added warnings.filterwarnings to suppress Polars join_asof noise.

5. llm_reports/TIER_1_FOUNDATIONS_REPORT.md
   - Act 3 section appended with full verification results.

================================================================
VERIFICATION RESULTS (verbatim, run at commit time)
================================================================

  $ python -c "import ast; ast.parse(open('src/execution/__init__.py').read())" && echo "Init Syntax: OK"
  Init Syntax: OK

  $ python -c "import sys; sys.path.append('src'); from execution import FactoryOrchestrator, RiskManager; print('Factory Imports: OK')"
  Factory Imports: OK

================================================================
KNOWN ENVIRONMENT BLOCKER (NOT resolved — needs Captain B)
================================================================

polars is not installed in the pipenv virtualenv. MLStrategy uses Polars
DataFrames for feature generation. The factory import path is now clean,
but you CANNOT run a live smoke test until this is resolved.

Fix: run `pipenv install polars` or `pipenv sync` in the project root.

================================================================
ARCHITECTURAL INVARIANTS — DO NOT VIOLATE
================================================================

- Signal.raw_sl_distance and raw_tp_distance carry RAW ATR (no multipliers).
  RiskManager.calculate_bracket() applies sl_atr_multiplier and
  tp_atr_multiplier exclusively. Applying them anywhere else re-introduces
  the double-multiplication bug fixed in this act.

- Crypto symbols contain "/" (e.g. BTC/USD). Use this to set is_crypto=True
  and switch buying-power source to account.cash in calculate_quantity().

- calculate_bracket() returning None = low volatility, skip the trade.
  Do not fall back to a default bracket. That was the A3 bug.

- $50 minimum notional: calculate_quantity() returns 0.0 below this.
  Check for qty == 0.0 (not <= 0) as the skip sentinel.

- Per-symbol asyncio.Lock wraps BOTH entry and exit order submission.
  Do not move order logic outside the lock context.

================================================================
QUARANTINED FILES — DO NOT IMPORT OR MODIFY
================================================================

  src/execution/live_orchestrator.py
    Tier 3 decoupling work is deferred. This file is not on the factory
    path and must not be touched until that work is formally scoped.

================================================================
WHAT COMES NEXT (suggested)
================================================================

1. Captain B runs: pipenv install polars (or pipenv sync)
2. Smoke test: run FactoryOrchestrator against Alpaca paper account
3. Verify A3 filter fires correctly on low-volatility symbols
4. Verify crypto sizing uses account.cash (not buying_power)
5. Tier 3 scoping: decouple / rewrite live_orchestrator.py

Full report log: llm_reports/  (read TIER_1_FOUNDATIONS_REPORT.md for
complete Act-by-Act history; PATH_ALPHA_REFACTOR_2026-05-02.md for
detailed before/after diffs of this act's logic changes.)
