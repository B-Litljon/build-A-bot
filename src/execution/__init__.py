"""
src/execution — trade lifecycle orchestration for the live lanes.

Exports:
    RiskManager          — bracket sizing, position sizing, Gate C entry windows

Note: OandaForexOrchestrator -- the orchestrator actually running live -- is
      intentionally not exported here; run_oanda.py imports it by path.
      The Alpaca scalper lane (LiveOrchestrator, FactoryOrchestrator) was
      DELETED 2026-09-16 with the rest of its dormant stack; git history
      has it if it is ever resurrected.

Glossary:
    __all__ -- the deliberately short public surface: importing this package
        must not pull in the heavy orchestrators.
"""

from .risk_manager import RiskManager

__all__ = ["RiskManager"]
