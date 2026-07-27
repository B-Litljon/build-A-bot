"""
src/execution — Factory trading orchestration layer.

Exports:
    FactoryOrchestrator  — async trade lifecycle manager (factory path)
    RiskManager          — bracket sizing, position sizing, A3 chop filter

Note: LiveOrchestrator (legacy monolith) is intentionally excluded.
      It lives in live_orchestrator.py and is quarantined until Tier 3
      decoupling is complete.

Note also that OandaScalperOrchestrator -- the orchestrator actually running
live -- is likewise not exported here; run_oanda.py imports it by path.

Glossary:
    __all__ -- the deliberately short public surface. The exclusions above are
        intentional, not oversights: importing this package must not pull in
        the heavier orchestrators.
"""

from .factory_orchestrator import FactoryOrchestrator
from .risk_manager import RiskManager

__all__ = ["FactoryOrchestrator", "RiskManager"]
