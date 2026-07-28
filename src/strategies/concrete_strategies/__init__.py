"""
Concrete strategy implementations, plus a name-to-class registry.

Glossary:
    STRATEGIES -- maps a config string to a strategy class, so a strategy can
        be selected by name. Currently holds only "ml_strategy"; note
        MLFactoryStrategy is deliberately NOT registered here and is imported
        directly by the Factory orchestrator instead.
"""

from .ml_strategy import MLStrategy

STRATEGIES = {
    "ml_strategy": MLStrategy,
}
