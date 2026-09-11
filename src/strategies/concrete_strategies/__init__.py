"""
Concrete strategy implementations, plus a name-to-class registry.

The registry resolves **lazily**. Selecting one strategy must not import the
other six, because `run_oanda.py` — the live bot's entry point — reads this
registry, and the experimental library modules have no business in the live
import graph. Before 2026-09-02 this file imported all seven eagerly, which
meant a typo in an unused research strategy would stop the bot booting and the
watchdog's crash-loop brake would then hold it down for 15 minutes.

`MLStrategy` is the exception and is imported eagerly: it is the strategy the
soak actually serves, and every live module imports it directly anyway.

Glossary:
    STRATEGIES -- maps a config string to a strategy class. A Mapping, not a
        dict: keys() is free, but a lookup imports that one module on demand and
        caches it. `STRATEGIES[name]()` behaves exactly as it did when this was
        a plain dict.
    _REGISTRY -- the underlying "module:ClassName" strings. Strings, not
        classes, so nothing is imported at package import time.
    build_strategy -- name + kwargs -> instance. The registry's first real
        consumer; prefer it over reaching into STRATEGIES directly.
    __getattr__ -- PEP 562 module-level hook, so
        `from strategies.concrete_strategies import DonchianBreakoutStrategy`
        still works and still imports only that module.

⚠️ Of the seven registered strategies only `ml_strategy` has ever passed a gate.
The five library strategies measured ZERO gross expectancy across 28,492 trades
(llm_reports/recons/2026-09-02_strategy-library-behavior-matrix.md) and
`regime_router` currently routes every regime to a stand-down. They are research
instruments; `run_oanda.py` warns if you select one.
"""

from importlib import import_module
from typing import Any, Dict, Iterator, Mapping, Type

# Eager: this is the live path. Every orchestrator imports it directly too.
from .ml_strategy import MLStrategy

# name -> "module:ClassName", relative to this package.
_REGISTRY: Dict[str, str] = {
    "ml_strategy": "ml_strategy:MLStrategy",
    "sma_crossover": "sma_crossover:SMACrossoverStrategy",
    "rsi_mean_reversion": "rsi_mean_reversion:RSIMeanReversionStrategy",
    "bollinger_breakout": "bollinger_breakout:BollingerBreakoutStrategy",
    "donchian_breakout": "donchian_breakout:DonchianBreakoutStrategy",
    "momentum": "momentum:MomentumStrategy",
    "regime_router": "regime_router:RegimeRouterStrategy",
}


def _resolve(spec: str) -> Type[Any]:
    module_name, class_name = spec.split(":", 1)
    return getattr(import_module(f"{__name__}.{module_name}"), class_name)


class _LazyStrategyRegistry(Mapping):
    """
    A read-only name -> class mapping that imports on lookup.

    Iterating names and calling ``keys()`` import nothing, which is what lets
    ``argparse(choices=sorted(STRATEGIES.keys()))`` enumerate every strategy
    without loading any of them.
    """

    def __getitem__(self, name: str) -> Type[Any]:
        try:
            spec = _REGISTRY[name]
        except KeyError:
            raise KeyError(
                f"unknown strategy {name!r}; known: {sorted(_REGISTRY)}"
            ) from None
        return _resolve(spec)

    def __iter__(self) -> Iterator[str]:
        return iter(_REGISTRY)

    def __len__(self) -> int:
        return len(_REGISTRY)

    def __repr__(self) -> str:
        return f"<lazy strategy registry: {sorted(_REGISTRY)}>"


STRATEGIES = _LazyStrategyRegistry()


def build_strategy(name: str, **params: Any) -> Any:
    """Instantiate a registered strategy by name, importing it on demand."""
    return STRATEGIES[name](**params)


# ClassName -> module, for the PEP 562 hook below.
_CLASS_TO_MODULE: Dict[str, str] = {
    spec.split(":", 1)[1]: spec.split(":", 1)[0] for spec in _REGISTRY.values()
}


def __getattr__(name: str) -> Any:
    """
    Lazily expose the strategy classes as package attributes.

    Keeps `from strategies.concrete_strategies import MomentumStrategy` working
    for tests and analysis tooling, while importing only that one module.
    """
    module_name = _CLASS_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module_name}"), name)


def __dir__() -> list:
    return sorted(set(globals()) | set(_CLASS_TO_MODULE))


__all__ = [
    "STRATEGIES",
    "build_strategy",
    "MLStrategy",
    "SMACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "BollingerBreakoutStrategy",
    "DonchianBreakoutStrategy",
    "MomentumStrategy",
    "RegimeRouterStrategy",
]
