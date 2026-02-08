from strategies.concrete_strategies.rsi_bbands import RSIBBands
from strategies.concrete_strategies.sma_crossover import SMACrossover


STRATEGY_REGISTRY = {
    "rsi_bollinger": RSIBBands,
    "sma_crossover": SMACrossover,
}


def create_strategy(name: str, **kwargs):
    strategy_cls = STRATEGY_REGISTRY.get(name)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    return strategy_cls(**kwargs)
