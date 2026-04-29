STRATEGY_REGISTRY = {}


def create_strategy(name: str, **kwargs):
    strategy_cls = STRATEGY_REGISTRY.get(name)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    return strategy_cls(**kwargs)
