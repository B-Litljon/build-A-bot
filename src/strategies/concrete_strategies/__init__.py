from .rsi_bbands import RSIBBands
from .sma_crossover import SMACrossover
from .ml_strategy import MLStrategy

STRATEGIES = {
    "rsi_bollinger": RSIBBands,
    "sma_crossover": SMACrossover,
    "ml_strategy": MLStrategy,
}
