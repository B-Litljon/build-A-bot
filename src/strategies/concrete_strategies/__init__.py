from .rsi_bbands import RSIBBands
from .sma_crossover import SMACrossover

STRATEGIES = {
    "rsi_bollinger": RSIBBands,
    "sma_crossover": SMACrossover,
}
