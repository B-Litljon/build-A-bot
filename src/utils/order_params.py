class OrderParams:
    """
    Defines parameters for order calculation and risk management.

    Attributes:
        risk_percentage (float): Percentage of capital to risk per trade.
        tp_multiplier (float): Multiplier to calculate take-profit level from entry price.
        sl_multiplier (float): Multiplier to calculate stop-loss level from entry price.
        sma_short_period (int, optional): Period for short-term SMA (trailing stop).
        sma_long_period (int, optional): Period for long-term SMA (trailing stop).
        sma_crossover_type (str, optional): "long" or "short" for trailing stop type.
        use_trailing_stop (bool, optional): Whether to use trailing stop-loss. Defaults to False.
        **kwargs: For adding other custom parameters.
    """

    def __init__(self, risk_percentage: float, tp_multiplier: float, sl_multiplier: float,
                 sma_short_period: int = None, sma_long_period: int = None,
                 sma_crossover_type: str = None, use_trailing_stop: bool = False, **kwargs):
        self.risk_percentage = risk_percentage
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.sma_short_period = sma_short_period
        self.sma_long_period = sma_long_period
        self.sma_crossover_type = sma_crossover_type
        self.use_trailing_stop = use_trailing_stop
        self.kwargs = kwargs

    def __str__(self):
        return f"OrderParams(risk_percentage={self.risk_percentage}, tp_multiplier={self.tp_multiplier}, sl_multiplier={self.sl_multiplier}, use_trailing_stop={self.use_trailing_stop}, ...)"
