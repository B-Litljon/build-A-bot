"""
OrderParams — minimal risk-config dataclass for backtest scenarios.

Note on scope: this dataclass is *backtest-oriented*. It uses percentage
multipliers (e.g. sl_multiplier=0.998 for 0.2% stop loss) which are
incompatible with the ATR-based bracket sizing used in live execution.

Live trading code paths compute brackets directly from base.Signal's
raw_sl_distance / raw_tp_distance fields, populated by strategies from
ATR. OrderParams is retained here only because grid_search_backtest*.py
need a parameter sweep target.

Do not wire this into live execution. If a future feature requires
strategy-specified risk parameters in production, redesign at that point —
this dataclass is not the right home.

STATUS (glossary pass, 2026-07-27): nothing in the repo imports OrderParams,
and the ``grid_search_backtest*.py`` scripts named above no longer exist (only
the stale ``grid_search_results.txt`` output remains). This module is dead as
written. Flagged, not removed.

Glossary:
    OrderParams -- backtest-only risk config. Percentage-multiplier based, so
        it is structurally incompatible with the live ATR-based brackets; see
        GLOSSARY.md ("bracket").
    risk_percentage -- fraction of capital risked per trade (0.02 = 2%).
    tp_multiplier -- take-profit price = entry x this. Above 1.0 for a long
        (1.005 = +0.5%).
    sl_multiplier -- stop-loss price = entry x this. Below 1.0 for a long
        (0.998 = -0.2%).
    use_trailing_stop -- placeholder flag; no code reads it.
"""

from dataclasses import dataclass


@dataclass
class OrderParams:
    """Risk configuration for backtest scenarios. Backtest-only."""

    risk_percentage: float  # fraction of capital risked per trade (e.g. 0.02 = 2%)
    tp_multiplier: float  # take-profit price = entry * tp_multiplier (>1 for long)
    sl_multiplier: float  # stop-loss price = entry * sl_multiplier (<1 for long)
    use_trailing_stop: bool = (
        False  # placeholder; not consumed by current backtest code
    )
