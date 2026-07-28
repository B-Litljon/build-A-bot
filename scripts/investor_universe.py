"""
V4 Investor stock universe -- the list of companies the monthly ranker chooses
from, plus each one's sector.

Single source of truth, imported by the data miner, the feature pipeline and
the portfolio orchestrator, so the three can never disagree about which
companies exist.

Glossary:
    UNIVERSE -- the 96 tickers considered each month. Widened from 46 on
        2026-07-03 to cover all 11 sectors. Every name was chosen to have a
        full 5-year daily history -- no recent listings or spin-offs -- so that
        walk-forward folds have data at every point and are not silently
        unbalanced by companies that did not yet exist.
    SECTORS -- ticker to sector name. Used by the orchestrator's diversified
        selection to cap how many picks may come from one sector, so a single
        sector's bad month cannot sink the whole basket. MUST be kept in sync
        with UNIVERSE.
    GICS -- the standard 11-sector classification scheme these names follow.
"""

# V4 Investor Stock Universe — Sector-Balanced Large-Caps (all 11 GICS sectors)
# Deduplicated definition used by data miner, feature pipeline, and orchestrator.
#
# 2026-07-03 widening: 46 → 96 names. Adds Real Estate (previously absent),
# roughly doubles Utilities/Materials/Energy, caps Financials at ~10% of the
# basket. Every name has a full 5-year daily price history (no recent IPOs /
# spin-offs) so walk-forward folds stay balanced.

# GICS sector for every universe member. Used by the orchestrator's
# diversified selection (max picks per sector) — keep in sync with UNIVERSE.
SECTORS: dict[str, str] = {
    **dict.fromkeys(
        ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "TXN"],
        "tech"),
    **dict.fromkeys(
        ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA", "TMUS"],
        "communication"),
    **dict.fromkeys(
        ["AMZN", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "GM", "YUM"],
        "consumer_discretionary"),
    **dict.fromkeys(
        ["WMT", "PG", "KO", "PEP", "COST", "MDLZ", "CL", "GIS", "MO"],
        "consumer_staples"),
    **dict.fromkeys(
        ["JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK"],
        "financials"),
    **dict.fromkeys(
        ["JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "MDT"],
        "health_care"),
    **dict.fromkeys(
        ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "OXY"],
        "energy"),
    **dict.fromkeys(
        ["CAT", "HON", "UPS", "GE", "DE", "LMT", "RTX", "UNP", "MMM", "ETN"],
        "industrials"),
    **dict.fromkeys(
        ["LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE"],
        "materials"),
    **dict.fromkeys(
        ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL"],
        "utilities"),
    **dict.fromkeys(
        ["PLD", "AMT", "EQIX", "SPG", "O", "PSA", "CCI", "WELL"],
        "real_estate"),
}

UNIVERSE: list[str] = [
    # Information Technology (10)
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "TXN",
    # Communication Services (8)
    "GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA", "TMUS",
    # Consumer Discretionary (10)
    "AMZN", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "GM", "YUM",
    # Consumer Staples (9)
    "WMT", "PG", "KO", "PEP", "COST", "MDLZ", "CL", "GIS", "MO",
    # Financials (10)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK",
    # Health Care (10)
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "MDT",
    # Energy (7)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "OXY",
    # Industrials (10)
    "CAT", "HON", "UPS", "GE", "DE", "LMT", "RTX", "UNP", "MMM", "ETN",
    # Materials (7)
    "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE",
    # Utilities (7)
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL",
    # Real Estate (8)
    "PLD", "AMT", "EQIX", "SPG", "O", "PSA", "CCI", "WELL",
]
