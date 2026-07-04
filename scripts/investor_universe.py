# V4 Investor Stock Universe — Sector-Balanced Large-Caps (all 11 GICS sectors)
# Deduplicated definition used by data miner, feature pipeline, and orchestrator.
#
# 2026-07-03 widening: 46 → 96 names. Adds Real Estate (previously absent),
# roughly doubles Utilities/Materials/Energy, caps Financials at ~10% of the
# basket. Every name has a full 5-year daily price history (no recent IPOs /
# spin-offs) so walk-forward folds stay balanced.

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
