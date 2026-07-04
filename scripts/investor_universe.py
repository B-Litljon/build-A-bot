# V4 Investor Stock Universe - Sector-Balanced Large-Caps
# Deduplicated definition used by data miner, feature pipeline, and orchestrator.

UNIVERSE: list[str] = [
    # Tech
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO",
    # Communication
    "GOOGL", "META", "NFLX", "DIS", "T",
    # Consumer Discretionary
    "AMZN", "HD", "MCD", "NKE", "LOW",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "COST",
    # Financials
    "JPM", "BAC", "WFC", "GS", "V", "MA",
    # Health Care
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK",
    # Energy
    "XOM", "CVX", "COP",
    # Industrials
    "CAT", "HON", "UPS", "GE",
    # Materials
    "LIN", "SHW",
    # Utilities
    "NEE", "DUK",
]
