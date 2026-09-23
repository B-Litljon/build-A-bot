"""Bar loading for the lab -- cached parquet first, provider second.

Thin wrapper around analysis.build_strategy_matrix.load_basket, which already
implements the cache contract (per-symbol parquet under
analysis_cache/strategy_matrix, provider fallback, common-window trim). The lab
adds nothing to the format; it only pins the defaults and exposes ``refresh``.

Glossary:
    DEFAULT_CACHE_DIR -- analysis_cache/strategy_matrix, the same cache
        scripts/evaluate_barriers.py and build_strategy_matrix.py use. The M15
        fiat basket there spans ~730 days and needs no network.
    load_bars -- {symbol: bars} for a spec's window, trimmed to the common
        span across symbols. Raises when a symbol yields nothing, rather than
        measuring a silently shrunken basket.
    resolve_alpha_table -- the spec's per-instrument cost table as a plain
        {symbol: alpha} dict, or None when the spec's switch is off.
    refresh -- delete the spec's symbols from the cache first, forcing a
        provider fetch. Destructive only for the named symbols' cache files,
        which the loader recreates; never touches models/ or logs/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import polars as pl

DEFAULT_CACHE_DIR = Path("analysis_cache/strategy_matrix")


def load_bars(
    spec,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> Dict[str, pl.DataFrame]:
    """Load the spec's basket, preferring local parquet over the provider."""
    from analysis.build_strategy_matrix import load_basket

    cache_dir = Path(cache_dir)
    if refresh:
        for sym in spec.symbols:
            cached = cache_dir / f"{sym}_M{spec.granularity}.parquet"
            if cached.exists():
                cached.unlink()

    bars = load_basket(
        list(spec.symbols),
        days_back=spec.days_back,
        granularity=spec.granularity,
        cache_dir=cache_dir,
    )
    missing = [s for s in spec.symbols if s not in bars or bars[s].is_empty()]
    if missing:
        raise ValueError(
            f"no bars loaded for {missing}; refusing to build a frame on a "
            "partial basket"
        )
    return bars


def resolve_alpha_table(spec) -> Optional[Dict[str, float]]:
    """The spec's cost table, or None when use_spread_table is off."""
    return spec.alpha_table()
