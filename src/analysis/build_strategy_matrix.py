"""
build_strategy_matrix.py -- scores the strategy library per market behavior and
emits a routing table.

THIS IS THE ONLY LEGITIMATE SOURCE OF A ROUTING TABLE. The tables in config/
named ``*.example.json`` are hand-authored templates with no empirical basis.

1. Fetches or loads bars for a BASKET of instruments (pooling matters: one
   symbol cannot fill nine behavior cells at n >= 30).
2. Tags every bar causally with behavior_tagger labels.
3. Backtests each library strategy through the LIVE gates, with measured
   per-instrument spread costs.
4. Scores each (behavior x strategy) cell from the ledger's own realised R.
5. Emits the matrix and a routing table in which a regime earns a strategy only
   on evidence, and stands down otherwise.

Glossary:
    prepare_tagged_frame -- adds natr_14 / ppo / behavior_label to a bar frame.
    run_strategy_matrix -- the whole sweep; returns (matrix, routing_table).
    score_cells_from_ledger -- per-cell stats computed from the ledger's real
        per-trade net_r, NOT from a binary win flag times a flat toll. With a
        per-instrument toll, net_ev_r stops being an affine transform of
        win_rate and the cell carries two independent numbers plus n.
    MIN_CELL_TRADES -- 30, the house floor. A 12-trade cell will happily show a
        profit factor of 3.0 by luck.
    require_significant -- a cell must also have a bootstrap CI on net
        expectancy that excludes zero before it can win a regime.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import polars as pl
import talib

from analysis.behavior_matrix import (
    BOOTSTRAP_N,
    MIN_CELL_TRADES,
    Candidate,
    CellStats,
    DEFAULT_TOLL_R,
    _bootstrap_ci,
    _profit_factor,
    recommend,
    score_ledger,
    tag_frame,
    to_frame,
)
from analysis.strategy_backtester import run_backtest
from strategies.concrete_strategies import (
    BollingerBreakoutStrategy,
    DonchianBreakoutStrategy,
    MomentumStrategy,
    RSIMeanReversionStrategy,
    SMACrossoverStrategy,
)

logger = logging.getLogger(__name__)


def prepare_tagged_frame(df: pl.DataFrame, symbol: str = "ASSET") -> pl.DataFrame:
    """Ensure symbol, natr_14, ppo exist, then tag frame with behavior labels."""
    if "symbol" not in df.columns:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

    close = df["close"].to_numpy().astype(float)
    high = df["high"].to_numpy().astype(float)
    low = df["low"].to_numpy().astype(float)

    natr = talib.NATR(high, low, close, timeperiod=14)
    ppo = talib.PPO(close, fastperiod=12, slowperiod=26, matype=talib.MA_Type.SMA)

    df = df.with_columns(
        pl.Series("natr_14", natr),
        pl.Series("ppo", ppo),
    )
    # tag_frame adds behavior_label column
    return tag_frame(df)


def score_cells_from_ledger(
    ledger: pl.DataFrame,
    candidate_name: str,
    *,
    min_cell_trades: int = MIN_CELL_TRADES,
    seed: int = 0,
) -> List[CellStats]:
    """
    Per-behavior stats from the ledger's OWN realised R.

    behavior_matrix.score_ledger re-derives R from a binary win flag times a
    flat toll, which makes every metric in a cell an affine transform of the win
    rate -- one number wearing six hats. The backtester now books realised R per
    trade with a per-instrument cost, so scoring from that column gives the cell
    genuinely independent information.

    ``cold`` rows are dropped, never pooled: an unwarmed window is an absence of
    evidence, not a regime.
    """
    from ml.regimes.behavior_tagger import LABEL_COLD

    out: List[CellStats] = []
    warm = ledger.filter(
        (pl.col("behavior_label") != LABEL_COLD)
        & (pl.col("behavior_label") != "unknown")
    )
    for behavior in sorted(warm["behavior_label"].unique().to_list()):
        cell = warm.filter(pl.col("behavior_label") == behavior)
        n = cell.height
        if n == 0:
            continue
        net = cell["net_r"].to_numpy().astype(float)
        gross = cell["gross_r"].to_numpy().astype(float)
        wins = int(cell["macro_win"].sum())
        lo, hi = _bootstrap_ci(net, n_boot=BOOTSTRAP_N, seed=seed)
        out.append(
            CellStats(
                behavior=behavior,
                candidate=candidate_name,
                n=n,
                win_rate=wins / n,
                gross_ev_r=float(np.mean(gross)),
                net_ev_r=float(np.mean(net)),
                profit_factor_gross=_profit_factor(gross),
                profit_factor_net=_profit_factor(net),
                ci_low=lo,
                ci_high=hi,
                informative=n >= min_cell_trades,
            )
        )
    return out


def run_strategy_matrix(
    frames: Dict[str, pl.DataFrame],
    *,
    sl_mult: float = 2.0,
    tp_mult: float = 4.0,
    toll_r: float = DEFAULT_TOLL_R,
    min_cell_trades: int = MIN_CELL_TRADES,
    apply_gates: bool = True,
    spread_alphas: Optional[Dict[str, float]] = None,
    require_significant: bool = True,
    max_hold_bars: int = 45,
) -> tuple[pl.DataFrame, Dict[str, Optional[str]], Dict[str, Dict[str, int]]]:
    """
    Evaluate the library across a BASKET and return
    (matrix, routing_table, gate_funnel).

    Pooling across instruments is not optional. Nine behavior cells x five
    strategies needs far more trades than one symbol produces, and a cell built
    from a single pair measures that pair, not that regime.
    """
    risk_manager = None
    if apply_gates:
        from execution.risk_manager import RiskManager, RiskProfile

        risk_manager = RiskManager(RiskProfile.for_asset_class("forex"))

    strategies = [
        SMACrossoverStrategy(),
        RSIMeanReversionStrategy(),
        BollingerBreakoutStrategy(),
        DonchianBreakoutStrategy(),
        MomentumStrategy(),
    ]

    all_cells: List[CellStats] = []
    funnel: Dict[str, Dict[str, int]] = {}

    tagged: Dict[str, pl.DataFrame] = {
        sym: prepare_tagged_frame(df, symbol=sym) for sym, df in frames.items()
    }

    for strat in strategies:
        name = strat.__class__.__name__
        ledgers: List[pl.DataFrame] = []
        strat_funnel: Dict[str, int] = {}

        for sym, df in tagged.items():
            logger.info("Backtesting %s on %s (%d bars) ...", name, sym, len(df))
            report = run_backtest(
                strat,
                df,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
                toll_r=toll_r,
                max_hold_bars=max_hold_bars,
                symbol_override=sym,
                risk_manager=risk_manager,
                spread_alphas=spread_alphas,
            )
            for gate, count in report.gate_rejections.items():
                strat_funnel[gate] = strat_funnel.get(gate, 0) + count
            led = report.to_ledger()
            if not led.is_empty():
                ledgers.append(led)

        funnel[name] = strat_funnel
        if not ledgers:
            logger.warning("Strategy %s generated 0 trades across the basket", name)
            continue

        pooled = pl.concat(ledgers, how="vertical_relaxed")
        all_cells.extend(
            score_cells_from_ledger(pooled, name, min_cell_trades=min_cell_trades)
        )

    matrix_frame = to_frame(all_cells)

    # A regime earns a strategy only on evidence. `recommend` already requires
    # informative + positive expectancy; we additionally require the bootstrap
    # interval to exclude zero, so a cell cannot win on a point estimate alone.
    eligible = [c for c in all_cells if c.significant] if require_significant else all_cells
    routing_table = recommend(eligible, require_informative=True)

    # Any behavior seen in the data but winning nothing must still appear, as an
    # explicit stand-down rather than a missing key.
    for c in all_cells:
        routing_table.setdefault(c.behavior, None)

    return matrix_frame, routing_table, funnel


DEFAULT_BASKET = "AUD_JPY,EUR_JPY,GBP_JPY,NZD_JPY,GBP_AUD,GBP_NZD"
"""The six tradeable crosses. XAU_USD / XAG_USD are deliberately absent: they
are broker-dead (a live XAG order was rejected INSTRUMENT_NOT_TRADEABLE on
2026-07-14) yet made up 38-60% of prior model picks, which is exactly how
earlier scores got distorted."""


def load_basket(
    symbols: Sequence[str],
    *,
    days_back: int,
    granularity: int,
    cache_dir: Path,
) -> Dict[str, pl.DataFrame]:
    """
    Load bars per symbol, preferring a local parquet and falling back to OANDA.

    Fetched frames are cached so a re-run costs nothing and the measurement is
    reproducible against the exact bars that produced it.
    """
    from datetime import datetime, timedelta, timezone

    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, pl.DataFrame] = {}

    provider = None
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    for sym in symbols:
        cached = cache_dir / f"{sym}_M{granularity}.parquet"
        legacy = Path("data/raw") / f"{sym}_M{granularity}.parquet"
        if cached.exists():
            out[sym] = pl.read_parquet(cached)
            logger.info("%s: %d bars from cache", sym, len(out[sym]))
            continue
        if legacy.exists():
            out[sym] = pl.read_parquet(legacy)
            logger.info("%s: %d bars from data/raw", sym, len(out[sym]))
            continue

        if provider is None:
            from data.factory import get_market_provider

            provider = get_market_provider()
        logger.info("%s: fetching %d days from the provider ...", sym, days_back)
        df = provider.get_historical_bars(sym, granularity, start, end)
        if df.is_empty():
            logger.warning("%s: provider returned no bars, skipping", sym)
            continue
        df.write_parquet(cached)
        out[sym] = df
        logger.info("%s: %d bars fetched and cached", sym, len(df))

    # Trim every frame to the window they ALL cover. A legacy parquet in
    # data/raw can span a different period from a fresh fetch, and pooling
    # mismatched windows silently mixes market eras -- a cell would then
    # describe "GBP_JPY in 2023" and "EUR_JPY in 2025" as if they were the same
    # regime. Align first, measure second.
    if len(out) > 1 and all("timestamp" in d.columns for d in out.values()):
        lo = max(d["timestamp"].min() for d in out.values())
        hi = min(d["timestamp"].max() for d in out.values())
        trimmed = {}
        for sym, d in out.items():
            before = len(d)
            d = d.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") <= hi))
            if before != len(d):
                logger.info(
                    "%s: trimmed %d -> %d bars to the common window", sym, before, len(d)
                )
            trimmed[sym] = d
        logger.info("Common window: %s -> %s", lo, hi)
        out = trimmed

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the strategy regime matrix and a routing table."
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Single parquet file (legacy single-symbol mode)")
    parser.add_argument("--symbols", type=str, default=DEFAULT_BASKET,
                        help=f"Comma-separated basket (default: {DEFAULT_BASKET})")
    parser.add_argument("--symbol", type=str, default="ASSET",
                        help="Symbol name when using --data")
    parser.add_argument("--days-back", type=int, default=730)
    parser.add_argument("--granularity", type=int, default=15)
    parser.add_argument("--cache-dir", type=str, default="analysis_cache/strategy_matrix")
    parser.add_argument("--output", type=str, default="config/regime_routing_forex.json")
    parser.add_argument("--matrix-out", type=str, default="logs/strategy_matrix.csv")
    parser.add_argument("--toll-r", type=float, default=DEFAULT_TOLL_R,
                        help="Flat fallback toll; ignored where spread alphas apply")
    parser.add_argument("--min-trades", type=int, default=MIN_CELL_TRADES)
    parser.add_argument("--sl-mult", type=float, default=2.0,
                        help="Stop as a multiple of ATR (live forex: 2.0)")
    parser.add_argument("--tp-mult", type=float, default=4.0,
                        help="Target as a multiple of ATR (live forex: 4.0)")
    parser.add_argument("--max-hold", type=int, default=45,
                        help="Bars before a trade times out. Widening the target "
                             "without widening this just converts wins to timeouts.")
    parser.add_argument("--no-gates", action="store_true",
                        help="Skip the live chop filter. Measures a population "
                             "the bot would never trade; diagnostics only.")
    parser.add_argument("--no-significance", action="store_true",
                        help="Let a cell win on a point estimate. Not advised.")
    parser.add_argument("--spread-table", type=str,
                        default="config/spread_alphas_m15.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.data:
        path = Path(args.data)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        frames = {args.symbol: pl.read_parquet(path)}
        logger.info("Loaded %d bars from %s", len(frames[args.symbol]), path)
    else:
        frames = load_basket(
            [s.strip() for s in args.symbols.split(",") if s.strip()],
            days_back=args.days_back,
            granularity=args.granularity,
            cache_dir=Path(args.cache_dir),
        )
    if not frames:
        raise SystemExit("No data loaded; nothing to score.")

    spread_alphas = None
    table_path = Path(args.spread_table)
    if table_path.exists():
        spread_alphas = json.load(open(table_path)).get("alphas")
        logger.info("Loaded %d spread alphas from %s", len(spread_alphas or {}), table_path)
    else:
        logger.warning("No spread table at %s -- falling back to a flat toll", table_path)

    matrix_df, routing_table, funnel = run_strategy_matrix(
        frames,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        max_hold_bars=args.max_hold,
        toll_r=args.toll_r,
        min_cell_trades=args.min_trades,
        apply_gates=not args.no_gates,
        spread_alphas=spread_alphas,
        require_significant=not args.no_significance,
    )

    print("\n" + "=" * 78)
    print("STRATEGY x BEHAVIOR MATRIX")
    print("=" * 78)
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(matrix_df)

    print("\n" + "=" * 78)
    print("GATE FUNNEL (signals proposed but vetoed by the live filter)")
    print("=" * 78)
    for strat, gates in sorted(funnel.items()):
        total = sum(gates.values())
        detail = ", ".join(f"{g}={c}" for g, c in sorted(gates.items())) or "none"
        print(f"  {strat:<28} vetoed {total:>6}   ({detail})")

    print("\n" + "=" * 78)
    print("ROUTING TABLE")
    print("=" * 78)
    routed = 0
    for regime, strat in sorted(routing_table.items()):
        if strat is None:
            print(f"  {regime:<18} -> stand down")
        else:
            routed += 1
            print(f"  {regime:<18} -> {strat}")
    print(f"\n  {routed}/{len(routing_table)} regimes earned a strategy.")
    if routed == 0:
        print("  Nothing cleared the evidence bar. A table of all stand-downs is")
        print("  a legitimate result, and it means the router has nothing to do.")

    Path(args.matrix_out).parent.mkdir(parents=True, exist_ok=True)
    matrix_df.write_csv(args.matrix_out)
    logger.info("Wrote matrix to %s", args.matrix_out)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_generated_by": "src/analysis/build_strategy_matrix.py",
        "_generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "_basket": sorted(frames),
        "_gates_applied": not args.no_gates,
        "_significance_required": not args.no_significance,
        "_min_cell_trades": args.min_trades,
        "_bracket": f"{args.sl_mult}x/{args.tp_mult}x/{args.max_hold}bar",
        **routing_table,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved routing table to %s", out_path)


if __name__ == "__main__":
    main()
