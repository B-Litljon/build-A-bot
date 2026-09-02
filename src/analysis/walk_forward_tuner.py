"""
walk_forward_tuner.py -- walk-forward parameter search for trading strategies.

Resists the naive "fit until perfect" overfitting trap by evaluating candidate
parameters strictly out-of-sample (OOS) across rolling or expanding chronological folds.

Glossary:
    param_grid -- dictionary of param_name -> list of candidate values.
    generate_folds -- builds the (train_start, train_end, val_start, val_end)
        index windows; expanding train, fixed-width validation.
    FoldMetrics -- one fold's in-sample and out-of-sample result, side by side.
        The GAP between them is the overfitting signal, not either alone.
    TunedParamResult -- aggregated out-of-sample performance for one parameter
        combination.
    pooled_oos_cp_lb -- Clopper-Pearson 95% one-sided lower bound on the pooled
        out-of-sample win rate. The evidential number; the point estimate is not.
    breakeven_win_rate -- the win rate this bracket shape needs just to pay for
        itself, 1 / (1 + tp_mult/sl_mult). 33.3% at the live 2:1.
    robust -- the promotion-style verdict: positive OOS expectancy, >= 30 OOS
        trades, >= 50% of folds positive, AND the lower bound clearing
        break-even. All four, or it is not robust.
    val_prefix -- warmup bars of prior history prepended to each validation
        slice so the fold starts warm; without it every fold silently discarded
        its first warmup_period bars and reset trailing state live never resets.
    tune_strategy -- main entry point to perform walk-forward parameter search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type
import numpy as np
import polars as pl

from analysis.behavior_matrix import DEFAULT_TOLL_R, _profit_factor
from analysis.strategy_backtester import run_backtest, _clopper_pearson_lower_bound
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    fold_idx: int
    is_trades: int
    is_net_ev: float
    is_pf: float
    oos_trades: int
    oos_net_ev: float
    oos_pf: float


@dataclass
class TunedParamResult:
    params: Dict[str, Any]
    total_oos_trades: int
    pooled_oos_win_rate: float
    pooled_oos_net_ev: float
    pooled_oos_pf: float
    positive_folds_ratio: float
    fold_metrics: List[FoldMetrics] = field(default_factory=list)
    # Clopper-Pearson 95% one-sided lower bound on the pooled OOS win rate, and
    # the win rate a bracket of this shape needs just to break even.
    pooled_oos_cp_lb: float = 0.0
    breakeven_win_rate: float = 0.0

    @property
    def robust(self) -> bool:
        """
        True only when the parameter set clears every bar, including the
        evidential one.

        The house style is that a point estimate is not evidence -- the
        promotion gate leans on Clopper-Pearson bounds precisely because an
        audit once found the same config passing at PF 1.444 and failing at
        0.982 an hour apart. This flag previously computed a CP bound and then
        ignored it, so a lucky 31-trade fold could read as robust. It no longer
        can: the lower bound on the win rate must clear the bracket's own
        break-even win rate, 1 / (1 + payoff).
        """
        return (
            self.pooled_oos_net_ev > 0.0
            and self.total_oos_trades >= 30
            and self.positive_folds_ratio >= 0.5
            and self.pooled_oos_cp_lb > self.breakeven_win_rate
        )


def generate_folds(n_bars: int, n_folds: int = 3) -> List[tuple[int, int, int, int]]:
    """
    Generate expanding walk-forward train and validation index ranges.
    
    Returns list of (train_start, train_end, val_start, val_end).
    """
    # Reserve first 20% for initial train
    step = n_bars // (n_folds + 2)
    folds = []
    for k in range(n_folds):
        train_start = 0
        train_end = step * (k + 2)
        val_start = train_end
        val_end = min(n_bars, train_end + step)
        folds.append((train_start, train_end, val_start, val_end))
    return folds


def tune_strategy(
    strategy_cls: Type[BaseStrategy],
    param_grid: Dict[str, Sequence[Any]],
    df: pl.DataFrame,
    *,
    n_folds: int = 3,
    sl_mult: float = 2.0,
    tp_mult: float = 4.0,
    toll_r: float = DEFAULT_TOLL_R,
    min_trades: int = 30,
) -> List[TunedParamResult]:
    """
    Perform expanding walk-forward parameter search for strategy_cls over df.
    """
    n_bars = len(df)
    folds = generate_folds(n_bars, n_folds=n_folds)

    keys = list(param_grid.keys())
    value_combinations = list(itertools.product(*param_grid.values()))

    logger.info(
        "Tuning %s: %d parameter combinations across %d walk-forward folds (%d total bars)",
        strategy_cls.__name__,
        len(value_combinations),
        n_folds,
        n_bars,
    )

    results: List[TunedParamResult] = []

    for vals in value_combinations:
        param_dict = dict(zip(keys, vals))
        # Warmup is a property of the parameter set (a 200-period SMA needs more
        # history than a 20-period one), so it is resolved per combination.
        warmup_bars = int(getattr(strategy_cls(**param_dict), "warmup_period", 60))

        fold_metrics_list: List[FoldMetrics] = []
        all_oos_trades = []

        for fold_idx, (tr_start, tr_end, val_start, val_end) in enumerate(folds):
            train_df = df.slice(tr_start, tr_end - tr_start)

            # The validation slice is PREFIXED with warmup bars of prior history
            # (2026-09-02). run_backtest skips its first `warmup` bars, so a
            # slice that began exactly at val_start spent the whole warmup
            # window unable to trade -- silently discarding the first
            # warmup_period bars of every fold and resetting the trailing regime
            # state that live never resets. Prefixing restores live continuity;
            # scoring still starts precisely at val_start because the prefix is
            # exactly consumed by the warmup.
            val_prefix = min(warmup_bars, val_start)
            val_df = df.slice(val_start - val_prefix, (val_end - val_start) + val_prefix)

            # A FRESH instance per fold. Today's strategies are stateless so
            # this is belt-and-braces, but the moment one carries fitted state a
            # shared instance would leak fold k's fit into fold k+1 invisibly.
            strat_instance = strategy_cls(**param_dict)

            # In-sample backtest
            is_rep = run_backtest(
                strat_instance,
                train_df,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
                toll_r=toll_r,
            )

            # Out-of-sample backtest
            oos_rep = run_backtest(
                strat_instance,
                val_df,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
                toll_r=toll_r,
            )

            fold_metrics_list.append(
                FoldMetrics(
                    fold_idx=fold_idx + 1,
                    is_trades=is_rep.total_trades,
                    is_net_ev=is_rep.net_ev_r,
                    is_pf=is_rep.profit_factor_net,
                    oos_trades=oos_rep.total_trades,
                    oos_net_ev=oos_rep.net_ev_r,
                    oos_pf=oos_rep.profit_factor_net,
                )
            )
            all_oos_trades.extend(oos_rep.trades)

        if not all_oos_trades:
            continue

        oos_wins = sum(t.macro_win for t in all_oos_trades)
        total_oos = len(all_oos_trades)
        pooled_wr = oos_wins / total_oos
        oos_net_rs = np.array([t.net_r for t in all_oos_trades])
        pooled_net_ev = float(np.mean(oos_net_rs))
        pooled_pf = _profit_factor(oos_net_rs)

        pos_folds = sum(1 for m in fold_metrics_list if m.oos_net_ev > 0)
        pos_ratio = pos_folds / len(fold_metrics_list) if fold_metrics_list else 0.0

        # Evidential bar: the win rate's lower bound against the win rate this
        # bracket needs to break even. payoff = tp/sl, so break-even = 1/(1+payoff).
        cp_lb = _clopper_pearson_lower_bound(oos_wins, total_oos)
        payoff = float(tp_mult) / float(sl_mult)
        breakeven = 1.0 / (1.0 + payoff)

        results.append(
            TunedParamResult(
                params=param_dict,
                total_oos_trades=total_oos,
                pooled_oos_win_rate=pooled_wr,
                pooled_oos_net_ev=pooled_net_ev,
                pooled_oos_pf=pooled_pf,
                positive_folds_ratio=pos_ratio,
                fold_metrics=fold_metrics_list,
                pooled_oos_cp_lb=cp_lb,
                breakeven_win_rate=breakeven,
            )
        )

    # Sort results by pooled OOS Net EV descending
    results.sort(key=lambda r: (r.robust, r.pooled_oos_net_ev), reverse=True)
    return results


def to_frame(results: Sequence[TunedParamResult]) -> pl.DataFrame:
    """Format tuning results as a Polars DataFrame."""
    if not results:
        return pl.DataFrame()
    rows = []
    for r in results:
        row = {f"param_{k}": v for k, v in r.params.items()}
        row.update({
            "oos_trades": r.total_oos_trades,
            "oos_win_rate": round(r.pooled_oos_win_rate, 4),
            "oos_net_ev_r": round(r.pooled_oos_net_ev, 4),
            "oos_pf": round(r.pooled_oos_pf, 4),
            "pos_folds_pct": round(r.positive_folds_ratio * 100, 1),
            "oos_cp_lb": round(r.pooled_oos_cp_lb, 4),
            "breakeven_wr": round(r.breakeven_win_rate, 4),
            "robust": r.robust,
        })
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward parameter tuner.")
    parser.add_argument("--strategy", type=str, default="sma_crossover", help="Strategy name")
    parser.add_argument("--data", type=str, required=True, help="Data parquet path")
    parser.add_argument("--folds", type=int, default=3, help="Number of walk-forward folds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from strategies.concrete_strategies import (
        BollingerBreakoutStrategy,
        DonchianBreakoutStrategy,
        MomentumStrategy,
        RSIMeanReversionStrategy,
        SMACrossoverStrategy,
    )

    strategy_map = {
        "sma_crossover": SMACrossoverStrategy,
        "rsi_mean_reversion": RSIMeanReversionStrategy,
        "bollinger_breakout": BollingerBreakoutStrategy,
        "donchian_breakout": DonchianBreakoutStrategy,
        "momentum": MomentumStrategy,
    }

    if args.strategy not in strategy_map:
        raise ValueError(f"Unknown strategy: {args.strategy}. Choose from {list(strategy_map.keys())}")

    strat_cls = strategy_map[args.strategy]
    df = pl.read_parquet(args.data)

    if args.strategy == "sma_crossover":
        grid = {"fast_period": [10, 20], "slow_period": [40, 50]}
    elif args.strategy == "rsi_mean_reversion":
        grid = {"rsi_period": [10, 14], "oversold": [25.0, 30.0], "overbought": [70.0, 75.0]}
    elif args.strategy == "bollinger_breakout":
        grid = {"bb_period": [15, 20], "bb_std": [1.5, 2.0]}
    elif args.strategy == "donchian_breakout":
        grid = {"channel_period": [15, 20, 25]}
    else:  # momentum
        grid = {"fast_period": [10, 12], "slow_period": [20, 26]}

    results = tune_strategy(strat_cls, grid, df, n_folds=args.folds)
    frame = to_frame(results)

    print()
    print("=" * 80)
    print("WALK-FORWARD TUNING RESULTS FOR " + strat_cls.__name__ + ":")
    print("=" * 80)
    print(frame)

    if results and results[0].robust:
        print("\nRecommended robust parameters:", results[0].params)
    elif results:
        print("\nBest parameters (below robustness threshold):", results[0].params)
    else:
        print("\nNo valid trades produced.")


if __name__ == "__main__":
    main()
