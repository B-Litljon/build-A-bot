"""
strategy_backtester.py -- honest, walk-forward, realistic-cost backtesting harness for BaseStrategy.

Simulates sequential bar-by-bar execution without lookahead:
1. Evaluates generate_signals() against trailing historical slices.
2. Sizes the bracket -- through the live RiskManager and its gates when one is
   supplied, otherwise from this module's sl_mult / tp_mult defaults.
3. Walks subsequent bars for a gap, an intrabar SL/TP breach, or a timeout.
4. Books the REALISED move in R and deducts the spread toll.
5. Emits a trade ledger compatible with behavior_matrix.score_ledger().

Two conventions worth knowing before reading a number out of this file:

* ``macro_win`` is bracket resolution -- 1 only when the target was reached.
  A timeout is 0 regardless of where price ended, matching
  ``retrainer._compute_devil_targets_atr``. So ``win_rate`` answers "how often
  did the target resolve first?".
* ``gross_r`` / ``net_r`` are the REALISED move over the stop distance. A
  timeout that drifted a little way up pays that little way, not the bracket's
  nominal payoff. The two columns answer different questions and will not agree
  on a trade that timed out; that is intended.

Glossary:
    BacktestTrade -- execution record of one simulated trade, including the fill
        price actually used and why the trade ended.
    BacktestReport -- aggregate stats (win rate, PF gross/net, net EV in R, max
        drawdown in R, Clopper-Pearson win-rate lower bound, gate funnel).
    run_backtest -- main entry point; evaluates a BaseStrategy over a Polars frame.
    gate_rejections -- proposed-but-vetoed signal counts per live gate.
    trade_toll -- per-trade cost in R; measured per instrument when spread
        alphas are supplied, else the flat toll_r.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import numpy as np
import polars as pl
from scipy.stats import beta

from strategies.base import BaseStrategy, Signal
from analysis.behavior_matrix import DEFAULT_TOLL_R, _profit_factor

if TYPE_CHECKING:  # pragma: no cover - annotation only, never imported at runtime
    from execution.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    entry_idx: int
    entry_timestamp: Any
    symbol: str
    direction: str
    entry_price: float
    raw_sl_distance: float
    sl_price: float
    tp_price: float
    exit_idx: int
    exit_timestamp: Any
    exit_price: float
    exit_reason: str  # "tp" | "sl" | "tp_gap" | "sl_gap" | "timeout"
    macro_win: int   # 1 if win, 0 if loss
    gross_r: float
    net_r: float
    behavior_label: Optional[str] = None


@dataclass
class BacktestReport:
    strategy_name: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    gross_ev_r: float
    net_ev_r: float
    profit_factor_gross: float
    profit_factor_net: float
    max_drawdown_r: float
    clopper_pearson_lb: float
    trades: List[BacktestTrade] = field(default_factory=list)
    # Signals proposed but vetoed by the live chop filter, keyed by gate
    # (GATE_SPREAD / GATE_REGIME / GATE_TIME / GATE_STATIC). Empty when no
    # RiskManager was supplied. The funnel is a result in its own right: a cell
    # whose picks are mostly gate-vetoed is not reachable live.
    gate_rejections: Dict[str, int] = field(default_factory=dict)

    def to_ledger(self) -> pl.DataFrame:
        """Convert trades into a standardized ledger DataFrame."""
        if not self.trades:
            return pl.DataFrame(
                schema={
                    "entry_idx": pl.Int64,
                    "timestamp": pl.Utf8,
                    "symbol": pl.Utf8,
                    "direction": pl.Utf8,
                    "close": pl.Float64,
                    "raw_sl_distance": pl.Float64,
                    "sl_price": pl.Float64,
                    "tp_price": pl.Float64,
                    "exit_price": pl.Float64,
                    "exit_reason": pl.Utf8,
                    "macro_win": pl.Int64,
                    "gross_r": pl.Float64,
                    "net_r": pl.Float64,
                    "behavior_label": pl.Utf8,
                }
            )
        return pl.DataFrame(
            [
                {
                    "entry_idx": t.entry_idx,
                    "timestamp": str(t.entry_timestamp),
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "close": t.entry_price,
                    "raw_sl_distance": t.raw_sl_distance,
                    "sl_price": t.sl_price,
                    "tp_price": t.tp_price,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "macro_win": t.macro_win,
                    "gross_r": t.gross_r,
                    "net_r": t.net_r,
                    "behavior_label": t.behavior_label or "unknown",
                }
                for t in self.trades
            ]
        )


def _clopper_pearson_lower_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact Clopper-Pearson 95% one-sided lower bound on binomial proportion."""
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def run_backtest(
    strategy: BaseStrategy,
    df: pl.DataFrame,
    *,
    sl_mult: float = 2.0,
    tp_mult: float = 4.0,
    toll_r: float = DEFAULT_TOLL_R,
    max_hold_bars: int = 45,
    cooldown_bars: int = 1,
    symbol_override: Optional[str] = None,
    risk_manager: Optional["RiskManager"] = None,
    spread_alphas: Optional[Dict[str, float]] = None,
    default_alpha: float = 0.15,
) -> BacktestReport:
    """
    Run an honest bar-by-bar backtest of strategy over df.

    df must have columns: open, high, low, close.
    Optional columns: timestamp, symbol, behavior_label, natr_14.

    Args:
        risk_manager: when supplied, every proposed signal is put through the
            LIVE chop filter (Gate C time blackout, Gate B regime percentile,
            Gate A spread/cost) via ``calculate_bracket``, and vetoed signals are
            counted rather than traded. Omitting it measures a population the
            live bot would never take -- Gate B alone vetoes the bottom 20% of
            the volatility window, which is ~60% of the tagger's ``*_low`` band.
        spread_alphas: per-instrument spread alphas (the ``alphas`` block of
            ``config/spread_alphas_m15.json``). When supplied the per-trade toll
            is measured per symbol instead of the flat ``toll_r``; the shipped
            alphas span 0.072 to 0.903, a 12.6x range that a single constant
            cannot represent at either end.
        default_alpha: alpha for symbols absent from the table.

    ``risk_manager`` is passed ``raw_atr``, NOT a stop distance -- it applies
    ``sl_atr_multiplier`` itself. See risk_manager.calculate_bracket.
    """
    strategy.validate_input(df)

    n_bars = len(df)
    warmup = getattr(strategy, "warmup_period", 60)
    if n_bars <= warmup:
        return BacktestReport(
            strategy_name=strategy.name,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            gross_ev_r=0.0,
            net_ev_r=0.0,
            profit_factor_gross=0.0,
            profit_factor_net=0.0,
            max_drawdown_r=0.0,
            clopper_pearson_lb=0.0,
        )

    # Pre-extract numpy arrays for fast evaluation
    open_ = df["open"].to_numpy().astype(float)
    high = df["high"].to_numpy().astype(float)
    low = df["low"].to_numpy().astype(float)
    close = df["close"].to_numpy().astype(float)
    # Trailing NATR feeds Gate B's percentile rank. Absent -> Gate B is bypassed,
    # exactly as the live RiskManager bypasses it on a cold window.
    natr = (
        df["natr_14"].to_numpy().astype(float)
        if "natr_14" in df.columns
        else None
    )
    timestamps = (
        df["timestamp"].to_list()
        if "timestamp" in df.columns
        else list(range(n_bars))
    )
    symbol = (
        symbol_override
        or (str(df["symbol"][0]) if "symbol" in df.columns else "UNKNOWN")
    )
    behavior_labels = (
        df["behavior_label"].to_list()
        if "behavior_label" in df.columns
        else [None] * n_bars
    )

    trades: List[BacktestTrade] = []
    gate_rejections: Dict[str, int] = {}
    use_real_timestamps = "timestamp" in df.columns
    i = warmup

    # Sizing R-payoff ratio: tp_mult / sl_mult
    payoff_ratio = float(tp_mult) / float(sl_mult)

    while i < n_bars - 1:
        # Slice history up to bar i (lookahead-free)
        # Note: Polars slicing df[:i+1]
        # Bounded window covers warmup_period while keeping evaluation O(1) per bar
        win_size = max(warmup * 2, 400)
        start_idx = max(0, i - win_size + 1)
        window_df = df.slice(start_idx, i - start_idx + 1)

        signal: Optional[Signal] = strategy.generate_signals(window_df)
        if signal is None:
            i += 1
            continue

        entry_idx = i
        entry_price = signal.entry_price
        raw_sl = signal.raw_sl_distance
        direction = signal.direction.lower()

        # Bracket distances. With a RiskManager the LIVE gates decide, and the
        # multipliers/rounding/floors come from the served RiskProfile rather
        # than this function's defaults -- so the backtest brackets are the ones
        # the bot would actually place.
        if risk_manager is not None:
            regime_window = (
                natr[max(0, entry_idx - risk_manager.profile.regime_window + 1) : entry_idx + 1]
                if natr is not None
                else None
            )
            bracket = risk_manager.calculate_bracket(
                entry_price,
                raw_sl,                      # raw ATR — the RiskManager multiplies
                symbol=symbol,
                spread=None,
                spread_fresh=False,          # offline: use the volatility proxy
                regime_series=regime_window,
                timestamp=timestamps[entry_idx] if use_real_timestamps else None,
            )
            if bracket is None:
                gate = getattr(risk_manager, "last_veto_gate", "unknown")
                gate_rejections[gate] = gate_rejections.get(gate, 0) + 1
                i += 1
                continue
            sl_dist, tp_dist = bracket
        else:
            sl_dist = raw_sl * sl_mult
            tp_dist = raw_sl * tp_mult

        # Per-instrument toll when a measured spread table is supplied. Mirrors
        # Gate A's own proxy: alpha * median(trailing NATR) * price / 100.
        trade_toll = toll_r
        if spread_alphas is not None and sl_dist > 0:
            alpha = float(spread_alphas.get(symbol, default_alpha))
            if natr is not None:
                lo = max(0, entry_idx - (risk_manager.profile.regime_window if risk_manager else 260) + 1)
                baseline_natr = float(np.nanmedian(natr[lo : entry_idx + 1]))
            else:
                baseline_natr = float("nan")
            if np.isfinite(baseline_natr):
                spread_proxy = alpha * baseline_natr * entry_price / 100.0
                trade_toll = spread_proxy / sl_dist

        if direction == "long":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # Simulate subsequent bars.
        #
        # Fill realism (2026-09-02): a bar that OPENS beyond a bracket level did
        # not trade through it -- it gapped past it, and the fill is the open,
        # not the level. Filling at the level books a price that never traded
        # and flatters stop-outs. Checked before the intrabar test because a gap
        # resolves the bar before any within-bar path matters.
        exit_idx = min(entry_idx + max_hold_bars, n_bars - 1)
        exit_price = close[exit_idx]
        exit_reason = "timeout"
        macro_win = 0

        for j in range(entry_idx + 1, exit_idx + 1):
            bar_o = open_[j]
            bar_h = high[j]
            bar_l = low[j]

            if direction == "long":
                gap_sl = bar_o <= sl_price
                gap_tp = bar_o >= tp_price
                hit_sl = bar_l <= sl_price
                hit_tp = bar_h >= tp_price
            else:  # short
                gap_sl = bar_o >= sl_price
                gap_tp = bar_o <= tp_price
                hit_sl = bar_h >= sl_price
                hit_tp = bar_l <= tp_price

            # Gap resolution first. If the bar opened through BOTH levels the
            # stop is the conservative read, matching the intrabar convention.
            if gap_sl:
                exit_idx, exit_price, exit_reason, macro_win = j, bar_o, "sl_gap", 0
                break
            if gap_tp:
                exit_idx, exit_price, exit_reason, macro_win = j, bar_o, "tp_gap", 1
                break

            # Intrabar. Stop assumed hit first when a single bar spans both --
            # we cannot see the within-bar path, so we take the worse branch.
            if hit_sl:
                exit_idx, exit_price, exit_reason, macro_win = j, sl_price, "sl", 0
                break
            if hit_tp:
                exit_idx, exit_price, exit_reason, macro_win = j, tp_price, "tp", 1
                break

        # R is booked from the REALISED move, not from the bracket's nominal
        # payoff (2026-09-02).
        #
        # The prior implementation booked every exit at exactly +payoff or -1R
        # based on the sign of the move, so a timeout one pip above entry paid a
        # full +2R. Measured on GBP_JPY M15 that inflated win rate by 10-12
        # points across every strategy in the library. A trade that never
        # touched either level did not earn the bracket's payoff and must not be
        # paid it.
        #
        # Realised R is exact for clean level hits (a TP fill at tp_price is
        # precisely +payoff), naturally bounded inside (-1, +payoff) for a
        # timeout that touched nothing, and correctly worse than -1R when a gap
        # blew through the stop -- which is a real loss the old code hid.
        signed_move = (
            exit_price - entry_price if direction == "long" else entry_price - exit_price
        )
        gross_r_val = signed_move / sl_dist if sl_dist > 0 else 0.0
        net_r_val = gross_r_val - trade_toll

        # macro_win stays BINARY and bracket-resolution-based so the ledger
        # remains compatible with behavior_matrix.score_ledger(): 1 only when the
        # target was reached. A timeout is not a win at any realised R --
        # matching retrainer._compute_devil_targets_atr, which scores timeout 0.
        # win_rate therefore answers "how often did the target resolve first?"
        # while the R columns answer "what did it actually pay?".

        trade = BacktestTrade(
            entry_idx=entry_idx,
            entry_timestamp=timestamps[entry_idx],
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            raw_sl_distance=raw_sl,
            sl_price=sl_price,
            tp_price=tp_price,
            exit_idx=exit_idx,
            exit_timestamp=timestamps[exit_idx],
            exit_price=exit_price,
            exit_reason=exit_reason,
            macro_win=macro_win,
            gross_r=gross_r_val,
            net_r=net_r_val,
            behavior_label=behavior_labels[entry_idx],
        )
        trades.append(trade)

        # Advance pointer past trade exit + cooldown (no overlapping positions)
        i = exit_idx + cooldown_bars

    if not trades:
        return BacktestReport(
            strategy_name=strategy.name,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            gross_ev_r=0.0,
            net_ev_r=0.0,
            profit_factor_gross=0.0,
            profit_factor_net=0.0,
            max_drawdown_r=0.0,
            clopper_pearson_lb=0.0,
            gate_rejections=gate_rejections,
        )

    wins = sum(t.macro_win for t in trades)
    n_trades = len(trades)
    win_rate = wins / n_trades
    gross_rs = np.array([t.gross_r for t in trades])
    net_rs = np.array([t.net_r for t in trades])

    pf_gross = _profit_factor(gross_rs)
    pf_net = _profit_factor(net_rs)

    # Calculate Max Drawdown in R
    cum_r = np.cumsum(net_rs)
    running_max = np.maximum.accumulate(cum_r)
    drawdowns = running_max - cum_r
    max_dd = float(np.max(drawdowns)) if drawdowns.size else 0.0

    cp_lb = _clopper_pearson_lower_bound(wins, n_trades)

    return BacktestReport(
        strategy_name=strategy.name,
        total_trades=n_trades,
        wins=wins,
        losses=n_trades - wins,
        win_rate=win_rate,
        gross_ev_r=float(np.mean(gross_rs)),
        net_ev_r=float(np.mean(net_rs)),
        profit_factor_gross=pf_gross,
        profit_factor_net=pf_net,
        max_drawdown_r=max_dd,
        clopper_pearson_lb=cp_lb,
        trades=trades,
        gate_rejections=gate_rejections,
    )
