"""Model-aware, live-gated backtest of a lab candidate.

The gate answers "would this promote?"; this module answers "what would it have
paid?". ``LabModelStrategy`` replays the trained candidate models bar by bar
through the SAME ``analysis.strategy_backtester.run_backtest`` the strategy
library uses, with a live ``RiskManager`` attached, so:

    * Gate A (spread/cost), Gate B (regime percentile) and Gate C (time
      blackout) veto entries exactly as the live bot does, and vetoes are
      counted in ``gate_rejections``;
    * the bracket comes from the served RiskProfile (2.0x/4.0x/45 for forex);
    * the toll is measured per instrument when the spread table is active.

Two entry points share one replay body: ``run_model_backtest`` scores the
gate's freshly-trained fold models, ``run_artifact_backtest`` scores a served
artifact at its own pinned bars (``lab.artifact``).

The one alignment trap, pinned by tests: the strategy's probability arrays are
indexed by the frame the backtest walks. Pass the POST-CLEAN frame (the
FrameResult's own df) to the runner; if a caller strips rows between building
the frame and backtesting it, every bar reads another bar's score.

Glossary:
    LabModelStrategy -- a BaseStrategy wrapper around the frozen models:
        precomputes Angel and Devil probabilities once, then emits a long
        Signal on any bar both stages approve. raw_sl_distance is raw ATR in
        PRICE units (close * natr_14 / 100), exactly what MLStrategy emits —
        the RiskManager applies the ATR multipliers itself.
    _replay_models -- the shared per-symbol loop + pooling, so the gate and
        artifact paths cannot drift apart.
    run_model_backtest -- per-symbol backtests + a pooled summary, gate models.
    run_artifact_backtest -- the same replay for a served artifact (duck-typed
        to avoid an import cycle with lab.artifact).
    LabBacktestReport -- pooled net/gross EV in R, win rate, net PF, drawdown
        on a timestamp-sorted trade sequence, the live-gate veto funnel, and
        the per-symbol reports. ``toll_mode`` says whether measured alphas or a
        flat toll priced the trades.
    angel_threshold / devil_threshold -- the frozen bars to replay at: the
        gate's calibrated values, or the served ``threshold.json`` pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from analysis.behavior_matrix import _profit_factor
from analysis.strategy_backtester import BacktestReport, run_backtest
from strategies.base import BaseStrategy, Signal


class LabModelStrategy(BaseStrategy):
    """Replays a gate-trained Angel/Devil pair as a BaseStrategy."""

    def __init__(
        self,
        frame_df: pl.DataFrame,
        feature_cols: Tuple[str, ...],
        angel_model,
        devil_model,
        angel_threshold: float,
        devil_threshold: float,
        *,
        angel_feature_cols: Optional[Tuple[str, ...]] = None,
        devil_feature_cols: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        self.frame_df = frame_df
        self.feature_cols = tuple(feature_cols)
        # The fit-time schema when the estimator declares one (a served
        # artifact), else the frame's columns (gate models are fit on numpy
        # arrays and have no names). Getting this order wrong is a silent
        # per-column scramble, so artifact callers pass it explicitly.
        self.angel_feature_cols = tuple(angel_feature_cols or feature_cols)
        self.devil_feature_cols = tuple(
            devil_feature_cols or (*self.angel_feature_cols, "angel_prob")
        )
        self.angel_model = angel_model
        self.devil_model = devil_model
        self.angel_threshold = float(angel_threshold)
        self.devil_threshold = float(devil_threshold)
        # The frame is already cleaned/warm; every bar is scoreable.
        self.warmup_period = 0

        self._row_index: Dict[object, int] = {
            ts: i for i, ts in enumerate(frame_df["timestamp"].to_list())
        }
        X = frame_df.select(list(self.angel_feature_cols)).to_numpy()
        self._angel_probs = angel_model.predict_proba(X)[:, 1]
        meta = pl.DataFrame(X, schema=list(self.angel_feature_cols)).with_columns(
            pl.Series("angel_prob", self._angel_probs)
        )
        if self.devil_feature_cols != (*self.angel_feature_cols, "angel_prob"):
            meta = meta.select(list(self.devil_feature_cols))
        devil_proba = devil_model.predict_proba(meta)
        if devil_proba.shape[1] == 1:  # single-class Devil (tiny frames)
            only = int(devil_model.classes_[0])
            self._devil_probs = np.full(len(meta), 1.0 if only == 1 else 0.0)
        else:
            self._devil_probs = devil_proba[:, 1]

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        i = self._row_index.get(df["timestamp"][-1])
        if i is None:
            return None
        if self._angel_probs[i] < self.angel_threshold:
            return None
        if self._devil_probs[i] < self.devil_threshold:
            return None
        entry_price = float(df["close"][-1])
        natr = float(df["natr_14"][-1])
        if not np.isfinite(natr) or natr <= 0:
            return None
        raw_atr = entry_price * natr / 100.0
        return Signal(
            direction="long",
            entry_price=entry_price,
            raw_sl_distance=raw_atr,
            raw_tp_distance=raw_atr,
            metadata={
                "angel_prob": float(self._angel_probs[i]),
                "devil_prob": float(self._devil_probs[i]),
            },
        )


@dataclass
class LabBacktestReport:
    """Pooled lab backtest outcome over the whole basket."""

    total_trades: int
    wins: int
    win_rate: float
    gross_ev_r: float
    net_ev_r: float
    profit_factor_net: float
    max_drawdown_r: float
    gate_rejections: Dict[str, int] = field(default_factory=dict)
    per_symbol: Dict[str, BacktestReport] = field(default_factory=dict)
    toll_mode: str = "flat"

    @property
    def symbol_count(self) -> int:
        return len(self.per_symbol)


def _replay_models(
    frame,
    spec,
    *,
    angel_model,
    devil_model,
    angel_threshold: float,
    devil_threshold: float,
    angel_feature_cols: Optional[Tuple[str, ...]] = None,
    devil_feature_cols: Optional[Tuple[str, ...]] = None,
) -> LabBacktestReport:
    """Shared per-symbol replay: gate-trained models and served artifacts both
    come through here, so the pooled math can never drift between the two."""
    from execution.risk_manager import RiskManager, RiskProfile

    profile = RiskProfile.for_asset_class(spec.asset_class)
    risk_manager = RiskManager(profile, alpha_overrides=frame.alpha_table or {})
    alphas = frame.alpha_table

    per_symbol: Dict[str, BacktestReport] = {}
    for sym in sorted(frame.df["symbol"].unique().to_list()):
        sub = frame.df.filter(pl.col("symbol") == sym)
        if sub.height < 2:
            continue
        strategy = LabModelStrategy(
            sub,
            frame.feature_cols,
            angel_model,
            devil_model,
            angel_threshold,
            devil_threshold,
            angel_feature_cols=angel_feature_cols,
            devil_feature_cols=devil_feature_cols,
        )
        per_symbol[sym] = run_backtest(
            strategy,
            sub,
            sl_mult=spec.geometry.sl_mult,
            tp_mult=spec.geometry.tp_mult,
            max_hold_bars=spec.geometry.max_hold,
            risk_manager=risk_manager,
            spread_alphas=alphas,
            default_alpha=profile.spread_atr_alpha,
            symbol_override=sym,
        )

    all_trades = [t for report in per_symbol.values() for t in report.trades]
    funnel: Dict[str, int] = {}
    for report in per_symbol.values():
        for gate_name, count in report.gate_rejections.items():
            funnel[gate_name] = funnel.get(gate_name, 0) + count

    if not all_trades:
        return LabBacktestReport(
            total_trades=0,
            wins=0,
            win_rate=0.0,
            gross_ev_r=0.0,
            net_ev_r=0.0,
            profit_factor_net=0.0,
            max_drawdown_r=0.0,
            gate_rejections=funnel,
            per_symbol=per_symbol,
            toll_mode="spread_table" if alphas else "flat",
        )

    # Pooled drawdown needs an ordering; entry timestamps are the only real
    # clock shared across symbols (the per-symbol runs are independent, so this
    # is the standard one-basket approximation, not a portfolio simulation).
    ordered = sorted(all_trades, key=lambda t: t.entry_timestamp)
    net_rs = np.array([t.net_r for t in ordered])
    gross_rs = np.array([t.gross_r for t in ordered])
    cum = np.cumsum(net_rs)
    drawdowns = np.maximum.accumulate(cum) - cum

    wins = int(sum(t.macro_win for t in ordered))
    return LabBacktestReport(
        total_trades=len(ordered),
        wins=wins,
        win_rate=wins / len(ordered),
        gross_ev_r=float(gross_rs.mean()),
        net_ev_r=float(net_rs.mean()),
        profit_factor_net=_profit_factor(net_rs),
        max_drawdown_r=float(drawdowns.max()) if drawdowns.size else 0.0,
        gate_rejections=funnel,
        per_symbol=per_symbol,
        toll_mode="spread_table" if alphas else "flat",
    )


def run_model_backtest(frame, gate, spec) -> LabBacktestReport:
    """
    Backtest the gate's models over the frame, symbol by symbol, live-gated.

    ``frame`` is the FrameResult the gate scored; ``gate`` is its GateResult.
    """
    return _replay_models(
        frame,
        spec,
        angel_model=gate.angel_model,
        devil_model=gate.devil_model,
        angel_threshold=gate.report.production_angel_threshold,
        devil_threshold=gate.production_threshold,
    )


def run_artifact_backtest(frame, spec, artifact) -> LabBacktestReport:
    """
    Replay a SERVED artifact over the frame at its own pinned bars.

    ``artifact`` is a ``lab.artifact.ServedArtifact`` (duck-typed here to keep
    this module free of an import cycle): the Angel/Devil pair, the
    ``threshold.json`` bars the live bot uses, and the fit-time feature order.
    """
    return _replay_models(
        frame,
        spec,
        angel_model=artifact.angel_model,
        devil_model=artifact.devil_model,
        angel_threshold=artifact.angel_threshold,
        devil_threshold=artifact.devil_threshold,
        angel_feature_cols=tuple(artifact.angel_feature_cols),
        devil_feature_cols=tuple(artifact.devil_feature_cols),
    )
