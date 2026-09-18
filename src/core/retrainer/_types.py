"""FoldMetrics, HoldoutMetrics, ValidationReport — the structured results a retrain run returns.
Split out of core/retrainer.py on 2026-09-16.
"""
from __future__ import annotations

from ._common import (
    logger,
    List,
    Optional,
    dataclass,
    field,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward CV fold."""

    fold_number: int
    train_size: int
    val_size: int
    brier_score: float
    expected_value: float
    angel_proposed_trades: int
    devil_approved_trades: int
    win_rate: float
    # Macro (45-bar bracket) wins among this fold's scored approvals — the
    # raw (wins, trades) evidence the pooled Clopper-Pearson PF lower bound
    # is computed from. 0 on the degenerate worst-case paths.
    macro_wins: int = 0
    # What a random long entry would have won on the same fold's tradeable bars,
    # under the same bracket. The benchmark this fold's macro win rate has to beat
    # to mean anything (see _macro_base_rate). nan when the frame carries no
    # label; 0.0 on the degenerate worst-case paths, where it is unused.
    base_rate: float = float("nan")


@dataclass
class HoldoutMetrics:
    """Metrics for the chronologically last holdout slice on the final artifact."""

    used: bool
    fraction: float
    start_date: Optional[str]
    end_date: Optional[str]
    brier_score: float
    expected_value: float
    win_rate: float
    profit_factor: float
    trades: int
    angel_proposed_trades: int
    bypass_reason: Optional[str] = None
    # Macro wins among the scored trades -- the PF gate's raw evidence, carried
    # so the confidence bound can be recomputed from metadata alone.
    wins: int = 0
    # Lower confidence bound on the holdout PF (see HOLDOUT_PF_CONFIDENCE).
    # 0.0 for a zero-trade scored holdout; None when the holdout was never
    # scored (bypassed, disabled, or no fold models to score).
    pf_lower_bound: Optional[float] = None
    pf_confidence: Optional[float] = None
    # Rows dropped at the tail of this slice because their macro walk ran off
    # the end of the fetched window (unresolvable outcome).
    purged_tail_rows: int = 0
    # True when the fold gate already failed and the holdout was scored for
    # diagnostics only -- the metrics inform, the fold verdict stands.
    diagnostic_only: bool = False


@dataclass
class ValidationReport:
    """Aggregated validation report across all walk-forward folds."""

    fold_metrics: List[FoldMetrics]
    mean_brier: float
    mean_ev: float
    final_profit_factor: float
    final_win_rate: float
    final_total_trades: int
    gate_passed: bool
    pooled_oos_trades: int = 0
    chop_veto_rate: float = 0.0
    effective_trade_floor: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)
    holdout: Optional[HoldoutMetrics] = None
    # Pooled macro wins across folds — with pooled_oos_trades, the evidence
    # behind pooled_pf_lower_bound.
    pooled_oos_wins: int = 0
    # Clopper-Pearson PF lower bounds (see _holdout_pf_lower_bound) on the
    # pooled fold trades and on Fold 3 alone — the fold gate's evidential
    # bars, replacing the old flat 300-trade floor and Fold-3 point PF.
    pooled_pf_lower_bound: float = 0.0
    fold3_pf_lower_bound: float = 0.0
    # The Angel proposal bar the returned models were trained with:
    # calibrated from OOF probabilities unless ANGEL_THRESHOLD pinned a fixed
    # value. Written into threshold.json on promotion; MLStrategy prefers the
    # pinned value over the constant, keeping train/serve symmetry.
    production_angel_threshold: float = 0.0
    # Pooled edge over the bracket's own base rate: the fold win rate minus what a
    # random long entry would have won on the same bars, pooled across folds. The
    # one number that separates skill from a favourable regime — see
    # _macro_base_rate for the measurement that motivated it.
    pooled_base_rate: float = float("nan")
    edge_over_random: float = float("nan")
