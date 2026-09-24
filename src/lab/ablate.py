"""lab ablate -- feature-INTERACTION ablation: what does one family add in situ?

Single-feature-at-a-time answers the wrong question. A family's honest in-situ
effect on the model's decisions is

    edge(full cocktail) - edge(cocktail - X)

measured on the SAME frame, labels, veto and geometry with only ``feature_sets``
varying — the polypharmacy question. A lone-feature run is a diagnostic of that
feature in isolation, not evidence about its contribution to the production
stack; families interact through the model's splits and through the veto, so
the only honest measure of a family is this delta.

The frame cache makes this cheap: every variant shares the base spec's data
window, geometry, labels and cost-table switch, so the N+1 runs differ in
``feature_sets`` only, and after W1 the cache key covers family versions and
generator state, so a variant never reuses another variant's frame.

Delta intervals: each arm's win rate gets a Clopper-Pearson bound (the same
exact-binomial conservatism the gate's PF lower bounds use — Wilson
under-covers at n < ~40, the regime these runs actually live in), and the
delta is reported with an interval derived from the two arms' bounds. The
delta's point estimate is NEVER presented alone: a delta consistent with zero
on thin trades is the expected outcome for the v3_base regime (~30-40 pooled
trades) and must not be read as "no effect" or "effect".

Glossary:
    AblationVariant -- one cocktail-minus-X spec plus the metrics its gate run
        produced: pooled trades/wins, edge_over_random, pooled and fold-3 PF
        lower bounds.
    AblationResult -- the full cocktail's metrics, every variant, and the
        per-family deltas with their Clopper-Pearson intervals.
    ablate -- expand a spec into N+1 variant specs and run the gate on each,
        sharing one bar load and the frame cache.
    delta_with_ci -- (delta, ci_low, ci_high) for one family: the full
        cocktail's edge minus the minus-X arm's edge, interval from both
        arms' CP win-rate bounds combined in quadrature (conservative, not
        exact — the arms share the same base-rate denominator, so their
        errors are positively correlated and exact joint intervals would be
        NARROWER, i.e. this is the safe direction).
    thin_trades -- the pooled-trade threshold under which a delta is flagged
        as noise-dominated (the v3_base regime is ~30-40 trades; the caveat
        always fires there).
    expand_variants -- spec -> (full spec, [minus-X specs]); geometry, labels,
        data and cost table are carried over UNCHANGED — only feature_sets
        differs.
    THIN_TRADES -- the pooled-trade threshold that arms the caveat.
    CP_CONFIDENCE -- the one-sided binomial confidence level for delta
        bounds, mirroring the gate's conservatism.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

THIN_TRADES = 100
CP_CONFIDENCE = 0.90


@dataclass
class AblationVariant:
    """One cocktail-minus-X arm: the spec plus its gate metrics."""

    family: str
    spec: object
    gate: object
    frame: object
    frame_from_cache: bool

    @property
    def report(self):
        return self.gate.report


@dataclass
class AblationResult:
    """The full cocktail, every minus-X variant, and the per-family deltas."""

    spec: object
    full: AblationVariant
    variants: List[AblationVariant]
    deltas: Dict[str, dict] = field(default_factory=dict)
    run_seconds: float = 0.0

    def summary(self) -> dict:
        full = self.full.report

        def _arm_metrics(variant: AblationVariant) -> dict:
            rep = variant.report
            return {
                "pooled_oos_trades": rep.pooled_oos_trades,
                "pooled_oos_wins": rep.pooled_oos_wins,
                "edge_over_random": rep.edge_over_random,
                "pooled_pf_lower_bound": rep.pooled_pf_lower_bound,
                "fold3_pf_lower_bound": rep.fold3_pf_lower_bound,
                "gate_passed": rep.gate_passed,
                "content_hash": variant.spec.content_hash(),
                "frame_from_cache": variant.frame_from_cache,
            }

        return {
            "name": self.spec.name,
            "model_family": self.full.gate.model_family,
            "run_seconds": self.run_seconds,
            "full": _arm_metrics(self.full),
            "variants": [
                {
                    "family": var.family,
                    **_arm_metrics(var),
                    "delta": self.deltas.get(var.family),
                }
                for var in self.variants
            ],
        }


def _cp_win_rate_bound(wins: int, trades: int, *, upper: bool) -> float:
    """One-sided Clopper-Pearson bound on a binomial win rate.

    The exact interval the gate's PF lower bounds are built from
    (core/retrainer/_gate.py) — Wilson under-covers at n < ~40. Upper bound
    uses Beta(1-conf; wins+1, losses); lower uses Beta(1-conf; wins, losses+1).
    """
    trades = int(trades)
    wins = min(max(int(wins), 0), trades)
    if trades <= 0:
        return float("nan")
    if wins == 0 and not upper:
        return 0.0
    if wins == trades and upper:
        return 1.0
    from scipy.stats import beta as _scipy_beta

    if upper:
        return float(_scipy_beta.ppf(CP_CONFIDENCE, wins + 1, trades - wins))
    return float(_scipy_beta.ppf(1.0 - CP_CONFIDENCE, wins, trades - wins + 1))


def delta_with_ci(
    full_wins: int,
    full_trades: int,
    minus_wins: int,
    minus_trades: int,
) -> dict:
    """
    The full-cocktail edge minus the minus-X edge, with an interval.

    Both edges are in WIN-RATE units (edge_over_random's own currency —
    see the 2026-09-14 edge-budget work for why it is not R). Each arm's
    win rate is bounded with Clopper-Pearson; the delta's interval comes
    from the two bounds, widened in quadrature. The arms share one base
    population so their sampling errors are positively correlated — the
    exact joint interval is NARROWER than this one, so the direction of
    conservatism (never over-claiming a real delta) is the safe one.
    """
    full_wr = full_wins / full_trades if full_trades else float("nan")
    minus_wr = minus_wins / minus_trades if minus_trades else float("nan")
    delta = full_wr - minus_wr
    full_lo = _cp_win_rate_bound(full_wins, full_trades, upper=False)
    full_hi = _cp_win_rate_bound(full_wins, full_trades, upper=True)
    minus_lo = _cp_win_rate_bound(minus_wins, minus_trades, upper=False)
    minus_hi = _cp_win_rate_bound(minus_wins, minus_trades, upper=True)

    if any(_isnan(x) for x in (full_lo, full_hi, minus_lo, minus_hi)):
        return {
            "delta": delta,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "full_wr": full_wr,
            "minus_wr": minus_wr,
            "full_trades": full_trades,
            "minus_trades": minus_trades,
            "distinguishable_from_zero": False,
        }

    # full_wr in [full_lo, full_hi], minus_wr in [minus_lo, minus_hi]; each
    # arm's error is at most half its width. Independent-error worst case
    # (quadrature) — conservative because the arms share bars. The interval
    # is then clipped to the win-rate feasible range [-1, 1]: a win-rate
    # delta cannot exceed it, and an unclipped bound (e.g. a perfect arm vs
    # a dead one) would over-claim in exactly the direction a finding must
    # never go.
    full_half = (full_hi - full_lo) / 2.0
    minus_half = (minus_hi - minus_lo) / 2.0
    width = (full_half**2 + minus_half**2) ** 0.5
    return {
        "delta": delta,
        "ci_low": max(-1.0, delta - width),
        "ci_high": min(1.0, delta + width),
        "full_wr": full_wr,
        "minus_wr": minus_wr,
        "full_trades": full_trades,
        "minus_trades": minus_trades,
        "distinguishable_from_zero": not (delta - width <= 0.0 <= delta + width),
    }


def _isnan(value: float) -> bool:
    return value != value


def expand_variants(spec) -> Tuple[object, List[object]]:
    """A spec -> (full, [one spec per family with that family removed]).

    Labels, veto, geometry, symbols, window and cost table are carried over
    UNCHANGED — only ``feature_sets`` differs across variants, which is what
    makes the delta readable as a family effect. extra_generators are kept in
    every variant (they are the spec's fixed tail, not a registered family).
    """
    if len(spec.feature_sets) < 2:
        raise ValueError(
            f"ablation needs >= 2 feature families to have a cocktail to "
            f"subtract against; spec {spec.name!r} declares "
            f"{list(spec.feature_sets)}. Use `run` for a single-family spec."
        )
    full = spec
    variants = [
        replace(spec, name=f"{spec.name}_minus_{family}", feature_sets=tuple(s for s in spec.feature_sets if s != family))
        for family in spec.feature_sets
    ]
    return full, variants


def _run_variant(runner, spec, *, family: str, n_folds: Optional[int] = None) -> AblationVariant:
    """One gate run over a prepared frame."""
    from lab.gate import run_gate

    frame, from_cache = runner.prepare_frame(spec)
    gate = run_gate(frame, spec, n_folds=n_folds)
    return AblationVariant(
        family=family,
        spec=spec,
        gate=gate,
        frame=frame,
        frame_from_cache=from_cache,
    )


def _omitted_family(spec) -> str:
    """The family this minus-X variant omits, from its expanded name."""
    suffix = "_minus_"
    parts = spec.name.rsplit(suffix, 1)
    return parts[1] if len(parts) > 1 else ""


def ablate(spec, *, runner=None, n_folds: Optional[int] = None) -> AblationResult:
    """
    Run the full cocktail and every minus-X variant through the real gate.

    ``runner`` is an ``ExperimentRunner`` (bar cache shared across all N+1
    arms, frame cache keyed by the W1 content hash); a default one is created
    when omitted. ``n_folds`` overrides the spec's fold count for EVERY arm
    identically, so the deltas compare like with like.
    """
    from lab.experiments import ExperimentRunner

    started = time.monotonic()
    runner = runner or ExperimentRunner()

    full_spec, variant_specs = expand_variants(spec)

    full_variant = _run_variant(runner, full_spec, family="full", n_folds=n_folds)
    variants = [
        _run_variant(runner, v_spec, family=_omitted_family(v_spec), n_folds=n_folds)
        for v_spec in variant_specs
    ]

    full_rep = full_variant.report
    deltas: Dict[str, dict] = {}
    for variant in variants:
        deltas[variant.family] = delta_with_ci(
            full_rep.pooled_oos_wins,
            full_rep.pooled_oos_trades,
            variant.report.pooled_oos_wins,
            variant.report.pooled_oos_trades,
        )

    return AblationResult(
        spec=spec,
        full=full_variant,
        variants=variants,
        deltas=deltas,
        run_seconds=time.monotonic() - started,
    )