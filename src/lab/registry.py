"""Feature-family registry -- the only seam a new lab feature needs to touch.

A family is a named builder of BaseFeatureGenerators plus the model-facing
columns they produce. ``feature_sets=("v3_base", "microstructure")`` resolves to
the generators in that order and to exactly those columns, so adding a candidate
feature never edits FeaturePipeline, the retrainer, or the live strategy.

Built-ins are registered lazily (their modules import talib/lightgbm), which
keeps this module import-light: the registry contract is dependency-free.

Glossary:
    FeatureFamily -- name -> (build, columns, description). ``build(spec,
        alpha_table)`` returns generator INSTANCES; ``columns(spec,
        alpha_table)`` returns the column names the model may see, which is a
        subset of what the generators append (bb_upper, sma_50, htf_rsi_14 and
        friends are intermediates or dead features, deliberately not model
        inputs).
    register_family -- functional registration for families whose generators
        take configuration.
    register_feature -- decorator for a no-argument generator class; columns
        come from the class's ``feature_cols`` attribute or the ``columns``
        argument.
    get_generators -- expand a spec's feature_sets (plus extra_generators) into
        the ordered generator list, raising on an unknown family rather than
        silently skipping it (a silent drop would score a different experiment
        than the spec names).
    feature_columns -- the matching model-facing column list, de-duplicated in
        first-seen order.
    v3_base -- the production V3 feature stack (V3Base + V3HTF + V3Session +
        V3Cost, in that order), whose columns equal the retrainer's
        BASE_FEATURE_COLS (+ cost_ratio when the cost table is active). This is
        the control family: a lab frame built from it alone must equal the
        retrainer's engineer_features_and_labels output row for row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ml.core.interfaces import BaseFeatureGenerator

BuildFn = Callable[[object, Optional[dict]], List[BaseFeatureGenerator]]
ColumnsFn = Callable[[object, Optional[dict]], Tuple[str, ...]]


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    build: BuildFn
    columns: ColumnsFn
    description: str = ""


_REGISTRY: Dict[str, FeatureFamily] = {}


def register_family(
    name: str,
    *,
    build: BuildFn,
    columns: ColumnsFn,
    description: str = "",
    replace: bool = False,
) -> FeatureFamily:
    """Register a feature family; duplicate names raise unless ``replace``."""
    if not name.strip():
        raise ValueError("feature family name must be non-empty")
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"feature family {name!r} is already registered "
            f"({_REGISTRY[name].description or 'no description'})"
        )
    family = FeatureFamily(name=name, build=build, columns=columns, description=description)
    _REGISTRY[name] = family
    return family


def register_feature(
    name: str,
    *,
    columns: Optional[Tuple[str, ...]] = None,
    description: str = "",
    replace: bool = False,
):
    """
    Decorator registering a no-argument BaseFeatureGenerator subclass.

    ``columns`` may also live on the class as a ``feature_cols`` attribute.
    Generators that need constructor configuration should use register_family.
    """

    def decorator(cls: type) -> type:
        if not issubclass(cls, BaseFeatureGenerator):
            raise TypeError(f"{cls!r} is not a BaseFeatureGenerator")
        declared = columns or getattr(cls, "feature_cols", None)
        if not declared:
            raise ValueError(
                f"register_feature({name!r}): pass columns=... or give {cls.__name__} "
                "a feature_cols attribute"
            )
        doc_lines = (cls.__doc__ or "").strip().splitlines()
        register_family(
            name,
            build=lambda spec, table: [cls()],
            columns=lambda spec, table: tuple(declared),
            description=description or (doc_lines[0] if doc_lines else f"family {name}"),
            replace=replace,
        )
        return cls

    return decorator


def available() -> List[FeatureFamily]:
    """Every registered family, sorted by name."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def get_generators(spec, alpha_table: Optional[dict] = None) -> List[BaseFeatureGenerator]:
    """Expand a spec's feature families (then extras) into generator instances."""
    out: List[BaseFeatureGenerator] = []
    for family_name in spec.feature_sets:
        family = _REGISTRY.get(family_name)
        if family is None:
            raise KeyError(
                f"unknown feature family {family_name!r}; registered: "
                f"{sorted(_REGISTRY)}"
            )
        out.extend(family.build(spec, alpha_table))
    out.extend(spec.extra_generators)
    return out


def feature_columns(spec, alpha_table: Optional[dict] = None) -> List[str]:
    """The model-facing columns for a spec, de-duplicated in first-seen order."""
    cols: List[str] = []
    for family_name in spec.feature_sets:
        family = _REGISTRY.get(family_name)
        if family is None:
            raise KeyError(
                f"unknown feature family {family_name!r}; registered: "
                f"{sorted(_REGISTRY)}"
            )
        for col in family.columns(spec, alpha_table):
            if col not in cols:
                cols.append(col)
    for gen in spec.extra_generators:
        for col in getattr(gen, "feature_cols", ()):  # validated in FeatureSpec
            if col not in cols:
                cols.append(col)
    return cols


# ═══════════════════════════════════════════════════════════════════════════
# Built-in: the production V3 stack, as one control family
# ═══════════════════════════════════════════════════════════════════════════


def _build_v3_base(spec, alpha_table: Optional[dict]) -> List[BaseFeatureGenerator]:
    from execution.risk_manager import RiskProfile
    from ml.features.v3_features import (
        V3BaseFeatures,
        V3CostFeatures,
        V3HTFFeatures,
        V3SessionFeatures,
    )

    profile = RiskProfile.for_asset_class(spec.asset_class)
    return [
        V3BaseFeatures(),
        V3HTFFeatures(timeframe=spec.htf_timeframe),
        V3SessionFeatures(),
        # No-op when alpha_table is None; needs natr_14 -> after V3Base.
        V3CostFeatures(
            alpha_table=alpha_table,
            default_alpha=profile.spread_atr_alpha,
            regime_window=profile.regime_window,
        ),
    ]


def _cols_v3_base(spec, alpha_table: Optional[dict]) -> Tuple[str, ...]:
    from core.retrainer._common import BASE_FEATURE_COLS

    cols = list(BASE_FEATURE_COLS)
    if alpha_table:
        cols.append("cost_ratio")
    return tuple(cols)


register_family(
    "v3_base",
    build=_build_v3_base,
    columns=_cols_v3_base,
    description=(
        "The production V3 stack (Base + HTF + Session + Cost). Its columns are "
        "the retrainer's BASE_FEATURE_COLS (+ cost_ratio with the spread table)."
    ),
)
