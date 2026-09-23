"""Seed specs -- the first three experiments the plan calls for.

Each is a plain FeatureSpec and runs through the same loop; they exist to prove
the lab reproduces production before it is trusted with a new idea.

    1. v3_base_control    -- the production 17 features, served geometry, cost
        table OFF (the served model's configuration). This is the calibration
        run: it must reproduce the gate's known numbers (~+0.045 edge over
        random) or the lab is wrong, not the model.
    2. spread_table_control -- identical except the cost table is ON, closing
        the one configured-but-unused asymmetry (the 2026-07-07 experiment was
        confounded by a window shift and never re-run).
    3. microstructure     -- control + the seed candidate family, cost table
        off so the only changed variable is the feature set.

Glossary:
    seed_specs -- name -> FeatureSpec; also imports lab.features so the
        "microstructure" family is registered before any spec resolves.
"""

from __future__ import annotations

from typing import Dict

from lab.spec import FeatureSpec


def _base(name: str, **kwargs) -> FeatureSpec:
    return FeatureSpec(name=name, days_back=730, granularity=15, **kwargs)


def seed_specs() -> Dict[str, FeatureSpec]:
    """The three seed experiments, keyed by CLI-friendly name."""
    from lab import features  # noqa: F401  (registers the candidate family)

    return {
        "v3_base_control": _base(
            "v3_base_control",
            feature_sets=("v3_base",),
            use_spread_table=False,
        ),
        "spread_table_control": _base(
            "spread_table_control",
            feature_sets=("v3_base",),
            use_spread_table=True,
        ),
        "microstructure": _base(
            "microstructure",
            feature_sets=("v3_base", "microstructure"),
            use_spread_table=False,
        ),
    }
