"""run_gate -- score a lab frame with the retrainer's own promotion gate.

No new metric exists here. ``validate_candidate`` is the exact function a
production retrain calls, so a lab PASS means "would have promoted" with no
translation, and ``edge_over_random`` (added 2026-09-14) is the primary number
the edge-budget work says matters. See src/lab/README.md for what v1 does NOT
cover (the artifact holdout, which needs the production feature stack).

Glossary:
    GateResult -- the ValidationReport plus the returned models and the frozen
        production thresholds, so a backtest can replay exactly the artifacts
        the gate judged.
    run_gate -- builds the hyperparameters, validates the spec's estimator
        family against the loaded retrainer, and runs the walk-forward folds.
    devil_label_context -- sets RETRAIN_DEVIL_LABEL from the spec for the
        duration of the run (the retrainer reads it per call) and restores the
        previous value, warning when it had to override an explicit setting.
    edge_over_random -- ValidationReport field, pooled fold win rate minus the
        bracket's base rate: the skill-vs-regime separator. The lab's primary
        decision metric.
    require_model_family -- MODEL_FAMILY is read when the retrainer is
        imported, so the lab refuses a spec that disagrees with the loaded
        value instead of silently scoring the wrong estimator. Launch with
        MODEL_FAMILY=<family> to select it. The CLI run/ablate path passes
        allow_env_override=True: an env-selected family IS the run's arm
        (the W4 estimator A/B launches a lightgbm seed spec under catboost).
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from core.retrainer._types import ValidationReport

_DEVIL_LABEL_ENV = "RETRAIN_DEVIL_LABEL"


@dataclass
class GateResult:
    """What run_gate returns: verdict, artifacts, and the frozen thresholds."""

    report: ValidationReport
    angel_model: object
    devil_model: object
    angel_features: List[str]
    devil_features: List[str]
    production_threshold: float
    hmm_models: Optional[dict]
    model_family: str
    elapsed_s: float

    @property
    def edge_over_random(self) -> float:
        return self.report.edge_over_random


def require_model_family(declared: str, *, allow_env_override: bool = False) -> str:
    """Fail loudly when the spec's estimator is not the loaded one.

    With ``allow_env_override`` (the CLI run/ablate path), an env-selected
    MODEL_FAMILY is the run's own arm — the estimator A/B (W4) launches a
    lightgbm-pinned seed spec under ``MODEL_FAMILY=catboost`` and the guard
    must accept that instead of refusing it. The override is still LOUD: the
    returned family (what the report records) is the loaded one, and the
    frame hash excludes the estimator entirely, so the two arms share one
    cached frame by construction. Without the flag the original contract
    stands: a spec file naming an estimator must match the loaded retrainer.
    """
    from core.retrainer._common import MODEL_FAMILY as loaded

    wanted = declared.strip().lower()
    if wanted != loaded:
        if not allow_env_override:
            raise RuntimeError(
                f"spec requests model_family={wanted!r} but the retrainer was imported "
                f"with MODEL_FAMILY={loaded!r}. MODEL_FAMILY is read at import time — "
                f"relaunch with MODEL_FAMILY={wanted}."
            )
        import logging

        logging.getLogger(__name__).warning(
            "spec %r declares model_family=%r but the process was launched with "
            "MODEL_FAMILY=%r — running the %s arm (env-selected estimator A/B); "
            "the frame is shared either way (the frame hash excludes the estimator)",
            "<spec>", wanted, loaded, loaded,
        )
    return loaded


@contextlib.contextmanager
def devil_label_context(kind: str):
    """Run a block with RETRAIN_DEVIL_LABEL pinned to the spec's label kind."""
    previous = os.environ.get(_DEVIL_LABEL_ENV)
    if previous is not None and previous.strip().lower() != kind:
        import logging

        logging.getLogger(__name__).warning(
            "%s=%r overridden to %r by the spec for this run (restored after)",
            _DEVIL_LABEL_ENV, previous, kind,
        )
    os.environ[_DEVIL_LABEL_ENV] = kind
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_DEVIL_LABEL_ENV, None)
        else:
            os.environ[_DEVIL_LABEL_ENV] = previous


def run_gate(frame, spec, *, n_folds: Optional[int] = None) -> GateResult:
    """
    Run the retrainer's walk-forward gate over a built frame.

    ``n_folds`` defaults to the spec's GateConfig.n_folds (3, production's).
    The frame's own chop_veto_rate is threaded through so the trade-count
    backstop scales the same way a production run's does.
    """
    from core.retrainer._common import get_hyperparameters
    from core.retrainer._gate import validate_candidate

    family = require_model_family(spec.gate.model_family, allow_env_override=True)
    angel_params, devil_params = get_hyperparameters(spec.asset_class)
    fold_count = int(n_folds if n_folds is not None else spec.gate.n_folds)

    started = time.monotonic()
    with devil_label_context(spec.label.kind):
        (
            report,
            angel_model,
            devil_model,
            angel_feats,
            devil_feats,
            production_threshold,
            hmm_models,
        ) = validate_candidate(
            frame.df,
            list(frame.feature_cols),
            sl_mult=spec.geometry.sl_mult,
            tp_mult=spec.geometry.tp_mult,
            n_folds=fold_count,
            angel_params=angel_params,
            devil_params=devil_params,
            chop_veto_rate=frame.chop_veto_rate,
        )
    elapsed = time.monotonic() - started
    return GateResult(
        report=report,
        angel_model=angel_model,
        devil_model=devil_model,
        angel_features=list(angel_feats),
        devil_features=list(devil_feats),
        production_threshold=float(production_threshold),
        hmm_models=hmm_models,
        model_family=family,
        elapsed_s=elapsed,
    )
