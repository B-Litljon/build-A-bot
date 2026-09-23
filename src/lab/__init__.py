"""src/lab — the offline feature lab.

A tool for answering one question: does a candidate feature set move the
retrainer's own promotion verdict? It composes existing pieces — the ml feature
generators, core.retrainer's labels/gate, analysis.strategy_backtester — and
adds no trading code. Nothing in src/execution or run_oanda imports it, so a
broken lab cannot reach the live bot.

The spec/registry/frames surface is import-light (polars only, heavy imports
are lazy inside the builder functions). Gate, backtest, experiments, report and
CLI are exposed lazily via PEP 562 so a spec-only script never boots LightGBM.

See src/lab/README.md for the loop and the v1 limits.
"""

from lab.data import load_bars, resolve_alpha_table
from lab.frames import FrameResult, build_frame, stack_bars
from lab.registry import (
    FeatureFamily,
    available,
    feature_columns,
    get_generators,
    register_family,
    register_feature,
)
from lab.spec import (
    DEFAULT_TRADEABLE_6,
    FeatureSpec,
    GateConfig,
    GeometrySpec,
    LabelSpec,
)

_LAZY = {
    "GateResult": "lab.gate",
    "run_gate": "lab.gate",
    "LabBacktestReport": "lab.backtest",
    "LabModelStrategy": "lab.backtest",
    "run_artifact_backtest": "lab.backtest",
    "run_model_backtest": "lab.backtest",
    "ServedArtifact": "lab.artifact",
    "ArtifactReplayResult": "lab.artifact",
    "load_served_artifact": "lab.artifact",
    "replay_served_artifact": "lab.artifact",
    "DEFAULT_FRAME_CACHE": "lab.experiments",
    "ExperimentResult": "lab.experiments",
    "ExperimentRunner": "lab.experiments",
    "DEFAULT_REPORT_DIR": "lab.report",
    "render_report": "lab.report",
    "write_report": "lab.report",
    "render_artifact_report": "lab.report",
    "write_artifact_report": "lab.report",
    "seed_specs": "lab.specs",
    "main": "lab.cli",
}

__all__ = [
    "DEFAULT_TRADEABLE_6",
    "FeatureSpec",
    "GateConfig",
    "GeometrySpec",
    "LabelSpec",
    "FrameResult",
    "build_frame",
    "stack_bars",
    "load_bars",
    "resolve_alpha_table",
    "FeatureFamily",
    "available",
    "feature_columns",
    "get_generators",
    "register_family",
    "register_feature",
] + sorted(_LAZY)


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'lab' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
