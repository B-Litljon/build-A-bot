"""CLI for the feature lab.

    PYTHONPATH=src:. python -m lab.cli list
    PYTHONPATH=src:. python -m lab.cli run --name v3_base_control
    PYTHONPATH=src:. python -m lab.cli run --spec specs/my_feature.py
    PYTHONPATH=src:. python -m lab.cli replay --name v3_base_control

``run`` answers "would this feature set promote?" (the gate). ``replay``
answers "what would the SERVED artifact have done on this frame?" — it loads
``--model-dir`` (default: ``OANDA_MODEL_DIR`` or models/forex_m15_wide), pins
the artifact's own ``threshold.json`` bars, and produces a replay report. It
does not retrain and has no PASS/FAIL.

Exit codes mirror the retrainer: 0 gate passed, 2 gate rejected, 1 error — so a
shell loop can branch the same way it does on a retrain. ``replay`` exits 0 on
a completed replay.

``--spec`` loads a Python file that defines a module-level ``SPEC`` (or a
``build_spec()`` function). Spec files must stay declarative: they may import
lab.spec / lab.features, but any import of the training stack freezes
MODEL_FAMILY before the CLI can apply the spec's estimator choice.

Glossary:
    _load_spec_file -- import a user spec by path (no sys.path games, no
        network) and pull SPEC/build_spec().
    _apply_model_family -- sets MODEL_FAMILY from the flag, the environment,
        or the spec — in that precedence — BEFORE the retrainer is imported,
        which is the only moment it can take effect. The env beats the spec
        so the W4 estimator A/B can run a lightgbm-pinned seed spec under
        catboost (the frame hash excludes gate.model_family, so both arms
        share one cached frame).
    _resolve_spec -- --name/--spec validation and seed lookup, shared by run
        and replay.
    _cmd_replay -- the served-artifact path: no MODEL_FAMILY, no gate; loads
        the artifact, prepares the frame, runs the live-gated replay, writes
        the report.
    main -- argparse dispatch; derives the report directory and the runner
        switches, prints the JSON summary, writes the recon report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from lab.spec import FeatureSpec  # noqa: F401  (type for spec files / clarity)


def _load_spec_file(path: str) -> FeatureSpec:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"spec file not found: {file_path}")
    module_spec = importlib.util.spec_from_file_location(
        f"lab_user_spec_{file_path.stem}", file_path
    )
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"cannot import spec file: {file_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if hasattr(module, "SPEC"):
        spec = module.SPEC
    elif hasattr(module, "build_spec"):
        spec = module.build_spec()
    else:
        raise SystemExit(
            f"{file_path} must define a module-level SPEC or a build_spec() function"
        )
    if not isinstance(spec, FeatureSpec):
        raise SystemExit(
            f"{file_path} returned {type(spec).__name__}, expected FeatureSpec"
        )
    return spec


def _apply_model_family(spec: FeatureSpec, explicit: str | None) -> None:
    """Pin MODEL_FAMILY before the retrainer imports (it is read at import).

    Precedence (W4, 2026-09-22): an explicit --model-family wins, then an
    env-selected family (MODEL_FAMILY=catboost python -m lab.cli run ... —
    the estimator A/B seam), then the spec's declared family. The env beats
    the spec because a seed spec is pinned to lightgbm and the A/B needs to
    run THAT spec under catboost; the content hash deliberately excludes
    gate.model_family, so both arms reuse the same cached frame.
    """
    current = os.environ.get("MODEL_FAMILY", "").strip().lower()
    if explicit:
        family = explicit.strip().lower()
        if family != spec.gate.model_family:
            raise SystemExit(
                f"--model-family {family!r} disagrees with the spec's "
                f"{spec.gate.model_family!r}; the spec is the contract"
            )
    elif current:
        family = current  # env-selected A/B arm wins over the spec default
    else:
        family = spec.gate.model_family.strip().lower()
    if family != current:
        os.environ["MODEL_FAMILY"] = family


def _cmd_list() -> int:
    from lab.registry import available
    from lab.specs import seed_specs

    print("Feature families:")
    for family in available():
        print(f"  {family.name:<18} {family.description}")
    print("\nSeed specs:")
    for name in sorted(seed_specs()):
        print(f"  {name}")
    return 0


def _resolve_spec(args) -> FeatureSpec:
    """--name/--spec validation and lookup, shared by run and replay."""
    if args.name and args.spec:
        raise SystemExit("pass either --name or --spec, not both")
    if not args.name and not args.spec:
        raise SystemExit("pass --name <seed> or --spec <path>")

    if args.spec:
        return _load_spec_file(args.spec)

    from lab.specs import seed_specs

    seeds = seed_specs()
    if args.name not in seeds:
        raise SystemExit(
            f"unknown seed {args.name!r}; available: {sorted(seeds)}"
        )
    return seeds[args.name]


def _cmd_run(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args)
    command = (
        f"PYTHONPATH=src:. python -m lab.cli run --spec {args.spec}"
        if args.spec
        else f"PYTHONPATH=src:. python -m lab.cli run --name {args.name}"
    )

    # MODEL_FAMILY must be set before anything imports the retrainer.
    _apply_model_family(spec, args.model_family)

    from lab.experiments import DEFAULT_FRAME_CACHE, ExperimentRunner
    from lab.report import DEFAULT_REPORT_DIR, write_report

    runner = ExperimentRunner(
        frame_cache_dir=Path(args.frame_cache_dir or DEFAULT_FRAME_CACHE),
        refresh_bars=args.refresh_bars,
        use_frame_cache=not args.no_frame_cache,
        do_backtest=not args.no_backtest,
    )
    result = runner.run(spec)
    print(json.dumps(result.summary(), indent=2, default=str))

    if not args.no_report:
        path = write_report(
            result,
            out_dir=Path(args.report_dir or DEFAULT_REPORT_DIR),
            command=command,
        )
        print(f"\nReport: {path}", file=sys.stderr)

    return 0 if result.gate.report.gate_passed else 2


def _cmd_replay(args: argparse.Namespace) -> int:
    """Replay the served artifact over the spec's frame; no gate, no retrain."""
    spec = _resolve_spec(args)
    command = (
        f"PYTHONPATH=src:. python -m lab.cli replay --spec {args.spec}"
        if args.spec
        else f"PYTHONPATH=src:. python -m lab.cli replay --name {args.name}"
    )
    if args.model_dir:
        command += f" --model-dir {args.model_dir}"

    from lab.artifact import load_served_artifact, replay_served_artifact
    from lab.experiments import DEFAULT_FRAME_CACHE, ExperimentRunner
    from lab.report import DEFAULT_REPORT_DIR, write_artifact_report

    artifact = load_served_artifact(args.model_dir)
    runner = ExperimentRunner(
        frame_cache_dir=Path(args.frame_cache_dir or DEFAULT_FRAME_CACHE),
        refresh_bars=args.refresh_bars,
        use_frame_cache=not args.no_frame_cache,
    )
    result = replay_served_artifact(spec, artifact, runner=runner)
    print(json.dumps(result.summary(), indent=2, default=str))

    if not args.no_report:
        path = write_artifact_report(
            result,
            out_dir=Path(args.report_dir or DEFAULT_REPORT_DIR),
            command=command,
            slug=args.report_slug,
        )
        print(f"\nReport: {path}", file=sys.stderr)

    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    """Feature-interaction ablation: edge(full) - edge(full - X), with CIs."""
    spec = _resolve_spec(args)
    command = (
        f"PYTHONPATH=src:. python -m lab.cli ablate --spec {args.spec}"
        if args.spec
        else f"PYTHONPATH=src:. python -m lab.cli ablate --name {args.name}"
    )

    # MODEL_FAMILY must be set before anything imports the retrainer.
    _apply_model_family(spec, args.model_family)

    from lab.ablate import ablate
    from lab.experiments import DEFAULT_FRAME_CACHE, ExperimentRunner
    from lab.report import DEFAULT_REPORT_DIR, write_ablation_report

    runner = ExperimentRunner(
        bar_cache_dir=Path(args.bar_cache_dir or "analysis_cache/strategy_matrix"),
        frame_cache_dir=Path(args.frame_cache_dir or DEFAULT_FRAME_CACHE),
        refresh_bars=args.refresh_bars,
        use_frame_cache=not args.no_frame_cache,
    )
    result = ablate(spec, runner=runner)
    print(json.dumps(result.summary(), indent=2, default=str))

    if not args.no_report:
        path = write_ablation_report(
            result,
            out_dir=Path(args.report_dir or DEFAULT_REPORT_DIR),
            command=command,
        )
        print(f"\nReport: {path}", file=sys.stderr)

    return 0


def _add_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", help="seed spec name (see `list`)")
    parser.add_argument("--spec", help="path to a Python file defining SPEC/build_spec()")
    parser.add_argument(
        "--no-frame-cache", action="store_true", help="rebuild the frame every run"
    )
    parser.add_argument(
        "--refresh-bars", action="store_true", help="re-fetch bars from the provider"
    )
    parser.add_argument("--frame-cache-dir", default=None)
    parser.add_argument("--bar-cache-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--no-report", action="store_true")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lab.cli",
        description="Feature lab: build a frame, run the production gate, backtest it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list registered feature families and seed specs")

    run = sub.add_parser("run", help="run one gate experiment")
    _add_spec_args(run)
    run.add_argument(
        "--model-family",
        default=None,
        help="must match the spec; set before the retrainer imports (default: spec's)",
    )
    run.add_argument("--no-backtest", action="store_true", help="skip the replay")

    replay = sub.add_parser(
        "replay", help="replay the served artifact on a spec's frame (no gate)"
    )
    _add_spec_args(replay)
    replay.add_argument(
        "--model-dir",
        default=None,
        help="served model dir (default: OANDA_MODEL_DIR or models/forex_m15_wide)",
    )
    replay.add_argument(
        "--report-slug",
        default=None,
        help="override the report filename slug (default: served-artifact-<spec>)",
    )

    ablate = sub.add_parser(
        "ablate",
        help="edge(full cocktail) - edge(cocktail - X) per family, with CIs",
    )
    _add_spec_args(ablate)
    ablate.add_argument(
        "--model-family",
        default=None,
        help="must match the spec; set before the retrainer imports (default: spec's)",
    )

    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list()
    if args.command == "replay":
        return _cmd_replay(args)
    if args.command == "ablate":
        return _cmd_ablate(args)
    return _cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
