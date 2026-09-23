"""Served-artifact replay -- score the model the bot actually runs.

The gate (`lab.gate`) answers "would a candidate feature set promote?" by
training fresh fold models on a lab frame. This module answers the other
baseline question: **"what would the SERVED artifact have done on that same
frame?"** It loads the model directory the soak runs (``OANDA_MODEL_DIR``),
pins the bars the artifact was fitted with (``threshold.json``, with the same
precedence ``MLStrategy`` uses), and replays it through the same live-gated
backtester the gate path uses. Nothing is retrained, promoted, or written
outside ``llm_reports/``; nothing in ``src/execution`` imports this.

One trap this module exists to avoid: the strategy must read the artifact's
OWN feature order (``feature_names_in_``, CatBoost's ``feature_names_``), not
the frame's column order. LightGBM's numpy predict is positional, so passing
the frame order would silently score every bar against permuted columns. The
schema check in ``replay_served_artifact`` makes a missing column loud instead.

Glossary:
    DEFAULT_MODEL_DIR -- models/forex_m15_wide unless ``OANDA_MODEL_DIR`` says
        otherwise; the same env var ``soak.service`` declares.
    ServedArtifact -- the loaded Angel/Devil pair plus provenance: pinned bars,
        their source, the fit-time feature schemas, and ``metadata.json``.
    load_served_artifact -- read the pkls, ``threshold.json`` and
        ``metadata.json``; raise when an estimator carries no feature schema (a
        numpy-fitted model cannot be replayed safely) or when the Devil's
        schema names anything beyond the Angel's features + ``angel_prob``.
    predict_probabilities -- the exact scoring pair the strategy uses, exposed
        for population stats so the report and the replay never disagree.
    ArtifactReplayResult -- frame + raw population + per-window split + gated
        backtest; ``summary()`` is the machine-readable CLI output.
    replay_served_artifact -- prepare the spec's frame (cache-aware), validate
        the artifact against it, and run ``run_artifact_backtest``.
    measured_alpha_table -- the per-instrument cost table from
        ``config/spread_alphas_m15.json``, used by the report's honest-toll
        note. Not applied to the replay itself (the artifact has no
        ``cost_ratio`` input; the live bot prices at the flat default).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import polars as pl

from lab.backtest import LabBacktestReport, run_artifact_backtest

DEFAULT_MODEL_DIR = Path("models/forex_m15_wide")
_DEVIL_ONLY_COLUMN = "angel_prob"
_DEFAULT_DEVIL_THRESHOLD = 0.50
_SPREAD_TABLE_PATH = Path("config/spread_alphas_m15.json")


@dataclass
class ServedArtifact:
    """A loaded served model directory plus everything needed to replay it."""

    model_dir: Path
    angel_model: object
    devil_model: object
    angel_threshold: float
    devil_threshold: float
    threshold_source: str
    angel_feature_cols: Tuple[str, ...]
    devil_feature_cols: Tuple[str, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def trained_at(self) -> Optional[str]:
        return self.metadata.get("trained_at")

    @property
    def trained_on_symbols(self) -> Optional[List[str]]:
        return self.metadata.get("trained_on_symbols")


def _feature_names(model, path: Path) -> Tuple[str, ...]:
    """The fit-time column order, or a loud failure (positional predict)."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = getattr(model, "feature_names_", None)  # CatBoost spelling
    if names is None:
        raise ValueError(
            f"{path} exposes no feature_names_in_/feature_names_. A model fitted "
            "on a bare numpy array has no declared schema, so a replay cannot "
            "know the column order LightGBM/CatBoost predict against. Refusing "
            "to guess."
        )
    names = tuple(str(name) for name in names)  # numpy arrays are not truthy
    if not names:
        raise ValueError(f"{path} declares an empty feature schema")
    return names


def _load_thresholds(model_dir: Path) -> Tuple[float, float, str]:
    """
    The pinned bars, with live precedence: threshold.json over constants.

    ``MLStrategy`` reads both keys from the served directory's threshold.json
    and otherwise falls back to the global constants; a replay that used any
    other bar would not be the model the bot runs.
    """
    from core.thresholds import ANGEL_THRESHOLD

    path = model_dir / "threshold.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        angel = data.get("angel_threshold")
        devil = data.get("devil_threshold")
        if angel is not None and devil is not None:
            return float(angel), float(devil), "threshold.json"
        missing = [
            key
            for key, value in (("angel_threshold", angel), ("devil_threshold", devil))
            if value is None
        ]
        return (
            float(angel if angel is not None else ANGEL_THRESHOLD),
            float(devil if devil is not None else _DEFAULT_DEVIL_THRESHOLD),
            f"threshold.json (missing {', '.join(missing)} -> constants)",
        )
    return float(ANGEL_THRESHOLD), float(_DEFAULT_DEVIL_THRESHOLD), "constants"


def load_served_artifact(model_dir: Optional[str] = None) -> ServedArtifact:
    """Load the served model dir; the default is the soak's own declaration."""
    resolved = Path(model_dir or os.getenv("OANDA_MODEL_DIR", DEFAULT_MODEL_DIR))
    angel_path = resolved / "angel_latest.pkl"
    devil_path = resolved / "devil_latest.pkl"
    for path in (angel_path, devil_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"served artifact incomplete: {path} not found "
                f"(model dir {resolved})"
            )

    angel_model = joblib.load(angel_path)
    devil_model = joblib.load(devil_path)
    angel_features = _feature_names(angel_model, angel_path)
    devil_features = _feature_names(devil_model, devil_path)

    allowed = set(angel_features) | {_DEVIL_ONLY_COLUMN}
    extra = [c for c in devil_features if c not in allowed]
    if extra:
        raise ValueError(
            f"{devil_path} declares feature(s) {extra} beyond the Angel's schema "
            f"+ {_DEVIL_ONLY_COLUMN!r}; a replay cannot build that input."
        )

    metadata: dict = {}
    metadata_path = resolved / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            metadata = {}

    angel_threshold, devil_threshold, source = _load_thresholds(resolved)
    return ServedArtifact(
        model_dir=resolved,
        angel_model=angel_model,
        devil_model=devil_model,
        angel_threshold=angel_threshold,
        devil_threshold=devil_threshold,
        threshold_source=source,
        angel_feature_cols=angel_features,
        devil_feature_cols=devil_features,
        metadata=metadata,
    )


def predict_probabilities(
    frame_df: pl.DataFrame, artifact: ServedArtifact
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Angel and Devil probabilities for every row, in the artifact's own order.

    Exposed (rather than inlined in the strategy) so the population stats and
    the replay read the same numbers; a divergence between them would be a
    silent measurement bug.
    """
    schema = list(artifact.angel_feature_cols)
    X = frame_df.select(schema).to_numpy()
    angel_probs = artifact.angel_model.predict_proba(X)[:, 1]

    meta = pl.DataFrame(X, schema=schema).with_columns(
        pl.Series(_DEVIL_ONLY_COLUMN, angel_probs)
    )
    if tuple(artifact.devil_feature_cols) != (*artifact.angel_feature_cols, _DEVIL_ONLY_COLUMN):
        meta = meta.select(list(artifact.devil_feature_cols))

    devil_proba = artifact.devil_model.predict_proba(meta)
    if devil_proba.shape[1] == 1:  # single-class Devil (degenerate artifact)
        only = int(artifact.devil_model.classes_[0])
        devil_probs = np.full(len(meta), 1.0 if only == 1 else 0.0)
    else:
        devil_probs = devil_proba[:, 1]
    return angel_probs, devil_probs


def _bracket_profit_factor(
    wins: int, losses: int, sl_mult: float, tp_mult: float
) -> float:
    if losses == 0:
        return float("inf") if wins else 0.0
    return (wins * tp_mult) / (losses * sl_mult)


def _population_block(
    mask_proposed: np.ndarray,
    mask_approved: np.ndarray,
    targets: np.ndarray,
    base_rate: float,
    spec,
) -> dict:
    """Raw proposal/approval stats, before the live Gates A/B/C."""
    proposed = int(mask_proposed.sum())
    approved = int(mask_approved.sum())
    if approved:
        wins = int(targets[mask_approved].sum())
        win_rate = wins / approved
        pf = _bracket_profit_factor(
            wins, approved - wins, spec.geometry.sl_mult, spec.geometry.tp_mult
        )
        edge = win_rate - base_rate
    else:
        wins = 0
        win_rate = float("nan")
        pf = 0.0
        edge = float("nan")
    return {
        "proposed": proposed,
        "approved": approved,
        "approved_wins": wins,
        "approved_win_rate": win_rate,
        "edge_over_random": edge,
        "profit_factor": pf,
    }


def _resolve_holdout_bounds(metadata: dict):
    """The artifact's recorded holdout window, as (start, end_exclusive)."""
    holdout = (metadata or {}).get("holdout") or {}
    start_s, end_s = holdout.get("start_date"), holdout.get("end_date")
    if not start_s or not end_s:
        return None
    try:
        start = datetime.fromisoformat(start_s).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(end_s).replace(tzinfo=timezone.utc) + timedelta(
            days=1
        )
    except ValueError:
        return None
    return start, end


def _window_breakdown(
    frame_df: pl.DataFrame,
    artifact: ServedArtifact,
    backtest: LabBacktestReport,
    spec,
    angel_probs: np.ndarray,
    devil_probs: np.ndarray,
) -> List[dict]:
    """
    Population and traded results split at the artifact's recorded holdout.

    The whole frame is mostly in-sample for a served artifact (it was trained on
    it), and the only slice it never saw is the window ``metadata.json``
    records. Separating that slice is the difference between a baseline and an
    overfit reading, so it is computed automatically whenever metadata carries
    the dates.
    """
    bounds = _resolve_holdout_bounds(artifact.metadata)
    if bounds is None:
        return []
    start, end = bounds

    ts = frame_df["timestamp"]
    last_holdout_day = (end - timedelta(days=1)).date().isoformat()
    # Series comparisons produce expressions; select them back to numpy masks.
    masks = {
        f"before {start.date().isoformat()}": frame_df.select(
            ts < pl.lit(start)
        ).to_series().to_numpy(),
        f"{start.date().isoformat()} to {last_holdout_day} (recorded holdout)": (
            frame_df.select((ts >= pl.lit(start)) & (ts < pl.lit(end)))
            .to_series()
            .to_numpy()
        ),
        f"after {last_holdout_day}": frame_df.select(
            ts >= pl.lit(end)
        ).to_series().to_numpy(),
    }
    targets = frame_df["devil_target_macro"].to_numpy()
    mask_proposed = angel_probs >= artifact.angel_threshold
    mask_approved = mask_proposed & (devil_probs >= artifact.devil_threshold)

    trades = [t for rep in backtest.per_symbol.values() for t in rep.trades]
    out: List[dict] = []
    for label, window_mask in masks.items():
        rows = int(window_mask.sum())
        if rows == 0:
            continue
        if label.startswith("before "):
            window_trades = [t for t in trades if _as_utc(t.entry_timestamp) < start]
        elif label.startswith("after "):
            window_trades = [t for t in trades if _as_utc(t.entry_timestamp) >= end]
        else:
            window_trades = [
                t for t in trades if start <= _as_utc(t.entry_timestamp) < end
            ]

        base_rate = float(targets[window_mask].mean())
        block = _population_block(
            mask_proposed & window_mask,
            mask_approved & window_mask,
            targets,
            base_rate,
            spec,
        )
        trade_wins = sum(1 for t in window_trades if t.macro_win)
        net_rs = [t.net_r for t in window_trades]
        block.update(
            {
                "label": label,
                "rows": rows,
                "base_rate": base_rate,
                "trades": len(window_trades),
                "trade_wins": trade_wins,
                "trade_win_rate": (trade_wins / len(window_trades))
                if window_trades
                else float("nan"),
                "net_ev_r": float(np.mean(net_rs)) if net_rs else float("nan"),
                "gross_ev_r": float(np.mean([t.gross_r for t in window_trades]))
                if window_trades
                else float("nan"),
            }
        )
        out.append(block)
    return out


def _as_utc(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


@dataclass
class ArtifactReplayResult:
    """One served-artifact replay: frame, raw population, windows, backtest."""

    spec: object
    artifact: ServedArtifact
    frame: object
    backtest: LabBacktestReport
    population: dict
    windows: List[dict]
    frame_from_cache: bool
    run_seconds: float

    def summary(self) -> dict:
        out = {
            "name": self.spec.name,
            "content_hash": self.frame.content_hash,
            "frame_from_cache": self.frame_from_cache,
            "run_seconds": self.run_seconds,
            "rows": self.frame.n_rows,
            "feature_count": len(self.frame.feature_cols),
            "chop_veto_rate": self.frame.chop_veto_rate,
            "purged_tail_rows": self.frame.purged_tail_rows,
            "artifact": {
                "model_dir": str(self.artifact.model_dir),
                "trained_at": self.artifact.trained_at,
                "trained_on_symbols": self.artifact.trained_on_symbols,
                "angel_threshold": self.artifact.angel_threshold,
                "devil_threshold": self.artifact.devil_threshold,
                "threshold_source": self.artifact.threshold_source,
                "feature_count": len(self.artifact.angel_feature_cols),
                "angel_features": list(self.artifact.angel_feature_cols),
                "devil_features": list(self.artifact.devil_feature_cols),
            },
            "population": self.population,
            "windows": self.windows,
        }
        bt = self.backtest
        if bt is not None:
            out["backtest"] = {
                "total_trades": bt.total_trades,
                "wins": bt.wins,
                "win_rate": bt.win_rate,
                "gross_ev_r": bt.gross_ev_r,
                "net_ev_r": bt.net_ev_r,
                "profit_factor_net": bt.profit_factor_net,
                "max_drawdown_r": bt.max_drawdown_r,
                "gate_rejections": dict(bt.gate_rejections),
                "toll_mode": bt.toll_mode,
            }
        return out


def measured_alpha_table() -> Optional[Dict[str, float]]:
    """The per-instrument alphas, if the config table exists (report note)."""
    try:
        return json.loads(_SPREAD_TABLE_PATH.read_text())["alphas"]
    except (OSError, ValueError, KeyError):
        return None


def replay_served_artifact(
    spec,
    artifact: ServedArtifact,
    *,
    runner=None,
) -> ArtifactReplayResult:
    """
    Prepare the spec's frame and replay ``artifact`` over it, live-gated.

    ``runner`` is an ``ExperimentRunner`` (for the frame/bar caches); a default
    one is created when omitted.
    """
    import time

    from lab.experiments import ExperimentRunner

    started = time.monotonic()
    runner = runner or ExperimentRunner()
    frame, from_cache = runner.prepare_frame(spec)

    missing = [c for c in artifact.angel_feature_cols if c not in frame.feature_cols]
    if missing:
        raise ValueError(
            f"artifact {artifact.model_dir} needs feature(s) {missing} that spec "
            f"{spec.name!r} does not build; its schema is "
            f"{list(artifact.angel_feature_cols)}"
        )

    angel_probs, devil_probs = predict_probabilities(frame.df, artifact)
    base_rate = float(frame.df["devil_target_macro"].mean())
    population = _population_block(
        angel_probs >= artifact.angel_threshold,
        (angel_probs >= artifact.angel_threshold)
        & (devil_probs >= artifact.devil_threshold),
        frame.df["devil_target_macro"].to_numpy(),
        base_rate,
        spec,
    )
    population.update(
        {
            "rows": frame.n_rows,
            "base_rate": base_rate,
            "angel_bar": artifact.angel_threshold,
            "devil_bar": artifact.devil_threshold,
        }
    )

    backtest = run_artifact_backtest(frame, spec, artifact)
    windows = _window_breakdown(
        frame.df, artifact, backtest, spec, angel_probs, devil_probs
    )
    return ArtifactReplayResult(
        spec=spec,
        artifact=artifact,
        frame=frame,
        backtest=backtest,
        population=population,
        windows=windows,
        frame_from_cache=from_cache,
        run_seconds=time.monotonic() - started,
    )
