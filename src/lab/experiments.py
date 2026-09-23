"""ExperimentRunner -- spec -> frame -> gate -> backtest, with a frame cache.

The loop the feature lab exists for. A run is four steps:

    1. load bars (cached parquet basket),
    2. build or load the frame keyed by the spec's content hash,
    3. run the retrainer's promotion gate,
    4. replay the gate's models through the live-gated backtester.

Only step 2 is cached, and only on a content hash that covers the spread table
bytes and the frame-affecting environment (see spec.content_hash), so a stale
hit is impossible by construction: change a lookback, a veto switch, or the
cost table and the key changes.

Glossary:
    DEFAULT_FRAME_CACHE -- analysis_cache/lab_frames/; one parquet + one JSON
        sidecar per content hash.
    ExperimentResult -- everything one run produced, plus whether the frame
        came from cache and how long the gate took.
    ExperimentRunner -- holds the cache/refresh switches so a batch of specs
        shares one bar load.
    prepare_frame -- the frame half of a run, shared with the artifact replay:
        cached load if the content hash matches, otherwise build + atomic save.
    _load_cached_frame -- a cache hit requires BOTH the parquet and a sidecar
        whose hash matches; a parquet alone is ignored (it could be half of an
        interrupted write).
    _save_frame -- atomic temp+rename for both files, matching the repo's
        model-artifact convention.
    summary -- the machine-readable run result used by the CLI and report.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import polars as pl

from lab.backtest import LabBacktestReport, run_model_backtest
from lab.data import DEFAULT_CACHE_DIR, load_bars
from lab.frames import FrameResult, build_frame
from lab.gate import GateResult, run_gate

DEFAULT_FRAME_CACHE = Path("analysis_cache/lab_frames")


@dataclass
class ExperimentResult:
    """One completed lab run."""

    spec: object
    frame: FrameResult
    gate: GateResult
    backtest: Optional[LabBacktestReport]
    frame_from_cache: bool
    run_seconds: float

    def summary(self) -> dict:
        report = self.gate.report
        out = {
            "name": self.spec.name,
            "content_hash": self.frame.content_hash,
            "model_family": self.gate.model_family,
            "frame_from_cache": self.frame_from_cache,
            "rows": self.frame.n_rows,
            "feature_count": len(self.frame.feature_cols),
            "chop_veto_rate": self.frame.chop_veto_rate,
            "purged_tail_rows": self.frame.purged_tail_rows,
            "gate_passed": report.gate_passed,
            "rejection_reasons": list(report.rejection_reasons),
            "mean_brier": report.mean_brier,
            "mean_ev": report.mean_ev,
            "pooled_oos_trades": report.pooled_oos_trades,
            "pooled_oos_wins": report.pooled_oos_wins,
            "pooled_pf_lower_bound": report.pooled_pf_lower_bound,
            "fold3_pf_lower_bound": report.fold3_pf_lower_bound,
            "pooled_base_rate": report.pooled_base_rate,
            "edge_over_random": report.edge_over_random,
            "production_angel_threshold": report.production_angel_threshold,
            "production_devil_threshold": self.gate.production_threshold,
            "run_seconds": self.run_seconds,
            "folds": [
                {
                    "fold": fm.fold_number,
                    "train_size": fm.train_size,
                    "val_size": fm.val_size,
                    "brier": fm.brier_score,
                    "ev": fm.expected_value,
                    "angel_proposed": fm.angel_proposed_trades,
                    "devil_approved": fm.devil_approved_trades,
                    "win_rate": fm.win_rate,
                    "macro_wins": fm.macro_wins,
                    "base_rate": fm.base_rate,
                }
                for fm in report.fold_metrics
            ],
        }
        if self.backtest is not None:
            bt = self.backtest
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


def _frame_paths(cache_dir: Path, content_hash: str) -> Tuple[Path, Path]:
    return cache_dir / f"{content_hash}.parquet", cache_dir / f"{content_hash}.json"


def _load_cached_frame(cache_dir: Path, spec) -> Optional[FrameResult]:
    parquet_path, sidecar_path = _frame_paths(cache_dir, spec.content_hash())
    if not (parquet_path.is_file() and sidecar_path.is_file()):
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (OSError, ValueError):
        return None
    if sidecar.get("content_hash") != spec.content_hash():
        return None
    if sidecar.get("spec_name") != spec.name:
        return None
    try:
        df = pl.read_parquet(parquet_path)
    except Exception:
        return None
    table = spec.alpha_table()
    return FrameResult(
        spec_name=spec.name,
        content_hash=sidecar["content_hash"],
        df=df,
        feature_cols=tuple(sidecar["feature_cols"]),
        chop_veto_rate=float(sidecar["chop_veto_rate"]),
        alpha_table=table,
        purged_tail_rows=int(sidecar.get("purged_tail_rows", 0)),
    )


def _save_frame(cache_dir: Path, spec, frame: FrameResult) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path, sidecar_path = _frame_paths(cache_dir, frame.content_hash)
    tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
    tmp_sidecar = sidecar_path.with_suffix(".json.tmp")
    frame.df.write_parquet(tmp_parquet)
    tmp_sidecar.write_text(
        json.dumps(
            {
                "content_hash": frame.content_hash,
                "spec_name": frame.spec_name,
                "feature_cols": list(frame.feature_cols),
                "chop_veto_rate": frame.chop_veto_rate,
                "purged_tail_rows": frame.purged_tail_rows,
                "rows": frame.n_rows,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    os.replace(tmp_parquet, parquet_path)
    os.replace(tmp_sidecar, sidecar_path)


class ExperimentRunner:
    """Runs specs end to end, sharing the bar cache across a batch."""

    def __init__(
        self,
        *,
        bar_cache_dir: Path = DEFAULT_CACHE_DIR,
        frame_cache_dir: Path = DEFAULT_FRAME_CACHE,
        refresh_bars: bool = False,
        use_frame_cache: bool = True,
        do_backtest: bool = True,
    ) -> None:
        self.bar_cache_dir = Path(bar_cache_dir)
        self.frame_cache_dir = Path(frame_cache_dir)
        self.refresh_bars = refresh_bars
        self.use_frame_cache = use_frame_cache
        self.do_backtest = do_backtest
        self._bars: Optional[Dict[str, pl.DataFrame]] = None

    def _bars_for(self, spec) -> Dict[str, pl.DataFrame]:
        if self._bars is None:
            self._bars = load_bars(
                spec, cache_dir=self.bar_cache_dir, refresh=self.refresh_bars
            )
        return self._bars

    def prepare_frame(self, spec) -> Tuple[FrameResult, bool]:
        """Build or load the spec's frame; returns ``(frame, from_cache)``.

        Public because the served-artifact replay needs the identical frame
        contract without running the gate: same cache key, same atomic write.
        """
        bars = self._bars_for(spec)

        frame: Optional[FrameResult] = None
        from_cache = False
        if self.use_frame_cache:
            frame = _load_cached_frame(self.frame_cache_dir, spec)
            from_cache = frame is not None
        if frame is None:
            frame = build_frame(spec, bars)
            if self.use_frame_cache:
                _save_frame(self.frame_cache_dir, spec, frame)
        return frame, from_cache

    def run(self, spec) -> ExperimentResult:
        started = time.monotonic()
        frame, from_cache = self.prepare_frame(spec)

        gate = run_gate(frame, spec)
        backtest = run_model_backtest(frame, gate, spec) if self.do_backtest else None
        return ExperimentResult(
            spec=spec,
            frame=frame,
            gate=gate,
            backtest=backtest,
            frame_from_cache=from_cache,
            run_seconds=time.monotonic() - started,
        )
