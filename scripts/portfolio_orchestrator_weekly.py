"""Lane-2 WEEKLY portfolio orchestrator — RESEARCH ONLY, not scheduled.

A parallel to ``scripts/portfolio_orchestrator.py`` (the live monthly V4 lane,
which is production and untouched). This weekly variant ranks the same 96-name
universe with the **weekly** model (``models/v4_investor_weekly_lgbm.txt``),
built by the Lane-2 weekly feature pipeline + trainer at a 5-trading-day hold.

Reuses the production lane's Alpaca execution helpers
(``_build_alpaca_clients`` / ``execute_rebalance`` / ``latest_per_symbol``) by
import so the order path cannot drift from the reviewed one. It does NOT
install a crontab, is NOT wired into ``run_investor_rebalance.sh``, and a gate
REJECT from the weekly trainer leaves no model artifact, in which case this
script refuses to run.

Because the weekly result is a **research finding with near-zero per-name
ranking skill** (per-fold Spearman IC ~ 0.03; see
``llm_reports/recons/2026-09-24_lab-equity-factor-pead.md``), the default mode
is ``--dry-run`` and a live paper rebalance is opt-in via ``--live``.

Invocation:
    PYTHONPATH=src:. python scripts/portfolio_orchestrator_weekly.py            # dry run
    PYTHONPATH=src:. python scripts/portfolio_orchestrator_weekly.py --skip-refresh
    PYTHONPATH=src:. python scripts/portfolio_orchestrator_weekly.py --live     # paper orders

Glossary:
    TOP_K / SECTOR_CAP / TARGET_WEIGHT -- 8 / 2 / 12.5%, identical to the
        monthly lane so the two are comparable (see
        scripts/portfolio_orchestrator.py Glossary).
    _WEEKLY_MODEL -- models/v4_investor_weekly_lgbm.txt; written only when the
        weekly trainer's falsification gate PASSES. Its absence means "no
        promotable weekly model", which this orchestrator honors by refusing.
    _WEEKLY_INFER -- data/processed/v4_weekly_inference_features.parquet from
        investor_feature_pipeline_weekly.py --inference (embargo rows retained
        so today's row survives, mirroring the monthly --inference mode).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

load_dotenv(_PROJECT_ROOT / ".env")

from investor_universe import UNIVERSE  # noqa: E402
# Reuse the reviewed production execution path verbatim (read-only import).
from portfolio_orchestrator import (  # noqa: E402
    SECTOR_CAP,
    TARGET_WEIGHT,
    TOP_K,
    _build_alpaca_clients,
    execute_rebalance,
    latest_per_symbol,
    predict_and_rank,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("portfolio_orchestrator_weekly")

_WEEKLY_MODEL = _PROJECT_ROOT / "models" / "v4_investor_weekly_lgbm.txt"
_WEEKLY_INFER = _PROJECT_ROOT / "data" / "processed" / "v4_weekly_inference_features.parquet"
_FEATURE_SCRIPT = _PROJECT_ROOT / "scripts" / "investor_feature_pipeline_weekly.py"


def _refresh_inference_features() -> None:
    if not _FEATURE_SCRIPT.exists():
        raise FileNotFoundError(_FEATURE_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(_FEATURE_SCRIPT), "--inference"],
        cwd=str(_PROJECT_ROOT), check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"weekly feature pipeline exited {proc.returncode}")
    if not _WEEKLY_INFER.exists():
        raise FileNotFoundError(f"weekly inference features not produced: {_WEEKLY_INFER}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Lane-2 WEEKLY portfolio orchestrator (research).")
    p.add_argument("--live", action="store_true",
                   help="submit paper orders (default is a dry run)")
    p.add_argument("--skip-refresh", action="store_true",
                   help="reuse existing v4_weekly_inference_features.parquet")
    args = p.parse_args(argv)
    dry_run = not args.live

    logger.info("=" * 70)
    logger.info("Lane-2 WEEKLY Portfolio Orchestrator (RESEARCH ONLY)")
    logger.info("mode=%s | top_k=%d cap=%d weight=%.1f%%",
                "DRY-RUN" if dry_run else "LIVE (paper)", TOP_K, SECTOR_CAP, TARGET_WEIGHT * 100)
    logger.info("=" * 70)

    try:
        if not _WEEKLY_MODEL.exists():
            raise FileNotFoundError(
                f"{_WEEKLY_MODEL} missing — the weekly trainer only writes it on a "
                "GATE PASS. Run scripts/investor_train_model_weekly.py; absent model "
                "means no promotable weekly model exists."
            )
        if not args.skip_refresh:
            _refresh_inference_features()
        elif not _WEEKLY_INFER.exists():
            raise FileNotFoundError(f"{_WEEKLY_INFER} missing — drop --skip-refresh")

        booster = lgb.Booster(model_file=str(_WEEKLY_MODEL))
        snapshot = latest_per_symbol(_WEEKLY_INFER, UNIVERSE)
        top_k, _ranked = predict_and_rank(booster, snapshot, TOP_K)

        if dry_run:
            # Research dry-run: the ranking IS the deliverable. Production's
            # execute_rebalance builds live Alpaca clients even in dry-run, so
            # we short-circuit before it rather than require credentials.
            logger.info(
                "DRY-RUN complete — would rebalance to top-%d: %s (no Alpaca "
                "credentials needed; pass --live on a funded paper account to trade).",
                TOP_K, top_k,
            )
            return 0

        trading, data_client = _build_alpaca_clients()
        execute_rebalance(trading, data_client, top_k, dry_run=dry_run)
        logger.info("weekly orchestrator complete (dry_run=%s).", dry_run)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("weekly orchestrator failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
