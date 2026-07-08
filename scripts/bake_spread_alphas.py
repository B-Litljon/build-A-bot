#!/usr/bin/env python
"""
Bake a per-instrument spread-alpha table from a soak log's SPREAD_CALIB lines.

The live orchestrator (execution.oanda_scalper_orchestrator) periodically logs
empirical spread calibration per instrument:

    SPREAD_CALIB XAU_USD | n=294 med_spread_pct=0.01481 [p25=... p75=...] \
        med_baseline_natr=0.20738 alpha_emp=0.0714

alpha_emp = median(spread_pct) / median(baseline_natr) — the spread cost as a
fraction of a typical move.  This script takes the LAST (largest-n) line per
symbol and writes a JSON table consumed by:

  * the retrainer (RETRAIN_SPREAD_TABLE env) — per-instrument chop-veto alphas
    and the V3CostFeatures ``cost_ratio`` feature;
  * live inference — copied into the model dir as ``spread_alphas.json`` on
    gate pass, loaded by MLStrategy / run_oanda.

``denomination_minutes`` records the bar size the alphas were measured on:
baseline NATR is timeframe-dependent, so an M15-denominated table must never
silently price an M1 retrain (the retrainer warns on mismatch).

Usage:
    PYTHONPATH=src:. python scripts/bake_spread_alphas.py \
        logs/soak_2026-07-02_0215.log config/spread_alphas_m15.json \
        --denomination-minutes 15 [--min-n 120]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Matches the orchestrator's _log_spread_calibration format.
_CALIB_RE = re.compile(
    r"SPREAD_CALIB (\S+) \| n=(\d+) .*alpha_emp=([\d.]+|nan)"
)

DEFAULT_ALPHA = 0.15  # RiskProfile.spread_atr_alpha fallback for unlisted symbols


def bake(log_path: Path, min_n: int) -> tuple[dict[str, float], dict[str, int]]:
    """Return ({symbol: alpha_emp}, {symbol: n}) from the last line per symbol."""
    alphas: dict[str, float] = {}
    samples: dict[str, int] = {}
    with open(log_path, "r") as fh:
        for line in fh:
            m = _CALIB_RE.search(line)
            if not m:
                continue
            sym, n, alpha = m.group(1), int(m.group(2)), m.group(3)
            if alpha == "nan":
                continue
            # Later lines have larger n (deques only grow); last one wins.
            alphas[sym] = float(alpha)
            samples[sym] = n

    thin = {s: n for s, n in samples.items() if n < min_n}
    if thin:
        raise SystemExit(
            f"Refusing to bake: sample count below --min-n {min_n} for {thin}. "
            "Let the soak collect more SPREAD_CALIB samples first."
        )
    if not alphas:
        raise SystemExit(f"No SPREAD_CALIB lines found in {log_path}")
    return alphas, samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="soak log with SPREAD_CALIB lines")
    parser.add_argument("out", type=Path, help="output JSON path")
    parser.add_argument(
        "--denomination-minutes", type=int, required=True,
        help="bar size (minutes) the soak ran on — alphas are only valid there",
    )
    parser.add_argument(
        "--min-n", type=int, default=120,
        help="refuse symbols with fewer calibration samples (default 120)",
    )
    args = parser.parse_args()

    alphas, samples = bake(args.log, args.min_n)

    table = {
        "denomination_minutes": args.denomination_minutes,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_log": str(args.log),
        "default_alpha": DEFAULT_ALPHA,
        "samples": dict(sorted(samples.items())),
        "alphas": dict(sorted(alphas.items())),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(table, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {args.out} ({len(alphas)} symbols):")
    for sym, alpha in sorted(alphas.items()):
        print(f"  {sym:<10} alpha={alpha:.4f}  (n={samples[sym]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
