"""
Rebuild the live-decision report: grade every decision the soak recorded
against what price actually did next.

Driver for ``analysis.decision_grader`` — the grader itself is a library, so
this script is the thing that actually runs it end to end:

  1. collect every ``ev="bar"`` record from logs/events-*.jsonl
  2. build the answer key by replaying the production bracket walk
     (``retrainer._compute_devil_targets_atr``, SL 2.0x / TP 4.0x / hold 45,
     SL-checked-first, timeout = loss) over the cached M15 basket
  3. join, then emit calibration bands, a net-of-cost threshold sweep and a
     per-behavior breakdown, to stdout and to logs/decision_report_<date>.txt
  4. persist the row-level frame to logs/graded_decisions.parquet (atomic)

The bracket-walk labeler is LONG-only (mirrors how the retrainer labels the
macro target); the 2026-09-06 report used the same convention, so numbers are
comparable across reports. Direction-split grading would be a separate,
larger change to the grader itself.

Glossary:
    decisions -- every bar evaluation the soak recorded, from telemetry.
    answer_key -- per-bar won/lost frame from the bracket walk over cached
        bars; the ground truth the decisions are graded against. The walk runs
        over the full frame (it needs the contiguous price path); the last
        LOOKAHEAD_BARS bars per symbol come out censored (dense-labeler
        default 0), and the grader purges them before joining.
    graded -- decisions joined to the answer key; rows the 45-bar walk could
        not resolve (too close to the end of history) are dropped, never
        invented.
    GRADED_OUT -- logs/graded_decisions.parquet, the row-level artifact the
        reports read back.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.decision_grader import (  # noqa: E402
    behavior_breakdown,
    calibration_table,
    grade_decisions,
    load_decisions,
    threshold_sweep,
)
from analysis.build_strategy_matrix import load_basket, prepare_tagged_frame  # noqa: E402
from core.retrainer import _compute_devil_targets_atr  # noqa: E402

GRADED_OUT = Path("logs/graded_decisions.parquet")
SL_MULT, TP_MULT, MAX_HOLD = 2.0, 4.0, 45
TOLL_R = 0.10

HEADER = (
    "LIVE DECISION REPORT — {when}\n"
    "decisions recorded : {n_dec:,}\n"
    "span               : {span}\n"
    "graded             : {n_graded:,}   (SL={sl}xATR TP={tp}xATR hold={hold} "
    "htf=1h, long-walk convention)\n"
    "base rate (random) : {base:.1f}%   break-even needs {be:.1f}%\n"
)


def main() -> int:
    decisions = load_decisions("logs/events-*.jsonl")
    if decisions.height == 0:
        print("no telemetry found; nothing to grade", file=sys.stderr)
        return 1

    frames = load_basket(
        ["AUD_JPY", "EUR_JPY", "GBP_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD"],
        days_back=730,
        granularity=15,
        cache_dir=Path("analysis_cache/strategy_matrix"),
    )
    tagged = pl.concat(
        [prepare_tagged_frame(df, sym) for sym, df in frames.items()],
        how="vertical_relaxed",
    )
    # The walk runs over the FULL frame (the bracket path must stay
    # contiguous or resolvable bars would newly truncate); grade_decisions
    # purges the censored tail (last LOOKAHEAD_BARS per symbol, which the
    # dense labeler defaulted to 0) before it joins.
    tagged = tagged.with_columns(
        pl.Series("won", _compute_devil_targets_atr(tagged, SL_MULT, TP_MULT, MAX_HOLD))
    )
    graded = grade_decisions(decisions, tagged)
    if graded.height == 0:
        print("no decisions could be graded", file=sys.stderr)
        return 1

    calib = calibration_table(graded)
    sweep = threshold_sweep(graded, SL_MULT, TP_MULT, TOLL_R)
    behavior = behavior_breakdown(graded)
    base = float(graded["won"].mean())
    breakeven = 1.0 / (1.0 + TP_MULT / SL_MULT)

    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    header = HEADER.format(
        when=when,
        n_dec=decisions.height,
        span=f"{decisions['bar_ts'].min()} -> {decisions['bar_ts'].max()}",
        n_graded=graded.height,
        sl=SL_MULT, tp=TP_MULT, hold=MAX_HOLD,
        base=100 * base, be=100 * breakeven,
    )

    with pl.Config(tbl_rows=50, tbl_width_chars=120):
        body = "\n".join(
            [
                header,
                "CALIBRATION — does confidence mean anything?",
                str(calib), "",
                f"THRESHOLD SWEEP (net of {TOLL_R}R toll)",
                str(sweep), "",
                "BY MARKET BEHAVIOR",
                str(behavior), "",
                f"row-level data: {GRADED_OUT}",
            ]
        )
    print(body)

    report_path = Path(f"logs/decision_report_{datetime.now(timezone.utc):%Y-%m-%d}.txt")
    report_path.write_text(body + "\n")

    # Atomic: a reader (or a rerun) must never see a half-written parquet.
    fd, tmp = tempfile.mkstemp(dir=GRADED_OUT.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        pl.DataFrame(graded).write_parquet(tmp)
        os.replace(tmp, GRADED_OUT)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    print(f"\nwrote {report_path} and {GRADED_OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())