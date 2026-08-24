#!/usr/bin/env bash
# Grade the live bot's recorded decisions against what price actually did.
#
# The bot scores ~500 bars a day and writes each verdict to logs/events-*.jsonl,
# but only ~1 a week becomes a fill. Judging the model on fills alone throws
# away 99.9% of the evidence it has already produced. This refetches the bars
# that followed each decision, walks the same ATR bracket the model was trained
# against, and reports how the SERVED artifact actually performed.
#
# READ-ONLY with respect to the live bot: it reads logs and broker history and
# writes a report. It never touches models/, never places an order, and does
# not import anything the live process uses.
#
#   bash run_decision_grader.sh            # report to logs/decision_report_<date>.txt
#   GRADER_DAYS=90 bash run_decision_grader.sh
#
# Install weekly (Sundays 12:00 PT, before the market reopens at 14:05):
#   0 12 * * 0 /mnt/storage/mystuf/development/build-A-bot/run_decision_grader.sh
set -euo pipefail
cd /mnt/storage/mystuf/development/build-A-bot

set -a; source .env; set +a
export PYTHONPATH=src:.
export DATA_SOURCE="${DATA_SOURCE:-oanda}"
export RETRAIN_TIMEFRAME_MINUTES="${RETRAIN_TIMEFRAME_MINUTES:-15}"
VENV=/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python

mkdir -p logs
OUT="logs/decision_report_$(date +%Y-%m-%d).txt"

"$VENV" - "$OUT" <<'PY' | tee "$OUT"
import sys, logging, datetime
sys.path.insert(0, "src")
logging.basicConfig(level=logging.WARNING, format="%(message)s")
import polars as pl
from core import retrainer as R
from data.factory import get_market_provider
from analysis.decision_grader import (load_decisions, grade_decisions,
                                      calibration_table, threshold_sweep,
                                      behavior_breakdown)
from analysis.behavior_matrix import tag_frame
import os

days = int(os.getenv("GRADER_DAYS", "60"))
dec = load_decisions()
print(f"LIVE DECISION REPORT — {datetime.datetime.now():%Y-%m-%d %H:%M}")
print(f"decisions recorded : {dec.height:,}")
if dec.height == 0:
    print("no decisions found — is the soak running with telemetry enabled?")
    raise SystemExit(0)
print(f"span               : {dec['bar_ts'].min()[:16]} -> {dec['bar_ts'].max()[:16]}")

cfg = R.get_asset_config("oanda")
syms = sorted(dec["symbol"].unique().to_list())
raw = R.fetch_training_data(get_market_provider(), symbols=syms, days_back=days,
                            timeframe_minutes=cfg["timeframe_minutes"])
feats, _c, _r = R.engineer_features_and_labels(
    raw, sl_mult=cfg["sl_mult"], angel_mult=cfg["angel_mult"], tp_mult=cfg["tp_mult"],
    max_hold=cfg["max_hold"], survival_bars=cfg["survival_bars"],
    htf_timeframe=cfg["htf_timeframe"],
    risk_profile=None,   # no chop veto: we want an answer key for EVERY bar
    alpha_table=None)
feats = tag_frame(feats)

g = grade_decisions(dec, feats.select([
    pl.col("symbol"), pl.col("timestamp"),
    pl.col("devil_target_macro").cast(pl.Int8).alias("won"),
    pl.col("behavior_label")]))

print(f"graded             : {g.height:,}")
print(f"brackets           : SL={cfg['sl_mult']}xATR TP={cfg['tp_mult']}xATR "
      f"hold={cfg['max_hold']} htf={cfg['htf_timeframe']}")
print(f"base rate (random) : {float(g['won'].mean()):.1%}   break-even needs "
      f"{1/(1+cfg['tp_mult']/cfg['sl_mult']):.1%}\n")
print("CALIBRATION — does confidence mean anything?")
print(calibration_table(g))
print("\nTHRESHOLD SWEEP (net of 0.10R toll)")
print(threshold_sweep(g, cfg["sl_mult"], cfg["tp_mult"], 0.10))
print("\nBY MARKET BEHAVIOR")
print(behavior_breakdown(g))
g.write_parquet("logs/graded_decisions.parquet")
print("\nrow-level data: logs/graded_decisions.parquet")
PY

echo "report written to $OUT"
