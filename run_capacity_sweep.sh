#!/usr/bin/env bash
# Does a smaller model generalize better?
#
#   bash run_capacity_sweep.sh
#
# Trains 5 model variants on identical data at 3 pinned windows and reports
# which ones actually clear the honest holdout gate. Takes ~15 minutes.
#
#   SWEEP_PINS=2026-08-23 bash run_capacity_sweep.sh   # one window, ~5 min
#
# Read-only: trains in memory, writes NO model, never touches the live bot.
# Output: logs/capacity_sweep.txt (readable) + logs/capacity_sweep.json (data).
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
export PYTHONPATH=src:.
export DATA_SOURCE="${DATA_SOURCE:-oanda}"
export RETRAIN_TIMEFRAME_MINUTES="${RETRAIN_TIMEFRAME_MINUTES:-15}"
export RETRAIN_DAYS_BACK="${RETRAIN_DAYS_BACK:-730}"
VENV=/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python
mkdir -p logs
"$VENV" scripts/capacity_sweep.py 2>&1 | tee logs/capacity_sweep.txt
echo
echo "readable report : logs/capacity_sweep.txt"
echo "data for plots  : logs/capacity_sweep.json"
