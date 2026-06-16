#!/usr/bin/env bash
# V5 OANDA forex paper soak — runs the promoted scalper on the practice
# account and accumulates per-instrument spread samples so we can derive an
# empirical spread_atr_alpha (grep SPREAD_CALIB in the log).
#
#   bash run_soak.sh            # trained basket, M1, practice
#
# Stop with: kill "$(cat /tmp/soak.pid)"  (flattens positions on SIGTERM).
set -euo pipefail
cd /mnt/storage/mystuf/development/build-A-bot
mkdir -p logs
LOG="logs/soak_$(date +%Y-%m-%d_%H%M).log"
echo "$LOG" > /tmp/soak_logpath
echo "$$" > /tmp/soak.pid
set -a; source .env; set +a
export PYTHONPATH=src:.
VENV=/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python
echo "Launching V5 soak (spread calibration) -> $LOG"
exec "$VENV" -u run_oanda.py --daemon --env practice > "$LOG" 2>&1
