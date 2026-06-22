#!/usr/bin/env bash
set -euo pipefail
cd /mnt/storage/mystuf/development/build-A-bot
LOG="logs/chop_ab_$(date +%Y-%m-%d_%H%M).log"
echo "$LOG" > /tmp/chop_ab_logpath
set -a; source .env; set +a
export DATA_SOURCE=oanda RETRAIN_DAYS_BACK=365 PYTHONPATH=src:.
VENV=/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python
echo "Launching A/B to $LOG"
exec "$VENV" -u chop_ab_test.py > "$LOG" 2>&1
