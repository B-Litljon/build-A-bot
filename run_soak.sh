#!/usr/bin/env bash
# V5 OANDA forex paper soak — runs the promoted scalper on the practice
# account and accumulates per-instrument spread samples so we can derive an
# empirical spread_atr_alpha (grep SPREAD_CALIB in the log).
#
#   bash run_soak.sh                      # full trained basket, M1, practice
#   bash run_soak.sh XAU_USD,XAG_USD      # restricted basket (e.g. metals-only)
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
SYMBOLS="${1:-}"
if [ -n "$SYMBOLS" ]; then
  echo "Launching V5 soak (symbols=$SYMBOLS) -> $LOG"
  exec "$VENV" -u run_oanda.py --daemon --env practice --symbols "$SYMBOLS" > "$LOG" 2>&1
fi
echo "Launching V5 soak (full trained basket) -> $LOG"
exec "$VENV" -u run_oanda.py --daemon --env practice > "$LOG" 2>&1
