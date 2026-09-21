#!/usr/bin/env bash
# V5 OANDA forex paper soak — runs the promoted forex bot on the practice
# account and accumulates per-instrument spread samples so we can derive an
# empirical spread_atr_alpha (grep SPREAD_CALIB in the log).
#
#   bash run_soak.sh                      # full trained basket, M1, practice
#   bash run_soak.sh XAU_USD,XAG_USD      # restricted basket (e.g. metals-only)
#   bash run_soak.sh "" 15                # full basket on M15 bars
#   SOAK_GRANULARITY=15 bash run_soak.sh  # same, via env
#
# Stop with: kill "$(cat /tmp/soak.pid)"  (flattens positions on SIGTERM).
set -euo pipefail
cd /mnt/storage/mystuf/development/build-A-bot
mkdir -p logs

# Log hygiene on every launch (2026-09-17): gzip soak logs older than 30 days,
# drop gzips older than 180 days. Runs before the new live log is created, so
# the file this run is about to write is never eligible. A chatty single run
# once produced a 427 MB log (soak_2026-08-16_1405.log.gz) — this bounds that.
find logs -maxdepth 1 -name 'soak_*.log' -mtime +30 -exec gzip -9 {} + 2>/dev/null || true
find logs -maxdepth 1 -name 'soak_*.log.gz' -mtime +180 -delete 2>/dev/null || true

LOG="logs/soak_$(date +%Y-%m-%d_%H%M).log"
echo "$LOG" > /tmp/soak_logpath
echo "$$" > /tmp/soak.pid
set -a; source .env; set +a
export PYTHONPATH=src:.
VENV=/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python
SYMBOLS="${1:-}"
GRANULARITY="${2:-${SOAK_GRANULARITY:-1}}"
ARGS=(--daemon --env practice --granularity "$GRANULARITY")
if [ -n "$SYMBOLS" ]; then
  ARGS+=(--symbols "$SYMBOLS")
  echo "Launching V5 soak (symbols=$SYMBOLS granularity=${GRANULARITY}m) -> $LOG"
else
  echo "Launching V5 soak (full trained basket, granularity=${GRANULARITY}m) -> $LOG"
fi
exec "$VENV" -u run_oanda.py "${ARGS[@]}" > "$LOG" 2>&1
