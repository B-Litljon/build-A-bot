#!/usr/bin/env bash
# Soak watchdog — relaunches the M15 paper soak if it isn't running.
# Runs from cron every 5 minutes; cron survives reboots, so this closes the
# gap that killed the 2026-07-02 soak (kernel-update reboot on 07-08 left it
# down for 3 days unnoticed).
#
# To stop the soak WITHOUT the watchdog bringing it back:
#   touch soak.off && kill "$(cat /tmp/soak.pid)"
# Remove soak.off to re-arm. To retire the watchdog entirely, delete its
# crontab line (crontab -e).
#
# Guards, in order:
#   1. soak.off kill switch          -> do nothing
#   2. soak already running          -> do nothing
#   3. before first-launch gate      -> wait (Sun 2026-07-12 14:05 PT market open)
#   4. weekend blackout              -> no cold starts Fri 14:00 - Sun 14:05 PT
#                                       (forex closed; a RUNNING soak is left alone)
#   5. crash-loop brake              -> at most one launch per 15 min
#
# Test knobs: WATCHDOG_NOW=<epoch> fakes the clock; DRY_RUN=1 logs the
# decision instead of launching. `soak_watchdog.sh selftest` just logs a
# line and exits (used to prove cron can execute this script).
set -euo pipefail
REPO=/mnt/storage/mystuf/development/build-A-bot
cd "$REPO"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') [watchdog] $*"; }

if [ "${1:-}" = "selftest" ]; then
  log "selftest: cron executed this script OK"
  exit 0
fi

# 1. kill switch
if [ -e "$REPO/soak.off" ]; then
  exit 0
fi

# 2. already running? (tight pattern: the venv python running run_oanda.py,
#    so an editor or grep with the filename in its argv doesn't count)
if pgrep -f 'bin/python -u run_oanda\.py' > /dev/null 2>&1; then
  exit 0
fi

NOW="${WATCHDOG_NOW:-$(date +%s)}"

# 3. don't launch before the scheduled Sunday relaunch (machine TZ is PT)
NOT_BEFORE=$(date -d '2026-07-12 14:05' +%s)
if [ "$NOW" -lt "$NOT_BEFORE" ]; then
  exit 0
fi

# 4. weekend blackout — forex is closed Fri 14:00 PT -> Sun 14:00 PT (tracks
#    5pm New York year-round since ET/PT shift DST together). Don't cold-start
#    into a closed market; retry ticks land at Sun 14:05.
DOW=$(date -d "@$NOW" +%u)              # 1=Mon .. 7=Sun
HHMM=$((10#$(date -d "@$NOW" +%H%M)))
if { [ "$DOW" -eq 5 ] && [ "$HHMM" -ge 1400 ]; } || [ "$DOW" -eq 6 ] \
   || { [ "$DOW" -eq 7 ] && [ "$HHMM" -lt 1405 ]; }; then
  exit 0
fi

# 5. crash-loop brake
STATE=/tmp/soak_watchdog_last_launch
if [ -f "$STATE" ]; then
  LAST=$(cat "$STATE")
  if [ $((NOW - LAST)) -lt 900 ]; then
    log "soak is down but last launch was $((NOW - LAST))s ago — holding off (possible crash loop, check newest logs/soak_*.log)"
    exit 0
  fi
fi

log "soak not running — launching M15 soak (OANDA_MODEL_DIR=models/forex_m15, granularity 15)"
if [ "${DRY_RUN:-0}" = "1" ]; then
  log 'DRY_RUN=1 — would run: setsid --fork env OANDA_MODEL_DIR=models/forex_m15 bash run_soak.sh "" 15'
  exit 0
fi
echo "$NOW" > "$STATE"
setsid --fork env OANDA_MODEL_DIR=models/forex_m15 bash "$REPO/run_soak.sh" "" 15
sleep 5
if pgrep -f 'bin/python -u run_oanda\.py' > /dev/null 2>&1; then
  log "launch OK — pid $(cat /tmp/soak.pid 2>/dev/null || echo '?'), log $(cat /tmp/soak_logpath 2>/dev/null || echo '?')"
else
  log "LAUNCH FAILED — no soak process 5s after start; check $(cat /tmp/soak_logpath 2>/dev/null || echo 'logs/soak_*.log')"
fi
