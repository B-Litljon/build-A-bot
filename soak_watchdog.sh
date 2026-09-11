#!/usr/bin/env bash
# Soak watchdog — relaunches the M15 paper soak if it isn't running.
# Runs from cron every 5 minutes; cron survives reboots, so this closes the
# gap that killed the 2026-07-02 soak (kernel-update reboot on 07-08 left it
# down for 3 days unnoticed).
#
# The soak itself runs as the systemd user service soak.service (see that
# file). This script decides WHEN to start it; systemd runs it and records how
# it died. Handy commands:
#   systemctl --user status soak.service        # running? how did it exit?
#   journalctl --user -u soak.service           # the exit record over time
#
# To stop the soak WITHOUT the watchdog bringing it back:
#   touch soak.off && systemctl --user stop soak.service
# (SIGTERM flattens open positions; systemd waits up to 90s for that.)
# Remove soak.off to re-arm. To retire the watchdog entirely, delete its
# crontab line (crontab -e).
#
# Guards, in order:
#   1. soak.off kill switch          -> do nothing
#   2. soak already running          -> staleness check, then do nothing
#   3. weekend blackout              -> no cold starts Fri 14:00 - Sun 14:05 PT
#                                       (forex closed; a RUNNING soak is left alone)
#   4. crash-loop brake              -> at most one launch per 15 min
#
# Test knobs: WATCHDOG_NOW=<epoch> fakes the clock; DRY_RUN=1 logs the
# decision instead of launching. `soak_watchdog.sh selftest` just logs a
# line and exits (used to prove cron can execute this script).
set -euo pipefail
REPO=/mnt/storage/mystuf/development/build-A-bot
cd "$REPO"

# The soak runs as a systemd user service so that a death is *recorded*
# (exit code or signal) instead of the process just vanishing, which is what
# happened six times in twelve days under the old `setsid` launch. The served
# model is declared in soak.service — that file is now the single source of
# truth for it, and carries the bracket-matching warning.
#
# cron gives us no session bus, so point systemctl at the user manager
# ourselves. Lingering must be on (`loginctl enable-linger`) or there is no
# user manager to talk to when the desktop session is closed.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
SYSTEMCTL=(systemctl --user)
UNIT=soak.service

# Staleness check (added 2026-09-09): pgrep proves the PROCESS is alive, not
# that it is producing bars. A wedged event loop with open positions satisfies
# pgrep forever while its software stops watch nothing — the worst failure
# CLAUDE.md names. status.json is rewritten every M15 bar; older than
# STALE_SECS means at least one bar was missed. Ticks are 5 min apart, so
# STALE_TICKS consecutive stale ticks escalate to a restart (SIGTERM flattens,
# boot reconcile adopts whatever the broker holds).
STALE_SECS=${STALE_SECS:-1200}   # 20 min = one missed bar + buffer
STALE_TICKS=${STALE_TICKS:-3}    # ~15 min of provable silence -> restart
STALE_STATE="$REPO/logs/soak_watchdog_stale"
NTFY_URL=${NTFY_URL:-http://127.0.0.1:8091/Reminders}
ntfy() {
  curl -sf -m 5 -H "Title: $1" -H "Priority: $2" -H "Tags: $3" \
    -d "$4" "$NTFY_URL" >/dev/null 2>&1 || true
}

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
  NOW="${WATCHDOG_NOW:-$(date +%s)}"
  # Market-open test: skip staleness when the market is closed, when the run
  # is too young to have written a bar, or when status.json is missing.
  DOW=$(TZ=America/Los_Angeles date -d "@$NOW" +%u)
  HHMM=$((10#$(TZ=America/Los_Angeles date -d "@$NOW" +%H%M)))
  market_closed=0
  { [ "$DOW" -eq 5 ] && [ "$HHMM" -ge 1400 ]; } || [ "$DOW" -eq 6 ] \
    || { [ "$DOW" -eq 7 ] && [ "$HHMM" -lt 1405 ]; } && market_closed=1
  if [ "$market_closed" -eq 0 ] && [ -f "$REPO/logs/status.json" ]; then
    status_age=$(( NOW - $(stat -c %Y "$REPO/logs/status.json") ))
    started=$(python3 -c "import json;print(json.load(open('$REPO/logs/status.json')).get('started',''))" 2>/dev/null || true)
    start_epoch=$(date -d "$started" +%s 2>/dev/null || echo 0)
    run_age=$(( NOW - start_epoch ))
    # Younger than ~2 bars or older than STALE_SECS both matter: a young run
    # has not had a bar to record yet; a stale one has stopped recording.
    if [ "$status_age" -gt "$STALE_SECS" ] && [ "$run_age" -gt "$STALE_SECS" ]; then
      ticks=$(( $(cat "$STALE_STATE" 2>/dev/null || echo 0) + 1 ))
      echo "$ticks" > "$STALE_STATE"
      log "STALE: status.json is ${status_age}s old (bar missed) — stale tick $ticks/$STALE_TICKS"
      ntfy "soak wedge suspected" "high" "warning" \
        "status.json ${status_age}s stale on tick $ticks/$STALE_TICKS"
      if [ "$ticks" -ge "$STALE_TICKS" ]; then
        log "STALE: $ticks consecutive stale ticks — restarting soak (SIGTERM flatten + boot reconcile)"
        ntfy "soak restarting after wedge" "urgent" "rotating_light" \
          "status.json stale for $ticks ticks; restarting $UNIT"
        if [ "${DRY_RUN:-0}" = "1" ]; then
          log "DRY_RUN=1 — would run: systemctl --user restart $UNIT"
        else
          "${SYSTEMCTL[@]}" restart "$UNIT"
          rm -f "$STALE_STATE"
        fi
      fi
    else
      rm -f "$STALE_STATE"  # bars flowing again: reset the counter
    fi
  fi
  exit 0
fi

NOW="${WATCHDOG_NOW:-$(date +%s)}"

# 3. weekend blackout — forex is closed Fri 14:00 PT -> Sun 14:00 PT (tracks
#    5pm New York year-round since ET/PT shift DST together). Don't cold-start
#    into a closed market; retry ticks land at Sun 14:05. Computed in
#    America/Los_Angeles so it stays correct if the machine TZ ever changes.
DOW=$(TZ=America/Los_Angeles date -d "@$NOW" +%u)   # 1=Mon .. 7=Sun
HHMM=$((10#$(TZ=America/Los_Angeles date -d "@$NOW" +%H%M)))
if { [ "$DOW" -eq 5 ] && [ "$HHMM" -ge 1400 ]; } || [ "$DOW" -eq 6 ] \
   || { [ "$DOW" -eq 7 ] && [ "$HHMM" -lt 1405 ]; }; then
  exit 0
fi

# 4. crash-loop brake (state in logs/, not /tmp: tmpfs is wiped on reboot and
#    a reboot mid-loop would otherwise re-arm the loop)
STATE="$REPO/logs/soak_watchdog_last_launch"
if [ -f "$STATE" ]; then
  LAST=$(cat "$STATE")
  if [ $((NOW - LAST)) -lt 900 ]; then
    log "soak is down but last launch was $((NOW - LAST))s ago — holding off (possible crash loop, check newest logs/soak_*.log)"
    exit 0
  fi
fi

# Post-mortem on the run that just ended. This is the reason the soak moved
# under systemd: record HOW it died, in the file we actually read. Result is
# systemd's verdict — "signal" (with which one), "oom-kill", "exit-code",
# "timeout", or "success" for a clean stop.
post_mortem() {
  local result code status
  result=$("${SYSTEMCTL[@]}" show "$UNIT" -p Result --value 2>/dev/null || echo "?")
  code=$("${SYSTEMCTL[@]}" show "$UNIT" -p ExecMainCode --value 2>/dev/null || echo "?")
  status=$("${SYSTEMCTL[@]}" show "$UNIT" -p ExecMainStatus --value 2>/dev/null || echo "?")
  # ExecMainCode follows the SI_* codes: 1=exited, 2=killed, 3=dumped. It is
  # reset to 0 once a unit stops cleanly, so 0 is ambiguous — split it on
  # Result. Always log exactly one line: a silent post-mortem is the failure
  # mode this whole change exists to remove.
  case "$code" in
    1) log "previous run: exited on its own with status $status (Result=$result)" ;;
    2) log "previous run: KILLED by signal $status (Result=$result)" ;;
    3) log "previous run: DUMPED CORE on signal $status (Result=$result)" ;;
    0)
      if [ "$result" = "success" ]; then
        log "previous run: stopped cleanly, no crash recorded"
      else
        log "previous run: no systemd record yet (first launch under $UNIT)"
      fi
      ;;
    *) log "previous run: unrecognised systemd record (code=$code Result=$result)" ;;
  esac
}

log "soak not running — launching M15 soak via $UNIT"
post_mortem
if [ "${DRY_RUN:-0}" = "1" ]; then
  log "DRY_RUN=1 — would run: systemctl --user start $UNIT"
  exit 0
fi
echo "$NOW" > "$STATE"
if ! "${SYSTEMCTL[@]}" start "$UNIT"; then
  log "LAUNCH FAILED — systemctl start $UNIT returned non-zero; try: systemctl --user status $UNIT"
  exit 0
fi
sleep 15
if pgrep -f 'bin/python -u run_oanda\.py' > /dev/null 2>&1; then
  log "launch OK — pid $(cat /tmp/soak.pid 2>/dev/null || echo '?'), log $(cat /tmp/soak_logpath 2>/dev/null || echo '?')"
else
  log "LAUNCH FAILED — no soak process 15s after start; check $(cat /tmp/soak_logpath 2>/dev/null || echo 'logs/soak_*.log') and journalctl --user -u $UNIT"
fi
