#!/usr/bin/env bash
# V4 Equities Investor Rebalance Launcher
#
# Runs the monthly portfolio rebalance using the trained LightGBM ranker model.
#
# Usage:
#   ./run_investor_rebalance.sh --dry-run               # Dry-run mode, no trades submitted
#   ./run_investor_rebalance.sh --skip-refresh          # Rebalance using existing features
#   ./run_investor_rebalance.sh                         # Live paper rebalance
#
# Scheduling:
#   Recommended crontab entry for monthly rebalances (keep --dry-run until ready):
#   30 16 1 * *  cd /mnt/storage/mystuf/development/build-A-bot && ./run_investor_rebalance.sh --dry-run
#
set -euo pipefail

# Ensure we are in the repo root directory
REPO_ROOT="/mnt/storage/mystuf/development/build-A-bot"
cd "$REPO_ROOT"

mkdir -p logs
LOG="logs/investor_rebalance_$(date +%Y-%m-%d_%H%M).log"

# Load environment variables if .env exists
if [ -f .env ]; then
  set -a; source .env; set +a
fi

export PYTHONPATH=src:.
VENV="/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python"

echo "Launching V4 investor rebalance (args: $*) -> logging to $LOG"
exec "$VENV" -u scripts/portfolio_orchestrator.py "$@" > "$LOG" 2>&1
