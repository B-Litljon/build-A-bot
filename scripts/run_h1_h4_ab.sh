#!/usr/bin/env bash
set -euo pipefail
cd /mnt/storage/mystuf/development/build-A-bot
set -a; source .env; set +a
export PYTHONPATH=src:.
VENV=/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python

echo "=== Launching H1 and H4 CatBoost A/B evaluations in parallel ==="

DATA_SOURCE=oanda RETRAIN_TIMEFRAME_MINUTES=60 RETRAIN_HTF_TIMEFRAME=4h CB_DAYS_BACK=730 CB_TAG=h1 "$VENV" scripts/run_catboost_ab.py > logs/ab_catboost_h1.log 2>&1 &
PID_H1=$!
echo "H1 evaluation launched (PID $PID_H1) -> logs/ab_catboost_h1.log"

DATA_SOURCE=oanda RETRAIN_TIMEFRAME_MINUTES=240 RETRAIN_HTF_TIMEFRAME=1d CB_DAYS_BACK=730 CB_TAG=h4 "$VENV" scripts/run_catboost_ab.py > logs/ab_catboost_h4.log 2>&1 &
PID_H4=$!
echo "H4 evaluation launched (PID $PID_H4) -> logs/ab_catboost_h4.log"

# Wait for both background processes
wait $PID_H1 || true
echo "H1 evaluation finished."

wait $PID_H4 || true
echo "H4 evaluation finished."

echo "=== Both evaluations completed ==="
