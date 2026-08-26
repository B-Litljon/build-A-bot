#!/usr/bin/env bash
# Holdout-stability demonstration batch: run the FIXED gate exactly as
# production runs it, at three pinned window endpoints, for the 2-year and
# 5-year configs -- the m2m brief's section 8 deliverable.
#
#   bash scripts/run_stability_batch.sh
#
# Six full retrainer runs (fetch -> carve -> engineer -> purge -> fold gate ->
# artifact holdout), SEQUENTIALLY -- one run at a time, each solo on the
# machine. Threads stay at LightGBM defaults (n_jobs=-1, 12 hw threads). This
# box has 6 physical cores, and any two-way overlap collapses fit speed ~100x
# once both jobs are mid-refit (measured 2026-08-24: 3.5s solo fit -> 9+ min
# under 2-way contention); the 2026-08-24 recon's ~85-minute runs hid the
# same collapse behind concurrent fetch phases. One run at a time is both
# faster end to end and keeps every pin measured under identical conditions.
# Nothing touches models/forex_m15_wide -- every run writes to a side
# RETRAIN_MODEL_DIR. Verdicts land in logs/stability_<cfg>_<pin>.log; side
# model dirs (models/forex_m15_stability_*) are the caller's to remove.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export PYTHONPATH=src:.
export DATA_SOURCE=oanda
export RETRAIN_TIMEFRAME_MINUTES=15
PY="/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python"

run_one() {  # <days_back> <pin>
  local days="$1" pin="$2" tag rc=0
  tag="2yr"; [ "$days" = "1825" ] && tag="5yr"
  # `|| rc=$?` keeps the expected exit-2 rejection from tripping set -e
  # before the verdict line prints.
  RETRAIN_DAYS_BACK="$days" \
  RETRAIN_END_DATE="$pin" \
  RETRAIN_MODEL_DIR="models/forex_m15_stability_${tag}_${pin//-/}" \
    "$PY" -m src.core.retrainer \
    > "logs/stability_${tag}_${pin//-/}.log" 2>&1 || rc=$?
  echo "[${tag} @ ${pin}] exit ${rc}"
}

for pin in 2026-08-09 2026-08-16 2026-08-23; do
  run_one 730  "$pin"
  run_one 1825 "$pin"
done
echo "batch complete"
