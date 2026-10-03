#!/usr/bin/env bash
# Run ONE retrain experiment arm with every setting that could drift pinned.
#
# An "arm" is one change under test: the baseline (untouched main) or main plus
# exactly one merged lane. Each arm writes its own log and its own candidate
# model dir, so two arms can never be confused for each other and NO arm can
# touch the served model at models/forex_m15_wide.
#
#   bash scripts/retrain_arm.sh baseline
#   bash scripts/retrain_arm.sh b-table-off              # B merged, table OFF = null control
#   bash scripts/retrain_arm.sh b-table-on   --spread-table
#   bash scripts/retrain_arm.sh d-rollover
#   bash scripts/retrain_arm.sh htf-partition
#   bash scripts/retrain_arm.sh all-three    --spread-table
#   bash scripts/retrain_arm.sh d-rollover   --end=2026-09-01   # re-pin the window
#   bash scripts/retrain_arm.sh d-rollover   --days=1095        # re-pin the window LENGTH
#
# The script does NOT create branches or merge lanes -- do that yourself, one
# lane at a time, then run the arm in the resulting tree. It records which tree
# it found in the provenance file, so a mis-aimed arm is visible afterwards.
#
# Glossary:
#   ARM -- the arm name; becomes the log slug and the candidate dir name.
#   --spread-table -- turn the measured per-instrument cost table ON
#       (RETRAIN_SPREAD_TABLE=config/spread_alphas_m15.json). WITHOUT it the
#       spread-label fix is inert: labels stay byte-identical to the incumbent's,
#       which is what makes `b-table-off` a useful null control.
#   PINNED_TF / PINNED_HTF -- 15 and "1h". THE TRAP: get_asset_config() defaults
#       BOTH asset classes to `timeframe = 1` (one-minute bars), so a bare
#       `python -m src.core.retrainer` silently retrains an M1 model -- 15x the
#       fetch, and nothing like the served M15 artifact. htf_timeframe then
#       derives from the bar size via _HTF_FOR_TIMEFRAME (1->5m, 5->30m, 15->1h);
#       it is pinned explicitly too because the code's own comment names a flat
#       "5m" default that "silently shipped skew" for M15 retrains. Cost a
#       six-minute aborted run to find on 2026-10-02.
#   PINNED_END / --end, PINNED_DAYS / --days -- all arms fetch the SAME window.
#       Without this the retrain reads "up to now", so two arms run an hour apart
#       see different data and stop being comparable. Defaults are 2026-10-01 and
#       730 days (the window the first six arms ran on); `--end=YYYY-MM-DD` and
#       `--days=NNNN` re-pin ONE run, which is how an arm is checked across
#       windows instead of trusted from one. A longer window is the way to buy
#       more trades at the gate, which is where the binding constraint sits.
#   preflight -- resolves the real config and refuses to fetch unless the
#       recipe matches the incumbent (forex, M15, 1h HTF, 730 days, 0.18
#       holdout). Cheap, and it runs BEFORE the expensive data pull.
#   candidate dir -- models/candidates/<ARM>. Must not already exist:
#       save_models MERGES into an existing metadata.json (see _persist.py), so
#       a reused dir would carry stale keys from an earlier arm.
#   provenance -- arm_provenance.txt inside the candidate dir records branch,
#       HEAD, commits ahead of main, dirty state, the table setting and the
#       resolved recipe, so every artifact explains which tree produced it.
#   Discord -- suppressed for arms (DISCORD_WEBHOOK_URL unset after sourcing
#       .env). NotificationManager then no-ops with a warning instead of
#       posting an embed report for every arm.
#   exit code -- 0 promoted into the candidate dir, 2 rejected (nothing
#       written), 1 error. A rejected arm is a RESULT, not a script failure.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ARM="${1:-}"
if [ -z "$ARM" ]; then
  echo "usage: retrain_arm.sh <arm-name> [--spread-table] [--end=YYYY-MM-DD] [--days=NNNN]" >&2
  exit 2
fi
shift || true

PINNED_DAYS=730   # default window LENGTH; re-pin one run with --days=NNNN
DAYS_BACK="$PINNED_DAYS"
PINNED_END=2026-10-01   # default window; re-pin one run with --end=YYYY-MM-DD
END_DATE="$PINNED_END"
PINNED_HOLDOUT=0.18
PINNED_TF=15
PINNED_HTF=1h
PINNED_SOURCE=oanda
CAND_DIR="models/candidates/$ARM"
LOG="logs/arm-$ARM.log"
VENV=/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python

if [ -e "$CAND_DIR" ]; then
  echo "REFUSING: $CAND_DIR already exists." >&2
  echo "Each arm gets a fresh dir -- save_models merges into an existing metadata.json." >&2
  exit 3
fi

TABLE=""
for arg in "$@"; do
  case "$arg" in
    --spread-table) TABLE="config/spread_alphas_m15.json" ;;
    --end=*)        END_DATE="${arg#--end=}" ;;
    --days=*)       DAYS_BACK="${arg#--days=}" ;;
    *)
      echo "REFUSING: unknown argument '$arg' (accepted: --spread-table, --end=YYYY-MM-DD, --days=NNNN)." >&2
      exit 2
      ;;
  esac
done
if ! printf '%s' "$END_DATE" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'; then
  echo "REFUSING: --end must be YYYY-MM-DD, got '$END_DATE'." >&2
  exit 2
fi
if ! printf '%s' "$DAYS_BACK" | grep -qE '^[0-9]+$' || [ "$DAYS_BACK" -lt 30 ]; then
  echo "REFUSING: --days must be a whole number of days >= 30, got '$DAYS_BACK'." >&2
  exit 2
fi

# Arm self-check: --spread-table is only meaningful if the tree carries the fix.
# Without it the table would change cost_ratio and the veto, but NOT the labels.
if [ -n "$TABLE" ] && ! grep -q "alpha_table" src/core/retrainer/_labels.py; then
  echo "REFUSING: this tree has no alpha_table in src/core/retrainer/_labels.py," >&2
  echo "so the spread-label fix is not merged and --spread-table would not price" >&2
  echo "the labels. Merge lane/fix-spread-labels first." >&2
  exit 4
fi

mkdir -p "$CAND_DIR" logs

# Credentials by name, never echoed; then drop the webhook for the arms.
set -a; . ./.env; set +a
unset DISCORD_WEBHOOK_URL

export PYTHONPATH=src:.
export DATA_SOURCE="$PINNED_SOURCE"
export RETRAIN_DAYS_BACK="$DAYS_BACK"
export ARM_EXPECT_DAYS="$DAYS_BACK"   # the preflight compares the resolved value against this
export RETRAIN_END_DATE="$END_DATE"
export RETRAIN_HOLDOUT_FRAC="$PINNED_HOLDOUT"
export RETRAIN_TIMEFRAME_MINUTES="$PINNED_TF"
export RETRAIN_HTF_TIMEFRAME="$PINNED_HTF"
export RETRAIN_MODEL_DIR="$CAND_DIR"
if [ -n "$TABLE" ]; then export RETRAIN_SPREAD_TABLE="$TABLE"; fi

# ── preflight: resolve the real config before paying for the fetch ──────────
set +e
RECIPE="$("$VENV" - <<'PY'
import os, sys
from core.retrainer._common import get_asset_config, DAYS_BACK, HOLDOUT_FRAC
cfg = get_asset_config(os.getenv("DATA_SOURCE", "alpaca"))
bad = []
if cfg["asset_class"] != "forex":           bad.append("asset_class!=forex")
if cfg["timeframe_minutes"] != 15:          bad.append("timeframe_minutes!=15")
if cfg["htf_timeframe"] != "1h":            bad.append("htf_timeframe!=1h")
if int(DAYS_BACK) != int(os.environ["ARM_EXPECT_DAYS"]):
    bad.append(f"DAYS_BACK!={os.environ['ARM_EXPECT_DAYS']}")
if abs(float(HOLDOUT_FRAC) - 0.18) > 1e-9:  bad.append("HOLDOUT_FRAC!=0.18")
print(
    f"asset_class={cfg['asset_class']} bars={cfg['timeframe_minutes']}min "
    f"htf={cfg['htf_timeframe']} sl={cfg['sl_mult']} tp={cfg['tp_mult']} "
    f"max_hold={cfg['max_hold']} survival={cfg['survival_bars']} "
    f"days={int(DAYS_BACK)} holdout={float(HOLDOUT_FRAC)} "
    f"symbols={len(cfg['tickers'])} model_dir={cfg['model_dir']}"
)
if bad:
    sys.stderr.write("PREFLIGHT FAILED: " + ", ".join(bad) + "\n")
    sys.exit(9)
PY
)"
PRE_RC=$?
set -e
if [ "$PRE_RC" -ne 0 ]; then
  echo "$RECIPE" >&2
  echo "REFUSING to run: the pinned recipe did not resolve as intended." >&2
  rm -rf "$CAND_DIR"
  exit 5
fi

{
  echo "arm:              $ARM"
  echo "when:             $(date -Is)"
  echo "branch:           $(git rev-parse --abbrev-ref HEAD)"
  echo "head:             $(git rev-parse HEAD)"
  echo "ahead-of-main:    $(git rev-list --count main..HEAD) commit(s)"
  echo "commits-in-arm:"
  git log --oneline main..HEAD | sed 's/^/                  /'
  echo "dirty-paths:      $(git status --porcelain | wc -l)"
  echo "recipe:           $RECIPE"
  echo "spread_table:     ${TABLE:-<off>}"
  echo "pinned:           days=$DAYS_BACK end=$END_DATE holdout=$PINNED_HOLDOUT bars=${PINNED_TF}min htf=$PINNED_HTF source=$PINNED_SOURCE"
  echo "model_dir:        $CAND_DIR"
} | tee "$CAND_DIR/arm_provenance.txt"

echo
echo "=== running retrain (log: $LOG) ==="
set +e
"$VENV" -m src.core.retrainer > "$LOG" 2>&1
RC=$?
set -e

SUMMARY_PATTERNS='Timeframe:|Combined dataset:|Generated devil_target_macro|Generated devil_target \(|Hybrid chop veto dropped|Final dataset:|Dynamic Angel threshold:|\[Fold [0-9]+\] Brier|Mean Brier Score|Fold 3 PF|Pooled PF \(folds\)|HOLDOUT METRICS:|Gate Result|MODELS PROMOTED|MODELS REJECTED'

{
  echo "arm: $ARM   branch: $(git rev-parse --abbrev-ref HEAD)   exit: $RC"
  echo
  grep -E "$SUMMARY_PATTERNS" "$LOG" || echo "(no summary lines matched -- read $LOG)"
} | tee "$CAND_DIR/arm_summary.txt"

echo
case "$RC" in
  0) echo "exit 0 -- PROMOTED into $CAND_DIR (served model untouched)";;
  2) echo "exit 2 -- REJECTED: production weights retained, this is a result not a failure";;
  *) echo "exit $RC -- ERROR: read $LOG";;
esac
exit "$RC"
