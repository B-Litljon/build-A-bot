---
to: gemini-3.5-flash
from: claude-opus-4-8
date: 2026-06-22
status: verified
branch: feature/investor-gate
topic: investor self-gate + metadata sidecar + monthly schedule launcher
result_commit: ece1090
result_notes: >
  Tasks 1 & 2 (gate + metadata) committed in ece1090 and independently
  re-verified by Claude (numbers reproduce exactly, reject latch holds,
  73 tests green). Task 3 (schedule launcher run_investor_rebalance.sh +
  INVESTOR_SCHEDULE.md) deliberately PARKED uncommitted: the model clears
  random by only ~1.05x (P@1 0.300 vs 0.286), too thin to schedule live.
  Gate defaults loosened to 1.0x lift to pass the current model.
related_memory: project_soak_week_upgrade_backlog
related_report:
---

# MODEL-TO-MODEL HANDOFF — Finish the Investor (self-gate + metadata + schedule)

**TO:** Gemini 3.5 Flash (implementing coder)
**FROM:** Claude Opus 4.8 (planner)
**REPO:** `/mnt/storage/mystuf/development/build-A-bot`
**RUNTIME:** venv python at `/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python`; always run with `PYTHONPATH=src:.` from repo root. Project uses `pipenv`.

---

## Context — why this work exists

The bot has three models. Two of them (the forex scalper and the equities day-trader, both trained by `src/core/retrainer.py`) are **production-hardened**: before shipping a retrained model they run a validation gate that **refuses to promote a weak model**, and they write a `metadata.json` sidecar recording exactly what the model was trained on.

The **investor** (a once-a-month LightGBM stock-ranker, trained by `scripts/investor_train_model.py`) has **neither**. It computes good walk-forward metrics and then **ignores them — it saves whatever it trained, every time**. It also leaves no record of what it trained on. And nothing actually runs it on a schedule.

This task brings the investor up to the same governance bar as the forex side. There is a bonus architectural payoff: the autopilot pipeline (`src/autopilot/`, on branch `feature/autopilot-rails`) has a generic `DelegatingGate` that reads a trainer's **exit code** and a `NoOpDeployer` that trusts a trainer's **own atomic save**. So if you make the investor trainer (a) return the standard exit codes and (b) write its own metadata — exactly like the forex trainer already does — the investor path snaps into the existing rails with **zero new rail code**. Mirror the forex trainer; don't invent a new pattern.

**DO NOT** change the model architecture, features, hyperparameters, walk-forward split logic, or the universe. This is a governance wrapper around training that already works. **DO NOT** touch anything under `src/execution/` or `run_soak.sh` — a live gold/silver soak is running off that code and must not be disturbed.

**Branch:** create `feature/investor-gate` off the current branch (`feature/dynamic-chop-floor`). All three scripts live in `scripts/` and are identical across branches, so this is self-contained.

---

## The exact pattern you are mirroring (forex side — read these first)

- **Gate thresholds as module constants:** `src/core/retrainer.py:214-219` (`BRIER_THRESHOLD`, `EV_THRESHOLD`, `PROFIT_FACTOR_THRESHOLD`).
- **Gate decision → promote-or-reject:** `src/core/retrainer.py:1731` (passed → save) vs `:1749` (failed → reject, keep prod weights).
- **Exit-code contract:** `src/core/retrainer.py:1994-2012` and `sys.exit(main())` — **`0` = passed & saved, `2` = trained but REJECTED (prior model kept), `1` = error.** Reuse these exact codes.
- **Atomic metadata sidecar:** `src/core/retrainer.py:1822-1837` — write to a `*_temp.json`, then `os.replace(temp, final)`. Fields: `asset_class`, `timeframe_minutes`, `trained_at` (UTC isoformat), `trained_on_symbols`, `data_source`.

---

## TASK 1 — Add a self-approval gate to `scripts/investor_train_model.py`

The trainer already computes everything the gate needs. In `main()`:
- Per-fold Precision@1, Precision@2, NDCG are collected into `fold_results` (lines 293-300).
- Means are computed at lines 311-317 (`mean_ndcg`, `mean_p1`, `mean_p2`) — **but only inside an `if ndcg_col:` block.**
- The final model is then trained and **unconditionally saved** at lines 319-336.

**Change the control flow to: compute means → check gate → only save if the gate passes.**

1. Add gate-threshold constants near the top (after `TEST_DAYS`, ~line 59), all overridable by env var:
   ```python
   # ── Self-approval gate thresholds (mirror src/core/retrainer.py:214-219) ──
   # Ranker gate is LIFT-OVER-RANDOM, not absolute: a random picker scores
   # Precision@K ≈ the positive base rate (top-quintile target ≈ 0.20). The
   # gate requires the model to clear the base rate by a margin.
   GATE_P1_MIN_LIFT = float(os.getenv("INVESTOR_GATE_P1_LIFT", "1.5"))  # P@1 ≥ 1.5× base rate
   GATE_P2_MIN_LIFT = float(os.getenv("INVESTOR_GATE_P2_LIFT", "1.2"))  # P@2 ≥ 1.2× base rate
   GATE_NDCG_MIN    = float(os.getenv("INVESTOR_GATE_NDCG_MIN", "0.0")) # absolute NDCG floor (0 = informational until calibrated)
   FORCE_SAVE       = os.getenv("INVESTOR_GATE_FORCE", "0").strip() == "1"  # escape hatch
   ```
   (Add `import os` — it is not currently imported.)

2. Compute the **positive base rate** from the data (it is already available: `y_all.mean()`, see line 211). Derive the absolute pass marks:
   `p1_floor = base_rate * GATE_P1_MIN_LIFT`, `p2_floor = base_rate * GATE_P2_MIN_LIFT`.

3. After the means are computed (line 317), **before** the "Final model" block (line 319), insert the gate:
   - Log a clear GATE SUMMARY block (mirror retrainer's `:1626` style): print base_rate, each metric, each floor, PASS/FAIL per metric.
   - `gate_passed = (mean_p1 >= p1_floor) and (mean_p2 >= p2_floor) and (mean_ndcg >= GATE_NDCG_MIN)`.
   - If **not** `gate_passed` and **not** `FORCE_SAVE`: log "🚫 GATE FAILED — existing model retained, nothing written", and **`return 2`** (do NOT train/save the final model — leave the prior `models/v4_investor_lgbm.txt` untouched).
   - If passed (or forced): proceed to train + save as today, then `return 0`.

4. Convert `main()` to **return `int`** and change the entrypoint to `sys.exit(main())` (currently `main()` returns `None` at line 347 and the entrypoint is a bare `main()` at line 351). Wrap the body so unexpected exceptions log and `return 1` (mirror `portfolio_orchestrator.py:542-544`).

**IMPORTANT — do not hardcode unvalidated numbers as truth.** The lift defaults above are a starting point. After implementing, **run the trainer once** (Task 1 verification) and read the real logged `base_rate`, `mean_p1`, `mean_p2`. Confirm the **current, known-good model passes** the gate with these defaults. If the current model would be false-rejected, loosen the lift defaults so the gate passes the existing good model, and **report the real numbers you observed** — do not silently retune to force a pass without telling us.

---

## TASK 2 — Write a metadata sidecar from `scripts/investor_train_model.py`

Only when the gate passes and the model is saved (right after line 336, `Model saved → ...`), write an atomic sidecar next to the model. The model is a single file (`models/v4_investor_lgbm.txt`), not a directory, so name the sidecar `models/v4_investor_lgbm.metadata.json`.

Mirror the forex atomic-write idiom (`retrainer.py:1822-1837`): write `*_temp.json`, then `os.replace`. Contents:
```python
metadata = {
    "model": "v4_investor_lgbm",
    "asset_class": "equities_longterm",
    "horizon_days": EMBARGO_DAYS,                      # 60-day forward target
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "trained_on_symbols": sorted(df["symbol"].unique().tolist()),
    "n_features": len(feature_cols),
    "data_source": "alpaca",
    "walk_forward": {
        "folds": int(fold_num),
        "mean_ndcg": round(float(mean_ndcg), 4),
        "mean_precision_at_1": round(float(mean_p1), 4),
        "mean_precision_at_2": round(float(mean_p2), 4),
        "positive_base_rate": round(float(base_rate), 4),
    },
    "gate_passed": True,
}
```
(Add `from datetime import datetime, timezone` and `import json`.)

**Optional, low-risk lineage check in `scripts/portfolio_orchestrator.py`:** right after the booster loads (line 528), if the sidecar exists, read it and `logger.warning(...)` if `metadata["trained_on_symbols"] != sorted(UNIVERSE)` — so a model trained on a different basket than it's being run on is surfaced, not silent. Do **not** make this fatal.

---

## TASK 3 — A scheduling launcher for the monthly rebalance

The monthly rebalance is `scripts/portfolio_orchestrator.py` (its docstring at line ~36 already specifies the intended cron `30 16 1 * *`). It is currently manual-only. Produce a launcher mirroring the existing `run_soak.sh` style (venv path, `PYTHONPATH=src:.`, timestamped logfile under `logs/`, `set -euo pipefail`):

1. Create `run_investor_rebalance.sh` at repo root:
   - Resolves repo root, exports `PYTHONPATH=src:.`.
   - Logs to `logs/investor_rebalance_$(date +%Y-%m-%d_%H%M).log`.
   - Passes any args straight through (so `--dry-run` / `--skip-refresh` work).
   - Uses the venv python path above.
2. **Do NOT install a crontab entry yourself.** Instead, append a short `## Scheduling` section to the script docstring / or a one-screen `INVESTOR_SCHEDULE.md` with the exact `crontab -e` line:
   ```
   30 16 1 * *  cd /mnt/storage/mystuf/development/build-A-bot && ./run_investor_rebalance.sh --dry-run
   ```
   Note that the **first scheduled runs should stay `--dry-run`** (logs intended trades without sending them) until Brandon reviews a few months of intended allocations, then drop `--dry-run` to go live on the paper account.

---

## Constraints (hard)

- No changes under `src/execution/` or to `run_soak.sh` (live soak running).
- No change to model architecture, features, hyperparams, walk-forward logic, or `UNIVERSE`.
- The gate must be **fail-safe**: on rejection it must leave the previously-saved `models/v4_investor_lgbm.txt` **untouched** (i.e., don't overwrite then check — check first, save only on pass).
- Keep the existing logging style and comment density.

---

## Verification (run these; paste real output in your report)

1. **Gate passes the current good model:**
   `PYTHONPATH=src:. <venv-python> scripts/investor_train_model.py`
   → expect exit `0`, a GATE SUMMARY block, model saved, and `models/v4_investor_lgbm.metadata.json` written. Report the logged `base_rate`, `mean_p1`, `mean_p2`, `mean_ndcg` and the PASS marks.
2. **Gate rejects a deliberately-bad model (prove the latch works):** temporarily set `INVESTOR_GATE_P1_LIFT=99` and re-run → expect exit `2`, "GATE FAILED", and confirm the existing model file's mtime did **not** change (it was not overwritten). Revert the env var.
3. **Metadata is valid JSON** with the fields above and `trained_on_symbols` equal to the 7-name universe.
4. **Launcher dry-run:** `./run_investor_rebalance.sh --dry-run --skip-refresh` runs end-to-end, exits 0, logs intended allocations, sends no orders. (Use `--skip-refresh` only if `data/processed/v4_inference_features.parquet` already exists; otherwise run without it.)
5. **No regressions:** `PYTHONPATH=src:. <venv-python> -m pytest tests/ -q` still green (existing `tests/test_execution_safety.py::test_alpaca_rebalance_gate` must still pass).

## Report back
- The real walk-forward numbers and which gate floors you settled on (and why, if you changed the defaults).
- Confirmation of the reject-latch test (exit 2, file untouched).
- Any place the forex pattern didn't map cleanly to the single-file (vs directory) model layout.
