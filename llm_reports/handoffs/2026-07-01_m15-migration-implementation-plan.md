---
type: handoff
date: 2026-07-01
time: 23:55 PDT
agent: Claude Fable 5
model: claude-fable-5
trigger: Brandon approved the M1→M15 Angel/Devil migration plan; implementation deferred to a later Claude Sonnet 5 session — this is the execution brief
head: 8883b9e3e711b2905ac03182809805836c28f54d
scope: read-only
related:
  - handoffs/2026-07-01_feature-desync-rebuttal.md
  - handoffs/2026-06-19_gate-c-and-m1-tradeability.md
files_touched: []
---

# MODEL-TO-MODEL HANDOFF

**FROM:** Claude Fable 5 (planning session, 2026-07-01)
**TO:** Claude Sonnet 5 (implementing session, when Brandon says go)
**RE:** Approved plan — migrate the Angel/Devil forex model from M1 to M15 bars
**HEAD:** `8883b9e` on `main` (local main is 3 commits ahead of origin — do not "fix" this, it's known)
**WORKING TREE:** two deliberate uncommitted items, see §Context
**TONE:** execution brief; decisions are final, don't re-litigate them

---

## Context

The M1 scalper can't trade: the 9-day metals soak ended 2026-07-01 with zero trades (angel_prob median ~0.17 vs 0.40 threshold, see `logs/soak_2026-06-22_0120.log`), and the June cost-gate finding showed only metals clear the spread toll on M1 — the 6 crosses can't. M15 bars make the typical per-bar move ~4× larger while the spread stays fixed, so profit targets can clear the toll on all 8 instruments.

**Decisions Brandon has already locked (do not re-ask):**
1. Keep standard bar-count indicator periods (RSI-14, BB-20, SMA-50, NATR-14 unchanged — the retrained model learns their M15 meaning). Do NOT wall-clock-rescale them.
2. Bump the higher-timeframe context from `5m` to `1h` (5m would be *below* the new base timeframe).
3. Full 8-instrument basket (metals + the 6 crosses).
4. `RETRAIN_DAYS_BACK=730` (2 years; compensates for M15's 15× fewer bars/day).

**Working tree state you'll inherit (uncommitted, deliberate):** the `run_oanda.py` trained-timeframe mismatch warning (`_trained_timeframe()` + warning in `_main()`) and `llm_reports/handoffs/2026-07-01_feature-desync-rebuttal.md`. Keep both; commit them with this work's first commit.

## Investigation (already done — don't re-derive)

- **Features** (`src/ml/features/v3_features.py`): all base lookbacks are module-level constants — **leave every one untouched**. `V3HTFFeatures(timeframe=...)` already takes the resample string; `"1h"` is valid polars `group_by_dynamic` syntax. `V3SessionFeatures` is UTC-hour based, timeframe-agnostic. HTF instance periods (`_htf_rsi_period=14` etc., v3_features.py:179-182) also stay.
- **Retrainer** (`src/core/retrainer.py`): fully env-parameterized — `RETRAIN_TIMEFRAME_MINUTES`, `RETRAIN_HTF_TIMEFRAME`, `RETRAIN_DAYS_BACK`, `RETRAIN_MODEL_DIR` (side-model isolation; keeps `models/forex/` prod artifacts safe), `RETRAIN_SYMBOLS` (defaults to the 8-basket), `RETRAIN_MAX_HOLD` (45 bars ≈ 11h holds on M15 — intended), `RETRAIN_SURVIVAL` (5). Labels are SL 0.5×ATR / TP 3.0×ATR walk-forward — ATR auto-scales on M15. **No retrainer code changes needed** except possibly the metadata one-liner in step 5.
- **Provider** (`src/data/oanda_provider.py`): M15 fully supported — `_GRANULARITY` has `15: "M15"`, history paginates 5000/request, tick aggregation honors `stream_granularity_minutes=15`. No change.
- **Gate C** (rollover blackout, `src/execution/risk_manager.py:133-138`): America/New_York wall-clock anchored — timeframe-agnostic. No change.
- **Promotion gate**: Brier ≤0.30, EV ≥0.0005, PF ≥1.2, pooled OOS trade floor 300 (`RETRAIN_POOLED_TRADE_FLOOR`). **Do not loosen any threshold.**

## Findings / Changes — the implementation steps

### 1. `run_oanda.py` — M15 wiring (small)
Replace the two hand-rolled ternaries at ~line 175 (`htf_tf = "30m" if args.granularity == 5 else "5m"`; `warmup_pd = 300 if ...`) with:
```python
_GRANULARITY_PROFILES = {1: ("5m", 260), 5: ("30m", 300), 15: ("1h", 260)}
```
(260 M15 bars = 65 one-hour HTF bars ≥ the 50 needed for HTF SMA-50 — more warmup margin than either existing profile.) Unknown granularity → fail fast listing valid keys. Keep the existing trained-timeframe mismatch warning as-is.

### 2. `run_soak.sh` — granularity passthrough (tiny)
Optional granularity (env `SOAK_GRANULARITY` or second positional arg) forwarded as `--granularity N`; default stays 1.

### 3. Side-model loading for live/soak — verify, add only if missing
Check how `MLStrategy.__init__` resolves model artifact paths. `RETRAIN_MODEL_DIR` isolates *training* output, but to soak the M15 candidate without clobbering prod, the live side needs an equivalent override (env or constructor arg). If one exists, document it; if not, add a minimal env-based override following the `RETRAIN_MODEL_DIR` pattern, defaulting to prod paths.

### 4. Audit forex absolute floors for M15 sanity (read + document; likely no code change)
`RiskProfile.for_asset_class("forex")` constants (`min_sl_pct`, chop-floor values, `spread_atr_alpha=0.15`) were tuned on M1. On M15 the ATR is ~4× larger so floors bind less — likely fine — but read them and note in the report which are M1-calibrated. **Known item:** the June soak's empirical spread alphas (XAU 0.294, XAG 0.520) are M1-denominated and do NOT transfer to M15; the 0.15 default proxy stands until an M15 soak's SPREAD_CALIB recalibrates. State this in the report.

### 5. Run the M15 retrain (the long step)
```bash
set -a; source .env; set +a
RETRAIN_MODEL_DIR=models/forex_m15 \
RETRAIN_TIMEFRAME_MINUTES=15 \
RETRAIN_HTF_TIMEFRAME=1h \
RETRAIN_DAYS_BACK=730 \
PYTHONPATH=src:. \
/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python \
  -m src.core.retrainer 2>&1 | tee logs/retrain_m15_$(date +%Y-%m-%d_%H%M).log
```
Sanity-check `models/forex_m15/` doesn't collide first. Expect ~48k bars/instrument fetch (paginated, several minutes). Confirm the retrainer writes `metadata.json` with `timeframe_minutes: 15` into the side dir (metadata-write path near retrainer.py:1829); if `htf_timeframe` isn't recorded there, add it (one line) so deployment can derive the right HTF later.

### 6. Interpret the gate verdict — from the log only
Quote fold metrics verbatim from the retrain log (house rule: retrain claims must be rerun-verifiable — agent retrain handoffs have shipped fabricated metrics before). Three outcomes:
- **PROMOTED** → candidate sits in `models/forex_m15/`. Deployment (copy to `models/forex/`, M15 soak) is Brandon's explicit call — stage, don't do.
- **REJECTED on sample size** with good Brier/PF/separation → report; note the known LightGBM narrow-probability pattern; do not lower the floor.
- **REJECTED on quality** → report; a negative answer to the M15 hypothesis is still the deliverable.

### 7. Report
`llm_reports/refactors/<actual-date>_m15-timeframe-migration.md` per `llm_reports/README.md` (frontmatter, `files_touched`, six sections). Cover: what changed and why, gate verdict with log-quoted metrics, the alpha-recalibration caveat, next steps.

## Verification

1. **Feature smoke BEFORE the long retrain**: fetch ~1 week of M15 XAU_USD via `get_historical_bars`, run `FeaturePipeline([V3BaseFeatures(), V3HTFFeatures(timeframe="1h"), V3SessionFeatures()])`, assert non-null feature rows after warmup and populated `htf_*` columns (proves the `"1h"` resample end-to-end).
2. `run_oanda.py --granularity 15` boot smoke on practice: correct profile (htf=1h, warmup=260) and — while prod is still the M1 model — the timeframe-mismatch warning firing.
3. `pipenv run pytest tests/` after code changes.
4. The retrain log is the primary artifact: per-instrument fetch counts, fold metrics, gate verdict.

## Risk & follow-ups

- Empirical spread alphas need M15 recalibration via a future soak (see step 4).
- Sample size: 730d of M15 ≈ 48k bars/instrument, ~380k pooled — the 300-trade pooled floor should be reachable, but if the gate rejects on samples with good separation, that's a finding, not a knob to turn.
- Out of scope: deploying/promoting to `models/forex/`; launching an M15 soak; pushing `main` to origin (known 3-commits-ahead state, Brandon's call); the pre-existing Gate C retrainer-symmetry TODO.

## Files touched

None by this handoff (planning only). The implementation will touch:
- `run_oanda.py` (granularity profile map; builds on the uncommitted mismatch warning)
- `run_soak.sh` (granularity passthrough)
- `src/strategies/concrete_strategies/ml_strategy.py` (only if the side-model path override is missing)
- `src/core/retrainer.py` (only if metadata lacks `htf_timeframe`)
- `models/forex_m15/` (new side-model artifacts, gitignored)
- `llm_reports/refactors/<date>_m15-timeframe-migration.md` (new)

---

**END HANDOFF.**
