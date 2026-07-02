---
type: refactor
date: 2026-07-02
time: 01:15 PDT
agent: Claude Sonnet 5 (plan authored by Claude Fable 5)
model: claude-sonnet-5
trigger: Brandon approved migrating Angel/Devil from M1 to M15 bars after the 9-day metals soak produced zero trades; plan at handoffs/2026-07-01_m15-migration-implementation-plan.md
head: 8883b9e3e711b2905ac03182809805836c28f54d
scope: modifies-source
related:
  - handoffs/2026-07-01_m15-migration-implementation-plan.md
  - handoffs/2026-07-01_feature-desync-rebuttal.md
files_touched:
  - run_oanda.py
  - run_soak.sh
  - src/strategies/concrete_strategies/ml_strategy.py
  - src/core/retrainer.py
  - models/forex_m15/ (new side-model artifacts, gitignored)
---

# M15 Timeframe Migration — code changes + PROMOTED M15 candidate

## Context

The M1 scalper couldn't trade: the 2026-06-22→07-01 metals soak made zero trades (angel_prob median ~0.17 vs 0.40 threshold), and only metals cleared the M1 spread-cost gate. M15 bars carry a ~4× larger typical move against the same fixed spread. Brandon locked the decisions on 2026-07-01: keep standard bar-count indicator periods, bump HTF context 5m→1h, full 8-instrument basket, 730 days of history, side-model isolation, gate thresholds untouched.

**Headline result: the M15 candidate PASSED the validation gate** — the first promoted model since the original 2026-05 M1 model — and sits staged in `models/forex_m15/` awaiting Brandon's deployment call.

## Investigation

Covered in the plan handoff (features/retrainer/provider/Gate C all pre-verified). Two findings added during implementation:

1. **`MLStrategy` sidecar paths were hardcoded** to `models/<asset_class>/` while the pkl paths were constructor-overridable — a side model would have loaded prod's threshold.json/metadata.json (wrong threshold, wrong basket, wrong timeframe guard).
2. **OANDA intermittently 401s burst candle requests** ("Insufficient authorization" that succeeds seconds later). `fetch_training_data` swallowed the failure and continued, so the first retrain attempt (log `retrain_m15_2026-07-02_0059.log`, exit 2) silently trained on 6/8 instruments — XAG_USD and GBP_JPY dropped — while metadata would have claimed all 8.

Cost-gate sanity check from the feature smoke: M15 XAU_USD NATR ≈ 0.215% vs the soak's M1 baseline 0.055% (~3.9×), so gold's effective spread ratio drops from 0.294 to ~0.08 — far below the 0.667 tradeability ceiling. The M15 rationale holds empirically.

## Findings / Changes

**`run_oanda.py`**
- New `_GRANULARITY_PROFILES = {1: ("5m", 260), 5: ("30m", 300), 15: ("1h", 260)}` replaces hand-rolled ternaries; `--granularity` now uses argparse `choices` (fails fast on unsupported values). 260 M15 bars = 65 one-hour HTF bars ≥ the 50 needed for HTF SMA-50.
- New `OANDA_MODEL_DIR` env override redirects the whole artifact set (pkls + metadata + threshold) to a side model, mirroring `RETRAIN_MODEL_DIR`. Default unchanged (`models/forex`).
- (Carried in from 2026-07-01, previously uncommitted) `_trained_timeframe()` + loud warning when `--granularity` mismatches the model's trained `timeframe_minutes`.

**`src/strategies/concrete_strategies/ml_strategy.py`**
- `_validate_metadata()` and `_load_threshold()` now resolve their sidecar files from `self.angel_path.parent` instead of hardcoded `models/<asset_class>/`. Identical behavior for prod; side models now carry their own sidecars.

**`run_soak.sh`**
- Optional granularity: second positional arg or `SOAK_GRANULARITY` env, forwarded as `--granularity N`; default 1 (unchanged).

**`src/core/retrainer.py`**
- `fetch_training_data`: per-symbol retry (3 attempts, 5s·n backoff, `RETRAIN_FETCH_RETRIES` env) and **hard failure if any requested symbol is still missing** — refuses to train on a silently shrunk basket.
- Metadata sidecar now records `htf_timeframe`.
- Fixed hardcoded "Timeframe: 1-minute bars" log line.

**The retrain (log `logs/retrain_m15_2026-07-02_0102.log`, exit 0 = promoted):**
- Fetch: all 8 instruments, 47,269–49,724 bars each, 392,842 combined rows (2024-07-02 → 2026-07-02).
- Chop veto dropped 88,892 rows (22.6%) — nearly identical to M1's historical drop rate.
- Gate summary, verbatim:
  ```
  Mean Brier Score : 0.2956 (threshold ≤ 0.3)
  Mean EV          : 0.723318 (threshold ≥ 0.0005)
  Profit Factor    : 1.5385 (threshold ≥ 1.2, Fold 3 OOS)
  Pooled OOS Trades: 340 across 3 folds (dynamic floor ≥ 232 = 300×(1−22.6%) | Fold 3 PF trades=138)
  Gate Result      : PASSED ✅
  ```
- Per fold: F1 Brier=0.2991 EV=0.800 WR=60.0% (90 trades) | F2 Brier=0.2960 EV=0.848 WR=61.6% (112) | F3 Brier=0.2916 EV=0.522 WR=50.7% (138), PF=120/78=1.5385, frozen OOS threshold 0.48.
- Artifacts: `models/forex_m15/{angel,devil}_latest.pkl` + `metadata.json` (timeframe_minutes=15, htf_timeframe=1h, all 8 symbols) + `threshold.json` (0.48). **Prod `models/forex/` untouched** (June 14 mtimes verified).

## Verification

1. Feature smoke pre-retrain: 736 real M15 XAU_USD bars → pipeline with `V3HTFFeatures(timeframe="1h")` → all 26 features non-null on all 679 post-warmup rows; HTF values constant within each hour (correct resample).
2. `pytest tests/` — 73 passed, before and after the retrainer changes.
3. Boot smoke: `OANDA_MODEL_DIR=models/forex_m15 python run_oanda.py --daemon --env practice --granularity 15` — models/threshold/metadata all load from the side dir, no mismatch warning (correctly silent), full 8-basket defaults from side metadata, position sync normal. Killed after 50s.
4. Guard negative case: `--granularity 7` rejected by argparse; prod-metadata mismatch warning verified earlier (2026-07-01).
5. First (invalid) retrain attempt preserved at `logs/retrain_m15_2026-07-02_0059.log` for comparison: 6-instrument basket REJECTED (Brier 0.3208, pooled 212<232) — the full basket is what passes, which is itself evidence the crosses contribute signal on M15.

## Risk & follow-ups

- **Spread alphas are M1-denominated** (XAU 0.294, XAG 0.520 from the June soak) and do not transfer; training used the default `spread_atr_alpha=0.15` proxy, which *overestimates* cost on M15 (conservative direction). An M15 soak's SPREAD_CALIB should recalibrate.
- `regime_window=260` bars now spans ~2.7 trading days instead of ~4.3 hours — symmetric between training and live, but a semantic shift worth remembering.
- Devil threshold dropped to 0.48 (from prod's 0.52); Angel threshold stays 0.40. Watch the first soak's proposal rate — M15 heartbeats arrive 15× slower.
- 45-bar max hold now means ~11-hour positions; weekend-gap exposure is a new risk class the M1 scalper never had.
- **Staged, not deployed:** running an M15 soak (`OANDA_MODEL_DIR=models/forex_m15 SOAK_GRANULARITY=15 bash run_soak.sh`... note: run_soak.sh doesn't forward OANDA_MODEL_DIR — export it in the environment first) and any promotion to `models/forex/` are Brandon's explicit calls.
- `main` remains unpushed to origin (now 3 commits + this work ahead).

## Files touched

- `run_oanda.py` — granularity profiles, OANDA_MODEL_DIR, mismatch warning (~45 lines)
- `run_soak.sh` — granularity passthrough (~10 lines)
- `src/strategies/concrete_strategies/ml_strategy.py` — sidecar resolution via angel_path.parent (2 methods)
- `src/core/retrainer.py` — fetch retries + partial-basket hard fail + htf_timeframe metadata (~40 lines)
- `models/forex_m15/` — promoted M15 candidate (gitignored, local only)
- `logs/retrain_m15_2026-07-02_0059.log`, `logs/retrain_m15_2026-07-02_0102.log` — evidence artifacts (untracked)
