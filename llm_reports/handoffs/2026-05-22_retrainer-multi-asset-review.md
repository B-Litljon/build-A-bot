---
type: handoff
date: 2026-05-22
time: PDT
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: Peer review of Gemini's "Modular, Multi-Asset Retraining Pipeline" implementation plan + partial uncommitted diff
head: c14298d (with uncommitted modifications to src/core/retrainer.py)
scope: report-only — punch-list for Gemini to action
related:
  - handoffs/2026-05-22_ml-pipeline-state-to-gemini.md
  - refactors/2026-05-14_ml-pipeline-refactor.md
files_touched: []
---

# MODEL-TO-MODEL HANDOFF

**FROM:** Claude Opus 4.7 (Claude Code, repo-resident execution agent)
**TO:** Gemini (Gemini CLI, Antigravity — repo-aware architect peer)
**RE:** Multi-asset retraining pipeline — partial diff review, punch-list before commit
**HEAD:** `c14298d` on `feature/v5-history-prime`, with `src/core/retrainer.py` uncommitted (~343-line diff)
**TONE:** peer-to-peer, prescriptive. Captain B asked me to detail this thoroughly to support a Flash-tier model — code snippets included.

---

## TL;DR

Plan is architecturally sound. The factory-based provider swap, env-var override pattern, and signature parameterization of `engineer_features_and_labels()` / `validate_candidate()` are all the right calls. But the current uncommitted diff has **3 showstopper issues** that will either break runtime or silently violate an existing project ruling. Fix these before commit. Detailed punch-list below with file:line evidence and concrete patch prescriptions.

---

## Part 1 — What you already nailed (verified against the working tree)

1. **Alpaca imports removed** (diff lines ~36-58). `StockHistoricalDataClient`, `StockBarsRequest`, `_AlpacaTimeFrame`, `_ALPACA_TFU`, `_to_alpaca_timeframe()` all gone. Clean.
2. **Factory imports wired** at `src/core/retrainer.py:46-47`:
   ```python
   from src.data.factory import get_market_provider
   from src.data.market_provider import MarketDataProvider
   ```
3. **`get_asset_config()`** at `:66-83` correctly branches on `data_source == "oanda"` with env-var overrides for tickers, SL/TP multipliers, max hold, and survival bars.
4. **`fetch_training_data()` refactor** at `:203-265` properly takes a `MarketDataProvider + symbols` instead of an Alpaca client, calls `provider.get_historical_bars()`, handles empty frames, and adds `symbol` column. Good.
5. **`engineer_features_and_labels()` parameterized** at `:402-407` with `sl_mult`, `tp_mult`, `max_hold`, `survival_bars`. All downstream calls (`_compute_devil_targets_atr`, `_compute_devil_survival_target`) receive them. Good.
6. **`validate_candidate()` parameterized** at `:843-846` with `sl_mult`, `tp_mult`, threaded through to `_find_optimal_threshold()` and the EV/PF calculations at `:1166-1213`. Good.
7. **OANDA provider's `get_historical_bars()`** at `src/data/oanda_provider.py:229-300` already handles pagination (5000-candle chunks, forward-paged via `from + count`). So your fetch will work for 60-day M5 (~12k candles, ~3 requests per symbol). Good.

---

## Part 2 — Showstoppers (block commit)

### Issue #1 — `main()` is broken (won't run)

**Location:** `src/core/retrainer.py:1479-1481`
**Current state:**
```python
# ─── Phase 1: Fetch data ────────────────────────────────────────────
client = get_alpaca_client()           # ← NameError; you deleted this function
logger.info("Alpaca client initialized")
raw_data = fetch_training_data(client) # ← wrong signature now
```

Your refactor deleted `get_alpaca_client()` (was at lines 202-214 pre-diff) and changed `fetch_training_data()` to take `provider, symbols`. But `main()` was never updated. **The retrainer currently does not compile-run.** If you ran it again right now, you'd hit `NameError: name 'get_alpaca_client' is not defined` immediately.

**Fix:** Replace lines 1478-1484 of `main()` with:

```python
# ─── Phase 1: Initialize provider + load asset config ──────────────
data_source = os.getenv("DATA_SOURCE", "alpaca").strip().lower()
asset_config = get_asset_config(data_source)
provider = get_market_provider()
logger.info(
    f"Provider initialized: {provider.__class__.__name__} "
    f"| Asset config: {asset_config['tickers']} "
    f"| SL={asset_config['sl_mult']}× TP={asset_config['tp_mult']}× "
    f"max_hold={asset_config['max_hold']} survival={asset_config['survival_bars']}"
)

# ─── Phase 2: Fetch data ────────────────────────────────────────────
timeframe_minutes = int(os.getenv("RETRAIN_TIMEFRAME_MINUTES", "5" if data_source == "oanda" else "1"))
raw_data = fetch_training_data(
    provider=provider,
    symbols=asset_config["tickers"],
    days_back=DAYS_BACK,
)

# ─── Phase 3: Engineer features ─────────────────────────────────────
features_df, feature_cols = engineer_features_and_labels(
    raw_data,
    sl_mult=asset_config["sl_mult"],
    tp_mult=asset_config["tp_mult"],
    max_hold=asset_config["max_hold"],
    survival_bars=asset_config["survival_bars"],
)
```

And update the `validate_candidate()` call at `:1503` to pass the multipliers:
```python
(
    report,
    angel_model,
    devil_model,
    angel_feats,
    devil_feats,
    optimal_threshold,
) = validate_candidate(
    features_df,
    feature_cols,
    sl_mult=asset_config["sl_mult"],
    tp_mult=asset_config["tp_mult"],
    n_folds=3,
)
```

Also: you also need to thread `timeframe_minutes` into `fetch_training_data()` — right now its signature doesn't accept it and the function hardcodes `timeframe_minutes=1` (diff line ~228). See Issue #4 below for the full fix on that.

---

### Issue #2 — Model path collision (asset classes overwrite each other)

**Location:** `src/core/retrainer.py:86-88`
**Current state:**
```python
MODEL_DIR = Path("models")
ANGEL_PATH = MODEL_DIR / "angel_latest.pkl"
DEVIL_PATH = MODEL_DIR / "devil_latest.pkl"
```

These are global constants. There is **no asset-class differentiation in the save path**. Consequence: if you run equities retraining Tuesday and forex retraining Wednesday, Wednesday's forex weights overwrite `models/angel_latest.pkl` and `models/devil_latest.pkl` — the exact paths the live Oanda scalper AND any equities strategy both load from. There is no way for the strategy at load time to know which asset class the model on disk was trained on.

**Why this is dangerous beyond just "files overwriting":** the schema drift guard at `src/strategies/concrete_strategies/ml_strategy.py:167` only validates **feature column names**. If both forex and equities retraining produce models with the same 18 V3 features (they do — feature engineering is asset-agnostic in this design), the guard cannot tell them apart. An equities-trained model loaded into the OANDA scalper passes every existing safety check.

**Fix:** Namespace the paths by asset class. Two options:

**Option A — Directory namespace (preferred, no config changes needed elsewhere):**

```python
ASSET_CLASS = "forex" if os.getenv("DATA_SOURCE", "alpaca").strip().lower() == "oanda" else "equities"
MODEL_DIR = Path("models") / ASSET_CLASS
ANGEL_PATH = MODEL_DIR / "angel_latest.pkl"
DEVIL_PATH = MODEL_DIR / "devil_latest.pkl"
```

This requires updating `MLStrategy.__init__` defaults (`src/strategies/concrete_strategies/ml_strategy.py:67-68`) from `"models/angel_latest.pkl"` to the asset-specific path the strategy was instantiated for. Suggest passing an `asset_class` kwarg, or letting the caller (`run_oanda.py`) pass explicit paths.

**Option B — Suffix in filename (lighter touch):**

```python
asset_class = "forex" if data_source == "oanda" else "equities"
ANGEL_PATH = MODEL_DIR / f"angel_{asset_class}_latest.pkl"
DEVIL_PATH = MODEL_DIR / f"devil_{asset_class}_latest.pkl"
```

Same MLStrategy update needed.

**Either way: also stamp the asset class into model metadata** (a sidecar `models/<class>/metadata.json` with `{"asset_class": "forex", "trained_at": "...", "trained_on_symbols": [...], "data_source": "oanda"}`) so post-hoc audits don't depend on the filename. MLStrategy should validate metadata against the asset class it expects.

---

### Issue #3 — Training-execution asymmetry (violates `[[tp-distance-ruling]]`)

**This is the architectural one. Read carefully.**

**Project ruling on record** (memory `project_tp_distance_ruling.md`):
> *"signal.raw_tp_distance intentionally discarded; RiskManager multipliers own bracket sizing to preserve training-execution symmetry"*

The principle: whatever SL/TP regime the Devil was trained under MUST match what the live `RiskManager` applies at execution. Otherwise the Devil's "this trade will survive" prediction is calibrated for a different stop than the trade actually gets.

**Your proposed forex config** (`src/core/retrainer.py:71-73`):
```python
"sl_mult": float(os.getenv("RETRAIN_SL_MULT", "1.0")),
"tp_mult": float(os.getenv("RETRAIN_TP_MULT", "2.0")),
```

**Live execution config** (`src/execution/risk_manager.py:9-10`):
```python
sl_atr_multiplier: float = 0.5
tp_atr_multiplier: float = 3.0
```

**The asymmetry:**
- Devil is trained to predict "did the trade survive a stop at 1.0×ATR for 5 bars?"
- Live execution sets the stop at **0.5×ATR** (half the distance — tighter, easier to breach)
- Therefore: a Devil that says "high survival probability" was trained against a stop **twice as far away** as the one the trade actually runs with. The trade gets stopped out far more often than the Devil's training distribution would predict.

The TP side has the inverse problem: Devil trained on 2.0×ATR target, execution aims for 3.0×ATR target. Devil's macro-target win rate (used during threshold calibration) was calibrated against a target the trade may not actually reach.

**This is exactly the bug the user's TP-distance ruling exists to prevent.** Per the memory: *"RiskManager multipliers own bracket sizing."* You cannot diverge the retrainer's multipliers from the live `RiskProfile`'s multipliers without re-opening that ruling.

**Resolution paths (in order of architectural cleanness):**

**Path A — Make `RiskProfile` asset-class-aware, match retrainer to it (best):**

Add a factory in `src/execution/risk_manager.py`:
```python
@classmethod
def for_asset_class(cls, asset_class: str) -> "RiskProfile":
    if asset_class == "forex":
        return cls(
            sl_atr_multiplier=1.0,
            tp_atr_multiplier=2.0,
            min_sl_pips=2.0,
            round_precision=5,
        )
    return cls()  # equities defaults
```

Then in `run_oanda.py:130`:
```python
risk_profile = RiskProfile.for_asset_class("forex")
```

And in the retrainer, instead of duplicating the multiplier values in `get_asset_config()`, **read them from `RiskProfile.for_asset_class(...)`** so they cannot drift:
```python
from src.execution.risk_manager import RiskProfile
def get_asset_config(data_source: str) -> dict:
    asset_class = "forex" if data_source == "oanda" else "equities"
    profile = RiskProfile.for_asset_class(asset_class)
    return {
        "tickers": [...],
        "sl_mult": profile.sl_atr_multiplier,
        "tp_mult": profile.tp_atr_multiplier,
        "max_hold": ...,
        "survival_bars": ...,
    }
```

This is the symmetry-preserving design. Single source of truth (`RiskProfile.for_asset_class`), both retrainer and live execution read from it.

**Path B — Keep retrainer multipliers as the source of truth, derive RiskProfile from them at runtime:**

Less clean (forces execution to read asset-config at risk-manager construction time), but viable.

**Path C — Reject the asymmetry: force retrainer to use `RiskProfile()` defaults regardless of asset class:**

```python
"sl_mult": 0.5,  # MUST match RiskProfile.sl_atr_multiplier
"tp_mult": 3.0,  # MUST match RiskProfile.tp_atr_multiplier
```

This makes the asymmetry impossible but throws away your asset-class tuning. The argument *for* this path: maybe forex doesn't actually need wider stops — the V3.3 Angel/Devil were designed around 0.5×ATR / 3.0×ATR and the asset-class tuning is a guess. The argument *against*: forex spreads compress the effective stop distance, so a 0.5×ATR stop on EUR/USD may be inside the spread on volatile bars.

**My recommendation: Path A.** It's the right place for asset-class config (the risk layer), and it eliminates the silent-drift class of bug entirely. The user's TP-distance ruling was written to prevent exactly this kind of divergence.

**Do NOT proceed with the forex retrain until this is resolved.** A forex model trained under Path A's matched multipliers is valid. A forex model trained under your current `1.0 / 2.0` defaults with live execution at `0.5 / 3.0` is broken-by-design.

---

## Part 3 — Important issues (fix before forex deployment)

### Issue #4 — Forex timeframe mismatch (M1 train, M5 deploy)

**Location:** Your `fetch_training_data()` at the diff's line ~228 hardcodes `timeframe_minutes=1`:
```python
df = provider.get_historical_bars(
    symbol=ticker,
    timeframe_minutes=1,
    start=start_date,
    end=end_date,
)
```

**Live forex stream defaults to M5** via `run_oanda.py:88` (`--granularity 5`), and `MLStrategy` is instantiated with `timeframe=5, htf_timeframe="30m"` (lines 116-122 of `run_oanda.py`).

So: retrainer fetches M1 forex bars → trains Devil on M1 microstructure → live deploys against M5 bars. Same class of asymmetry as Issue #3, applied to the bar resolution. The Devil's microstructure features (range_coil_10, bar_body_pct, bar_upper_wick_pct, bar_lower_wick_pct) have completely different statistical distributions at M1 vs. M5.

**Fix:** Add `timeframe_minutes` to `get_asset_config()` and pass it through to `fetch_training_data()`:

```python
# in get_asset_config(), oanda branch:
"timeframe_minutes": int(os.getenv("RETRAIN_TIMEFRAME_MINUTES", "5")),
# equities branch:
"timeframe_minutes": int(os.getenv("RETRAIN_TIMEFRAME_MINUTES", "1")),
```

And update the `fetch_training_data()` signature to accept it. Default it to 1 for backward compat with the existing equities flow.

---

### Issue #5 — Schema drift guard cannot catch distribution drift

This isn't a code fix — it's a documentation / architectural awareness point that needs to land in the project memory before forex models go live.

The guard at `ml_strategy.py:167-172` validates `feature_names_in_`. It compares column names, in order. It does NOT validate:
- The training data's asset class
- The training data's timeframe
- The training data's date range
- The statistical distribution of features at training time vs. inference time

In your design, equities and forex retraining produce models with **identical feature schemas** (the 18 V3 features + `angel_prob` for Devil). The guard cannot tell them apart. A forex model loaded into an equities strategy passes. An equities model loaded into the forex strategy passes.

**Mitigation:** The per-asset-class model paths from Issue #2 give you path-level isolation. Layer on a sidecar `metadata.json` per model with explicit asset class, timeframe, training symbols, and training date range. Have `MLStrategy.__init__` validate `metadata.asset_class == self.expected_asset_class`. This closes the loop.

---

## Part 4 — Polish (not blocking, do when convenient)

### Issue #6 — Asset class detection by `DATA_SOURCE` is fragile

`get_asset_config()` keys off `data_source == "oanda"`. What happens when:
- A second forex provider gets added (e.g., `data_source="ig"` or `data_source="oanda_sandbox"`)?
- Someone wants to backtest forex strategies using a hypothetical equity-side-emulated provider?

Better: explicit `ASSET_CLASS` env var (`forex` / `equities` / `crypto`), or a helper `_asset_class_for_source(data_source: str) -> str` with a registry. Conceptually, asset class and data source are orthogonal axes.

### Issue #7 — Stylistic redundancy in default tickers

`EQUITIES_TICKERS` and `FOREX_TICKERS` are module-level constants AND the `os.getenv` default. Slight DRY violation. Could collapse to:

```python
_DEFAULT_TICKERS_BY_CLASS = {
    "forex": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"],
    "equities": ["TSLA", "NVDA", "MARA", "COIN", "SMCI"],
}
```

Cosmetic, defer until other changes settle.

---

## Part 5 — Suggested execution order

If Captain B greenlights this punch-list:

1. **Fix Issue #1 first** (wire `main()`). Without it, nothing else can be verified. After patch, run with `DATA_SOURCE=alpaca` (equities) end-to-end. Expected outcome: equities retrain completes successfully, produces models at the equities-namespaced path, validation gates pass. If equities path now breaks where it didn't before, the refactor introduced a regression — diagnose before continuing.
2. **Fix Issue #2** (path namespacing). Equities models already exist at `models/angel_latest.pkl` etc. Need a migration: either move them to `models/equities/` or leave `latest` symlinks in place. Do not orphan the production models.
3. **Resolve Issue #3** with Captain B (architectural decision, not a code call). My recommendation is Path A but it's his ruling to make.
4. **Fix Issue #4** (timeframe parameter). Trivial once #1 is done.
5. **Add Issue #5 mitigation** (metadata sidecar). Worth a small protocol design — propose a schema before implementing.
6. **Then and only then**: forex dry-run retrain. `DATA_SOURCE=oanda` should produce forex-namespaced models, validation gates should pass independently, MLStrategy should load them under matched RiskProfile.

---

## Part 6 — One more verification flag for your review

The validation gate results from your previous equities retrain (Brier 0.2676, EV 1.809180, PF 3.4) are suspiciously rosy for the V3.3 Angel/Devil meta-labeling architecture on a 60-day window. Possibilities:
- Genuine improvement from the Polars-native fit fix (unlikely — that's a metadata-only change)
- Walk-forward leakage somewhere in `validate_candidate()`
- The `BRIER_THRESHOLD = 0.30` (raised from 0.25 per the inline comment at `:130-133`) is generous given the ~45% base rate

Not asking you to investigate now — just flagging it. After Issues #1-4 are fixed, when you run the *next* equities retrain, compare validation metrics to this run. If they're radically different, the discrepancy might point to a leakage bug that the rosy numbers are masking.

---

**END HANDOFF.**

Captain B will adjudicate Issue #3. The rest are yours to pick up at your discretion.
