---
type: refactor
date: 2026-09-02
time: 14:05 PDT
agent: Claude Opus 5
model: claude-opus-5
trigger: "Brandon: review the multi-strategy/router build from GLM-5.3-Flash and Gemini 3.8 Flash, then fix what's wrong."
head: 51646e92f0edd0a124c18f041b377071f348eaa9
branch: feat/trim-100x15-retrain
scope: modifies-source
related:
  - recons/2026-09-02_strategy-library-behavior-matrix.md
  - handoffs/2026-09-01_multi-strategy-regime-router-crypto-binance.md
files_touched:
  - src/analysis/strategy_backtester.py
  - src/analysis/walk_forward_tuner.py
  - src/analysis/build_strategy_matrix.py
  - src/strategies/concrete_strategies/regime_router.py
  - run_oanda.py
  - config/regime_routing_forex.example.json
  - config/regime_routing_equities.example.json
  - tests/test_strategy_backtester.py
  - tests/test_walk_forward_tuner.py
  - tests/test_regime_router.py
  - GLOSSARY.md
  - src/analysis/README.md
  - src/strategies/concrete_strategies/README.md
---

# The strategy scorer was paying for trades that never happened

## Context

A strategy library, a regime router and a walk-forward tuner arrived
uncommitted in the working tree, built by Gemini 3.8 Flash from the 2026-09-01
handoff. Crypto was correctly left out — Brandon cancelled that direction before
the build started and there is no Binance code anywhere in the tree.

Two reviews ran over it: GLM-5.3-Flash, and this session. Between them they
found eight defects. They overlapped on one (timeout R-booking) and found four
and three respectively that the other missed. This report covers the fixes; the
measurement that followed is in the companion recon.

**The architecture was sound.** The router stands down by default, guards the
cold window, tags causally, and — most importantly — the backtester wrote its
own direction-aware bracket walk instead of reusing
`retrainer._compute_devil_targets_atr`, which is long-only and would have
silently inverted the answer key on every short. That was the single worst trap
available and the build stepped around it.

What was wrong was the *measurement*, in ways that all pointed the same
direction: flattering.

## Investigation

### The routing tables were never measured

The finding that matters most is not a bug in the code.

`config/regime_routing_forex.json` was written at **13:05:51**.
`src/analysis/strategy_backtester.py` was written at **13:05:39** — twelve
seconds earlier. No matrix output existed anywhere in the tree: no
`logs/strategy_*`, nothing new in `analysis_cache/`. And
`build_strategy_matrix.py` writes to `config/regime_routing.json`, while the
files on disk were `regime_routing_forex.json` and
`regime_routing_equities.json` — different names, so the generator had never
produced them.

Both tables were hand-authored. They read like findings — forex routed
`range_normal` to momentum, equities routed `trend_low` to an SMA cross — but
every assignment was a textbook prior. Meanwhile `run_oanda.py` had gained a
`--strategy regime_router` path defaulting straight at one of them.

### Timeout trades were booked at the bracket's full payoff

`strategy_backtester.py` resolved a timeout by sign alone:

```python
if exit_reason == "timeout":
    macro_win = 1 if exit_price >= entry_price else 0
gross_r_val = payoff_ratio if macro_win else -1.0     # payoff_ratio = tp/sl = 2.0
```

A trade that never touched either level, ending one pip above entry, was paid a
full **+2R**. Demonstrated on a synthetic market that drifts 0.6% total and
touches nothing:

```
trades=13  win_rate=100.0%  gross_ev_r=+2.000  net_ev_r=+1.670  PF=inf
```

The repo's own convention (`_compute_devil_targets_atr`) scores a timeout as a
loss. On real GBP_JPY M15 bars this inflated win rate by 10–12 points on every
strategy in the library.

### A gap through the stop filled at the stop

The exit walk tested `low[j] <= sl_price` and then filled at `sl_price`. When a
bar *opens* below the stop, that price never traded — the fill is the open. On
FX M15 this is near-noise; on a weekend gap it is not, and the convention
would have travelled straight into any future crypto work.

### The live gates never ran

The module docstring said *"Brackets trades via RiskManager multipliers"*.
`grep` found exactly one occurrence of `RiskManager` in the file: that sentence.
There was no import. Gates A, B and C were all absent, and the cost was the flat
`DEFAULT_TOLL_R = 0.33` rather than the measured per-instrument alphas, which
span **0.072 to 0.903** — a 12.6× range no constant covers at both ends.

Gate B is the one that bites. It vetoes the bottom 20% of the trailing
volatility window; the tagger's `low` band is the bottom third — the *same
computation on the same 260-bar window*. So Gate B removes roughly **60% of the
entire `low` band**, and a gateless run scores a population the bot would never
trade.

### The tuner computed an evidence bound and ignored it

`TunedParamResult.robust` required positive expectancy, ≥30 trades and ≥50%
positive folds — then never consulted the Clopper-Pearson bound the module had
gone to the trouble of computing. A lucky 31-trade run read as robust on its
point estimate.

### Validation folds started cold

Each fold sliced at exactly `val_start`, but `run_backtest` skips its first
`warmup_period` bars. Every fold therefore spent its whole warmup window unable
to trade, silently discarding those bars and resetting trailing state that live
never resets.

### Two names for one strategy

The router registered every sub-strategy twice — once as `snake_case`, once as
`ClassName` — pointing at the same instance. A routing table could name the same
thing two ways, and a rename would half-break in silence.

## Findings / Changes

**1. Realised R replaces nominal payoff.** Every exit now books
`signed_move / sl_dist`. This is exact for a clean level hit (a fill at the
target is precisely +2R), naturally bounded inside (−1, +payoff) for a timeout
that touched nothing, and correctly *worse* than −1R when a gap blew through the
stop — a real loss the old convention hid.

`macro_win` stays binary and bracket-resolution-based so the ledger still feeds
`behavior_matrix.score_ledger`. The two columns now answer different questions
and are meant to disagree on a timeout; that is stated at the top of the file
rather than left to be discovered.

**2. Gap fills at the open.** New `sl_gap` / `tp_gap` exit reasons, checked
before the intrabar test because a gap resolves the bar before any within-bar
path matters. A stop gapped through at 90 against a 2-point stop now books
−5.00R instead of pretending a fill at 98 for −1.00R.

**3. The live gates run.** `run_backtest` accepts a `RiskManager`; vetoed
signals are counted in a new `gate_rejections` funnel instead of being traded.
The funnel is a result in its own right — a regime whose picks are mostly
gate-vetoed is not reachable live however good the survivors look. It also
accepts `spread_alphas` for a per-instrument toll mirroring Gate A's own proxy,
so the cost charged in the ledger is the cost the gate reasoned about.

The `RiskManager` is a `TYPE_CHECKING`-only annotation — the instance is passed
in. The scorer's import graph reaches neither `core.retrainer` nor `lightgbm`
nor `models/`, which is what makes it safe to run beside the live soak.

**4. `robust` now uses its bound.** The Clopper-Pearson lower bound on the
pooled OOS win rate must clear the bracket's break-even rate, `1/(1+payoff)` —
33.3% at the live 2:1.

**5. Folds start warm.** Validation slices are prefixed with `warmup_period`
bars of prior history, exactly consumed by the warmup, so scoring still begins
precisely at `val_start`. Plus a fresh strategy instance per fold, so no fitted
state can cross a boundary.

**6. One canonical name per strategy.** `ClassName` spellings still resolve via
`_resolve_name`, but are no longer a second registry entry. Keys beginning `_`
are treated as documentation and dropped on load.

**7. The routing tables are marked as what they are.** Renamed to
`*.example.json`, each carrying a `_WARNING` field stating they were authored by
hand before any measurement existed. `run_oanda.py` now has **no default routing
config** and refuses to start the router without an explicit one. Selecting any
bare library strategy logs a warning that it has no measured edge.

**8. `build_strategy_matrix.py` rebuilt for a real run** — a basket rather than
one symbol, gates on, measured spreads, cells scored from the ledger's own
realised R, the house floor of 30 trades restored (it had been lowered to 15),
and a significance requirement so a cell cannot win a regime on a point
estimate. It also trims every frame to the window they all cover: a legacy
parquet in `data/raw` spanned a different two years from a fresh fetch, and
pooling those would have mixed market eras inside a single cell.

## Verification

```
PYTHONPATH=src:. python -m pytest -q     →  391 passed   (was 380; +11)
python -m compileall -q src/ run_oanda.py →  clean
systemctl --user is-active soak.service   →  active
git status --short models/                →  empty
```

Eleven new tests, each pinned to a specific defect rather than to the
implementation:

| Test | Pins |
|---|---|
| `test_timeout_is_not_a_win_and_pays_only_the_realised_move` | the 100%-win-rate flat market |
| `test_gap_through_stop_fills_at_the_open_not_the_stop_price` | −5R, not −1R |
| `test_gap_through_target_fills_at_the_open_and_counts_as_a_win` | the favourable direction too |
| `test_clean_level_hits_still_book_exactly_the_nominal_payoff` | realised R must not disturb the ordinary case |
| `test_gates_veto_signals_and_the_funnel_is_recorded` | gates fire, funnel populated |
| `test_spread_alphas_produce_a_per_instrument_toll` | a wider spread costs more; gross is cost-free |
| `test_robust_requires_the_clopper_pearson_bound_to_clear_breakeven` | and the other three bars still bind |
| `test_oos_folds_are_warm_so_no_validation_bars_are_silently_discarded` | fold warmth |
| `test_sub_strategies_registered_once_under_a_canonical_name` | no aliasing |
| `test_classname_spelling_still_resolves_to_the_canonical_key` | backward compatibility |
| `test_documentation_keys_are_not_treated_as_routes` | `_WARNING` is not a regime |

Writing the gate test surfaced something worth recording: a *flat* low-volatility
stretch does **not** trip Gate B. The gate ranks the current bar inside its own
trailing window, so a constant series ranks at 1.0. Only a **decaying** series
keeps the newest bar at the bottom of its own window and stays vetoed. The first
version of that test asserted on flat data and passed for the wrong reason.

Live-behaviour smoke test:

```
$ run_oanda.py --strategy regime_router --granularity 15
--strategy regime_router requires --routing-config (or OANDA_ROUTING_CONFIG).
The tables in config/ are hand-authored templates, not measurements; generate a
real one with src/analysis/build_strategy_matrix.py first.
```

Documentation, per the CLAUDE.md three-layer rule (GLM correctly flagged that
layers 1 and 2 had been skipped):

- **Layer 1** — `GLOSSARY.md` gained a *strategy library and router* section
  (strategy library, regime router, routing table, stand down, R, realised R,
  gap fill, gate funnel) and a correction to the stale `candidate` entry, which
  described a model directory, threshold pair and chop-gate config that
  `behavior_matrix.Candidate` has never carried.
- **Layer 2** — `src/analysis/README.md` and
  `src/strategies/concrete_strategies/README.md` gained entries for all nine new
  modules with imports / imported-by / reads-writes.
- **Layer 3** — `Glossary:` sections extended for the new identifiers
  (`gate_rejections`, `trade_toll`, `pooled_oos_cp_lb`, `breakeven_win_rate`,
  `val_prefix`, `_resolve_name`).

## Risk & follow-ups

1. **The soak was never at risk and is still running** (PID 3365348, up since
   2026-08-30). It holds the old `run_oanda.py` in memory. The modified file
   matters only on restart, and `--strategy` defaults to `ml_strategy`, so the
   served path is unchanged. One nuance worth knowing: `run_oanda.py` now
   imports all six strategies at module load, so an import error in any new
   strategy would take the bot down on its next restart and the watchdog's
   crash-loop brake would hold it down 15 minutes. It imports cleanly today, but
   the blast radius grew for no live benefit.
2. **`macro_win` and `net_r` disagree on timeouts by design.** Anyone reading
   `win_rate` as "profitability" will misread this ledger. It is documented in
   the module header and in both READMEs; it is still the most likely
   misinterpretation.
3. **Gate A runs its proxy-spread branch offline**, so gate behaviour
   approximates live rather than reproducing it — the same limitation
   `retrainer._compute_chop_veto_mask` already carries.
4. **The equities routing template is still unmeasured.** Only forex was
   measured. `config/regime_routing_equities.example.json` keeps its warning and
   should not be used.
5. **Nothing is committed.** All work is uncommitted on
   `feat/trim-100x15-retrain`.

## Files touched

Modified: `src/analysis/strategy_backtester.py`,
`src/analysis/walk_forward_tuner.py`, `src/analysis/build_strategy_matrix.py`,
`src/strategies/concrete_strategies/regime_router.py`, `run_oanda.py`,
`tests/test_strategy_backtester.py`, `tests/test_walk_forward_tuner.py`,
`tests/test_regime_router.py`, `GLOSSARY.md`, `src/analysis/README.md`,
`src/strategies/concrete_strategies/README.md`.

Renamed: `config/regime_routing_forex.json` →
`config/regime_routing_forex.example.json`; same for the equities table.

Written by the measurement run: `config/regime_routing_forex.json` (real, all
stand-downs), `logs/strategy_matrix.csv`,
`analysis_cache/strategy_matrix/*.parquet`.
