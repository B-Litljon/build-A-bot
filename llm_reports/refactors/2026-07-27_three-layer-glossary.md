---
type: refactor
date: 2026-07-27
time: 00:02 PDT
agent: Claude Fable 5
model: claude-fable-5
trigger: "Brandon: build a three-layer glossary documentation layer for the entire repo so he can relearn a codebase that agentic tools grew faster than he could follow"
head: 52a0d9e02903cb959f790c764b5bf681ad574971
scope: modifies-source
related:
  - refactors/2026-07-26_concurrency-race-fix.md
files_touched:
  - GLOSSARY.md (new, 419 lines)
  - CLAUDE.md (new)
  - 19 folder README.md files (new)
  - 97 .py files (module docstrings only)
---

## Context

Brandon built the early version of this codebase by hand, then agentic tools
grew it faster than he could follow. He no longer knows what many identifiers
mean. The ask was a documentation layer that lets him relearn his own system,
under one explicit constraint: **accuracy over completeness — a confidently
wrong explanation is worse than no explanation.** Never guess a meaning from a
name; derive it from the code and its call sites.

Three layers were specified:

1. **`GLOSSARY.md`** — repository map + the project's shared vocabulary.
2. **`README.md` per code folder** — per file: what it is, what it imports from
   the repo, what imports it, which data artifacts it touches.
3. **`Glossary:` in every module docstring** — the meaningful identifiers inside
   that file.

Hard constraints: docs only, zero executable-code changes; byte-compile and the
test suite must stay as green as before; work folder-by-folder with one commit
per folder; add a maintenance rule to `CLAUDE.md`.

**Precondition (gate).** `git status` was run first and the gate tripped: the
2026-07-26 concurrency race fix was uncommitted (by Brandon's own prior
instruction to leave it for review), as was older watchdog work
(`soak_watchdog.sh` + a `.gitignore` line, live in cron since 2026-07-11 but
never tracked). Work stopped and reported. Brandon said "go ahead and commit."
Both were committed separately (`76a1b5a` fix, `b47dade` chore) on `main` before
branching `docs/glossary`, so the docs diff stays untangled.

## Investigation

Every file was read before being documented. Nothing was inferred from a
filename. Where a docstring made a claim, the claim was checked against the
code. Representative traces:

**`V3RandomForestTrainer` holds LightGBM.** The class wraps
`RandomForestClassifier`, but `load()` does `self.model = joblib.load(path)`,
replacing the estimator wholesale. Traced the call site:
`ml_strategy.py:124-126` constructs `V3RandomForestTrainer()` then immediately
`.load()`s a pickle the retrainer wrote via `joblib.dump(angel_model)` where
`angel_model` is an `lgb.LGBMClassifier`. So a variable named for a random
forest holds a gradient-boosted model at runtime. Documented as verified, not
assumed.

**`Signal.raw_tp_distance` is dead on arrival.** Grepped every reference:
`ml_strategy.py:652-653` writes both distance fields to the same value
(`atr_abs`); the only consumers are `factory_orchestrator.py:115` and
`oanda_scalper_orchestrator.py:530`, and **both read only `raw_sl_distance`**.
Nothing anywhere reads `raw_tp_distance`. This matches the intentional ruling
that `RiskManager` multipliers own target sizing, so it is documented as
vestigial-by-design rather than a bug.

**Bracket multipliers differ by asset class.** `retrainer.SL_ATR_MULTIPLIER` /
`TP_ATR_MULTIPLIER` are 0.5/3.0, but `get_asset_config()` sources the real
values from `RiskProfile.for_asset_class()`, and the forex branch
(`risk_manager.py:144-146`) overrides to 1.0/2.0. An earlier draft of the
retrainer glossary described the constants as authoritative; this was **caught
and corrected before commit** — reasoning from the module constants alone is
wrong for every forex run.

**`resolver.py` is orphaned.** Its own usage line says
`python -m src.core.resolver`, but `run_pipeline.sh:157` Phase 3 actually runs
`python -m src.evaluate_performance`. Nothing imports resolver. Corroborated
later by a format split: `replay_test.py` writes `signal_ledger.**parquet**`
and `evaluate_performance.py` / `reinforcement_voter.py` read parquet, while
`resolver.py` reads `signal_ledger.**csv**` — and both files exist on disk.

**`backtest_60.py` is broken.** It calls
`MLStrategy(model_path=..., threshold=0.60)` — neither kwarg exists on the
current constructor, so both are swallowed by `**kwargs` and the model paths
fall back to `models/equities/`, which does not exist → `FileNotFoundError`.
It then calls `strategy.get_order_params()`; grepping `def get_order_params`
across the repo returns nothing. It expects the `OrderParams` shape from
`core/order_management.py`, itself dead. The two are dead together.

**`tests/verify_warmup.py` never runs.** It contains a real
`IsolatedAsyncioTestCase`. Counting `def test_` across the suite gives 115, but
`pytest --collect-only` reports 114. `pyproject.toml` sets
`testpaths = ["tests"]`, but collection *also* requires the default `test_*.py`
filename pattern, which `verify_warmup.py` does not match.

**Train/live symmetry confirmed (a positive finding).** The feature-generator
order in `ml_strategy.py:170-177` and `retrainer.py:878-891` is identical:
`[V3BaseFeatures, V3HTFFeatures, V3SessionFeatures, V3CostFeatures]`. This is
the property the whole `src/ml` design exists to protect, and it holds.

**Operational check.** `ps` confirmed the M15 soak is live (PID 10385, up since
Jul 13) with `soak_watchdog.sh` in cron every 5 minutes and no `soak.off`
present. Since the watchdog relaunches from the working tree, `src/execution/`
edits were audited especially carefully (see Verification).

## Findings / Changes

### Coverage

| Metric | Result |
|---|---|
| Python files with a `Glossary:` section | **97 / 97 (100%)** |
| Folder READMEs written | **19** |
| Domain terms defined in `GLOSSARY.md` | **78** |
| Branch total | 118 files changed, 4,682 insertions, **4 deletions** |
| Commits | 10 (one per folder group + Layer 1) |

The 4 deletions are one-line docstrings expanded into multi-line ones
(`ml/regimes/__init__.py`, `strategies/base.py`,
`tests/test_composite_fundamentals.py`, `tests/test_feature_stats.py`).
**Zero executable lines were removed or altered anywhere on the branch.**

### Per-layer

**Layer 1 — `GLOSSARY.md` (419 lines).** Repository map covering every
top-level folder and each `src/` subpackage, with what runs when
(training / analysis / live). Domain glossary of 78 terms grouped by theme
(the two-stage model; bars and time; prices, volatility and cost; the gates;
training; scoring; diagnosing a quiet model; live operation; artifacts). It
deliberately disambiguates the words that mean two different things here —
`Signal`, `regime`, `watchdog`, `heartbeat`, `fetch_training_data` — because
each is a live trap. It complements `table-o-content.md` (the narrative tour)
rather than duplicating it.

**Layer 2 — 19 folder READMEs.** Each opens with how the folder fits the
pipeline, then one entry per file with repo-internal imports, importers, and
data artifacts, all derived from actual import statements and I/O call sites.

**Layer 3 — 97 module headers.** Format
`name -- one plain-language line: what it is, units/range, why it exists`.
Domain terms point at `GLOSSARY.md` instead of being redefined.

**`CLAUDE.md` (new).** Carries the required maintenance rule verbatim, plus the
Layer-3 conventions, operational safety notes (the soak runs from the working
tree; `touch soak.off` before killing), and a "things that will mislead you"
list of the verified name/reality mismatches.

### Flagged — dead, duplicated, or misnamed (**flag only, nothing fixed**)

**Broken:**

1. `backtest_60.py` — cannot run: nonexistent constructor kwargs → missing
   `models/equities/`; and `get_order_params()` is defined nowhere.
2. `tests/verify_warmup.py` — a real test that pytest never collects
   (filename doesn't match `test_*.py`). Renaming would fix it and move the
   suite count from 114 to 115.

**Dead / orphaned:**

3. `src/utils/risk_management.py` — 0 bytes, nothing imports it. The real
   `RiskManager` is `src/execution/risk_manager.py:178`. Name collision.
4. `src/core/order_management.py` (`OrderParams`) — nothing imports it, and the
   `grid_search_backtest*.py` it says it exists for no longer exists.
5. `src/core/ws_stream_simulator.py` — nothing imports it.
6. `src/core/resolver.py` — orphaned; `run_pipeline.sh` runs
   `evaluate_performance` instead, despite resolver's own usage line.
7. `src/data/fetch_training_data.py` — dead module, **and** it collides by name
   with the live `retrainer.fetch_training_data` *function* that
   `chop_ab_test.py` and `scripts/generate_feature_stats.py` actually import.
8. `src/data/discovery.py` — dormant; nothing imports it, no script runs it.
9. `src/autopilot/`, `src/research/` — only stale `__pycache__`, no source.

**Misnamed / misleading:**

10. `V3RandomForestTrainer` holds a LightGBM model at runtime (verified).
11. `yf_macro.py`: `"2Y_YIELD"` maps to `^IRX`, which is the **13-week T-bill**,
    not the 2-year; and `^TNX` returns yield **×10** with no rescaling.
12. OANDA `volume` is **tick count**, not traded size. (Training uses the same
    proxy, so they agree — but the name misleads.)
13. `"regime"` = volatility band in `risk_manager.py`/`reinforcement_voter.py`,
    but HMM hidden state in `src/ml/regimes/`.

**Stale claims in existing docs/comments:**

14. `"18-feature"` in `strategies/base.py` and `ml_factory_strategy.py` — the
    live set is 22 (23 with `cost_ratio`), and `MLStrategy` reads the schema
    from the model rather than hardcoding a count.
15. `replay_test.py` docstring says "CSV export"; it writes parquet.
16. `scripts/smoke_test.py` header says "DO NOT COMMIT"; it is committed, and
    equivalent assertions live in `tests/test_risk_manager.py`.
17. `table-o-content.md` references three files that no longer exist:
    `src/core/trading_bot.py`, `main.py`, `src/main.py`.
18. `retrainer.MIN_OOS_TRADES_FOR_PF` is commented "superseded" but still
    defined and referenced.

**Inconsistencies (cosmetic, no action taken):**

19. Mixed import prefixes repo-wide (`core.X` vs `src.core.X`) depending on
    whether `src` or the repo root is on `PYTHONPATH`.
20. `__init__.py` present in some packages (`utils`, `core`, `ml`,
    `ml/regimes`) and absent in others (`data`, `data/providers`, `ml/core`,
    `ml/features`, `ml/targets`, `ml/trainers`, which work as namespace
    packages).
21. `src/data/feed.py` `warmup_history` has leftover unconditional
    `print("[DEBUG] …")` calls that fire on every warm-up regardless of log
    level.
22. `src/analysis/*` targets the legacy root-level `models/angel_latest.pkl`
    rather than the current `models/<asset_class>/` layout. Those artifacts
    still exist, so the scripts run — they just diagnose the old models.

## Verification

Byte-compile and the full suite were run **after every folder**, before each
commit:

```
python -m compileall -q src/          → OK (every folder, every time)
PYTHONPATH=src:. python -m pytest -q  → 114 passed, 2 warnings
```

Baseline before any documentation work was **114 passed**; final state is
**114 passed**. No test was added, removed, or modified in substance.

**Docs-only proof.** Rather than trusting that docstring edits are inert, the
branch diff was audited for removed lines:

```
git diff main...HEAD | grep "^-" | grep -v "^---"
```

returns exactly four lines, all one-line docstrings that were expanded into
multi-line ones. No executable line was deleted or changed anywhere on the
branch.

**Extra check for `src/execution/`** (the folder feeding the running soak):
`git diff --numstat -- src/execution/` showed **385 insertions, 0 deletions**
across all seven files — every edit a pure docstring insertion. The live soak
(PID 10385) already holds its code in memory and is unaffected; a watchdog
relaunch would pick up byte-identical behaviour.

## Risk & follow-ups

1. **The branch is not merged and not pushed.** `docs/glossary` sits 10 commits
   ahead of `main`; `main` itself is now 4 ahead of `origin/main` (2 pre-existing
   + the race fix + the watchdog chore). Nothing was pushed.
2. **⚠️ The soak is running from this working tree while a non-`main` branch is
   checked out.** That is true of any branch checkout, not something these
   changes caused, but it is worth knowing: if the watchdog relaunches now, it
   launches the `docs/glossary` working tree. Behaviour is identical (docstrings
   only, compile-verified), but merging or returning to `main` removes the
   ambiguity.
3. **Two flagged items are cheap, real fixes** if you want them: renaming
   `tests/verify_warmup.py` → `test_warmup.py` (recovers a test that has never
   run), and deleting the four dead modules (items 3–5, 7). Both are code
   changes and so were deliberately out of scope here.
4. **The `^IRX` / `^TNX` issues (item 11) are live correctness bugs** if anything
   consumes those series numerically — a 10× scale error and a wrong instrument.
   Nothing in the current investor path appears to read them, but that deserves
   its own check rather than my assurance.
5. **Docs rot unless the rule is enforced.** The `CLAUDE.md` rule is the
   mechanism; it only works if agents actually read `CLAUDE.md`, which they do
   automatically, and if reviewers reject changes that add an identifier without
   its glossary line.
6. **No `[UNCLEAR — verify]` markers were needed.** Every identifier documented
   was resolvable by reading code and call sites — so there are no open
   questions for Brandon in this pass. That is a statement about coverage, not
   a guarantee of zero errors: the glossary is 4,682 lines of prose about
   ~25,000 lines of code, and spot-checking any entry against its file is
   welcome.

## Files touched

**New (21):**

- `GLOSSARY.md` (419 lines), `CLAUDE.md`
- `src/README.md`, `scripts/README.md`, `tests/README.md`,
  `tests/execution/README.md`
- `src/{utils,core,data,ml,strategies,execution,analysis,day_trading}/README.md`
- `src/data/providers/README.md`,
  `src/ml/{core,features,targets,trainers,regimes}/README.md`,
  `src/strategies/concrete_strategies/README.md`

**Modified (97 `.py` files — module docstrings only):** all of `src/**`,
`scripts/*.py`, `tests/**`, and the root entry points
(`run_oanda.py`, `run_live.py`, `run_factory.py`, `chop_ab_test.py`,
`backtest_60.py`, `trading_mcp.py`).

**Commits (10, one per folder group):** `f522457` utils · `bd2ad62` core ·
`0033da0` data · `f07578b` ml · `fa7f9ef` strategies · `834d198` execution ·
`981935c` analysis+day_trading · `ab1c162` scripts+entrypoints ·
`3180e2a` tests · `52a0d9e` GLOSSARY.md + CLAUDE.md

**Also committed to `main` before branching** (the precondition gate):
`76a1b5a` the 2026-07-26 concurrency race fix, `b47dade` the untracked
`soak_watchdog.sh` + `.gitignore` entry.
