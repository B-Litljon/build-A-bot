---
type: handoff
date: 2026-09-22
time: 09:05 PDT
agent: Claude (architect sub-agent)
model: kimi-k3
trigger: "Brandon: green light on the v2 feature-lab work, dependency-ordered; a build handoff for the next agent to follow."
head: a1a60a1
scope: read-only
related:
  - handoffs/2026-09-22_feature-lab-v2.md
  - handoffs/2026-09-21_feature-lab-plan.md
  - refactors/2026-09-21_feature-lab-built.md
---

# Feature lab v2 — build tasks, dependency-ordered (for the next agent)

## Context

Brandon has green-lit the v2 feature-lab work. This is the executable handoff. The
review that produced it is `handoffs/2026-09-22_feature-lab-v2.md` — read that for
the *why*; this file is the *what*, in the exact order to do it. The lab itself was
built 2026-09-21 and is committed (`a1a60a1`); the working tree is clean.

**Dependency order is not optional.** W1 must land first, because W3's ablation runs
cache their frames on `FeatureSpec.content_hash` — while W1 is open, an ablation run
can silently reuse a stale frame. W4 is a gate at the end, not code: it decides
whether a bespoke architecture harness is ever built (deliberately out of scope here).

## Agent contract — read before starting

1. **Branch first.** `git status` must be clean; then `git checkout -b <name>` and do
   all work there. **Do not commit.** Brandon reviews and commits. Suggested branch:
   `lab/w1-w4`. (This is the standing rule Brandon set this session.)
2. **Never touch the live path.** The OANDA soak runs from this tree under
   `soak.service` (verify with `systemctl --user is-active soak.service` before and
   after). Do not modify anything under `src/execution/`, `run_oanda.py`, or
   `models/forex_m15_wide/`. The lab is offline-only; keep it that way — nothing in
   the live boot path may import `src/lab/`.
3. **No placeholders.** No `pass` bodies, no `TODO` stubs, no `Any`-typed public
   surfaces added where a real type/Protocol exists. Keep the existing type-annotation
   discipline (the current lab is fully annotated).
4. **Tests match repo convention** (`tests/README.md`): deterministic seeds, no
   network, orchestrators via `__new__`, failure-path-first. Add files under
   `tests/test_lab_*.py`.
5. **Verify at the end** and quote real output in your report:
   `PYTHONPATH=src:. <venv python> -m pytest -q` and `python -m compileall -q src/`.
   (Venv: `/home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python`.)
6. **When done, write** `llm_reports/recons/<date>_lab-v2-build.md` (or refactors/ if
   you also ship the `_features.py`-adjacent docs) with the six-standard sections,
   listing every file touched. Report the branch name and that nothing is committed.

---

## W1 — Frame-cache hash covers generator state (the polypharmacy-era cache hole)

**Why first:** `FeatureSpec.content_hash()` (`src/lab/spec.py:167-185`) currently
hashes spec fields + spread-table bytes + a risk-profile fingerprint + frame-affecting
env keys, but reduces any `BaseFeatureGenerator` to `_generator_id(gen) =
f"{module}.{qualname}"` (`spec.py:188-196`). Editing a lookback constant *inside* a
registered generator (e.g. `LabMicrostructureFeatures._WINDOW`) leaves the hash
unchanged → a stale cached frame is silently reused, while the docs promise the
opposite. Two instances of the same class with different constructor args also collide.

**Do:**
- Require a per-family version and fold it into the hash: extend
  `registry.register_family`/`register_feature` to take a required `version: int`
  (fail loudly when it's missing), and include `(name, version)` for every family in
  `spec.feature_sets` when hashing.
- For `extra_generators`: hash the generator's resolved state
  (`dataclasses.asdict(gen)` if a dataclass, else `vars(gen)`, with a deterministic
  sort) alongside its class id. **If a generator's state is not JSON-h**ashable,
  `content_hash()` must **raise**, not fall back to the class-only id — no silent
  best-effort hash (the current bug is exactly such a fallthrough).
- Keep `model_family` **out** of the frame hash — it changes the model, not the
  features (it belongs to run provenance; this matters for W4 frame reuse).

**Tests:** a generator defining a constant and re-registering it with a bumped
version changes the hash; a dataclass generator with differing constructor args
hashes differently; a non-hashable-state generator raises from `content_hash()`.
A test asserting the *claim in the README* ("change a lookback → new frame") now
holds end-to-end.

## W2 — Backtest uses the gate's own feature lists, not a re-derived copy

**Why:** `run_model_backtest` (`src/lab/backtest.py:240-253`) discards
`gate.angel_features`/`gate.devil_features` and re-derives them from
`frame.feature_cols`. Correct only by construction today — the gate trains on
whatever it's handed, and HMM features (if ever enabled) are appended at
`_gate.py:606` — so the identity breaks silently under the first spec that diverges.

**Do:** thread `gate.angel_features` and `gate.devil_features` from `GateResult`
through `run_model_backtest` into `LabModelStrategy` (replacing the frame-derived
ones). `run_artifact_backtest` already takes explicit lists; keep the two paths
consistent through the shared `_replay_models`.

**Tests:** construct a `GateResult` whose feature lists contain a column absent from
`frame.feature_cols` (a synthetic HMM stand-in) and assert the backtest consumes the
gate's list, i.e. no silent fallback to frame columns.

## W3 — `lab ablate`: feature-interaction ablation (the polypharmacy question)

**Why:** single-feature-at-a-time answers the wrong question. The honest measure of
family X *in situ* is `edge(full cocktail) − edge(cocktail − X)`. A lone-feature run
is a diagnostic, not evidence. This is the piece the user explicitly asked for.

**Do:**
- New `src/lab/ablate.py` + CLI verb:
  `PYTHONPATH=src:. python -m lab.cli ablate --name <spec>`.
- Expand the spec into N+1 `FeatureSpec`s: the base spec unchanged, plus one per
  family in `spec.feature_sets` with that family removed. **Labels, veto, geometry,
  data identical across variants; only `feature_sets` differs.** Frame caching handles
  the reuse (correctly, once W1 lands).
- For each variant emit: pooled trades, pooled wins, `edge_over_random`, pooled and
  fold-3 PF lower bounds, and **the delta** against the full cocktail — *with a
  confidence interval on the delta* (Clopper-Pearson on the underlying win
  difference, mirroring the gate's conservatism). The report must not present a point
  delta as a finding without its interval.
- Report renderer: an ablation table (one row per variant) + a Caveats section that
  fires whenever pooled trades are thin (the v3_base regime is ~30-40 trades), stating
  that a delta consistent with zero is not a drop decision.
- If `feature_sets` has exactly one family, `ablate` refuses with a clear message
  (ablation needs ≥2 to have a cocktail to subtract against) and suggests `run`.

**Tests:** expansion correctness (N+1 specs, right families omitted, geometry/labels
unchanged); delta computation on synthetic numbers; thin-trade caveat fires;
single-family refusal.

## W4 — Estimator A/B via the existing MODEL_FAMILY seam (gate, not build)

**Why:** the estimator is already isolated behind `make_classifier` /
`MODEL_FAMILY` (`_common.py:410-416`), so "same frame under LightGBM vs CatBoost" is
two runs, zero new lab code. This answers "is the incumbent estimator the bottleneck"
*before* anyone builds a bespoke architecture. The 2026-09-14 H4/CatBoost A/B was
confounded (window shift + metals); this is the controlled re-measurement.

**Do (after W1–W3 are green):**
- Run the identical seed spec(s) under both estimators. CatBoost is selected by
  `MODEL_FAMILY=catboost` in the environment (read at retrainer import time — the
  CLI sets it before import); the default stays lightgbm.
  ```
  PYTHONPATH=src:.                        python -m lab.cli run --name v3_base_control
  MODEL_FAMILY=catboost PYTHONPATH=src:. python -m lab.cli run --name v3_base_control
  ```
  Same content hash → same cached frame; only the fitted estimator changes.
- File `llm_reports/recons/<date>_lab-model-family-ab.md`: gate verdict, pooled
  trades, `edge_over_random`, both PF bounds for each family, side by side, with the
  explicit statement of whether the delta is distinguishable from sampling noise on
  this trade count.
- **Decision rule for the record:** if CatBoost does not produce a promotable
  `edge_over_random` on the same frame + geometry + cost accounting, then a bespoke
  architecture harness is *not* the next lever — the answer is features (or targets),
  consistent with the 2026-09-14 edge-budget conclusion. Write that verdict down.

## What "done" looks like

- `pytest -q` green, `compileall -q src/` clean, quoted output in the build report.
- Four W's complete, in this order, each with its tests.
- The soak untouched: `systemctl --user is-active soak.service` still `active`.
- Nothing committed; branch named in the report; a clean diff Brandon can review.
- Any new module gets a `Glossary:` docstring section; new domain terms go in
  `GLOSSARY.md`; `src/lab/README.md` updated for the new verb, per `CLAUDE.md`'s
  three-layer documentation rule (this is part of the task, not a follow-up).

## Files touched

- Created: `llm_reports/handoffs/2026-09-22_feature-lab-v2-build-tasks.md` (this file).
- Read: the lab source (`spec.py:160-204`, `backtest.py:240-253`,
  `_common.py:355-419`), the prior handoff and the build refactor report.
