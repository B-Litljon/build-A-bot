# CLAUDE.md

Instructions for AI assistants working in this repo.

## More than one agent may be in this checkout

Several agents (and several humans) work in this tree, sometimes on the same
feature at the same time. Two conventions exist for that, and using them is not
optional when someone else is active:

- **Leave a channel, not a private report.** A one-way brief or an audit nobody
  else reads cannot coordinate two writers. Open or append to a **live thread**
  in [`llm_reports/m2m-prompts/`](llm_reports/m2m-prompts/) — append-only, one
  `## [timestamp] <from> → <to>` block per message, every claim marked VERIFIED
  (with the command and its real output), ASK, OFFER, BLOCKER or DECIDED. Read
  that folder's README for the index of open threads. **Re-read the thread
  immediately before appending**, and say so if you edit a file the other agent
  may be mid-edit in — a stale read in a shared tree silently clobbers work.
- **Check the tree before you believe a note, including a note you wrote.** Read
  `git log`/`git status` and the file itself; commits here are often made by
  whoever finishes last, with a message that describes only their own half.

## Documentation maintenance (required)

This repo carries a three-layer documentation system. **Keeping it accurate is
part of making a change, not a follow-up task.**

> **Any change that adds, renames, or removes a documented identifier must
> update that file's `Glossary:` section in the same change. New files get a
> Glossary header; new folders get a README entry; new recurring domain terms go
> in `GLOSSARY.md`.**

The three layers:

| Layer | Where | Contains |
|---|---|---|
| 1 | [`GLOSSARY.md`](GLOSSARY.md) | Repository map + the project's shared vocabulary |
| 2 | `README.md` in each code folder | Per file: what it is, what it imports, what imports it, what it reads/writes |
| 3 | `Glossary:` in each module docstring | Meaningful identifiers inside that file |

Conventions for Layer 3:

- Format: `name -- one plain-language line: what it is, units/range if numeric,
  why it exists.`
- **Meaningful identifiers only**: module constants, class attributes, config
  fields, load-bearing locals. Skip loop counters, obvious temporaries, and
  standard-library idioms.
- For domain terms, write "see GLOSSARY.md" rather than redefining them.
- Roughly 10–30 lines; long files may warrant more, but never pad.
- **Accuracy over completeness.** Derive meaning from reading the code and its
  call sites, never from the identifier's name — several names in this repo are
  actively misleading (see below). If a purpose is genuinely undeterminable,
  write `[UNCLEAR — verify]` and move on.

## Operational safety

- **A live bot may be running.** The M15 forex soak runs from the working tree
  (`run_oanda.py --daemon --env practice --granularity 15`). Check with
  `ps aux | grep run_oanda` before assuming otherwise.
- **`soak_watchdog.sh` is in cron every 5 minutes** and will relaunch the soak
  if it dies — from whatever is in the working tree, on whatever branch is
  checked out. To stop the soak, `touch soak.off` **before** stopping it, or it
  comes back within 5 minutes.
- **The soak runs as the `soak.service` systemd user unit** (since 2026-08-22).
  Stop it with `systemctl --user stop soak.service`; how a run ended is in
  `systemctl --user status soak.service` and in the watchdog's post-mortem line
  in `logs/watchdog.log`. The served model dir is declared in `soak.service` —
  that is the single source of truth, and it must match the tree's brackets.
- **Stops and targets are enforced in software**, by the bot process itself. A
  dead bot means an unwatched open position. Treat anything that could crash or
  hang `src/execution/` as a money-losing bug, not a cosmetic one.
- Model artifacts are written atomically (temp file + rename) because the live
  strategy hot-reloads them. Preserve that pattern.

## Things that will mislead you

Verified during the 2026-07-27 glossary pass:

- **One `Signal` class now.** The Alpaca path's `core.signal.Signal` was
  deleted with that lane on 2026-09-16. `strategies.base.Signal` (explicit
  distance fields) is the only one.
- **`V3RandomForestTrainer` holds a LightGBM model.** `.load()` unpickles
  whatever is on disk; production models have been LightGBM since 2026-05-23.
  Candidates are not so uniform — the CatBoost A/B runs leave
  `CatBoostClassifier` pickles — which is why `feature_names_in_` reads BOTH
  spellings (`feature_names_in_`, CatBoost's `feature_names_`). A CatBoost
  artifact used to fail the strategy's boot with "exposes no feature_names_in_"
  (2026-09-14).
- **`Signal.raw_tp_distance` is written but never read.** Target sizing belongs
  to `RiskManager`'s multipliers, deliberately.
- **Bracket multipliers differ by asset class.** The module constants say
  0.5×/3.0×, but `RiskProfile.for_asset_class("forex")` overrides to 2.0×/4.0×
  (was 1.0×/2.0× until 2026-08-08). `spread_k_base` is 3.0 for forex, not the
  module default 1.5 — it is a toll cap, admitting a trade only when the spread
  eats at most `1/k` of the stop distance.
- **"regime" means two things** (volatility band vs HMM hidden state), and so
  does **"watchdog"** (in-process stop monitor vs the cron restarter) and
  **"heartbeat"** (OANDA keepalive vs the strategy's periodic log).
- **Don't use textbook PSI thresholds.** Market bars are autocorrelated; use the
  null calibration in `feature_stats.json`.
- **`fetch_training_data`** is a function living in `core/retrainer/_data.py`
  (the retrainer became a package on 2026-09-16; `core/retrainer` re-exports
  it). The same-named dead module in `src/data/` was deleted 2026-09-16.

## Testing

```bash
# Use the project venv python — system python 3.14 lacks the deps
# (alpaca, mcp, …) and fails test COLLECTION with 23 import errors.
PYTHONPATH=src:. \
  /home/tha_magick_man/.local/share/virtualenvs/build-A-bot-A3hTUWzK/bin/python \
  -m pytest -q     # 505 passed / 6 subtests passed (2026-09-14; the suite has
                   # grown a lot — treat the count as a sanity check, not a target)
```

`PYTHONPATH=src:.` is required — entry points prepend `src/` to the path, which
is why modules import as `data.factory` rather than `src.data.factory`.

Before committing changes to `src/`, also run
`python -m compileall -q src/`.
