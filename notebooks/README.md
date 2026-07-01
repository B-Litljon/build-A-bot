# Feature Research Lab

A scientific way to test whether a feature idea has **real predictive signal** —
*before* spending days building a model and praying it passes a gate.

Core idea: a feature is never judged against price alone, but against a **forward-looking
target**. We measure the **Information Coefficient** (does the feature's ranking line up
with the future move, *consistently over time*) and the **decile staircase** (sort by the
feature — does the top bucket actually move more?). What separates this from fooling
ourselves: out-of-sample testing, year-by-year stability, cost-awareness, redundancy
checks, and a built-in `noise_random` control that **must** score ~0.

## Two pieces
- **`src/research/feature_lab.py`** — the engine. Fetches data, runs the *exact*
  production feature pipeline (zero train/inference skew), builds the targets, and scores
  every feature. Runs as a script.
- **`notebooks/feature_lab.ipynb`** — the visual layer (IC ranking, decile staircases,
  stability, redundancy heatmap) and the **"define your candidate feature"** cell.

## Workflow
1. **Generate the panel** (once per data refresh), from the repo root with creds loaded:
   ```bash
   set -a; . ./.env; set +a
   PYTHONPATH=src:. <venv-python> -m research.feature_lab --profile swing --export
   ```
   This prints a ranked table and writes two portable files to `data/research/`:
   `feature_panel_swing.parquet` (all features + targets) and `feature_report_swing.csv`.

2. **Explore in the notebook.** Open `notebooks/feature_lab.ipynb`:
   - **Local kernel** (IDE Colab extension on the venv) — can also re-run step 1 inline.
   - **Cloud Colab / share with Gemini** — upload `feature_panel_swing.parquet` and run;
     it's pure pandas, no repo or TA-Lib needed.

3. **Test your idea.** In the last cell, write a `df -> df` function that adds your
   feature, then read its IC (full + out-of-sample), decile staircase, and whether it
   beats the `noise_random` control. Keep it or kill it. Repeat.

## Reading the numbers
| column | meaning | what's good |
|--------|---------|-------------|
| `ic_fwd` | rank-corr with the future move (all data) | bigger \|value\| |
| `ic_oos` | same, on the held-back recent year | same sign as `ic_fwd`, not ~0 |
| `ic_ir_is` | IC mean ÷ IC std (consistency) | bigger \|value\| = steadier |
| `decile_bps` | top-minus-bottom decile return, basis points | bigger than your cost |
| `beats_cost` | does the decile spread clear `--cost-bps`? | `true` |
| `max_abs_corr` | correlation with its most-similar feature | low = adds something new |
| `ic_angel` / `ic_devil` | rank-corr with the model's actual trade labels | same story as `ic_fwd` |

**The discipline:** a feature only earns a spot if it clearly beats `noise_random`,
keeps its sign out-of-sample, and its decile spread beats trading cost. Statistically
"consistent" but sub-cost is **not** tradeable — the table will show you both.

## Profiles
`--profile swing` (H1, ~6h ahead, cross-pairs + metals — the default) or `--profile
scalper` (M1, ~3 bars, metals). Add more in `PROFILES` in `feature_lab.py`. Register new
candidate features with the `@candidate("name")` decorator in the same file to have them
scored on every run.
