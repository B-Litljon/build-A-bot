---
type: refactor
date: 2026-05-11
time: 00:15 PDT
agent: Claude Opus 4.7
model: claude-opus-4-7
trigger: User asked for a reorganized llm_reports/ structure that is readable at a glance and uniformly formatted, following a deep audit of the codebase
head: 7addd18c85f38c299eaa194ea2c74862c0b006cf
scope: modifies-config
files_touched:
  - .gitignore
  - llm_reports/README.md
  - llm_reports/_TEMPLATE.md
  - llm_reports/audits/2026-04-25_alpaca-surface-area.md
  - llm_reports/audits/2026-05-05_v3.4-roadmap.md
  - llm_reports/audits/2026-05-10_polars-and-logic.md
  - llm_reports/handoffs/2026-05-02_act3.md
  - llm_reports/handoffs/2026-05-03_maiden-voyage.md
  - llm_reports/recons/2026-04-25_sdk-decoupling-branch.md
  - llm_reports/recons/2026-04-25_sdk.md
  - llm_reports/recons/2026-04-29_grid-search.md
  - llm_reports/recons/2026-04-29_mlstrategy-interpretation.md
  - llm_reports/recons/2026-04-30_bracket-and-sizing.md
  - llm_reports/recons/2026-04-30_smoke-test-gate.md
  - llm_reports/recons/2026-04-30_smoke-test-gate-rerun.md
  - llm_reports/refactors/2026-04-26_tier-1-foundations.md
  - llm_reports/refactors/2026-05-02_path-alpha.md
  - llm_reports/refactors/2026-05-03_tier-2-decoupling.md
  - llm_reports/refactors/2026-05-03_v4-data-pipeline.md
  - llm_reports/refactors/2026-05-10_oanda-integration.md
  - llm_reports/stops/2026-04-27_unexpected-v1-deps.md
  - memory/feedback_llm_reports.md
  - memory/MEMORY.md
---

# llm_reports/ Reorganization

## Context

Prior to this session all 18 reports in `llm_reports/` lived in a flat directory with inconsistent file names (all-caps, mixed separators, no date prefix on several) and no uniform body structure — each agent had used whatever heading style felt natural at the time. The user asked for a format that was "readable at a glance" and that required agents to show their reasoning (why/where/what/how) so that both the user and a future agent could reconstruct decisions without re-reading commits.

The session that immediately preceded this reorganization produced a 25-finding critical audit (`audits/2026-05-10_polars-and-logic.md`), which itself was the first report to use the new format — written just before the overhaul was formalized.

## Investigation

**Problem diagnosis:**

- `ls llm_reports/` returned files named `ACT3_HANDOFF_LETTER_2026-05-02.md`, `ALPACA_SURFACE_AREA_AUDIT.md`, `V4_DATA_PIPELINE_REPORT_2026-05-03.md`, etc. — a mix of all-caps, underscores, partial dates, and implicit types baked into the name. No two files followed the same convention.
- Bodies ranged from bullet-list summaries to prose narratives to structured numbered findings; a future agent couldn't programmatically tell what kind of work each report documented.
- `git ls-files llm_reports/` revealed only 6 of the 18 files were tracked; the directory was listed in `.gitignore`, meaning 12 reports existed only on disk.

**Design decisions:**

1. **Five category folders** (`audits/`, `refactors/`, `recons/`, `handoffs/`, `stops/`) chosen to match the natural modes of agent work. Category is determined by what was *done*, not by the topic. This means `SDK_DECOUPLING_BRANCH_REPORT` went to `recons/` (no code written) even though it sounds like a refactor.
2. **Date-first kebab-case names** (`YYYY-MM-DD_topic.md`) so `ls` inside any category folder sorts chronologically without additional tooling.
3. **YAML frontmatter** as the machine-parseable header. Fields: `type`, `date`, `time`, `agent`, `model`, `trigger`, `head`, `scope`, plus conditional `files_touched` and optional `related`. An `imported_from` field marks legacy reports so consumers know not to expect schema compliance.
4. **Six-section body** (Context → Investigation → Findings/Changes → Verification → Risk & follow-ups → Files touched) satisfies the four W's the user asked for: WHY (Context), WHERE (Files touched + citations), WHAT (Findings), HOW (Investigation + Verification). Sections that don't apply get a single `_n/a_` line so the shape is constant.
5. **Legacy body preservation**: Bodies of all 18 existing reports were left exactly as written — only YAML frontmatter was prepended. This avoids re-summarizing content that was correct at the time, and the `imported_from` marker tells future agents the body predates the convention.

**Classification of each existing report:**

| Original filename | Category | Rationale |
|---|---|---|
| ACT3_HANDOFF_LETTER | handoffs/ | Explicitly a continuation letter to next agent |
| MAIDEN_VOYAGE_REPORT | handoffs/ | Post-session summary / handoff |
| ALPACA_SURFACE_AREA_AUDIT | audits/ | Read-only analysis, numbered findings |
| V3.4_ROADMAP_AUDIT | audits/ | Read-only, roadmap review |
| SDK_DECOUPLING_BRANCH_REPORT | recons/ | Explicitly states "read-only, no checkout performed" |
| SDK_NOTES | recons/ | Investigative notes, answered a specific question |
| GRID_SEARCH_NOTES | recons/ | Investigation to inform hyperparameter decision |
| MLSTRATEGY_INTERP | recons/ | Interpretation/investigation, no code shipped |
| BRACKET_AND_SIZING | recons/ | Investigation into risk sizing logic |
| SMOKE_TEST_GATE | recons/ | Gate investigation, no code |
| SMOKE_TEST_GATE_RERUN | recons/ | Follow-up investigation of same question |
| TIER_1_FOUNDATIONS_REPORT | refactors/ | Code shipped (Tier 1 foundation work) |
| PATH_ALPHA_REFACTOR | refactors/ | Code shipped |
| TIER_2_DECOUPLING_REPORT | refactors/ | Code shipped (Tier 2 decoupling) |
| V4_DATA_PIPELINE_REPORT | refactors/ | Code shipped |
| OANDA_INTEGRATION | refactors/ | Code shipped (most recent prior session) |
| STOP_REPORT | stops/ | Work blocked, needs user input |

## Findings / Changes

**New files created:**

- `llm_reports/README.md` — full convention documentation: category table, naming rules, YAML frontmatter schema with required vs optional fields, six-section body template, `imported_from` marker explanation, example `cp` command for starting a new report
- `llm_reports/_TEMPLATE.md` — drop-in starter (YAML scaffold + six section headings with inline authoring instructions)

**Existing reports restructured:** All 18 reports moved to category subfolder with date-first kebab-case name. YAML frontmatter prepended. Bodies untouched.

**git mechanics:** 6 previously-tracked files moved via `git mv` (staged as renames in git history); 12 untracked files moved via plain `mv`. The `.gitignore` line `llm_reports/` was removed so all reports are now tracked.

**Memory updated:**
- `memory/feedback_llm_reports.md` — rewritten with new convention; source of truth delegated to `llm_reports/README.md`
- `memory/MEMORY.md` — index entry updated from "write detailed recon reports" to "file under `llm_reports/<category>/YYYY-MM-DD_topic.md` with YAML frontmatter + six-section body; canonical spec in `llm_reports/README.md`"

**CLAUDE.md recommendation (deferred):** To make other LLMs check `llm_reports/` automatically, a "For AI Agents" section should be added to `CLAUDE.md` instructing agents to read `llm_reports/README.md` and browse recent reports at session start. The M2M handoff template should include a step 0 for the same. This was not implemented in this session — user confirmed the intent but it remains a follow-up.

## Verification

```
find llm_reports -type f | sort
```

Output confirmed all 21 files (18 reports + README + _TEMPLATE + this report) in their correct category subfolders with date-first names.

Spot-checked three frontmatter blocks:
- `audits/2026-05-10_polars-and-logic.md` — full schema-compliant YAML with `files_touched` list
- `stops/2026-04-27_unexpected-v1-deps.md` — `imported_from: STOP_REPORT_2026-04-27.md`, body untouched
- `refactors/2026-05-10_oanda-integration.md` — `imported_from` marker, body intact

`.gitignore` verified — `llm_reports/` line removed, all other rules preserved.

`git status` after staging confirms 6 renames + their body modifications, plus all 15 previously-untracked files staged for first commit.

## Risk & follow-ups

1. **CLAUDE.md "For AI Agents" section** — not yet written. Until it is, non-Claude agents won't know to check `llm_reports/` unless explicitly told in the M2M handoff. Add at next opportunity.
2. **M2M handoff template step 0** — user's architect AI template should include "Read `llm_reports/README.md` and recent reports before forming any plan." Currently relies on user memory to include this instruction.
3. **`imported_from` reports are body-noncompliant** — the 18 legacy bodies don't have the six-section structure. This is intentional (bodies preserved). A future agent doing a deep audit might want to backfill the structure for the most-referenced reports (ACT3, MAIDEN_VOYAGE, TIER_1/2 foundations) so they're machine-parseable.
4. **`package-lock.json` untracked** — appeared in `git status` during this session. Not included in this commit. User should decide whether it belongs in the repo.

## Files touched

All files listed in frontmatter `files_touched`. Key line ranges:

- `.gitignore:37` — removed `llm_reports/` line
- `llm_reports/README.md` — new file, 140 lines
- `llm_reports/_TEMPLATE.md` — new file, 51 lines
- All 18 reports: frontmatter prepended (lines 1–15 of each file are new YAML)
- `memory/feedback_llm_reports.md` — full rewrite (~25 lines)
- `memory/MEMORY.md:4` — index entry updated (1 line)
