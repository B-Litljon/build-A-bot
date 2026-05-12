# llm_reports/ — Convention

Long-form notes written by AI agents (Claude, Gemini, Kimi K2.6, GPT, etc.) as
they audit, refactor, or investigate this codebase. The folder serves two
purposes:

1. **An audit trail** for the user (Brandon) so non-trivial agent work can be
   reviewed without re-reading commits.
2. **Onboarding context** for the *next* agent — handoff letters, recon notes,
   and refactor explanations let a fresh session pick up where the prior one
   stopped.

## What goes where

| Folder | Scope | Body characteristics |
|---|---|---|
| **`audits/`** | Read-only analysis. Findings, severity ratings, no edits to source. | Long-form; numbered findings; severity tiers. |
| **`refactors/`** | Code changes that shipped. Documents the **what + why** of a commit (or sequence). | Frontmatter must list `files_touched`; body shows before/after for non-obvious changes. |
| **`recons/`** | Targeted investigation to inform a decision. Not yet a refactor; not a broad audit. | Body leads with the question being answered; concludes with a recommended next step. |
| **`handoffs/`** | Session-to-session or agent-to-agent continuation notes. | Body has a clear "what to do next" section the incoming agent can follow. |
| **`stops/`** | Work blocked by an unexpected dependency / risk / question. | Short. Body has "blocker" + "what's needed to unblock." |

Pick the folder by the **default mode** of the work, not by the topic:

- Wrote 500 lines of code and shipped a commit → `refactors/`
- Wrote no code; produced a numbered findings list → `audits/`
- Wrote no code; answered a specific question to unblock a decision → `recons/`
- Stopping mid-task because of an upstream issue → `stops/`
- Briefing the next agent on what's queued up → `handoffs/`

## Naming

```
<YYYY-MM-DD>_<kebab-case-topic>.md
```

- **Date** is ISO 8601, always at the start. Sorts chronologically inside any
  folder.
- **Topic** is kebab-case (lowercase, hyphens, no spaces).
- **Category** is implied by the folder — never repeated in the filename.
- No version suffixes (`_v2`, `_RERUN`, `_FINAL`). If a follow-up report
  rehashes the same topic, the new date differentiates it; reference the prior
  via `related:` in the frontmatter.

Examples:
- `audits/2026-05-10_polars-and-logic.md`
- `refactors/2026-05-10_oanda-integration.md`
- `recons/2026-04-30_bracket-and-sizing.md`
- `handoffs/2026-05-02_act3.md`
- `stops/2026-04-27_unexpected-v1-deps.md`

## Frontmatter

Every new report opens with a YAML block:

```yaml
---
type: audit | refactor | recon | handoff | stop
date: 2026-05-10                                   # ISO date of authoring
time: 20:37 PDT                                    # local time + tz
agent: Claude Opus 4.7                             # human-readable name
model: claude-opus-4-7                             # machine-parseable id
trigger: <one-line — what prompted this report>
head: 7addd18c85f38c299eaa194ea2c74862c0b006cf     # full SHA preferred
scope: read-only | modifies-source | modifies-config | modifies-data
related:                                           # optional
  - audits/2026-05-05_v3.4-roadmap.md
files_touched:                                     # required if scope != read-only
  - src/data/oanda_provider.py
---
```

### Required vs optional

- **Always required:** `type`, `date`, `time`, `agent`, `model`, `trigger`,
  `head`, `scope`.
- **Conditionally required:** `files_touched` when `scope != read-only`.
- **Optional:** `related`.

### Special markers (legacy / imported reports only)

- `imported_from: <original-filename>` — present on reports that were
  backfilled from the pre-convention era (2026-04 to 2026-05-10). Their
  bodies retain their original heading style and some fields may be
  `unknown`. Trust the `imported_from` marker as the signal that this is
  partial metadata, not full schema compliance.

## Body — "show your work"

Every new report body has these six sections, in this order:

```markdown
## Context
Why this work happened. The user's ask, the prior incident, the upstream
report that triggered it. State the question being answered up front.

## Investigation
The "show your work" trace. What was read, grepped, hypothesized. Quote
file:line citations and small code excerpts. This is the section that lets
a future agent reconstruct your reasoning without re-doing the search.

## Findings / Changes
For audits + recons: numbered findings with severity.
For refactors: per-file before/after summaries.
For handoffs + stops: the substance — what's true now.

## Verification
How correctness was confirmed. Commands run, tests passed, smoke tests,
manual checks. For audits: how citations were spot-checked. For refactors:
how the change was validated end-to-end.

## Risk & follow-ups
Open items, regressions to watch, related work. Items that should become
their own report later.

## Files touched
Paths + line ranges. Mandatory for refactors. For audits/recons, list files
*read* (so a follow-up agent knows what's already been examined).
```

Sections that don't apply for a given report type stay as a heading with
one line of `_n/a — read-only audit_` (or similar) so the shape stays
constant across reports.

The six sections satisfy the four W's:

- **WHY** → Context
- **WHERE** → Files touched + citations in Investigation
- **WHAT** → Findings / Changes
- **HOW** → Investigation + Verification

## Starting a new report

```sh
# pick the right category folder
cp llm_reports/_TEMPLATE.md llm_reports/audits/$(date -u +%Y-%m-%d)_my-topic.md
```

Then fill in the frontmatter, write the six sections, delete the inline
instructions as you go.
