---
type: <audit | refactor | recon | handoff | stop>
date: <YYYY-MM-DD>
time: <HH:MM TZ>
agent: <human-readable name, e.g. "Claude Opus 4.7">
model: <machine-parseable model id, e.g. "claude-opus-4-7">
trigger: <one-line summary of what prompted this report>
head: <full HEAD commit SHA at start of work>
scope: <read-only | modifies-source | modifies-config | modifies-data>
# Optional fields below — delete the line if not used.
related:
  - <category>/<prior-report.md>
files_touched:           # REQUIRED if scope != read-only
  - <path/to/file.py>
---

# <Title — short, descriptive, no all-caps>

## Context

_Why this work happened. The user's ask, the prior incident, the upstream
report that triggered it. State the question being answered up front._

## Investigation

_The "show your work" trace. What was read, grepped, hypothesized. Quote
file:line citations and small code excerpts so a future agent can
reconstruct your reasoning without re-doing the search._

## Findings / Changes

_For audits + recons: numbered findings with severity ratings.
For refactors: per-file before/after summaries.
For handoffs + stops: the substance — what's true now._

## Verification

_How correctness was confirmed. Commands run, tests passed, smoke tests,
manual checks. For audits: how citations were spot-checked. For refactors:
how the change was validated end-to-end._

## Risk & follow-ups

_Open items, regressions to watch, related work. Items that should become
their own report later._

## Files touched

_Paths + line ranges. Mandatory for refactors. For audits/recons, list
files **read** so a follow-up agent knows what's already been examined._
