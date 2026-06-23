# M2M Prompts — a ledger of model-to-model handoffs

This folder holds **instructions we hand to other LLMs** (coder models like Gemini, intern models like Haiku) to implement a task. It is the *input* counterpart to `llm_reports/`:

- `llm_reports/` = **outputs** — findings, post-mortems, what happened.
- `m2m_prompts/` = **inputs** — the exact instructions we dispatched, kept so we can diff *what we asked for* against *what actually got committed*.

Keeping them in separate drawers keeps that input/output distinction clean.

## Filename convention

```
m2m_prompts/YYYY-MM-DD_topic.md
```

e.g. `2026-06-22_investor-gate.md`. Dated names sort themselves — **no index file** (volume is low).

## Frontmatter (every entry)

```yaml
---
to: gemini-3.5-flash        # assignee model
from: claude-opus-4-8       # author
date: 2026-06-22
status: drafted             # drafted → dispatched → completed → verified
branch: feature/investor-gate
topic: short description of the task
result_commit:              # SHA, filled in AFTER the work lands — closes the audit loop
related_memory:             # optional: memory slug this traces back to
related_report:             # optional: llm_reports/ link
---
```

## Status lifecycle

| status | meaning |
|---|---|
| `drafted` | written, not yet sent |
| `dispatched` | handed to the assignee model |
| `completed` | work landed; `result_commit` filled in |
| `verified` | we re-ran / checked the result ourselves (see the verify-handoffs-by-rerun rule) |

The point of `result_commit`: the prompt points to the commit, the commit points back to the prompt. That two-way link is the ledger — you can always reconstruct *why* a change was made and *whether it matched the instruction*.
