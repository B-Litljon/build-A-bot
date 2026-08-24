# `llm_reports/m2m`

Model-to-model handoffs written **as reports** — a finding and the work it
implies, in one document, handed to an outside model.

Distinct from the two neighbouring drawers:

| folder | holds |
|---|---|
| `m2m_prompts/` | pure *instructions* dispatched to another model (inputs) |
| `llm_reports/<category>/` | our own findings and post-mortems (outputs) |
| `llm_reports/m2m/` | a finding **and** its work order, for an outside model |

Use this when the evidence is the brief — when the assignee needs to understand
*why* before they can sensibly do *what*.

## Conventions

Same filename and frontmatter as `m2m_prompts/` (see that README), plus `head:`
recording the commit the brief was verified against.

**Every entry must open with a FRESHNESS CONTRACT.** On 2026-08-11 a brief sent
into a notebook holding months-old context came back as a stale work order that
would have reverted a live fix — every detail correct as of June, which is what
made it convincing. The contract asserts current state up front, names what is
already built, and requires the assignee to flag contradictions instead of
acting on them. See commit b76f375.
