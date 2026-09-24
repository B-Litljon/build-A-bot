"""Report emitter -- a lab run into llm_reports/recons/<date>_<slug>.md.

Fills the numbers that were measured and leaves an empty ``## Interpretation``
for whoever ran the experiment. The report follows llm_reports/README.md's
frontmatter convention; the folder is ``recons`` because a lab run answers a
specific question ("does this feature set move the verdict?").

A second renderer covers the served-artifact replay (``lab.artifact``): no gate
verdict, but the raw pre-gate population, the artifact's recorded-holdout
window split, and the live-gated replay.

Glossary:
    render_report -- ExperimentResult + command -> markdown text.
    write_report -- render, then write to llm_reports/recons/ with an atomic
        temp+rename; returns the path.
    _artifact_caveats -- the honesty box for a replay: no gate verdict, flat
        toll (the served schema has no cost_ratio), thin populations, and the
        in-sample nature of the frame.
    render_artifact_report -- ArtifactReplayResult + command -> markdown text.
    write_artifact_report -- the replay's writer; ``slug`` defaults to
        ``served-artifact-<spec>`` so it can never overwrite a gate report.
    render_ablation_report -- AblationResult + command -> markdown: the
        ablation table (one row per variant: pooled trades, wins,
        edge_over_random, both PF bounds, and the delta WITH its
        Clopper-Pearson interval) plus a Caveats section that fires on thin
        pooled trades — a delta consistent with zero is never a drop decision
        by itself.
    write_ablation_report -- the ablation's writer; slug ``ablate-<spec>`` so
        it can never overwrite a gate or replay report.
    _git_head -- short commit hash for the frontmatter, or "unknown" outside a
        git checkout (never raises: a report is not worth losing to a git call).
    _fmt -- float formatting helper with an explicit nan fallback, because the
        gate reports nan for metrics that could not be computed.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_REPORT_DIR = Path("llm_reports/recons")


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _fmt(value, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "nan"
    if f != f:  # nan
        return "nan"
    return f"{f:.{digits}f}"


def _caveats(result) -> list:
    """The honesty box: conditions under which the numbers above mislead."""
    out = []
    report = result.gate.report
    if not report.gate_passed:
        out.append(
            "Gate FAILED — the replayed models are the Fold-3 placeholders, not a "
            "promoted artifact. Backtest numbers are indicative only."
        )
    if report.pooled_oos_trades < 100:
        out.append(
            f"Only {report.pooled_oos_trades} pooled OOS trades. At this count the "
            "win rate (and therefore edge-over-random) is dominated by sampling "
            "noise — a positive number here is not evidence of skill."
        )
    if not result.spec.use_spread_table:
        out.append(
            "Cost table OFF — the backtest priced trades at the flat default toll, "
            "not per-instrument measured alphas. Cross-instrument comparisons in "
            "the backtest table are especially weak."
        )
    return out


def render_report(result, *, command: str = "") -> str:
    """Markdown for one experiment result."""
    s = result.summary()
    spec = result.spec
    now = datetime.now().astimezone()
    lines = [
        "---",
        "type: recon",
        f"date: {now.date().isoformat()}",
        f"time: {now.strftime('%H:%M %Z')}",
        "agent: opencode",
        "model: deepseek-flash",
        'trigger: "Feature-lab run {name} (spec hash {hash})"'.format(
            name=spec.name, hash=s["content_hash"]
        ),
        f"head: {_git_head()}",
        "scope: lab run only — no production model, config, or live path touched",
        "related:",
        "  - handoffs/2026-09-21_feature-lab-plan.md",
        "---",
        "",
        f"# Feature lab — {spec.name}",
        "",
        "## Spec",
        "",
        f"- content hash: `{s['content_hash']}` (frame cache key)",
        f"- feature families: `{', '.join(spec.feature_sets)}`"
        + (" + extra generators" if spec.extra_generators else ""),
        f"- symbols: {', '.join(spec.symbols)}",
        f"- window: {spec.days_back} days @ M{spec.granularity}, htf {spec.htf_timeframe}",
        f"- geometry: {spec.geometry.sl_mult}x/{spec.geometry.tp_mult}x/{spec.geometry.max_hold} bars",
        f"- labels: kind={spec.label.kind}, survival_bars={spec.label.survival_bars}",
        f"- spread table: {'on' if spec.use_spread_table else 'off'}"
        + (f" ({spec.spread_table_path})" if spec.use_spread_table else ""),
        f"- estimator: {s['model_family']}",
        "",
        "## Frame",
        "",
        f"- rows: {s['rows']:,} | features: {s['feature_count']} | "
        f"chop/behavior veto drop: {s['chop_veto_rate']:.2%} | "
        f"unresolvable tail purged: {s['purged_tail_rows']:,}",
        f"- {'loaded from frame cache' if s['frame_from_cache'] else 'built this run'}",
        "",
        "## Gate (retrainer `validate_candidate`)",
        "",
        f"- verdict: **{'PASS' if s['gate_passed'] else 'FAIL'}**",
        f"- mean Brier {_fmt(s['mean_brier'])} | mean EV {_fmt(s['mean_ev'], 6)} | "
        f"pooled trades {s['pooled_oos_trades']} | pooled wins {s['pooled_oos_wins']}",
        f"- pooled PF lower bound {_fmt(s['pooled_pf_lower_bound'])} | "
        f"fold-3 PF lower bound {_fmt(s['fold3_pf_lower_bound'])}",
        f"- pooled base rate {_fmt(s['pooled_base_rate'])} | "
        f"**edge over random {_fmt(s['edge_over_random'])}**",
        f"- production thresholds: Angel {_fmt(s['production_angel_threshold'])}, "
        f"Devil {_fmt(s['production_devil_threshold'])}",
        f"- gate wall time: {s['run_seconds']:.1f}s",
        "",
        "| fold | train | val | Brier | EV | proposed | approved | win rate | base rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fm in s["folds"]:
        lines.append(
            f"| {fm['fold']} | {fm['train_size']:,} | {fm['val_size']:,} | "
            f"{_fmt(fm['brier'])} | {_fmt(fm['ev'], 6)} | {fm['angel_proposed']} | "
            f"{fm['devil_approved']} | {fm['win_rate']:.2%} | {_fmt(fm['base_rate'])} |"
        )
    if s["rejection_reasons"]:
        lines += ["", "Rejection reasons:", ""]
        lines += [f"- {r}" for r in s["rejection_reasons"]]

    lines += ["", "## Backtest (live-gated replay)", ""]
    bt = s.get("backtest")
    if bt is None:
        lines.append("Not run for this experiment (`--no-backtest`).")
    else:
        lines += [
            f"- toll pricing: {bt['toll_mode']} | trades {bt['total_trades']} | "
            f"wins {bt['wins']} | win rate {bt['win_rate']:.2%}",
            f"- gross EV {_fmt(bt['gross_ev_r'])}R | net EV {_fmt(bt['net_ev_r'])}R | "
            f"net PF {_fmt(bt['profit_factor_net'])} | max drawdown {_fmt(bt['max_drawdown_r'])}R",
            f"- gate veto funnel: "
            + (", ".join(f"{k}={v}" for k, v in sorted(bt["gate_rejections"].items())) or "none"),
            "",
            "| symbol | trades | win rate | net EV (R) | net PF |",
            "|---|---:|---:|---:|---:|",
        ]
        for sym in sorted(result.backtest.per_symbol):
            rep = result.backtest.per_symbol[sym]
            lines.append(
                f"| {sym} | {rep.total_trades} | {rep.win_rate:.2%} | "
                f"{_fmt(rep.net_ev_r)} | {_fmt(rep.profit_factor_net)} |"
            )

    caveats = _caveats(result)
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in caveats] if caveats else ["- None."]

    lines += [
        "",
        "## Not run by v1",
        "",
        "The artifact-level holdout gate is not reproduced here: it engineers the",
        "holdout slice with the PRODUCTION feature list, so a candidate feature set",
        "cannot be scored by it without generalizing `_score_artifact_holdout`.",
        "The fold gate's `edge_over_random` is the lab's verdict; fold 3's",
        "validation window is the recent-regime check.",
        "",
        "## Interpretation",
        "",
        "_To be written after reading the numbers above._",
        "",
    ]
    if command:
        lines += ["## Command", "", "```bash", command, "```", ""]
    return "\n".join(lines)


def write_report(result, *, out_dir: Path = DEFAULT_REPORT_DIR, command: str = "") -> Path:
    """Render and write the report atomically; returns the file path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().date().isoformat()
    slug = result.spec.name.strip().lower().replace(" ", "-").replace("_", "-")
    path = out_dir / f"{stamp}_lab-{slug}.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(render_report(result, command=command))
    os.replace(tmp, path)
    return path


def _artifact_caveats(result) -> list:
    """The honesty box for a served-artifact replay."""
    out = []
    bt = result.backtest
    if bt.toll_mode == "flat":
        out.append(
            "Flat toll — the served artifact's schema has no `cost_ratio`, so trades "
            "are priced at the live default alpha (what the soak currently runs). "
            "The measured per-instrument alphas are a separate arm, not applied here."
        )
    else:
        out.append(
            "Toll priced with measured per-instrument alphas; the live bot currently "
            "prices at the flat default instead, so this replay is not its exact cost."
        )
    if bt.total_trades < 100:
        out.append(
            f"Only {bt.total_trades} replayed trades; win rate and net EV at this "
            "count are dominated by sampling noise."
        )
    out.append(
        "Replay, not a promotion gate: ONE artifact, ONE cached frame, no PASS/FAIL. "
        "The frame is largely in-sample for the artifact; the recorded-holdout rows "
        "in the window table are the only slice it never saw."
    )
    return out


def render_artifact_report(result, *, command: str = "") -> str:
    """Markdown for one served-artifact replay (lab.artifact)."""
    s = result.summary()
    spec = result.spec
    artifact = s["artifact"]
    now = datetime.now().astimezone()
    lines = [
        "---",
        "type: recon",
        f"date: {now.date().isoformat()}",
        f"time: {now.strftime('%H:%M %Z')}",
        "agent: opencode",
        "model: deepseek-flash",
        'trigger: "Served-artifact replay {dir} on spec {name} (hash {hash})"'.format(
            dir=artifact["model_dir"], name=spec.name, hash=s["content_hash"]
        ),
        f"head: {_git_head()}",
        "scope: lab replay only — no production model, config, or live path touched",
        "related:",
        "  - handoffs/2026-09-21_feature-lab-plan.md",
        "---",
        "",
        f"# Feature lab — served-artifact replay (`{spec.name}`)",
        "",
        "## Served artifact",
        "",
        f"- model dir: `{artifact['model_dir']}` | trained: "
        f"{artifact['trained_at'] or 'unknown'} | bars: Angel "
        f"{_fmt(artifact['angel_threshold'])} / Devil {_fmt(artifact['devil_threshold'])} "
        f"(from {artifact['threshold_source']})",
        f"- schema: {artifact['feature_count']} Angel features (`feature_names_in_`); "
        "Devil adds `angel_prob`",
    ]
    if artifact.get("trained_on_symbols"):
        lines.append(f"- trained on: {', '.join(artifact['trained_on_symbols'])}")
    lines += [
        "",
        "## Frame",
        "",
        f"- rows: {s['rows']:,} | features: {s['feature_count']} | "
        f"chop/behavior veto drop: {s['chop_veto_rate']:.2%} | "
        f"unresolvable tail purged: {s['purged_tail_rows']:,}",
        f"- {'loaded from frame cache' if s['frame_from_cache'] else 'built this run'}",
        "",
        "## Raw population (pre-live-gates)",
        "",
    ]
    population = s.get("population") or {}
    if population:
        lines += [
            f"- rows {population['rows']:,} | base rate {_fmt(population['base_rate'])}",
            f"- Angel proposals {population['proposed']} | Devil approvals "
            f"{population['approved']} | approval win rate "
            f"{_fmt(population['approved_win_rate'])} | edge over random "
            f"{_fmt(population['edge_over_random'])} | bracket PF "
            f"{_fmt(population['profit_factor'])}",
        ]
    else:
        lines.append("- Not computed for this replay.")

    if s.get("windows"):
        lines += [
            "",
            "## Window split (artifact's recorded holdout)",
            "",
            "| window | rows | base | proposed | approved | approval WR | edge | "
            "trades | trade WR | net EV (R) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for window in s["windows"]:
            lines.append(
                f"| {window['label']} | {window['rows']:,} | "
                f"{_fmt(window['base_rate'])} | {window['proposed']} | "
                f"{window['approved']} | {_fmt(window['approved_win_rate'])} | "
                f"{_fmt(window['edge_over_random'])} | {window['trades']} | "
                f"{_fmt(window['trade_win_rate'])} | {_fmt(window['net_ev_r'])} |"
            )

    lines += ["", "## Replay (live-gated)", ""]
    bt = s.get("backtest")
    if bt is None:
        lines.append("Not run for this replay.")
    else:
        lines += [
            f"- toll pricing: {bt['toll_mode']} | trades {bt['total_trades']} | "
            f"wins {bt['wins']} | win rate {bt['win_rate']:.2%}",
            f"- gross EV {_fmt(bt['gross_ev_r'])}R | net EV {_fmt(bt['net_ev_r'])}R | "
            f"net PF {_fmt(bt['profit_factor_net'])} | max drawdown "
            f"{_fmt(bt['max_drawdown_r'])}R",
            "- gate veto funnel: "
            + (
                ", ".join(f"{k}={v}" for k, v in sorted(bt["gate_rejections"].items()))
                or "none"
            ),
            "",
            "| symbol | trades | win rate | net EV (R) | net PF |",
            "|---|---:|---:|---:|---:|",
        ]
        for sym in sorted(result.backtest.per_symbol):
            rep = result.backtest.per_symbol[sym]
            lines.append(
                f"| {sym} | {rep.total_trades} | {rep.win_rate:.2%} | "
                f"{_fmt(rep.net_ev_r)} | {_fmt(rep.profit_factor_net)} |"
            )

    caveats = _artifact_caveats(result)
    lines += ["", "## Caveats", ""]
    lines += [f"- {caveat}" for caveat in caveats] if caveats else ["- None."]
    lines += [
        "",
        "## Interpretation",
        "",
        "_To be written after reading the numbers above._",
        "",
    ]
    if command:
        lines += ["## Command", "", "```bash", command, "```", ""]
    return "\n".join(lines)


def write_artifact_report(
    result,
    *,
    out_dir: Path = DEFAULT_REPORT_DIR,
    command: str = "",
    slug: Optional[str] = None,
) -> Path:
    """Render and write a replay report atomically; returns the file path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().date().isoformat()
    if not slug:
        spec_slug = result.spec.name.strip().lower().replace(" ", "-").replace("_", "-")
        slug = f"served-artifact-{spec_slug}"
    path = out_dir / f"{stamp}_lab-{slug}.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(render_artifact_report(result, command=command))
    os.replace(tmp, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Ablation reports (lab.ablate, W3 2026-09-22)
# ═══════════════════════════════════════════════════════════════════════════


def _ablation_caveats(result) -> list:
    """The honesty box for an ablation: thin trades dominate deltas."""
    from lab.ablate import THIN_TRADES

    out = []
    full = result.full.report
    if not full.gate_passed:
        out.append(
            "Full-cocktail gate FAILED — every arm replays Fold-3 placeholder "
            "models; the deltas compare two failed runs, not two promoted ones."
        )
    thin = [
        v.family
        for v in result.variants
        if v.report.pooled_oos_trades < THIN_TRADES
        or full.pooled_oos_trades < THIN_TRADES
    ]
    if full.pooled_oos_trades < THIN_TRADES or any(
        v.report.pooled_oos_trades < THIN_TRADES for v in result.variants
    ):
        out.append(
            f"Pooled OOS trades are thin (full {full.pooled_oos_trades}; "
            + ", ".join(f"{v.family} {v.report.pooled_oos_trades}" for v in result.variants)
            + f"; the caveat threshold is {THIN_TRADES}). At these counts a "
            "delta whose interval spans zero is NOT a drop decision and NOT "
            "evidence of no effect — it is sampling noise wearing a point "
            "estimate. Single runs of anything stochastic are noise."
        )
    out.append(
        "edge_over_random is in WIN-RATE units, not R (see the 2026-09-14 "
        "edge-budget work). Deltas inherit that currency."
    )
    out.append(
        "Each variant is a full walk-forward gate run: the delta compares two "
        "trained model PAIRS, not one model with a column removed. Family "
        "interactions are inside the models' fits, which is the question — "
        "but it also means a delta folds in every calibration shift the "
        "smaller feature set induces, not just the family's columns."
    )
    return out


def render_ablation_report(result, *, command: str = "") -> str:
    """Markdown for one ablation run (lab.ablate.AblationResult)."""
    s = result.summary()
    spec = result.spec
    now = datetime.now().astimezone()
    full = s["full"]
    lines = [
        "---",
        "type: recon",
        f"date: {now.date().isoformat()}",
        f"time: {now.strftime('%H:%M %Z')}",
        "agent: opencode",
        "model: deepseek-flash",
        'trigger: "Feature-interaction ablation {name} (full hash {hash})"'.format(
            name=spec.name, hash=full["content_hash"]
        ),
        f"head: {_git_head()}",
        "scope: lab run only — no production model, config, or live path touched",
        "related:",
        "  - handoffs/2026-09-22_feature-lab-v2.md",
        "  - handoffs/2026-09-22_feature-lab-v2-build-tasks.md",
        "---",
        "",
        f"# Feature lab — ablation (`{spec.name}`)",
        "",
        "## Question",
        "",
        "For each registered family X: **edge(full cocktail) − edge(cocktail − X)**,",
        "measured with only `feature_sets` varying — same bars, labels, veto,",
        "geometry, cost table and folds. The delta is a family's in-situ effect,",
        "not a lone-feature diagnostic.",
        "",
        f"- estimator: {s['model_family']} | wall time {s['run_seconds']:.1f}s",
        f"- full cocktail: {', '.join(spec.feature_sets)}",
        "",
        "## Ablation table",
        "",
        "| cocktail | trades | wins | edge over random | pooled PF lb | fold-3 PF lb | "
        "Δ edge (full − minus-X) | 90% CI on delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| **full** | {full['pooled_oos_trades']} | {full['pooled_oos_wins']} | "
        f"{_fmt(full['edge_over_random'])} | {_fmt(full['pooled_pf_lower_bound'])} | "
        f"{_fmt(full['fold3_pf_lower_bound'])} | — | — |",
    ]
    for row in s["variants"]:
        delta = row["delta"] or {}
        lines.append(
            f"| − {row['family']} | {row['pooled_oos_trades']} | "
            f"{row['pooled_oos_wins']} | {_fmt(row['edge_over_random'])} | "
            f"{_fmt(row['pooled_pf_lower_bound'])} | {_fmt(row['fold3_pf_lower_bound'])} | "
            f"**{_fmt(delta.get('delta'))}** | "
            f"[{_fmt(delta.get('ci_low'))}, {_fmt(delta.get('ci_high'))}] |"
        )

    lines += ["", "## Frame hashes", "", f"- full cocktail: `{full['content_hash']}`"]
    for row in s["variants"]:
        cache_note = "cached" if row["frame_from_cache"] else "built this run"
        lines.append(f"- − {row['family']}: `{row['content_hash']}` ({cache_note})")

    caveats = _ablation_caveats(result)
    lines += ["", "## Caveats", ""]
    lines += [f"- {caveat}" for caveat in caveats]
    lines += [
        "",
        "## Reading rule",
        "",
        "A family is a CANDIDATE FOR REMOVAL only when its delta's interval",
        "excludes zero on a trade count the gate itself calls evidential. A",
        "delta consistent with zero on thin trades is the expected regime for",
        "this basket — it is not a drop decision and not evidence of no effect.",
        "",
        "## Interpretation",
        "",
        "_To be written after reading the numbers above._",
        "",
    ]
    if command:
        lines += ["## Command", "", "```bash", command, "```", ""]
    return "\n".join(lines)


def write_ablation_report(
    result,
    *,
    out_dir: Path = DEFAULT_REPORT_DIR,
    command: str = "",
) -> Path:
    """Render and write an ablation report atomically; returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().date().isoformat()
    spec_slug = result.spec.name.strip().lower().replace(" ", "-").replace("_", "-")
    path = out_dir / f"{stamp}_lab-ablate-{spec_slug}.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(render_ablation_report(result, command=command))
    os.replace(tmp, path)
    return path
