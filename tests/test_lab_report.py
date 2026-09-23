"""
The lab's report emitter: a failed gate and a thin population must be called
out in the report, not just buried in the table.

The recon report is the artefact a human reads after an experiment; if a
30-trade +0.179 edge renders as a headline with no caveat, the lab re-creates
the exact misreading the 2026-09-14 session had to withdraw. These tests pin
the honesty box and the atomic write.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from lab.frames import FrameResult  # noqa: E402
from lab.report import render_report, write_report  # noqa: E402
from lab.spec import FeatureSpec  # noqa: E402


def fake_result(*, gate_passed=False, trades=30, use_spread_table=False):
    from lab.experiments import ExperimentResult

    report = SimpleNamespace(
        gate_passed=gate_passed,
        rejection_reasons=["EV below threshold"],
        mean_brier=0.2,
        mean_ev=-0.01,
        pooled_oos_trades=trades,
        pooled_oos_wins=max(1, trades // 2),
        pooled_pf_lower_bound=0.77,
        fold3_pf_lower_bound=0.59,
        pooled_base_rate=0.254,
        edge_over_random=0.179,
        production_angel_threshold=0.356,
        fold_metrics=[],
    )
    spec = FeatureSpec(
        name="caveat-test", feature_sets=("v3_base",), use_spread_table=use_spread_table
    )
    frame = FrameResult(
        spec_name=spec.name,
        content_hash="deadbeefdeadbeef",
        df=pl.DataFrame({"x": [1.0]}),
        feature_cols=("x",),
        chop_veto_rate=0.2,
        alpha_table=None,
        purged_tail_rows=3,
    )
    return ExperimentResult(
        spec=spec,
        frame=frame,
        gate=SimpleNamespace(
            report=report,
            model_family="lightgbm",
            production_threshold=0.66,
            elapsed_s=12.5,
        ),
        backtest=SimpleNamespace(
            total_trades=0,
            wins=0,
            win_rate=0.0,
            gross_ev_r=0.0,
            net_ev_r=0.0,
            profit_factor_net=0.0,
            max_drawdown_r=0.0,
            gate_rejections={},
            toll_mode="flat",
            per_symbol={},
        ),
        frame_from_cache=False,
        run_seconds=12.5,
    )


class TestRenderReport(unittest.TestCase):
    def test_thin_population_and_failed_gate_are_caveated(self):
        text = render_report(fake_result(trades=30), command="cmd")
        self.assertIn("## Caveats", text)
        self.assertIn("Gate FAILED", text)
        self.assertIn("30 pooled OOS trades", text)
        self.assertIn("Cost table OFF", text)
        self.assertIn("edge over random 0.1790", text)

    def test_thick_population_drops_the_noise_caveat(self):
        text = render_report(fake_result(trades=5000), command="cmd")
        self.assertNotIn("pooled OOS trades. At this count", text)
        self.assertIn("Gate FAILED", text)

    def test_spread_table_on_drops_the_cost_caveat(self):
        text = render_report(fake_result(use_spread_table=True), command="cmd")
        self.assertNotIn("Cost table OFF", text)

    def test_frontmatter_and_command_are_emitted(self):
        text = render_report(fake_result(), command="PYTHONPATH=src:. python -m lab.cli")
        self.assertTrue(text.startswith("---\ntype: recon\n"))
        self.assertIn("handoffs/2026-09-21_feature-lab-plan.md", text)
        self.assertIn("PYTHONPATH=src:. python -m lab.cli", text)


class TestWriteReport(unittest.TestCase):
    def test_writes_atomically_and_names_the_slug(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_report(fake_result(), out_dir=Path(tmp), command="cmd")
            self.assertTrue(out.is_file())
            self.assertIn("lab-caveat-test", out.name)
            leftovers = list(Path(tmp).glob("*.tmp"))
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
