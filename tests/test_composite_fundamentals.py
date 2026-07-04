"""Unit tests for CompositeFundamentalProvider — no network required."""

import sys
import unittest
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.fundamentals import FundamentalProvider
from data.providers.composite_fundamentals import CompositeFundamentalProvider


# ── Minimal fakes ─────────────────────────────────────────────────────────────

class _EmptyProvider(FundamentalProvider):
    def get_company_info(self, symbol): return {}
    def get_valuation_metrics(self, symbol): return {}
    def get_quarterly_financials(self, symbol): return pd.DataFrame()


class _FilledProvider(FundamentalProvider):
    def __init__(self, suffix="A"):
        self._suffix = suffix

    def get_company_info(self, symbol):
        return {"source": self._suffix, "sector": f"Tech{self._suffix}"}

    def get_valuation_metrics(self, symbol):
        return {"source": self._suffix, "trailingPE": 25.0}

    def get_quarterly_financials(self, symbol):
        return pd.DataFrame(
            {"Total Revenue": [1e9]},
            index=pd.DatetimeIndex(["2024-03-31"], name="period_end"),
        )


class _RaisingProvider(FundamentalProvider):
    """Belt-and-suspenders: child that violates the never-raise contract."""
    def get_company_info(self, symbol): raise RuntimeError("boom")
    def get_valuation_metrics(self, symbol): raise RuntimeError("boom")
    def get_quarterly_financials(self, symbol): raise RuntimeError("boom")


# ── Test cases ────────────────────────────────────────────────────────────────

class TestCompositeFundamentalProvider(unittest.TestCase):

    def test_first_non_empty_wins_company_info(self):
        """First provider with non-empty result is used."""
        comp = CompositeFundamentalProvider([_FilledProvider("A"), _FilledProvider("B")])
        result = comp.get_company_info("AAPL")
        self.assertEqual(result["source"], "A")

    def test_falls_through_to_second_when_first_empty_company_info(self):
        comp = CompositeFundamentalProvider([_EmptyProvider(), _FilledProvider("B")])
        result = comp.get_company_info("AAPL")
        self.assertEqual(result["source"], "B")

    def test_all_empty_returns_empty_dict(self):
        comp = CompositeFundamentalProvider([_EmptyProvider(), _EmptyProvider()])
        self.assertEqual(comp.get_company_info("AAPL"), {})

    def test_empty_provider_list_returns_empty_dict(self):
        comp = CompositeFundamentalProvider([])
        self.assertEqual(comp.get_company_info("AAPL"), {})

    def test_first_non_empty_wins_valuation_metrics(self):
        comp = CompositeFundamentalProvider([_FilledProvider("A"), _FilledProvider("B")])
        result = comp.get_valuation_metrics("AAPL")
        self.assertEqual(result["source"], "A")

    def test_falls_through_to_second_when_first_empty_valuation(self):
        comp = CompositeFundamentalProvider([_EmptyProvider(), _FilledProvider("B")])
        result = comp.get_valuation_metrics("AAPL")
        self.assertEqual(result["source"], "B")

    def test_all_empty_returns_empty_dict_valuation(self):
        comp = CompositeFundamentalProvider([_EmptyProvider(), _EmptyProvider()])
        self.assertEqual(comp.get_valuation_metrics("AAPL"), {})

    def test_empty_provider_list_returns_empty_dict_valuation(self):
        comp = CompositeFundamentalProvider([])
        self.assertEqual(comp.get_valuation_metrics("AAPL"), {})

    def test_first_non_empty_wins_quarterly_financials(self):
        comp = CompositeFundamentalProvider([_FilledProvider("A"), _FilledProvider("B")])
        result = comp.get_quarterly_financials("AAPL")
        self.assertFalse(result.empty)
        self.assertIn("Total Revenue", result.columns)

    def test_falls_through_to_second_when_first_empty_financials(self):
        comp = CompositeFundamentalProvider([_EmptyProvider(), _FilledProvider("B")])
        result = comp.get_quarterly_financials("AAPL")
        self.assertFalse(result.empty)

    def test_all_empty_returns_empty_dataframe(self):
        comp = CompositeFundamentalProvider([_EmptyProvider(), _EmptyProvider()])
        self.assertTrue(comp.get_quarterly_financials("AAPL").empty)

    def test_empty_provider_list_returns_empty_dataframe(self):
        comp = CompositeFundamentalProvider([])
        self.assertTrue(comp.get_quarterly_financials("AAPL").empty)

    def test_raising_child_does_not_propagate(self):
        """A child that violates the contract must not propagate the exception."""
        comp = CompositeFundamentalProvider([_RaisingProvider(), _FilledProvider("B")])
        # Should not raise; should fall through to the filled provider
        info = comp.get_company_info("AAPL")
        self.assertEqual(info["source"], "B")
        metrics = comp.get_valuation_metrics("AAPL")
        self.assertEqual(metrics["source"], "B")
        df = comp.get_quarterly_financials("AAPL")
        self.assertFalse(df.empty)


class TestGetFundamentalProviderFactory(unittest.TestCase):
    """Factory builds the right composite; SimFin constructor mocked to avoid key requirement."""

    def setUp(self):
        import os
        self._orig_env = os.environ.copy()
        # SimFin constructor only validates key presence — a fake value is enough
        os.environ.setdefault("SIMFIN_API_KEY", "test-fake-key")

    def tearDown(self):
        import os
        os.environ.clear()
        os.environ.update(self._orig_env)

    def _get(self, sources=None):
        import os
        from data.factory import get_fundamental_provider

        if sources is not None:
            os.environ["FUNDAMENTAL_SOURCES"] = sources
        else:
            os.environ.pop("FUNDAMENTAL_SOURCES", None)

        return get_fundamental_provider()

    def test_default_is_simfin_only(self):
        prov = self._get()
        self.assertEqual(type(prov).__name__, "CompositeFundamentalProvider")
        self.assertEqual(len(prov._providers), 1)
        self.assertEqual(type(prov._providers[0]).__name__, "SimFinFundamentalProvider")

    def test_simfin_explicit(self):
        prov = self._get("simfin")
        self.assertEqual(len(prov._providers), 1)
        self.assertEqual(type(prov._providers[0]).__name__, "SimFinFundamentalProvider")

    def test_simfin_comma_yfinance(self):
        prov = self._get("simfin,yfinance")
        self.assertEqual(len(prov._providers), 2)
        self.assertEqual(type(prov._providers[0]).__name__, "SimFinFundamentalProvider")
        self.assertEqual(type(prov._providers[1]).__name__, "YFinanceFundamentalProvider")

    def test_none_token_returns_empty_composite(self):
        prov = self._get("none")
        self.assertEqual(len(prov._providers), 0)

    def test_unknown_token_raises(self):
        import os
        from data.factory import get_fundamental_provider
        os.environ["FUNDAMENTAL_SOURCES"] = "bogussource"
        with self.assertRaises(ValueError):
            get_fundamental_provider()


if __name__ == "__main__":
    unittest.main()
