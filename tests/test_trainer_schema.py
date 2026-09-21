"""
The estimator-schema contract: how a loaded model declares its columns.

`MLStrategy` sources its ENTIRE inference schema from
`V3RandomForestTrainer.feature_names_in_` and raises at boot when it is None —
deliberate, because guessing the column order is a silent-wrong-answer bug.
What broke (2026-09-14): CatBoost spells the same thing `feature_names_`, so a
CatBoost classifier reported no schema and `models/forex_h4_catboost` refused to
serve at all. These tests pin the fallback, including the case it must NOT
rescue (a model with no declared names stays schema-less rather than inventing
one).

Glossary:
    _NamedStub -- a stand-in exposing one spelling or the other, so the property
        is tested as a contract rather than against whatever library is
        installed.
    TestCatBoostSchemaFallback -- the fallback, both directions.
    TestStrategyBootsOnCatBoostPair -- the regression that mattered: an
        MLStrategy boot on a genuine CatBoost-typed Angel/Devil pair.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from core.retrainer import BASE_FEATURE_COLS
from ml.trainers.v3_rf_trainer import V3RandomForestTrainer

try:  # pragma: no cover - environment dependent
    from catboost import CatBoostClassifier

    _HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    _HAS_CATBOOST = False


class _NamedStub:
    """A model stand-in that declares column names the way one library does."""

    def __init__(self, attr: str, names):
        setattr(self, attr, names)


class TestCatBoostSchemaFallback(unittest.TestCase):
    def _trainer_holding(self, model) -> V3RandomForestTrainer:
        t = V3RandomForestTrainer()
        t.model = model
        return t

    def test_sklearn_spelling_wins_when_present(self):
        t = self._trainer_holding(
            _NamedStub("feature_names_in_", np.array(["a", "b"]))
        )
        self.assertEqual(t.feature_names_in_, ["a", "b"])

    def test_catboost_spelling_is_read(self):
        """The gap: CatBoost says feature_names_, not feature_names_in_."""
        t = self._trainer_holding(_NamedStub("feature_names_", ["a", "b"]))
        self.assertEqual(t.feature_names_in_, ["a", "b"])

    def test_ndarray_and_list_are_both_normalised(self):
        for names in (np.array(["a", "b"]), ["a", "b"], ("a", "b")):
            t = self._trainer_holding(_NamedStub("feature_names_", names))
            self.assertEqual(t.feature_names_in_, ["a", "b"])

    def test_no_declared_names_stays_none(self):
        """A model fitted on a bare numpy array declares nothing. The property
        must return None rather than invent a schema — MLStrategy's refusal to
        boot is the correct behaviour there, not a bug to paper over."""
        self.assertIsNone(self._trainer_holding(object()).feature_names_in_)

    def test_reloaded_stub_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.pkl"
            joblib.dump(_NamedStub("feature_names_", ["x", "y"]), path)
            t = V3RandomForestTrainer()
            t.load(str(path))
            self.assertEqual(t.feature_names_in_, ["x", "y"])


@unittest.skipUnless(_HAS_CATBOOST, "catboost not installed")
class TestStrategyBootsOnCatBoostPair(unittest.TestCase):
    """
    The regression that mattered: `models/forex_h4_catboost` raised
    "Loaded Angel model exposes no feature_names_in_" at boot, which took the
    whole H4 lane offline regardless of barriers.
    """

    @staticmethod
    def _fit(name: str, cols, n=400):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(n, len(cols)))
        y = (X[:, 0] > 0).astype(int)
        model = CatBoostClassifier(
            iterations=10, depth=3, verbose=0, allow_writing_files=False
        )
        model.fit(pl.DataFrame(X, schema=cols).to_pandas(), y)
        return model

    def test_ml_strategy_boots_on_catboost_angel_and_devil(self):
        from strategies.concrete_strategies.ml_strategy import MLStrategy

        cols = list(BASE_FEATURE_COLS)
        devil_cols = cols + ["angel_prob"]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            joblib.dump(self._fit("angel", cols), td / "angel_latest.pkl")
            joblib.dump(self._fit("devil", devil_cols), td / "devil_latest.pkl")
            # The constructor is the assertion: it raises rather than serve a
            # model whose schema it cannot read.
            strat = MLStrategy(
                asset_class="forex",
                angel_path=td / "angel_latest.pkl",
                devil_path=td / "devil_latest.pkl",
                warmup_period=10,
            )
        self.assertEqual(strat.feature_names, cols)
        self.assertEqual(strat.devil_trainer.feature_names_in_, devil_cols)


if __name__ == "__main__":
    unittest.main()
