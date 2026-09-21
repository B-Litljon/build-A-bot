"""
Thin BaseTrainer wrapper around scikit-learn's RandomForestClassifier.

⚠️ NAME vs REALITY: despite "RandomForest" in the name, this class is now used
mostly as a LOAD-AND-PREDICT shell. Production models are LightGBM (the switch
happened 2026-05-23 and was what finally cleared the validation gate), and
``load()`` simply unpickles whatever estimator is on disk -- so a
``V3RandomForestTrainer`` in the live strategy is typically holding a LightGBM
model. Only the legacy ``train_model.py`` path still trains an actual forest
through it.

Glossary:
    V3RandomForestTrainer -- the wrapper. Constructor kwargs pass straight
        through to RandomForestClassifier; the live paths never use that.
    model -- the wrapped estimator. Whatever ``load()`` read from the pickle.
    train / predict_proba / predict -- straight delegations.
    save / load -- joblib pickle round-trip. Note ``load`` REPLACES ``model``
        entirely, which is how a LightGBM model ends up inside a class named
        for random forests.
    feature_names_in_ -- the column names the fitted model expects, or None.
        Used to check that live features line up with training features; a
        mismatch here is a silent-wrong-answer bug, not a crash. Reads BOTH
        spellings: sklearn says ``feature_names_in_``, CatBoost says
        ``feature_names_``, and without the fallback a CatBoost classifier
        reports no schema at all — which is what refused to serve the H4
        CatBoost candidate (2026-09-14).
"""

from sklearn.ensemble import RandomForestClassifier
import joblib
from ml.core.interfaces import BaseTrainer

class V3RandomForestTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        # Default Random Forest parameters for V3 architecture
        # Override these by passing kwargs to __init__
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

    @property
    def feature_names_in_(self):
        """
        The column names the loaded estimator was fitted on, or None.

        Both spellings are read because the library changed under this class:
        sklearn and LightGBM expose ``feature_names_in_``, CatBoost exposes
        ``feature_names_`` (no "in"). They mean the same thing — the training
        column ORDER — so the fallback is exact rather than approximate, and
        both are normalised to a list because sklearn returns an ndarray while
        CatBoost returns a list.

        Load-bearing, not cosmetic: MLStrategy sources its whole inference
        schema from this property and REFUSES to boot when it is None, so a
        CatBoost artifact used to be unserveable no matter how good it was
        (verified 2026-09-14: models/forex_h4_catboost raised at boot). A model
        fitted on a bare numpy array carries neither spelling and still returns
        None — that case remains a genuine "no declared schema".
        """
        names = getattr(self.model, "feature_names_in_", None)
        if names is None:
            names = getattr(self.model, "feature_names_", None)
        return list(names) if names is not None else None
