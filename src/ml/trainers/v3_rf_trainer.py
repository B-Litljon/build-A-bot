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
        mismatch here is a silent-wrong-answer bug, not a crash.
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
        return getattr(self.model, "feature_names_in_", None)
