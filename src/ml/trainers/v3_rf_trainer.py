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
