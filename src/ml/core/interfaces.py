from abc import ABC, abstractmethod
import polars as pl

class BaseFeatureGenerator(ABC):
    @abstractmethod
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generates features and appends them to the dataframe."""
        pass

class BaseTargetGenerator(ABC):
    @abstractmethod
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generates targets and appends them to the dataframe."""
        pass

class BaseTrainer(ABC):
    @abstractmethod
    def train(self, X, y):
        """Fits the model."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Returns continuous probability scores."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Saves the model artifact to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Loads the model artifact from disk."""
        pass
