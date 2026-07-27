"""
The three plug-in contracts the ML factory is built from.

Everything in src/ml is assembled from these: a pipeline is just a list of
feature generators plus an optional target generator, so adding a new feature
family means writing one class, not editing the pipeline. Deliberately tiny and
dependency-free -- this file must stay importable from anywhere.

Glossary:
    BaseFeatureGenerator -- takes a bar DataFrame, returns it with new feature
        columns APPENDED. Generators must not drop or reorder existing rows;
        several of them depend on columns an earlier generator added, so order
        in the pipeline matters.
    BaseTargetGenerator -- same shape, but adds the answer column the model is
        trained to predict. Only used at training time; live inference has no
        target.
    BaseTrainer -- wraps a model: train, predict_proba, save, load.
    predict_proba -- returns probabilities rather than hard yes/no labels. The
        whole system depends on this: the thresholds that decide whether to
        trade are tuned against these continuous scores.
"""

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
