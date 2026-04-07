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
