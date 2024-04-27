from imblearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator

from src.types.types import TransformerStep, ColumnType


def column_transformer(
    steps: list[BaseEstimator],
    for_columns: ColumnType
) -> TransformerStep:
    return make_pipeline(*steps), for_columns


def numerical_transformer(steps: list[BaseEstimator]) -> TransformerStep:
    return column_transformer(steps, ColumnType.numerical)


def categorical_transformer(steps: list[BaseEstimator]) -> TransformerStep:
    return column_transformer(steps, ColumnType.categorical)
