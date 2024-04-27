from enum import Enum
from typing import Tuple, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector


class ColumnType(Enum):
    numerical = make_column_selector(dtype_include=['int', 'int32', 'int64', 'float', 'float32', 'float64'])
    categorical = make_column_selector(dtype_include=['object', 'category', 'bool'])


TransformerStep = Tuple[Union[BaseEstimator, TransformerMixin], ColumnType]
Step = BaseEstimator
