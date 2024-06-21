from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipelines.builder import pipeline_builder
from util.scikitlearn_utils import numerical_transformer, categorical_transformer


def logistic_regression() -> GridSearchCV:
    return pipeline_builder(
        preprocessing=[RandomUnderSampler(random_state=42)],
        transformers=[
            numerical_transformer([
                SimpleImputer(strategy='most_frequent'),
                StandardScaler()
            ]),
            categorical_transformer([
                SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ]),
        ],
        steps=[
            LogisticRegression()
        ],
        param_grid={
            'logisticregression__penalty': ['l2'],
            'logisticregression__max_iter': [1000, 2000],
            'logisticregression__C': [0.1, 1],
        }
    )
