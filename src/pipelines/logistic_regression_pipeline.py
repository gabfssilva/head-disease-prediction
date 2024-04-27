from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipelines.builder import pipeline_builder
from src.util.scikitlearn_utils import numerical_transformer, categorical_transformer


def logistic_regression() -> GridSearchCV:
    return pipeline_builder(
        preprocessing_transformers=[
            numerical_transformer([
                KNNImputer(),
                StandardScaler()
            ]),
            categorical_transformer([
                SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ]),
        ],
        steps=[PCA(), LogisticRegression()],
        param_grid={
            'preprocessor__numerical__knnimputer__n_neighbors': [2, 3, 5, 10],

            'logisticregression__penalty': ['l2'],
            'logisticregression__max_iter': [2000, 10000],
            'logisticregression__C': [0.1, 1, 10],

            'pca__n_components': [2, 3, 5, 10, 20]
        },
        positive_class='1'
    )
