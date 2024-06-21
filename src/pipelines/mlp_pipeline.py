from imblearn.under_sampling import RandomUnderSampler

from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

import numpy as np

from pipelines.builder import pipeline_builder
from util.scikitlearn_utils import *


def mlp() -> GridSearchCV:
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
            MLPClassifier(random_state=42)
        ],
        param_grid={
            'mlpclassifier__hidden_layer_sizes': [
                (100, 100, 50, 50, 25, 25, 25, 10, 10,),
                (100,25,5,),
                (100,25,25,25,50,10,5,),
             ],
            'mlpclassifier__activation': ['logistic'],
            'mlpclassifier__solver': ['adam'],
            'mlpclassifier__alpha': [0.1, 0.01],
            'mlpclassifier__learning_rate': ['adaptive'],
            'mlpclassifier__max_iter': [250, 500, 1000],
            # 'mlpclassifier__batch_size': [128],
        }
    )
