from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pipelines.builder import pipeline_builder
from util.scikitlearn_utils import numerical_transformer, categorical_transformer


def decision_tree() -> GridSearchCV:
    return pipeline_builder(
        preprocessing=[RandomUnderSampler(random_state=42)],
        transformers=[
            numerical_transformer([StandardScaler()]),
            categorical_transformer([OneHotEncoder(sparse_output=False, handle_unknown='ignore')]),
        ],
        steps=[
            DecisionTreeClassifier()
        ],
        param_grid={
            'decisiontreeclassifier__max_depth': [50, 100, 200],
            'decisiontreeclassifier__min_samples_split': [20, 50],
            'decisiontreeclassifier__min_samples_leaf': [3, 10],
            # 'decisiontreeclassifier__max_features': ['sqrt', 'log2', 500],
            'decisiontreeclassifier__criterion': ['gini', 'entropy']
        }
    )
