from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from pipelines.builder import pipeline_builder
from util.scikitlearn_utils import numerical_transformer, categorical_transformer


def random_forest() -> GridSearchCV:
    return pipeline_builder(
        preprocessing=[RandomUnderSampler(random_state=42)],
        transformers=[
            numerical_transformer([StandardScaler()]),
            categorical_transformer([OneHotEncoder(sparse_output=False, handle_unknown='ignore')]),
        ],
        steps=[
            RandomForestClassifier(random_state=42),
        ],
        param_grid={
            'randomforestclassifier__n_estimators': [10, 15],
            'randomforestclassifier__max_depth': [10, 20, 50, 100],
            'randomforestclassifier__min_samples_split': [10, 15, 20],
        }
    )
