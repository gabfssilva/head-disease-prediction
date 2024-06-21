from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from tps.types import TransformerStep, Step


def pipeline_builder(
    preprocessing: list[Step],
    transformers: list[TransformerStep],
    steps: list[Step],
    param_grid: dict,
    positive_class: any = 'True'
) -> GridSearchCV:
    """
    Constructs a highly opinionated machine learning pipeline for binary classification
    and performs hyperparameter tuning using GridSearchCV.

    This function is designed to create a custom pipeline by integrating preprocessing steps and
    a sequence of transformations and estimators, followed by a strategy for hyperparameter optimization
    using cross-validation.

    Parameters:
    - transformers (list[TransformerStep]): A list of tuples, where each tuple contains
      a transformer and its associated metadata. These are used to preprocess the data.
    - steps (list[Step]): A list of estimators or transformers that define the steps of the pipeline
      after preprocessing.
    - param_grid (dict): A dictionary where keys are parameter names (strings) and values are lists of
      parameter settings to try as candidates.
    - positive_class (any): The label of the positive class which is crucial for computing metrics like
      precision, recall, and F1-score.

    Returns:
    - GridSearchCV: An instance of GridSearchCV which will execute the specified cross-validation strategy
      for hyperparameter tuning.

    Example:
    ```
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
    ```

    In this example, the function configures a pipeline for logistic regression, including preprocessing
    for both numerical and categorical data, principal component analysis (PCA), and logistic regression itself.
    The hyperparameters such as the number of neighbors for KNN imputation, regularization parameters for logistic
    regression, and the number of components for PCA are optimized using GridSearchCV.
    """

    precision_scorer = make_scorer(precision_score, pos_label=positive_class)
    recall_scorer = make_scorer(recall_score, pos_label=positive_class)
    f1_scorer = make_scorer(f1_score, pos_label=positive_class)

    cv_strategy = StratifiedKFold(
        n_splits=10,
        shuffle=False
    )

    preprocessor = ColumnTransformer(
        transformers=list(map(lambda x: (x[1].name, x[0], x[1].value), transformers))
    )

    pipeline = Pipeline(
        steps=[
            *make_pipeline(*preprocessing).steps,
            ('preprocessor', preprocessor),
            *make_pipeline(*steps).steps
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            'accuracy': 'accuracy',
            'f1': f1_scorer,
            'recall': recall_scorer,
            'precision': precision_scorer
        },
        n_jobs=-1,
        verbose=3,
        cv=cv_strategy,
        refit='f1'
    )
