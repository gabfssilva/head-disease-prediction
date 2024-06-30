from sk.all import *

all_scoring = [
    f1, 
    recall,
    precision, 
    accuracy, 
    balanced_accuracy, 
]

dt = lambda: pipeline_builder(
    scoring=all_scoring,
    steps=[
        random_under_sampler(random_state=42),
        column_transformers([
            numerical([standard_scaler()]),
            categorical([one_hot_encoder(sparse_output=False, handle_unknown='ignore')]),
        ]),
        decision_tree_classifier(random_state=42)
    ],
    param_grid={
        'decisiontreeclassifier__max_depth': [100],
        'decisiontreeclassifier__min_samples_split': [3],
        'decisiontreeclassifier__min_samples_leaf': [10],
        'decisiontreeclassifier__max_features': ["log2"],
        'decisiontreeclassifier__criterion': ['gini']
    }
)


lr = lambda: pipeline_builder(
    scoring=all_scoring,
    steps=[
        random_under_sampler(random_state=42),
        column_transformers([
            numerical([
                simple_imputer(),
                standard_scaler()
            ]),
            categorical([
                simple_imputer(),
                one_hot_encoder(sparse_output=False, handle_unknown='ignore')
            ])
        ]),
        pca(random_state=42),
        logistic_regression(random_state=42)
    ],
    param_grid={
        'columntransformer__numerical__simpleimputer__strategy': ['mean'],
        'columntransformer__categorical__simpleimputer__strategy': ['constant'],
        'columntransformer__categorical__simpleimputer__fill_value': ['missing'],
        'columntransformer__categorical__simpleimputer__missing_values': ['<NA>'],

        'pca__n_components': [None],
        'logisticregression__penalty': ['l2'],
        'logisticregression__max_iter': [100, 250, 500, 1000],
        'logisticregression__C': [1, 0.1],
    }
)

mlp = lambda: pipeline_builder(
    scoring=all_scoring,
    steps=[
        random_under_sampler(random_state=42),
        column_transformers([
            numerical([
                simple_imputer(),
                standard_scaler()
            ]),
            categorical([
                simple_imputer(),
                one_hot_encoder(sparse_output=False, handle_unknown='ignore')
            ]),
        ]),
        mlp_classifier(random_state=42)
    ],
    param_grid={
        'columntransformer__numerical__simpleimputer__strategy': ['median'],
        'columntransformer__categorical__simpleimputer__strategy': ['constant'],
        'columntransformer__categorical__simpleimputer__fill_value': ['missing'],
        'columntransformer__categorical__simpleimputer__missing_values': ['<NA>'],
        'mlpclassifier__hidden_layer_sizes': [
            (100, 25, 10),
            (75, 20, 5),
            (50, 15, 5),
        ],
        'mlpclassifier__activation': ['logistic'],
        'mlpclassifier__solver': ['adam'],
        'mlpclassifier__alpha': [0.1],
        'mlpclassifier__learning_rate': ['adaptive', 'constant'],
        'mlpclassifier__max_iter': [50, 100],
    }
)


rf = lambda: pipeline_builder(
    scoring=all_scoring,
    steps=[
        random_under_sampler(random_state=42),
        column_transformers([
            numerical([standard_scaler()]),
            categorical([one_hot_encoder(sparse_output=False, handle_unknown='ignore')]),
        ]),
        random_forest_classifier(random_state=42)
    ],
    param_grid={
        'randomforestclassifier__n_estimators': [250, 500],
        'randomforestclassifier__max_depth': [50, 100],
        'randomforestclassifier__min_samples_split': [10, 20],
        'randomforestclassifier__min_samples_leaf': [5, 10],
        'randomforestclassifier__max_features': ["sqrt"],
        'randomforestclassifier__criterion': ['log_loss', 'gini']
    }
)

svm = lambda: pipeline_builder(
    scoring=all_scoring,
    steps=[
        random_under_sampler(random_state=42),
        column_transformers([
            numerical([
                simple_imputer(),
                standard_scaler()
            ]),
            categorical([
                simple_imputer(),
                one_hot_encoder(sparse_output=False, handle_unknown='ignore')
            ])
        ]),
        svc(random_state=42)
    ],
    param_grid={
        'columntransformer__numerical__simpleimputer__strategy': ['mean'],
        'columntransformer__categorical__simpleimputer__strategy': ['constant'],
        'columntransformer__categorical__simpleimputer__fill_value': ['missing'],
        'columntransformer__categorical__simpleimputer__missing_values': ['<NA>'],

        'svc__C': [0.1, 1],
    }
)
