from sk.all import *

dt = lambda: pipeline_builder(
    scoring=[precision, f1, accuracy, balanced_accuracy, recall],
    steps=[
        random_under_sampler(random_state=42),
        column_transformers([
            numerical([standard_scaler()]),
            categorical([one_hot_encoding(sparse_output=False, handle_unknown='ignore')]),
        ]),
        decision_tree_classifier(random_state=42)
    ],
    param_grid={
        'decisiontreeclassifier__max_depth': [500],
        'decisiontreeclassifier__min_samples_split': [20],
        'decisiontreeclassifier__min_samples_leaf': [10],
        'decisiontreeclassifier__max_features': [25],
        'decisiontreeclassifier__criterion': ['gini']
    }
)


lr = lambda: pipeline_builder(
    scoring=[precision, f1, accuracy, balanced_accuracy, recall],
    steps=[
        random_under_sampler(random_state=42),
        column_transformers([
            numerical([
                simple_imputer(),
                standard_scaler()
            ]),
            categorical([
                simple_imputer(),
                one_hot_encoding(sparse_output=False, handle_unknown='ignore')
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
        'logisticregression__max_iter': [500],
        'logisticregression__C': [1],
    }
)

mlp = lambda: pipeline_builder(
    scoring=[precision, f1, accuracy, balanced_accuracy, recall],
    steps=[
        random_under_sampler(random_state=42),
        column_transformers([
            numerical([
                simple_imputer(),
                standard_scaler()
            ]),
            categorical([
                simple_imputer(),
                one_hot_encoding(sparse_output=False, handle_unknown='ignore')
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
            # (50, 25, 10, 5,),
            # (100, 25, 25, 5,),
            (25, 15, 10),
        ],

        'mlpclassifier__activation': ['logistic', 'relu'],
        'mlpclassifier__solver': ['adam'],
        'mlpclassifier__alpha': [0.01],
        'mlpclassifier__learning_rate': ['adaptive', 'constant', 'invscaling'],
        'mlpclassifier__max_iter': [500],
    }
)


rf = lambda: pipeline_builder(
    scoring=[precision, f1, accuracy, balanced_accuracy, recall],
    steps=[
        random_under_sampler(random_state=42),
        column_transformers([
            numerical([standard_scaler()]),
            categorical([one_hot_encoding(sparse_output=False, handle_unknown='ignore')]),
        ]),
        random_forest_classifier(random_state=42)
    ],
    param_grid={
        'randomforestclassifier__n_estimators': [100],
        'randomforestclassifier__max_depth': [100],
        'randomforestclassifier__min_samples_split': [15],
        'randomforestclassifier__max_features': ["sqrt"]
    }
)
