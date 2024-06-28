from collections import Counter
from enum import Enum
from functools import reduce
from itertools import product
from pprint import pprint
from typing import Any, Dict, List, Tuple, Union

import atomics
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, \
    balanced_accuracy_score, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, LearningCurveDisplay, ValidationCurveDisplay, \
    ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sk.enhanced_grid_search import EnhancedGridSearchCV

def _insert_line_breaks(title, max_len=60):
    words = title.split()
    final_title = ""
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > max_len:
            final_title += current_line + '\n'
            current_line = word + ' '
        else:
            current_line += word + ' '
    final_title += current_line
    return final_title.strip()


class Display:
    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y

    def plot_grid_search(self):
        cv_results = self.estimator.cv_results_
        param_grid = self.estimator.param_grid
        refit_metric = self.estimator.refit

        mean_test_score_key = _expression(
            predicate=refit_metric,
            value=lambda: f'mean_test_{refit_metric}',
            otherwise=lambda: 'mean_test_score'
        )

        std_test_score_key = _expression(
            predicate=self.estimator.refit,
            value=lambda: f'std_test_{refit_metric}',
            otherwise=lambda: 'std_test_score'
        )

        scores_mean = cv_results[mean_test_score_key]
        scores_sd = cv_results[std_test_score_key]

        # Generate combinations of parameter values
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())

        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())

        # Reshape scores_mean and scores_sd to match the parameter grid shape
        shape = tuple(len(param_grid[param]) for param in param_names)
        scores_mean = np.array(scores_mean).reshape(shape)
        scores_sd = np.array(scores_sd).reshape(shape)

        # Create subplots
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))

        # Plot Grid search scores for each combination
        for idx, param_comb in enumerate(param_combinations):
            param_label = ', '.join([f"{param_names[j]}: {param_comb[j]}" for j in range(len(param_comb))])

            param_label = param_label.replace("columntransformer", "ct")
            param_label = param_label.replace("simpleimputer", "si")
            param_label = param_label.replace("standardscalar", "ss")
            param_label = param_label.replace("categorical", "cat")
            param_label = param_label.replace("numerical", "num")
            param_label = param_label.replace("logisticregression", "logreg")
            param_label = param_label.replace("decisiontreeclassifier", "dtc")
            param_label = param_label.replace("randomforestclassifier", "rfc")

            param_label = reduce(lambda s, e: e + "\n" + s, param_label.split(", "), "")

            # param_label = _insert_line_breaks(param_label, 60)
            index_tuple = tuple(param_grid[param].index(param_comb[j]) for j, param in enumerate(param_names))
            score = scores_mean[index_tuple]
            ax.plot(idx, score, 'o', label=param_label)

        title = f"Grid Search Scores ({refit_metric})" if refit_metric else "Grid Search Scores"
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xlabel('Parameter Combination Index', fontsize=16)
        ax.set_ylabel('CV Average Score', fontsize=16)
        ax.grid('on')

        # Move the legend to the bottom
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

        plt.xticks(ticks=range(len(param_combinations)), labels=range(len(param_combinations)))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)  # Adjust bottom margin to make room for the legend
        plt.show()

    def confusion_matrix(self, ax=None):
        ax = _expression(ax is None, lambda: plt.gca(), lambda: ax)
        ConfusionMatrixDisplay.from_estimator(self.estimator, self.X, self.y, ax=ax)
        plt.show()

    def precision_recall(self, ax=None):
        ax = _expression(ax is None, lambda: plt.gca(), lambda: ax)
        PrecisionRecallDisplay.from_estimator(self.estimator, self.X, self.y, ax=ax)
        plt.show()

    def validation_curve(
        self,
        param_name,
        param_range,
        groups=None,
        cv=None,
        scoring=None,
        n_jobs=None,
        pre_dispatch="all",
        verbose=0,
        error_score=np.nan,
        fit_params=None,
        ax=None,
        negate_score=False,
        score_name=None,
        score_type="both",
        std_display_style="fill_between",
        line_kw=None,
        fill_between_kw=None,
        errorbar_kw=None,
    ):
        ax = _expression(ax is None, lambda: plt.gca(), lambda: ax)

        ValidationCurveDisplay.from_estimator(
            self.estimator, self.X, self.y,
            param_name=param_name,
            param_range=param_range,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
            error_score=error_score,
            fit_params=fit_params,
            ax=ax,
            negate_score=negate_score,
            score_name=score_name,
            score_type=score_type,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw
        )

        plt.show()

    def learning_curve(
        self,
        groups=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=None,
        scoring=None,
        exploit_incremental_learning=False,
        n_jobs=None,
        pre_dispatch="all",
        verbose=0,
        shuffle=False,
        random_state=None,
        error_score=np.nan,
        fit_params=None,
        ax=None,
        negate_score=False,
        score_name=None,
        score_type="both",
        std_display_style="fill_between",
        line_kw=None,
        fill_between_kw=None,
        errorbar_kw=None
    ):
        ax = _expression(ax is None, lambda: plt.gca(), lambda: ax)

        LearningCurveDisplay.from_estimator(
            self.estimator, self.X, self.y,
            groups=groups,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            exploit_incremental_learning=exploit_incremental_learning,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
            shuffle=shuffle,
            random_state=random_state,
            error_score=error_score,
            fit_params=fit_params,
            ax=ax,
            negate_score=negate_score,
            score_name=score_name,
            score_type=score_type,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw
        )

        plt.show()


class ColumnType(Enum):
    numerical = make_column_selector(dtype_include=['int', 'int32', 'int64', 'float', 'float32', 'float64'])
    categorical = make_column_selector(dtype_include=['object', 'category', 'bool'])
    all = make_column_selector(
        dtype_include=['int', 'int32', 'int64', 'float', 'float32', 'float64', 'object', 'category', 'bool'])


TransformerStep = Tuple[Union[BaseEstimator, TransformerMixin], ColumnType]
Step = BaseEstimator


def pca(
    n_components=None,
    copy=True,
    whiten=False,
    svd_solver="auto",
    tol=0.0,
    iterated_power="auto",
    n_oversamples=10,
    power_iteration_normalizer="auto",
    random_state=None
) -> PCA:
    return PCA(
        n_components,
        copy=copy,
        whiten=whiten,
        svd_solver=svd_solver,
        tol=tol,
        iterated_power=iterated_power,
        n_oversamples=n_oversamples,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state
    )


def standard_scaler(
    copy=True,
    with_mean=True,
    with_std=True
) -> StandardScaler:
    return StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)


def simple_imputer(
    missing_values=np.nan,
    strategy="mean",
    fill_value=None,
    copy=True,
    add_indicator=False,
    keep_empty_features=False,
) -> SimpleImputer:
    return SimpleImputer(
        missing_values=missing_values,
        strategy=strategy,
        fill_value=fill_value,
        copy=copy,
        add_indicator=add_indicator,
        keep_empty_features=keep_empty_features
    )


def logistic_regression(
    penalty="l2",
    dual=False,
    tol=1e-4,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver="lbfgs",
    max_iter=100,
    multi_class="deprecated",
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None,
) -> LogisticRegression:
    return LogisticRegression(
        penalty,
        dual=dual,
        tol=tol,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=random_state,
        solver=solver,
        max_iter=max_iter,
        multi_class=multi_class,
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs,
        l1_ratio=l1_ratio
    )


def random_forest_classifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        warm_start=warm_start,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        max_samples=max_samples
    )


def one_hot_encoding(
    categories="auto",
    drop=None,
    sparse_output=True,
    dtype=np.float64,
    handle_unknown="error",
    min_frequency=None,
    max_categories=None,
    feature_name_combiner="concat",
) -> OneHotEncoder:
    return OneHotEncoder(
        categories=categories,
        drop=drop,
        sparse_output=sparse_output,
        dtype=dtype,
        handle_unknown=handle_unknown,
        min_frequency=min_frequency,
        max_categories=max_categories,
        feature_name_combiner=feature_name_combiner
    )


def knn_imputer(
    missing_values=np.nan,
    n_neighbors=5,
    weights="uniform",
    metric="nan_euclidean",
    copy=True,
    add_indicator=False,
    keep_empty_features=False,
) -> KNNImputer:
    return KNNImputer(
        missing_values=missing_values,
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        copy=copy,
        add_indicator=add_indicator,
        keep_empty_features=keep_empty_features,
    )


def decision_tree_classifier(
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        random_state=random_state,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        monotonic_cst=monotonic_cst
    )


def svc(
    C=1.0,
    kernel="rbf",
    degree=3,
    gamma="scale",
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=1e-3,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape="ovr",
    break_ties=False,
    random_state=None,
) -> SVC:
    return SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=probability,
        tol=tol,
        cache_size=cache_size,
        class_weight=class_weight,
        verbose=verbose,
        max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        break_ties=break_ties,
        random_state=random_state
    )


def mlp_classifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="constant",
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=200,
    shuffle=True,
    random_state=None,
    tol=1e-4,
    verbose=False,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    validation_fraction=0.1,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    n_iter_no_change=10,
    max_fun=15000,
) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        power_t=power_t,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state,
        tol=tol,
        verbose=verbose,
        warm_start=warm_start,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        n_iter_no_change=n_iter_no_change,
        max_fun=max_fun
    )


def random_under_sampler(
    sampling_strategy="auto",
    random_state=None,
    replacement=False
) -> RandomUnderSampler:
    return RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        replacement=replacement
    )


def random_over_sampler(
    sampling_strategy: str = "auto",
    random_state=None,
    shrinkage=None,
) -> RandomOverSampler:
    return RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        shrinkage=shrinkage
    )


precision = lambda pos_label: {'precision': make_scorer(precision_score, pos_label=pos_label)}
accuracy = lambda *_: {'accuracy': make_scorer(accuracy_score)}
balanced_accuracy = lambda *_: {'balanced_accuracy': make_scorer(balanced_accuracy_score)}
recall = lambda pos_label: {'recall': make_scorer(recall_score, pos_label=pos_label)}
f1 = lambda pos_label: {'f1': make_scorer(f1_score, pos_label=pos_label)}


def kf(n_splits=5, shuffle=False, random_state=None) -> KFold:
    return KFold(n_splits, shuffle=shuffle, random_state=random_state)


def skf(n_splits=5, shuffle=False, random_state=None) -> KFold:
    return StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)


def column_transformers(
    transformers,
    remainder="drop",
    sparse_threshold=0.3,
    n_jobs=None,
    transformer_weights=None,
    verbose=False,
    verbose_feature_names_out=True,
    force_int_remainder_cols=True,
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=list(map(lambda x: (x[1].name, x[0], x[1].value), transformers)),
        remainder=remainder,
        sparse_threshold=sparse_threshold,
        n_jobs=n_jobs,
        transformer_weights=transformer_weights,
        verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out,
        force_int_remainder_cols=force_int_remainder_cols
    )


def numerical(
    transformers: list[BaseEstimator],
) -> TransformerStep:
    return make_pipeline(*transformers), ColumnType.numerical


def categorical(
    transformers: list[BaseEstimator],
) -> TransformerStep:
    return make_pipeline(*transformers), ColumnType.categorical


def pipeline_builder(
    steps: list[Step],
    param_grid: dict,
    positive_class: any = 'True',
    cross_validation=skf(n_splits=10, shuffle=False),
    scoring=[f1, accuracy, precision, recall, balanced_accuracy],
    n_jobs=7,
) -> EnhancedGridSearchCV:
    score_list = list(map(lambda s: s(positive_class), scoring))

    return EnhancedGridSearchCV(
        estimator=make_pipeline(*steps),
        param_grid=param_grid,
        scoring=reduce(lambda f, s: {**f, **s}, score_list, {}),
        n_jobs=n_jobs,
        cross_validation=cross_validation,
        positive_class=positive_class
    )


def results(estimator) -> pd.DataFrame:
    # Extract split scores
    split_scores = {k: v for (k, v) in estimator.cv_results_.items() if k.startswith('split')}

    # Extract parameters and flatten them
    params = estimator.cv_results_["params"]

    # Create a list to hold rows of data
    rows = []

    # Iterate over each parameter combination
    for idx, param_comb in enumerate(params):
        for fold in range(estimator.cv.n_splits):
            row = param_comb.copy()
            row['fold'] = fold
            for metric_name in split_scores.keys():
                if metric_name.startswith(f'split{fold}_'):
                    score = split_scores[metric_name][idx]
                    row[f'score_{metric_name.split("_")[2]}'] = score
            rows.append(row)

    # Convert rows to DataFrame
    df = pd.DataFrame(rows)

    return df

def display(estimator, X, y) -> Display:
    return Display(estimator, X, y)


def _expression(predicate: bool, value: callable, otherwise: callable = lambda: None):
    if predicate:
        return value()
    else:
        return otherwise()
