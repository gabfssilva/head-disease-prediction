from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from joblib import Parallel, delayed
import numpy as np
from itertools import product
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from sk.all import *

@dataclass
class ExecutionInfo:
    scoring: str # main score or refit
    current_best_score: float  # best global based on refit
    current_fold: int  # current fold number
    current_execution: int  # the current execution of the total (splits * total combinations)


def eval_pipeline_with_progress_bar(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    positive_label='True',
    cv=StratifiedKFold(n_splits=10),
    scoring=[f1, accuracy, accuracy, precision, recall],
    workers: int = 7
) -> Dict[str, Any]:
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    total_executions = cv.n_splits * len(param_combinations)

    pb = tqdm(
        total=total_executions, 
        colour='blue', 
        desc='Initializing...', 
        bar_format="{l_bar}{postfix}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
    )

    def update_progress(info: ExecutionInfo):
        pb.set_description(f"Best score based on {info.scoring}")
        pb.set_postfix(info.current_best_score)
        pb.update(1)

    try:
        result = eval_pipeline(
            X, y, estimator, param_grid, 
            positive_label=positive_label,
            cv=cv, 
            scoring=scoring, 
            workers=workers, 
            callback=update_progress
        )
    finally:
        pb.close()

    return result


def eval_pipeline(
    X,
    y,
    estimator,
    param_grid,
    positive_label='True',
    cv=StratifiedKFold(n_splits=10),
    scoring=[f1, accuracy, precision, recall],
    callback: Callable[[int, int], None] = None,
    workers: int = 7
):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    scoring = list(map(lambda s: s(positive_label), scoring))
    scoring = reduce(lambda d, s: { **d, **s }, scoring, {})

    refit_score = list(scoring.keys())[0]

    best_score = -np.inf
    best_params = None
    results = []
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    execution = 0
    fold = 0

    def process_fold(estimator, params, X, y, train_index, test_index, scoring):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        estimator.set_params(**params)
        estimator.fit(X_train, y_train)

        return  {
            score_name: scorer(estimator, X_test, y_test) for score_name, scorer in scoring.items()
        }

    for params in param_combinations:
        fold_results = Parallel(n_jobs=workers, verbose=0, return_as='generator')(
            delayed(process_fold)(estimator, params, X, y, train_index, test_index, scoring)
            for train_index, test_index in cv.split(X, y)
        )

        fold_scores = {score_name: [] for score_name in scoring.keys()}
        for scores in fold_results:
            for score_name in scores:
                fold_scores[score_name].append(scores[score_name])

            execution += 1
            
            if callback:
                info = ExecutionInfo(
                    scoring=refit_score,
                    current_best_score=best_score,
                    current_fold=fold,
                    current_execution=execution
                )

                callback(info)

        avg_scores = {score_name: np.mean(values) for score_name, values in fold_scores.items()}
        results.append({
            "params": params,
            "avg_scores": avg_scores
        })

        current_score = avg_scores[refit_score]
        if current_score > best_score:
            best_score = current_score
            best_params = params
        
        fold += 1

    return {
        'best_params': best_params,
        'best_score': best_score,
        'results': results
    }

