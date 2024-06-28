import pandas as pd
from IPython.display import display
from imblearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report

from tps.types import TransformerStep, ColumnType


def column_transformer(
    steps: list[BaseEstimator],
    for_columns: ColumnType = ColumnType.all
) -> TransformerStep:
    return make_pipeline(*steps), for_columns


def numerical_transformer(steps: list[BaseEstimator]) -> TransformerStep:
    return column_transformer(steps, ColumnType.numerical)


def categorical_transformer(steps: list[BaseEstimator]) -> TransformerStep:
    return column_transformer(steps, ColumnType.categorical)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    if hasattr(model, 'best_params_'):
        print("Best Parameters:")
        print(model.best_params_)

    display("Eval on train set")
    print_confusion_matrix(y_train, model.predict(X_train), model)

    display("Eval on test set")
    print_confusion_matrix(y_test, model.predict(X_test), model)


def print_confusion_matrix(y_true, y_pred, model):
    class_names = {"False": "haven't suffered a heart attack", "True": "suffered a heart attack"}

    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(cm, index=[f'Actual {class_names[cls]}' for cls in model.classes_],
                         columns=[f'Predicted {class_names[cls]}' for cls in model.classes_])
    styled_cm = cm_df.style.background_gradient(cmap='Blues').format("{:.0f}")
    print("Confusion Matrix:")
    display(styled_cm)

    report = classification_report(
        y_true,
        y_pred,
        target_names=[class_names[cls] for cls in model.classes_],
        output_dict=True
    )

    metrics_df = pd.DataFrame(report).transpose()
    styled_metrics = metrics_df.style.highlight_max(
        subset=['f1-score', 'precision', 'recall'],
        color='lightgreen',
        axis=0
    ).format("{:.2f}")

    print("\nEvaluation Metrics:")
    display(styled_metrics)
