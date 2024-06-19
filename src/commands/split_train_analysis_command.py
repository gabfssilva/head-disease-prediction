import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_analysis(context: dict) -> bool:
    X_train = context['X_train']
    y_train = context['y_train']

    _, X_analysis, _, y_analysis = train_test_split(X_train, y_train,
                                                    test_size=0.1,
                                                    stratify=y_train,
                                                    random_state=42)

    context['X_analysis'] = X_analysis
    context['y_analysis'] = y_analysis

    return True
