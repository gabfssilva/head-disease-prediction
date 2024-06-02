import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_analysis(context: dict) -> bool:
    X_train = context['X_train']
    y_train = context['y_train']

    _, X_analysis, _, y_analysis = train_test_split(X_train, y_train,
                                                    test_size=0.1,
                                                    stratify=y_train,
                                                    random_state=42
                                                    )

    y_analysis_df = pd.DataFrame(y_analysis, columns=['CVDINFR4'])
    analysis_dataframe = pd.concat([X_analysis, y_analysis_df], axis=1)
    feature_names = X_analysis.columns.tolist() + ['CVDINFR4']
    context['analysis_dataframe'] = analysis_dataframe[feature_names]

    return True
