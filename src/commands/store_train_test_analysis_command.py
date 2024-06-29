import pandas as pd


def store_train_test_analysis(context: dict) -> bool:
    X_train = context['X_train']
    y_train = context['y_train']

    X_test = context['X_test']
    y_test = context['y_test']

    X_analysis = context['X_analysis']
    y_analysis = context['y_analysis']

    def store(X, y, name):
        y_df = pd.DataFrame(y, columns=['target'])
        df = pd.concat([X, y_df], axis=1)
        feature_names = X.columns.tolist() + ['target']
        df[feature_names].to_parquet(f'resources/processed/{name}', index=False)

    store(X_train, y_train, 'train.parquet')
    store(X_test, y_test, 'test.parquet')
    store(X_analysis, y_analysis, 'analysis.parquet')

    return True
