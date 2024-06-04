from sklearn.model_selection import train_test_split


def split_train_test(context: dict) -> bool:
    dataset = context['raw_dataset']
    y = dataset['CVDINFR4']
    X = dataset.drop('CVDINFR4', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)

    context['X_train'] = X_train
    context['X_test'] = X_test
    context['y_train'] = y_train
    context['y_test'] = y_test

    return True
