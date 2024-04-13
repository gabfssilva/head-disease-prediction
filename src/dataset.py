import pandas as pd


def dataset() -> pd.DataFrame:
    # TODO: load dataset using the provided header mappings
    return pd.read_fwf('resources/dataset.zip')
