import pandas as pd


def load_raw_dataset(context: dict) -> bool:
    headers = context['headers']

    context['raw_dataset'] = pd.read_fwf(
        filepath_or_buffer='resources/generated/dataset/LLCP2022.ASC ',
        colspecs=list(map(lambda x: (x['starts_at']-1, x['ends_at']-1), headers.values())),
        names=list(headers.keys()),
        header=None
    )

    return True


def load_raw_dataset_parquet(context: dict) -> bool:
    context['raw_dataset'] = pd.read_parquet('resources/generated/raw_dataset.parquet')

    return True
