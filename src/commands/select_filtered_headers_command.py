import pandas as pd


def select_filtered_columns(context: dict) -> bool:
    headers = pd.read_json('resources/processed/selected_headers.json')
    columns = headers.columns.tolist()
    df = context['raw_dataset']
    df = df[columns]
    context['raw_dataset'] = df
    return True
