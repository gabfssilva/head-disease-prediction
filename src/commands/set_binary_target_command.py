import pandas as pd


def set_binary_target(context: dict) -> bool:
    df: pd.DataFrame = context['raw_dataset']
    df = df[df['CVDINFR4'].isin(['1', '2'])]
    print(df)
    df['target'] = (df['CVDINFR4'] == '1').astype(str)
    df.drop(['CVDINFR4'], axis=1, inplace=True)
    context['raw_dataset'] = df
    return True
