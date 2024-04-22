import pandas as pd

def raw_dataset() -> pd.DataFrame:
    return pd.read_parquet('../resources/generated/raw_dataset.parquet')

# CVDSTRK3 = Have yoy ever had a stroke?
print(raw_dataset()['CVDSTRK3'].value_counts())

print('-' * 50)

# CVDCRHD4 = Have you ever been diagnosed with angina or coronary heart disease?
print(raw_dataset()['CVDCRHD4'].value_counts())
