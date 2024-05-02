import pandas as pd

dataset: pd.DataFrame = (
    pd.read_parquet('../resources/generated/raw_dataset.parquet'))

# CVDSTRK3 = Have yoy ever had a stroke?
print(dataset['CVDSTRK3'].value_counts())

print('-' * 50)

# CVDCRHD4 = Have you ever been diagnosed with angina or coronary heart disease?
print(dataset['CVDCRHD4'].value_counts())
