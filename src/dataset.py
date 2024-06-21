import pandas as pd

from pipelines.decision_tree_pipeline import *

df = pd.read_parquet('../resources/processed/train.parquet')
df_test = pd.read_parquet('../resources/processed/test.parquet')

y_train = df['target']
X_train = df.drop(['CVDINFR4', 'target'], axis=1)

y_test = df_test['target']
X_test = df_test.drop(['CVDINFR4', 'target'], axis=1)
