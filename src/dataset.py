# import pandas as pd
# from pipelines.all_pipelines import *

# df = pd.read_parquet('resources/processed/train.parquet')
# df_test = pd.read_parquet('resources/processed/test.parquet')

# y_train = df['target']
# X_train = df.drop('target', axis=1)

# y_test = df_test['target']
# X_test = df_test.drop('target', axis=1)

# lr = lr()
# lr.fit(X_train, y_train)


# # mlp = mlp()
# # mlp.fit(X_train, y_train)

# # dt = dt()
# # dt.fit(X_train, y_train)

# # rf = rf()
# # rf.fit(X_train, y_train)