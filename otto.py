import pandas as pd
import numpy as np
import seaborn as sn
import sklearn
from sklearn_pandas import DataFrameMapper, cross_val_score
import re

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')
feats = [key for key in df.keys() if re.match('.*feat.*',key)]
mapper = DataFrameMapper([(feats,sklearn.preprocessing.StandardScaler())])
data_train_scaled = mapper.fit_transform(df_train)
data_test_scaled = mapper.transform(df_test)
data_test = df_test[feats]
data_train =df_train[feats]