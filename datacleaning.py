from collections import Counter
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import parallel_backend
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd 
from utils import timer
import numpy as np
import config
import duckdb
from joblib import Parallel, delayed
import numba

bids_data=pd.read_csv("./database/bids.csv.zip")

def make_feature_speedup(series_temp):
    # result=pd.Series()
    df=pd.DataFrame.from_dict({'bidder_id':[series_temp['bidder_id']]})
    temp_df=duckdb.sql("SELECT * FROM  df, bids_data as Q  WHERE df.bidder_id==Q.bidder_id")
    if duckdb.sql("SELECT EXISTS(SELECT * FROM temp_df)").fetchall()[0][0]:
        return 1
    else:
        return 0
train_data=pd.read_csv("./database/train.csv.zip")
temp_feature=[]
for ind, i in train_data.iterrows():
    temp_feature.append(make_feature_speedup(i))
a=np.array(temp_feature)
ind=(a==1)
result=train_data[ind]
result.to_csv("train_clean.csv")
            
