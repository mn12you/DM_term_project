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

bids_data=pd.read_csv(config.IN_DIR/"bids.csv.zip")

def plot_bar_dic(my_dict):
    keys = list(my_dict.keys())
    # get values in the same order as keys, and parse percentage values
    vals = [float(my_dict[k]) for k in keys]
    key_len=np.arange(len(keys))
    sns.barplot(x=key_len, y=vals)

# def bid_per_auction(temp_bids):
#     auc_counter=Counter(temp_bids['auction'])
#     CC=np.array(list(auc_counter.values()))
#     return CC.mean()

# def med_bid_time(temp_bids):
#     temp_time=list(temp_bids['time'])
#     temp_time.sort()
#     temp_time_before=np.array(temp_time[0:-1])
#     temp_time_after=np.array(temp_time[1:])
#     if len(temp_time_after)>0:
#         return np.median(np.array(temp_time_after-temp_time_before))
#     else:
#         return temp_time[0]
    
def make_feature_speedup(series_temp):
    # result=pd.Series()
    df=pd.DataFrame.from_dict({'bidder_id':[series_temp['bidder_id']]})
    temp_df=duckdb.sql("SELECT * FROM  df, bids_data as Q  WHERE df.bidder_id==Q.bidder_id")
    if duckdb.sql("SELECT EXISTS(SELECT * FROM temp_df)").fetchall()[0][0]:
        # result['bid_per_auction']=bid_per_auction_speedup(temp_df)
        # result['med_bid_time']= med_bid_time_speedup(temp_df)
        result1=bid_per_auction_speedup(temp_df)
        result2=med_bid_time_speedup(temp_df)
    else:
        # result['bid_per_auction']=0
        # result['med_bid_time']=0
        result1=0
        result2=0
    return result1,result2
    
def bid_per_auction_speedup(temp_df):
    temp=duckdb.sql("SELECT COUNT(*) as count_num FROM  temp_df GROUP BY temp_df.auction")
    temp2=duckdb.sql("SELECT MEAN(temp.count_num) as mean_num FROM  temp ").fetchall()
    return temp2[0][0]

def med_bid_time_speedup(temp_df):
    temp=duckdb.query("SELECT  temp_df.time FROM  temp_df ORDER BY temp_df.time").fetchall()
    array_temp=np.array(temp)
    if array_temp.shape[0]>1:
        array_temp2=array_temp[1:,:]-array_temp[0:-1,:]
        return np.median(array_temp2)
    else:
        return array_temp[0,0]

@timer
def make_feature(train_data):
    feature=pd.DataFrame()
    temp_feature= Parallel(n_jobs=4)(
            delayed(make_feature_speedup)(i) for ind, i in train_data.iterrows())
    # feature=train_data.apply(make_feature_speed?=up,axis=1)
    feature['bid_per_auction']=[i[0] for i in temp_feature]
    feature['med_bid_time']=[i[1] for i in temp_feature]
    feature['out']=train_data['outcome']
    return feature
    
@timer 
def train_classifier(feature_data):
    x=feature_data.drop(columns=['out'])
    x= preprocessing.normalize(x.values, axis=0)
    y=feature_data['out']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf=RandomForestClassifier(n_jobs=4,n_estimators=1, max_depth=None, min_samples_leaf=1, random_state=42, criterion='entropy')
    # clf=HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
    clf.fit(X_train, y_train)
    score=clf.score(X_test, y_test)
    print("Classifier score: ", score)
    return clf

@timer 
def make_feature_test(test_data)->pd.DataFrame:
    feature=pd.DataFrame()
    temp_feature= Parallel(n_jobs=4)(
            delayed(make_feature_speedup)(i) for ind, i in test_data.iterrows())
    # feature=train_data.apply(make_feature_speed?=up,axis=1)
    feature['bid_per_auction']=[i[0] for i in temp_feature]
    feature['med_bid_time']=[i[1] for i in temp_feature]
            
    return feature

# def test_submission(test_data,clf):
#     feature=make_feature_test(test_data)
#     with parallel_backend('threading', n_jobs=4):
#         output=clf.predict(feature)
#     return output
    