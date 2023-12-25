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

bids_data=pd.read_csv(config.IN_DIR/"bids.csv.zip")

def plot_bar_dic(my_dict):
    keys = list(my_dict.keys())
    # get values in the same order as keys, and parse percentage values
    vals = [float(my_dict[k]) for k in keys]
    key_len=np.arange(len(keys))
    sns.barplot(x=key_len, y=vals)

def bid_per_auction(temp_bids):
    auc_counter=Counter(temp_bids['auction'])
    CC=np.array(list(auc_counter.values()))
    return CC.mean()

def med_bid_time(temp_bids):
    temp_time=list(temp_bids['time'])
    temp_time.sort()
    temp_time_before=np.array(temp_time[0:-1])
    temp_time_after=np.array(temp_time[1:])
    if len(temp_time_after)>0:
        return np.median(np.array(temp_time_after-temp_time_before))
    else:
        return temp_time[0]
def bid_per_auction_speedup(series_temp):
    df=pd.DataFrame.from_dict({'bidder_id':[series_temp['bidder_id']]})
    temp=duckdb.query("SELECT COUNT(*) as count_num FROM  df, bids_data as Q  WHERE df.bidder_id==Q.bidder_id GROUP BY Q.auction")
    if temp!=None:
        temp2=duckdb.query("SELECT MEAN(temp.count_num) as mean_num FROM  temp").fetchall()
        return temp2[0]
    else:
        return 0
@timer
def make_feature(train_data):
    feature=pd.DataFrame()
    feature['bid_per_auction']=train_data.apply(bid_per_auction_speedup,axis=1)
    feature['out']=train_data['outcome']
    print(feature)
    return feature
    

    