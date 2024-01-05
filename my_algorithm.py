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
from numba import jit
import datetime
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier



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
    result=[]
    df=pd.DataFrame.from_dict({'bidder_id':[series_temp['bidder_id']]})
    temp_df=duckdb.sql("SELECT * FROM  df, bids_data as Q  WHERE df.bidder_id==Q.bidder_id")
    # if duckdb.sql("SELECT EXISTS(SELECT * FROM temp_df)").fetchall()[0][0]:
        # result['bid_per_auction']=bid_per_auction_speedup(temp_df)
        # result['med_bid_time']= med_bid_time_speedup(temp_df)
    result.append(bid_per_auction(temp_df))
    result.append(med_bid_time(temp_df))
    result.append(ip_entropy(temp_df))
    result.append(bid_entropy(temp_df))
    result.append(URL_entropy(temp_df))
    result.append(mean_bid_url(temp_df))
    result.append(bid_count(temp_df))
    result.append(entropy_per_auction(temp_df))
    # temp=min_bid_time(temp_df)
    # result.append(temp[0])
    # result.append(temp[1])
    # else:
    #     # result['bid_per_auction']=0
    #     # result['med_bid_time']=0
    #     result1=0
    #     result2=0
    #     result3=0
    # # return result1,result2,result3
    return result

def med_bid_time(temp_df):
    temp=duckdb.query("SELECT  temp_df.time FROM  temp_df ORDER BY temp_df.time").fetchall()
    array_temp=np.array(temp)
    if array_temp.shape[0]>1:
        array_temp2=array_temp[1:,:]-array_temp[0:-1,:]
        return np.median(array_temp2)
    else:
        return array_temp[0,0]
    
def bid_per_auction(temp_df):
    temp=duckdb.sql("SELECT COUNT(*) as count_num FROM  temp_df GROUP BY temp_df.auction")
    temp2=duckdb.sql("SELECT MEAN(temp.count_num) as mean_num FROM  temp ").fetchall()
    return temp2[0][0]

def bid_entropy(temp_df):
    temp=duckdb.query("SELECT temp_df.time FROM  temp_df ").fetchall()
    np_array=self_time(np.array(temp))
    week=pd.DataFrame(np_array,columns=['day'])
    temp=duckdb.query("SELECT COUNT(*) FROM  week GROUP BY week.day").fetchall()
    return self_entropy(np.array(temp).astype(np.float32))

def entropy_per_auction(temp_df):
    temp=duckdb.sql("SELECT temp_df.auction FROM  temp_df GROUP BY temp_df.auction").df()  
    temp2=temp_df.df()
    def speed_up(temp_inside):
        temp_array=temp2[temp2['auction']==temp_inside.auction]
        ip=duckdb.sql("SELECT COUNT(*) FROM  temp_array GROUP BY temp_array.ip").fetchall()
        url=duckdb.sql("SELECT COUNT(*) FROM  temp_array GROUP BY temp_array.url").fetchall()
        return self_entropy(np.array(ip)), self_entropy(np.array(url))
    
    feature=pd.DataFrame()
    # temp_feature[]
    feature[['ip','url']]= temp.apply(speed_up,axis=1,result_type='expand')
    ip=duckdb.sql("SELECT MEAN(feature.ip) FROM  feature ").fetchall() 
    url=duckdb.sql("SELECT MEAN(feature.url) FROM  feature ").fetchall() 
    result=(ip[0][0]+url[0][0])/2
    return result

def mean_bid_url(temp_df):
    temp=duckdb.query("SELECT  COUNT(*) as count_num FROM  temp_df GROUP BY temp_df.url ")
    temp2=duckdb.sql("SELECT MEAN(temp.count_num) as mean_num FROM  temp").fetchall()
    return temp2[0][0]

def bid_count(temp_df):
    temp=duckdb.query("SELECT  COUNT(*) FROM  temp_df ").fetchall()
    return temp[0][0]

def ip_entropy(temp_df):
    temp=duckdb.query("SELECT  COUNT(*) FROM  temp_df GROUP BY temp_df.ip").fetchall()
    array_temp=np.array(temp).astype(np.float32)
    return self_entropy(array_temp)

def URL_entropy(temp_df):
    temp=duckdb.query("SELECT  COUNT(*) FROM  temp_df GROUP BY temp_df.url").fetchall()
    array_temp=np.array(temp).astype(np.float32)
    return self_entropy(array_temp)


@jit(nopython=True, nogil=True)
def self_time(np_array):
    day_array=np_array/24
    week_array=np.ceil(day_array%7)
    return week_array

@jit(nopython=True, nogil=True)
def self_entropy(np_array):
    all_array=np.sum(np_array)
    np_array=np_array/all_array
    log_array=np.log(np_array)
    array_temp=-log_array*np_array
    result=np.sum(array_temp)
    return result
@timer
def make_feature(train_data):
    feature=pd.DataFrame()
    temp_feature= Parallel(n_jobs=-1)(
            delayed(make_feature_speedup)(i) for ind, i in train_data.iterrows())
    # feature=train_data.apply(make_feature_speed?=up,axis=1)
    feature['bid_per_auction']=[i[0] for i in temp_feature]
    feature['med_bid_time']=[i[1] for i in temp_feature]
    feature['ip_entropy']=[i[2] for i in temp_feature]
    feature['bid_entropy']=[i[3] for i in temp_feature]
    feature['url_entropy']=[i[4] for i in temp_feature]
    feature['mean_bid_url']=[i[5] for i in temp_feature]
    feature['bid_count']=[i[6] for i in temp_feature]
    feature['entropy_per_auction']=[i[7] for i in temp_feature]
    feature['out']=train_data['outcome']
    return feature
    
@timer 
def train_classifier(feature_data,a):
    x=feature_data.drop(columns=['out'])
    x= preprocessing.normalize(x.values, axis=0)
    y=feature_data['out']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train,y_train)
    # clf=RandomForestClassifier(n_jobs=4,n_estimators=5, random_state=42, criterion='entropy')
    clf=GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, random_state=42).fit(X_res, y_res)

    clf.fit(X_res, y_res)
    score=clf.score(X_test, y_test)
    print("Classifier score: ", score)
    model_path=config.MODEL_DIR/a.model
    dump(clf,model_path)
    return clf

def make_feature_test_speedup(series_temp):
    # result=pd.Series()
    result=[]
    df=pd.DataFrame.from_dict({'bidder_id':[series_temp['bidder_id']]})
    temp_df=duckdb.sql("SELECT * FROM  df, bids_data as Q  WHERE df.bidder_id==Q.bidder_id")
    if duckdb.sql("SELECT EXISTS(SELECT * FROM temp_df)").fetchall()[0][0]:
        result.append(bid_per_auction(temp_df))
        result.append(med_bid_time(temp_df))
        result.append(ip_entropy(temp_df))
        result.append(bid_entropy(temp_df))
        result.append(URL_entropy(temp_df))
        result.append(mean_bid_url(temp_df))
        result.append(bid_count(temp_df))
        result.append(entropy_per_auction(temp_df))
    else:
        result=[0,0,0,0,0,0,0,0,0]
    return result


@timer 
def make_feature_test(test_data)->pd.DataFrame:
    feature=pd.DataFrame()
    temp_feature= Parallel(n_jobs=-1)(
            delayed(make_feature_test_speedup)(i) for ind, i in test_data.iterrows())
    # feature=train_data.apply(make_feature_speed?=up,axis=1)
    feature['bid_per_auction']=[i[0] for i in temp_feature]
    feature['med_bid_time']=[i[1] for i in temp_feature]
    feature['ip_entropy']=[i[2] for i in temp_feature]
    feature['bid_entropy']=[i[3] for i in temp_feature]
    feature['url_entropy']=[i[4] for i in temp_feature]
    feature['mean_bid_url']=[i[5] for i in temp_feature]
    feature['bid_count']=[i[6] for i in temp_feature]
    feature['entropy_per_auction']=[i[7] for i in temp_feature]
    return feature
    



# def test_submission(test_data,clf):
#     feature=make_feature_test(test_data)
#     with parallel_backend('threading', n_jobs=4):
#         output=clf.predict(feature)
#     return output
    