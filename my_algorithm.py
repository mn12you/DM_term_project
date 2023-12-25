from collections import Counter
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import parallel_backend
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd 
from utils import timer

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
@timer 
def make_feature(train_data,bids_data):
    train_feature=pd.DataFrame()
    bid_per_auction_list=[]
    med_bid_time_list=[]
    label=[]
    for ind,name in enumerate(train_data['bidder_id']):
        mask=bids_data['bidder_id']==name
        temp_bids=bids_data[mask]
        if len(temp_bids)>0:
            bid_per_auction_list.append(bid_per_auction(temp_bids))
            med_bid_time_list.append(med_bid_time(temp_bids))
            label.append(train_data.loc[ind,'outcome'])
    train_feature['bid_per_auction'] =bid_per_auction_list
    train_feature['med_bid_time'] =med_bid_time_list
    train_feature['out']=label
            
    return train_feature

def make_feature_speedup(train_data,bids_data):
    train_feature=pd.DataFrame()
    bid_per_auction_list=[]
    med_bid_time_list=[]
    label=[]
    bids=bids_data.set_index(['bidder_id'])
    for ind,name in enumerate(train_data['bidder_id']):
        temp_bids=bids.loc[name]
@timer 
def train_classifier(feature_data):
    x=feature_data.drop(columns=['out'])
    x= preprocessing.normalize(x.values, axis=0)
    y=feature_data['out']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # clf=RandomForestClassifier(n_jobs=4,n_estimators=800, max_depth=None, min_samples_leaf=1, random_state=42, criterion='entropy')
    clf=HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
    clf.fit(X_train, y_train)
    score=clf.score(X_test, y_test)
    print(score)
    return clf
@timer 
def make_feature_test(test_data,bids_data):
    test_feature=pd.DataFrame()
    bid_per_auction_list=[]
    med_bid_time_list=[]
    label=[]
    for ind,name in enumerate(test_data['bidder_id']):
        mask=bids_data['bidder_id']==name
        temp_bids=bids_data[mask]
        if len(temp_bids)>0:
            bid_per_auction_list.append(bid_per_auction(temp_bids))
            med_bid_time_list.append(med_bid_time(temp_bids))
        else:
            bid_per_auction_list.append(0)
            med_bid_time_list.append(0)

    test_feature['bid_per_auction'] =bid_per_auction_list
    test_feature['med_bid_time'] =med_bid_time_list
            
    return test_feature

def test_submission(test_data,bids_data,clf):
    feature=make_feature_test(test_data,bids_data)
    with parallel_backend('threading', n_jobs=4):
        output=clf.predict(feature)
    return output
   