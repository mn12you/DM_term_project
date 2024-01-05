import args
import config
import utils
import my_algorithm
import argparse
import pandas as pd
from pathlib import Path
from joblib import dump, load
from joblib import parallel_backend
def train(a:argparse.ArgumentParser)->None:
    model_path=config.MODEL_DIR/a.model
    input_data=utils.read_file(config.IN_DIR/"train_clean.csv")
    train_feature=config.IN_DIR/"feature.csv"
    if  train_feature.exists() and a.feature==False:
        temp= pd.read_csv(train_feature)
    else:
        temp=my_algorithm.make_feature(input_data)
        temp.to_csv(config.IN_DIR/"feature.csv",index=False)   
    my_algorithm.train_classifier(temp,a)
def test(a:argparse.ArgumentParser,model_path:Path)->None:
    sub=pd.DataFrame()
    clf=load(model_path)
    input_data=utils.read_file(config.IN_DIR/"test.csv.zip")
    test_feature=config.IN_DIR/"test_feature.csv"
    if test_feature.exists()  and a.feature==False :
        with parallel_backend('threading', n_jobs=4):
            temp=pd.read_csv(test_feature)
            output=clf.predict_proba(temp)[:,[1]]
    else:
        temp=my_algorithm.make_feature_test(input_data)
        temp.to_csv(config.IN_DIR/"test_feature.csv",index=False)
        with parallel_backend('threading', n_jobs=4):
            output=clf.predict_proba(temp)[:,[1]]
    sub['bidder_id']=input_data['bidder_id']
    sub['prediction']=output
    
    utils.write_file(sub,config.OUT_DIR/a.output)

if __name__=="__main__":
    a=args.parse_arg()
    if a.phase=='train':
        train(a)    
    else: 
        model_path=config.MODEL_DIR/a.model
        if model_path.exists():
            test(a,model_path)
        else:
            train(a)
            test(a,model_path)
