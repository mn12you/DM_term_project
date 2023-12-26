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
    input_data=utils.read_file(config.IN_DIR/"train.csv.zip")
    temp=my_algorithm.make_feature(input_data)
    temp.to_csv(config.IN_DIR/"feature.csv")
    clf=my_algorithm.train_classifier(temp)
    dump(clf,model_path)
def test(a:argparse.ArgumentParser,model_path:Path)->None:
    sub=pd.DataFrame()
    clf=load(model_path)
    input_data=utils.read_file(config.IN_DIR/"test.csv.zip")
    temp=my_algorithm.make_feature_test(input_data)
    with parallel_backend('threading', n_jobs=4):
        output=clf.predict(temp)
    sub['bidder_id']=input_data['bidder_id']
    sub['predict']=output
    
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
