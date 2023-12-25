import args
import config
import utils
import my_algorithm
import argparse
from pathlib import Path
from joblib import dump, load
def train(a:argparse.ArgumentParser)->None:
    model_path=config.MODEL_DIR/a.model
    input_data,bids_data=utils.read_file(config.IN_DIR/a.dataset)
    print("ALL")
    temp=my_algorithm.make_feature(input_data,bids_data)
    clf=my_algorithm.train_classifier(temp)
    dump(clf,model_path)
def test(a:argparse.ArgumentParser,model_path:Path)->None:
    clf=load(model_path)
    input_data,bids_data=utils.read_file(config.IN_DIR/a.dataset)
    temp=my_algorithm.make_feature_test(input_data,bids_data)
    output=clf.predict(temp)
    utils.write_file(output,config.OUT_DIR/a.output)

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