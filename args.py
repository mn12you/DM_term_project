import argparse

def parse_arg():
    args_custom = argparse.ArgumentParser()
    def add_arg(*args, **kwargs):
        args_custom.add_argument(*args, **kwargs)

    add_arg('--dataset', type=str, default='train.csv.zip', help='Dataset to use')
    add_arg('--model', type=str, default='clf.joblib', help='Model to use')
    add_arg('--phase', type=str, default='Train', help='Train or test')
    add_arg('--output', type=str, default='submission.csv', help='Submission filename')
   
    return args_custom.parse_args()