from train import load_config, load_train_val_data, select_columns, build_onehot_encoder, dummify_columns

import pandas as pd
import xgboost
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--submission_filepath', type=str, default='data/submission.csv')
parser.add_argument('--config', type=str)
parser.add_argument('--max_depth', type=int, default=2)
parser.add_argument('--eta', type=int, default=1)
parser.add_argument('--silent', type=int, default=1)
parser.add_argument('--objective', type=str, default='binary:logistic')
parser.add_argument('--nthread', type=int, default=4)
parser.add_argument('--num_round', type=int, default=200)
parser.add_argument('--columns', type=str, nargs='+', default=None)


if __name__ == '__main__':

    args = parser.parse_args()
    config = load_config(args)

    train_filepaths = [f'data/cross-validation/cv{i}.csv' for i in range(10)]

    train_X, train_y = load_train_val_data(train_filepaths)
    test_X = pd.read_csv('data/test.csv')

    train_selected_X = select_columns(train_X)
    test_selected_X = select_columns(test_X)

    onehot_encoder, dummified_columns = build_onehot_encoder(train_selected_X)

    train_dummified_X = dummify_columns(train_selected_X, onehot_encoder, dummified_columns)
    test_dummified_X = dummify_columns(test_selected_X, onehot_encoder, dummified_columns)

    dtrain = xgboost.DMatrix(train_dummified_X, label=train_y)
    dtest = xgboost.DMatrix(test_dummified_X)

    param = {
        'max_depth': config['max_depth'],
        'eta': config['eta'],
        'silent': config['silent'],
        'objective': config['objective'],
        'nthread': config['nthread'],
        'eval_metric': 'auc'}

    evallist = [(dtrain, 'train')]

    bst = xgboost.train(param, dtrain, config['num_round'], evallist)

    test_predicted = bst.predict(dtest)

    pd.DataFrame({'target': test_predicted, 'id': test_X['id']}).to_csv(config['submission_filepath'], index=False)