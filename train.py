import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import xgboost
import numpy as np
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--config', type=int, default=2)
parser.add_argument('--max_depth', type=int, default=2)
parser.add_argument('--eta', type=int, default=1)
parser.add_argument('--silent', type=int, default=1)
parser.add_argument('--objective', type=str, default='binary:logistic')
parser.add_argument('--nthread', type=int, default=4)
parser.add_argument('--num_round', type=int, default=200)
parser.add_argument('--columns', type=str, nargs='+', default=None)


def load_config(args):
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)

        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
    else:
        config = vars(args)  # convert to dictionary

    return config


def load_train_val_data(filepaths):
    dataframes = []
    for filepath in filepaths:
        dataframes.append(pd.read_csv(filepath))

    data = pd.concat(dataframes)
    X = data.drop(columns=['target'])
    y = data['target']

    return X, y


def select_columns(dataframe, columns=None):
    if columns is None:
        return dataframe.drop(columns=['id'])
    else:
        return dataframe[columns]


def build_onehot_encoder(selected_X):
    cat_columns = [column for column in selected_X.columns if column.endswith('cat')]
    categorical_data = selected_X[cat_columns]
    non_negative_data = categorical_data + 1

    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    onehot_encoder.fit(non_negative_data)

    dummified_columns = []
    for column, categories in zip(cat_columns, onehot_encoder.categories_):
        for category in categories:
            dummified_columns.append(column + '_' + str(category))

    return onehot_encoder, dummified_columns


def dummify_columns(dataframe, onehot_encoder, dummified_columns):
    cat_columns = [column for column in dataframe.columns if column.endswith('cat')]
    categorical_data = dataframe[cat_columns]
    non_negative_data = categorical_data + 1
    transformed_array = onehot_encoder.transform(non_negative_data)
    transformed_df = pd.DataFrame(transformed_array, columns=dummified_columns, index=dataframe.index)
#     dummified_dataframe = pd.get_dummies(dataframe, columns=cat_columns)
    dummified_dataframe = pd.concat([dataframe.drop(columns=cat_columns), transformed_df], axis='columns')
    return dummified_dataframe


def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


if __name__ == '__main__':

    args = parser.parse_args()
    config = load_config(args)

    train_filepaths = [f'data/cross-validation/cv{i}.csv' for i in range(9)]
    val_filepaths = ['data/cross-validation/cv9.csv']

    train_X, train_y = load_train_val_data(train_filepaths)
    val_X, val_y = load_train_val_data(val_filepaths)

    train_selected_X = select_columns(train_X)
    val_selected_X = select_columns(val_X)

    onehot_encoder, dummified_columns = build_onehot_encoder(pd.concat([train_selected_X, val_selected_X]))

    train_dummified_X = dummify_columns(train_selected_X, onehot_encoder, dummified_columns)
    val_dummified_X = dummify_columns(val_selected_X, onehot_encoder, dummified_columns)

    dtrain = xgboost.DMatrix(train_dummified_X, label=train_y)
    dval = xgboost.DMatrix(val_dummified_X, label=val_y)

    param = {
        'max_depth': config['max_depth'],
        'eta': config['eta'],
        'silent': config['silent'],
        'objective': config['objective'],
        'nthread': config['nthread'],
        'eval_metric': 'auc'}

    evallist = [(dval, 'eval'), (dtrain, 'train')]

    bst = xgboost.train(param, dtrain, config['num_round'], evallist)

    val_predicted = bst.predict(dval)

    print(gini_normalized(val_y, val_predicted))