from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--split', action='store_true')
parser.add_argument('--cv', type=int, default=10)


def split_cross_validation_data(raw_filepath='data/train.csv', cv=10):
    raw_data = pd.read_csv(raw_filepath)
    raw_data_shuffled = raw_data.sample(frac=1)
    cv_data_size = len(raw_data_shuffled) // cv
    for i in range(cv):
        start = i * cv_data_size
        end = (i+1) * cv_data_size if i < cv - 1 else None
        cv_dataframe = raw_data_shuffled[start:end]
        cv_dataframe.to_csv(f'data/cross-validation/cv{i}.csv', index=False)


if __name__ == '__main__':

    args = parser.parse_args()
    if args.split:
        split_cross_validation_data(cv=args.cv)
